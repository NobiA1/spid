"""
Market data loader: fetch quotes, fundamentals, and macro data with local caching.

Anonymized release note: real vendor endpoints and credentials have been replaced
by placeholder values. A synthetic offline dataset can be produced with
``fake_data/make_fake_data.py`` so the pipeline runs without external services.
"""

import hashlib
import os
import pickle
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    yaml = None

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CACHE_DIR = "/tmp/market_data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def _is_us_symbol(stock: str) -> bool:
    return "." not in str(stock)


def _all_us_symbols(stocks: List[str]) -> bool:
    return bool(stocks) and all(_is_us_symbol(s) for s in stocks)


def _local_bundle_roots() -> List[str]:
    roots = [
        os.path.join(PROJECT_ROOT, "data_config", "market_2025_retail_merged"),
        os.path.join(PROJECT_ROOT, "data_config", "market_2025"),
    ]
    return [root for root in roots if os.path.exists(root)]


def _first_existing_path(*parts: str) -> Optional[str]:
    for root in _local_bundle_roots():
        path = os.path.join(root, *parts)
        if os.path.exists(path):
            return path
    return None

def _load_token(yaml_path: str = None) -> str:
    if yaml_path is None:
        yaml_path = os.path.join(PROJECT_ROOT, "data_config", "tokens.yaml")
    
    if yaml is not None:
        try:
            with open(yaml_path, "r") as f:
                cfg = yaml.safe_load(f)
            return cfg["market_data"]["token"]
        except Exception:
            pass
            
    # Fallback to manual parsing if yaml is missing or failed
    try:
        with open(yaml_path, "r") as f:
            for line in f:
                if "token:" in line and "token_news:" not in line:
                    return line.split("token:")[1].strip()
    except Exception:
        pass
    return ""

def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}.pkl")


def _stocks_cache_suffix(stocks: List[str]) -> str:
    material = ",".join(sorted(str(s) for s in stocks)).encode("utf-8")
    return hashlib.sha1(material).hexdigest()[:12]

def _load_cache(name: str):
    p = _cache_path(name)
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    return None

def _save_cache(name: str, data):
    with open(_cache_path(name), "wb") as f:
        pickle.dump(data, f)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_stock_list(csv_path: str = None) -> List[Dict]:
    """Load a stock-universe CSV → list of dicts with stk_code, firm_name, ind."""
    if csv_path is None:
        csv_path = os.path.join(PROJECT_ROOT, "data_config", "MSCI_A50.csv")
    import csv
    stocks = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = (row.get("stk_code") or row.get("ts_code") or "").strip()
            if not code:
                continue
            firm = (row.get("firm_name") or row.get("name") or "").strip()
            ind = (
                row.get("l3_name")
                or row.get("ind")
                or ""
            ).strip()
            stocks.append({
                "stk_code": code,
                "firm_name": firm,
                "ind": ind,
            })
    return stocks


def fetch_daily_prices(
    stocks: List[str],
    start_date: str,
    end_date: str,
    token: str = None,
    pre_days: int = 60,
) -> Dict[str, Dict[str, dict]]:
    """
    Fetch daily OHLCV for each stock from Tushare.
    Returns: {stk_code: {date_str: {open, high, low, close, vol, amount}}}
    date_str format: "YYYYMMDD"
    pre_days: fetch extra days before start_date for history (MA, momentum).
    """
    cache_key = f"prices_{start_date}_{end_date}_{_stocks_cache_suffix(stocks)}"
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    if _all_us_symbols(stocks):
        try:
            result = _fetch_yfinance_daily_prices(stocks, start_date, end_date, pre_days=pre_days)
            _save_cache(cache_key, result)
            print(f"[DataLoader] Cached yFinance daily prices for {len(result)} US stocks")
            return result
        except Exception as e:
            print(f"[DataLoader] yFinance daily fetch failed, falling back to Tushare: {e}")

    # Only use bundled local CSV when the requested range is within 2025.
    # Otherwise, fall back to Tushare API to avoid empty slices (e.g., 2026 dates).
    use_local_bundle = str(end_date).startswith("2025")
    local_path = _first_existing_path("stocks", "daily_2025.csv") if use_local_bundle else None
    if local_path is not None:
        import pandas as pd
        df = pd.read_csv(local_path)
        df["trade_date"] = df["trade_date"].astype(str)
        dt_start = datetime.strptime(start_date, "%Y%m%d")
        dt_expanded = dt_start - timedelta(days=pre_days + 30)
        expanded_start = dt_expanded.strftime("%Y%m%d")
        df = df[(df["ts_code"].isin(stocks)) & (df["trade_date"] >= expanded_start) & (df["trade_date"] <= end_date)]
        result: Dict[str, Dict[str, dict]] = {}
        for stk, sdf in df.groupby("ts_code"):
            stk_data = {}
            sdf = sdf.sort_values("trade_date")
            for _, row in sdf.iterrows():
                stk_data[str(row["trade_date"])] = {
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "vol": float(row["vol"]) * 100,
                    "amount": float(row["amount"]) * 1000,
                }
            result[stk] = stk_data
        _save_cache(cache_key, result)
        print(f"[DataLoader] Loaded local daily prices for {len(result)} stocks from {local_path}")
        return result

    if token is None:
        token = _load_token()
    import tushare as ts
    pro = ts.pro_api(token)

    # Expand start for historical context
    dt_start = datetime.strptime(start_date, "%Y%m%d")
    dt_expanded = dt_start - timedelta(days=pre_days + 30)  # extra margin for holidays
    expanded_start = dt_expanded.strftime("%Y%m%d")

    result = {}
    for i, stk in enumerate(stocks):
        try:
            df = pro.daily(ts_code=stk, start_date=expanded_start, end_date=end_date)
            if df is None or df.empty:
                print(f"  [DataLoader] {stk}: no data")
                continue
            df = df.sort_values("trade_date")
            stk_data = {}
            for _, row in df.iterrows():
                stk_data[row["trade_date"]] = {
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "vol": float(row["vol"]) * 100,  # tushare vol unit = lots (100 shares)
                    "amount": float(row["amount"]) * 1000,  # tushare amount unit = thousand CNY
                }
            result[stk] = stk_data
            if (i + 1) % 30 == 0:
                print(f"  [DataLoader] Fetched {i+1}/{len(stocks)} stocks")
                time.sleep(0.3)
        except Exception as e:
            print(f"  [DataLoader] {stk} error: {e}")
            time.sleep(0.5)

    _save_cache(cache_key, result)
    print(f"[DataLoader] Cached {len(result)} stocks prices")
    return result


def fetch_circ_mv(
    stocks: List[str],
    trade_date: str,
    token: str = None,
) -> Dict[str, float]:
    """Fetch circulating market cap (in units of 10k CNY) for synthetic float calculation."""
    cache_key = f"circ_mv_{trade_date}_{_stocks_cache_suffix(stocks)}"
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    if _all_us_symbols(stocks):
        result = _fetch_yfinance_market_caps(stocks)
        _save_cache(cache_key, result)
        print(f"[DataLoader] Loaded yFinance market caps for {len(result)} US stocks")
        return result

    use_local_bundle = str(trade_date).startswith("2025")
    local_path = _first_existing_path("stocks", "daily_basic_2025.csv") if use_local_bundle else None
    if local_path is not None:
        import pandas as pd
        df = pd.read_csv(local_path)
        df["trade_date"] = df["trade_date"].astype(str)
        trade_date_str = str(trade_date)
        df = df[(df["ts_code"].isin(stocks)) & (df["trade_date"] == trade_date_str)]
        result = {row["ts_code"]: float(row["circ_mv"]) for _, row in df.iterrows() if pd.notna(row.get("circ_mv"))}
        _save_cache(cache_key, result)
        print(f"[DataLoader] Loaded local circ_mv for {len(result)} stocks from {local_path}")
        return result

    if token is None:
        token = _load_token()
    import tushare as ts
    pro = ts.pro_api(token)

    result = {}
    for stk in stocks:
        try:
            df = pro.daily_basic(ts_code=stk, trade_date=trade_date, fields="ts_code,circ_mv")
            if df is not None and not df.empty:
                result[stk] = float(df.iloc[0]["circ_mv"])
            time.sleep(0.05)
        except Exception as e:
            print(f"  [circ_mv] {stk}: {e}")

    _save_cache(cache_key, result)
    return result


def fetch_fina_indicator(
    stocks: List[str],
    token: str = None,
) -> Dict[str, Dict]:
    """Fetch latest financial indicators (EPS, BPS, ROE etc.)."""
    cache_key = f"fina_indicator_latest_{_stocks_cache_suffix(stocks)}"
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    if _all_us_symbols(stocks):
        result = _fetch_yfinance_fina_indicator(stocks)
        _save_cache(cache_key, result)
        print(f"[DataLoader] Loaded yFinance financial proxies for {len(result)} US stocks")
        return result

    # Use bundled financials only as a 2025-local proxy. For other years, prefer API.
    local_income = _first_existing_path("stocks", "income_2025.csv")
    local_balance = _first_existing_path("stocks", "balancesheet_2025.csv")
    use_local_bundle = bool(local_income and local_balance)
    if use_local_bundle and os.environ.get("FORCE_TUSHARE_API", "").lower() not in ("1", "true", "yes"):
        import pandas as pd
        income_df = pd.read_csv(local_income)
        bal_df = pd.read_csv(local_balance)
        result = {}
        for stk in stocks:
            inc = income_df[income_df["ts_code"] == stk].copy()
            bal = bal_df[bal_df["ts_code"] == stk].copy()
            if inc.empty or bal.empty:
                continue
            inc["end_date"] = inc["end_date"].astype(str)
            bal["end_date"] = bal["end_date"].astype(str)
            inc_row = inc.sort_values(["end_date", "ann_date"], ascending=False).iloc[0]
            bal_row = bal.sort_values(["end_date", "ann_date"], ascending=False).iloc[0]

            total_share = float(bal_row.get("total_share", 0) or 0)
            equity = float(bal_row.get("total_hldr_eqy_exc_min_int", 0) or bal_row.get("total_hldr_eqy_inc_min_int", 0) or 0)
            bps = equity / total_share if total_share > 0 else 1.0
            net_income = float(inc_row.get("n_income", 0) or 0)
            roe_waa = net_income / equity if equity > 0 else 0.0
            result[stk] = {
                "eps": float(inc_row.get("basic_eps", 0) or 0),
                "bps": float(bps if bps > 0 else 1.0),
                "roe_waa": float(roe_waa),
                "end_date": str(inc_row.get("end_date", "")),
            }
        _save_cache(cache_key, result)
        print(f"[DataLoader] Loaded local fina_indicator proxy for {len(result)} stocks")
        return result

    if token is None:
        token = _load_token()
    import tushare as ts
    pro = ts.pro_api(token)

    result = {}
    fields = "ts_code,end_date,eps,bps,roe_waa,netprofit_margin,grossprofit_margin,debt_to_assets,current_ratio,ocf_to_or"
    for stk in stocks:
        try:
            df = pro.fina_indicator(ts_code=stk, fields=fields)
            if df is not None and not df.empty:
                row = df.sort_values("end_date", ascending=False).iloc[0]
                result[stk] = {
                    "eps": float(row.get("eps", 0) or 0),
                    "bps": float(row.get("bps", 1) or 1),
                    "roe_waa": float(row.get("roe_waa", 0) or 0),
                    "end_date": str(row.get("end_date", "")),
                }
            time.sleep(0.15)
        except Exception as e:
            print(f"  [fina] {stk}: {e}")

    _save_cache(cache_key, result)
    print(f"[DataLoader] Cached fina_indicator for {len(result)} stocks")
    return result


def fetch_macro_rate(token: str = None) -> Dict[str, float]:
    """Fetch 10Y bond rate and LPR 1Y for g_B calculation."""
    cache_key = "macro_rate"
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    # Default fallback values (approximate 2024–2025 levels)
    result = {"rate_10y": 0.023, "lpr_1y": 0.031}

    local_shibor = _first_existing_path("macro", "shibor.csv")
    if local_shibor is not None:
        import pandas as pd
        df = pd.read_csv(local_shibor)
        if not df.empty:
            row = df.sort_values("date", ascending=False).iloc[0]
            # Use the 1Y SHIBOR as a proxy for rate_10y when bond data is unavailable.
            result = {
                "rate_10y": float(row.get("1y", result["rate_10y"] * 100) or result["rate_10y"] * 100) / 100.0,
                "lpr_1y": float(row.get("1w", result["lpr_1y"] * 100) or result["lpr_1y"] * 100) / 100.0,
            }
            _save_cache(cache_key, result)
            return result

    try:
        if token is None:
            token = _load_token()
        import tushare as ts
        pro = ts.pro_api(token)
        # Try to get CN bond yield
        df = pro.cn_gdp()  # placeholder — Tushare macro data can be flaky
        # Use defaults for now
    except Exception:
        pass

    _save_cache(cache_key, result)
    return result


def get_trading_dates(
    price_data: Dict[str, Dict[str, dict]],
    start_date: str,
    end_date: str,
) -> List[str]:
    """Extract sorted trading dates within [start, end] from price_data."""
    all_dates = set()
    for stk_data in price_data.values():
        for d in stk_data.keys():
            if start_date <= d <= end_date:
                all_dates.add(d)
    return sorted(all_dates)


def get_price_on_date(
    price_data: Dict[str, Dict[str, dict]],
    stock: str,
    date: str,
) -> Optional[dict]:
    """Get OHLCV dict for a stock on a specific date."""
    return price_data.get(stock, {}).get(date)


def get_close_price(
    price_data: Dict[str, Dict[str, dict]],
    stock: str,
    date: str,
    fallback: float = 100.0,
) -> float:
    """Get closing price for a stock on a specific date."""
    d = get_price_on_date(price_data, stock, date)
    return d["close"] if d else fallback


def get_history_before(
    price_data: Dict[str, Dict[str, dict]],
    stock: str,
    date: str,
    n: int = 20,
) -> List[dict]:
    """Get up to n trading days of OHLCV before (not including) date, most recent first."""
    stk_data = price_data.get(stock, {})
    dates = sorted(d for d in stk_data.keys() if d < date)
    dates = dates[-n:]  # take last n
    dates.reverse()  # most recent first
    return [stk_data[d] for d in dates]


def _fetch_yfinance_daily_prices(
    stocks: List[str],
    start_date: str,
    end_date: str,
    pre_days: int = 60,
) -> Dict[str, Dict[str, dict]]:
    import pandas as pd
    import yfinance as yf

    dt_start = datetime.strptime(start_date, "%Y%m%d")
    dt_end = datetime.strptime(end_date, "%Y%m%d")
    expanded_start = (dt_start - timedelta(days=pre_days + 30)).strftime("%Y-%m-%d")
    # yfinance end is exclusive; add a few calendar days for weekends/holidays.
    yf_end = (dt_end + timedelta(days=5)).strftime("%Y-%m-%d")

    result: Dict[str, Dict[str, dict]] = {}
    for stock in stocks:
        stk_data: Dict[str, dict] = {}
        if os.environ.get("SUM_US_PRICE_SOURCE", "").lower() != "akshare":
            try:
                df = yf.download(
                    stock,
                    start=expanded_start,
                    end=yf_end,
                    progress=False,
                    auto_adjust=False,
                    threads=False,
                )
                if df is not None and not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [c[0] for c in df.columns]
                    df = df.reset_index()
                    for _, row in df.iterrows():
                        dt = pd.to_datetime(row["Date"]).to_pydatetime()
                        d = dt.strftime("%Y%m%d")
                        if d > end_date:
                            continue
                        close = float(row.get("Close", row.get("Adj Close", 0)) or 0)
                        if close <= 0:
                            continue
                        vol = float(row.get("Volume", 0) or 0)
                        stk_data[d] = {
                            "open": float(row.get("Open", close) or close),
                            "high": float(row.get("High", close) or close),
                            "low": float(row.get("Low", close) or close),
                            "close": close,
                            "vol": vol,
                            "amount": vol * close,
                        }
            except Exception as e:
                print(f"  [yFinance] {stock}: {e}")
        if not stk_data:
            stk_data = _fetch_akshare_us_daily(stock, expanded_start.replace("-", ""), end_date)
        if not stk_data:
            stk_data = _fetch_example_daily(stock, expanded_start.replace("-", ""), end_date)
        if not stk_data:
            print(f"  [USPrices] {stock}: no data")
            continue
        result[stock] = stk_data
        time.sleep(0.1)
    return result


def _fetch_akshare_us_daily(stock: str, start_date: str, end_date: str) -> Dict[str, dict]:
    try:
        import akshare as ak
        import pandas as pd
        df = ak.stock_us_daily(symbol=stock, adjust="")
    except Exception as e:
        print(f"  [AkShare] {stock}: {e}")
        return {}
    if df is None or df.empty:
        return {}
    result: Dict[str, dict] = {}
    for _, row in df.iterrows():
        dt = pd.to_datetime(row.get("date")).to_pydatetime()
        d = dt.strftime("%Y%m%d")
        if d < start_date or d > end_date:
            continue
        close = float(row.get("close") or 0)
        if close <= 0:
            continue
        vol = float(row.get("volume") or 0)
        result[d] = {
            "open": float(row.get("open") or close),
            "high": float(row.get("high") or close),
            "low": float(row.get("low") or close),
            "close": close,
            "vol": vol,
            "amount": vol * close,
        }
    if result:
        print(f"  [AkShare] {stock}: loaded {len(result)} rows")
    return result


def _fetch_example_daily(stock: str, start_date: str, end_date: str) -> Dict[str, dict]:
    import requests

    symbol = f"{stock}.US"
    url = f"https://api.example-data.invalid/api/eod/{symbol}"
    params = {
        "from": f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}",
        "to": f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}",
        "period": "d",
        "fmt": "json",
        "api_token": os.environ.get("EXAMPLE_DATA_TOKEN", "demo"),
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code != 200:
            return {}
        rows = response.json()
    except Exception as e:
        print(f"  [ExampleData] {stock}: {e}")
        return {}
    result: Dict[str, dict] = {}
    if not isinstance(rows, list):
        return result
    for row in rows:
        d = str(row.get("date", "")).replace("-", "")
        close = float(row.get("close") or row.get("adjusted_close") or 0)
        if not d or close <= 0:
            continue
        vol = float(row.get("volume") or 0)
        result[d] = {
            "open": float(row.get("open") or close),
            "high": float(row.get("high") or close),
            "low": float(row.get("low") or close),
            "close": close,
            "vol": vol,
            "amount": vol * close,
        }
    if result:
        print(f"  [ExampleData] {stock}: loaded {len(result)} rows with demo token")
    return result


def _fetch_yfinance_market_caps(stocks: List[str]) -> Dict[str, float]:
    if os.environ.get("SUM_US_PRICE_SOURCE", "").lower() == "akshare":
        approx = {
            "AAPL": 3.0e12,
            "MSFT": 3.5e12,
            "NVDA": 3.4e12,
            "AMZN": 2.2e12,
            "META": 1.6e12,
            "GOOGL": 2.1e12,
            "TSLA": 1.0e12,
        }
        return {s: approx[s] for s in stocks if s in approx}
    import yfinance as yf

    result: Dict[str, float] = {}
    for stock in stocks:
        cap = None
        try:
            fast_info = getattr(yf.Ticker(stock), "fast_info", {}) or {}
            cap = fast_info.get("market_cap") if hasattr(fast_info, "get") else None
        except Exception:
            cap = None
        if not cap:
            try:
                cap = (yf.Ticker(stock).get_info() or {}).get("marketCap")
            except Exception:
                cap = None
        if cap:
            result[stock] = float(cap)
        time.sleep(0.1)
    return result


def _fetch_yfinance_fina_indicator(stocks: List[str]) -> Dict[str, Dict]:
    if os.environ.get("SUM_US_PRICE_SOURCE", "").lower() == "akshare":
        approx = {
            "AAPL": {"eps": 6.0, "bps": 4.5, "roe_waa": 1.3},
            "MSFT": {"eps": 12.0, "bps": 40.0, "roe_waa": 0.35},
            "NVDA": {"eps": 2.5, "bps": 4.0, "roe_waa": 0.9},
            "AMZN": {"eps": 5.0, "bps": 30.0, "roe_waa": 0.2},
            "META": {"eps": 23.0, "bps": 75.0, "roe_waa": 0.35},
            "GOOGL": {"eps": 8.0, "bps": 25.0, "roe_waa": 0.3},
            "TSLA": {"eps": 2.0, "bps": 22.0, "roe_waa": 0.1},
        }
        return {
            s: {
                "eps": approx.get(s, {}).get("eps", 1.0),
                "bps": approx.get(s, {}).get("bps", 10.0),
                "roe_waa": approx.get(s, {}).get("roe_waa", 0.1),
                "end_date": "",
            }
            for s in stocks
        }
    import yfinance as yf

    result: Dict[str, Dict] = {}
    for stock in stocks:
        bps = 1.0
        roe = 0.0
        eps = 0.0
        try:
            info = yf.Ticker(stock).get_info() or {}
            bps = float(info.get("bookValue") or bps)
            roe = float(info.get("returnOnEquity") or roe)
            eps = float(info.get("trailingEps") or eps)
        except Exception:
            pass
        result[stock] = {
            "eps": eps,
            "bps": max(bps, 0.01),
            "roe_waa": roe,
            "end_date": "",
        }
        time.sleep(0.1)
    return result
