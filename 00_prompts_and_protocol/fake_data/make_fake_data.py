#!/usr/bin/env python3
"""Generate small synthetic market and news inputs for the anonymized release.

The real price series and news corpora are not redistributed (they are large and
carry identifying provenance). This script fabricates a tiny, clearly synthetic
dataset with the exact schemas the pipeline expects, so every script can be
executed offline end to end:

  sample_config/MSCI_A50.csv                  stock universe (`stk_code`, `firm_name`, `ind`)
  sample_config/stocks/daily_2025.csv         daily OHLCV for 2025
  sample_config/stocks/daily_basic_2025.csv   daily circulating market cap
  sample_config/macro/shibor.csv              macro rate proxy
  sample_news/merged/2025-04.csv              merged news digest (`llm_items_json`)
  sample_news/monthly/news_annotated_2025_4.csv  flat annotated news

All names, tickers, prices and headlines are fictional placeholders.
"""

from __future__ import annotations

import csv
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

BASE = Path(__file__).resolve().parent
CFG_DIR = BASE / "sample_config"
NEWS_DIR = BASE / "sample_news"

N_STOCKS = 50
N_DAYS = 21  # April 2025 style window
START = datetime(2025, 4, 1)
SEED = 42

INDUSTRIES = [
    "Banking", "Insurance", "Securities", "Liquor", "Food & Beverage",
    "Electrical Equipment", "Automobiles", "Electronic Components",
    "Pharmaceuticals", "Real Estate", "Energy", "Telecom",
]


def trading_days(start: datetime, n: int) -> list[datetime]:
    days: list[datetime] = []
    cursor = start
    while len(days) < n:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def write_stock_universe() -> list[dict]:
    stocks = []
    for i in range(N_STOCKS):
        code = f"EX{i:04d}.SZ" if i % 2 == 0 else f"EX{i:04d}.SH"
        stocks.append(
            {
                "stk_code": code,
                "firm_name": f"Example Corp {i:02d}",
                "ind": INDUSTRIES[i % len(INDUSTRIES)],
            }
        )
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    with (CFG_DIR / "MSCI_A50.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stk_code", "firm_name", "ind", "l3_name"])
        writer.writeheader()
        for s in stocks:
            writer.writerow({**s, "l3_name": s["ind"]})
    return stocks


def write_prices(stocks: list[dict]) -> None:
    rng = random.Random(SEED)
    days = trading_days(START, N_DAYS)
    stocks_dir = CFG_DIR / "stocks"
    stocks_dir.mkdir(parents=True, exist_ok=True)

    price_rows = []
    basic_rows = []
    for s in stocks:
        price = rng.uniform(8.0, 120.0)
        float_shares = rng.uniform(2e8, 5e9)
        for d in days:
            drift = rng.gauss(0.0, 0.02)
            price = max(1.0, price * (1.0 + drift))
            high = price * (1.0 + abs(rng.gauss(0.0, 0.01)))
            low = price * (1.0 - abs(rng.gauss(0.0, 0.01)))
            open_ = price * (1.0 + rng.gauss(0.0, 0.005))
            vol = rng.uniform(1e6, 5e7)
            ds = d.strftime("%Y%m%d")
            price_rows.append(
                {
                    "ts_code": s["stk_code"],
                    "trade_date": ds,
                    "open": round(open_, 2),
                    "high": round(high, 2),
                    "low": round(low, 2),
                    "close": round(price, 2),
                    "vol": round(vol / 100.0, 2),
                    "amount": round(vol * price / 1000.0, 2),
                }
            )
            if d == days[-1]:
                basic_rows.append(
                    {"ts_code": s["stk_code"], "trade_date": ds, "circ_mv": round(float_shares / 1e4, 2)}
                )

    with (stocks_dir / "daily_2025.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"]
        )
        writer.writeheader()
        writer.writerows(price_rows)

    with (stocks_dir / "daily_basic_2025.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ts_code", "trade_date", "circ_mv"])
        writer.writeheader()
        writer.writerows(basic_rows)

    # Minimal income + balance sheets so the local financials proxy has data.
    income_rows = []
    balance_rows = []
    for s in stocks:
        income_rows.append(
            {
                "ts_code": s["stk_code"],
                "end_date": "20241231",
                "ann_date": "20250331",
                "n_income": round(rng.uniform(1e8, 5e9), 2),
                "basic_eps": round(rng.uniform(0.1, 3.0), 3),
            }
        )
        balance_rows.append(
            {
                "ts_code": s["stk_code"],
                "end_date": "20241231",
                "ann_date": "20250331",
                "total_share": round(rng.uniform(1e8, 1e10), 2),
                "total_hldr_eqy_exc_min_int": round(rng.uniform(1e9, 5e11), 2),
            }
        )
    with (stocks_dir / "income_2025.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ts_code", "end_date", "ann_date", "n_income", "basic_eps"])
        writer.writeheader()
        writer.writerows(income_rows)
    with (stocks_dir / "balancesheet_2025.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["ts_code", "end_date", "ann_date", "total_share", "total_hldr_eqy_exc_min_int"],
        )
        writer.writeheader()
        writer.writerows(balance_rows)


def write_macro() -> None:
    macro_dir = CFG_DIR / "macro"
    macro_dir.mkdir(parents=True, exist_ok=True)
    with (macro_dir / "shibor.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "1w", "1y"])
        writer.writerow(["2025-03-31", "1.80", "2.30"])


def _fake_news_items(rng: random.Random, firm: str, ind: str) -> list[dict]:
    templates = [
        ("company_update", 0.7, "Example Corp reports a modest change in quarterly expectations."),
        ("industry_policy", 0.6, "A regional policy update affects the broader sector outlook."),
        ("market_comment", 0.4, "Commentators note mixed near-term market sentiment."),
    ]
    items = []
    for event_type, importance, content in rng.sample(templates, k=rng.randint(1, 3)):
        scope_stock = event_type == "company_update"
        items.append(
            {
                "event_type": event_type,
                "importance": importance,
                "clean_content": content,
                "stock": [firm] if scope_stock else [],
                "ind": [] if scope_stock else [ind],
            }
        )
    return items


def write_news(stocks: list[dict]) -> None:
    rng = random.Random(SEED + 1)
    merged_dir = NEWS_DIR / "merged"
    monthly_dir = NEWS_DIR / "monthly"
    merged_dir.mkdir(parents=True, exist_ok=True)
    monthly_dir.mkdir(parents=True, exist_ok=True)

    days = trading_days(START, N_DAYS)
    merged_rows = []
    monthly_rows = []
    for idx, d in enumerate(days):
        for j in range(3):
            s = stocks[(idx * 3 + j) % len(stocks)]
            ts = d.replace(hour=9 + j, minute=15)
            items = _fake_news_items(rng, s["firm_name"], s["ind"])
            news_id = f"N{idx:03d}{j}"
            merged_rows.append(
                {
                    "ID": news_id,
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "llm_items_json": json.dumps({"items": items}),
                }
            )
            top = items[0]
            monthly_rows.append(
                {
                    "ID": news_id,
                    "type": top["event_type"],
                    "scope": "stock" if top["stock"] else "industry",
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "content": top["clean_content"],
                    "stock": repr(top["stock"]),
                    "ind": repr(top["ind"]),
                    "importance": top["importance"],
                    "event_type": top["event_type"],
                }
            )

    with (merged_dir / "2025-04.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "timestamp", "llm_items_json"])
        writer.writeheader()
        writer.writerows(merged_rows)

    with (monthly_dir / "news_annotated_2025_4.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["ID", "type", "scope", "timestamp", "content", "stock", "ind", "importance", "event_type"],
        )
        writer.writeheader()
        writer.writerows(monthly_rows)


def main() -> None:
    stocks = write_stock_universe()
    write_prices(stocks)
    write_macro()
    write_news(stocks)
    print(f"Wrote synthetic config to {CFG_DIR}")
    print(f"Wrote synthetic news to {NEWS_DIR}")


if __name__ == "__main__":
    main()
