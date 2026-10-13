import math
from datetime import datetime
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Iterable, Tuple


INDUSTRY_ALIASES = {
    "Securities": {"Securities", "Non-bank Finance"},
    "Electrical Equipment": {"Electrical Equipment", "Power Equipment"},
    "Automobiles": {"Automobiles"},
    "Electronic Components": {"Electronic Components", "Electronics"},
    "Liquor": {"Liquor", "Food & Beverage"},
}


def _industry_terms(ind: str) -> set:
    if not ind:
        return set()
    return INDUSTRY_ALIASES.get(ind, {ind})


def load_new_dataset(csv_path: str) -> List[Dict]:
    print(f"[News] Loading new dataset from {csv_path} and parsing llm_json...")
    df = pd.read_csv(csv_path)
    news_list = []
    
    for _, row in df.iterrows():
        ts_str = str(row.get('timestamp', ''))
        try:
            ts = datetime.strptime(ts_str.strip(), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
            
        items = []
        for col in ['llm_items_json', 'sample_items_json', 'llm_json']:
            if pd.notna(row.get(col)) and isinstance(row[col], str):
                try:
                    parsed = json.loads(row[col])
                    if isinstance(parsed, dict) and 'items' in parsed:
                        items = parsed['items']
                        break
                    elif isinstance(parsed, list):
                        items = parsed
                        break
                except json.JSONDecodeError:
                    continue
                    
        if not items:
            continue
            
        for item in items:
            event_type = item.get('event_type', '')
            importance = float(item.get('importance', 0.5))
            stock_list = item.get('stock', [])
            ind_list = item.get('ind', [])
            
            n_item = {
                "id": str(row.get("ID", "")),
                "timestamp": ts,
                "date_str": ts.strftime("%Y%m%d"),
                "content": item.get('clean_content', ''),
                "stock": stock_list,
                "ind": ind_list,
                "scope": "macro" if not stock_list and not ind_list else "micro",
                "importance": importance,
                "event_type": event_type
            }
            news_list.append(n_item)
    print(f"[News] Loaded {len(news_list)} valid news items.")
    return news_list


def load_news_months(csv_paths: Iterable[str]) -> List[Dict]:
    """Load and concatenate multiple merged monthly news CSVs."""
    merged: List[Dict] = []
    for csv_path in csv_paths:
        path = Path(csv_path)
        if not path.exists():
            print(f"[News] Skip missing file: {path}")
            continue
        merged.extend(load_new_dataset(str(path)))
    merged.sort(key=lambda x: (x["timestamp"], x["importance"]), reverse=False)
    return merged


def _is_stock_or_industry_hit(item: Dict, firm_name: str, ind: str) -> bool:
    item_inds = set(item.get("ind", []) or [])
    return (firm_name in item.get("stock", [])) or bool(_industry_terms(ind) & item_inds)


def select_for_stock_day_policy(
    daily_news: List[Dict],
    firm_name: str,
    ind: str,
    current_time: datetime,
) -> List[Dict]:
    """
    Day-level institution news policy.

    - Same-day and next-day news:
      macro news requires importance >= 0.5;
      stock/industry news are kept in full.
    - Yesterday's news:
      macro news requires importance >= 0.7;
      stock/industry news require importance >= 0.5.
    """
    selected: List[Dict] = []
    current_day = current_time.date()

    for item in daily_news:
        item_day = item["timestamp"].date()
        day_diff = (item_day - current_day).days
        importance = float(item.get("importance", 0.5))
        is_hit = _is_stock_or_industry_hit(item, firm_name, ind)
        is_macro = not item.get("stock") and not item.get("ind")

        if day_diff in (0, 1):
            if is_hit or (is_macro and importance >= 0.5):
                selected.append(item)
        elif day_diff == -1:
            if (is_hit and importance >= 0.5) or (is_macro and importance >= 0.7):
                selected.append(item)

    selected.sort(key=lambda x: (x["timestamp"], x["importance"]), reverse=True)
    return selected

def select_for_stock_v3(
    daily_news: List[Dict],
    firm_name: str,
    ind: str,
    current_time: datetime,
) -> List[Dict]:
    """
    New filtering scheme:
    1. Within 1h: importance > 0.5 OR hits industry/stock
    2. Beyond 1h: (importance > 0.5 AND hits industry/stock) OR importance >= 0.7
    """
    selected = []
    
    for item in daily_news:
        is_hit = (firm_name in item["stock"]) or bool(_industry_terms(ind) & set(item.get("ind", []) or []))
        delta_seconds = (current_time - item["timestamp"]).total_seconds()
        delta_hours = delta_seconds / 3600.0
        
        importance = item.get("importance", 0.5)
        
        if delta_hours <= 1.0:
            # Within 1h: keep all news with importance > 0.5 OR any stock/industry hit
            if importance > 0.5 or is_hit:
                selected.append(item)
        else:
            # Beyond 1h: (importance > 0.5 AND stock/industry hit) OR importance >= 0.7
            if (importance > 0.5 and is_hit) or importance >= 0.7:
                selected.append(item)
                
    # Sort by importance and then by time (latest first)
    selected.sort(key=lambda x: (x['importance'], x['timestamp']), reverse=True)
    return selected


def select_for_stock_relevance_tiered(
    daily_news: List[Dict],
    firm_name: str,
    ind: str,
    current_time: datetime,
    *,
    n_total: int = 20,
    n_stock: int = 20,
    n_industry: int = 20,
    n_macro: int = 20,
    min_macro_importance: float = 0.6,
) -> List[Dict]:
    """
    Relevance-first selection (recommended):
      stock direct hit > industry fill > macro fallback > drop unrelated.

    Within each tier, rank primarily by importance, then recency.
    The caller is responsible for passing a trading-window slice, typically
    previous trading-day close to current trading-day close. We intentionally
    avoid continuous hour-level decay here: weekend or holiday news should not
    be suppressed merely because the market was closed.

    The default 20-item budget is allocated sequentially rather than by fixed
    quota: keep all available stock-level news first, fill the remaining slots
    with industry news, and only use high-importance macro news when stock and
    industry news are insufficient.
    """
    def _score(item: Dict) -> float:
        return float(item.get("importance", 0.5) or 0.5)

    def _rank_key(item: Dict):
        return (_score(item), item.get("timestamp"))

    stock_hits: List[Dict] = []
    ind_hits: List[Dict] = []
    macro_hits: List[Dict] = []

    for item in daily_news:
        stocks = item.get("stock") or []
        inds = item.get("ind") or []
        is_stock = firm_name in stocks
        is_ind = bool(_industry_terms(ind) & set(inds))
        is_macro = (not stocks and not inds) or (item.get("scope") == "macro")
        if is_stock:
            stock_hits.append(item)
        elif is_ind:
            ind_hits.append(item)
        elif is_macro and float(item.get("importance", 0.5) or 0.5) >= min_macro_importance:
            macro_hits.append(item)

    stock_hits.sort(key=_rank_key, reverse=True)
    ind_hits.sort(key=_rank_key, reverse=True)
    macro_hits.sort(key=_rank_key, reverse=True)

    selected: List[Dict] = []
    selected.extend(stock_hits[:min(n_stock, n_total)])

    slots = max(0, n_total - len(selected))
    if slots:
        selected.extend(ind_hits[:min(n_industry, slots)])

    slots = max(0, n_total - len(selected))
    if slots:
        selected.extend(macro_hits[:min(n_macro, slots)])

    selected = selected[:n_total]
    selected.sort(key=lambda x: (float(x.get("importance", 0.5) or 0.5), x.get("timestamp")), reverse=True)
    return selected
