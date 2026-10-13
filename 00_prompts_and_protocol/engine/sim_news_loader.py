"""
News loader: read annotated news CSVs under data_config/monthly/ and filter by date/stock.
"""

import ast
import csv
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple


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


def _parse_list_field(raw: str) -> List[str]:
    """Parse a string like "['Banking', 'Insurance']" into a Python list."""
    if not raw or raw.strip() in ("[]", ""):
        return []
    try:
        return ast.literal_eval(raw.strip())
    except (ValueError, SyntaxError):
        return []


def load_news_for_month(year: int, month: int, data_dir: str = None) -> List[Dict]:
    """Load all news from a monthly CSV file."""
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data_config", "monthly")
    fname = f"news_annotated_{year}_{month:02d}.csv"
    fpath = os.path.join(data_dir, fname)
    if not os.path.exists(fpath):
        print(f"[NewsLoader] File not found: {fpath}")
        return []

    news = []
    with open(fpath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_str = row.get("timestamp", "")
            try:
                ts = datetime.strptime(ts_str.strip(), "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            news.append({
                "id": row.get("ID", ""),
                "type": row.get("type", ""),
                "scope": row.get("scope", ""),
                "timestamp": ts,
                "date_str": ts.strftime("%Y%m%d"),
                "content": row.get("content", ""),
                "stock": _parse_list_field(row.get("stock", "[]")),
                "ind": _parse_list_field(row.get("ind", "[]")),
                # Keep compatibility with the newer merged dataset fields if present.
                "importance": float(row.get("importance", 0.5) or 0.5),
                "event_type": row.get("event_type", "") or row.get("type", ""),
            })
    print(f"[NewsLoader] Loaded {len(news)} items from {fname}")
    return news


def filter_news_by_date(news: List[Dict], date_str: str) -> List[Dict]:
    """Filter news to only those on a specific date (YYYYMMDD)."""
    return [n for n in news if n["date_str"] == date_str]


def select_for_stock(
    daily_news: List[Dict],
    firm_name: str,
    ind: str,
    n: int = 5,
) -> List[Dict]:
    """
    Select top-n news items relevant to a stock by:
    1. Direct stock mention (firm_name in news['stock'])
    2. Industry match (ind in news['ind'])
    3. Macro news (scope == 'macro')
    """
    scored = []
    for item in daily_news:
        score = 0
        if firm_name in item["stock"]:
            score = 3  # Direct mention
        elif ind and ind in item["ind"]:
            score = 2  # Industry match
        elif item["scope"] == "macro":
            score = 1  # Macro
        if score > 0:
            scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:n]]

def _classify_relevance(item: Dict, firm_name: str, ind: str) -> Tuple[str, str]:
    """
    Returns:
      - scope: stock | industry | macro | other
      - relevance: direct | industry | market | unrelated
    """
    stock_hits = firm_name in (item.get("stock") or [])
    ind_hits = bool(_industry_terms(ind) & set(item.get("ind") or []))
    is_macro = (item.get("scope") == "macro") or (not item.get("stock") and not item.get("ind"))
    if stock_hits:
        return "stock", "direct"
    if ind_hits:
        return "industry", "industry"
    if is_macro:
        return "macro", "market"
    return "other", "unrelated"


def format_news_text(
    news_items: List[Dict],
    *,
    firm_name: str = "",
    ind: str = "",
    current_time: Optional[datetime] = None,
    max_chars: int = 6500,
) -> str:
    """
    Format news items into a structured digest for LLM prompt.

    Key design:
    - Expose metadata so LLM knows *why* this news is included.
    - Avoid over-generalization: explicitly mark scope/relevance.
    """
    if not news_items:
        return "No relevant news"
    lines = []
    total = 0
    for item in news_items:
        ts = item.get("timestamp")
        ts_str = ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else "????-??-?? ??:??"
        importance = float(item.get("importance", 0.5) or 0.5)
        event_type = str(item.get("event_type", "") or item.get("type", "") or "")
        scope, relevance = _classify_relevance(item, firm_name, ind)

        lag_h = None
        if current_time is not None and hasattr(ts, "timestamp"):
            try:
                lag_h = max(0.0, (current_time - ts).total_seconds() / 3600.0)
            except Exception:
                lag_h = None
        lag_str = f"{lag_h:.1f}h" if isinstance(lag_h, (int, float)) else "NA"

        hit_stock = firm_name in (item.get("stock") or [])
        hit_ind = bool(_industry_terms(ind) & set(item.get("ind") or []))
        hit_tags = []
        if hit_stock:
            hit_tags.append("hit_stock")
        if hit_ind:
            hit_tags.append("hit_industry")
        if not hit_tags and scope == "macro":
            hit_tags.append("macro")
        if not hit_tags:
            hit_tags.append("unrelated")

        meta = (
            f"scope={scope}; relevance={relevance}; "
            f"importance={importance:.2f}; event_type={event_type or 'unknown'}; "
            f"time_lag_hours={lag_str}; tags={','.join(hit_tags)}"
        )
        line = f"[{len(lines) + 1:02d}] [{ts_str}] ({meta}) {item.get('content', '')}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines)
