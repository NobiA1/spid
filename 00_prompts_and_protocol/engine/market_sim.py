#!/usr/bin/env python3
"""
market_sim.py — Self-contained market simulation orchestrator.

Usage:
    python3 run_code/market_sim.py --start 20250102 --end 20250106 [--mock-llm] [--seed 42]

Outputs saved to output/:
    index_comparison.png, orders.json, trades.json,
    wealth_summary.json, gini_history.json
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Allow running from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from run_code.simulation.market_engine import (
    UnifiedMarket, StockInfo, AgentOrder, Direction,
)
from run_code.utils.market_data_loader import (
    load_stock_list, fetch_daily_prices, fetch_circ_mv,
    fetch_fina_indicator, fetch_macro_rate,
    get_trading_dates, get_close_price, get_history_before,
)
from run_code.utils.sim_news_loader import load_news_for_month, filter_news_by_date, select_for_stock, format_news_text
from run_code.utils.sim_news_filter import (
    load_new_dataset,
    select_for_stock_day_policy,
    select_for_stock_v3,
    select_for_stock_relevance_tiered,
)
from run_code.simulation.agent_behaviors import (
    RetailAgent, InstitutionAgent, RetailFlowStats, CIODecision,
    initialize_agents, allocate_budgeted_shares,
    value_compute_R_hat, momentum_compute_signals, momentum_compute_R_hat,
    noise_compute_S_ext, noise_update_rho,
    apply_signal_pipeline, generate_retail_order, sum_retail_order_prefilter,
)
from run_code.simulation.social_forum import ForumModule
from run_code.utils.llm_router import (
    call_value_agent_llm, call_momentum_agent_llm, call_noise_agent_llm,
    call_inst_fundamental_quarterly_llm, call_inst_fundamental_daily_llm,
    call_inst_quant_llm, call_inst_sentiment_llm, call_inst_cio_llm,
)
from run_code.baselines.stockagent_agents import call_stockagent_retail
from run_code.baselines.twinmarket_agents import call_twinmarket_retail
from run_code.baselines.common import result_from_payload
import concurrent.futures
from run_code.utils.performance_metrics import (
    compute_gini, compute_equal_weight_index, plot_comparison,
    compute_weighted_market_index, plot_weighted_comparison,
    export_orders, export_trades, export_wealth_summary, export_gini_history,
)
from run_code.simulation.market_maker import VirtualMarketMaker
from run_code.utils.simulation_audit import WeeklyMarketSupervisor



DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

NO_NEWS_TECHNICAL_ONLY_PLACEHOLDER = (
    "No news (control experiment: the real news stream is cut off; the LLM decides based only on technical signals such as price/volume)."
)


def _suppress_news_for_llm(ablation_mode: str) -> bool:
    return ablation_mode == "no_news_technical_only"


def _suppress_forum(ablation_mode: str) -> bool:
    return ablation_mode == "no_forum"


def iter_months(start_date: str, end_date: str) -> List[Tuple[int, int]]:
    """Return inclusive (year, month) pairs covering the date range."""
    dt_start = datetime.strptime(start_date, "%Y%m%d")
    dt_end = datetime.strptime(end_date, "%Y%m%d")
    cursor = datetime(dt_start.year, dt_start.month, 1)
    months = []
    while cursor <= dt_end:
        months.append((cursor.year, cursor.month))
        if cursor.month == 12:
            cursor = datetime(cursor.year + 1, 1, 1)
        else:
            cursor = datetime(cursor.year, cursor.month + 1, 1)
    return months


def load_monthly_news_range(start_date: str, end_date: str, news_root: str) -> Dict[Tuple[int, int], List[Dict]]:
    """Load all merged monthly news CSVs in the target date range."""
    cache: Dict[Tuple[int, int], List[Dict]] = {}
    for year, month in iter_months(start_date, end_date):
        fpath = os.path.join(news_root, "merged", f"{year}-{month:02d}.csv")
        cache[(year, month)] = load_new_dataset(fpath) if os.path.exists(fpath) else []
    return cache


def compute_day_aggregates(
    value_signals: Dict[str, dict],
    momentum_signals_data: Dict[str, dict],
    noise_signals_data: Dict[str, dict],
    obi_by_stock: Dict[str, float],
    step_orders: List,
    step_trades: List,
) -> Dict[str, float]:
    """Aggregate signals for weekly supervision."""
    def _mean(values: List[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    return {
        "mean_value_signal": _mean([v.get("delta_earnings", 0.0) + v.get("delta_quality", 0.0) + v.get("macro_signal", 0.0) for v in value_signals.values()]),
        "mean_momentum_signal": _mean([m.get("catalyst_signal", 0.0) + m.get("diffusion_signal", 0.0) for m in momentum_signals_data.values()]),
        "mean_noise_signal": _mean([n.get("forum_sentiment", 0.0) + n.get("news_reaction", 0.0) for n in noise_signals_data.values()]),
        "mean_obi": _mean(list(obi_by_stock.values())),
        "order_count": len(step_orders),
        "trade_count": len(step_trades),
    }


def compute_obi_from_snapshot(snapshot: Optional[dict]) -> float:
    """
    OBI (Order Book Imbalance):
        (sum_bid_vol - sum_ask_vol) / (sum_bid_vol + sum_ask_vol)
    Returns 0 when snapshot/depth is unavailable.
    """
    if not snapshot:
        return 0.0
    ob = snapshot.get("order_book", {})
    bids = ob.get("bid", []) or []
    asks = ob.get("ask", []) or []
    bid_vol = sum((lvl.get("vol", {}) or {}).get("value", 0) for lvl in bids)
    ask_vol = sum((lvl.get("vol", {}) or {}).get("value", 0) for lvl in asks)
    denom = bid_vol + ask_vol
    if denom <= 0:
        return 0.0
    return (bid_vol - ask_vol) / denom


def build_intraday_times(steps_per_day: int) -> List[str]:
    """Return equally spaced observation endpoints over A-share trading sessions."""
    steps = max(1, int(steps_per_day or 1))
    if steps == 1:
        return ["15:00:00"]

    sessions = [
        (datetime.strptime("09:30:00", "%H:%M:%S"), datetime.strptime("11:30:00", "%H:%M:%S")),
        (datetime.strptime("13:00:00", "%H:%M:%S"), datetime.strptime("15:00:00", "%H:%M:%S")),
    ]
    total_minutes = sum(int((end - start).total_seconds() // 60) for start, end in sessions)
    raw_counts = [steps * int((end - start).total_seconds() // 60) / total_minutes for start, end in sessions]
    counts = [max(1, int(math.floor(x))) for x in raw_counts]
    while sum(counts) < steps:
        residuals = [raw_counts[i] - math.floor(raw_counts[i]) for i in range(len(raw_counts))]
        counts[max(range(len(counts)), key=lambda i: residuals[i])] += 1
    while sum(counts) > steps:
        idx = max(range(len(counts)), key=lambda i: counts[i])
        if counts[idx] > 1:
            counts[idx] -= 1
        else:
            break

    times: List[str] = []
    for (start, end), count in zip(sessions, counts):
        span_seconds = (end - start).total_seconds()
        for i in range(1, count + 1):
            t = start + timedelta(seconds=span_seconds * i / count)
            times.append(t.strftime("%H:%M:%S"))
    return times[:steps]


# ═══════════════════════════════════════════════════════════════════════════
# Mock LLM signals (used when --mock-llm flag is set)
# ═══════════════════════════════════════════════════════════════════════════

def mock_value_signals(stock: str, rng: random.Random) -> dict:
    """Mock LLM output for value agent daily analysis."""
    return {
        "delta_earnings": rng.uniform(-0.15, 0.15),
        "delta_quality": rng.uniform(-0.10, 0.10),
        "macro_signal": rng.uniform(-0.20, 0.20),
        "post_text": "Valuation reasonable, watching marginal fundamental changes",
    }


def mock_momentum_signals(stock: str, rng: random.Random) -> dict:
    """Mock LLM output for momentum agent catalyst analysis."""
    return {
        "catalyst_signal": rng.uniform(-0.3, 0.3),
        "diffusion_signal": rng.uniform(-0.2, 0.2),
        "post_text": "Trend in progress, volume support is acceptable",
    }


def mock_noise_signals(stock: str, rng: random.Random) -> dict:
    """Mock LLM output for noise agent sentiment extraction."""
    return {
        "forum_sentiment": rng.uniform(-0.5, 0.5),
        "news_reaction": rng.uniform(-0.3, 0.3),
        "post_text": "Feels like it's going up, going in!" if rng.random() > 0.5 else "Not good, getting out!",
    }


# ── Mock Institution Sub-Agent LLM outputs ──

def mock_inst_fundamental_quarterly(stock: str, rng: random.Random) -> dict:
    """Mock quarterly fundamental cache."""
    return {
        "fundamental_quality": rng.uniform(-0.3, 0.5),
        "fair_pb_mid": rng.uniform(0.8, 2.5),
        "fair_pb_range": [rng.uniform(0.5, 1.0), rng.uniform(1.5, 3.0)],
        "key_expectation_factors": ["policy change", "industry competitive landscape", "macro rate environment"],
        "risk_flag": rng.uniform(0.1, 0.5),
        "context_summary": "Fundamental quality above average; watch policy and industry marginal changes",
    }


def mock_inst_fundamental_daily(stock: str, rng: random.Random) -> dict:
    """Mock daily expectation delta."""
    return {
        "expectation_delta": rng.uniform(-0.4, 0.4),
        "delta_confidence": rng.uniform(0.2, 0.8),
        "catalyst_type": rng.choice(["policy", "industry", "company_event", "macro", "none"]),
    }


def mock_inst_quant(stock: str, rng: random.Random) -> dict:
    """Mock quantitative strategist output."""
    return {
        "technical_signal": rng.uniform(-0.5, 0.5),
        "capital_flow_signal": rng.uniform(-0.4, 0.4),
        "volatility_forecast": rng.uniform(0.1, 0.6),
        "key_levels": {"support": rng.uniform(-0.05, -0.01), "resistance": rng.uniform(0.01, 0.05)},
        "context_summary": "Technicals neutral-to-bullish, no clear capital-flow anomaly",
    }


def mock_inst_sentiment(stock: str, rng: random.Random) -> dict:
    """Mock sentiment & flow analyst output."""
    return {
        "retail_sentiment_extreme": rng.uniform(-0.5, 0.5),
        "retail_pain_signal": rng.uniform(-0.3, 0.3),
        "contrarian_opportunity": rng.uniform(0.0, 0.5),
        "context_summary": "Retail sentiment neutral, no clear extreme signal yet",
    }


def mock_inst_cio(stock: str, sub_reports: dict, rng: random.Random) -> dict:
    """Mock CIO decision based on sub-agent reports."""
    # Simple heuristic: combine signals
    exp_delta = sub_reports.get("expectation_delta", 0)
    tech_sig = sub_reports.get("technical_signal", 0)
    retail_ext = sub_reports.get("retail_sentiment_extreme", 0)
    contra = sub_reports.get("contrarian_opportunity", 0)

    # Blended signal
    combined = 0.4 * exp_delta + 0.3 * tech_sig - 0.2 * retail_ext + 0.1 * contra
    combined += rng.gauss(0, 0.05)

    if abs(combined) < 0.10:
        action = "hold"
        delta_w = 0.0
    elif combined > 0:
        action = "buy"
        delta_w = min(0.05, combined * 0.10)
    else:
        action = "sell"
        delta_w = max(-0.05, combined * 0.10)

    return {
        "action": action,
        "conviction": min(1.0, abs(combined) * 2),
        "target_weight_delta": round(delta_w, 4),
        "urgency": min(1.0, abs(combined) * 1.5),
        "risk_override": False,
        "reasoning": f"combined signal={combined:.2f}, {'increase' if action=='buy' else 'reduce' if action=='sell' else 'hold'}",
    }


def select_for_stock_v3(daily_news: List[Dict], firm_name: str, ind: str, current_time: datetime) -> List[Dict]:
    """
    Intraday-frequency news selection:
    1. Within 1h (delta_hours <= 1.0): importance > 0.5 OR stock/industry hit
    2. Beyond 1h: (importance > 0.5 AND stock/industry hit) OR importance >= 0.7
    """
    selected = []
    for item in daily_news:
        is_hit = (firm_name in item["stock"]) or (ind in item["ind"])
        delta_seconds = (current_time - item["timestamp"]).total_seconds()
        delta_hours = delta_seconds / 3600.0
        
        importance = item.get("importance", 0.5)
        
        if delta_hours <= 1.0:
            if importance > 0.5 or is_hit:
                selected.append(item)
        else:
            if (importance > 0.5 and is_hit) or importance >= 0.7:
                selected.append(item)
                
    selected.sort(key=lambda x: (x['importance'], x['timestamp']), reverse=True)
    return selected


# ═══════════════════════════════════════════════════════════════════════════
# Main Simulation
# ═══════════════════════════════════════════════════════════════════════════

def run_simulation(
    start_date: str = "20250102",
    end_date: str = "20250630",
    seed: int = 42,
    mock_llm: bool = True,
    N_retail: int = 300,
    N_inst: int = 1,
    stock_universe_csv: Optional[str] = None,
    news_root: Optional[str] = None,
    output_dir: Optional[str] = None,
    inst_early_news: int = 1,
    retail_mode: str = "signal",  # "signal" (default) | "stockagent" | "twinmarket"
    prompt_baseline_workers: int = 10,
    record_prompts: bool = True,
    max_stocks: int = 56,
    full_decision_log: bool = False,
    llm_workers: int = 10,
    ablation_mode: str = "none",
    semantic_cache_run: Optional[str] = None,
    institution_cache_run: Optional[str] = None,
    prompt_baseline_cache_run: Optional[str] = None,
    minimal_output: bool = False,
    lob_snapshot: bool = False,
    intraday_steps_per_day: int = 1,
):
    t0 = time.time()
    rng = random.Random(seed)
    np.random.seed(seed)
    out_dir = output_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    retail_decisions_jsonl = os.path.join(out_dir, "retail_decisions.jsonl")
    if full_decision_log:
        with open(retail_decisions_jsonl, "w", encoding="utf-8") as f:
            f.write("")

    print("=" * 60)
    print("SUM Market Simulation V2")
    print(f"  Period: {start_date} → {end_date}")
    print(f"  Seed: {seed}   Mock LLM: {mock_llm}   Retail Mode: {retail_mode}")
    print(f"  Retail Agents: {N_retail}   Institutions: {N_inst}")
    print(f"  Prompt Baseline Workers: {prompt_baseline_workers}   Record Prompts: {record_prompts}")
    print(f"  SUM LLM Workers: {llm_workers}")
    print(f"  Intraday Steps Per Day: {intraday_steps_per_day}")
    print(f"  Max Stocks: {max_stocks}   Full Decision Log: {full_decision_log}")
    print(f"  Minimal Output: {minimal_output}")
    print(f"  Ablation Mode: {ablation_mode}")
    if semantic_cache_run:
        print(f"  Semantic Cache Run: {semantic_cache_run}")
    if institution_cache_run:
        print(f"  Institution Cache Run: {institution_cache_run}")
    if prompt_baseline_cache_run:
        print(f"  Prompt Baseline Cache Run: {prompt_baseline_cache_run}")
    if stock_universe_csv:
        print(f"  Universe CSV: {stock_universe_csv}")
    if news_root:
        print(f"  News Root:    {news_root}")
    print("=" * 60)

    print("\n--- [News feed scheme] ---")
    print("Goal: keep enough important news when the context budget allows, avoiding recent intraday news overwriting overnight/holiday risk.")
    print("Time window: sliced by observation frequency; daily level uses previous trading day 15:00 to current 15:00 by default; intraday experiments use previous observation point to current one.")
    print("Selection: relevance-first (keep stock news first, fill remaining slots with industry news; fall back to high-importance macro only when stock/industry news is insufficient).")
    print("Context size: keep at most 20 structured digest items by default, including scope/relevance/importance/event_type/time_lag_hours.")
    print("Parsing: read llm_items_json / llm_json, extract their items structure, and recompose to obtain more accurate fine-grained attributes (including importance and stock classification).")
    print("------------------------------------\n")

    # ─── Step 1: Load stock list ───
    stock_list = load_stock_list(stock_universe_csv)
    stk_codes = [s["stk_code"] for s in stock_list]
    firm_map = {s["stk_code"]: s["firm_name"] for s in stock_list}
    ind_map = {s["stk_code"]: s["ind"] for s in stock_list}
    print(f"\n[Init] {len(stk_codes)} stocks loaded")

    # ─── Step 2: Fetch market data ───
    print("[Init] Fetching price data from the market-data provider...")
    token = None
    price_data = fetch_daily_prices(stk_codes, start_date, end_date, token, pre_days=60)

    # Filter out stocks with no data and cap universe size for controlled runs.
    valid_stk = [s for s in stk_codes if s in price_data and len(price_data[s]) > 0][:max_stocks]
    stk_codes = valid_stk
    stock_list = [s for s in stock_list if s["stk_code"] in valid_stk]
    print(f"[Init] {len(stk_codes)} stocks selected (truncated to {max_stocks})")

    trading_dates = get_trading_dates(price_data, start_date, end_date)
    print(f"[Init] Trading dates: {trading_dates}")

    if not trading_dates:
        print("[ERROR] No trading dates found. Exiting.")
        return

    # Initial prices = close on first trading date (or last pre-sim day)
    first_date = trading_dates[0]
    # Use the day before first_date as initial price reference
    prices: Dict[str, float] = {}
    for stk in stk_codes:
        hist = get_history_before(price_data, stk, first_date, 1)
        if hist:
            prices[stk] = hist[0]["close"]
        else:
            price_row = price_data.get(stk, {}).get(first_date)
            prices[stk] = price_row["open"] if price_row else 100.0

    print(f"[Init] Initial prices loaded (sample: {list(prices.items())[:3]})")

    # Fetch financial indicators (for value agents)
    print("[Init] Fetching financial indicators...")
    fina_data = fetch_fina_indicator(stk_codes)

    # Macro rate
    macro = fetch_macro_rate()
    g_B = 0.7 * macro["rate_10y"] + 0.3 * macro["lpr_1y"]
    print(f"[Init] g_B = {g_B:.4f}")

    # Circulating market cap for synthetic float weighting
    circ_mv = fetch_circ_mv(stk_codes, first_date, token)
    print(f"[Init] circ_mv fetched for {len(circ_mv)} stocks")

    # ─── Step 3: Initialize agents ───
    print("[Init] Initializing agents...")
    retail_agents, inst_agents = initialize_agents(
        stock_list, prices, price_data, first_date, rng, circ_mv=circ_mv, N=N_retail, N_inst=N_inst,
    )
    ablation_meta: Dict[str, Any] = {"mode": ablation_mode}

    if ablation_mode == "micro_default_gamma_sigma":
        for agent in retail_agents:
            agent.sigma = 0.0175
            agent.gamma_0 = 0.08
            agent.gamma_t = 0.08
        ablation_meta.update({"retail_sigma": 0.0175, "retail_gamma_0": 0.08, "retail_gamma_t": 0.08})

    if ablation_mode == "macro_sum_institution":
        total_inst_wealth = sum(ia.initial_wealth for ia in inst_agents)
        stock_weights = {s: 1.0 for s in stk_codes}
        macro_inst_agents: List[RetailAgent] = []
        sleeve_specs = [("value", 3), ("momentum", 3), ("noise", 4)]
        sleeve_idx = 0
        for style, count in sleeve_specs:
            for _ in range(count):
                W0 = total_inst_wealth * count / 10.0 / count
                alpha_stock = {"value": 0.60, "momentum": 0.40, "noise": 0.50}[style]
                holdings, total_invested = allocate_budgeted_shares(W0 * alpha_stock, stock_weights, prices)
                if style == "value":
                    psi = 0.50 + 0.45 * rng.betavariate(5, 2)
                elif style == "momentum":
                    psi = 0.35 + 0.60 * rng.betavariate(4, 2)
                else:
                    psi = 0.10 + 0.70 * rng.betavariate(2, 5)
                sigma = rng.uniform(0.005, 0.030)
                gamma_0 = rng.uniform(0.04, 0.12)
                inst = RetailAgent(
                    agent_id=f"INST_SUM_{style.upper()}_{sleeve_idx:02d}",
                    style=style,
                    psi=psi,
                    sigma=sigma,
                    gamma_0=gamma_0,
                    gamma_t=gamma_0,
                    holdings=holdings,
                    cash=W0 - total_invested,
                    initial_wealth=W0,
                    stock_list=list(stk_codes),
                    rho=0.0,
                )
                inst.wealth_history = [W0]
                macro_inst_agents.append(inst)
                sleeve_idx += 1
        inst_agents = macro_inst_agents
        ablation_meta.update({
            "institution_architecture": "SUM_retail_signal_sleeves",
            "institution_sleeves": len(inst_agents),
            "institution_principal_split": "3:3:4 across value:momentum:noise",
            "institution_total_initial_wealth": total_inst_wealth,
        })

    if ablation_mode == "no_news_technical_only":
        ablation_meta.update({
            "news_input": "suppressed",
            "technical_input": "preserved",
            "forum_input": "preserved",
            "inst_early_news_effective": False,
            "description": (
                "Cut real news flow to all LLM perception layers; "
                "keep price/volume/OBI/vol_ratio/kline signals and simulated forum."
            ),
        })

    if ablation_mode == "no_forum":
        ablation_meta.update({
            "news_input": "preserved",
            "technical_input": "preserved",
            "forum_input": "suppressed",
            "forum_publication": "disabled",
            "description": (
                "Cut endogenous forum information diffusion: no prior posts or forum "
                "counts are injected into retail/institution prompts, no new posts are "
                "published, and forum_sentiment is clamped to zero."
            ),
        })

    n_val = sum(1 for a in retail_agents if a.style == "value")
    n_mom = sum(1 for a in retail_agents if a.style == "momentum")
    n_noi = sum(1 for a in retail_agents if a.style == "noise")
    print(f"[Init] Retail: Value={n_val} Momentum={n_mom} Noise={n_noi}")
    total_retail_wealth = sum(a.get_wealth(prices) for a in retail_agents)
    total_inst_wealth = sum(ia.get_wealth(prices) for ia in inst_agents)
    print(f"[Init] Total retail wealth: ¥{total_retail_wealth:,.0f}")
    print(f"[Init] Total institution wealth: ¥{total_inst_wealth:,.0f}")

    # ─── Step 4: Compute synthetic float shares & VMM ───
    print("[Init] Computing synthetic float shares...")
    W_total = total_retail_wealth + total_inst_wealth
    weight_base = {
        s: max(circ_mv.get(s, prices.get(s, 0.01) * 1e6), 0.01)
        for s in stk_codes
    }
    log_caps = {s: math.log(v) for s, v in weight_base.items()}
    total_log = sum(log_caps.values())
    log_weights = {
        s: log_caps[s] / total_log if total_log > 0 else 1.0 / len(stk_codes)
        for s in stk_codes
    }

    float_shares: Dict[str, int] = {}
    for s in stk_codes:
        w = log_weights[s]
        p = max(prices[s], 0.01)
        float_shares[s] = max(1, int(w * W_total / p))

    # Aggregate agent holdings
    agent_holdings: Dict[str, int] = {}
    for a in retail_agents:
        for s, q in a.holdings.items():
            agent_holdings[s] = agent_holdings.get(s, 0) + q
    for ia in inst_agents:
        for s, q in ia.holdings.items():
            agent_holdings[s] = agent_holdings.get(s, 0) + q

    # Validate and fix float shares
    for s in stk_codes:
        if agent_holdings.get(s, 0) > float_shares.get(s, 0):
            float_shares[s] = agent_holdings[s] + 100  # ensure VMM gets some

    vmm = VirtualMarketMaker()
    vmm.initialize(float_shares, agent_holdings, log_weights)
    vmm_total = sum(vmm.holdings.values())
    print(f"[Init] VMM initialized: {vmm_total} total shares across {sum(1 for v in vmm.holdings.values() if v > 0)} stocks")

    # ─── Step 5: Initialize market engine ───
    market = UnifiedMarket()
    for s in stk_codes:
        market.register_stock(StockInfo(
            stock_code=s,
            name=firm_map.get(s, s),
            prev_close=prices[s],
            float_shares=float_shares.get(s, 10000),
        ))

    # Register cash and positions for all agents
    for a in retail_agents:
        market.set_cash(a.agent_id, a.cash)
        for s, q in a.holdings.items():
            market.set_position(a.agent_id, s, q)

    for ia in inst_agents:
        market.set_cash(ia.agent_id, ia.cash)
        for s, q in ia.holdings.items():
            market.set_position(ia.agent_id, s, q)

    market.set_cash(vmm.agent_id, 1e12)  # VMM unlimited cash for buying
    for s, q in vmm.holdings.items():
        market.set_position(vmm.agent_id, s, q)

    # ─── Step 6: Initialize forum & news ───
    forum = ForumModule()
    news_cache = {}

    forum_disabled = _suppress_forum(ablation_mode)

    def _forum_posts_text(stock: str, current_day: int, reader_type: str) -> str:
        if forum_disabled:
            return "No posts yet (control experiment: forum information diffusion is cut off)"
        return forum.get_posts_text(stock, current_day, reader_type)

    def _forum_stats(stock: str, current_day: int) -> Dict[str, int]:
        if forum_disabled:
            return {"n_posts": 0, "n_bull": 0, "n_bear": 0, "n_neutral": 0}
        return forum.get_stats(stock, current_day)

    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data_config"))
    news_root = news_root or os.path.join(data_root, "news")
    print(f"\n[News] Loading merged monthly news from: {news_root}")
    news_cache = load_monthly_news_range(start_date, end_date, news_root)
    total_news_items = sum(len(v) for v in news_cache.values())
    print(f"[News] Loaded {total_news_items} parsed news items across {len(news_cache)} months")
    all_news_obj = [item for items in news_cache.values() for item in items]
    all_news_obj.sort(key=lambda x: x["timestamp"])

    semantic_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    semantic_cache_hits = 0
    semantic_cache_misses = 0
    if semantic_cache_run:
        cache_path = Path(semantic_cache_run) / "shared_semantics.json"
        if not cache_path.is_absolute():
            cache_path = Path.cwd() / cache_path
        with cache_path.open("r", encoding="utf-8") as f:
            cached_records = json.load(f)
        for record in cached_records:
            key = (str(record.get("date", "")), str(record.get("stock", "")))
            semantic_cache[key] = record
        print(f"[SemanticCache] Loaded {len(semantic_cache)} stock-day records from {cache_path}")

    institution_cache: Dict[Tuple[int, str], Dict[str, Any]] = {}
    institution_cache_hits = 0
    institution_cache_misses = 0
    if institution_cache_run:
        inst_cache_path = Path(institution_cache_run) / "inst_decisions.json"
        if not inst_cache_path.is_absolute():
            inst_cache_path = Path.cwd() / inst_cache_path
        with inst_cache_path.open("r", encoding="utf-8") as f:
            cached_inst_records = json.load(f)
        for record in cached_inst_records:
            key = (int(record.get("step", 0)), str(record.get("stock", "")))
            institution_cache[key] = record
        print(f"[InstitutionCache] Loaded {len(institution_cache)} stock-step records from {inst_cache_path}")

    prompt_baseline_cache: Dict[Tuple[int, str, str, str], Dict[str, Any]] = {}
    prompt_baseline_cache_hits = 0
    prompt_baseline_cache_misses = 0
    if prompt_baseline_cache_run:
        cache_run_dir = Path(prompt_baseline_cache_run)
        if not cache_run_dir.is_absolute():
            cache_run_dir = Path.cwd() / cache_run_dir
        prompt_cache_path = cache_run_dir / "retail_decisions.json"
        cache_meta_path = cache_run_dir / "run_metadata.json"
        cached_mode = retail_mode
        if cache_meta_path.exists():
            with cache_meta_path.open("r", encoding="utf-8") as f:
                cache_meta = json.load(f)
            cached_mode = str(cache_meta.get("retail_mode", retail_mode))
        if cached_mode != retail_mode:
            raise ValueError(
                "Cross-baseline prompt cache is disabled: "
                f"cache run mode={cached_mode!r} but current retail_mode={retail_mode!r}. "
                "Only same-framework replay is supported (e.g. stockagent -> stockagent)."
            )
        with prompt_cache_path.open("r", encoding="utf-8") as f:
            cached_prompt_records = json.load(f)
        skipped_cross_mode = 0
        for record in cached_prompt_records:
            record_mode = str(record.get("mode", retail_mode))
            if record_mode != retail_mode:
                skipped_cross_mode += 1
                continue
            key = (
                int(record.get("step", 0)),
                str(record.get("agent_id", "")),
                str(record.get("stock", "")),
                record_mode,
            )
            prompt_baseline_cache[key] = record
        print(
            f"[PromptBaselineCache] Loaded {len(prompt_baseline_cache)} retail decisions "
            f"from {prompt_cache_path} (mode={retail_mode})"
        )
        if skipped_cross_mode:
            print(
                f"[PromptBaselineCache] Skipped {skipped_cross_mode} cross-mode records "
                f"(cross-baseline replay disabled)."
            )

    # ─── Step 7: Validation ───
    initial_gini = compute_gini([a.get_wealth(prices) for a in retail_agents])
    print(f"\n[Validation] Initial Gini: {initial_gini:.4f}")
    print(f"[Validation] Stock count: {len(stk_codes)}")

    # ════════════════════════════════════════════════════════════════════
    #  SIMULATION LOOP
    # ════════════════════════════════════════════════════════════════════

    all_orders_log = []
    all_trades_log = []
    all_agent_states_log = []
    all_order_attempts_log = []
    all_retail_decisions_log = []
    all_shared_semantics_log = []
    all_daily_assets_log = []
    all_step_metrics_log = []
    lob_snapshot_log: list | None = [] if lob_snapshot else None
    cumulative_order_attempts = 0
    cumulative_orders = 0
    cumulative_trades = 0
    total_retail_decisions_logged = 0
    total_llm_tokens = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_llm_calls = 0
    total_failed_llm_calls = 0
    gini_history = [{"step": 0, "date": "init", "gini": initial_gini}]
    sim_prices_history: Dict[str, List[float]] = {s: [prices[s]] for s in stk_codes}
    real_prices_history: Dict[str, List[float]] = {s: [prices[s]] for s in stk_codes}
    weighted_sim_index_history = [1000.0]
    weighted_real_index_history = [1000.0]
    weekly_supervisor = WeeklyMarketSupervisor(weight_base)
    weekly_snapshots: List[Dict] = []
    weekly_dates_buffer: List[str] = []
    available_trading_dates = sorted({
        d
        for s in stk_codes
        for d in price_data.get(s, {}).keys()
        if d <= end_date
    })

    def _previous_available_trading_date(date_str: str) -> Optional[str]:
        prev = [d for d in available_trading_dates if d < date_str]
        return prev[-1] if prev else None

    # Pre-compute v_bar for value agents (long-run mean of ln(bps/P))
    v_bar_cache: Dict[str, float] = {}
    for s in stk_codes:
        bps = fina_data.get(s, {}).get("bps", 1.0)
        v_bar_cache[s] = math.log(max(bps, 0.01) / max(prices[s], 0.01))

    print(f"\n{'='*60}")
    print(
        f"Starting simulation: {len(trading_dates)} trading days, "
        f"{max(1, int(intraday_steps_per_day or 1))} matching steps/day"
    )
    print(f"{'='*60}\n")

    sim_step = 0

    def _append_retail_decisions_jsonl(records: List[Dict[str, Any]]) -> None:
        if not full_decision_log or not records:
            return
        with open(retail_decisions_jsonl, "a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _split_order_for_intraday(order: AgentOrder, n_slices: int, slice_idx: int) -> AgentOrder:
        n = max(1, int(n_slices))
        return AgentOrder(
            stock=order.stock,
            direction=order.direction,
            price_type=order.price_type,
            price_offset=order.price_offset,
            deal_amount=float(order.deal_amount) / n,
            limit_price=order.limit_price,
            quantity=None,
            agent_id=order.agent_id,
            client_order_id=f"{order.client_order_id}-IT{slice_idx:02d}",
            reasoning=f"Intraday cache slice {slice_idx}/{n}: {order.reasoning}",
        )

    for day_idx, date in enumerate(trading_dates):
        print(f"\n>>> Day {day_idx+1}: {date}")
        
        # End of day sync (for real price tracking and gini)
        # Real prices for today
        real_today: Dict[str, float] = {}
        for s in stk_codes:
            price_row = price_data.get(s, {}).get(date)
            real_today[s] = price_row["close"] if price_row else prices.get(s, 100)

        day_value_signal_means: List[float] = []
        day_momentum_signal_means: List[float] = []
        day_noise_signal_means: List[float] = []
        day_obi_means: List[float] = []
        day_order_count = 0
        day_trade_count = 0
        day_llm_tokens_start = total_llm_tokens
        day_llm_calls_start = total_llm_calls
        day_prompt_tokens_start = total_prompt_tokens
        day_completion_tokens_start = total_completion_tokens
        day_failed_llm_calls_start = total_failed_llm_calls

        def _asset_snapshot(agent: Any) -> Dict[str, Any]:
            holdings = {s: int(q) for s, q in getattr(agent, "holdings", {}).items() if q}
            wealth = float(agent.get_wealth(prices))
            position_value = float(sum(q * prices.get(s, 0.0) for s, q in holdings.items()))
            return {
                "wealth": wealth,
                "cash": float(getattr(agent, "cash", 0.0)),
                "position_value": position_value,
                "position_ratio": position_value / wealth if wealth > 0 else 0.0,
                "holdings": holdings,
                "n_positions": len(holdings),
            }

        day_start_assets: Dict[str, Dict[str, Any]] = {}
        for a in retail_agents:
            day_start_assets[a.agent_id] = _asset_snapshot(a)
        for ia in inst_agents:
            day_start_assets[ia.agent_id] = _asset_snapshot(ia)
        day_start_assets[vmm.agent_id] = _asset_snapshot(vmm)

        prev_date = trading_dates[day_idx - 1] if day_idx - 1 >= 0 else _previous_available_trading_date(date)
        intraday_times = build_intraday_times(intraday_steps_per_day)
        news_interval_start = (
            datetime.strptime(f"{prev_date} 15:00:00", "%Y%m%d %H:%M:%S")
            if prev_date is not None
            else datetime.strptime(f"{date} 00:00:00", "%Y%m%d %H:%M:%S")
        )

        for intraday_idx, h_str in enumerate(intraday_times, start=1):
            is_day_final_step = intraday_idx == len(intraday_times)
            # Intraday matching: each observation gets a fresh news slice and LOB match.
            step_t0 = time.time()
            current_time = datetime.strptime(f"{date} {h_str}", "%Y%m%d %H:%M:%S")
            sim_step += 1
            step_llm_tokens_start = total_llm_tokens
            step_llm_calls_start = total_llm_calls
            step_prompt_tokens_start = total_prompt_tokens
            step_completion_tokens_start = total_completion_tokens
            step_failed_llm_calls_start = total_failed_llm_calls

            # ── Market state layer (for prompts) ──
            # Use real market breadth to avoid feedback loops from simulated prices.
            news_window_start = news_interval_start

            # Load only the news slice newly available at this observation time.
            daily_news = [
                n for n in all_news_obj
                if news_window_start < n["timestamp"] <= current_time
            ]
            news_interval_start = current_time

            rets = []
            adv = dec = 0
            if prev_date is not None:
                for s in stk_codes:
                    p0 = price_data.get(s, {}).get(prev_date, {}).get("close", prices.get(s, 0.0))
                    p1 = price_data.get(s, {}).get(date, {}).get("close", prices.get(s, 0.0))
                    if p0 and p1:
                        r = p1 / max(p0, 0.01) - 1.0
                        rets.append(r)
                        if r > 0:
                            adv += 1
                        elif r < 0:
                            dec += 1
            mean_ret = float(np.mean(rets)) if rets else 0.0
            breadth = (adv - dec) / max((adv + dec), 1)
            # Vol/OBI will be filled below; placeholder now.
            market_regime = "neutral"
            if mean_ret < -0.01 and breadth < -0.25:
                market_regime = "risk_off"
            elif mean_ret > 0.01 and breadth > 0.25:
                market_regime = "risk_on"
            news_items_in_window = 0 if _suppress_news_for_llm(ablation_mode) else len(daily_news)
            market_state_text = (
                f"market_regime={market_regime}; "
                f"mean_return_1d={mean_ret:+.3%}; breadth={breadth:+.2f}; "
                f"news_window=({news_window_start.strftime('%Y-%m-%d %H:%M')} -> {current_time.strftime('%Y-%m-%d %H:%M')}); "
                f"news_items_in_window={news_items_in_window}"
            )

            # ── Compute OBI and vol_ratio ──
            vol_ratio_today: Dict[str, float] = {}
            for s in stk_codes:
                price_row = price_data.get(s, {}).get(date)
                if price_row:
                    hist = get_history_before(price_data, s, date, 20)
                    avg_vol = sum(h["vol"] for h in hist) / max(len(hist), 1) if hist else 1
                    # Daily-level matching: use daily volume relative to its past average
                    vol_ratio_today[s] = float(price_row["vol"]) / max(float(avg_vol), 1.0)
                else:
                    vol_ratio_today[s] = 1.0

            # ── Generate kline data ──
            kline_data: Dict[str, List[dict]] = {}
            for s in stk_codes:
                hist = get_history_before(price_data, s, date, 20)
                kline = []
                p_today = prices.get(s, 100)
                for h in hist:
                    kline.append({
                        "close_ratio": (h["close"] - p_today) / max(p_today, 0.01),
                        "close_val": h["close"],
                        "high_val": h["high"],
                        "low_val": h["low"],
                        "vol_ratio": vol_ratio_today.get(s, 1.0),
                        "body_ratio": (h["close"] - h["open"]) / max(h["open"], 0.01),
                        "ma5_ratio": 0.0,
                        "ma20_ratio": 0.0,
                    })
                kline_data[s] = kline

            mom_quant_signals: Dict[str, dict] = {}
            for s in stk_codes:
                mom_quant_signals[s] = momentum_compute_signals(kline_data.get(s, []))

            # ── Order-book derived OBI from current market state (before new step orders) ──
            obi_by_stock: Dict[str, float] = {}
            for s in stk_codes:
                snap = market.get_snapshot(s)
                obi_by_stock[s] = compute_obi_from_snapshot(snap)
            market_state_text = (
                market_state_text
                + f"; mean_OBI={float(np.mean(list(obi_by_stock.values()))):+.3f}"
                + f"; mean_vol_ratio={float(np.mean(list(vol_ratio_today.values()))):.2f}x"
            )

            # ── Per-stock LLM signals ──
            semantic_t0 = time.time()
            value_signals: Dict[str, dict] = {}
            momentum_signals_data: Dict[str, dict] = {}
            noise_signals_data: Dict[str, dict] = {}

            if retail_mode in {"stockagent", "twinmarket"}:
                # Per-agent prompt baselines: each retail agent directly interprets
                # the public/private state and emits an order. Do not precompute
                # SUM shared semantic signals here, otherwise the baseline would
                # include both architectures' LLM costs.
                for s in stk_codes:
                    value_signals[s] = {
                        "delta_earnings": 0.0,
                        "delta_quality": 0.0,
                        "macro_signal": 0.0,
                        "post_text": "",
                    }
                    momentum_signals_data[s] = {
                        "catalyst_signal": 0.0,
                        "diffusion_signal": 0.0,
                        "post_text": "",
                    }
                    noise_signals_data[s] = {
                        "forum_sentiment": 0.0,
                        "news_reaction": 0.0,
                        "post_text": "",
                    }
            elif retail_mode == "rule_based":
                raise ValueError(
                    "Rule-based is now a standalone baseline. Use "
                    "run_code/baselines/rule_based_market_sim.py instead of --retail-mode rule_based."
                )
            elif semantic_cache:
                for s in stk_codes:
                    cached = semantic_cache.get((date, s))
                    if cached:
                        semantic_cache_hits += 1
                        value_signals[s] = cached.get("value_signal", {}) or {}
                        momentum_signals_data[s] = cached.get("momentum_signal", {}) or {}
                        noise_signals_data[s] = cached.get("noise_signal", {}) or {}
                        value_signals[s].setdefault("delta_earnings", 0.0)
                        value_signals[s].setdefault("delta_quality", 0.0)
                        value_signals[s].setdefault("macro_signal", 0.0)
                        value_signals[s].setdefault("post_text", "")
                        momentum_signals_data[s].setdefault("catalyst_signal", 0.0)
                        momentum_signals_data[s].setdefault("diffusion_signal", 0.0)
                        momentum_signals_data[s].setdefault("post_text", "")
                        noise_signals_data[s].setdefault("forum_sentiment", 0.0)
                        noise_signals_data[s].setdefault("news_reaction", 0.0)
                        noise_signals_data[s].setdefault("post_text", "")
                    else:
                        semantic_cache_misses += 1
                        value_signals[s] = {"delta_earnings": 0.0, "delta_quality": 0.0, "macro_signal": 0.0, "post_text": ""}
                        momentum_signals_data[s] = {"catalyst_signal": 0.0, "diffusion_signal": 0.0, "post_text": ""}
                        noise_signals_data[s] = {"forum_sentiment": 0.0, "news_reaction": 0.0, "post_text": ""}
            elif mock_llm:
                for s in stk_codes:
                    value_signals[s] = mock_value_signals(s, rng)
                    momentum_signals_data[s] = mock_momentum_signals(s, rng)
                    noise_signals_data[s] = mock_noise_signals(s, rng)
            else:
                def _fetch_stock_llm(s):

                    if _suppress_news_for_llm(ablation_mode):
                        news_text = NO_NEWS_TECHNICAL_ONLY_PLACEHOLDER
                    else:
                        target_news = select_for_stock_relevance_tiered(daily_news, firm_map[s], ind_map[s], current_time)
                        news_text = format_news_text(
                            target_news,
                            firm_name=firm_map[s],
                            ind=ind_map[s],
                            current_time=current_time,
                        )

                    v_ctx = {
                        "firm_name": firm_map[s],
                        "pb_ratio": max(prices.get(s, 1) / max(fina_data.get(s, {}).get("bps", 1.0), 0.01), 0.01),
                        "bps": fina_data.get(s, {}).get("bps", 1.0),
                        "market_state_text": market_state_text,
                        "news_text": news_text,
                        "posts_text": _forum_posts_text(s, sim_step, "value"),
                    }
                    m_ctx = {
                        "firm_name": firm_map[s],
                        "kline_ratios": [k.get("close_ratio", 0) for k in kline_data.get(s, [])],
                        "range_pos": mom_quant_signals[s]["range_pos"],
                        "market_state_text": market_state_text,
                        "news_text": news_text,
                        "posts_text": _forum_posts_text(s, sim_step, "momentum"),
                        **_forum_stats(s, sim_step),
                    }
                    n_ctx = {
                        "firm_name": firm_map[s],
                        "market_state_text": market_state_text,
                        "news_text": news_text,
                        "posts_text": _forum_posts_text(s, sim_step, "noise"),
                    }
                    v_res, v_tok = call_value_agent_llm(v_ctx); m_res, m_tok = call_momentum_agent_llm(m_ctx); n_res, n_tok = call_noise_agent_llm(n_ctx)
                    return s, v_res, m_res, n_res, v_tok + m_tok + n_tok

                with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(llm_workers or 1))) as executor:
                    futures = {executor.submit(_fetch_stock_llm, s): s for s in stk_codes}
                    for future in concurrent.futures.as_completed(futures):
                        s, v_res, m_res, n_res, tok = future.result()
                        total_llm_tokens += tok; total_llm_calls += 3
                        value_signals[s] = v_res; momentum_signals_data[s] = m_res; noise_signals_data[s] = n_res
            semantic_seconds = time.time() - semantic_t0

            if forum_disabled:
                for s in stk_codes:
                    noise_signals_data[s]["forum_sentiment"] = 0.0
                    noise_signals_data[s]["post_text"] = ""
                    value_signals[s]["post_text"] = ""
                    momentum_signals_data[s]["post_text"] = ""

            noise_S_ext: Dict[str, float] = {}
            for s in stk_codes:
                ns = noise_signals_data[s]
                noise_S_ext[s] = noise_compute_S_ext(
                    ns["forum_sentiment"],
                    ns["news_reaction"],
                    obi_by_stock.get(s, 0.0),
                    vol_ratio_today.get(s, 1.0),
                )

            # Weekly supervisor feedback: bias the next week's interpretation.
            # Cached semantic records already include the corresponding cached run's
            # post-feedback shared signals, so do not apply another feedback pass.
            for s in stk_codes:
                trend_bias = weekly_supervisor.get_bias(s)
                if (not semantic_cache) and abs(trend_bias) > 1e-9:
                    value_signals[s]["macro_signal"] = max(-1.0, min(1.0, value_signals[s]["macro_signal"] + 0.35 * trend_bias))
                    momentum_signals_data[s]["catalyst_signal"] = max(-1.0, min(1.0, momentum_signals_data[s]["catalyst_signal"] + 0.45 * trend_bias))
                    momentum_signals_data[s]["diffusion_signal"] = max(-1.0, min(1.0, momentum_signals_data[s]["diffusion_signal"] + 0.25 * trend_bias))
                    noise_signals_data[s]["news_reaction"] = max(-1.0, min(1.0, noise_signals_data[s]["news_reaction"] + 0.30 * trend_bias))
                    noise_signals_data[s]["forum_sentiment"] = max(-1.0, min(1.0, noise_signals_data[s]["forum_sentiment"] + 0.20 * trend_bias))
                    noise_S_ext[s] = noise_compute_S_ext(
                        noise_signals_data[s]["forum_sentiment"],
                        noise_signals_data[s]["news_reaction"],
                        obi_by_stock.get(s, 0.0),
                        vol_ratio_today.get(s, 1.0),
                    )

            day_value_signal_means.append(float(np.mean([
                v.get("delta_earnings", 0.0) + v.get("delta_quality", 0.0) + v.get("macro_signal", 0.0)
                for v in value_signals.values()
            ])))
            day_momentum_signal_means.append(float(np.mean([
                m.get("catalyst_signal", 0.0) + m.get("diffusion_signal", 0.0)
                for m in momentum_signals_data.values()
            ])))
            day_noise_signal_means.append(float(np.mean([
                n.get("forum_sentiment", 0.0) + n.get("news_reaction", 0.0)
                for n in noise_signals_data.values()
            ])))
            day_obi_means.append(float(np.mean(list(obi_by_stock.values()))) if obi_by_stock else 0.0)
            for s in stk_codes:
                all_shared_semantics_log.append({
                    "step": sim_step,
                    "date": date,
                    "hour": h_str,
                    "stock": s,
                    "firm_name": firm_map.get(s, s),
                    "value_signal": value_signals.get(s, {}),
                    "momentum_signal": momentum_signals_data.get(s, {}),
                    "noise_signal": noise_signals_data.get(s, {}),
                    "obi": obi_by_stock.get(s, 0.0),
                    "noise_S_ext": noise_S_ext.get(s, 0.0),
                    "vol_ratio": vol_ratio_today.get(s, 1.0),
                })

            # ── Generate retail agent orders & Collect poster candidates ──
            retail_order_t0 = time.time()
            step_orders = []
            step_retail_decision_records: List[Dict[str, Any]] = []
            all_stock_poster_candidates = {s: [] for s in stk_codes}

            if retail_mode in {"stockagent", "twinmarket"}:
                # Per-agent prompt retail decisions are independent conditional on the
                # current public/private state, so they can be safely parallelized.
                def _run_prompt_baseline_retail_decision(agent: RetailAgent, stock: str) -> Dict[str, Any]:
                    p = prices.get(stock, 0.0)
                    base_log: Dict[str, Any] = {
                        "step": sim_step,
                        "date": date,
                        "hour": h_str,
                        "agent_id": agent.agent_id,
                        "style": agent.style,
                        "stock": stock,
                        "mode": retail_mode,
                    }
                    if stock not in stk_codes or p <= 0:
                        return {**base_log, "orders": [], "poster": None, "tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "llm_call": False, "success": False, "error": "invalid_stock_or_price"}

                    wealth = agent.get_wealth(prices)
                    current_shares = int(agent.holdings.get(stock, 0))
                    position_value = current_shares * p
                    max_trade_value_ratio = {"value": 0.60, "momentum": 0.70, "noise": 0.50}.get(agent.style, 0.50)

                    llm_vars = {
                        "firm_name": firm_map.get(stock, stock),
                        "stock_code": stock,
                        "sim_step": sim_step,
                        "current_price": p,
                        "upper_limit": market.books[stock].stock_info.upper_limit,
                        "lower_limit": market.books[stock].stock_info.lower_limit,
                        "min_price_tick": market.config.get("min_price_tick", 0.01),
                        "agent_id": agent.agent_id,
                        "cash": float(agent.cash),
                        "current_shares": current_shares,
                        "current_position_value": position_value,
                        "wealth": wealth,
                        "initial_wealth": agent.initial_wealth,
                        "psi": agent.psi,
                        "sigma": agent.sigma,
                        "gamma_t": getattr(agent, "gamma_t", 0.08),
                        "max_trade_value_ratio": max_trade_value_ratio,
                        "min_order_quantity": market.config.get("min_order_quantity", 1),
                        "quantity_step": market.config.get("quantity_step", 1),
                        "max_order_quantity": market.config.get("max_order_quantity", 1_000_000),
                    }
                    llm_vars["market_state_text"] = market_state_text

                    if _suppress_news_for_llm(ablation_mode):
                        llm_vars["news_text"] = NO_NEWS_TECHNICAL_ONLY_PLACEHOLDER
                    else:
                        target_news = select_for_stock_relevance_tiered(daily_news, firm_map[stock], ind_map[stock], current_time)
                        llm_vars["news_text"] = format_news_text(
                            target_news,
                            firm_name=firm_map[stock],
                            ind=ind_map[stock],
                            current_time=current_time,
                        )
                    if agent.style == "value":
                        bps = fina_data.get(stock, {}).get("bps", 1.0)
                        llm_vars["bps"] = bps
                        llm_vars["pb_ratio"] = max(p / max(bps, 0.01), 0.01)
                        llm_vars["posts_text"] = _forum_posts_text(stock, sim_step, "value")
                    elif agent.style == "momentum":
                        llm_vars["kline_ratios"] = [k.get("close_ratio", 0) for k in kline_data.get(stock, [])]
                        llm_vars["range_pos"] = mom_quant_signals.get(stock, {}).get("range_pos", 0.5)
                        llm_vars["posts_text"] = _forum_posts_text(stock, sim_step, "momentum")
                        llm_vars.update(_forum_stats(stock, sim_step))
                    else:
                        llm_vars["rho_prev"] = getattr(agent, "rho", 0.0)
                        llm_vars["obi"] = obi_by_stock.get(stock, 0.0)
                        llm_vars["vol_ratio"] = vol_ratio_today.get(stock, 1.0)
                        llm_vars["posts_text"] = _forum_posts_text(stock, sim_step, "noise")

                    used_prompt_cache = False
                    if prompt_baseline_cache:
                        nonlocal prompt_baseline_cache_hits, prompt_baseline_cache_misses
                        cache_key = (sim_step, agent.agent_id, stock, retail_mode)
                        cached_prompt = prompt_baseline_cache.get(cache_key)
                        if cached_prompt:
                            prompt_baseline_cache_hits += 1
                            used_prompt_cache = True
                            usage = {
                                "success": cached_prompt.get("success", False),
                                "total_tokens": int(cached_prompt.get("tokens", 0) or 0),
                                "prompt_tokens": int(cached_prompt.get("prompt_tokens", 0) or 0),
                                "completion_tokens": int(cached_prompt.get("completion_tokens", 0) or 0),
                                "latency_seconds": float(cached_prompt.get("latency_seconds", 0.0) or 0.0),
                                "error": str(cached_prompt.get("error", "")),
                                "raw_content": str(cached_prompt.get("raw_content", "")),
                            }
                            res = result_from_payload(cached_prompt.get("raw_response", {}) or {}, usage, "", "")
                        else:
                            prompt_baseline_cache_misses += 1
                            res = call_stockagent_retail(agent.style, llm_vars, mock=mock_llm) if retail_mode == "stockagent" else call_twinmarket_retail(agent.style, llm_vars, mock=mock_llm)
                    elif retail_mode == "stockagent":
                        res = call_stockagent_retail(agent.style, llm_vars, mock=mock_llm)
                    else:
                        res = call_twinmarket_retail(agent.style, llm_vars, mock=mock_llm)

                    raw_order_count = len(res.orders)
                    filtered_orders: List[AgentOrder] = []
                    prefilter_reasons: List[str] = []
                    for order in res.orders:
                        filtered, reason = sum_retail_order_prefilter(
                            order,
                            current_price=p,
                            cash=float(agent.cash),
                            current_shares=current_shares,
                        )
                        if filtered is None:
                            prefilter_reasons.append(reason)
                        else:
                            filtered_orders.append(filtered)
                    res_orders = filtered_orders
                    poster = None
                    if res_orders:
                        o0 = res_orders[0]
                        vol_r = float(o0.deal_amount) / max(wealth, 1.0)
                        R_exec = vol_r if o0.direction == "BUY" else -vol_r
                        if abs(R_exec) > forum.TAU_POST:
                            poster = (abs(R_exec), agent.agent_id, R_exec, vol_r, res.post_text, agent.style)
                    elif res.post_text:
                        poster = (0.0, agent.agent_id, 0.0, 0.0, res.post_text, agent.style)

                    return {
                        **base_log,
                        "orders": res_orders,
                        "poster": poster,
                        "tokens": res.tokens,
                        "prompt_tokens": res.prompt_tokens,
                        "completion_tokens": res.completion_tokens,
                        "latency_seconds": round(res.latency_seconds, 4),
                        "llm_call": (not mock_llm) and (not used_prompt_cache),
                        "success": res.success,
                        "error": res.error,
                        "post_text": res.post_text,
                        "raw_response": res.raw,
                        "raw_content": res.raw_content,
                        "system_prompt": res.system_prompt if record_prompts else "",
                        "user_prompt": res.user_prompt if record_prompts else "",
                        "parsed_order_count": raw_order_count,
                        "postfilter_order_count": len(res_orders),
                        "baseline_prefilter_drop_reasons": prefilter_reasons,
                        "used_prompt_baseline_cache": used_prompt_cache,
                        "action": res_orders[0].direction if res_orders else "HOLD",
                        "reasoning": res_orders[0].reasoning if res_orders else "",
                        "wealth": round(wealth, 2),
                        "cash": round(float(agent.cash), 2),
                        "current_shares": current_shares,
                        "position_value": round(position_value, 2),
                    }

                prompt_tasks: List[Tuple[RetailAgent, str]] = [
                    (agent, stock)
                    for agent in retail_agents
                    for stock in agent.stock_list
                    if stock in stk_codes
                ]
                max_workers = max(1, int(prompt_baseline_workers or 1))
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(_run_prompt_baseline_retail_decision, agent, stock) for agent, stock in prompt_tasks]
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            decision_log = future.result()
                        except Exception as exc:
                            all_retail_decisions_log.append({
                                "step": sim_step,
                                "date": date,
                                "hour": h_str,
                                "mode": retail_mode,
                                "success": False,
                                "error": f"future_exception: {exc}",
                            })
                            total_failed_llm_calls += 1
                            continue
                        total_llm_tokens += int(decision_log.get("tokens", 0))
                        total_prompt_tokens += int(decision_log.get("prompt_tokens", 0))
                        total_completion_tokens += int(decision_log.get("completion_tokens", 0))
                        if decision_log.get("llm_call"):
                            total_llm_calls += 1
                        if decision_log.get("llm_call") and not decision_log.get("success", False):
                            total_failed_llm_calls += 1
                        poster = decision_log.pop("poster", None)
                        orders = decision_log.pop("orders", [])
                        if poster is not None:
                            all_stock_poster_candidates[decision_log["stock"]].append(poster)
                        step_orders.extend(orders)
                        all_retail_decisions_log.append(decision_log)
            else:
                for agent in retail_agents:
                    for stock in agent.stock_list:
                        if stock not in stk_codes:
                            continue
                        # Original SUM: LLM produces semantic signals; rules map signals -> orders.
                        kline_close_ratios = [k.get("close_ratio", 0) for k in kline_data.get(stock, [])]

                        channel_style = "noise" if ablation_mode == "meso_noise_channel" else agent.style
                        if channel_style == "value":
                            vs = value_signals[stock]
                            R_hat = value_compute_R_hat(
                                math.log(max(fina_data.get(stock, {}).get("bps", 1.0), 0.01) / max(prices.get(stock, 100), 0.01)),
                                v_bar_cache.get(stock, 0.0),
                                agent.psi,
                                vs["delta_earnings"],
                                vs["delta_quality"],
                                vs["macro_signal"],
                                g_B,
                            )
                            w_max = 0.60
                        elif channel_style == "momentum":
                            mq = mom_quant_signals[stock]
                            ms = momentum_signals_data[stock]
                            R_hat = momentum_compute_R_hat(agent.psi, mq["M_t"], ms["catalyst_signal"], ms["diffusion_signal"])
                            w_max = 0.70
                        else:
                            agent.rho = noise_update_rho(agent.psi, agent.rho, noise_S_ext[stock])
                            R_hat = agent.rho
                            w_max = 0.50

                        R_adj, vol_r, dir_str, R_exec = apply_signal_pipeline(R_hat, agent, stock, kline_close_ratios, w_max)

                        post_text = {"value": value_signals, "momentum": momentum_signals_data, "noise": noise_signals_data}[channel_style][stock].get("post_text", "")
                        if abs(R_exec) > forum.TAU_POST:
                            all_stock_poster_candidates[stock].append((abs(R_exec), agent.agent_id, R_exec, vol_r, post_text, agent.style))

                        order = generate_retail_order(agent, stock, R_adj, vol_r, dir_str, prices, sim_step, channel_style=channel_style)
                        if full_decision_log:
                            W = agent.get_wealth(prices)
                            signal_payload: Dict[str, Any] = {}
                            if channel_style == "value":
                                signal_payload = {
                                    "value_signal": value_signals.get(stock, {}),
                                    "v_t": math.log(max(fina_data.get(stock, {}).get("bps", 1.0), 0.01) / max(prices.get(stock, 100), 0.01)),
                                    "v_bar": v_bar_cache.get(stock, 0.0),
                                    "g_B": g_B,
                                }
                            elif channel_style == "momentum":
                                signal_payload = {
                                    "momentum_signal": momentum_signals_data.get(stock, {}),
                                    "momentum_quant_signal": mom_quant_signals.get(stock, {}),
                                }
                            else:
                                signal_payload = {
                                    "noise_signal": noise_signals_data.get(stock, {}),
                                    "noise_S_ext": noise_S_ext.get(stock, 0.0),
                                    "rho": getattr(agent, "rho", 0.0),
                                }
                            step_retail_decision_records.append({
                                "step": sim_step,
                                "date": date,
                                "hour": h_str,
                                "mode": retail_mode,
                                "agent_id": agent.agent_id,
                                "style": agent.style,
                                "channel_style": channel_style,
                                "stock": stock,
                                "price": round(float(prices.get(stock, 0.0)), 4),
                                "wealth": round(float(W), 2),
                                "cash": round(float(agent.cash), 2),
                                "holding_shares": int(agent.holdings.get(stock, 0)),
                                "psi": round(float(agent.psi), 6),
                                "sigma": round(float(agent.sigma), 6),
                                "gamma_t": round(float(agent.gamma_t), 6),
                                "R_hat": round(float(R_hat), 8),
                                "R_adj": round(float(R_adj), 8),
                                "volume_ratio": round(float(vol_r), 8),
                                "direction": dir_str,
                                "R_exec": round(float(R_exec), 8),
                                "w_max": round(float(w_max), 6),
                                "order_generated": order is not None,
                                "client_order_id": order.client_order_id if order else "",
                                "deal_amount": round(float(order.deal_amount), 2) if order else 0.0,
                                "decision_reasoning": (
                                    f"{'Rule-based utility' if retail_mode == 'rule_based' else 'SUM signal'} decision: "
                                    f"style={agent.style}; channel={channel_style}; stock={stock}; "
                                    f"R_hat={R_hat:.6f}; bounded-rationality adjusted R_adj={R_adj:.6f}; "
                                    f"direction={dir_str}; volume_ratio={vol_r:.6f}; "
                                    f"order_generated={order is not None}."
                                ),
                                "order_reasoning": order.reasoning if order else "",
                                **signal_payload,
                            })
                        if order:
                            step_orders.append(order)
            _append_retail_decisions_jsonl(step_retail_decision_records)
            total_retail_decisions_logged += len(step_retail_decision_records)
            retail_order_seconds = time.time() - retail_order_t0

            # ── Filter Top 10 Posters per Stock ──
            if not forum_disabled:
                for stock, candidates in all_stock_poster_candidates.items():
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    for intensity, author_id, R_exec, vol_r, text, style in candidates[:10]:
                        forum.try_publish(stock, sim_step, date, style, author_id, R_exec, vol_r, text)

            # ── Institution: Multi-Agent Debate Pipeline (institution_design.md §4) ──
            institution_t0 = time.time()
            for ia in inst_agents:
                if ablation_mode == "macro_sum_institution":
                    if not hasattr(ia, "decision_log"):
                        ia.decision_log = []
                    for stock in getattr(ia, "stock_list", stk_codes):
                        if stock not in stk_codes:
                            continue
                        kline_close_ratios = [k.get("close_ratio", 0) for k in kline_data.get(stock, [])]
                        if ia.style == "value":
                            vs = value_signals[stock]
                            R_hat = value_compute_R_hat(
                                math.log(max(fina_data.get(stock, {}).get("bps", 1.0), 0.01) / max(prices.get(stock, 100), 0.01)),
                                v_bar_cache.get(stock, 0.0),
                                ia.psi,
                                vs["delta_earnings"],
                                vs["delta_quality"],
                                vs["macro_signal"],
                                g_B,
                            )
                            w_max = 0.60
                        elif ia.style == "momentum":
                            mq = mom_quant_signals[stock]
                            ms = momentum_signals_data[stock]
                            R_hat = momentum_compute_R_hat(ia.psi, mq["M_t"], ms["catalyst_signal"], ms["diffusion_signal"])
                            w_max = 0.70
                        else:
                            ia.rho = noise_update_rho(ia.psi, ia.rho, noise_S_ext[stock])
                            R_hat = ia.rho
                            w_max = 0.50

                        R_adj, vol_r, dir_str, R_exec = apply_signal_pipeline(R_hat, ia, stock, kline_close_ratios, w_max)
                        order = generate_retail_order(ia, stock, R_adj, vol_r, dir_str, prices, sim_step, channel_style=ia.style)
                        ia.decision_log.append({
                            "step": sim_step,
                            "date": date,
                            "hour": h_str,
                            "stock": stock,
                            "architecture": "SUM_retail_signal_sleeve",
                            "style": ia.style,
                            "action": dir_str if order else "HOLD",
                            "R_hat": round(float(R_hat), 8),
                            "R_adj": round(float(R_adj), 8),
                            "volume_ratio": round(float(vol_r), 8),
                            "order_generated": order is not None,
                            "reasoning": order.reasoning if order else "",
                        })
                        if order:
                            step_orders.append(order)
                    continue

                # Flush pending TWAP slices from previous steps
                step_orders.extend(ia.flush_twap_orders(sim_step))
                if institution_cache and len(intraday_times) > 1 and intraday_idx > 1:
                    continue

                # §3.1 News time advantage (configurable):
                # inst_early_news=1 → institution gets t+1 news; inst_early_news=0 → no early access
                next_date = None
                if day_idx + 1 < len(trading_dates):
                    next_date = trading_dates[day_idx + 1]
                inst_news = list(daily_news)  # copy current news
                if (
                    not _suppress_news_for_llm(ablation_mode)
                    and inst_early_news == 1
                    and next_date is not None
                ):
                    tomorrow_news = [n for n in all_news_obj if n["date_str"] == next_date]
                    inst_news.extend(tomorrow_news)

                inst_W = ia.get_wealth(prices)
                cio_decisions: Dict[str, CIODecision] = {}
                vol_forecasts: Dict[str, float] = {}

                def _process_inst_stock(stock):
                    p = prices.get(stock, 0)
                    if p <= 0: return None
                    nonlocal institution_cache_hits, institution_cache_misses
                    if institution_cache:
                        cache_step = day_idx + 1 if len(intraday_times) > 1 else sim_step
                        cached_inst = institution_cache.get((cache_step, stock))
                        if cached_inst:
                            institution_cache_hits += 1
                            action = cached_inst.get("action", "hold")
                            target_delta = float(cached_inst.get("target_weight_delta", 0.0) or 0.0)
                            urgency = float(cached_inst.get("urgency", 0.0) or 0.0)
                            decision = CIODecision(
                                action=action,
                                conviction=1.0 if action != "hold" else 0.0,
                                target_weight_delta=target_delta,
                                urgency=urgency,
                                risk_override=False,
                                reasoning=str(cached_inst.get("reasoning", "")),
                            )
                            log_entry = {
                                "step": sim_step,
                                "cache_step": cache_step,
                                "date": date,
                                "hour": h_str,
                                "stock": stock,
                                "action": action,
                                "target_weight_delta": target_delta,
                                "urgency": urgency,
                                "reasoning": cached_inst.get("reasoning", ""),
                                "source": "institution_cache",
                            }
                            return stock, decision, 0.3, log_entry, 0
                        institution_cache_misses += 1
                    forum_stats_s = _forum_stats(stock, sim_step)
                    retail_stats = InstitutionAgent.compute_retail_flow_stats(stock, retail_agents, prices, forum_stats_s)
                    bps = fina_data.get(stock, {}).get('bps', 1.0)
                    pb_ratio = p / max(bps, 0.01)
                    cost_vs_price = (retail_stats.avg_cost / max(p, 0.01) - 1)
                    if _suppress_news_for_llm(ablation_mode):
                        news_inst_text = NO_NEWS_TECHNICAL_ONLY_PLACEHOLDER
                    else:
                        inst_target_news = select_for_stock_relevance_tiered(inst_news, firm_map[stock], ind_map[stock], current_time)
                        news_inst_text = format_news_text(
                            inst_target_news,
                            firm_name=firm_map[stock],
                            ind=ind_map[stock],
                            current_time=current_time,
                        )
                    posts_text = _forum_posts_text(stock, sim_step, 'value')
                    local_tokens = 0
                    if mock_llm:
                        if stock not in ia.fundamental_cache:
                            ia.fundamental_cache[stock] = mock_inst_fundamental_quarterly(stock, rng)
                        fund_cache = ia.fundamental_cache[stock]
                        fund_daily = mock_inst_fundamental_daily(stock, rng)
                        quant_res = mock_inst_quant(stock, rng)
                        sentiment_res = mock_inst_sentiment(stock, rng)
                        sub_reports = {'expectation_delta': fund_daily['expectation_delta'], 'technical_signal': quant_res['technical_signal'], 'retail_sentiment_extreme': sentiment_res['retail_sentiment_extreme'], 'contrarian_opportunity': sentiment_res['contrarian_opportunity']}
                        cio_res = mock_inst_cio(stock, sub_reports, rng)
                    else:
                        if stock not in ia.fundamental_cache:
                            fund_ctx = {'firm_name': firm_map[stock], 'industry': ind_map[stock], 'bps': bps, 'roe_waa': fina_data.get(stock, {}).get('roe_waa', 0), 'end_date': fina_data.get(stock, {}).get('end_date', ''), 'revenue': 0, 'n_income': 0, 'grprofit_margin': fina_data.get(stock, {}).get('grossprofit_margin', 0) or 0, 'netprofit_margin': fina_data.get(stock, {}).get('netprofit_margin', 0) or 0, 'basic_eps': fina_data.get(stock, {}).get('eps', 0), 'total_assets': 0, 'total_liab': 0, 'debt_to_assets': fina_data.get(stock, {}).get('debt_to_assets', 0) or 0, 'current_ratio': fina_data.get(stock, {}).get('current_ratio', 0) or 0, 'n_cashflow_act': 0, 'free_cashflow': 0, 'roa': 0, 'r_10Y': macro.get('rate_10y', 0.023), 'r_LPR': macro.get('lpr_1y', 0.031), 'main_business': ''}
                            fund_q_res, tok = call_inst_fundamental_quarterly_llm(fund_ctx)
                            local_tokens += tok
                            ia.fundamental_cache[stock] = fund_q_res
                        fund_cache = ia.fundamental_cache[stock]
                        fund_daily_ctx = {'firm_name': firm_map[stock], 'industry': ind_map[stock], 'context_summary': fund_cache.get('context_summary', ''), 'key_expectation_factors': fund_cache.get('key_expectation_factors', []), 'pb_ratio': pb_ratio, 'fair_pb_mid': fund_cache.get('fair_pb_mid', 1.0), 'news_inst_text': news_inst_text}
                        fund_daily, tok = call_inst_fundamental_daily_llm(fund_daily_ctx)
                        local_tokens += tok
                        quant_ctx = {'firm_name': firm_map[stock], 'industry': ind_map[stock], 'kline_ratios': [k.get('close_ratio', 0) for k in kline_data.get(stock, [])], 'vol_ratios': [k.get('vol_ratio', 1.0) for k in kline_data.get(stock, [])], 'ma5_ratio': kline_data.get(stock, [{}])[0].get('ma5_ratio', 0) if kline_data.get(stock) else 0, 'ma20_ratio': kline_data.get(stock, [{}])[0].get('ma20_ratio', 0) if kline_data.get(stock) else 0, 'OBI': obi_by_stock.get(stock, 0.0), 'range_pos': mom_quant_signals.get(stock, {}).get('range_pos', 0.5), 'rzye_chg': 0, 'rqye_chg': 0, 'block_trade_section': ''}
                        quant_res, tok = call_inst_quant_llm(quant_ctx)
                        local_tokens += tok
                        sentiment_ctx = {'firm_name': firm_map[stock], 'industry': ind_map[stock], 'n_hold': retail_stats.n_hold, 'delta_n': retail_stats.delta_n, 'avg_cost': retail_stats.avg_cost, 'cost_vs_price': cost_vs_price, 'std_cost': retail_stats.std_cost, 'avg_rho': retail_stats.avg_rho, 'std_rho': retail_stats.std_rho, 'n_bull': retail_stats.n_bull, 'n_bear': retail_stats.n_bear, 'n_neutral': retail_stats.n_neutral, 'posts_text': posts_text, 'news_inst_text': news_inst_text, 'vol_ratio': vol_ratio_today.get(stock, 1.0), 'OBI': obi_by_stock.get(stock, 0.0)}
                        sentiment_res, tok = call_inst_sentiment_llm(sentiment_ctx)
                        local_tokens += tok
                        current_shares = ia.holdings.get(stock, 0)
                        position_value = current_shares * p
                        position_pct = (position_value / inst_W * 100) if inst_W > 0 else 0
                        total_pos_val = sum(ia.holdings.get(s2, 0) * prices.get(s2, 0) for s2 in stk_codes)
                        total_position_pct = (total_pos_val / inst_W * 100) if inst_W > 0 else 0
                        cio_ctx = {'firm_name': firm_map[stock], 'last_price': p, 'fundamental_quality': fund_cache.get('fundamental_quality', 0), 'fair_pb_mid': fund_cache.get('fair_pb_mid', 1.0), 'pb_ratio': pb_ratio, 'expectation_delta': fund_daily.get('expectation_delta', 0), 'delta_confidence': fund_daily.get('delta_confidence', 0), 'catalyst_type': fund_daily.get('catalyst_type', 'none'), 'risk_flag': fund_cache.get('risk_flag', 0), 'fundamental_summary': fund_cache.get('context_summary', ''), 'technical_signal': quant_res.get('technical_signal', 0), 'capital_flow_signal': quant_res.get('capital_flow_signal', 0), 'volatility_forecast': quant_res.get('volatility_forecast', 0), 'support': quant_res.get('key_levels', {}).get('support', 0), 'resistance': quant_res.get('key_levels', {}).get('resistance', 0), 'quant_summary': quant_res.get('context_summary', ''), 'retail_sentiment_extreme': sentiment_res.get('retail_sentiment_extreme', 0), 'retail_pain_signal': sentiment_res.get('retail_pain_signal', 0), 'contrarian_opportunity': sentiment_res.get('contrarian_opportunity', 0), 'sentiment_summary': sentiment_res.get('context_summary', ''), 'current_shares': current_shares, 'position_value': position_value, 'position_pct': position_pct, 'total_position_pct': total_position_pct}
                        cio_res, tok = call_inst_cio_llm(cio_ctx)
                        local_tokens += tok
                    trend_bias = weekly_supervisor.get_bias(stock)
                    if abs(trend_bias) > 1e-9:
                        cio_res['target_weight_delta'] = max(-0.05, min(0.05, float(cio_res.get('target_weight_delta', 0.0)) + 0.25 * trend_bias))
                        if cio_res.get('action', 'hold') == 'hold' and abs(trend_bias) > 0.01:
                            cio_res['action'] = 'buy' if trend_bias > 0 else 'sell'
                            cio_res['urgency'] = max(float(cio_res.get('urgency', 0.0)), min(1.0, abs(trend_bias) * 2))
                    decision = CIODecision(action=cio_res.get('action', 'hold'), conviction=float(cio_res.get('conviction', 0)), target_weight_delta=float(cio_res.get('target_weight_delta', 0)), urgency=float(cio_res.get('urgency', 0)), risk_override=bool(cio_res.get('risk_override', False)), reasoning=str(cio_res.get('reasoning', '')))
                    vol = float(quant_res.get('volatility_forecast', 0.3) if not mock_llm else quant_res.get('volatility_forecast', 0.3))
                    log_entry = {'step': sim_step, 'stock': stock, 'action': cio_res.get('action', 'hold'), 'target_weight_delta': cio_res.get('target_weight_delta', 0), 'urgency': cio_res.get('urgency', 0), 'reasoning': cio_res.get('reasoning', '')}
                    return stock, decision, vol, log_entry, local_tokens
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(llm_workers or 1))) as executor:
                    futures = {executor.submit(_process_inst_stock, s): s for s in stk_codes}
                    for future in concurrent.futures.as_completed(futures):
                        res = future.result()
                        if res is not None:
                            stk, dec, vol, log_enc, tok = res
                            cio_decisions[stk] = dec
                            vol_forecasts[stk] = vol
                            ia.decision_log.append(log_enc)
                            total_llm_tokens += tok
                # §5 Portfolio Risk Management
                final_deltas = ia.apply_portfolio_risk_mgmt(cio_decisions, prices, vol_forecasts)

                # §6 Order Execution
            inst_orders = ia.generate_orders_from_decisions(
                final_deltas, cio_decisions, prices, sim_step,
            )
            if institution_cache and len(intraday_times) > 1 and inst_orders:
                n_slices = len(intraday_times) - intraday_idx + 1
                split_now: List[AgentOrder] = []
                for order in inst_orders:
                    for k in range(n_slices):
                        sliced = _split_order_for_intraday(order, n_slices, k + 1)
                        if k == 0:
                            split_now.append(sliced)
                        else:
                            ia.twap_queue.append((sim_step + k, sliced))
                inst_orders = split_now
            step_orders.extend(inst_orders)
            institution_seconds = time.time() - institution_t0

            # ── VMM orders ──
            vmm_t0 = time.time()
            step_orders.extend(vmm.generate_orders(prices, sim_step))
            vmm_seconds = time.time() - vmm_t0

            # ── Submit orders to market ──
            matching_t0 = time.time()
            step_trades = []
            n_buy = sum(1 for o in step_orders if o.direction == "BUY")
            n_sell = len(step_orders) - n_buy
            n_valid = 0

            step_executed_orders = 0
            for order in step_orders:
                cumulative_order_attempts += 1
                val_res, trades = market.submit_agent_order(order)
                executed_price = getattr(val_res.order, "price", None) if getattr(val_res, "order", None) else None
                executed_qty = getattr(val_res.order, "quantity", None) if getattr(val_res, "order", None) else None
                status = (
                    getattr(val_res.order.status, "value", str(val_res.order.status))
                    if getattr(val_res, "order", None) else "REJECTED"
                )
                if not minimal_output:
                    order_attempt = {
                        "step": sim_step,
                        "date": date,
                        "hour": h_str,
                        "agent": order.agent_id,
                        "stock": order.stock,
                        "direction": order.direction,
                        "deal_amount": round(float(order.deal_amount), 2),
                        "price_type": order.price_type,
                        "price_offset": round(float(order.price_offset), 6),
                        "limit_price": round(float(order.limit_price), 4) if order.limit_price is not None else None,
                        "quantity_requested": order.quantity,
                        "client_order_id": order.client_order_id,
                        "reasoning": order.reasoning,
                        "is_valid": bool(val_res.is_valid),
                        "order_id": getattr(val_res.order, "order_id", ""),
                        "status": status,
                        "price": round(executed_price, 4) if executed_price is not None else None,
                        "quantity": executed_qty,
                        "reject_reason": val_res.reject_reason,
                        "warnings": val_res.warnings,
                        "n_trades_generated": len(trades),
                    }
                    all_order_attempts_log.append(order_attempt)
                if val_res.is_valid:
                    n_valid += 1
                    cumulative_orders += 1
                    if trades:
                        step_executed_orders += 1
                    cumulative_trades += len(trades)
                    if not minimal_output:
                        all_orders_log.append({
                            "step": sim_step,
                            "date": date,
                            "hour": h_str,
                            "agent": order.agent_id,
                            "stock": order.stock,
                            "direction": order.direction,
                            "deal_amount": round(order.deal_amount, 2),
                            "price_type": order.price_type,
                            "price_offset": round(float(order.price_offset), 6),
                            "client_order_id": order.client_order_id,
                            "reasoning": order.reasoning,
                            "order_id": getattr(val_res.order, "order_id", ""),
                            "status": getattr(val_res.order.status, "value", str(val_res.order.status)) if getattr(val_res, "order", None) else "",
                            "price": round(executed_price, 4) if executed_price is not None else None,
                            "quantity": executed_qty,
                            "reject_reason": val_res.reject_reason,
                            "warnings": val_res.warnings,
                        })
                    for t in trades:
                        tr = {
                            "step": sim_step,
                            "date": date,
                            "hour": h_str,
                            "trade_id": t.trade_id,
                            "stock": t.stock_code,
                            "price": round(t.price, 4),
                            "quantity": t.quantity,
                            "value": round(t.value, 2),
                            "buyer": t.buyer_id,
                            "seller": t.seller_id,
                            "buy_order_id": t.buy_order_id,
                            "sell_order_id": t.sell_order_id,
                            "timestamp": t.timestamp,
                        }
                        step_trades.append(tr)
                        if not minimal_output:
                            all_trades_log.append(tr)
            matching_seconds = time.time() - matching_t0

            # ── LOB snapshot (captured after matching, before end_of_day_roll) ──
            if lob_snapshot_log is not None:
                for s in stk_codes:
                    book = market.books.get(s)
                    if book is None:
                        continue
                    best_bid = book.get_best_bid()
                    best_ask = book.get_best_ask()
                    mid = (best_bid + best_ask) / 2 if (best_bid and best_ask) else None
                    bas_pct = (best_ask - best_bid) / mid * 100.0 if mid else None
                    # Depth: sum resting qty on each side (up to 5 levels)
                    from simulation.market_engine import OrderHeapItem
                    import heapq as _hq
                    raw_bids = sorted([it.order for it in book.bids], key=lambda o: o.price, reverse=True)
                    raw_asks = sorted([it.order for it in book.asks], key=lambda o: o.price)
                    depth_bid = sum(o.remaining_quantity for o in raw_bids[:5])
                    depth_ask = sum(o.remaining_quantity for o in raw_asks[:5])
                    n_bid_levels = len({o.price for o in raw_bids})
                    n_ask_levels = len({o.price for o in raw_asks})
                lob_snapshot_log.append({
                    "step": sim_step,
                    "date": date,
                    "hour": h_str,
                    "intraday_step": intraday_idx,
                    "stock": s,
                        "best_bid": best_bid,
                        "best_ask": best_ask,
                        "mid_price": mid,
                        "bas_pct": bas_pct,
                        "depth_bid_5lvl": int(depth_bid),
                        "depth_ask_5lvl": int(depth_ask),
                        "total_depth_5lvl": int(depth_bid + depth_ask),
                        "n_bid_levels": n_bid_levels,
                        "n_ask_levels": n_ask_levels,
                        "n_trades_today": len(book.today_trades),
                        "volume_today": sum(t.quantity for t in book.today_trades),
                        "last_price": book.last_price,
                    })

            day_order_count += len(step_orders)
            day_trade_count += len(step_trades)

            # Update prices and agent states
            state_update_t0 = time.time()
            for s in stk_codes:
                book = market.books.get(s)
                if book and book.last_price > 0: prices[s] = book.last_price

            for a in retail_agents:
                a.cash = market.get_cash(a.agent_id)
                for s in list(a.holdings.keys()):
                    a.holdings[s] = market.get_position(a.agent_id, s)
                w = a.get_wealth(prices)
                a.wealth_history.append(w)
                a.update_bounded_rationality()
                if not minimal_output:
                    holdings_snapshot = {s: q for s, q in a.holdings.items() if q}
                    position_value = sum(q * prices.get(s, 0.0) for s, q in holdings_snapshot.items())
                    all_agent_states_log.append({
                        "step": sim_step,
                        "date": date,
                        "hour": h_str,
                        "agent_id": a.agent_id,
                        "style": a.style,
                        "wealth": round(w, 2),
                        "cash": round(a.cash, 2),
                        "position_value": round(position_value, 2),
                        "position_ratio": round(position_value / w, 4) if w > 0 else 0.0,
                        "holdings": holdings_snapshot,
                        "n_positions": len(holdings_snapshot),
                        "pnl_T": round(a.get_pnl_T(), 4),
                        "gamma_t": round(float(getattr(a, "gamma_t", 0.0) or 0.0), 4),
                        "eta_t": round(float(getattr(a, "eta_t", 0.0) or 0.0), 4),
                        "h_t": round(float(getattr(a, "h_t", 0.0) or 0.0), 4),
                        "delta_t": round(float(getattr(a, "delta_t", 0.0) or 0.0), 4),
                        "rho": round(float(getattr(a, "rho", 0.0) or 0.0), 4),
                    })

            for ia in inst_agents:
                ia.cash = market.get_cash(ia.agent_id)
                for s in stk_codes:
                    ia.holdings[s] = market.get_position(ia.agent_id, s)
                w = ia.get_wealth(prices)
                ia.wealth_history.append(w)
                if hasattr(ia, "update_bounded_rationality"):
                    ia.update_bounded_rationality()
                if not minimal_output:
                    inst_holdings = {s: q for s, q in ia.holdings.items() if q}
                    inst_position_value = sum(q * prices.get(s, 0.0) for s, q in inst_holdings.items())
                    all_agent_states_log.append({
                        "step": sim_step,
                        "date": date,
                        "hour": h_str,
                        "agent_id": ia.agent_id,
                        "style": ia.style,
                        "wealth": round(w, 2),
                        "cash": round(ia.cash, 2),
                        "position_value": round(inst_position_value, 2),
                        "position_ratio": round(inst_position_value / w, 4) if w > 0 else 0.0,
                        "holdings": inst_holdings,
                        "n_positions": len(inst_holdings),
                        "decision_log_size": len(getattr(ia, "decision_log", [])),
                        "twap_queue_len": len(getattr(ia, "twap_queue", [])),
                    })

            vmm.cash = market.get_cash(vmm.agent_id)
            for s in stk_codes:
                vmm.holdings[s] = market.get_position(vmm.agent_id, s)
            if not minimal_output:
                vmm_position_value = sum(q * prices.get(s, 0.0) for s, q in vmm.holdings.items())
                all_agent_states_log.append({
                    "step": sim_step,
                    "date": date,
                    "hour": h_str,
                    "agent_id": vmm.agent_id,
                    "style": vmm.style,
                    "wealth": round(vmm.get_wealth(prices), 2),
                    "cash": round(vmm.cash, 2),
                    "position_value": round(vmm_position_value, 2),
                    "position_ratio": round(vmm_position_value / max(vmm.get_wealth(prices), 1.0), 4),
                    "holdings": {s: q for s, q in vmm.holdings.items() if q},
                    "n_positions": sum(1 for q in vmm.holdings.values() if q),
                })

            if not minimal_output:
                for agent_obj in [*retail_agents, *inst_agents, vmm]:
                    agent_id = agent_obj.agent_id
                    start_asset = day_start_assets.get(agent_id, {})
                    end_asset = _asset_snapshot(agent_obj)
                    all_daily_assets_log.append({
                    "step": sim_step,
                    "date": date,
                    "hour": h_str,
                    "intraday_step": intraday_idx,
                    "intraday_steps_per_day": len(intraday_times),
                    "agent_id": agent_id,
                        "style": getattr(agent_obj, "style", ""),
                        "role": "retail" if agent_id.startswith("R_") else "institution" if agent_id.startswith(("I_", "INST")) else "market_maker",
                        "wealth_start": round(float(start_asset.get("wealth", 0.0)), 2),
                        "wealth_end": round(float(end_asset.get("wealth", 0.0)), 2),
                        "daily_pnl": round(float(end_asset.get("wealth", 0.0)) - float(start_asset.get("wealth", 0.0)), 2),
                        "daily_return": round(float(end_asset.get("wealth", 0.0)) / float(start_asset.get("wealth", 1.0)) - 1.0, 6) if float(start_asset.get("wealth", 0.0)) > 0 else 0.0,
                        "cash_start": round(float(start_asset.get("cash", 0.0)), 2),
                        "cash_end": round(float(end_asset.get("cash", 0.0)), 2),
                        "position_value_start": round(float(start_asset.get("position_value", 0.0)), 2),
                        "position_value_end": round(float(end_asset.get("position_value", 0.0)), 2),
                        "position_ratio_start": round(float(start_asset.get("position_ratio", 0.0)), 6),
                        "position_ratio_end": round(float(end_asset.get("position_ratio", 0.0)), 6),
                        "holdings_start": start_asset.get("holdings", {}),
                        "holdings_end": end_asset.get("holdings", {}),
                        "n_positions_start": start_asset.get("n_positions", 0),
                        "n_positions_end": end_asset.get("n_positions", 0),
                    })
            state_update_seconds = time.time() - state_update_t0

            elapsed = time.time() - step_t0
            submitted_orders = len(step_orders)
            valid_orders = n_valid
            executed_orders = step_executed_orders if minimal_output else sum(
                1 for o in all_order_attempts_log
                if o.get("step") == sim_step and o.get("n_trades_generated", 0) > 0
            )
            step_metrics = {
                "step": sim_step,
                "date": date,
                "hour": h_str,
                "intraday_step": intraday_idx,
                "intraday_steps_per_day": len(intraday_times),
                "retail_mode": retail_mode,
                "n_retail": N_retail,
                "n_stocks": len(stk_codes),
                "submitted_orders": submitted_orders,
                "valid_orders": valid_orders,
                "invalid_orders": submitted_orders - valid_orders,
                "executed_orders": executed_orders,
                "execution_rate": round(executed_orders / submitted_orders, 6) if submitted_orders else 0.0,
                "trades": len(step_trades),
                "buy_orders": n_buy,
                "sell_orders": n_sell,
                "elapsed_seconds": round(elapsed, 4),
                "semantic_seconds": round(semantic_seconds, 4),
                "retail_order_seconds": round(retail_order_seconds, 4),
                "institution_seconds": round(institution_seconds, 4),
                "vmm_seconds": round(vmm_seconds, 4),
                "matching_seconds": round(matching_seconds, 4),
                "state_update_seconds": round(state_update_seconds, 4),
                "llm_calls": total_llm_calls - step_llm_calls_start,
                "failed_llm_calls": total_failed_llm_calls - step_failed_llm_calls_start,
                "prompt_tokens": total_prompt_tokens - step_prompt_tokens_start,
                "completion_tokens": total_completion_tokens - step_completion_tokens_start,
                "total_llm_tokens": total_llm_tokens - step_llm_tokens_start,
            }
            all_step_metrics_log.append(step_metrics)
            print(
                f"  [Step {sim_step:3d} D{day_idx+1:02d}/{len(trading_dates):02d} "
                f"I{intraday_idx:02d}/{len(intraday_times):02d}] {date} {h_str} | "
                f"Orders={len(step_orders)} (B{n_buy}/S{n_sell}) | Trades={len(step_trades)} | "
                f"LLM={step_metrics['llm_calls']} fail={step_metrics['failed_llm_calls']} | "
                f"Time={elapsed:.1f}s"
            )
            if full_decision_log:
                print(
                    f"  [DecisionLog] Retail decisions appended: "
                    f"{len(step_retail_decision_records)} this step, "
                    f"{total_retail_decisions_logged} total -> {retail_decisions_jsonl}"
                )

            # --- INTERMEDIATE STEP SAVE ---
            if not minimal_output:
                try:
                    export_orders(all_orders_log, os.path.join(out_dir, "orders.json"))
                    export_trades(all_trades_log, os.path.join(out_dir, "trades.json"))
                    with open(os.path.join(out_dir, "order_attempts.json"), "w", encoding="utf-8") as f:
                        json.dump(all_order_attempts_log, f, ensure_ascii=False, indent=2, default=str)
                    with open(os.path.join(out_dir, "retail_decisions.json"), "w", encoding="utf-8") as f:
                        json.dump(all_retail_decisions_log, f, ensure_ascii=False, indent=2, default=str)
                    with open(os.path.join(out_dir, "shared_semantics.json"), "w", encoding="utf-8") as f:
                        json.dump(all_shared_semantics_log, f, ensure_ascii=False, indent=2, default=str)
                    with open(os.path.join(out_dir, "daily_agent_assets.json"), "w", encoding="utf-8") as f:
                        json.dump(all_daily_assets_log, f, ensure_ascii=False, indent=2, default=str)
                    with open(os.path.join(out_dir, "step_metrics.json"), "w", encoding="utf-8") as f:
                        json.dump(all_step_metrics_log, f, ensure_ascii=False, indent=2, default=str)
                    with open(os.path.join(out_dir, "agent_states.json"), "w", encoding="utf-8") as f:
                        json.dump(all_agent_states_log, f, ensure_ascii=False, indent=2)

                    step_posts = [
                        {
                            "step": p.day, "stock": p.stock_code, "author_id": p.author_id,
                            "style": p.author_type, "text": p.text, "action": p.action
                        }
                        for p in forum.posts
                    ]
                    with open(os.path.join(out_dir, "forum_posts.json"), "w", encoding="utf-8") as f:
                        json.dump(step_posts, f, ensure_ascii=False, indent=2)

                    step_inst_dec = []
                    for ia_temp in inst_agents:
                        step_inst_dec.extend(ia_temp.decision_log)
                    with open(os.path.join(out_dir, "inst_decisions.json"), "w", encoding="utf-8") as f:
                        json.dump(step_inst_dec, f, ensure_ascii=False, indent=2)

                    export_wealth_summary(retail_agents, inst_agents, vmm, dict(prices),
                                          os.path.join(out_dir, "wealth_summary.json"))
                except Exception as e:
                    print(f"  [Warning] Failed to save intermediate step results: {e}")
            else:
                try:
                    with open(os.path.join(out_dir, "step_metrics.json"), "w", encoding="utf-8") as f:
                        json.dump(all_step_metrics_log, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"  [Warning] Failed to save step metrics: {e}")

        # ── End of day sync ──
        market.end_of_day_roll(date)
        for s in stk_codes:
            sim_prices_history[s].append(prices[s])
            real_prices_history[s].append(real_today[s])

        day_stats = {
            "mean_value_signal": float(np.mean(day_value_signal_means)) if day_value_signal_means else 0.0,
            "mean_momentum_signal": float(np.mean(day_momentum_signal_means)) if day_momentum_signal_means else 0.0,
            "mean_noise_signal": float(np.mean(day_noise_signal_means)) if day_noise_signal_means else 0.0,
            "mean_obi": float(np.mean(day_obi_means)) if day_obi_means else 0.0,
            "order_count": day_order_count,
            "trade_count": day_trade_count,
        }
        audit_day = weekly_supervisor.record_day(date, dict(prices), dict(real_today), day_stats)
        weighted_sim_index_history.append(audit_day.sim_index)
        weighted_real_index_history.append(audit_day.real_index)

        wealth_list = [a.get_wealth(prices) for a in retail_agents]
        gini = compute_gini(wealth_list)
        gini_history.append({"step": sim_step, "date": date, "gini": round(gini, 4)})
        print(f"  >>> End of Day {date} | Gini: {gini:.4f}")

        weekly_dates_buffer.append(date)
        current_week = datetime.strptime(date, "%Y%m%d").isocalendar()[:2]
        next_week = None
        if day_idx + 1 < len(trading_dates):
            next_week = datetime.strptime(trading_dates[day_idx + 1], "%Y%m%d").isocalendar()[:2]
        if next_week != current_week or day_idx + 1 == len(trading_dates):
            week_key = f"{current_week[0]}-W{current_week[1]:02d}"
            weekly_record = weekly_supervisor.close_week(week_key, weekly_dates_buffer)
            if weekly_record is not None:
                weekly_snapshots.append({
                    "week_key": weekly_record.week_key,
                    "start_date": weekly_record.start_date,
                    "end_date": weekly_record.end_date,
                    "sim_return": weekly_record.sim_return,
                    "real_return": weekly_record.real_return,
                    "gap": weekly_record.gap,
                    "abs_gap": weekly_record.abs_gap,
                    "root_cause": weekly_record.root_cause,
                    "correction_strength": weekly_record.correction_strength,
                })
            weekly_dates_buffer = []

    # ════════════════════════════════════════════════════════════════════
    #  OUTPUT
    # ════════════════════════════════════════════════════════════════════

    print(f"\n{'='*60}")
    print("Simulation Complete")
    print(f"{'='*60}")

    # Equal-weight indices
    sim_index = compute_equal_weight_index(sim_prices_history)
    real_index = compute_equal_weight_index(real_prices_history)

    date_labels = ["init"] + trading_dates
    if not minimal_output:
        plot_comparison(sim_index, real_index, date_labels,
                        os.path.join(out_dir, "index_comparison.png"))
        plot_weighted_comparison(weighted_sim_index_history, weighted_real_index_history, date_labels,
                                 os.path.join(out_dir, "float_weighted_index_comparison.png"))

    # Exports
    if not minimal_output:
        export_orders(all_orders_log, os.path.join(out_dir, "orders.json"))
        export_trades(all_trades_log, os.path.join(out_dir, "trades.json"))
        with open(os.path.join(out_dir, "order_attempts.json"), "w", encoding="utf-8") as f:
            json.dump(all_order_attempts_log, f, ensure_ascii=False, indent=2, default=str)
        with open(os.path.join(out_dir, "retail_decisions.json"), "w", encoding="utf-8") as f:
            json.dump(all_retail_decisions_log, f, ensure_ascii=False, indent=2, default=str)
        with open(os.path.join(out_dir, "shared_semantics.json"), "w", encoding="utf-8") as f:
            json.dump(all_shared_semantics_log, f, ensure_ascii=False, indent=2, default=str)
        with open(os.path.join(out_dir, "daily_agent_assets.json"), "w", encoding="utf-8") as f:
            json.dump(all_daily_assets_log, f, ensure_ascii=False, indent=2, default=str)

    with open(os.path.join(out_dir, "step_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_step_metrics_log, f, ensure_ascii=False, indent=2, default=str)
    if lob_snapshot_log is not None:
        with open(os.path.join(out_dir, "lob_snapshots.json"), "w", encoding="utf-8") as f:
            json.dump(lob_snapshot_log, f, ensure_ascii=False, indent=2, default=str)
        print(f"[LOB] Snapshot saved: {len(lob_snapshot_log)} records → lob_snapshots.json")
    export_wealth_summary(
        retail_agents, inst_agents, vmm, prices,
        os.path.join(out_dir, "wealth_summary.json"),
        compact=minimal_output,
    )
    export_gini_history(gini_history, os.path.join(out_dir, "gini_history.json"))

    if not minimal_output:
        with open(os.path.join(out_dir, "agent_states.json"), "w", encoding="utf-8") as f:
            json.dump(all_agent_states_log, f, ensure_ascii=False, indent=2)

        with open(os.path.join(out_dir, "forum_recommendations.json"), "w", encoding="utf-8") as f:
            json.dump(forum.recommendation_log, f, ensure_ascii=False, indent=2)

        all_posts = [
            {
                "step": p.day, "stock": p.stock_code, "author_id": p.author_id,
                "style": p.author_type, "text": p.text, "action": p.action
            }
            for p in forum.posts
        ]
        with open(os.path.join(out_dir, "forum_posts.json"), "w", encoding="utf-8") as f:
            json.dump(all_posts, f, ensure_ascii=False, indent=2)

        # Export institution decision logs
        all_inst_decisions = []
        for ia in inst_agents:
            all_inst_decisions.extend(getattr(ia, "decision_log", []))
        with open(os.path.join(out_dir, "inst_decisions.json"), "w", encoding="utf-8") as f:
            json.dump(all_inst_decisions, f, ensure_ascii=False, indent=2)

        with open(os.path.join(out_dir, "daily_audit.json"), "w", encoding="utf-8") as f:
            json.dump(weekly_supervisor.export_daily(), f, ensure_ascii=False, indent=2, default=str)

        with open(os.path.join(out_dir, "weekly_audit.json"), "w", encoding="utf-8") as f:
            json.dump(weekly_supervisor.export_weekly(), f, ensure_ascii=False, indent=2, default=str)

        with open(os.path.join(out_dir, "weekly_corrections.json"), "w", encoding="utf-8") as f:
            json.dump(weekly_snapshots, f, ensure_ascii=False, indent=2, default=str)

    # ── Explicit exports required by downstream analysis ──
    try:
        with open(os.path.join(out_dir, "stock_price_history_sim.json"), "w", encoding="utf-8") as f:
            json.dump(sim_prices_history, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "stock_price_history_real.json"), "w", encoding="utf-8") as f:
            json.dump(real_prices_history, f, ensure_ascii=False, indent=2)

        rows = []
        for s in stk_codes:
            r_init, r_final = real_prices_history[s][0], real_prices_history[s][-1]
            s_init, s_final = sim_prices_history[s][0], sim_prices_history[s][-1]
            r_ret = (r_final / r_init - 1.0) if r_init > 0 else 0.0
            s_ret = (s_final / s_init - 1.0) if s_init > 0 else 0.0
            rows.append({
                "stock": s,
                "real_init": r_init, "real_final": r_final, "real_return": r_ret,
                "sim_init": s_init, "sim_final": s_final, "sim_return": s_ret,
                "return_gap": (r_ret - s_ret),
            })
        pd.DataFrame(rows).sort_values("return_gap").to_csv(
            os.path.join(out_dir, "stock_price_changes.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        idx_df = pd.DataFrame({
            "date": date_labels,
            "equal_weight_sim": sim_index,
            "equal_weight_real": real_index,
            "float_weight_sim": weighted_sim_index_history,
            "float_weight_real": weighted_real_index_history,
        })
        idx_df.to_csv(os.path.join(out_dir, "index_series.csv"), index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"[Warning] Failed to export price histories: {e}")

    run_meta = {
        "start_date": start_date,
        "end_date": end_date,
        "seed": seed,
        "mock_llm": mock_llm,
        "retail_mode": retail_mode,
        "prompt_baseline_skips_shared_semantics": retail_mode in {"stockagent", "twinmarket"},
        "openai_config_section": os.environ.get("OPENAI_CONFIG_SECTION", "openai"),
        "n_retail": N_retail,
        "n_inst": N_inst,
        "inst_early_news": int(inst_early_news),
        "stock_universe_csv": stock_universe_csv or os.path.join(data_root, "retail_100.csv"),
        "news_root": news_root,
        "output_dir": out_dir,
        "total_orders": cumulative_orders if minimal_output else len(all_orders_log),
        "total_order_attempts": cumulative_order_attempts if minimal_output else len(all_order_attempts_log),
        "total_trades": cumulative_trades if minimal_output else len(all_trades_log),
        "initial_gini": round(initial_gini, 4),
        "final_gini": round(gini_history[-1]["gini"], 4),
        "elapsed_seconds": round(time.time() - t0, 2),
        "total_llm_tokens": total_llm_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_llm_calls": total_llm_calls,
        "total_failed_llm_calls": total_failed_llm_calls,
        "prompt_baseline_workers": prompt_baseline_workers,
        "record_prompts": record_prompts,
        "llm_workers": llm_workers,
        "intraday_steps_per_day": max(1, int(intraday_steps_per_day or 1)),
        "total_steps": len(all_step_metrics_log),
        "max_stocks": max_stocks,
        "full_decision_log": full_decision_log,
        "minimal_output": minimal_output,
        "ablation_mode": ablation_mode,
        "ablation": ablation_meta,
        "semantic_cache_run": semantic_cache_run or "",
        "semantic_cache_hits": semantic_cache_hits,
        "semantic_cache_misses": semantic_cache_misses,
        "institution_cache_run": institution_cache_run or "",
        "institution_cache_hits": institution_cache_hits,
        "institution_cache_misses": institution_cache_misses,
        "prompt_baseline_cache_run": prompt_baseline_cache_run or "",
        "prompt_baseline_cache_hits": prompt_baseline_cache_hits,
        "prompt_baseline_cache_misses": prompt_baseline_cache_misses,
        "retail_decisions_jsonl": retail_decisions_jsonl if full_decision_log else "",
        "total_retail_decisions_logged": total_retail_decisions_logged,
        "weekly_records": len(weekly_supervisor.weekly_records),
        "daily_records": len(weekly_supervisor.daily_records),
    }
    with open(os.path.join(out_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2, default=str)

    # Print summary
    total_trades = cumulative_trades if minimal_output else len(all_trades_log)
    total_orders = cumulative_orders if minimal_output else len(all_orders_log)
    final_gini = gini_history[-1]["gini"]

    print(f"\n  Total orders: {total_orders}")
    print(f"  Total trades: {total_trades}")
    print(f"  Initial Gini: {initial_gini:.4f}")
    print(f"  Final Gini:   {final_gini:.4f}")
    print(f"  Elapsed:      {time.time() - t0:.1f}s")
    if not mock_llm:
        print(f"  Total LLM Tokens Consumed: {total_llm_tokens}")

    for style in ["value", "momentum", "noise"]:
        group = [a for a in retail_agents if a.style == style]
        init_w = sum(a.initial_wealth for a in group)
        final_w = sum(a.get_wealth(prices) for a in group)
        pct = (final_w / init_w - 1) * 100 if init_w > 0 else 0
        print(f"  [{style:10s}] n={len(group):3d}  init=¥{init_w:>14,.0f}  final=¥{final_w:>14,.0f}  ({pct:+.2f}%)")

    if inst_agents:
        init_w = sum(ia.initial_wealth for ia in inst_agents)
        final_w = sum(ia.get_wealth(prices) for ia in inst_agents)
        pct = (final_w / init_w - 1) * 100 if init_w > 0 else 0
        print(f"  [institution] n={len(inst_agents):3d}  init=¥{init_w:>14,.0f}  final=¥{final_w:>14,.0f}  ({pct:+.2f}%)")

    print("\n============================================================")
    print("Stock Level Performance (Real vs Simulated)")
    print("============================================================")
    for s in stk_codes:
        r_init, r_final = real_prices_history[s][0], real_prices_history[s][-1]
        s_init, s_final = sim_prices_history[s][0], sim_prices_history[s][-1]
        r_ret = (r_final / r_init - 1) * 100 if r_init > 0 else 0
        s_ret = (s_final / s_init - 1) * 100 if s_init > 0 else 0
        print(f"[{s:10s}] Real: {r_init:7.2f} -> {r_final:7.2f} ({r_ret:+6.2f}%) | "
              f"Sim: {s_init:7.2f} -> {s_final:7.2f} ({s_ret:+6.2f}%)")
              
    print(f"\nResults saved to: {out_dir}/")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SUM Market Simulation V2")
    parser.add_argument("--start", default="20250102", help="Start date YYYYMMDD")
    parser.add_argument("--end", default="20250630", help="End date YYYYMMDD")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mock-llm", type=str, default="True", help="Set to False to use real LLM")
    parser.add_argument(
        "--retail-mode",
        type=str,
        default="signal",
        choices=["signal", "stockagent", "twinmarket"],
        help="Retail decision mode: signal (SUM), stockagent, or twinmarket",
    )
    parser.add_argument("--N", type=int, default=300, help="Number of retail agents")
    parser.add_argument("--inst-early-news", type=int, default=1, choices=[0, 1], help="Institution early read next-day news: 0=No, 1=Yes")
    parser.add_argument("--stock-universe-csv", default=os.path.join(os.path.dirname(__file__), "..", "data_config", "retail_100.csv"), help="Universe CSV path")
    parser.add_argument("--news-root", default=os.path.join(os.path.dirname(__file__), "..", "data_config", "news"), help="News root path")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: output)")
    parser.add_argument("--prompt-baseline-workers", type=int, default=10, help="Concurrent workers for prompt baseline retail decisions")
    parser.add_argument("--pure-llm-workers", type=int, default=10, help=argparse.SUPPRESS)
    parser.add_argument("--record-prompts", type=str, default="True", help="Whether to save rendered prompts in retail_decisions.json")
    parser.add_argument("--max-stocks", type=int, default=56, help="Maximum number of valid stocks to keep from the universe CSV")
    parser.add_argument("--full-decision-log", type=str, default="False", help="Stream every retail signal decision to retail_decisions.jsonl")
    parser.add_argument("--llm-workers", type=int, default=10, help="Concurrent workers for SUM shared semantics and institution LLM calls")
    parser.add_argument(
        "--ablation-mode",
        type=str,
        default="none",
        choices=[
            "none",
            "macro_sum_institution",
            "meso_noise_channel",
            "micro_default_gamma_sigma",
            "no_news_technical_only",
            "no_forum",
        ],
        help="Ablation mode (includes no_news_technical_only and no_forum controls).",
    )
    parser.add_argument("--semantic-cache-run", default=None, help="Optional run directory containing shared_semantics.json to reuse.")
    parser.add_argument("--institution-cache-run", default=None, help="Optional run directory containing inst_decisions.json to reuse.")
    parser.add_argument(
        "--prompt-baseline-cache-run",
        default=None,
        help=(
            "Same-framework replay only: directory of a prior stockagent or "
            "twinmarket run containing retail_decisions.json. Cross-baseline "
            "replay (e.g. pure_llm -> stockagent) is disabled."
        ),
    )
    parser.add_argument(
        "--minimal-output",
        type=str,
        default="False",
        help="Skip heavy per-agent logs for extreme-N cache replay runs.",
    )
    parser.add_argument(
        "--lob-snapshot",
        type=str,
        default="False",
        help="Capture LOB depth/spread after each day's matching phase.",
    )
    parser.add_argument(
        "--intraday-steps-per-day",
        type=int,
        default=1,
        help="Number of evenly spaced LOB observation/matching steps per trading day.",
    )
    args = parser.parse_args()
    
    use_mock_llm = args.mock_llm.lower() in ("true", "1", "t", "yes")
    record_prompts = args.record_prompts.lower() in ("true", "1", "t", "yes")
    full_decision_log = args.full_decision_log.lower() in ("true", "1", "t", "yes")
    minimal_output = args.minimal_output.lower() in ("true", "1", "t", "yes")
    use_lob_snapshot = args.lob_snapshot.lower() in ("true", "1", "t", "yes")

    run_simulation(
        start_date=args.start,
        end_date=args.end,
        seed=args.seed,
        mock_llm=use_mock_llm,
        N_retail=args.N,
        stock_universe_csv=args.stock_universe_csv,
        news_root=args.news_root,
        output_dir=args.output_dir,
        inst_early_news=args.inst_early_news,
        retail_mode=args.retail_mode,
        prompt_baseline_workers=args.prompt_baseline_workers or args.pure_llm_workers,
        record_prompts=record_prompts,
        max_stocks=args.max_stocks,
        full_decision_log=full_decision_log,
        llm_workers=args.llm_workers,
        ablation_mode=args.ablation_mode,
        semantic_cache_run=args.semantic_cache_run,
        institution_cache_run=args.institution_cache_run,
        prompt_baseline_cache_run=args.prompt_baseline_cache_run,
        minimal_output=minimal_output,
        lob_snapshot=use_lob_snapshot,
        intraday_steps_per_day=args.intraday_steps_per_day,
    )
