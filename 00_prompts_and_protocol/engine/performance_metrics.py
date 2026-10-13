"""
Metrics: Gini coefficient, equal-weight index, plotting, and export.
"""

import json
import math
import os
from typing import Dict, List, Optional

import numpy as np


def compute_gini(wealth_list: List[float]) -> float:
    """Compute Gini coefficient of a wealth distribution."""
    arr = np.array(sorted(wealth_list))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr)))


def compute_equal_weight_index(
    prices_history: Dict[str, List[float]],
    base: float = 1000.0,
) -> List[float]:
    """
    Compute equal-weight index from daily close prices.
    prices_history: {stock: [price_day0, price_day1, ...]}
    Returns: [index_day0, index_day1, ...]
    """
    stocks = list(prices_history.keys())
    if not stocks:
        return []
    n_days = min(len(v) for v in prices_history.values())
    if n_days == 0:
        return []

    index = [base]
    for d in range(1, n_days):
        returns = []
        for s in stocks:
            p_prev = prices_history[s][d - 1]
            p_curr = prices_history[s][d]
            if p_prev > 0:
                returns.append(p_curr / p_prev - 1)
        if returns:
            avg_r = sum(returns) / len(returns)
            index.append(index[-1] * (1 + avg_r))
        else:
            index.append(index[-1])
    return index


def compute_weighted_market_index(
    prices_history: Dict[str, List[float]],
    weights: Dict[str, float],
    base: float = 1000.0,
) -> List[float]:
    """
    Compute a market-cap weighted index from price histories.
    weights should already be normalized or will be normalized internally.
    """
    stocks = [s for s in prices_history.keys() if s in weights]
    if not stocks:
        return []

    n_days = min(len(prices_history[s]) for s in stocks)
    if n_days == 0:
        return []

    total_w = sum(max(weights.get(s, 0.0), 0.0) for s in stocks)
    if total_w <= 0:
        norm_weights = {s: 1.0 / len(stocks) for s in stocks}
    else:
        norm_weights = {s: max(weights.get(s, 0.0), 0.0) / total_w for s in stocks}

    index = [base]
    for d in range(1, n_days):
        weighted_ret = 0.0
        for s in stocks:
            p_prev = prices_history[s][d - 1]
            p_curr = prices_history[s][d]
            if p_prev > 0:
                weighted_ret += norm_weights[s] * (p_curr / p_prev - 1)
        index.append(index[-1] * (1 + weighted_ret))
    return index


def plot_comparison(
    sim_index: List[float],
    real_index: List[float],
    dates: List[str],
    output_path: str,
    title: str = "Simulated vs Real Equal-Weight Index",
):
    """Plot simulated vs real equal-weight index K-line."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        x = list(range(len(dates)))
        ax.plot(x, sim_index[:len(x)], "b-o", label="Simulated Equal-Weight Index", linewidth=2)
        if real_index:
            ax.plot(x, real_index[:len(x)], "r-s", label="Real Equal-Weight Index", linewidth=2)

        ax.set_xticks(x)
        ax.set_xticklabels(dates, rotation=45, ha="right")
        ax.set_ylabel("Index Value")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"[Metrics] Chart saved: {output_path}")
    except ImportError as e:
        print(f"[Metrics] Cannot plot (missing lib): {e}")


def plot_weighted_comparison(
    sim_index: List[float],
    real_index: List[float],
    dates: List[str],
    output_path: str,
    title: str = "Simulated vs Real Float-Weighted Index",
):
    """Plot weighted index comparison with the same style as the equal-weight chart."""
    plot_comparison(sim_index, real_index, dates, output_path, title=title)


def export_orders(orders: List[Dict], path: str):
    """Export order log to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(orders, f, ensure_ascii=False, indent=2, default=str)
    print(f"[Metrics] Orders saved: {path} ({len(orders)} records)")


def export_trades(trades: List[Dict], path: str):
    """Export trade log to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trades, f, ensure_ascii=False, indent=2, default=str)
    print(f"[Metrics] Trades saved: {path} ({len(trades)} records)")


def export_wealth_summary(
    retail_agents,
    inst_agents,
    vmm,
    prices: Dict[str, float],
    path: str,
    compact: bool = False,
):
    """Export per-type initial/final wealth summary."""
    summary = {"by_type": {}, "gini_final": 0.0, "compact": compact}

    # Group by type
    type_groups = {}
    wealth_list = []
    for a in retail_agents:
        w = a.get_wealth(prices)
        wealth_list.append(w)
        t = a.style
        if t not in type_groups:
            type_groups[t] = {"n": 0, "initial_total": 0, "final_total": 0}
            if not compact:
                type_groups[t]["agents"] = []
        type_groups[t]["n"] += 1
        type_groups[t]["initial_total"] += a.initial_wealth
        type_groups[t]["final_total"] += w
        if not compact:
            type_groups[t]["agents"].append({
                "id": a.agent_id, "initial": round(a.initial_wealth, 2), "final": round(w, 2)
            })

    inst_summary = {"n": 0, "initial_total": 0.0, "final_total": 0.0}
    if not compact:
        inst_summary["agents"] = []
    for ia in inst_agents:
        w = ia.get_wealth(prices)
        wealth_list.append(w)
        inst_summary["n"] += 1
        inst_summary["initial_total"] += ia.initial_wealth
        inst_summary["final_total"] += w
        if not compact:
            inst_summary["agents"].append({"id": ia.agent_id, "initial": round(ia.initial_wealth, 2), "final": round(w, 2)})
    if inst_summary["n"] > 0:
        type_groups["institution"] = inst_summary

    if vmm:
        vmm_w = vmm.get_wealth(prices)
        type_groups["vmm"] = {
            "n": 1,
            "initial_total": 0,
            "final_total": vmm_w,
        }

    summary["by_type"] = type_groups
    summary["gini_final"] = compute_gini(wealth_list)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"[Metrics] Wealth summary saved: {path}")


def export_gini_history(gini_list: List[Dict], path: str):
    """Export Gini coefficient per step."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(gini_list, f, ensure_ascii=False, indent=2, default=str)
    print(f"[Metrics] Gini history saved: {path}")
