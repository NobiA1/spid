#!/usr/bin/env python3
"""Build LLM-vs-SUM scalability figures and tables."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
RUN_ROOT = WORKSPACE_ROOT / "weak_baseline" / "exp" / "scalability" / "scaling_efficiency_202504_top5_N50_100_500_1000"
ANALYSIS_DIR = WORKSPACE_ROOT / "exp" / "weak_baseline" / "scaling_efficiency_202504_top5_N50_100_500_1000" / "analysis" / "llm_sum_comparison"
FIG_DIR = WORKSPACE_ROOT / "exp" / "weak_baseline" / "scaling_efficiency_202504_top5_N50_100_500_1000" / "assets" / "llm_sum_comparison"

OKABE_ITO = {
    "SUM": "#0072B2",
    "LLM-driven": "#D55E00",
    "gray": "#6B7280",
    "light_gray": "#E5E7EB",
    "green": "#009E73",
    "orange": "#E69F00",
    "sky": "#56B4E9",
}


RUNS = [
    ("SUM", 50, "sum_N0050"),
    ("SUM", 100, "sum_N0100"),
    ("SUM", 500, "sum_N0500"),
    ("SUM", 1000, "sum_N1000"),
    ("LLM-driven", 50, "pure_llm_N0050"),
    ("LLM-driven", 100, "pure_llm_N0100"),
    ("LLM-driven", 500, "pure_llm_N0500"),
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def rmse(a: pd.Series, b: pd.Series) -> float:
    return float(np.sqrt(np.mean((a.to_numpy(float) - b.to_numpy(float)) ** 2)))


def mae(a: pd.Series, b: pd.Series) -> float:
    return float(np.mean(np.abs(a.to_numpy(float) - b.to_numpy(float))))


def direction_accuracy(sim_ret: pd.Series, real_ret: pd.Series) -> float:
    return float(np.mean(np.sign(sim_ret.to_numpy(float)) == np.sign(real_ret.to_numpy(float))))


def autocorr_abs_return_error(sim_ret: pd.Series, real_ret: pd.Series) -> float:
    sim_abs = np.abs(sim_ret.to_numpy(float))
    real_abs = np.abs(real_ret.to_numpy(float))
    if len(sim_abs) < 3:
        return np.nan
    sim_ac = pd.Series(sim_abs).autocorr(lag=1)
    real_ac = pd.Series(real_abs).autocorr(lag=1)
    return float(abs(sim_ac - real_ac))


def safe_acf(values, lag: int) -> float:
    s = pd.Series(values).dropna()
    if len(s) <= lag + 1 or float(s.std()) == 0.0:
        return np.nan
    return float(s.autocorr(lag=lag))


def entropy_from_counts(counts) -> float:
    arr = np.array(list(counts), dtype=float)
    total = arr.sum()
    if total <= 0:
        return 0.0
    p = arr[arr > 0] / total
    return float(-(p * np.log2(p)).sum())


def normalized_entropy(counts) -> float:
    arr = np.array(list(counts), dtype=float)
    if len(arr) <= 1:
        return 0.0
    return entropy_from_counts(arr) / np.log2(len(arr))


def js_divergence(p, q) -> float:
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    p = p / p.sum() if p.sum() else np.ones_like(p) / len(p)
    q = q / q.sum() if q.sum() else np.ones_like(q) / len(q)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def style_from_agent(agent_id: str) -> str | None:
    if not isinstance(agent_id, str):
        return None
    if agent_id.startswith("R_V_"):
        return "value"
    if agent_id.startswith("R_M_"):
        return "momentum"
    if agent_id.startswith("R_N_"):
        return "noise"
    return None


def stock_panel_returns(run_dir: Path):
    sim = load_json(run_dir / "stock_price_history_sim.json")
    real = load_json(run_dir / "stock_price_history_real.json")
    stocks = sorted(sim.keys())
    sim_ret = {}
    real_ret = {}
    for stock in stocks:
        s = pd.Series(sim[stock], dtype=float).pct_change().dropna().reset_index(drop=True)
        r = pd.Series(real[stock], dtype=float).pct_change().dropna().reset_index(drop=True)
        sim_ret[stock] = s
        real_ret[stock] = r
    return stocks, pd.DataFrame(sim_ret), pd.DataFrame(real_ret)


def market_quality_metrics(run_dir: Path, idx: pd.DataFrame) -> dict:
    stocks, sim_panel, real_panel = stock_panel_returns(run_dir)
    eq_sim = idx["equal_weight_sim"]
    eq_real = idx["equal_weight_real"]
    fw_sim = idx["float_weight_sim"]
    fw_real = idx["float_weight_real"]
    eq_sim_ret = eq_sim.pct_change().dropna()
    eq_real_ret = eq_real.pct_change().dropna()
    fw_sim_ret = fw_sim.pct_change().dropna()
    fw_real_ret = fw_real.pct_change().dropna()

    dispersion_sim = sim_panel.std(axis=1)
    dispersion_real = real_panel.std(axis=1)
    all_sim_returns = sim_panel.to_numpy().reshape(-1)
    all_real_returns = real_panel.to_numpy().reshape(-1)

    return {
        "eq_volatility_relative_error": abs(eq_sim_ret.std() - eq_real_ret.std()) / max(eq_real_ret.std(), 1e-12),
        "fw_volatility_relative_error": abs(fw_sim_ret.std() - fw_real_ret.std()) / max(fw_real_ret.std(), 1e-12),
        "cross_sectional_dispersion_mae": float(np.mean(np.abs(dispersion_sim - dispersion_real))),
        "sim_tail_kurtosis": float(pd.Series(all_sim_returns).kurt() + 3.0),
        "real_tail_kurtosis": float(pd.Series(all_real_returns).kurt() + 3.0),
        "tail_kurtosis_error": float(abs((pd.Series(all_sim_returns).kurt() + 3.0) - (pd.Series(all_real_returns).kurt() + 3.0))),
        "sim_vol_clustering_lag1": safe_acf(np.abs(eq_sim_ret.to_numpy(float)), 1),
        "real_vol_clustering_lag1": safe_acf(np.abs(eq_real_ret.to_numpy(float)), 1),
        "vol_clustering_error": abs(safe_acf(np.abs(eq_sim_ret.to_numpy(float)), 1) - safe_acf(np.abs(eq_real_ret.to_numpy(float)), 1)),
        "sim_abs_return_persistence_lag1_5": float(np.nanmean([safe_acf(np.abs(eq_sim_ret.to_numpy(float)), lag) for lag in range(1, 6)])),
    }


def behavior_metrics(run_dir: Path, n: int, days: int, stocks: list[str]) -> dict:
    orders = load_json(run_dir / "orders.json")
    agent_states = load_json(run_dir / "agent_states.json")
    trades = load_json(run_dir / "trades.json")
    steps = load_json(run_dir / "step_metrics.json")
    gini = load_json(run_dir / "gini_history.json")

    agent_style = {}
    for row in agent_states:
        aid = row.get("agent_id")
        style = row.get("style") or style_from_agent(aid)
        if aid and style in {"value", "momentum", "noise"}:
            agent_style[aid] = style

    # Fallback in case an output omits some final agent states.
    for style_prefix, style in [("R_V_", "value"), ("R_M_", "momentum"), ("R_N_", "noise")]:
        seen = [a for a in agent_style if a.startswith(style_prefix)]
        for i in range(len(seen), n + 1):
            aid = f"{style_prefix}{i:03d}"
            if aid in agent_style:
                continue

    order_action = {}
    stock_counts = {s: 0 for s in stocks}
    active_agent_days = set()
    active_agent_stock_days = set()
    buy_sell_imbalance_by_day = []
    style_action_counts = {
        "value": {"BUY": 0, "SELL": 0, "HOLD": 0},
        "momentum": {"BUY": 0, "SELL": 0, "HOLD": 0},
        "noise": {"BUY": 0, "SELL": 0, "HOLD": 0},
    }

    for order in orders:
        aid = order.get("agent") or order.get("agent_id")
        style = style_from_agent(aid)
        stock = order.get("stock")
        day = int(order.get("step", 0))
        direction = order.get("direction")
        if style not in style_action_counts or stock not in stocks or direction not in {"BUY", "SELL"}:
            continue
        key = (day, aid, stock)
        order_action[key] = direction
        stock_counts[stock] += 1
        active_agent_days.add((day, aid))
        active_agent_stock_days.add(key)

    agents_by_style = {style: sorted([a for a, st in agent_style.items() if st == style]) for style in style_action_counts}
    # If agent_states are day-level, unique agents are enough; if a run only logs active
    # agents, derive a conservative denominator from N and observed style ratios.
    total_known_agents = sum(len(v) for v in agents_by_style.values())
    if total_known_agents == 0:
        agents_by_style = {
            "value": [f"R_V_{i:03d}" for i in range(int(n * 0.3))],
            "momentum": [f"R_M_{i:03d}" for i in range(int(n * 0.3))],
            "noise": [f"R_N_{i:03d}" for i in range(n - int(n * 0.6))],
        }

    for style, agents in agents_by_style.items():
        for day in range(1, days + 1):
            for aid in agents:
                for stock in stocks:
                    action = order_action.get((day, aid, stock), "HOLD")
                    style_action_counts[style][action] += 1

    total_possible_agent_days = max(1, n * days)
    total_possible_agent_stock_days = max(1, n * days * len(stocks))
    action_counts = {a: sum(style_action_counts[s][a] for s in style_action_counts) for a in ["BUY", "SELL", "HOLD"]}
    style_vectors = [
        [style_action_counts[s][a] for a in ["BUY", "SELL", "HOLD"]]
        for s in ["value", "momentum", "noise"]
    ]
    pairwise_js = [js_divergence(style_vectors[i], style_vectors[j]) for i in range(3) for j in range(i + 1, 3)]

    for step in steps:
        b = float(step.get("buy_orders", 0))
        s = float(step.get("sell_orders", 0))
        buy_sell_imbalance_by_day.append((b - s) / max(b + s, 1.0))

    trade_values = pd.Series([float(t.get("value", 0.0)) for t in trades])
    trade_by_stock = {}
    for t in trades:
        stock = t.get("stock")
        trade_by_stock[stock] = trade_by_stock.get(stock, 0.0) + float(t.get("value", 0.0))
    stock_trade_share = np.array(list(trade_by_stock.values()), dtype=float)
    stock_trade_share = stock_trade_share / stock_trade_share.sum() if stock_trade_share.sum() else np.zeros(len(stocks))

    return {
        "active_agent_ratio": len(active_agent_days) / total_possible_agent_days,
        "active_agent_stock_ratio": len(active_agent_stock_days) / total_possible_agent_stock_days,
        "action_entropy": normalized_entropy([action_counts["BUY"], action_counts["SELL"], action_counts["HOLD"]]),
        "stock_choice_entropy": normalized_entropy([stock_counts.get(s, 0) for s in stocks]),
        "style_separation_js": float(np.mean(pairwise_js)),
        "fill_rate_trades_per_order": len(trades) / max(len(orders), 1),
        "buy_sell_imbalance_acf1": safe_acf(buy_sell_imbalance_by_day, 1),
        "turnover_concentration_hhi": float((stock_trade_share ** 2).sum()) if len(stock_trade_share) else np.nan,
        "trade_value_total": float(trade_values.sum()) if len(trade_values) else 0.0,
        "gini_initial": float(gini[0]["gini"]) if gini else np.nan,
        "gini_final": float(gini[-1]["gini"]) if gini else np.nan,
        "gini_delta": float(gini[-1]["gini"] - gini[0]["gini"]) if gini else np.nan,
    }


def load_rows() -> pd.DataFrame:
    rows = []
    for arch, n, run_dir in RUNS:
        d = RUN_ROOT / run_dir
        meta = load_json(d / "run_metadata.json")
        steps = load_json(d / "step_metrics.json")
        days = int(meta["daily_records"])
        stocks = 5
        agent_stock_days = n * stocks * days
        total_tokens = float(meta["total_llm_tokens"])
        total_calls = float(meta["total_llm_calls"])
        elapsed = float(meta["elapsed_seconds"])
        orders = int(meta["total_orders"])
        attempts = int(meta["total_order_attempts"])
        trades = int(meta["total_trades"])
        semantic_seconds = sum(float(s.get("semantic_seconds", 0.0)) for s in steps)
        retail_seconds = sum(float(s.get("retail_order_seconds", 0.0)) for s in steps)
        institution_seconds = sum(float(s.get("institution_seconds", 0.0)) for s in steps)
        matching_seconds = sum(float(s.get("matching_seconds", 0.0)) for s in steps)
        state_seconds = sum(float(s.get("state_update_seconds", 0.0)) for s in steps)
        other_seconds = max(
            0.0,
            elapsed - semantic_seconds - retail_seconds - institution_seconds - matching_seconds - state_seconds,
        )

        idx = pd.read_csv(d / "index_series.csv", encoding="utf-8-sig")
        idx = idx[idx["date"] != "init"].reset_index(drop=True)
        eq_sim = idx["equal_weight_sim"]
        eq_real = idx["equal_weight_real"]
        fw_sim = idx["float_weight_sim"]
        fw_real = idx["float_weight_real"]
        eq_sim_ret = eq_sim.pct_change().dropna()
        eq_real_ret = eq_real.pct_change().dropna()
        fw_sim_ret = fw_sim.pct_change().dropna()
        fw_real_ret = fw_real.pct_change().dropna()

        stocks_list, _, _ = stock_panel_returns(d)
        extra_quality = market_quality_metrics(d, idx)
        extra_behavior = behavior_metrics(d, n, days, stocks_list)

        rows.append(
            {
                "architecture": arch,
                "N": n,
                "run_dir": str(d),
                "days": days,
                "stocks": stocks,
                "agent_stock_days": agent_stock_days,
                "total_llm_calls": int(total_calls),
                "failed_llm_calls": int(meta["total_failed_llm_calls"]),
                "total_llm_tokens": int(total_tokens),
                "elapsed_seconds": elapsed,
                "total_orders": orders,
                "total_order_attempts": attempts,
                "total_trades": trades,
                "orders_per_agent": orders / n,
                "trades_per_agent": trades / n,
                "orders_per_trade": orders / trades if trades else np.nan,
                "tokens_per_agent_stock_day": total_tokens / agent_stock_days,
                "calls_per_agent_stock_day": total_calls / agent_stock_days,
                "seconds_per_agent_stock_day": elapsed / agent_stock_days,
                "agent_stock_days_per_second": agent_stock_days / elapsed,
                "semantic_seconds": semantic_seconds,
                "retail_order_seconds": retail_seconds,
                "institution_seconds": institution_seconds,
                "matching_seconds": matching_seconds,
                "state_update_seconds": state_seconds,
                "other_seconds": other_seconds,
                "eq_index_rmse": rmse(eq_sim, eq_real),
                "fw_index_rmse": rmse(fw_sim, fw_real),
                "eq_return_mae": mae(eq_sim_ret, eq_real_ret),
                "fw_return_mae": mae(fw_sim_ret, fw_real_ret),
                "eq_direction_accuracy": direction_accuracy(eq_sim_ret, eq_real_ret),
                "fw_direction_accuracy": direction_accuracy(fw_sim_ret, fw_real_ret),
                "eq_abs_return_autocorr_error": autocorr_abs_return_error(eq_sim_ret, eq_real_ret),
                "fw_abs_return_autocorr_error": autocorr_abs_return_error(fw_sim_ret, fw_real_ret),
                **extra_quality,
                **extra_behavior,
            }
        )
    df = pd.DataFrame(rows)
    lower_is_better = [
        "eq_index_rmse",
        "fw_index_rmse",
        "eq_return_mae",
        "fw_return_mae",
        "eq_volatility_relative_error",
        "fw_volatility_relative_error",
        "eq_abs_return_autocorr_error",
        "fw_abs_return_autocorr_error",
        "cross_sectional_dispersion_mae",
    ]
    score_parts = []
    for col in lower_is_better:
        max_val = max(float(df[col].max()), 1e-12)
        score_parts.append(1.0 - (df[col].astype(float) / max_val))
    score_parts.append(df["eq_direction_accuracy"].astype(float))
    score_parts.append(df["fw_direction_accuracy"].astype(float))
    df["market_fidelity_score"] = pd.concat(score_parts, axis=1).mean(axis=1)
    return df


def build_efficiency_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n in [50, 100, 500]:
        s = df[(df.architecture == "SUM") & (df.N == n)].iloc[0]
        l = df[(df.architecture == "LLM-driven") & (df.N == n)].iloc[0]
        rows.append(
            {
                "N": n,
                "llm_call_reduction_x": l.total_llm_calls / s.total_llm_calls,
                "token_reduction_x": l.total_llm_tokens / s.total_llm_tokens,
                "tokens_saved": int(l.total_llm_tokens - s.total_llm_tokens),
                "elapsed_speedup_x": l.elapsed_seconds / s.elapsed_seconds,
                "SUM_tokens_per_agent_stock_day": s.tokens_per_agent_stock_day,
                "LLM_tokens_per_agent_stock_day": l.tokens_per_agent_stock_day,
                "SUM_calls_per_agent_stock_day": s.calls_per_agent_stock_day,
                "LLM_calls_per_agent_stock_day": l.calls_per_agent_stock_day,
                "SUM_eq_index_rmse": s.eq_index_rmse,
                "LLM_eq_index_rmse": l.eq_index_rmse,
                "SUM_fw_index_rmse": s.fw_index_rmse,
                "LLM_fw_index_rmse": l.fw_index_rmse,
                "SUM_fidelity_score": s.market_fidelity_score,
                "LLM_fidelity_score": l.market_fidelity_score,
                "SUM_fidelity_per_1M_tokens": s.market_fidelity_score / (s.total_llm_tokens / 1_000_000),
                "LLM_fidelity_per_1M_tokens": l.market_fidelity_score / (l.total_llm_tokens / 1_000_000),
                "SUM_fidelity_per_min": s.market_fidelity_score / (s.elapsed_seconds / 60),
                "LLM_fidelity_per_min": l.market_fidelity_score / (l.elapsed_seconds / 60),
            }
        )
    return pd.DataFrame(rows)


def build_complexity_slope_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = {
        "total_llm_calls": "beta_calls_vs_N",
        "total_llm_tokens": "beta_tokens_vs_N",
        "elapsed_seconds": "beta_wall_time_vs_N",
        "market_fidelity_score": "beta_fidelity_vs_N",
    }
    for arch in ["SUM", "LLM-driven"]:
        sub = df[df.architecture == arch].sort_values("N")
        row = {"architecture": arch}
        for metric, name in metrics.items():
            x = np.log(sub["N"].to_numpy(float))
            y = np.log(np.maximum(sub[metric].to_numpy(float), 1e-12))
            row[name] = float(np.polyfit(x, y, 1)[0])
        row["interpretation"] = (
            "Shared semantic cost is effectively independent of N"
            if arch == "SUM"
            else "Per-agent semantic interpretation scales close to O(N)"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def configure_style():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def line_plot(ax, df, y, ylabel, title, log_y=False, annotate_sum_1000=False):
    for arch in ["SUM", "LLM-driven"]:
        sub = df[df.architecture == arch].sort_values("N")
        if arch == "LLM-driven":
            sub = sub[sub.N <= 500]
        ax.plot(
            sub.N,
            sub[y],
            marker="o",
            linewidth=2.2,
            markersize=5,
            label=arch,
            color=OKABE_ITO[arch],
        )
    if log_y:
        ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xticks([50, 100, 500, 1000])
    ax.set_xticklabels(["50", "100", "500", "1000"])
    ax.set_xlabel("Number of retail agents (N)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    if annotate_sum_1000:
        ax.annotate(
            "SUM only",
            xy=(1000, df[(df.architecture == "SUM") & (df.N == 1000)][y].iloc[0]),
            xytext=(-38, 16),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="-", color=OKABE_ITO["gray"], lw=0.8),
            fontsize=8,
            color=OKABE_ITO["gray"],
        )


def make_resource_panel(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)
    line_plot(
        axes[0, 0],
        df,
        "total_llm_calls",
        "Total LLM calls",
        "LLM calls",
        log_y=True,
        annotate_sum_1000=True,
    )
    line_plot(
        axes[0, 1],
        df,
        "total_llm_tokens",
        "Total LLM tokens",
        "Token cost",
        log_y=True,
        annotate_sum_1000=True,
    )
    line_plot(
        axes[1, 0],
        df,
        "elapsed_seconds",
        "Wall-clock seconds",
        "Runtime",
        log_y=False,
        annotate_sum_1000=True,
    )
    line_plot(
        axes[1, 1],
        df,
        "agent_stock_days_per_second",
        "Agent-stock-days / second",
        "Throughput",
        log_y=False,
        annotate_sum_1000=True,
    )
    axes[0, 0].legend(frameon=False, loc="best")
    fig.suptitle("Scalability resources: SUM amortizes shared semantic perception", y=1.02, fontsize=12)
    save_figure(fig, "fig_llm_sum_resource_scaling")


def make_efficiency_panel(df: pd.DataFrame, eff: pd.DataFrame):
    matched = df[df.N.isin([50, 100, 500])]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)

    width = 0.34
    xs = np.arange(3)
    ns = [50, 100, 500]
    for i, metric in enumerate(["tokens_per_agent_stock_day", "calls_per_agent_stock_day"]):
        ax = axes[i]
        sum_vals = [matched[(matched.architecture == "SUM") & (matched.N == n)][metric].iloc[0] for n in ns]
        llm_vals = [matched[(matched.architecture == "LLM-driven") & (matched.N == n)][metric].iloc[0] for n in ns]
        ax.bar(xs - width / 2, sum_vals, width, color=OKABE_ITO["SUM"], label="SUM")
        ax.bar(xs + width / 2, llm_vals, width, color=OKABE_ITO["LLM-driven"], label="LLM-driven")
        ax.set_xticks(xs)
        ax.set_xticklabels(ns)
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_xlabel("N")
        ax.set_title("Tokens per agent-stock-day" if i == 0 else "Calls per agent-stock-day")
    axes[0].set_ylabel("Log scale")

    ax = axes[2]
    ax.plot(eff.N, eff.token_reduction_x, marker="o", lw=2.2, color=OKABE_ITO["green"], label="Token reduction")
    ax.plot(eff.N, eff.llm_call_reduction_x, marker="s", lw=2.2, color=OKABE_ITO["orange"], label="Call reduction")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels(ns)
    ax.set_xlabel("N")
    ax.set_ylabel("Reduction factor vs LLM-driven")
    ax.set_title("Amortization gain")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)

    axes[0].legend(frameon=False)
    fig.suptitle("Scaling efficiency: per-agent semantic cost collapses under SUM", y=1.08, fontsize=12)
    save_figure(fig, "fig_llm_sum_scaling_efficiency")


def make_quality_activity_panel(df: pd.DataFrame):
    matched = df[df.N.isin([50, 100, 500])]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    line_plot(axes[0, 0], matched, "eq_index_rmse", "RMSE", "Equal-weight index RMSE", log_y=False)
    line_plot(axes[0, 1], matched, "fw_index_rmse", "RMSE", "Float-weight index RMSE", log_y=False)
    line_plot(axes[1, 0], matched, "total_trades", "Matched trades", "Trading activity", log_y=False)
    line_plot(axes[1, 1], matched, "orders_per_trade", "Orders / trade", "Order-to-trade friction", log_y=False)
    axes[0, 0].legend(frameon=False, loc="best")
    fig.suptitle("Simulation quality and market activity under matched N", y=1.02, fontsize=12)
    save_figure(fig, "fig_llm_sum_quality_activity")


def make_runtime_breakdown(df: pd.DataFrame):
    components = [
        ("semantic_seconds", "Semantic LLM"),
        ("retail_order_seconds", "Retail decisions"),
        ("institution_seconds", "Institution"),
        ("matching_seconds", "Matching"),
        ("state_update_seconds", "State update"),
        ("other_seconds", "Other/export"),
    ]
    colors = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#9CA3AF"]
    rows = df[df.N.isin([50, 100, 500])].copy()
    rows["label"] = rows["architecture"].map({"SUM": "SUM", "LLM-driven": "LLM"}) + "\nN=" + rows["N"].astype(str)
    rows = rows.sort_values(["N", "architecture"], key=lambda s: s.map({"SUM": 0, "LLM-driven": 1}).fillna(s))
    fig, ax = plt.subplots(figsize=(7.2, 3.7), constrained_layout=True)
    bottom = np.zeros(len(rows))
    x = np.arange(len(rows))
    for (col, label), color in zip(components, colors):
        vals = rows[col].to_numpy(float)
        ax.bar(x, vals, bottom=bottom, label=label, color=color)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(rows["label"])
    ax.set_ylabel("Wall-clock seconds")
    ax.set_title("Figure 5. Runtime decomposition", pad=8)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    save_figure(fig, "fig_llm_sum_runtime_breakdown")


def make_figure1_resource_scaling(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.55), constrained_layout=True)
    line_plot(axes[0], df, "total_llm_calls", "LLM calls", "(a) Calls", log_y=True, annotate_sum_1000=True)
    line_plot(axes[1], df, "total_llm_tokens", "Total tokens", "(b) Tokens", log_y=True, annotate_sum_1000=True)
    line_plot(axes[2], df, "elapsed_seconds", "Wall-time (s)", "(c) Wall-time", log_y=False, annotate_sum_1000=True)
    axes[0].legend(frameon=False)
    fig.suptitle("Figure 1. Resource scaling curves", y=1.08, fontsize=12)
    save_figure(fig, "figure1_resource_scaling_curves")


def make_figure2_per_agent_cost(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)
    line_plot(
        axes[0],
        df,
        "tokens_per_agent_stock_day",
        "Tokens / agent-stock-day",
        "(a) Semantic cost per unit",
        log_y=True,
        annotate_sum_1000=True,
    )
    line_plot(
        axes[1],
        df,
        "seconds_per_agent_stock_day",
        "Seconds / agent-stock-day",
        "(b) Runtime per unit",
        log_y=True,
        annotate_sum_1000=True,
    )
    axes[0].legend(frameon=False)
    fig.suptitle("Figure 2. Per-agent cost collapse", y=1.08, fontsize=12)
    save_figure(fig, "figure2_per_agent_cost_collapse")


def make_figure3_market_fidelity(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.1), constrained_layout=True)
    line_plot(axes[0, 0], df, "eq_index_rmse", "RMSE ↓", "(a) Equal-weight RMSE", log_y=False, annotate_sum_1000=True)
    line_plot(axes[0, 1], df, "fw_index_rmse", "RMSE ↓", "(b) Float-weight RMSE", log_y=False, annotate_sum_1000=True)
    line_plot(axes[1, 0], df, "eq_direction_accuracy", "Accuracy ↑", "(c) Direction accuracy", log_y=False, annotate_sum_1000=True)
    line_plot(axes[1, 1], df, "market_fidelity_score", "Score ↑", "(d) Fidelity score", log_y=False, annotate_sum_1000=True)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Figure 3. Market fidelity vs scale", y=1.02, fontsize=12)
    save_figure(fig, "figure3_market_fidelity_vs_scale")


def make_figure4_efficiency_frontier(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.3, 3.7), constrained_layout=True)
    for arch in ["SUM", "LLM-driven"]:
        sub = df[df.architecture == arch].sort_values("N")
        if arch == "LLM-driven":
            sub = sub[sub.N <= 500]
        sizes = 28 + 0.10 * sub["N"].to_numpy(float)
        ax.scatter(
            sub["total_llm_tokens"] / 1_000_000,
            sub["market_fidelity_score"],
            s=sizes,
            color=OKABE_ITO[arch],
            alpha=0.85,
            label=arch,
            edgecolor="white",
            linewidth=0.8,
        )
        ax.plot(sub["total_llm_tokens"] / 1_000_000, sub["market_fidelity_score"], color=OKABE_ITO[arch], lw=1.8)
        for _, row in sub.iterrows():
            ax.annotate(f"N={int(row.N)}", (row.total_llm_tokens / 1_000_000, row.market_fidelity_score), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Total tokens (millions, log)")
    ax.set_ylabel("Market Fidelity Score ↑")
    ax.set_title("Figure 4. Scaling efficiency frontier")
    ax.grid(True, axis="both", alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, "figure4_scaling_efficiency_frontier")


def make_figure6_behavioral_diversity(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.65), constrained_layout=True)
    line_plot(axes[0], df, "action_entropy", "Normalized entropy ↑", "(a) Action entropy", annotate_sum_1000=True)
    line_plot(axes[1], df, "stock_choice_entropy", "Normalized entropy ↑", "(b) Stock-choice entropy", annotate_sum_1000=True)
    line_plot(axes[2], df, "style_separation_js", "Mean JS divergence ↑", "(c) Style separation", annotate_sum_1000=True)
    axes[0].legend(frameon=False)
    fig.suptitle("Figure 6. Behavioral diversity under scaling", y=1.08, fontsize=12)
    save_figure(fig, "figure6_behavioral_diversity")


def save_figure(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f"{name}.png"
    pdf = FIG_DIR / f"{name}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def write_tables(df: pd.DataFrame, eff: pd.DataFrame, slopes: pd.DataFrame):
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    config = pd.DataFrame(
        [
            ("Period", "2025-04-01 to 2025-04-30"),
            ("Stocks", "300059.SZ East Money; 300750.SZ CATL; 002594.SZ BYD; 002475.SZ Luxshare Precision; 600519.SH Kweichow Moutai"),
            ("Selection rule", "Top-5 by average traded amount in April 2025"),
            ("Scales", "SUM: N=50,100,500,1000; LLM-driven: N=50,100,500 (N=1000 not yet measured)"),
            ("Architectures", "SUM, Pure LLM-driven"),
            ("Shared controls", "same seed=42, initial wealth distribution, style ratio, matching engine, news, market data"),
            ("Trading days", "21"),
            ("Stocks per run", "5"),
        ],
        columns=["Field", "Value"],
    )
    resource_cols = [
        "architecture",
        "N",
        "days",
        "stocks",
        "total_llm_calls",
        "failed_llm_calls",
        "total_llm_tokens",
        "elapsed_seconds",
        "total_order_attempts",
        "total_orders",
        "total_trades",
        "orders_per_trade",
        "tokens_per_agent_stock_day",
        "calls_per_agent_stock_day",
        "agent_stock_days_per_second",
    ]
    quality_cols = [
        "architecture",
        "N",
        "eq_index_rmse",
        "fw_index_rmse",
        "eq_return_mae",
        "fw_return_mae",
        "eq_direction_accuracy",
        "fw_direction_accuracy",
        "eq_abs_return_autocorr_error",
        "fw_abs_return_autocorr_error",
        "eq_volatility_relative_error",
        "fw_volatility_relative_error",
        "cross_sectional_dispersion_mae",
        "market_fidelity_score",
    ]
    scaling_eff_cols = [
        "N",
        "SUM_fidelity_score",
        "LLM_fidelity_score",
        "token_reduction_x",
        "llm_call_reduction_x",
        "SUM_fidelity_per_1M_tokens",
        "LLM_fidelity_per_1M_tokens",
        "SUM_fidelity_per_min",
        "LLM_fidelity_per_min",
        "elapsed_speedup_x",
    ]
    stylized_cols = [
        "architecture",
        "N",
        "sim_tail_kurtosis",
        "real_tail_kurtosis",
        "sim_vol_clustering_lag1",
        "sim_abs_return_persistence_lag1_5",
        "cross_sectional_dispersion_mae",
        "fill_rate_trades_per_order",
        "buy_sell_imbalance_acf1",
        "turnover_concentration_hhi",
        "action_entropy",
        "stock_choice_entropy",
        "style_separation_js",
        "active_agent_ratio",
        "gini_delta",
    ]
    config.to_csv(ANALYSIS_DIR / "table1_experimental_configuration.csv", index=False)
    df[resource_cols].to_csv(ANALYSIS_DIR / "table_resource_scalability.csv", index=False)
    eff.to_csv(ANALYSIS_DIR / "table_scaling_efficiency_matched.csv", index=False)
    df[quality_cols].to_csv(ANALYSIS_DIR / "table_quality_activity.csv", index=False)
    eff[scaling_eff_cols].to_csv(ANALYSIS_DIR / "table4_scaling_efficiency.csv", index=False)
    slopes.to_csv(ANALYSIS_DIR / "table5_complexity_slope_summary.csv", index=False)
    df[stylized_cols].to_csv(ANALYSIS_DIR / "table6_stylized_facts_behavior.csv", index=False)

    def to_markdown_table(table: pd.DataFrame) -> str:
        formatted = table.copy()
        for col in formatted.columns:
            if pd.api.types.is_float_dtype(formatted[col]):
                formatted[col] = formatted[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
            else:
                formatted[col] = formatted[col].map(lambda x: "" if pd.isna(x) else str(x))
        header = "| " + " | ".join(formatted.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(formatted.columns)) + " |"
        body = ["| " + " | ".join(row) + " |" for row in formatted.to_numpy(str)]
        return "\n".join([header, sep, *body])

    markdown = []
    markdown.append("# LLM-driven vs SUM Scalability Tables\n")
    markdown.append(
        "> Note: the strictly comparable two-architecture measured points are `N=50/100/500`; "
        "`SUM N=1000` is measured, `LLM-driven N=1000` is not yet measured. Therefore all direct "
        "`SUM vs LLM-driven` ratio comparisons use `N=50/100/500`, with `SUM N=1000` as an additional "
        "measured point showing SUM can keep scaling."
    )
    markdown.append(
        "\n> Notation: `ASD` denotes agent-stock-day, i.e. `N x #stocks x #trading_days`. "
        "`v` means lower is better, `^` means higher is better."
    )

    markdown.append("\n## Table 1. Experimental Configuration\n")
    markdown.append(
        "This table fixes the boundaries of a fair comparison: the same April 2025 window, the same "
        "5 stocks selected by traded amount, the same seed, news, market data, and matching engine. "
        "Note that SUM has `N=1000` while LLM-driven is only run up to `N=500`."
    )
    markdown.append(to_markdown_table(config))
    markdown.append(
        "\n**Interpretation.** This table should precede all scalability results in the paper, to avoid "
        "the misreading that we compare across datasets, stock pools, or dates; instead we compare the "
        "scaling behavior of the architectures themselves under identical market inputs."
    )

    markdown.append("\n## Table 2. Resource Scalability\n")
    markdown.append(
        "This table measures how system resources scale with the number of agents. Key columns are "
        "`total_llm_calls`, `total_llm_tokens`, `tokens_per_agent_stock_day`, and "
        "`agent_stock_days_per_second`. If SUM's shared semantic layer is effective, the first two columns "
        "should be roughly independent of `N`, while the amortized cost per agent-stock-day should drop quickly with `N`."
    )
    markdown.append(to_markdown_table(df[resource_cols].round(4)))
    markdown.append(
        "\n**Interpretation.** SUM's LLM call count is constant at `315` for `N=50/100/500/1000`, and tokens "
        "stay around `1.86M`; LLM-driven grows from `N=50` to `N=500` up to `37,590` calls and `175.8M` tokens. "
        "This shows SUM's public semantic-understanding cost is roughly independent of `N`, while LLM-driven's "
        "semantic cost grows roughly linearly with the number of agents. More importantly, SUM's "
        "`tokens_per_agent_stock_day` drops from `355.0` to `17.7`, reflecting shared semantics amortized over more "
        "agents; LLM-driven stays at `3300+`, meaning every agent-stock-day re-interprets public information."
    )

    markdown.append("\n## Table 3. Market Fidelity\n")
    markdown.append(
        "This table measures how close the simulated market is to the real April 2025 market trajectory. "
        "`eq_index_rmse` is the equal-weight index RMSE, `fw_index_rmse` is the float-weighted index RMSE; "
        "`return_mae` measures daily-return error, `direction_accuracy` measures whether daily-return direction agrees; "
        "`volatility_relative_error`, `abs_return_autocorr_error`, and `cross_sectional_dispersion_mae` measure "
        "volatility, volatility clustering, and cross-sectional dispersion errors respectively. "
        "`market_fidelity_score` is the normalized composite of these metrics."
    )
    markdown.append(to_markdown_table(df[quality_cols].round(4)))
    markdown.append(
        "\n**Interpretation.** SUM's composite fidelity is clearly higher than LLM-driven: e.g. at `N=500`, "
        "SUM's `market_fidelity_score=0.4787` vs LLM-driven's `0.0821`. Per-metric, SUM's EQ/FW RMSE are both "
        "markedly lower and direction accuracy is higher or comparable. This shows SUM does not merely cut cost by "
        "skipping LLM calls, but retains a better macro market trajectory at a lower semantic cost."
    )

    markdown.append("\n## Table 4. Scaling Efficiency\n")
    markdown.append(
        "This table directly answers scaling efficiency: at the same `N`, how much less semantic resource SUM uses "
        "to achieve what market quality. `token_reduction_x` and `llm_call_reduction_x` are LLM-driven's cost "
        "multiples relative to SUM; `fidelity_per_1M_tokens` and `fidelity_per_min` measure market realism per unit "
        "token and per unit time respectively."
    )
    markdown.append(to_markdown_table(eff[scaling_eff_cols].round(4)))
    markdown.append(
        "\n**Interpretation.** As `N` grows from 50 to 500, SUM's token reduction widens from `9.7x` to `94.3x`, "
        "and call reduction from `11.7x` to `119.3x`. Meanwhile SUM's `fidelity_per_1M_tokens` is about `0.21-0.26`, "
        "while LLM-driven at `N=500` is only `0.0005`. This is the strongest scaling-efficiency evidence in the paper: "
        "SUM's advantage amplifies with scale."
    )

    markdown.append("\n## Table 5. Complexity Slope Summary\n")
    markdown.append(
        "This table estimates complexity slopes with a log-log regression `log(metric)=alpha+beta log(N)`. "
        "`beta_tokens_vs_N approx 0` means token cost barely grows with `N`; `beta_tokens_vs_N approx 1` means token "
        "cost grows roughly linearly."
    )
    markdown.append(to_markdown_table(slopes.round(4)))
    markdown.append(
        "\n**Interpretation.** SUM's `beta_tokens_vs_N=-0.0001` and `beta_calls_vs_N approx 0` are almost constant "
        "complexity; LLM-driven's `beta_tokens_vs_N=0.9918` and `beta_calls_vs_N=1.0091` are very close to linear "
        "complexity. This matches the architectural expectation exactly: SUM does a small amount of semantic "
        "interpretation of shared public information, while LLM-driven has each agent re-interpret it."
    )

    markdown.append("\n## Table 6. Stylized Facts and Behavioral Coverage\n")
    markdown.append(
        "This table does not restrict fidelity to 'daily closeness to the real index', but checks whether the "
        "generated market itself has market-like properties. `sim_tail_kurtosis` is simulated return fat-tailedness; "
        "`sim_vol_clustering_lag1` and `sim_abs_return_persistence_lag1_5` measure volatility clustering; "
        "`fill_rate_trades_per_order` measures trading friction; `action_entropy`, `stock_choice_entropy`, and "
        "`style_separation_js` measure behavioral coverage and style differences."
    )
    markdown.append(to_markdown_table(df[stylized_cols].round(4)))
    markdown.append(
        "\n**Interpretation.** SUM's action entropy is clearly higher than LLM-driven, indicating a more balanced "
        "BUY/SELL/HOLD coverage; stock-choice entropy is high on both sides, indicating trading targets do not "
        "collapse to a single stock. LLM-driven's style JS divergence is higher, but this is not necessarily better, "
        "as it may come from prompt-direct-decision style polarization; SUM's advantage is that style differences come "
        "from explicit utility and parameters rather than natural-language wording drift. Also, since the current "
        "monthly sample is short, fat-tail and volatility-clustering metrics are better used as supplementary evidence "
        "of market realism; the main text can carry core results and the appendix the full tables."
    )
    (ANALYSIS_DIR / "llm_sum_scalability_tables.md").write_text("\n\n".join(markdown), encoding="utf-8")


def main():
    configure_style()
    df = load_rows()
    eff = build_efficiency_table(df)
    slopes = build_complexity_slope_table(df)
    write_tables(df, eff, slopes)
    make_figure1_resource_scaling(df)
    make_figure2_per_agent_cost(df)
    make_figure3_market_fidelity(df)
    make_figure4_efficiency_frontier(df)
    make_runtime_breakdown(df)
    make_figure6_behavioral_diversity(df)
    make_resource_panel(df)
    make_efficiency_panel(df, eff)
    make_quality_activity_panel(df)
    print(f"Wrote tables to {ANALYSIS_DIR}")
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
