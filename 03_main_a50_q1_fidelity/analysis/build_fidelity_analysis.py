#!/usr/bin/env python3
"""Rebuild fidelity metrics and figures from the MSCI A50 2025 runs."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
EXP_FIDELITY = WORKSPACE_ROOT / "data" / "fidelity" / "full_year_msci_a50_2025" / "runs"
ASSET_DIR = WORKSPACE_ROOT / "outputs" / "fidelity" / "assets"

CANONICAL_SUM = (
    "sum_msci_a50_2025_full_N1000_seed42_full_decisions_20260502_120345"
)
RB_WEEK = "rb_msci_a50_2025_week1_N1000_seed42_20260526_150632"

H1_END = "20250630"
STYLE_ORDER = ["value", "momentum", "noise", "institution", "liquidity_provider"]


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def role_from_agent_id(agent_id: str) -> str:
    if agent_id.startswith("R_V_"):
        return "value"
    if agent_id.startswith("R_M_"):
        return "momentum"
    if agent_id.startswith("R_N_"):
        return "noise"
    if agent_id.startswith("I_") or agent_id.startswith("INST"):
        return "institution"
    if agent_id.upper().startswith("VMM") or agent_id.startswith("LP"):
        return "liquidity_provider"
    return "other"


def slice_index_df(df: pd.DataFrame, end_date: Optional[str] = None) -> pd.DataFrame:
    out = df[df["date"] != "init"].copy()
    if end_date:
        out = out[out["date"] <= end_date]
    return out.reset_index(drop=True)


def index_metrics(df: pd.DataFrame) -> Dict[str, float]:
    sim = df["equal_weight_sim"].astype(float).values
    real = df["equal_weight_real"].astype(float).values
    base = sim[0]
    sim_n = sim / base * 1000.0
    real_n = real / real[0] * 1000.0
    sim_ret = np.diff(sim) / sim[:-1]
    real_ret = np.diff(real) / real[:-1]
    abs_sim = np.abs(sim_ret)
    abs_real = np.abs(real_ret)
    lag1_sim = float(np.corrcoef(abs_sim[:-1], abs_sim[1:])[0, 1]) if len(abs_sim) > 2 else float("nan")
    lag1_real = float(np.corrcoef(abs_real[:-1], abs_real[1:])[0, 1]) if len(abs_real) > 2 else float("nan")
    rmse = float(np.sqrt(np.mean((sim_n - real_n) ** 2)))
    mape = float(np.mean(np.abs(sim_n - real_n) / np.maximum(real_n, 1e-9)) * 100.0)
    return {
        "trading_days": int(len(df)),
        "sum_return_pct": float((sim[-1] / sim[0] - 1.0) * 100.0),
        "real_return_pct": float((real[-1] / real[0] - 1.0) * 100.0),
        "ew_rmse": rmse,
        "ew_mape_pct": mape,
        "daily_return_corr": float(np.corrcoef(sim_ret, real_ret)[0, 1]) if len(sim_ret) > 1 else float("nan"),
        "absret_acf_lag1_sum": lag1_sim,
        "absret_acf_lag1_real": lag1_real,
        "fw_rmse": float(
            np.sqrt(
                np.mean(
                    (
                        df["float_weight_sim"].astype(float) / df["float_weight_sim"].astype(float).iloc[0] * 1000.0
                        - df["float_weight_real"].astype(float) / df["float_weight_real"].astype(float).iloc[0] * 1000.0
                    )
                    ** 2
                )
            )
        ),
    }


def wealth_metrics(run_dir: Path) -> Dict:
    summary = load_json(run_dir / "wealth_summary.json")
    by_type = summary.get("by_type", {})
    shares = {}
    returns = {}
    for style, block in by_type.items():
        init_w = float(block.get("initial_total", 0.0))
        final_w = float(block.get("final_total", 0.0))
        if init_w <= 0:
            continue
        shares[style] = final_w
        returns[style] = (final_w / init_w - 1.0) * 100.0
    total = sum(shares.values()) or 1.0
    share_pct = {k: v / total for k, v in shares.items()}
    gini_hist = load_json(run_dir / "gini_history.json")
    gini_start = float(gini_hist[0]["gini"]) if gini_hist else float("nan")
    gini_end = float(gini_hist[-1]["gini"]) if gini_hist else float(summary.get("gini_final", float("nan")))
    return {
        "gini_start": gini_start,
        "gini_end": gini_end,
        "share_end": share_pct,
        "mean_return_pct": returns,
    }


def counterparty_share(trades: List[dict]) -> Dict[str, Dict[str, float]]:
    matrix = defaultdict(lambda: defaultdict(float))
    for tr in trades:
        val = float(tr.get("value", 0.0))
        buyer = role_from_agent_id(str(tr.get("buyer", "")))
        seller = role_from_agent_id(str(tr.get("seller", "")))
        matrix[buyer][seller] += val
        matrix[seller][buyer] += val
    out = {}
    for focal, cps in matrix.items():
        total = sum(cps.values()) or 1.0
        out[focal] = {k: v / total * 100.0 for k, v in cps.items()}
    return out


def style_activity(trades: List[dict], n_days: int, agent_counts: Dict[str, int]) -> Dict[str, Dict[str, float]]:
    traded_value = defaultdict(float)
    active_days = defaultdict(set)
    for tr in trades:
        val = float(tr.get("value", 0.0))
        date = str(tr.get("date", ""))
        for side in ("buyer", "seller"):
            aid = str(tr.get(side, ""))
            role = role_from_agent_id(aid)
            if role not in {"value", "momentum", "noise", "institution"}:
                continue
            traded_value[role] += val
            active_days[role].add((aid, date))
    return {
        role: {
            "turnover_proxy": traded_value[role] / max(agent_counts.get(role, 1), 1) / 1e6,
            "active_ratio": len({a for a, _ in active_days[role]}) / max(agent_counts.get(role, 1) * n_days, 1),
            "unique_active_agents": len({a for a, _ in active_days[role]}),
        }
        for role in ["value", "momentum", "noise", "institution"]
    }


def agent_counts_from_wealth(run_dir: Path) -> Dict[str, int]:
    summary = load_json(run_dir / "wealth_summary.json")
    return {k: int(v.get("n", 0)) for k, v in summary.get("by_type", {}).items()}


def micro_audit(run_dir: Path) -> Dict:
    """First sigma / gamma_t per agent from sharded jsonl + terminal wealth."""
    wealth = load_json(run_dir / "wealth_summary.json")
    agent_wealth = {}
    for style, block in wealth.get("by_type", {}).items():
        if style not in {"value", "momentum", "noise"}:
            continue
        for a in block.get("agents", []):
            init_w = float(a["initial"])
            final_w = float(a["final"])
            agent_wealth[a["id"]] = {
                "style": style,
                "return_pct": (final_w / init_w - 1.0) * 100.0 if init_w > 0 else 0.0,
            }

    first_params: Dict[str, Tuple[float, float]] = {}
    parts = sorted(run_dir.glob("retail_decisions.jsonl.part*"))
    if not parts:
        single = run_dir / "retail_decisions.jsonl"
        parts = [single] if single.exists() else []
    for part in parts:
        with part.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                aid = row["agent_id"]
                if aid in first_params:
                    continue
                first_params[aid] = (float(row.get("sigma", 0.0)), float(row.get("gamma_t", 0.0)))
                if len(first_params) >= len(agent_wealth):
                    break
        if len(first_params) >= len(agent_wealth):
            break

    sigmas, gammas, rets = [], [], []
    for aid, (sigma, gamma) in first_params.items():
        if aid not in agent_wealth:
            continue
        sigmas.append(sigma)
        gammas.append(gamma)
        rets.append(agent_wealth[aid]["return_pct"])

    sigmas = np.array(sigmas)
    gammas = np.array(gammas)
    rets = np.array(rets)

    def corr(x, y):
        if len(x) < 3:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    def bin_stats(x, y, label):
        if len(x) == 0:
            return []
        qs = np.quantile(x, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        bins = []
        for i in range(5):
            lo, hi = qs[i], qs[i + 1]
            mask = (x >= lo) & (x <= hi if i == 4 else x < hi)
            if mask.sum() == 0:
                continue
            bins.append(
                {
                    f"{label}_q": i,
                    "x": float(np.mean(x[mask])),
                    "y": float(np.mean(y[mask])),
                    "n": int(mask.sum()),
                }
            )
        return bins

    return {
        "n_agents": len(rets),
        "mean_return_pct": float(np.mean(rets)) if len(rets) else float("nan"),
        "sigma_return_corr": corr(sigmas, rets),
        "gamma_return_corr": corr(gammas, rets),
        "sigma_bins": bin_stats(sigmas, rets / 100.0, "sigma"),
        "gamma_bins": bin_stats(gammas, rets / 100.0, "gamma"),
    }


def plot_index_panel(df_h1: pd.DataFrame, df_full: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    x = np.arange(len(df_h1))
    axes[0].plot(x, df_h1["equal_weight_sim"] / df_h1["equal_weight_sim"].iloc[0] * 1000, label="SUM sim", lw=2)
    axes[0].plot(x, df_h1["equal_weight_real"] / df_h1["equal_weight_real"].iloc[0] * 1000, label="Real", lw=2)
    axes[0].set_title("(a) EW index — 2025 H1")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    sim_ret = np.diff(df_h1["equal_weight_sim"].astype(float)) / df_h1["equal_weight_sim"].astype(float).iloc[:-1]
    abs_r = np.abs(sim_ret)
    lags = range(1, 6)
    acfs = [float(np.corrcoef(abs_r[:-l], abs_r[l:])[0, 1]) if len(abs_r) > l + 1 else 0 for l in lags]
    axes[1].bar(list(lags), acfs, color="#1f77b4")
    axes[1].set_title("(b) |return| ACF — H1")
    axes[1].set_xlabel("Lag")

    df_full = df_full.copy()
    df_full["sim_ret"] = df_full["equal_weight_sim"].pct_change()
    vol = df_full["equal_weight_sim"].pct_change().rolling(5).std()
    axes[2].scatter(vol, df_full["sim_ret"].abs(), alpha=0.35, s=12)
    axes[2].set_title("(c) Vol vs |return| — full year")
    axes[2].set_xlabel("5d rolling vol")
    axes[2].set_ylabel("|daily return|")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_wealth_flow(wealth: Dict, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    styles = [s for s in STYLE_ORDER if s in wealth["mean_return_pct"]]
    rets = [wealth["mean_return_pct"][s] for s in styles]
    axes[1].bar(styles, rets, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"][: len(styles)])
    axes[1].set_title("Cumulative return by type (%)")
    axes[1].tick_params(axis="x", rotation=20)

    shares = wealth["share_end"]
    axes[0].bar(list(shares.keys()), [shares[k] * 100 for k in shares], color="#4c78a8")
    axes[0].set_title("Terminal wealth share (%)")
    axes[0].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_micro(micro: Dict, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in [
        (axes[0], "sigma_bins", "Return vs σ"),
        (axes[1], "gamma_bins", "Return vs γ₀"),
    ]:
        bins = micro.get(key, [])
        if bins:
            ax.errorbar(
                [b["x"] for b in bins],
                [b["y"] * 100 for b in bins],
                fmt="o-",
                capsize=3,
            )
        ax.set_title(title)
        ax.set_ylabel("Mean return (%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def analyze_run(run_dir: Path, window: str = "h1") -> Dict:
    idx = pd.read_csv(run_dir / "index_series.csv")
    idx_slice = slice_index_df(idx, H1_END if window == "h1" else None)
    trades = load_json(run_dir / "trades.json")
    trades_slice = [t for t in trades if str(t.get("date", "")) <= H1_END] if window == "h1" else trades
    counts = agent_counts_from_wealth(run_dir)
    return {
        "run_dir": str(run_dir),
        "window": window,
        "index": index_metrics(idx_slice),
        "wealth": wealth_metrics(run_dir),
        "counterparty_share_pct": counterparty_share(trades_slice),
        "style_activity": style_activity(trades_slice, len({t["date"] for t in trades_slice}), counts),
    }


def main():
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    sum_dir = EXP_FIDELITY / CANONICAL_SUM
    rb_dir = EXP_FIDELITY / RB_WEEK

    if not sum_dir.exists():
        raise SystemExit(f"Missing canonical SUM run: {sum_dir}")

    idx_full = pd.read_csv(sum_dir / "index_series.csv")
    idx_h1 = slice_index_df(idx_full, H1_END)

    summary = {
        "source_runs": {
            "sum_canonical": CANONICAL_SUM,
            "rb_week": RB_WEEK,
        },
        "note": "Generated from new MSCI A50 2025 pipeline only; legacy/full_year_old excluded.",
        "trading_days_h1": int(len(idx_h1)),
        "trading_days_full": int(len(slice_index_df(idx_full))),
        "h1": analyze_run(sum_dir, "h1"),
        "full_year": analyze_run(sum_dir, "full"),
    }
    summary["h1"]["micro_audit"] = micro_audit(sum_dir)

    if rb_dir.exists():
        rb_h1 = analyze_run(rb_dir, "h1")
        summary["rb_week_h1"] = rb_h1

    metrics_path = ASSET_DIR / "fidelity_metrics_summary.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    plot_index_panel(idx_h1, slice_index_df(idx_full), ASSET_DIR / "fidelity_emergent_three_panel.png")
    plot_wealth_flow(summary["h1"]["wealth"], ASSET_DIR / "fidelity_meso_wealth_flow.png")
    plot_micro(summary["h1"]["micro_audit"], ASSET_DIR / "fidelity_micro_controls.png")

    print(f"Wrote {metrics_path}")
    print(f"Figures -> {ASSET_DIR}")
    h1 = summary["h1"]["index"]
    print(
        f"H1 EW return sim={h1['sum_return_pct']:.2f}% real={h1['real_return_pct']:.2f}% "
        f"RMSE={h1['ew_rmse']:.2f} MAPE={h1['ew_mape_pct']:.2f}% lag1ACF={h1['absret_acf_lag1_sum']:.3f}"
    )


if __name__ == "__main__":
    main()
