#!/usr/bin/env python3
"""Build figures, tables, and markdown for heterogeneity ablation analysis."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]  # repo root
ROOT = WORKSPACE_ROOT / "data" / "ablation"
ANALYSIS = WORKSPACE_ROOT / "outputs" / "ablation" / "heterogeneity_ablation_202506_top5_N1000" / "analysis"
FIG_DIR = ANALYSIS / "figures"
TABLE_DIR = ANALYSIS / "tables"
MD_PATH = ROOT / "ablation_analysis.md"

RUNS = [
    ("baseline_sum_N1000", "Baseline SUM", "Baseline"),
    ("macro_sum_institution", "Macro ablation", "Macro"),
    ("meso_noise_channel", "Meso ablation", "Meso"),
    ("micro_default_gamma_sigma", "Micro ablation", "Micro"),
]

COLORS = {
    "Baseline": "#0072B2",
    "Macro": "#E69F00",
    "Meso": "#009E73",
    "Micro": "#D55E00",
    "Real": "#111827",
    "grid": "#D1D5DB",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    summary_rows = []
    index_data: dict[str, pd.DataFrame] = {}
    stock_data: dict[str, pd.DataFrame] = {}
    step_data: dict[str, pd.DataFrame] = {}

    for folder, label, short in RUNS:
        run_dir = ROOT / folder
        meta = json.load(open(run_dir / "run_metadata.json", encoding="utf-8"))
        idx = pd.read_csv(run_dir / "index_series.csv", encoding="utf-8-sig")
        stocks = pd.read_csv(run_dir / "stock_price_changes.csv", encoding="utf-8-sig")
        steps = pd.read_json(run_dir / "step_metrics.json")

        idx_no_init = idx[idx["date"].astype(str) != "init"]

        def rmse(a: str, b: str) -> float:
            return float(np.sqrt(np.mean((idx_no_init[a].to_numpy(float) - idx_no_init[b].to_numpy(float)) ** 2)))

        total_attempts = float(meta.get("total_order_attempts") or meta.get("total_orders") or 0)
        total_trades = float(meta.get("total_trades") or 0)
        summary_rows.append(
            {
                "folder": folder,
                "experiment": label,
                "short": short,
                "daily_records": meta.get("daily_records"),
                "n_retail": meta.get("n_retail"),
                "n_inst": meta.get("n_inst"),
                "total_orders": meta.get("total_orders"),
                "total_order_attempts": meta.get("total_order_attempts"),
                "total_trades": meta.get("total_trades"),
                "trade_per_attempt": total_trades / total_attempts if total_attempts > 0 else np.nan,
                "initial_gini": meta.get("initial_gini"),
                "final_gini": meta.get("final_gini"),
                "delta_gini": (meta.get("final_gini") or 0) - (meta.get("initial_gini") or 0),
                "elapsed_seconds": meta.get("elapsed_seconds"),
                "total_llm_tokens": meta.get("total_llm_tokens"),
                "total_llm_calls": meta.get("total_llm_calls"),
                "semantic_cache_hits": meta.get("semantic_cache_hits", ""),
                "institution_cache_hits": meta.get("institution_cache_hits", ""),
                "eq_rmse": rmse("equal_weight_sim", "equal_weight_real"),
                "float_rmse": rmse("float_weight_sim", "float_weight_real"),
                "eq_final_gap": float(idx["equal_weight_sim"].iloc[-1] - idx["equal_weight_real"].iloc[-1]),
                "float_final_gap": float(idx["float_weight_sim"].iloc[-1] - idx["float_weight_real"].iloc[-1]),
            }
        )
        index_data[short] = idx
        stocks["experiment"] = short
        stocks["return_gap_pp"] = 100.0 * stocks["return_gap"]
        stock_data[short] = stocks
        step_data[short] = steps

    summary = pd.DataFrame(summary_rows)
    baseline = summary[summary["short"] == "Baseline"].iloc[0]
    for col in ["eq_rmse", "float_rmse", "total_orders", "total_trades", "trade_per_attempt"]:
        summary[f"{col}_delta_vs_baseline"] = summary[col] - baseline[col]
    return summary, index_data, stock_data, step_data


def save_tables(summary: pd.DataFrame, stock_data: dict[str, pd.DataFrame]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary_out = summary[
        [
            "experiment",
            "daily_records",
            "total_orders",
            "total_trades",
            "trade_per_attempt",
            "eq_rmse",
            "float_rmse",
            "eq_final_gap",
            "float_final_gap",
            "final_gini",
        ]
    ].copy()
    summary_out["trade_per_attempt"] *= 100.0
    summary_out.to_csv(TABLE_DIR / "table_ablation_main_metrics.csv", index=False, encoding="utf-8-sig")

    delta_out = summary[
        [
            "experiment",
            "eq_rmse_delta_vs_baseline",
            "float_rmse_delta_vs_baseline",
            "total_orders_delta_vs_baseline",
            "total_trades_delta_vs_baseline",
            "trade_per_attempt_delta_vs_baseline",
        ]
    ].copy()
    delta_out["trade_per_attempt_delta_vs_baseline"] *= 100.0
    delta_out.to_csv(TABLE_DIR / "table_ablation_delta_vs_baseline.csv", index=False, encoding="utf-8-sig")

    stocks = pd.concat(stock_data.values(), ignore_index=True)
    pivot = stocks.pivot(index="stock", columns="experiment", values="return_gap_pp")
    pivot = pivot[["Baseline", "Macro", "Meso", "Micro"]]
    pivot.to_csv(TABLE_DIR / "table_stock_return_gap_pp.csv", encoding="utf-8-sig")


def format_dates(dates: pd.Series) -> list[str]:
    labels = []
    for value in dates.astype(str):
        if value == "init":
            labels.append("Init")
        else:
            labels.append(f"{value[4:6]}-{value[6:8]}")
    return labels


def make_index_paths(index_data: dict[str, pd.DataFrame]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    base = index_data["Baseline"]
    x = np.arange(len(base))
    labels = format_dates(base["date"])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), constrained_layout=True)
    panels = [
        ("equal_weight", "Equal-weight index", "equal_weight_sim", "equal_weight_real"),
        ("float_weight", "Float-market-cap-weighted index", "float_weight_sim", "float_weight_real"),
    ]
    for ax, (_, title, sim_col, real_col) in zip(axes, panels):
        ax.plot(x, base[real_col], color=COLORS["Real"], lw=2.4, linestyle="--", label="Real market")
        for short, idx in index_data.items():
            ax.plot(x, idx[sim_col], color=COLORS[short], lw=1.9, marker="o", markersize=3.2, label=short)
        ax.set_title(title)
        ax.set_xticks(x[::2])
        ax.set_xticklabels(labels[::2], rotation=45, ha="right")
        ax.set_ylabel("Index value (base=1000)")
        ax.grid(True, axis="y", color=COLORS["grid"], alpha=0.6)
    axes[0].legend(frameon=False, ncol=2, loc="lower left")
    fig.suptitle("Index paths under three-level heterogeneity ablations", y=1.04, fontsize=11)
    fig.savefig(FIG_DIR / "fig_ablation_index_paths.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_ablation_index_paths.pdf", bbox_inches="tight")
    plt.close(fig)


def make_metric_bars(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.4), constrained_layout=True)
    order = ["Baseline", "Macro", "Meso", "Micro"]
    sub = summary.set_index("short").loc[order].reset_index()
    x = np.arange(len(sub))
    colors = [COLORS[s] for s in sub["short"]]

    ax = axes[0, 0]
    width = 0.36
    ax.bar(x - width / 2, sub["eq_rmse"], width, label="Equal-weight", color="#56B4E9")
    ax.bar(x + width / 2, sub["float_rmse"], width, label="Float-weighted", color="#E69F00")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["short"])
    ax.set_ylabel("RMSE")
    ax.set_title("Index tracking error")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", color=COLORS["grid"], alpha=0.55)

    ax = axes[0, 1]
    ax.bar(x - width / 2, sub["total_orders"], width, label="Orders", color="#0072B2")
    ax.bar(x + width / 2, sub["total_trades"], width, label="Trades", color="#D55E00")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["short"])
    ax.set_ylabel("Count")
    ax.set_title("Market activity")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", color=COLORS["grid"], alpha=0.55)

    ax = axes[1, 0]
    ax.bar(x, 100.0 * sub["trade_per_attempt"], color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["short"])
    ax.set_ylabel("Trades / order attempts (%)")
    ax.set_title("Matching efficiency")
    ax.grid(True, axis="y", color=COLORS["grid"], alpha=0.55)

    ax = axes[1, 1]
    ax.bar(x - width / 2, sub["eq_final_gap"], width, label="Equal-weight", color="#56B4E9")
    ax.bar(x + width / 2, sub["float_final_gap"], width, label="Float-weighted", color="#E69F00")
    ax.axhline(0, color="#111827", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["short"])
    ax.set_ylabel("Simulated minus real index")
    ax.set_title("Terminal index gap")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", color=COLORS["grid"], alpha=0.55)

    fig.suptitle("Aggregate ablation metrics", y=1.03, fontsize=11)
    fig.savefig(FIG_DIR / "fig_ablation_metric_bars.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_ablation_metric_bars.pdf", bbox_inches="tight")
    plt.close(fig)


def make_stock_heatmap(stock_data: dict[str, pd.DataFrame]) -> None:
    stocks = pd.concat(stock_data.values(), ignore_index=True)
    pivot = stocks.pivot(index="stock", columns="experiment", values="return_gap_pp")
    pivot = pivot[["Baseline", "Macro", "Meso", "Micro"]]
    fig, ax = plt.subplots(figsize=(6.8, 3.2), constrained_layout=True)
    vmax = max(1.0, float(np.nanmax(np.abs(pivot.to_numpy()))))
    im = ax.imshow(pivot.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Stock-level return gap (real minus simulated, pp)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=8, color="#111827")
    cbar = fig.colorbar(im, ax=ax, shrink=0.88)
    cbar.set_label("Return gap (percentage points)")
    fig.savefig(FIG_DIR / "fig_ablation_stock_return_gap_heatmap.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_ablation_stock_return_gap_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def make_daily_activity(step_data: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), constrained_layout=True)
    for short, steps in step_data.items():
        x = np.arange(1, len(steps) + 1)
        axes[0].plot(x, steps["submitted_orders"], lw=1.8, marker="o", markersize=3, color=COLORS[short], label=short)
        axes[1].plot(x, steps["trades"], lw=1.8, marker="o", markersize=3, color=COLORS[short], label=short)
    axes[0].set_title("Daily submitted orders")
    axes[1].set_title("Daily matched trades")
    for ax in axes:
        ax.set_xlabel("Trading day")
        ax.grid(True, axis="y", color=COLORS["grid"], alpha=0.6)
    axes[0].set_ylabel("Orders")
    axes[1].set_ylabel("Trades")
    axes[0].legend(frameon=False, ncol=2)
    fig.suptitle("Daily market activity across ablations", y=1.04, fontsize=11)
    fig.savefig(FIG_DIR / "fig_ablation_daily_activity.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_ablation_daily_activity.pdf", bbox_inches="tight")
    plt.close(fig)


def md_table(df: pd.DataFrame, float_cols: dict[str, int] | None = None) -> str:
    float_cols = float_cols or {}
    out = df.copy()
    for col, nd in float_cols.items():
        if col in out:
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{v:.{nd}f}")
    for col in out.columns:
        if col not in float_cols:
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else str(v))
    header = "| " + " | ".join(out.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(out.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in out.to_numpy(str)]
    return "\n".join([header, sep, *rows])


def write_markdown(summary: pd.DataFrame, stock_data: dict[str, pd.DataFrame]) -> None:
    """Write the ablation report in English with English chart/table labels."""
    order = ["Baseline", "Macro", "Meso", "Micro"]
    sub = summary.set_index("short").loc[order].reset_index()
    baseline = sub[sub["short"] == "Baseline"].iloc[0]
    macro = sub[sub["short"] == "Macro"].iloc[0]
    meso = sub[sub["short"] == "Meso"].iloc[0]
    micro = sub[sub["short"] == "Micro"].iloc[0]

    main_table = sub[
        [
            "short",
            "total_orders",
            "total_trades",
            "trade_per_attempt",
            "eq_rmse",
            "float_rmse",
            "eq_final_gap",
            "float_final_gap",
            "final_gini",
        ]
    ].rename(
        columns={
            "short": "Experiment",
            "total_orders": "Orders",
            "total_trades": "Trades",
            "trade_per_attempt": "Trade/Attempt",
            "eq_rmse": "EW RMSE",
            "float_rmse": "FW RMSE",
            "eq_final_gap": "EW Final Gap",
            "float_final_gap": "FW Final Gap",
            "final_gini": "Final Gini",
        }
    )
    main_table["Trade/Attempt"] *= 100.0

    delta = sub[
        [
            "short",
            "eq_rmse_delta_vs_baseline",
            "float_rmse_delta_vs_baseline",
            "total_orders_delta_vs_baseline",
            "total_trades_delta_vs_baseline",
            "trade_per_attempt_delta_vs_baseline",
        ]
    ].rename(
        columns={
            "short": "Experiment",
            "eq_rmse_delta_vs_baseline": "Delta EW RMSE",
            "float_rmse_delta_vs_baseline": "Delta FW RMSE",
            "total_orders_delta_vs_baseline": "Delta Orders",
            "total_trades_delta_vs_baseline": "Delta Trades",
            "trade_per_attempt_delta_vs_baseline": "Delta Trade/Attempt",
        }
    )
    delta["Delta Trade/Attempt"] *= 100.0

    stock_pivot = pd.concat(stock_data.values(), ignore_index=True).pivot(index="stock", columns="experiment", values="return_gap_pp")
    stock_pivot = stock_pivot[order].reset_index().rename(columns={"stock": "Stock"})

    stock_names = {
        "300059.SZ": "East Money",
        "300308.SZ": "Zhongji Innolight",
        "300502.SZ": "Eoptolink",
        "600519.SH": "Kweichow Moutai",
        "002594.SZ": "BYD",
    }
    stock_pivot["Stock"] = stock_pivot["Stock"].map(lambda s: f"{s} ({stock_names.get(s, s)})")

    md = f"""# Three-Level Heterogeneity Ablation Analysis, 2025-06

## Experimental Setup

This report presents the three-level heterogeneity ablation results for June 2025 and compares them with the standard SUM N=1000 baseline. The stock pool is the five MSCI A50 stocks with the highest average daily traded amount over the full month of 2025-06: East Money `300059.SZ`, Zhongji Innolight `300308.SZ`, Eoptolink `300502.SZ`, Kweichow Moutai `600519.SH`, and BYD `002594.SZ`. All four settings cover the 20 trading days from 2025-06-03 to 2025-06-30, with N=1000 retail agents, seed=42, and `mock_llm=False`.

The three ablations correspond to the three heterogeneity levels. The Macro ablation replaces the institution side with a homogeneous SUM-retail architecture and splits the institution capital into ten sleeves at value:momentum:noise = 3:3:4 that are settled uniformly. The Meso ablation keeps the number and labels of retail types but routes all SUM retail agents through the noise channel. The Micro ablation keeps the types and channels but fixes every SUM retail agent's `gamma` and `sigma` to the distribution mean, i.e. `gamma=0.08`, `sigma=0.0175`.

To keep the ablations as controlled-variable experiments, all three reuse the baseline's shared semantic cache; meso and micro also reuse the baseline's institution CIO decision cache. Therefore this report compares the re-matched orders, trades, price paths, wealth distribution, and per-stock return fit, and does not use additional tokens as an efficiency metric.

## Aggregate Results

**Table 1. Aggregate ablation metrics**

{md_table(main_table, {'Trade/Attempt': 2, 'EW RMSE': 2, 'FW RMSE': 2, 'EW Final Gap': 2, 'FW Final Gap': 2, 'Final Gini': 4})}

**Table 2. Metric changes relative to Baseline SUM**

{md_table(delta, {'Delta EW RMSE': 2, 'Delta FW RMSE': 2, 'Delta Orders': 0, 'Delta Trades': 0, 'Delta Trade/Attempt': 2})}

On index fit, the 2025-06 results show a clear multi-metric trade-off rather than one setting dominating on all metrics. The baseline EW RMSE is {baseline['eq_rmse']:.2f} and FW RMSE is {baseline['float_rmse']:.2f}. The Macro ablation lowers EW RMSE to {macro['eq_rmse']:.2f} but raises FW RMSE to {macro['float_rmse']:.2f}, showing that making the institution SUM-like moves the equal-weight path closer to the real index while worsening the bias on high-weight assets. The Meso ablation has FW RMSE {meso['float_rmse']:.2f}, slightly below the baseline, but EW RMSE rises to {meso['eq_rmse']:.2f} and the trade structure clearly deteriorates. The Micro ablation has EW RMSE {micro['eq_rmse']:.2f} and FW RMSE {micro['float_rmse']:.2f}, both worse than the baseline, indicating that individual response-parameter heterogeneity contributes stably to index fit.

![Index paths under three-level heterogeneity ablations](analysis/figures/fig_ablation_index_paths.png)

Figure 1 compares the equal-weight and float-weighted index paths. In the real June 2025 market, AI optical-module stocks such as Zhongji Innolight and Eoptolink rose strongly, while Kweichow Moutai and BYD fell, producing clear structural divergence. All four simulations produce an upward direction but underestimate the real equal-weight index's late-month rise. The Meso ablation's simulated path is smoother and more homogeneous, indicating that after unifying the noise channel, agents insufficiently follow strong-trend stocks while concentrating pressure on high-weight stocks.

![Aggregate ablation metrics](analysis/figures/fig_ablation_metric_bars.png)

Figure 2 summarizes index error, market activity, matching efficiency, and terminal index gap. The Macro ablation's order count is close to the baseline, with {int(macro['total_trades']):,} trades, slightly below the baseline's {int(baseline['total_trades']):,}, showing that splitting the institution into SUM-style sleeves does not significantly amplify activity and mainly affects local trade structure. The Meso ablation is the most distinctive: order count rises to {int(meso['total_orders']):,}, {int(meso['total_orders'] - baseline['total_orders']):,} more than the baseline, but trade count drops to {int(meso['total_trades']):,}. This indicates channel homogenization produces more same-direction orders rather than more effective counterparties. The Micro ablation's order and trade counts are close to the baseline, indicating micro parameter heterogeneity acts more like a mechanism that modulates response magnitude and fill probability than one that changes overall trading intent.

## Daily Activity

![Daily market activity across ablations](analysis/figures/fig_ablation_daily_activity.png)

Figure 3 shows per-day orders and trades. The Meso ablation submits more orders on most trading days but matched trades are clearly lower, with same-direction crowding more prominent in the mid-to-late days. This is important evidence for the three-level design: the value/momentum/noise semantic channels not only change the buy/sell direction but also determine whether enough two-sided quotes form. Macro and Micro are closer to the baseline, indicating that in this month's five-stock setting the main index-level sensitivity comes from the meso level.

## Per-Stock Error

The per-stock return gap is defined as `real return - simulated return` in percentage points. Positive means the simulation under-rose or over-fell; negative means the simulation over-rose or under-fell.

**Table 3. Stock-level return gap (real minus simulated, percentage points)**

{md_table(stock_pivot, {'Baseline': 2, 'Macro': 2, 'Meso': 2, 'Micro': 2})}

![Stock-level return gap heatmap](analysis/figures/fig_ablation_stock_return_gap_heatmap.png)

Figure 4 shows that all settings clearly underestimate the real rise of Zhongji Innolight `300308.SZ` and Eoptolink `300502.SZ`, the hardest structural moves to fit in the 2025-06 pool. The baseline return gap for Zhongji Innolight is {stock_pivot.loc[stock_pivot['Stock'].str.startswith('300308'), 'Baseline'].iloc[0]:.2f} pp and for Eoptolink {stock_pivot.loc[stock_pivot['Stock'].str.startswith('300502'), 'Baseline'].iloc[0]:.2f} pp. The Meso ablation further widens the Zhongji Innolight error, indicating that unifying the noise channel weakens the ability to follow trend moves.

On the other hand, Kweichow Moutai `600519.SH` and BYD `002594.SZ` fell in the real market but the simulated prices generally rose, so their gaps are negative. Under the Macro ablation, the Kweichow Moutai error widens to {stock_pivot.loc[stock_pivot['Stock'].str.startswith('600519'), 'Macro'].iloc[0]:.2f} pp, indicating that making the institution SUM-like strengthens buy pressure on some high-weight stocks. Under the Micro ablation, the BYD error is {stock_pivot.loc[stock_pivot['Stock'].str.startswith('002594'), 'Micro'].iloc[0]:.2f} pp, showing that fixing response parameters reduces agents' dispersed reaction to negative divergence.

## Conclusion

This 2025-06 ablation supports the necessity of the three-level heterogeneity design, but the evidence appears as a trade-off between fit metrics and market mechanisms. The Macro ablation lowers EW RMSE but significantly raises FW RMSE, showing the institution level reallocates price errors between low- and high-weight stocks. The Meso ablation slightly improves FW RMSE but sharply increases orders and reduces trades, showing that semantic-channel homogenization damages market microstructure. The Micro ablation is worse than the baseline on both index RMSEs, showing individual response-parameter heterogeneity directly contributes to stable fit.

Therefore, the three levels of heterogeneity are not merely added agent diversity but carry different market-generating mechanisms: the macro level controls how large capital is expressed, the meso level controls the interpretive division of public information, and the micro level controls the distribution of response magnitude under the same interpretation. Together they let the SUM market keep a relatively better balance among index path, trade activity, and per-stock error.
"""
    MD_PATH.write_text(md, encoding="utf-8")


def main() -> None:
    configure_style()
    summary, index_data, stock_data, step_data = load_data()
    save_tables(summary, stock_data)
    make_index_paths(index_data)
    make_metric_bars(summary)
    make_stock_heatmap(stock_data)
    make_daily_activity(step_data)
    write_markdown(summary, stock_data)
    print(f"Wrote {MD_PATH}")
    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote tables to {TABLE_DIR}")


if __name__ == "__main__":
    main()
