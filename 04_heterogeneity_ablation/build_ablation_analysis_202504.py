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
ANALYSIS = WORKSPACE_ROOT / "outputs" / "ablation" / "heterogeneity_ablation_202504_top5_N1000" / "analysis"
FIG_DIR = ROOT / "assets"
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
    order = ["Baseline", "Macro", "Meso", "Micro"]
    sub = summary.set_index("short").loc[order].reset_index()
    baseline = sub[sub["short"] == "Baseline"].iloc[0]

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

    macro = sub[sub["short"] == "Macro"].iloc[0]
    meso = sub[sub["short"] == "Meso"].iloc[0]
    micro = sub[sub["short"] == "Micro"].iloc[0]

    md = f"""# Three-Level Heterogeneity Ablation Analysis

## Experimental Setup

This section compares the standard SUM N=1000 experiment with three heterogeneity ablations. All four settings use the five MSCI A50 stocks with the highest average daily traded amount in April 2025, over 21 trading days, with 1000 retail agents and the same initial random seed and stock pool. The standard experiment comes from the scalability `sum_N1000`, copied to `baseline_sum_N1000`; the three ablations correspond to the macro, meso, and micro levels.

The three ablations are defined as follows. The Macro ablation replaces the institution-specific architecture with a homogeneous SUM-retail architecture and splits the institution capital into ten sleeves at value:momentum:noise = 3:3:4 that are settled uniformly. The Meso ablation keeps retail type labels and individual state but routes all SUM retail agents through the noise semantic channel. The Micro ablation keeps types and semantic channels but fixes every SUM retail agent's `gamma` and `sigma` to the distribution mean, i.e. `gamma=0.08`, `sigma=0.0175`.

To make each ablation a controlled-variable experiment that only removes one level of heterogeneity, all three reuse the standard SUM N=1000 shared semantic records; meso and micro also reuse the standard institution CIO decision records. Therefore the LLM token and wall-clock cost of the three ablations are no longer used for efficiency comparison, and the analysis focuses on the re-executed agent decisions, orders, matching, price paths, wealth distribution, and per-stock return fit.

## Aggregate Metrics

Table 1 gives the core results of the four settings. The standard SUM has an equal-weight RMSE of {baseline['eq_rmse']:.2f} and a float-weighted RMSE of {baseline['float_rmse']:.2f}. After the three ablations, equal-weight RMSE rises: macro, meso, micro are {macro['eq_rmse']:.2f}, {meso['eq_rmse']:.2f}, {micro['eq_rmse']:.2f} respectively; float-weighted RMSE is most sensitive to the meso ablation, rising to {meso['float_rmse']:.2f}.

**Table 1. Aggregate ablation metrics**

{md_table(main_table, {'Trade/Attempt': 2, 'EW RMSE': 2, 'FW RMSE': 2, 'EW Final Gap': 2, 'FW Final Gap': 2, 'Final Gini': 4})}

**Table 2. Metric changes relative to Baseline SUM**

{md_table(delta, {'Delta EW RMSE': 2, 'Delta FW RMSE': 2, 'Delta Orders': 0, 'Delta Trades': 0, 'Delta Trade/Attempt': 2})}

![Index paths under three-level heterogeneity ablations](assets/fig_ablation_index_paths.png)

Figure 1 shows that the standard SUM and all three ablations underestimate the real market's rapid drop in early April on the equal-weight index, but the subsequent paths diverge differently. The Macro ablation is closest to the standard SUM path, indicating that replacing the institution-level architecture perturbs the aggregate index path relatively mildly; its float-weighted RMSE is only {macro['float_rmse'] - baseline['float_rmse']:.2f} higher than the standard experiment. This suggests that under the current five-stock, monthly-window, single-institution setting, the main role of institution-level heterogeneity is not to determine the overall index trend but to affect local trades and price corrections of tail stocks.

The Meso ablation has the largest path difference, especially on the float-weighted index, where the terminal gap widens from the standard experiment's {baseline['float_final_gap']:.2f} to {meso['float_final_gap']:.2f}. This shows that collapsing the value, momentum, and noise retail types all into the noise channel significantly changes buy/sell pressure and matching efficiency on high-weight stocks. In other words, the mid-level mapping from type to semantic channel is not a decorative categorical variable but a key mechanism that determines how public information is converted into heterogeneous trade directions.

The Micro ablation has an equal-weight RMSE of {micro['eq_rmse']:.2f}, higher than the standard experiment but lower than macro and meso; however its float-weighted terminal gap reaches {micro['float_final_gap']:.2f}, more negative than the standard experiment. This shows that fixing `gamma` and `sigma` weakens the dispersion of individual response magnitude, so the market still retains some directionality but price pressure on high-weight assets is more easily synchronized.

![Aggregate ablation metrics](assets/fig_ablation_metric_bars.png)

## Market Activity and Matching Efficiency

The market-activity panels in Figure 2 show that the meso ablation produces the most orders, with a total of {int(meso['total_orders']):,}, {int(meso['total_orders'] - baseline['total_orders']):,} more than the standard SUM; but its trade count is only {int(meso['total_trades']):,}, {int(baseline['total_trades'] - meso['total_trades']):,} fewer than the standard experiment. This means unifying the noise channel does not bring more effective liquidity but instead makes order intents more crowded and same-direction, so more orders cannot be matched.

The Macro ablation's order count is close to the standard experiment, with slightly higher trades of {int(macro['total_trades']):,}. This matches the macro ablation design: after the institution capital is split into SUM-style sleeves, the institution level no longer forms large orders via a unified CIO risk control but expresses trading demand through multiple style accounts, so local matching opportunities increase but overall path-fit improvement is limited.

The Micro ablation's order count is {int(micro['total_orders']):,}, close to the standard experiment, but trades drop to {int(micro['total_trades']):,}. This shows that micro parameter heterogeneity mainly affects the dispersion of order magnitude and price offset; when `gamma` and `sigma` are fixed, different agents respond more uniformly to the same signal and the matching space shrinks.

![Daily market activity across ablations](assets/fig_ablation_daily_activity.png)

Figure 3 further shows daily orders and trades. The Meso ablation has significantly higher submitted orders on several trading days, but matched trades do not rise accordingly, showing that mid-level channel heterogeneity is very important for the "matchable difference" in a two-sided market. The standard SUM, macro, and micro have closer daily activity, with macro's trades smoother, reflecting that institution SUM-sleeves provide additional but not drastic counterparties to the order book.

## Per-Stock Error Structure

The per-stock return gap is expressed in percentage points as `real return - simulated return`. Negative means the simulated drop is smaller than the real market, i.e. underestimating the real fall; positive means the simulated drop is larger, i.e. over-falling.

**Table 3. Stock-level return gap (real minus simulated, percentage points)**

{md_table(stock_pivot, {'Baseline': 2, 'Macro': 2, 'Meso': 2, 'Micro': 2})}

![Stock-level return gap heatmap](assets/fig_ablation_stock_return_gap_heatmap.png)

Figure 4 shows that all settings clearly underestimate the real drop of Luxshare Precision `002475.SZ`, the hardest single stock to fit in this window. The standard SUM's return gap for this stock is -11.69 pp; the macro ablation improves it to -8.80 pp, while the meso ablation worsens it to -14.31 pp. This further supports the importance of the mid-level semantic channel: when all retail agents use the noise channel, the model finds it harder to respond in a differentiated way to stock-level fundamental and trend information.

Kweichow Moutai `600519.SH` shows an error in the opposite direction: the standard SUM's return gap is +8.04 pp, all three ablations remain positive, and micro reaches +9.52 pp. This shows the model applies excess downward pressure on high-weight defensive assets, especially when micro response parameters are fixed. Since the float-weighted index is more sensitive to high-weight stocks, this is also an important reason why micro and meso worsen on the float-weighted terminal gap.

## Interpreting Level Contributions

From the overall results of the three ablations, the contribution of the three heterogeneity levels to market simulation is not homogeneous. The Macro level mainly affects how institution capital enters the order book. After making the institution SUM-style sleeves, trades increase slightly, float-weighted RMSE barely changes, but per-stock error is reallocated. This shows that in the current setting, the institution-level architecture affects trade structure and local price pressure more than it dominates the index-level path.

The Meso level contributes the most. After unifying the noise channel, orders surge, matching efficiency drops, and float-weighted RMSE clearly worsens. This shows the value, momentum, and noise semantic channels provide a necessary mid-level behavioral division of labor: the value channel absorbs fundamentals and valuation corrections, the momentum channel absorbs trend and diffusion signals, and the noise channel absorbs sentiment and order-book disturbance. Removing this division leaves the market without enough heterogeneous counterparties, and public information is over-compressed into homogeneous sentiment responses.

The Micro level mainly controls the individual dispersion of response magnitude. After fixing `gamma` and `sigma`, equal-weight RMSE worsens only slightly, but the terminal deviation on high-weight stocks widens. This shows micro heterogeneity is not the most central source of direction judgment but is key to order intensity, fill probability, and the price elasticity of high-weight assets. It provides a lateral dispersion of "different agents reacting with different magnitudes under the same semantics" rather than a new semantic direction.

## Conclusion

This ablation set supports the necessity of the three-level heterogeneity design. The standard SUM achieves the lowest equal-weight RMSE and the lowest float-weighted RMSE among the four; removing any level degrades at least one market-level fit metric. The most critical level is meso semantic-channel heterogeneity, followed by micro response-parameter heterogeneity; macro institution-architecture heterogeneity has a milder effect in this single-month five-stock experiment but still changes trade structure and the per-stock error distribution.

Therefore, the three levels of heterogeneity are not a decorative add-on of agent diversity but correspond to three market-generating mechanisms: the macro level controls the structure of large capital, the meso level controls the interpretive division of public information, and the micro level controls the dispersion of response magnitude under the same interpretation. Only when all three act together does the N=1000 SUM market maintain a relatively balanced simulation across index path, trade activity, and per-stock error.
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
