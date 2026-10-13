#!/usr/bin/env python3
"""Three-panel boxplot: controllability across investor types and frameworks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

GAMMAS = [0.04, 0.06, 0.08, 0.10, 0.12]
STYLES = ["value", "momentum", "noise"]
STYLE_LABELS = {"value": "Value", "momentum": "Momentum", "noise": "Noise"}
FRAMEWORKS = ["stockagent", "twinmarket", "sum"]
FW_LABELS = {
    "stockagent": "StockAgent",
    "twinmarket": "TwinMarket",
    "sum": "SUM",
}
FW_COLORS = {
    "stockagent": "#0072B2",
    "twinmarket": "#E69F00",
    "sum": "#009E73",
}


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def framework_key(row: dict) -> str:
    return row.get("framework") or row.get("\ufeffframework", "")


def collect_values(rows: list[dict], framework: str, style: str, gamma: float) -> list[float]:
    return [
        float(r["aggressiveness"])
        for r in rows
        if framework_key(r) == framework
        and r["style"] == style
        and abs(float(r["gamma"]) - gamma) < 1e-9
    ]


def compute_summary(rows: list[dict]) -> list[dict]:
    summary = []
    for fw in FRAMEWORKS:
        for st in STYLES:
            means, stds = [], []
            for g in GAMMAS:
                vals = collect_values(rows, fw, st, g)
                means.append(float(np.mean(vals)) if vals else float("nan"))
                stds.append(float(np.std(vals)) if vals else float("nan"))
            rho, p = stats.spearmanr(GAMMAS, means)
            per_gamma_std = float(np.mean(stds))
            summary.append(
                {
                    "framework": fw,
                    "style": st,
                    "spearman_rho": float(rho),
                    "spearman_p": float(p),
                    "mean_within_gamma_std": per_gamma_std,
                    "gamma_means": means,
                }
            )
    return summary


def plot_three_panel(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
    positions = list(range(len(GAMMAS)))
    width = 0.24
    offsets = {"stockagent": -width, "twinmarket": 0.0, "sum": width}

    for ax, style in zip(axes, STYLES):
        for fw in FRAMEWORKS:
            data = [collect_values(rows, fw, style, g) for g in GAMMAS]
            pos = [p + offsets[fw] for p in positions]
            bp = ax.boxplot(
                data,
                positions=pos,
                widths=width * 0.85,
                patch_artist=True,
                showfliers=True,
                manage_ticks=False,
                flierprops=dict(marker=".", markersize=3, alpha=0.5),
            )
            color = FW_COLORS[fw]
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.35 if fw != "sum" else 0.55)
                patch.set_edgecolor(color)
            for key in ["whiskers", "caps", "medians"]:
                for item in bp[key]:
                    item.set_color(color)
                    if key == "medians":
                        item.set_linewidth(1.2)

        ax.set_title(STYLE_LABELS[style], fontsize=10, fontweight="bold")
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{g:.2f}" for g in GAMMAS], fontsize=8)
        ax.set_xlabel(r"$\gamma$", fontsize=9)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_ylim(-0.02, 0.72)

    axes[0].set_ylabel(r"Position adjustment $|E(\gamma)-E_0|$", fontsize=9)

    handles = [
        plt.Line2D([0], [0], color=FW_COLORS[fw], lw=6, alpha=0.45, label=FW_LABELS[fw])
        for fw in FRAMEWORKS
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"controllability_three_panel_boxplot.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=str(Path(__file__).resolve().parent / "sample_sweep_results.csv"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "figures"),
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    rows = load_rows(csv_path)
    plot_three_panel(rows, out_dir)

    summary = compute_summary(rows)
    print("Saved figures to", out_dir)
    for item in summary:
        print(
            f"{item['framework']:12s} {item['style']:8s} "
            f"ρ={item['spearman_rho']:+.3f}  mean-std={item['mean_within_gamma_std']:.4f}  "
            f"means={[round(x,3) for x in item['gamma_means']]}"
        )


if __name__ == "__main__":
    main()
