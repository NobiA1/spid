#!/usr/bin/env python3
"""Compare January 2025 fidelity runs: SUM vs StockAgent vs TwinMarket."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[3]  # repo root
ROOT = WORKSPACE / "data/fidelity/full_year_msci_a50_2025/runs"
SUM_DIR = ROOT / "sum_msci_a50_2025_full_N1000_seed42_full_decisions_20260502_120345"
JAN_END = "20250127"


def load_index(run_dir: Path):
    path = run_dir / "index_series.csv"
    dates, ew_sim, ew_real = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            d = row.get("date") or list(row.values())[0]
            if d == "init" or d > JAN_END:
                continue
            dates.append(d)
            ew_sim.append(float(row["equal_weight_sim"]))
            ew_real.append(float(row["equal_weight_real"]))
    return dates, np.array(ew_sim), np.array(ew_real)


def ret(p):
    return np.diff(p) / p[:-1]


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mape(sim, real):
    return float(np.mean(np.abs((sim - real) / (real + 1e-9))) * 100)


def direction_acc(sr, rr):
    return float((np.sign(sr) == np.sign(rr)).mean())


def metrics(name: str, run_dir: Path) -> dict:
    meta_path = run_dir / "run_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    dates, sim, real = load_index(run_dir)
    sr, rr = ret(sim), ret(real)
    return {
        "system": name,
        "path": str(run_dir),
        "n_retail": meta.get("n_retail"),
        "retail_mode": meta.get("retail_mode"),
        "trading_days": len(dates),
        "date_range": f"{dates[0]}–{dates[-1]}" if dates else "",
        "ew_rmse": rmse(sim, real),
        "ew_mape_pct": mape(sim, real),
        "return_corr": float(np.corrcoef(sr, rr)[0, 1]) if len(sr) > 2 else float("nan"),
        "direction_accuracy": direction_acc(sr, rr),
        "final_gini": meta.get("final_gini"),
        "total_llm_calls": meta.get("total_llm_calls"),
        "total_llm_tokens": meta.get("total_llm_tokens"),
        "elapsed_hours": round(float(meta.get("elapsed_seconds", 0)) / 3600, 2),
    }


def find_latest(prefix: str) -> Path | None:
    cands = sorted(ROOT.glob(prefix + "*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in cands:
        if (p / "index_series.csv").exists() and (p / "run_metadata.json").exists():
            return p
    return None


def main():
    sa_dir = find_latest("sa_msci_a50_2025_january_N500")
    tm_dir = find_latest("tm_msci_a50_2025_january_N500")
    rows = [metrics("SUM", SUM_DIR)]
    if sa_dir:
        rows.append(metrics("StockAgent", sa_dir))
    if tm_dir:
        rows.append(metrics("TwinMarket", tm_dir))

    out = ROOT.parent / "january_tm_sa_comparison.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"January comparison ({JAN_END} cutoff)\n")
    hdr = f"{'System':<12} {'N':>5} {'Days':>4} {'EW-RMSE':>9} {'MAPE%':>7} {'RetCorr':>8} {'DirAcc':>7} {'Gini':>6} {'LLM calls':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['system']:<12} {str(r.get('n_retail','?')):>5} {r['trading_days']:>4} "
            f"{r['ew_rmse']:>9.2f} {r['ew_mape_pct']:>7.2f} {r['return_corr']:>8.3f} "
            f"{r['direction_accuracy']:>7.3f} {str(r.get('final_gini','?')):>6} "
            f"{str(r.get('total_llm_calls','?')):>10}"
        )
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
