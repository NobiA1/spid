#!/usr/bin/env python3
"""
Runner for the pure LLM-driven retail baseline.

Note: `retail_mode=pure_llm` was removed from framework/run_code/market_sim.py.
Use the bundled outputs for reproduction, or temporarily restore pure_llm
support before re-running.
"""

import argparse
import os
import sys
from pathlib import Path

SUM_REPO_ROOT = Path(__file__).resolve().parents[4]
SUM_ARCH_ROOT = SUM_REPO_ROOT / "framework"
sys.path.insert(0, str(SUM_ARCH_ROOT))

from run_code.market_sim import run_simulation  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Pure LLM Retail Mode")
    parser.add_argument("--start", default="20260401")
    parser.add_argument("--end", default="20260407")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mock-llm", type=str, default="True")
    parser.add_argument("--N", type=int, default=300)
    parser.add_argument("--inst-early-news", type=int, default=1, choices=[0, 1])
    parser.add_argument("--pure-llm-workers", type=int, default=10)
    parser.add_argument("--record-prompts", type=str, default="True")
    parser.add_argument(
        "--stock-universe-csv",
        default=str(SUM_ARCH_ROOT / "data_config" / "retail_100.csv"),
    )
    parser.add_argument(
        "--news-root",
        default=str(SUM_ARCH_ROOT / "data_config" / "news"),
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if "pure_llm" not in getattr(run_simulation, "__code__", object()).co_varnames:
        raise SystemExit(
            "SUM_repo no longer supports retail_mode=pure_llm. "
            "See weak_baseline/exp/ for archived runs."
        )

    use_mock_llm = args.mock_llm.lower() in ("true", "1", "t", "yes")
    record_prompts = args.record_prompts.lower() in ("true", "1", "t", "yes")

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
        retail_mode="pure_llm",
        prompt_baseline_workers=args.pure_llm_workers,
        record_prompts=record_prompts,
    )


if __name__ == "__main__":
    main()
