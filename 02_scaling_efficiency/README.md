# 02 — Scaling efficiency and date sensitivity

## `april_top5_scaling/`
* `make_llm_sum_scalability_report.py` — builds the April-2025 top-5 scaling
  report: SUM vs a pure-LLM baseline over `N = 50/100/500/1000` (SUM) and
  `N = 50/100/500` (pure LLM), covering LLM call/token cost, per-agent-stock-day
  cost, and market-fidelity scores, plus all figures and markdown tables.
* `SCALING_REPORT.md` — the rendered report / full metric list for that experiment.

The script reads a run tree under `<workspace>/weak_baseline/exp/scalability/...`;
point `RUN_ROOT` at your own runs to regenerate.

## `june_top5_sensitivity/`
Launch and monitor shell scripts for the June-2025 top-5 reproduction, which re-checks
the scaling result on an independent date window.

| Script | Purpose |
|---|---|
| `run_week1_baselines.sh` | Week-1 StockAgent / TwinMarket runs (institution cache + optional prompt cache). |
| `run_june_full_baselines.sh` | Full-month StockAgent / TwinMarket runs reusing the week-1 retail cache. |
| `run_sum_intraday20_n500.sh` | SUM `N=500`, 20 intraday order-book observations per day, cached semantics. |
| `monitor_june_full.sh` | Progress monitor for the full-month baseline runs. |
| `monitor_sum_intraday20.sh` | One-shot progress report for the latest intraday-20 run. |

All scripts call `market_sim.py` with `--mock-llm False` and the released model
(configurable via `OPENAI_MODEL`, default `qwen3-8b`).
