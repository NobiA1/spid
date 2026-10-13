# 03 — Main fidelity experiment (A50, 2025) and date re-checks

The main market-fidelity experiment runs on the 50-stock `A50` universe for 2025
(quarter / half-year / full-year windows). No run outputs are redistributed; this
folder contains the analysis code and the launch scripts.

## `analysis/`
| File | Purpose |
|---|---|
| `build_fidelity_analysis.py` | Rebuilds the fidelity metrics summary and the three-panel figures (index path, wealth flow, micro controls `sigma`/`gamma`) from a run directory. |
| `fidelity_metrics_full.py` | Full market-level (RMSE, MAPE, skew/kurtosis, direction accuracy, KS distance, volatility clustering, leverage effect, Hill tail index) and micro-level (turnover, holding period, active ratio, momentum bias, disposition, `gamma`/`sigma`-return correlations) metrics from `index_series.csv` and `retail_decisions.jsonl`. |
| `compare_sum_rb_fullyear.py` | SUM-vs-rule-based comparison of market-level and investor-level metrics for a full-year window. |

Each script defines a `ROOT`/`WORKSPACE` constant pointing at a run tree of the form
`<workspace>/data/fidelity/full_year_msci_a50_2025/runs/<run_id>/`; adjust it to your
own layout.

## `full_year_msci_a50_2025/`
| File | Purpose |
|---|---|
| `compare_january_tm_sa.py` | Compares SUM vs StockAgent vs TwinMarket on a January window (RMSE, MAPE, return correlation, direction accuracy, Gini, LLM calls). |
| `launch_h1_N100_fidelity.sh` | H1 (Q1) fidelity runs at `N=100` reusing canonical semantic/institution caches. |
| `launch_h1_N500_fidelity.sh` | H1 fidelity StockAgent/TwinMarket runs at `N=500`. |
| `launch_tm_sa_fidelity.sh` | January or full-year TwinMarket/StockAgent `N=500` runs. |
