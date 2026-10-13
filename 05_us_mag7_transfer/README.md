# 05 — US Magnificent-7 cross-market transfer

This experiment repeats the SUM fidelity pipeline on the US Magnificent-7 universe
(June 2025, `N=500`, `M=3`) and compares it against an A-share comparator.
Equal-weight normalized metrics are used.

No runnable code, per-day price paths or news tables were present in the October 12
snapshot, so this folder contains only the disclosed metrics.

## `reported_metrics.csv`
Columns: `market, window, N, M, metric, value, units, source, notes`.

| Metric | US Mag7 | A-share comparator |
|---|---|---|
| NRMSE | 0.0833 | 0.0296 |
| NMAPE | 0.0693 | 0.0220 |
| RetCorr | 0.3787 | 0.3885 |
| Wasserstein distance | 0.00835 | 0.00923 |

The transfer experiment reuses the three retail investor prompts from
`00_prompts_and_protocol/sum_investor_prompts/llm_router.py`.
