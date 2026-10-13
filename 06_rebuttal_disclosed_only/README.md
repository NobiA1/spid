# 06 — Supplementary analyses (reported metrics only)

These analyses were disclosed only as summary tables; their raw per-day / per-order
sequences were not present in the October 12 snapshot, so no runnable code is
shipped. The disclosed metrics are preserved here for review.

| File | Analysis | Key columns |
|---|---|---|
| `M_scaling_metrics.csv` | Number of institution cohorts `M = {3,10,30,50}` | `cohorts_M, llm_calls, tokens_million, EW_RMSE, EW_MAPE, RetCorr, final_Gini` |
| `no_news_control_metrics.csv` | Technical-only control (real news stream cut) | `metric, standard_SUM, no_news_technical_only, units` |
| `no_forum_ablation_metrics.csv` | Forum-diffusion ablation on cached semantics | `setting, N, EW_RMSE, EW_MAPE, RetCorr, DirAcc, trades, final_Gini` |
| `lead_lag_metrics.csv` | Trajectory alignment vs trading-day lag `k = 0..5` | `lag_trading_days, EW_RMSE, RetCorr, DirAcc` |
| `N100k_microstructure_metrics.csv` | Microstructure at `N=500` vs `N=100k` | `metric, N500, N100000, units` |

The `M`-scaling, no-news and no-forum controls run on a June-2025 top-5 A-share
window; the `N=100k` extension reuses the same market/news/seed and semantic cache.
