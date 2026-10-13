# 04 — Three-level heterogeneity ablation

Two build scripts produce the figures, tables and markdown for the macro / meso /
micro heterogeneity ablations:

* `build_ablation_analysis_202504.py` — April window (top-5, 21 trading days,
  `N=1000`).
* `build_ablation_analysis_202506.py` — June window (top-5, 20 trading days,
  `N=1000`).

Each ablation isolates one level of heterogeneity:

| Level | What is removed | Implementation (see `engine/market_sim.py`) |
|---|---|---|
| Macro | institution-specific architecture | replaced by homogeneous SUM-retail sleeves (`macro_sum_institution`) |
| Meso | type→semantic-channel mapping | all retail agents routed through the noise channel (`meso_noise_channel`) |
| Micro | individual `gamma` / `sigma` dispersion | fixed to distribution means `gamma=0.08`, `sigma=0.0175` (`micro_default_gamma_sigma`) |

The scripts read `<workspace>/data/ablation/<run_id>/` directories containing
`run_metadata.json`, `index_series.csv`, `stock_price_changes.csv` and
`step_metrics.json`, and write tables, figures and an analysis markdown. Adjust the
`ROOT` / `WORKSPACE_ROOT` constants to your run tree.
