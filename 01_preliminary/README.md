# 01 — Preliminary / controllability experiments

## `p01_investor_risk_control/`
P01 tests whether a *natural-language* investor description maps to trading
behaviour in a stable, monotonic, controllable way, and contrasts it with (a) a
numeric parameter injected into the prompt and (b) an explicit utility control.

* `config.py` — scenarios (`MFT_BULLISH/BEARISH`, `MINIMAL_CONTROLLED`), the ten
  risk anchors `L0–L9`, prompt templates, the paraphrase meta-prompt, the action
  space and decoding constants (`qwen3-8b`, `temperature=0.6`, `top_p=0.95`,
  `top_k=20`).
* `run.py` — async runner: generate paraphrases (10 levels × 10 wordings), run the
  MFT checks, the NL gradient (DIR-A), the numeric control (DIR-B), the utility
  control (DIR-C) and attribute embeddings.
* `analysis.py` — Spearman ρ, Kendall τ, adjacent-inversion rate, JSD stability
  metrics, and the four P01 figures.
* `README.md` — protocol summary and reported results.

## `gamma_sweep_robustness/`
* `plot_three_panel_boxplot.py` — three-panel boxplot of normalized trade
  aggressiveness across the `gamma` grid, three investor types and three
  architectures (StockAgent / TwinMarket / SUM), with a Spearman summary.
* `sample_sweep_results.csv` — a small synthetic sweep table (columns
  `framework, style, gamma, aggressiveness`) so the plotting script runs without the
  full run output. Regenerate figures with
  `python plot_three_panel_boxplot.py`.

The full `gamma` sweep and the temperature/time-context robustness runs are not
redistributed; only the plotting code and a synthetic sample are included.
