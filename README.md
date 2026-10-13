# SUM — Anonymized Code Release

## Shared configuration

| Parameter | Value |
|---|---|
| Model id | `qwen3-8b` |
| Temperature | `0.6` |
| top_p | `0.95` |
| top_k | `20` |
| API style | OpenAI-compatible chat completions |
| Base URL (placeholder) | `https://api.example-llm.invalid/v1` |
| API key (placeholder) | `sk-placeholder-anonymized` |

Configuration is resolved through, in order of precedence:

1. environment variables `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`;
2. the `openai` section of `data_config/tokens.yaml` (see
   `00_prompts_and_protocol/config/tokens.example.yaml`);
3. safe in-code defaults.

The prompt-baseline helper additionally accepts a flat root `token.yaml`
(`api_key_openai` / `base_url_openai` / `model_openai`); see
`00_prompts_and_protocol/config/token.example.yaml`.

---

## Directory map

```
code_arrOct/
├── 00_prompts_and_protocol/   Shared engine, investor prompts and P01 prompt sources
├── 01_preliminary/            Preliminary / controllability experiments (P01, gamma sweep)
├── 02_scaling_efficiency/     Scaling-efficiency and date-sensitivity experiments
├── 03_main_a50_q1_fidelity/   Main fidelity experiment (A50, 2025) and analysis
├── 04_heterogeneity_ablation/ Three-level heterogeneity ablation analysis
├── 05_us_mag7_transfer/       US Magnificent-7 cross-market transfer (reported metrics)
└── 06_rebuttal_disclosed_only/Rebuttal-only supplementary analyses (reported metrics)
```

### `00_prompts_and_protocol/`
The shared machinery and prompt sources that every experiment depends on.

* `engine/` — core simulator and utilities: `market_sim.py` (orchestrator),
  `market_data_loader.py` (quote/fundamental/macro loading), `performance_metrics.py`
  (Gini, equal- and float-weighted indices, exports), `sim_news_filter.py` and
  `sim_news_loader.py` (news selection and prompt-formatting), `simulation_audit.py`
  (weekly market supervisor).
* `sum_investor_prompts/llm_router.py` — the value / momentum / noise retail prompt
  templates plus the institution sub-agents (fundamental, quant, sentiment) and the
  CIO aggregation prompt. This is the canonical definition of the SUM semantic layer.
* `prompt_baselines/` — StockAgent- and TwinMarket-style prompt baselines, the shared
  baseline helpers (`common.py`), the pure-LLM runner (`run_pure_llm_sim.py`) and the
  three investor prompt-template documents.
* `p01_prompt/` — the prompt/config/runner/analysis sources for the P01
  semantic-indistinguishability protocol.
* `config/` — anonymized example configuration files (fake endpoints/keys).
* `fake_data/` — synthetic news + market-data generator and the tiny sample dataset
  it produces.

### `01_preliminary/`
* `p01_investor_risk_control/` — P01: does a natural-language risk description map
  monotonically to trading behaviour? Contains the config, async runner and analysis
  (Spearman/Kendall, inversion rate, JSD, figures) plus the protocol README.
* `gamma_sweep_robustness/` — controllability sweep over the risk-tolerance
  parameter `gamma`, with `plot_three_panel_boxplot.py` to draw the three investor
  types across three architectures. A synthetic `sample_sweep_results.csv` is
  included so the plotting script runs immediately.

### `02_scaling_efficiency/`
* `april_top5_scaling/` — the April-2025 top-5 scaling report builder
  (`make_llm_sum_scalability_report.py`) and its `SCALING_REPORT.md`. Compares SUM and
  a pure-LLM baseline across `N = 50/100/500(/1000)` on token/call cost and fidelity.
* `june_top5_sensitivity/` — launch and monitor shell scripts for the June-2025
  top-5 reproduction (week-1 and full-month StockAgent/TwinMarket runs, and the
  intraday-20 SUM run).

### `03_main_a50_q1_fidelity/`
The main market-fidelity experiment on the 50-stock `A50` universe (2025), plus
date-window re-checks.
* `analysis/` — `build_fidelity_analysis.py` (rebuild metrics and figures),
  `fidelity_metrics_full.py` (market- and micro-level metrics incl. stylised facts),
  `compare_sum_rb_fullyear.py` (SUM vs rule-based).
* `full_year_msci_a50_2025/` — `compare_january_tm_sa.py` (SUM vs StockAgent vs
  TwinMarket on a January window) and the `launch_*_fidelity.sh` launch scripts for
  H1 / full-year N=100 and N=500 runs.

### `04_heterogeneity_ablation/`
`build_ablation_analysis_202504.py` and `build_ablation_analysis_202506.py` build the
figures, tables and markdown for the macro / meso / micro heterogeneity ablations
(institution architecture, semantic channel, and individual `gamma`/`sigma`).

### `05_us_mag7_transfer/`
Cross-market transfer to the US Magnificent-7 universe. No runnable code was present
in the snapshot; the folder holds the disclosed `reported_metrics.csv` and a README.

### `06_rebuttal_disclosed_only/`
Supplementary analyses disclosed only in the rebuttal: `M`-scaling, no-news and
no-forum controls, lead-lag sensitivity, and the `N=100k` microstructure extension.
Each is a reported-metrics CSV; see the folder README.

---

## Running the code

```bash
# 1) generate the synthetic (fake) news + market data
python code_arrOct/00_prompts_and_protocol/fake_data/make_fake_data.py

# 2) provide fake credentials (or export the env vars)
export OPENAI_API_KEY=sk-placeholder-anonymized
export OPENAI_BASE_URL=https://api.example-llm.invalid/v1
export OPENAI_MODEL=qwen3-8b

# 3) smoke-test the self-contained pieces
python code_arrOct/01_preliminary/gamma_sweep_robustness/plot_three_panel_boxplot.py
```

Scripts that need a completed run directory (the fidelity and ablation analyzers,
the scalability report builder, the shell launchers) expect a run tree of the form
`<workspace>/data/<experiment>/runs/<run_id>/` and are documented in their folder
READMEs. They are provided for completeness and review, not as a single-command
reproduction of the paper.
