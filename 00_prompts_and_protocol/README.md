# 00 — Prompts, engine and protocol sources

Shared code that the experiment folders depend on.

## `engine/`
| File | Purpose |
|---|---|
| `market_sim.py` | Simulation orchestrator: loads data/news, initializes agents, runs the daily/intraday matching loop, supports semantic/institution/prompt caches and ablation modes (`no_news_technical_only`, `no_forum`, `macro_sum_institution`, `micro_default_gamma_sigma`). |
| `market_data_loader.py` | Loads the stock universe, daily OHLCV, circulating market cap, latest financial indicators and macro rates, with a local cache. Vendor endpoints and credentials are placeholders. |
| `performance_metrics.py` | Gini coefficient, equal-weight and float-weighted index construction, plotting and JSON exports. |
| `sim_news_filter.py` | Parses merged monthly news CSVs and implements the relevance-tiered selection used to build the per-stock news digest. |
| `sim_news_loader.py` | Loads annotated monthly news and formats the structured digest (scope / relevance / importance / lag) injected into prompts. |
| `simulation_audit.py` | Weekly market supervisor: tracks simulated-vs-real index drift and derives correction biases fed back into the next week. |

## `sum_investor_prompts/llm_router.py`
Canonical SUM prompt layer: the value / momentum / noise retail prompts (returning
semantic signals, not orders) and the institution sub-agents — fundamental
(quarterly + daily), quantitative, sentiment/flow — plus the CIO arbitration prompt.
All calls use model `qwen3-8b`, `temperature=0.6`, `top_p=0.95`, `top_k=20`.

## `prompt_baselines/`
| File | Purpose |
|---|---|
| `common.py` | Shared helpers: OpenAI-compatible JSON call, order payload parsing, risk-profile text renderers. |
| `stockagent_agents.py` | StockAgent-style baseline (persona + direct JSON order). |
| `twinmarket_agents.py` | TwinMarket-style baseline (belief → desire → intention → order). |
| `run_pure_llm_sim.py` | Runner for the pure-LLM retail mode (each agent emits orders directly). |
| `*_investor_prompt_template.md` | Full prompt templates for the three investor types. |

## `p01_prompt/`
`config.py` (scenarios, anchors, templates, decoding), `run.py` (async runner:
paraphrase generation, MFT, DIR-A/B/C, embeddings) and `analysis.py`
(Spearman/Kendall, inversion rate, JSD, figures) for the P01 protocol.
Decoding defaults live in `config.py`: `qwen3-8b`, `0.6 / 0.95 / 20`.

## `config/`
Anonymized example configuration. `tokens.example.yaml` shows the `openai` and
`market_data` sections; `token.example.yaml` shows the flat root form used by the
prompt-baseline helper. All endpoints/keys are placeholders.

## `fake_data/`
`make_fake_data.py` fabricates the minimal synthetic inputs the pipeline needs
(stock universe, daily prices, financials, macro rates, merged + annotated news).
Run it to populate `sample_config/` and `sample_news/`.
