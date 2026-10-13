# SUM vs Pure LLM-driven Architecture: Scaling Efficiency

> Note: the strictly comparable two-architecture measured points are
> `N=50/100/500`; SUM also has a measured `N=1000` run, while LLM-driven `N=1000`
> is not included. Therefore all direct `SUM vs LLM-driven` ratio comparisons use
> `N=50/100/500`, with SUM `N=1000` as an additional point showing SUM can keep
> scaling.

> Notation: `ASD` denotes agent-stock-day, i.e. `N x #stocks x #trading_days`.
> `v` means lower is better, `^` means higher is better.

## Table 1. Experimental Configuration

This table fixes the boundaries of a fair comparison: the same April 2025
window, the same 5 stocks selected by traded amount, the same seed, news, market
data, and matching engine. Note that SUM has `N=1000` while LLM-driven is only
run up to `N=500`.

| Field | Value |
| --- | --- |
| Period | 2025-04-01 to 2025-04-30 |
| Stocks | 300059.SZ East Money; 300750.SZ CATL; 002594.SZ BYD; 002475.SZ Luxshare Precision; 600519.SH Kweichow Moutai |
| Selection rule | Top-5 by average traded amount in April 2025 |
| Scales | SUM: N=50,100,500,1000; LLM-driven: N=50,100,500 (N=1000 not yet measured) |
| Architectures | SUM, Pure LLM-driven |
| Shared controls | same seed=42, initial wealth distribution, style ratio, matching engine, news, market data |
| Trading days | 21 |
| Stocks per run | 5 |

**Interpretation.** This table should precede all scalability results, to avoid
the misreading that we compare across datasets, stock pools, or dates; instead we
compare the scaling behavior of the architectures themselves under identical
market inputs.

## Table 2. Resource Scalability

Measures how system resources scale with agent count. Key columns:
`total_llm_calls`, `total_llm_tokens`, `tokens_per_agent_stock_day`, and
`agent_stock_days_per_second`.

| architecture | N | days | stocks | total_llm_calls | failed_llm_calls | total_llm_tokens | elapsed_seconds | total_order_attempts | total_orders | total_trades | orders_per_trade | tokens_per_agent_stock_day | calls_per_agent_stock_day | agent_stock_days_per_second |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SUM | 50 | 21 | 5 | 315 | 0 | 1864002 | 728.5000 | 2889 | 2872 | 393 | 7.3079 | 355.0480 | 0.0600 | 7.2066 |
| SUM | 100 | 21 | 5 | 315 | 0 | 1864867 | 1161.6000 | 4522 | 4512 | 610 | 7.3967 | 177.6064 | 0.0300 | 9.0393 |
| SUM | 500 | 21 | 5 | 315 | 0 | 1864613 | 1196.3200 | 21865 | 21837 | 2761 | 7.9091 | 35.5164 | 0.0060 | 43.8846 |
| SUM | 1000 | 21 | 5 | 315 | 0 | 1863651 | 1113.0300 | 38109 | 38091 | 5195 | 7.3322 | 17.7491 | 0.0030 | 94.3371 |
| LLM-driven | 50 | 21 | 5 | 3696 | 0 | 18032543 | 577.8500 | 2431 | 2108 | 300 | 7.0267 | 3434.7701 | 0.7040 | 9.0854 |
| LLM-driven | 100 | 21 | 5 | 7308 | 0 | 34879077 | 811.1800 | 4420 | 3854 | 558 | 6.9068 | 3321.8169 | 0.6960 | 12.9441 |
| LLM-driven | 500 | 21 | 5 | 37590 | 0 | 175797556 | 1997.2100 | 20494 | 17296 | 1656 | 10.4444 | 3348.5249 | 0.7160 | 26.2867 |

**Interpretation.** SUM's LLM calls are constant at `315` for `N=50/100/500/1000`,
tokens stay around `1.86M`; LLM-driven grows from `N=50` to `N=500` to `37,590`
calls and `175.8M` tokens. SUM's public semantic cost is roughly independent of
`N`, while LLM-driven's grows roughly linearly. SUM's `tokens_per_agent_stock_day`
drops from `355.0` to `17.7` (shared semantics amortized over more agents);
LLM-driven stays at `3300+`.

## Table 3. Market Fidelity

| architecture | N | eq_index_rmse | fw_index_rmse | eq_return_mae | fw_return_mae | eq_direction_accuracy | fw_direction_accuracy | eq_abs_return_autocorr_error | fw_abs_return_autocorr_error | eq_volatility_relative_error | fw_volatility_relative_error | cross_sectional_dispersion_mae | market_fidelity_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SUM | 50 | 89.2721 | 30.6156 | 0.0144 | 0.0105 | 0.6000 | 0.5500 | 0.4647 | 0.3517 | 0.8726 | 0.8336 | 0.0117 | 0.3922 |
| SUM | 100 | 77.9643 | 23.8039 | 0.0141 | 0.0104 | 0.6000 | 0.7000 | 0.1394 | 0.2667 | 0.8669 | 0.8288 | 0.0115 | 0.4621 |
| SUM | 500 | 71.3169 | 33.8851 | 0.0136 | 0.0111 | 0.6500 | 0.5000 | 0.0385 | 0.0935 | 0.8558 | 0.7989 | 0.0110 | 0.4787 |
| SUM | 1000 | 55.0193 | 29.6474 | 0.0140 | 0.0113 | 0.6000 | 0.5500 | 0.3927 | 0.0898 | 0.8637 | 0.8178 | 0.0128 | 0.4407 |
| LLM-driven | 50 | 170.0022 | 121.7809 | 0.0171 | 0.0128 | 0.4000 | 0.4500 | 0.7209 | 0.4077 | 0.9484 | 0.9312 | 0.0170 | 0.1507 |
| LLM-driven | 100 | 167.8224 | 118.7700 | 0.0172 | 0.0126 | 0.4000 | 0.4500 | 0.8935 | 0.7594 | 0.9379 | 0.9172 | 0.0173 | 0.0983 |
| LLM-driven | 500 | 174.2411 | 125.7518 | 0.0173 | 0.0127 | 0.4000 | 0.4500 | 0.9373 | 0.7856 | 0.9327 | 0.9044 | 0.0175 | 0.0821 |

**Interpretation.** SUM's composite fidelity is clearly higher than LLM-driven:
at `N=500`, SUM `market_fidelity_score=0.4787` vs LLM-driven `0.0821`. SUM retains
a better macro market trajectory at a lower semantic cost.

## Table 4. Scaling Efficiency

| N | SUM_fidelity_score | LLM_fidelity_score | token_reduction_x | llm_call_reduction_x | SUM_fidelity_per_1M_tokens | LLM_fidelity_per_1M_tokens | SUM_fidelity_per_min | LLM_fidelity_per_min | elapsed_speedup_x |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 0.3922 | 0.1507 | 9.6741 | 11.7333 | 0.2104 | 0.0084 | 0.0323 | 0.0156 | 0.7932 |
| 100 | 0.4621 | 0.0983 | 18.7033 | 23.2000 | 0.2478 | 0.0028 | 0.0239 | 0.0073 | 0.6983 |
| 500 | 0.4787 | 0.0821 | 94.2810 | 119.3333 | 0.2567 | 0.0005 | 0.0240 | 0.0025 | 1.6695 |

**Interpretation.** As `N` grows from 50 to 500, SUM's token reduction widens from
`9.7x` to `94.3x`, and call reduction from `11.7x` to `119.3x`. SUM's advantage
amplifies with scale.

## Table 5. Complexity Slope Summary

Log-log regression `log(metric)=alpha+beta log(N)`.

| architecture | beta_calls_vs_N | beta_tokens_vs_N | beta_wall_time_vs_N | beta_fidelity_vs_N | interpretation |
| --- | --- | --- | --- | --- | --- |
| SUM | -0.0000 | -0.0001 | 0.1139 | 0.0351 | Shared semantic cost is effectively independent of N |
| LLM-driven | 1.0091 | 0.9918 | 0.5423 | -0.2370 | Per-agent semantic interpretation scales close to O(N) |

## Table 6. Stylized Facts and Behavioral Coverage

| architecture | N | sim_tail_kurtosis | real_tail_kurtosis | sim_vol_clustering_lag1 | sim_abs_return_persistence_lag1_5 | cross_sectional_dispersion_mae | fill_rate_trades_per_order | buy_sell_imbalance_acf1 | turnover_concentration_hhi | action_entropy | stock_choice_entropy | style_separation_js | active_agent_ratio | gini_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SUM | 50 | 1.4441 | 8.0182 | 0.4490 | -0.0026 | 0.0117 | 0.1368 | 0.8005 | 0.2404 | 0.9437 | 0.9975 | 0.0399 | 0.9467 | 0.0005 |
| SUM | 100 | 1.5258 | 8.0182 | 0.1238 | 0.1098 | 0.0115 | 0.1352 | 0.5107 | 0.2702 | 0.8673 | 0.9737 | 0.0358 | 0.8938 | 0.0008 |
| SUM | 500 | 1.3276 | 8.0182 | 0.0229 | -0.0203 | 0.0110 | 0.1264 | 0.5486 | 0.2426 | 0.8725 | 0.9596 | 0.0151 | 0.9172 | 0.0004 |
| SUM | 1000 | 2.3061 | 8.0182 | 0.3771 | -0.1454 | 0.0128 | 0.1364 | 0.3164 | 0.2437 | 0.7749 | 0.9776 | 0.0309 | 0.8443 | 0.0008 |
| LLM-driven | 50 | 2.9760 | 8.0182 | 0.7053 | 0.4783 | 0.0170 | 0.1423 | 0.1092 | 0.2213 | 0.5947 | 0.9940 | 0.1024 | 0.6657 | -0.0006 |
| LLM-driven | 100 | 2.9188 | 8.0182 | 0.8779 | 0.8222 | 0.0173 | 0.1448 | 0.6673 | 0.2157 | 0.5890 | 0.9915 | 0.0931 | 0.6614 | 0.0004 |
| LLM-driven | 500 | 3.7594 | 8.0182 | 0.9217 | 0.9108 | 0.0175 | 0.0957 | 0.4600 | 0.2200 | 0.5746 | 0.9974 | 0.1259 | 0.5829 | 0.0011 |

## Summary

This experiment compares the scalability of SUM and a pure LLM-driven
architecture under the same April 2025 market inputs. SUM performs shared
semantic interpretation over public information, then lets many heterogeneous
agents reuse that semantic layer through explicit utility responses; the pure
LLM-driven baseline asks each agent to interpret and act more directly. SUM's LLM
calls stay fixed at `315` from `N=50` through `N=1000` (tokens stable around
`1.86M`), while LLM-driven grows from `3,696` calls / `18.0M` tokens at `N=50` to
`37,590` calls / `175.8M` tokens at `N=500`, and SUM keeps substantially better
market-fidelity scores. This supports the paper's central scaling claim.

## Figures

![Scalability resource panel](assets/scalability_resource_panel.png)

![Scaling efficiency quality panel](assets/scaling_efficiency_quality_panel.png)

![Marginal scaling efficiency](assets/marginal_scaling_efficiency_corrected.png)

![SUM N1000 real index comparison](assets/sum_N1000_real_index_comparison.png)
