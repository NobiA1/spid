"""StockAgent-style prompt baseline.

The baseline preserves StockAgent's core interface: natural-language persona
conditioning and direct JSON trading action, while using SUM's LOB order schema.

Note: the prompts below were originally written in Chinese for an A-share market
scenario; they have been translated into English for this anonymized release.
"""

from __future__ import annotations

from typing import Any, Dict

from run_code.baselines.common import (
    BaselineRetailResult,
    base_order_constraints,
    call_json_llm,
    gamma_profile_text,
    result_from_payload,
    sigma_profile_text,
    style_desire_text,
    style_max_trade_value_ratio,
)


OFFSET_HINTS = {
    "value": "value type should lean toward passive limit orders: for BUY, price_offset in [-0.0100, -0.0010]; for SELL in [0.0010, 0.0100].",
    "momentum": "momentum type may chase price actively: for BUY, price_offset in [0.0010, 0.0110]; for SELL in [-0.0110, -0.0010].",
    "noise": "noise type places small impulsive limit orders: for BUY, price_offset in [0.0005, 0.0050]; for SELL in [-0.0050, -0.0005].",
}


def build_stockagent_prompts(style: str, variables: Dict[str, Any]) -> tuple[str, str]:
    gamma_text = gamma_profile_text(variables.get("gamma_t", variables.get("gamma", 0.08)))
    sigma_text = sigma_profile_text(variables.get("sigma", 0.0175))
    variables = {
        **variables,
        "style": style,
        "max_trade_value_ratio": variables.get("max_trade_value_ratio", style_max_trade_value_ratio(style)),
    }
    system_prompt = f"""You are a {style} retail investor in the Chinese A-share market. You must strictly follow the given persona, account constraints, and trading format.

Risk profile: {gamma_text}
Judgment stability: {sigma_text}
Investment style: {style_desire_text(style)}

Your task is to act like a trader in the StockAgent code and directly decide a single trading action based on the current market state.
You may output only valid JSON; when not trading, output {{"orders": [], "post_text": "..."}}.
orders contains at most one order, and price_type is fixed to LIMIT."""

    user_prompt = f"""[Target] {variables.get('firm_name', variables.get('stock_code', 'UNKNOWN'))} ({variables.get('stock_code', '')})
[Market state] {variables.get('market_state_text', 'market_regime=NA')}
[Today's news] {variables.get('news_text', 'no news')}
[Forum posts] {variables.get('posts_text', 'no posts')}

[Price / structured indicators]
- current_price: {variables.get('current_price', 0)}
- kline_ratios: {variables.get('kline_ratios', [])}
- pb_ratio: {variables.get('pb_ratio', '')}
- bps: {variables.get('bps', '')}
- range_pos: {variables.get('range_pos', '')}
- obi: {variables.get('obi', '')}
- vol_ratio: {variables.get('vol_ratio', '')}

[Account and trading constraints]
{base_order_constraints(variables)}

[Hard numeric trading requirements]
- If orders is non-empty, deal_amount must be a positive number, not 0, and must not copy the example placeholder.
- BUY deal_amount must be <= min(cash * 0.95, wealth * max_trade_value_ratio).
- SELL deal_amount must be <= min(current_shares * current_price, wealth * max_trade_value_ratio).
- If you think it is not worth trading, output empty orders rather than an order with deal_amount=0.
- {OFFSET_HINTS.get(style, '')}

[Output format]
Output only valid JSON. The 0.0 below are format placeholders; replace them with your real values when trading:
{{
  "orders": [
    {{
      "stock": "{variables.get('stock_code', '')}",
      "direction": "BUY or SELL",
      "price_type": "LIMIT",
      "price_offset": 0.0,
      "deal_amount": 0.0,
      "agent_id": "{variables.get('agent_id', '')}",
      "client_order_id": "ORD-{variables.get('sim_step', 0)}-{variables.get('agent_id', '')}-{variables.get('stock_code', '')}",
      "reasoning": "within 100 words"
    }}
  ],
  "post_text": "15-30 word forum view"
}}"""
    return system_prompt, user_prompt


def call_stockagent_retail(
    style: str,
    variables: Dict[str, Any],
    *,
    temperature: float = 0.6,
    mock: bool = False,
) -> BaselineRetailResult:
    system_prompt, user_prompt = build_stockagent_prompts(style, variables)
    if mock:
        payload = {"orders": [], "post_text": "mock StockAgent watching"}
        usage = {"success": True, "total_tokens": 0, "raw_content": "{}"}
    else:
        payload, usage = call_json_llm(system_prompt, user_prompt, temperature=temperature)
    return result_from_payload(payload, usage, system_prompt, user_prompt)
