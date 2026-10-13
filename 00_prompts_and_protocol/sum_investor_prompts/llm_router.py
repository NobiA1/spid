"""
LLM Router: call the language model to generate value / momentum / noise investor signals.

Note: the investor and institution prompts below were originally written in Chinese
for an A-share (Chinese mainland equity) market scenario; they have been translated
into English for this anonymized release. The JSON output schema is unchanged.
"""
import json
import os
import re
import time
from typing import Dict, Any, Tuple
try:
    import yaml
except ImportError:
    yaml = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

def _load_openai_config() -> dict:
    # utils/llm_router.py -> run_code/utils, project root is two levels up
    yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data_config", "tokens.yaml"))
    section = os.environ.get("OPENAI_CONFIG_SECTION", "openai")
    if yaml is not None:
        try:
            with open(yaml_path, "r") as f:
                cfg = yaml.safe_load(f)
            return cfg.get(section, {})
        except Exception:
            pass
            
    # Fallback manual parsing
    config = {}
    try:
        with open(yaml_path, "r") as f:
            in_openai = False
            for line in f:
                stripped = line.strip()
                if stripped == f"{section}:":
                    in_openai = True
                elif in_openai and ":" in line:
                    if not line.startswith(" "):
                         in_openai = False
                         continue
                    key, val = stripped.split(":", 1)
                    config[key.strip()] = val.strip()
    except Exception:
        pass
    return config

def _get_llm_runtime_config() -> dict:
    """
    Resolve runtime config with precedence:
      1) env: OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL
      2) data_config/tokens.yaml: openai.api_key / openai.base_url / openai.model
      3) safe defaults (model only)
    """
    cfg = _load_openai_config() or {}
    api_key = os.environ.get("OPENAI_API_KEY") or cfg.get("api_key") or cfg.get("API_KEY") or ""
    base_url = (
        os.environ.get("OPENAI_BASE_URL")
        or cfg.get("base_url")
        or cfg.get("BASE_URL")
        or "https://api.example-llm.invalid/v1"
    )
    model = os.environ.get("OPENAI_MODEL") or cfg.get("model") or cfg.get("MODEL") or "qwen3-8b"
    return {"api_key": api_key, "base_url": base_url, "model": model}

def _call_llm(prompt: str, json_format: bool = True) -> Tuple[str, int]:
    runtime = _get_llm_runtime_config()
    if OpenAI is None:
        print("[LLMRouter] openai package not installed; returning empty json.")
        return "{}", 0
    if not runtime.get("api_key"):
        print("[LLMRouter] Missing OPENAI_API_KEY (or tokens.yaml openai.api_key); returning empty json.")
        return "{}", 0

    client = OpenAI(api_key=runtime["api_key"], base_url=runtime.get("base_url"), timeout=60, max_retries=1)

    kwargs = {
        "model": runtime["model"],
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.6,
        "top_p": 0.95,
        "extra_body": {"top_k": 20},
    }
    
    if json_format:
        kwargs["response_format"] = {"type": "json_object"}

    max_attempts = int(os.environ.get("LLM_MAX_ATTEMPTS", "8") or "8")
    for attempt in range(max_attempts):
        try:
            completion = client.chat.completions.create(**kwargs)
            content = completion.choices[0].message.content
            if content is None or (isinstance(content, str) and not content.strip()):
                raise ValueError("empty LLM content")
            if json_format:
                parse_content = content
                match = re.search(r"```json\n(.*?)```", parse_content, re.DOTALL)
                if match:
                    parse_content = match.group(1)
                json.loads(parse_content)
            tokens = completion.usage.total_tokens if completion.usage else 0
            return content, tokens
        except Exception as e:
            print(f"[LLMRouter] Error calling OpenAI API (attempt {attempt + 1}/{max_attempts}): {e}")
            if attempt < max_attempts - 1:
                time.sleep(min(45.0, 3.0 * (attempt + 1)))
    return "{}", 0

def _parse_json_result(content: str, default_keys: dict, tokens: int) -> Tuple[dict, int]:
    if content is None:
        print("[LLMRouter] Empty LLM content; using default JSON.")
        return default_keys, tokens
    if not isinstance(content, str):
        print(f"[LLMRouter] Non-string LLM content ({type(content).__name__}); using default JSON.")
        return default_keys, tokens
    match = re.search(r"```json\n(.*?)```", content, re.DOTALL)
    if match:
        content = match.group(1)
    
    try:
        parsed = json.loads(content)
        for k in default_keys:
            if k not in parsed:
                parsed[k] = default_keys[k]
        return parsed, tokens
    except json.JSONDecodeError:
        print(f"[LLMRouter] Failed to parse JSON: {content}")
        return default_keys, tokens

# ----------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------

def call_value_agent_llm(context: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    prompt = f"""You are a **value investor**. Your task: judge marginal changes in fundamentals and decide a trading signal.

[Company]: {context.get('firm_name', 'unknown')}
[Current P/B]: {context.get('pb_ratio', 1.0):.2f} (book value per share BPS={context.get('bps', 1.0):.2f})

[Market State (risk regime)]
{context.get('market_state_text', 'market_regime=NA')}

[Today's new information]
--- News ---
{context.get('news_text', 'no news')}
--- Forum posts ---
{context.get('posts_text', 'no posts')}

[Task]
1. Combining today's information, assess the marginal change in the stock's earnings expectation (delta_earnings in [-1.0, 1.0])
2. Assess the marginal change in the stock's asset quality and risk profile (delta_quality in [-1.0, 1.0])
3. Assess the marginal impact of macro environment or industry policy changes (macro_signal in [-1.0, 1.0])
4. Write a one-line forum post expressing your value-oriented view (15-30 words)

[Important constraints (avoid macro over-generalization)]
- The news summary is a structured digest, typically covering the full window from the previous trading day's close to the current trading day's close, at most about 20 items; the system keeps stock-level news first, fills with industry news, and only falls back to high-importance macro news when both are insufficient. Do not look only at the earliest or latest news; you must combine overnight/holiday risk with intraday supportive information.
- The digest contains scope/relevance/importance/event_type/time_lag_hours/tags. You must use this metadata to judge whether it is relevant to this stock.
- If scope=macro and relevance=market: it usually reflects a change in risk appetite / discount rate / liquidity, not an improvement in the stock's earnings; unless there is also a scope=stock direct hit, do not treat macro news directly as a stock-specific fundamental positive.
- Distinguish "policy support / liquidity support" (e.g. ETF purchases, liquidity injection) from "fundamental positive / earnings improvement" (orders, results, products, competitive landscape).
- If within the same window there are both risk items (tariffs / trade conflict / export pressure / earnings pressure) and supportive items (buybacks / share increases / liquidity support), assess them separately: the former lowers short-term risk appetite and earnings expectation; the latter can only partly ease discount-rate or liquidity pressure and cannot automatically offset the risk shock.
- If market_regime=risk_off: even with policy support, prioritize the interpretation that short-term risk assets are under pressure.

Please output ONLY valid JSON, which must contain the following 4 keys: "delta_earnings", "delta_quality", "macro_signal", "post_text". Numeric values are floats.
"""
    content, tokens = _call_llm(prompt)
    return _parse_json_result(content, {
        "delta_earnings": 0.0, "delta_quality": 0.0,
        "macro_signal": 0.0, "post_text": "Valuation stable, no significant marginal change"
    }, tokens)


def call_momentum_agent_llm(context: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    prompt = f"""You are a **momentum investor**. You focus mainly on the continuation of price trends and catalysts, aided by price-volume patterns.

[Company]: {context.get('firm_name', 'unknown')}
[Last-20-day price ratio series (current as 0 baseline)]: {[round(x, 3) for x in context.get('kline_ratios', [])]}
[Current range position]: {context.get('range_pos', 0.5):.2f} (0=20-day low, 1=high)

[Market State (risk regime)]
{context.get('market_state_text', 'market_regime=NA')}

[Today's new information]
--- News ---
{context.get('news_text', 'no news')}
--- Forum discussion heat ---
Posts over the last 3 days: {context.get('n_posts', 0)} (bullish: {context.get('n_bull', 0)}, bearish: {context.get('n_bear', 0)})
{context.get('posts_text', 'no posts')}

[Task]
1. Assess whether the news or event constitutes a catalyst for price momentum continuation or reversal (catalyst_signal in [-1.0, 1.0]; +1 = strong upward-continuation catalyst, -1 = strong downward reversal)
2. Based on forum discussion heat, assess whether the information has already diffused sufficiently among retail investors; the more it has diffused, the more it is priced-in and the less momentum room remains (diffusion_signal in [-1.0, 1.0]; +1 = news just emerging with high potential, -1 = news over-diffused and potential exhausted)
3. Write a one-line technically-oriented forum post expressing your current view (15-30 words)

[Important constraints (avoid macro over-generalization)]
- The news summary is a structured digest, typically covering the full window from the previous trading day's close to the current trading day's close, at most about 20 items; the system keeps stock-level news first, fills with industry news, and only falls back to high-importance macro news when both are insufficient. Do not look only at the latest news; you must combine overnight/holiday risk, pre-open shocks, and intraday support.
- Prioritize scope=stock/direct and scope=industry/industry as "catalyst sources"; treat scope=macro/market as "risk environment" rather than a stock-specific catalyst.
- If the digest marks relevance=unrelated / scope=other: treat it as noise and do not use it as a core catalyst basis.
- If within the window there are both risk shocks and supportive news, interpret it as a mixed state of "risk repricing first, then supportive repair," rather than a one-sided chase-up signal.
- If market_regime=risk_off: reduce the tendency to chase gains and increase the weight on "reversal / drawdown risk" (catalyst_signal more cautious, diffusion_signal leaning toward "risk already priced-in").

Please output ONLY valid JSON, which must contain the following 3 keys: "catalyst_signal", "diffusion_signal", "post_text". Numeric values are floats.
"""
    content, tokens = _call_llm(prompt)
    return _parse_json_result(content, {
        "catalyst_signal": 0.0, "diffusion_signal": 0.0,
        "post_text": "Trend unclear, keep watching price-volume alignment"
    }, tokens)


def call_noise_agent_llm(context: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    prompt = f"""You are a **noise (sentiment) investor**. Your decisions depend heavily on forum sentiment, gut reactions to short-term news, and herd behavior, ignoring any fundamentals.

[Company]: {context.get('firm_name', 'unknown')}

[Market State (risk regime)]
{context.get('market_state_text', 'market_regime=NA')}

[Today's inciting information]
--- Short-term news gut feeling ---
{context.get('news_text', 'no news')}
--- Forum sentiment atmosphere ---
{context.get('posts_text', 'no posts')}

[Task]
1. Based on the news, judge the gut emotional reaction to positive/negative (news_reaction in [-1.0, 1.0])
2. Judge the strength of the current forum sentiment consensus (forum_sentiment in [-1.0, 1.0]: +1 extreme FOMO euphoria, -1 extreme panic stampede)
3. Shout your impulsive gut feeling in the group chat in one sentence, the more emotional and colloquial the better (post_text, 15-30 words)

[Important constraints (how to use the structured digest)]
- The news summary is a structured digest, typically covering the full window from the previous trading day's close to the current trading day's close, at most about 20 items; the system keeps stock-level news first, fills with industry news, and only falls back to high-importance macro news when both are insufficient. Do not be incited only by the latest supportive news; also react emotionally to overnight/holiday risk news.
- If scope=macro/market and market_regime=risk_off, or the digest contains tariffs, trade conflict, export pressure, earnings pressure, or escalating overseas risk, your gut should lean toward panic/caution rather than mechanically treating it as positive.
- For "buyback / share-increase / liquidity support / market-stabilizing" news, you may feel relief or a rebound, but weigh it against risk news; do not automatically output extreme optimism.
- For relevance=unrelated news, output values close to 0 (indifferent) to avoid being falsely incited by noise.

Please output ONLY valid JSON, which must contain the following 3 keys: "news_reaction", "forum_sentiment", "post_text". Numeric values are floats.
"""
    content, tokens = _call_llm(prompt)
    return _parse_json_result(content, {
        "news_reaction": 0.0, "forum_sentiment": 0.0,
        "post_text": "Can't read it, sitting out today"
    }, tokens)


# ======================================================================
# Institution Agent LLM Calls (institution_design.md, Section 4)
# ======================================================================

def call_inst_fundamental_quarterly_llm(context: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """Sub-Agent 1: Fundamental Analyst — quarterly full analysis (cached)."""
    prompt = f"""[Fundamental Analyst · Quarterly baseline]

You are a fundamental research analyst at an asset management firm. Task: establish an
intrinsic-value baseline and earnings-expectation framework for this company, to be used
for daily incremental updates over the coming weeks.

[Company profile]
{context.get('firm_name', 'unknown')} ({context.get('industry', '')})
Main business: {context.get('main_business', 'unknown')}

[Latest quarterly financials (reporting period {context.get('end_date', '')})]
Income statement: revenue {context.get('revenue', 0):,.0f} CNY / net income {context.get('n_income', 0):,.0f} CNY
        gross margin {context.get('grprofit_margin', 0):.1%} net margin {context.get('netprofit_margin', 0):.1%}
        basic EPS {context.get('basic_eps', 0):.3f} CNY

Balance sheet: total assets {context.get('total_assets', 0):,.0f} CNY / total liabilities {context.get('total_liab', 0):,.0f} CNY
            debt-to-assets {context.get('debt_to_assets', 0):.1%}  current ratio {context.get('current_ratio', 0):.2f}
            BPS {context.get('bps', 0):.3f} CNY

Cash flow: operating cash flow {context.get('n_cashflow_act', 0):,.0f} CNY / free cash flow {context.get('free_cashflow', 0):,.0f} CNY
Overall: ROE (weighted) {context.get('roe_waa', 0):.2%}  ROA {context.get('roa', 0):.2%}

[Macro environment]
  10Y government bond {context.get('r_10Y', 0.023):.2%} / LPR (1Y) {context.get('r_LPR', 0.031):.2%}

Please complete the following two tasks:

Task 1: Intrinsic value range
  Based on current financials and macro rates, give a reasonable P/B range (low / mid / high),
  and explain the main driving logic.

Task 2: Expectation sensitivity factors
  List the key factors that could change the market's earnings expectation for this company (3-5 items).

Please output ONLY valid JSON with the following keys:
"fundamental_quality"(float,-1~1), "fair_pb_mid"(float), "fair_pb_range"([float,float]),
"key_expectation_factors"([string,...]), "risk_flag"(float,0~1), "context_summary"(string, within 200 words)
"""
    content, tokens = _call_llm(prompt)
    return _parse_json_result(content, {
        "fundamental_quality": 0.0,
        "fair_pb_mid": 1.0,
        "fair_pb_range": [0.8, 1.5],
        "key_expectation_factors": ["policy change", "industry competition", "macro economy"],
        "risk_flag": 0.3,
        "context_summary": "Fundamental quality neutral; watch industry policy and macro changes",
    }, tokens)


def call_inst_fundamental_daily_llm(context: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """Sub-Agent 1: Fundamental Analyst — daily expectation delta."""
    prompt = f"""[Fundamental Analyst · Daily expectation-change tracking]

Company: {context.get('firm_name', 'unknown')} ({context.get('industry', '')})
Quarterly baseline summary: {context.get('context_summary', 'none')}
Key expectation-sensitive factors: {context.get('key_expectation_factors', [])}

[Current valuation state]
Current P/B: {context.get('pb_ratio', 1.0):.2f}  fair P/B mid: {context.get('fair_pb_mid', 1.0):.2f}
Valuation deviation: {'overvalued' if context.get('pb_ratio', 1) > context.get('fair_pb_mid', 1) else 'undervalued'} {abs(context.get('pb_ratio', 1) / max(context.get('fair_pb_mid', 1), 0.01) - 1):.1%}

[Today's new information (incl. pre-market flashes)]
{context.get('news_inst_text', 'no news')}

Core analysis task:
Judge whether today's information triggered the above expectation-sensitive factors, and the direction:

1. Direction of expectation deviation: will today's information cause the market to revise its expectation of the company's intrinsic value up or down?
   - Upward -> positive value; downward -> negative value; no substantive impact -> close to 0
2. Change certainty: is the above expectation change well-grounded or merely market speculation?
3. Can the expectation improvement drive the price (even if already overvalued)?

Please output ONLY valid JSON with the following keys:
"expectation_delta"(float,-1~1), "delta_confidence"(float,0~1),
"catalyst_type"(string: "policy"|"industry"|"company_event"|"macro"|"competitor"|"none")
"""
    content, tokens = _call_llm(prompt)
    return _parse_json_result(content, {
        "expectation_delta": 0.0,
        "delta_confidence": 0.3,
        "catalyst_type": "none",
    }, tokens)


def call_inst_quant_llm(context: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """Sub-Agent 2: Quantitative Strategist — technical + capital flow analysis."""
    prompt = f"""[Quantitative Strategist task]

You are a quantitative analyst at an asset management firm, responsible for judging short-term price movement from technicals and capital flow.

[Company] {context.get('firm_name', 'unknown')} ({context.get('industry', '')})

[Market technicals (simulated market data)]
Last-20-day close series (ratio to P_t): {context.get('kline_ratios', [])}
Last-20-day volume series (vol ratio): {context.get('vol_ratios', [])}
Moving averages: MA5/P_t = {context.get('ma5_ratio', 0):+.4f}, MA20/P_t = {context.get('ma20_ratio', 0):+.4f}
OBI (order-book imbalance): {context.get('OBI', 0):+.3f}
Range position: range_pos = {context.get('range_pos', 0.5):.0%}

[Capital-flow data]
Margin-buy balance change (last 5 days): {context.get('rzye_chg', 0):+.1%}
Margin-sell balance change (last 5 days): {context.get('rqye_chg', 0):+.1%}
{context.get('block_trade_section', '')}

Please analyze:
1. Technical trend direction and strength (momentum/reversal signals, support/resistance)
2. Capital-flow signals (margin trading direction, block-trade implications, unusual main-force flows)
3. Short-term (1-5 day) price movement forecast

Please output ONLY valid JSON with the following keys:
"technical_signal"(float,-1~1), "capital_flow_signal"(float,-1~1),
"volatility_forecast"(float,0~1),
"key_levels"({{"support":float,"resistance":float}}),
"context_summary"(string, within 100 words)
"""
    content, tokens = _call_llm(prompt)
    return _parse_json_result(content, {
        "technical_signal": 0.0,
        "capital_flow_signal": 0.0,
        "volatility_forecast": 0.3,
        "key_levels": {"support": -0.03, "resistance": 0.03},
        "context_summary": "Technicals neutral, no clear trend signal",
    }, tokens)


def call_inst_sentiment_llm(context: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """Sub-Agent 3: Sentiment & Flow Analyst — retail behavior analysis."""
    prompt = f"""[Sentiment & Flow Analyst task]

You are a behavioral-finance analyst at an asset management firm, specializing in retail behavior patterns and identifying arbitrage opportunities.

[Company] {context.get('firm_name', 'unknown')} ({context.get('industry', '')})

[Retail-cohort monitoring data (non-public)]
  Number of retail holders of this stock: {context.get('n_hold', 0)}
  Net buyers today: {context.get('delta_n', 0):+d}
  Retail average holding cost: {context.get('avg_cost', 0):.2f} (deviation vs today P_t: {context.get('cost_vs_price', 0):+.2%})
  Holding-cost std: {context.get('std_cost', 0):.2f}
  Noise-retail average sentiment: {context.get('avg_rho', 0):+.3f} (variance: {context.get('std_rho', 0):.3f})
  Forum bullish/bearish/neutral post counts: {context.get('n_bull', 0)}/{context.get('n_bear', 0)}/{context.get('n_neutral', 0)}

[Forum posts (public information)]
{context.get('posts_text', 'no posts')}

[News (incl. pre-market flashes)]
{context.get('news_inst_text', 'no news')}

[Market dynamics]
  Today's volume: {context.get('vol_ratio', 1.0):.2f}x average volume
  Order-book imbalance: OBI = {context.get('OBI', 0):+.3f}

Please analyze:
1. How extreme is the current retail-cohort sentiment: is there a consistent bullish/bearish stance?
2. Relationship between retail average cost and current price: are many retail investors in profit or in loss?
3. Are there signs of a "retail stampede" or "retail bottom-fishing"?
4. How should the institution use this information for contrarian trading?

Please output ONLY valid JSON with the following keys:
"retail_sentiment_extreme"(float,-1~1), "retail_pain_signal"(float,-1~1),
"contrarian_opportunity"(float,0~1), "context_summary"(string, within 100 words)
"""
    content, tokens = _call_llm(prompt)
    return _parse_json_result(content, {
        "retail_sentiment_extreme": 0.0,
        "retail_pain_signal": 0.0,
        "contrarian_opportunity": 0.2,
        "context_summary": "Retail sentiment neutral, no clear arbitrage opportunity",
    }, tokens)


def call_inst_cio_llm(context: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """CIO Agent: debate arbitration and final decision (Section 4.5)."""
    prompt = f"""[Chief Investment Officer · Integrated decision]

You are the Chief Investment Officer (CIO) of this asset management firm and must integrate the views of three analysts to make the final investment decision.

[Target] {context.get('firm_name', 'unknown')} (current price P_t = {context.get('last_price', 0):.2f})

[Analyst reports]

- Fundamental Analyst:
  Fundamental quality: {context.get('fundamental_quality', 0):+.2f}  fair P/B mid: {context.get('fair_pb_mid', 1.0):.2f} (current: {context.get('pb_ratio', 1.0):.2f})
  Valuation state: {'overvalued' if context.get('pb_ratio', 1) > context.get('fair_pb_mid', 1) else 'undervalued'} {abs(context.get('pb_ratio', 1) / max(context.get('fair_pb_mid', 1), 0.01) - 1):.1%}
  [Today's expectation change] expectation_delta: {context.get('expectation_delta', 0):+.2f} (certainty: {context.get('delta_confidence', 0):.2f}, type: {context.get('catalyst_type', 'none')})
  Risk alert: {context.get('risk_flag', 0):.2f}
  Conclusion: {context.get('fundamental_summary', '')}

  Note: valuation level alone is not a basis for action. Focus on the direction and certainty of expectation_delta:
     if |expectation_delta| < 0.15 and no clear catalyst -> keep the current position and avoid unnecessary rebalancing.

- Quantitative Strategist:
  Technical signal: {context.get('technical_signal', 0):+.2f}  capital-flow signal: {context.get('capital_flow_signal', 0):+.2f}
  Volatility forecast: {context.get('volatility_forecast', 0):.2f}
  Support: {context.get('support', 0):+.3f} x P_t  Resistance: {context.get('resistance', 0):+.3f} x P_t
  Conclusion: {context.get('quant_summary', '')}

- Sentiment/Flow Analyst:
  Retail sentiment extremity: {context.get('retail_sentiment_extreme', 0):+.2f}
  Retail pain signal: {context.get('retail_pain_signal', 0):+.2f}
  Contrarian opportunity: {context.get('contrarian_opportunity', 0):.2f}
  Conclusion: {context.get('sentiment_summary', '')}

[Current position state]
  Position in this stock: {context.get('current_shares', 0)} shares (market value {context.get('position_value', 0):,.0f} CNY, {context.get('position_pct', 0):.1%} of portfolio)
  Total portfolio position: {context.get('total_position_pct', 0):.1%}
  Single-stock cap: 15.0%

[Decision requirements]
1. Identify the core points of disagreement among the three analysts
2. Balance expected return against risk control
3. Consider the current position state and portfolio constraints
4. If analysts are highly aligned: you may be more aggressive; if there is significant disagreement: be more conservative
5. Pay special attention: does retail behavioral bias offer a high-certainty arbitrage opportunity?

Please output ONLY valid JSON with the following keys:
"action"(string:"buy"|"sell"|"hold"),
"conviction"(float,0~1), "target_weight_delta"(float,-0.10~0.10),
"urgency"(float,0~1), "risk_override"(bool), "reasoning"(string, within 100 words)
"""
    content, tokens = _call_llm(prompt)
    return _parse_json_result(content, {
        "action": "hold",
        "conviction": 0.0,
        "target_weight_delta": 0.0,
        "urgency": 0.3,
        "risk_override": False,
        "reasoning": "Insufficient signal, keep current position",
    }, tokens)
