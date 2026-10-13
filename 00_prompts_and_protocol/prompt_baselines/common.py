"""Shared helpers for prompt-based and rule-based retail baselines."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

from run_code.simulation.market_engine import AgentOrder


@dataclass(frozen=True)
class BaselineRetailResult:
    orders: List[AgentOrder]
    post_text: str
    raw: Dict[str, Any]
    tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    success: bool = True
    error: str = ""
    raw_content: str = ""
    system_prompt: str = ""
    user_prompt: str = ""


def coerce_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def clip(x: Any, lo: float = -1.0, hi: float = 1.0) -> float:
    v = coerce_float(x, 0.0)
    return max(lo, min(hi, v))


def find_repo_root(start: Optional[Path] = None) -> Path:
    cur = (start or Path(__file__)).resolve()
    for p in [cur, *cur.parents]:
        if (p / "token.yaml").exists():
            return p
    return Path.cwd()


def _parse_key_value_config(path: Path) -> Dict[str, str]:
    cfg: Dict[str, str] = {}
    if not path.exists():
        return cfg
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        cfg[key.strip()] = val.strip().strip("'\"")
    return cfg


def _load_yaml_config(path: Path) -> Dict[str, Any]:
    if yaml is None or not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def load_openai_runtime_config() -> Dict[str, Optional[str]]:
    """Load the requested root token.yaml OpenAI-compatible config first."""
    root = find_repo_root()
    kv = _parse_key_value_config(root / "token.yaml")
    # The baseline experiments are expected to use the root token.yaml
    # api_key_openai/base_url_openai/model_openai triple. Environment variables
    # are only fallback values for ad-hoc local runs.
    api_key = (
        kv.get("api_key_openai")
        or kv.get("api_openai")
        or kv.get("OPENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )
    base_url = kv.get("base_url_openai") or kv.get("BASE_URL_OPENAI") or os.environ.get("OPENAI_BASE_URL") or None
    model = kv.get("model_openai") or kv.get("MODEL_OPENAI") or os.environ.get("OPENAI_MODEL") or ""

    if not api_key or not model:
        data_cfg = _load_yaml_config(
            Path(__file__).resolve().parents[2] / "data_config" / "tokens.yaml"
        )
        section = data_cfg.get(os.environ.get("OPENAI_CONFIG_SECTION", "openai"), {})
        if isinstance(section, dict):
            api_key = api_key or section.get("api_key") or section.get("API_KEY") or ""
            base_url = base_url or section.get("base_url") or section.get("BASE_URL") or None
            model = model or section.get("model") or section.get("MODEL") or ""

    base_url = base_url or "https://api.example-llm.invalid/v1"
    return {"api_key": api_key, "base_url": base_url, "model": model or "qwen3-8b"}


def parse_json_obj(content: str) -> Dict[str, Any]:
    m = re.search(r"```(?:json)?\s*\n([\s\S]*?)\n```", content or "")
    if m:
        content = m.group(1)
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def call_json_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.6,
    max_attempts: int = 2,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    t0 = time.time()
    usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "latency_seconds": 0.0,
        "success": False,
        "error": "",
        "raw_content": "",
    }
    runtime = load_openai_runtime_config()
    if OpenAI is None or not runtime.get("api_key"):
        usage["error"] = "openai_unavailable_or_missing_api_key"
        usage["latency_seconds"] = time.time() - t0
        return {}, usage

    client = OpenAI(
        api_key=runtime["api_key"],
        base_url=runtime.get("base_url"),
        timeout=90,
        max_retries=1,
    )
    for attempt in range(max_attempts):
        try:
            completion = client.chat.completions.create(
                model=runtime["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                top_p=0.95,
                response_format={"type": "json_object"},
                extra_body={"top_k": 20},
            )
            content = completion.choices[0].message.content or "{}"
            usage["raw_content"] = content
            if completion.usage:
                usage["prompt_tokens"] = int(getattr(completion.usage, "prompt_tokens", 0) or 0)
                usage["completion_tokens"] = int(getattr(completion.usage, "completion_tokens", 0) or 0)
                usage["total_tokens"] = int(getattr(completion.usage, "total_tokens", 0) or 0)
            usage["success"] = True
            usage["latency_seconds"] = time.time() - t0
            return parse_json_obj(content), usage
        except Exception as exc:
            usage["error"] = f"attempt {attempt + 1}/{max_attempts}: {exc}"
            if attempt < max_attempts - 1:
                time.sleep(2.0 * (attempt + 1))
    usage["latency_seconds"] = time.time() - t0
    return {}, usage


def normalize_orders_payload(payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    orders = payload.get("orders", [])
    if not isinstance(orders, list):
        orders = []
    post_text = payload.get("post_text", "")
    if not isinstance(post_text, str):
        post_text = str(post_text) if post_text is not None else ""
    return orders[:1], post_text.strip()


def dict_to_agent_order(d: Dict[str, Any]) -> Optional[AgentOrder]:
    if not isinstance(d, dict):
        return None
    required = ["stock", "direction", "price_type", "price_offset", "deal_amount", "agent_id", "client_order_id"]
    if any(k not in d for k in required):
        return None
    direction = str(d.get("direction", "")).upper()
    if direction not in {"BUY", "SELL"}:
        return None
    return AgentOrder(
        stock=str(d.get("stock", "")),
        direction=direction,
        price_type=str(d.get("price_type", "LIMIT")).upper(),
        price_offset=coerce_float(d.get("price_offset", 0.0), 0.0),
        deal_amount=max(0.0, coerce_float(d.get("deal_amount", 0.0), 0.0)),
        agent_id=str(d.get("agent_id", "")),
        client_order_id=str(d.get("client_order_id", "")),
        reasoning=str(d.get("reasoning", ""))[:300],
    )


def apply_sum_style_order_prefilter(
    order: AgentOrder,
    *,
    current_price: float,
    cash: float,
    current_shares: int,
    min_notional_shares: int = 5,
    **_: Any,
) -> Tuple[Optional[AgentOrder], str]:
    """Backward-compatible alias for sum_retail_order_prefilter."""
    from run_code.simulation.agent_behaviors import sum_retail_order_prefilter

    return sum_retail_order_prefilter(
        order,
        current_price=current_price,
        cash=cash,
        current_shares=current_shares,
        min_notional_shares=min_notional_shares,
    )


def result_from_payload(
    payload: Dict[str, Any],
    usage: Dict[str, Any],
    system_prompt: str,
    user_prompt: str,
) -> BaselineRetailResult:
    orders_payload, post_text = normalize_orders_payload(payload or {})
    orders: List[AgentOrder] = []
    for od in orders_payload:
        order = dict_to_agent_order(od)
        if order:
            orders.append(order)
    return BaselineRetailResult(
        orders=orders,
        post_text=post_text,
        raw=payload or {},
        tokens=int(usage.get("total_tokens", 0) or 0),
        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        latency_seconds=float(usage.get("latency_seconds", 0.0) or 0.0),
        success=bool(usage.get("success", False)),
        error=str(usage.get("error", "")),
        raw_content=str(usage.get("raw_content", "")),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


# Note: the following profile strings were originally written in Chinese for an
# A-share market scenario; they have been translated to English for release.
def gamma_profile_text(gamma: Any) -> str:
    g = coerce_float(gamma, 0.08)
    if g < 0.056:
        return "Very high risk tolerance; willing to adjust position size aggressively and quote actively when signals are clear."
    if g < 0.072:
        return "Relatively high risk tolerance; leans toward active trading while still respecting position limits."
    if g < 0.088:
        return "Balanced risk tolerance; trades off between returns and drawdowns."
    if g < 0.104:
        return "Conservative; prefers small positions and cautious quotes."
    return "High risk aversion; prioritizes capital preservation, mostly waiting or placing small orders."


def sigma_profile_text(sigma: Any) -> str:
    s = coerce_float(sigma, 0.0175)
    if s < 0.012:
        return "Strong information-processing ability; judgments are relatively stable."
    if s < 0.023:
        return "Moderate information-processing ability; occasional misjudgments."
    return "High judgment noise; prone to misreading signals, so trading intensity should be reduced."


def style_desire_text(style: str) -> str:
    return {
        "value": "I adjust my position mainly based on valuation margin of safety, fundamental quality, and macro risk.",
        "momentum": "I adjust my position mainly based on price-volume trends, event catalysts, and the degree of information diffusion.",
        "noise": "I am mainly driven by short-term news, forum sentiment, order-book signals, and volume shocks.",
    }.get(style, "I adjust my position based on the given investment style and market information.")


def style_max_trade_value_ratio(style: str) -> float:
    return {"value": 0.60, "momentum": 0.70, "noise": 0.50}.get(style, 0.50)


def base_order_constraints(variables: Dict[str, Any]) -> str:
    return f"""agent_id: {variables.get('agent_id', '')}
style: {variables.get('style', '')}
cash: {variables.get('cash', 0)}
current_shares: {variables.get('current_shares', 0)}
current_position_value: {variables.get('current_position_value', 0)}
wealth: {variables.get('wealth', 0)}
initial_wealth: {variables.get('initial_wealth', 0)}
max_trade_value_ratio: {variables.get('max_trade_value_ratio', style_max_trade_value_ratio(str(variables.get('style', ''))))}
min_order_quantity: {variables.get('min_order_quantity', 1)}
quantity_step: {variables.get('quantity_step', 1)}
max_order_quantity: {variables.get('max_order_quantity', 1000000)}
min_price_tick: {variables.get('min_price_tick', 0.01)}
upper_limit: {variables.get('upper_limit', '')}
lower_limit: {variables.get('lower_limit', '')}"""
