"""P01 Experiment Configuration: Scenarios, Anchors, Templates, Constants."""

import json

# ── Model & Decoding ──────────────────────────────────────────────────────────
MODEL = "qwen3-8b"
EMBEDDING_MODEL = "qwen3-embedding-0.6b"
DECODING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 120}

# ── Concurrency & Sampling ────────────────────────────────────────────────────
CONCURRENCY = 20
MFT_SAMPLES = 20
DIR_SAMPLES = 50

# ── Action Space ──────────────────────────────────────────────────────────────
ACTION_SPACE = [-100, -75, -50, -25, 0, 25, 50, 75, 100]

# ── Scenarios ─────────────────────────────────────────────────────────────────
MFT_BULLISH = {
    "scenario_id": "mft_bullish_01",
    "task_type": "single_asset_position_adjustment",
    "asset": {
        "symbol": "Asset_Bull",
        "asset_type": "generic_equity",
        "current_price": 100.0,
        "position_pct_now": 25,
        "cash_available_pct": 75,
    },
    "signals": {
        "fundamental_upside_z": 1.4,
        "medium_term_momentum_z": 1.3,
        "crowd_optimism_z": 1.1,
        "valuation_expensiveness_z": 0.2,
        "short_term_drawdown_risk_z": 0.1,
        "volatility_risk_z": 0.2,
    },
    "textual_context": {
        "macro_news": "Macro conditions are supportive and financing conditions have eased.",
        "firm_news": "The company has materially improved its outlook and recent results exceeded expectations.",
        "crowd_summary": "Crowd sentiment is broadly optimistic and near-term expectations are positive.",
    },
}

MFT_BEARISH = {
    "scenario_id": "mft_bearish_01",
    "task_type": "single_asset_position_adjustment",
    "asset": {
        "symbol": "Asset_Bear",
        "asset_type": "generic_equity",
        "current_price": 100.0,
        "position_pct_now": 25,
        "cash_available_pct": 75,
    },
    "signals": {
        "fundamental_upside_z": 0.1,
        "medium_term_momentum_z": 0.2,
        "crowd_optimism_z": 0.1,
        "valuation_expensiveness_z": 1.4,
        "short_term_drawdown_risk_z": 1.3,
        "volatility_risk_z": 1.2,
    },
    "textual_context": {
        "macro_news": "Macro conditions remain fragile and risk appetite is deteriorating.",
        "firm_news": "The company outlook has weakened and the asset appears expensive relative to its fundamentals.",
        "crowd_summary": "Crowd sentiment is cautious and many participants expect further downside.",
    },
}

MINIMAL_CONTROLLED = {
    "scenario_id": "minimal_controlled_01",
    "task_type": "single_asset_position_adjustment",
    "asset": {
        "symbol": "Asset_A",
        "asset_type": "generic_equity",
        "current_price": 100.0,
        "position_pct_now": 25,
        "cash_available_pct": 75,
    },
    "signals": {
        "fundamental_upside_z": 1.1,
        "medium_term_momentum_z": 0.5,
        "crowd_optimism_z": 0.8,
        "valuation_expensiveness_z": 1.0,
        "short_term_drawdown_risk_z": 0.6,
        "volatility_risk_z": 0.7,
    },
    "textual_context": {
        "macro_news": "Macro conditions are stable, but financing conditions remain somewhat uncertain.",
        "firm_news": "The company outlook has improved, but analysts note that the stock already trades at an expensive valuation.",
        "crowd_summary": "Crowd sentiment is optimistic about a rebound, although opinions remain divided.",
    },
}

# ── NL Anchor Sentences (L0–L9) ──────────────────────────────────────────────
NL_ANCHORS = {
    0: "You are an extremely cautious investor who prioritizes capital preservation above all else.",
    1: "You are a very cautious investor who strongly avoids downside risk even at the cost of missing upside.",
    2: "You are a cautious investor who prefers safety and acts conservatively when signals are mixed.",
    3: "You are a somewhat cautious investor who needs strong conviction before taking larger risk.",
    4: "You are a mildly cautious investor who leans conservative but is open to measured opportunities.",
    5: "You are a balanced investor with no strong bias toward caution or aggressiveness.",
    6: "You are a mildly aggressive investor who is willing to take moderate risk for potential upside.",
    7: "You are a somewhat aggressive investor who is comfortable acting decisively when upside seems plausible.",
    8: "You are an aggressive investor who is willing to tolerate substantial downside for higher returns.",
    9: "You are an extremely aggressive investor who actively seeks upside even under significant uncertainty and risk.",
}

# ── Prompt Templates ──────────────────────────────────────────────────────────
SYSTEM_NL_TEMPLATE = """You are an equity investor.
Your defining behavioral trait is:
{ATTRIBUTE_SENTENCE}

You should make decisions in a way that is consistent with this trait.
Use only the information provided in the task.
Choose exactly one action from the allowed action space.
Do not explain your reasoning beyond the required JSON fields.
Return JSON only."""

SYSTEM_NUMERIC_TEMPLATE = """You are an equity investor.
Your risk_tolerance parameter is {RISK_TOLERANCE} on a 0-1 scale, where 0 means maximize safety even if upside is missed, and 1 means maximize upside even if downside risk is high.
You must align your decision strictly with this parameter.
Use only the information provided in the task.
Choose exactly one action from the allowed action space.
Do not explain your reasoning beyond the required JSON fields.
Return JSON only."""

USER_TEMPLATE = """Here is the market scenario:
{SCENARIO_JSON}

Select exactly one action from:
[-100, -75, -50, -25, 0, 25, 50, 75, 100]

Return JSON only in the following format:
{{
  "target_position_change_pct": ...,
  "confidence": ...,
  "short_reason": "..."
}}"""

# ── Paraphrase Meta-Prompt ────────────────────────────────────────────────────
PARAPHRASE_PROMPT = """Generate exactly 9 paraphrases of the following investor behavioral trait description.

Rules:
1. Preserve the EXACT same intended risk tolerance level
2. Only change wording, sentence structure, or synonyms
3. Do NOT introduce any new concepts such as:
   - value investing vs momentum trading
   - long-term vs short-term time horizon
   - rational vs emotional decision-making
   - institutional vs retail investor identity
   - specific sectors, markets, or strategies

Original sentence:
"{anchor}"

Return ONLY a JSON array of exactly 9 strings. No other text."""

# ── Utility Function Parameters ───────────────────────────────────────────────
UTILITY_ALPHA_SCALE = 1.0  # auto-calibrated at runtime if needed
UTILITY_POSITIONS = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]


def compute_utility_actions(scenario: dict, alpha_scale: float = 1.0) -> dict:
    """Compute optimal discrete actions for levels 0..9 using mean-variance utility."""
    sig = scenario["signals"]
    upside = (sig["fundamental_upside_z"] + sig["medium_term_momentum_z"]
              + sig["crowd_optimism_z"]) / 3.0
    downside = (sig["valuation_expensiveness_z"] + sig["short_term_drawdown_risk_z"]
                + sig["volatility_risk_z"]) / 3.0
    signal_composite = upside - downside
    sigma2 = ((sig["short_term_drawdown_risk_z"] + sig["volatility_risk_z"]) / 2.0) ** 2

    results = {}
    for level in range(10):
        alpha_l = (0.01 + 0.01 * level) * alpha_scale
        gamma_l = 5.0 - 0.4 * level

        best_w, best_u = 0.0, float("-inf")
        for w in UTILITY_POSITIONS:
            r_hat = alpha_l * signal_composite
            u = w * r_hat - (gamma_l / 2.0) * (w ** 2) * sigma2
            if u > best_u:
                best_u = u
                best_w = w

        a_star = int(100 * best_w)
        results[level] = {"w_star": best_w, "a_star": a_star, "utility": best_u,
                          "alpha": alpha_l, "gamma": gamma_l}
    return results


def calibrate_utility_alpha(scenario: dict, min_bins: int = 5) -> float:
    """Find smallest alpha_scale producing at least min_bins distinct actions."""
    for scale in [1.0, 10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]:
        actions = compute_utility_actions(scenario, scale)
        unique = len(set(v["a_star"] for v in actions.values()))
        if unique >= min_bins:
            return scale
    return 1000.0
