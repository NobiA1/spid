#!/usr/bin/env python3
"""
P01: Semantic Indistinguishability Behavioral Testing – Experiment Runner.

Execution order (per Section 8.7):
  1. Generate / load NL paraphrases  (10 levels × 10 paraphrases = 100 agents)
  2. MFT   (6 anchors × 2 scenarios × 20 samples = 240 calls)
  3. DIR-A  (100 NL agents × 50 samples           = 5000 calls)
  4. INV    (reuses DIR-A, 0 new calls)
  5. DIR-B  (10 numeric agents × 50 samples        = 500 calls)
  6. DIR-C  (utility function, 0 calls)
  7. Embeddings (100 attribute sentences)
"""

import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from openai import AsyncOpenAI

def log(msg: str):
    print(msg, flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    ACTION_SPACE, CONCURRENCY, DECODING, DIR_SAMPLES, EMBEDDING_MODEL,
    MFT_BEARISH, MFT_BULLISH, MFT_SAMPLES, MINIMAL_CONTROLLED, MODEL,
    NL_ANCHORS, PARAPHRASE_PROMPT, SYSTEM_NL_TEMPLATE, SYSTEM_NUMERIC_TEMPLATE,
    USER_TEMPLATE, calibrate_utility_alpha, compute_utility_actions,
)

BASE_DIR = Path(__file__).resolve().parent
SUM_REPO_DIR = BASE_DIR.parents[2]
ROOT_DIR = SUM_REPO_DIR
RESULTS_DIR = SUM_REPO_DIR / "data" / "controllability" / "P01" / "results" / "main"
RAW_DIR = RESULTS_DIR / "raw_responses"
AGENTS_DIR = RESULTS_DIR / "agents"
EMB_DIR = RESULTS_DIR / "embeddings"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_tokens() -> dict:
    env_path = os.environ.get("SUM_TOKENS_FILE")
    candidates = [
        Path(env_path) if env_path else None,
        SUM_REPO_DIR / "framework" / "data_config" / "tokens.yaml",
        SUM_REPO_DIR / "resources" / "tokens.local.yaml",
        ROOT_DIR / "token.yaml",
    ]
    for path in candidates:
        if path and path.exists():
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f)
    if os.environ.get("OPENAI_API_KEY"):
        return {"openai": {}}
    raise FileNotFoundError("No tokens file found. Set OPENAI_API_KEY or SUM_TOKENS_FILE.")


def get_client(tokens: dict) -> AsyncOpenAI:
    openai_cfg = tokens.get("openai", {}) if tokens else {}
    api_key = os.environ.get("OPENAI_API_KEY") or openai_cfg.get("api_key")
    base_url = (
        os.environ.get("OPENAI_BASE_URL")
        or openai_cfg.get("base_url")
        or "https://api.example-llm.invalid/v1"
    )
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for P01 LLM calls.")
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
    )


def snap_action(val: int | float) -> int:
    """Snap a numeric value to the nearest valid action in ACTION_SPACE."""
    return min(ACTION_SPACE, key=lambda a: abs(a - val))


def build_user_prompt(scenario: dict) -> str:
    return USER_TEMPLATE.format(SCENARIO_JSON=json.dumps(scenario, indent=2))


# ── Async LLM Caller ─────────────────────────────────────────────────────────

async def call_llm(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = DECODING["temperature"],
    top_p: float = DECODING["top_p"],
    top_k: int = DECODING["top_k"],
    max_tokens: int = DECODING["max_tokens"],
    retries: int = 3,
) -> dict:
    """Single LLM call with retry and rate-limiting semaphore."""
    for attempt in range(retries):
        try:
            async with sem:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    extra_body={"top_k": top_k},
                )
            raw_text = resp.choices[0].message.content
            parsed = json.loads(raw_text)
            action = parsed.get("target_position_change_pct")
            if action is None:
                raise ValueError("Missing target_position_change_pct")
            action = snap_action(int(round(float(action))))
            return {
                "action": action,
                "confidence": parsed.get("confidence"),
                "short_reason": parsed.get("short_reason", ""),
                "raw": raw_text,
            }
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                return {"action": None, "error": str(e), "raw": ""}


# ── Step 1: Generate Paraphrases ──────────────────────────────────────────────

async def generate_paraphrases(client: AsyncOpenAI) -> dict:
    """Generate or load 100 NL agent definitions (10 levels × 10 paraphrases)."""
    agents_file = AGENTS_DIR / "nl_agents.json"
    if agents_file.exists():
        log("  ↳ Loading existing NL agents …")
        with open(agents_file) as f:
            return json.load(f)

    log("  ↳ Generating paraphrases via LLM …")
    sem = asyncio.Semaphore(5)
    agents = {}

    async def gen_one_level(level: int):
        anchor = NL_ANCHORS[level]
        prompt = PARAPHRASE_PROMPT.format(anchor=anchor)
        for attempt in range(3):
            try:
                async with sem:
                    resp = await client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.6,
                        top_p=0.95,
                        max_tokens=1200,
                        response_format={"type": "json_object"},
                        extra_body={"top_k": 20},
                    )
                text = resp.choices[0].message.content
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    for v in parsed.values():
                        if isinstance(v, list):
                            parsed = v
                            break
                if not isinstance(parsed, list) or len(parsed) < 9:
                    raise ValueError(f"Expected list of 9, got {type(parsed)}")
                paraphrases = [anchor] + [str(p) for p in parsed[:9]]
                for idx, attr in enumerate(paraphrases):
                    aid = f"risk_L{level}_p{idx+1:02d}"
                    agents[aid] = {"agent_id": aid, "level": level,
                                   "paraphrase_idx": idx + 1, "attribute": attr}
                return
            except Exception as e:
                log(f"    Retry L{level} attempt {attempt+1}: {e}")
                await asyncio.sleep(2)

    await asyncio.gather(*(gen_one_level(l) for l in range(10)))

    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(agents_file, "w") as f:
        json.dump(agents, f, indent=2, ensure_ascii=False)
    log(f"  ↳ Saved {len(agents)} NL agents.")
    return agents


# ── Step 2: Run MFT ──────────────────────────────────────────────────────────

async def run_mft(client: AsyncOpenAI, nl_agents: dict) -> dict:
    """MFT: 6 anchors × 2 scenarios × 20 samples = 240 calls."""
    out_file = RAW_DIR / "mft_responses.json"
    if out_file.exists():
        log("  ↳ Loading existing MFT results …")
        with open(out_file) as f:
            return json.load(f)

    log("  ↳ Running MFT (240 calls) …")
    sem = asyncio.Semaphore(CONCURRENCY)

    nl_ids = ["risk_L0_p01", "risk_L5_p01", "risk_L9_p01"]
    struct_ids = [
        {"agent_id": "struct_risk_00", "risk_tolerance": 0.0},
        {"agent_id": "struct_risk_05", "risk_tolerance": 0.5},
        {"agent_id": "struct_risk_09", "risk_tolerance": 0.9},
    ]

    tasks = []
    for scenario in [MFT_BULLISH, MFT_BEARISH]:
        user_prompt = build_user_prompt(scenario)
        for aid in nl_ids:
            attr = nl_agents[aid]["attribute"]
            sys_p = SYSTEM_NL_TEMPLATE.format(ATTRIBUTE_SENTENCE=attr)
            for s in range(MFT_SAMPLES):
                tasks.append((scenario["scenario_id"], aid, "nl", s, sys_p, user_prompt))
        for sa in struct_ids:
            sys_p = SYSTEM_NUMERIC_TEMPLATE.format(RISK_TOLERANCE=sa["risk_tolerance"])
            for s in range(MFT_SAMPLES):
                tasks.append((scenario["scenario_id"], sa["agent_id"], "numeric", s, sys_p, user_prompt))

    random.shuffle(tasks)
    results = []

    async def run_one(t):
        scen_id, aid, cond, sample, sys_p, usr_p = t
        r = await call_llm(client, sem, sys_p, usr_p)
        r.update({"scenario_id": scen_id, "agent_id": aid, "condition": cond, "sample": sample})
        return r

    done = 0
    for batch_start in range(0, len(tasks), 50):
        batch = tasks[batch_start:batch_start + 50]
        batch_results = await asyncio.gather(*(run_one(t) for t in batch))
        results.extend(batch_results)
        done += len(batch)
        log(f"    MFT progress: {done}/{len(tasks)}")

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log(f"  ↳ MFT done, {len(results)} responses saved.")
    return results


# ── Step 3: Run DIR-A (NL) ───────────────────────────────────────────────────

async def run_dir_a(client: AsyncOpenAI, nl_agents: dict) -> list:
    """DIR-A: 100 NL agents × 50 samples = 5000 calls on minimal_controlled_01."""
    out_file = RAW_DIR / "dir_a_nl.json"
    if out_file.exists():
        log("  ↳ Loading existing DIR-A results …")
        with open(out_file) as f:
            return json.load(f)

    log("  ↳ Running DIR-A NL (5000 calls) …")
    sem = asyncio.Semaphore(CONCURRENCY)
    user_prompt = build_user_prompt(MINIMAL_CONTROLLED)

    tasks = []
    for aid, agent in nl_agents.items():
        sys_p = SYSTEM_NL_TEMPLATE.format(ATTRIBUTE_SENTENCE=agent["attribute"])
        for s in range(DIR_SAMPLES):
            tasks.append((aid, agent["level"], s, sys_p))

    random.shuffle(tasks)
    results = []

    async def run_one(t):
        aid, level, sample, sys_p = t
        r = await call_llm(client, sem, sys_p, user_prompt)
        r.update({"agent_id": aid, "level": level, "sample": sample})
        return r

    checkpoint_file = RAW_DIR / "dir_a_nl_checkpoint.json"
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            results = json.load(f)
        done = len(results)
        tasks = tasks[done:]
        log(f"    Resuming from checkpoint: {done} already done, {len(tasks)} remaining")
    else:
        done = 0

    batch_size = 100
    for batch_start in range(0, len(tasks), batch_size):
        batch = tasks[batch_start:batch_start + batch_size]
        batch_results = await asyncio.gather(*(run_one(t) for t in batch))
        results.extend(batch_results)
        done += len(batch)
        log(f"    DIR-A progress: {done}/5000")
        if done % 500 == 0:
            with open(checkpoint_file, "w") as f:
                json.dump(results, f, ensure_ascii=False)

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    log(f"  ↳ DIR-A done, {len(results)} responses saved.")
    return results


# ── Step 4: Run DIR-B (Numeric) ──────────────────────────────────────────────

async def run_dir_b(client: AsyncOpenAI) -> list:
    """DIR-B: 10 numeric agents × 50 samples = 500 calls on minimal_controlled_01."""
    out_file = RAW_DIR / "dir_b_numeric.json"
    if out_file.exists():
        log("  ↳ Loading existing DIR-B results …")
        with open(out_file) as f:
            return json.load(f)

    log("  ↳ Running DIR-B Numeric (500 calls) …")
    sem = asyncio.Semaphore(CONCURRENCY)
    user_prompt = build_user_prompt(MINIMAL_CONTROLLED)

    tasks = []
    for i in range(10):
        rt = round(i * 0.1, 1)
        aid = f"struct_risk_{i:02d}"
        sys_p = SYSTEM_NUMERIC_TEMPLATE.format(RISK_TOLERANCE=rt)
        for s in range(DIR_SAMPLES):
            tasks.append((aid, rt, s, sys_p))

    random.shuffle(tasks)
    results = []

    async def run_one(t):
        aid, rt, sample, sys_p = t
        r = await call_llm(client, sem, sys_p, user_prompt)
        r.update({"agent_id": aid, "risk_tolerance": rt, "sample": sample})
        return r

    done = 0
    for batch_start in range(0, len(tasks), 50):
        batch = tasks[batch_start:batch_start + 50]
        batch_results = await asyncio.gather(*(run_one(t) for t in batch))
        results.extend(batch_results)
        done += len(batch)
        log(f"    DIR-B progress: {done}/{len(tasks)}")

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log(f"  ↳ DIR-B done, {len(results)} responses saved.")
    return results


# ── Step 5: DIR-C (Utility) ──────────────────────────────────────────────────

def run_dir_c() -> dict:
    """DIR-C: Compute utility-optimal actions (0 LLM calls)."""
    out_file = RAW_DIR / "dir_c_utility.json"
    if out_file.exists():
        log("  ↳ Loading existing DIR-C results …")
        with open(out_file) as f:
            return json.load(f)

    log("  ↳ Computing DIR-C Utility …")
    alpha_scale = calibrate_utility_alpha(MINIMAL_CONTROLLED, min_bins=5)
    log(f"    Calibrated alpha_scale = {alpha_scale}")
    results = compute_utility_actions(MINIMAL_CONTROLLED, alpha_scale)
    out = {"alpha_scale": alpha_scale, "actions": {}}
    for level, data in results.items():
        out["actions"][str(level)] = data

    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)
    log(f"  ↳ DIR-C done, actions: {[results[l]['a_star'] for l in range(10)]}")
    return out


# ── Step 6: Compute Embeddings ────────────────────────────────────────────────

async def compute_embeddings(client: AsyncOpenAI, nl_agents: dict) -> dict:
    """Embed all 100 attribute sentences for Figure P01-4."""
    out_file = EMB_DIR / "attribute_embeddings.json"
    if out_file.exists():
        log("  ↳ Loading existing embeddings …")
        with open(out_file) as f:
            return json.load(f)

    log("  ↳ Computing embeddings (100 sentences) …")
    sem = asyncio.Semaphore(10)
    embeddings = {}

    sorted_agents = sorted(nl_agents.items())

    async def embed_one(aid: str, text: str):
        for attempt in range(3):
            try:
                async with sem:
                    resp = await client.embeddings.create(
                        model=EMBEDDING_MODEL,
                        input=text,
                    )
                return aid, resp.data[0].embedding
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    log(f"    Embedding failed for {aid}: {e}")
                    return aid, None

    tasks = [embed_one(aid, ag["attribute"]) for aid, ag in sorted_agents]
    for batch_start in range(0, len(tasks), 20):
        batch = tasks[batch_start:batch_start + 20]
        batch_results = await asyncio.gather(*batch)
        for aid, emb in batch_results:
            if emb is not None:
                embeddings[aid] = emb
        log(f"    Embeddings progress: {min(batch_start + 20, len(tasks))}/{len(tasks)}")

    with open(out_file, "w") as f:
        json.dump(embeddings, f)
    log(f"  ↳ Embeddings done, {len(embeddings)} vectors saved.")
    return embeddings


# ── Main Orchestrator ─────────────────────────────────────────────────────────

async def main():
    t0 = time.time()
    tokens = load_tokens()
    client = get_client(tokens)

    for d in [RAW_DIR, AGENTS_DIR, EMB_DIR, RESULTS_DIR / "figures"]:
        d.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("P01: Semantic Indistinguishability Experiment Runner")
    log("=" * 60)

    # Step 1: Generate / load NL agents
    log("\n[Step 1] NL Agent Generation")
    nl_agents = await generate_paraphrases(client)

    # Step 2: MFT
    log("\n[Step 2] MFT (Minimum Functionality Tests)")
    mft_results = await run_mft(client, nl_agents)

    # Step 3: DIR-A (NL)
    log("\n[Step 3] DIR-A (Natural-Language Risk Gradient)")
    dir_a_results = await run_dir_a(client, nl_agents)

    # Step 4: DIR-B (Numeric)
    log("\n[Step 4] DIR-B (Numeric-in-Prompt Control)")
    dir_b_results = await run_dir_b(client)

    # Step 5: DIR-C (Utility)
    log("\n[Step 5] DIR-C (Utility-Function Control)")
    dir_c_results = run_dir_c()

    # Step 6: Embeddings
    log("\n[Step 6] Attribute Embeddings")
    embeddings = await compute_embeddings(client, nl_agents)

    elapsed = time.time() - t0
    log(f"\n{'=' * 60}")
    log(f"All experiments completed in {elapsed:.1f}s")
    log(f"Results saved to {RESULTS_DIR}")

    # Save run manifest
    manifest = {
        "model": MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "decoding": DECODING,
        "concurrency": CONCURRENCY,
        "mft_samples": MFT_SAMPLES,
        "dir_samples": DIR_SAMPLES,
        "total_mft_calls": len([r for r in mft_results if r.get("action") is not None]),
        "total_dir_a_calls": len([r for r in dir_a_results if r.get("action") is not None]),
        "total_dir_b_calls": len([r for r in dir_b_results if r.get("action") is not None]),
        "dir_c_alpha_scale": dir_c_results.get("alpha_scale"),
        "total_embedding_calls": len(embeddings),
        "elapsed_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(RESULTS_DIR / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Run analysis
    log("\n[Step 7] Running Analysis …")
    from analysis import run_analysis
    run_analysis()


if __name__ == "__main__":
    asyncio.run(main())
