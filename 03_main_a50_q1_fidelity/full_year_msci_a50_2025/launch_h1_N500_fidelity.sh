#!/usr/bin/env bash
# H1 2025 fidelity baselines: MSCI A50 ×50, N=500, seed=42.
# Default workers=100 (~5 min/day target at N=500). Optional prompt cache replay.
set -euo pipefail

WORKERS_SA_TM="${WORKERS_SA_TM:-100}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$SCRIPT_DIR/../../.."
START=20250102
END=20250630
SEED=42
N=500

CODE_DIR="$WORKSPACE/framework/run_code"
EXP_DATA="$WORKSPACE/data/fidelity/full_year_msci_a50_2025"
CACHE_RUN="${EXP_DATA}/runs/sum_msci_a50_2025_full_N1000_seed42_full_decisions_20260502_120345"
NEWS_ROOT="$WORKSPACE/framework/data_config/news"
UNIVERSE="../data_config/MSCI_A50.csv"
TS="${1:-$(date +%Y%m%d_%H%M%S)}"

# Optional: PROMPT_CACHE_SA=/path/to/partial/run  PROMPT_CACHE_TM=/path/to/partial/run
PROMPT_CACHE_SA="${PROMPT_CACHE_SA:-}"
PROMPT_CACHE_TM="${PROMPT_CACHE_TM:-}"

mkdir -p "${EXP_DATA}/runs" "${EXP_DATA}/logs"

launch_baseline() {
  local mode="$1"
  local prefix="$2"
  local prompt_cache="${3:-}"
  local out="${EXP_DATA}/runs/${prefix}_msci_a50_2025_h1_N${N}_seed42_w${WORKERS_SA_TM}_${TS}"
  local log="${EXP_DATA}/logs/${prefix}_msci_a50_2025_h1_N${N}_w${WORKERS_SA_TM}_${TS}.log"
  local extra=()
  if [[ -n "${prompt_cache}" ]]; then
    extra=(--prompt-baseline-cache-run "${prompt_cache}")
  fi
  cd "${CODE_DIR}"
  PYTHONUNBUFFERED=1 nohup python3 market_sim.py \
    --start "${START}" --end "${END}" \
    --seed "${SEED}" --mock-llm False \
    --retail-mode "${mode}" \
    --N "${N}" --max-stocks 50 \
    --stock-universe-csv "${UNIVERSE}" \
    --news-root "${NEWS_ROOT}" \
    --output-dir "${out}" \
    --institution-cache-run "${CACHE_RUN}" \
    --prompt-baseline-workers "${WORKERS_SA_TM}" \
    --record-prompts False \
    --full-decision-log False \
    "${extra[@]}" \
    > "${log}" 2>&1 &
  echo "${prefix} pid=$! out=${out}"
}

launch_baseline stockagent sa "${PROMPT_CACHE_SA}"
launch_baseline twinmarket tm "${PROMPT_CACHE_TM}"
{
  echo "TS=${TS} N=${N} WORKERS=${WORKERS_SA_TM}"
  echo "H1=${START}-${END}"
  echo "PROMPT_CACHE_SA=${PROMPT_CACHE_SA}"
  echo "PROMPT_CACHE_TM=${PROMPT_CACHE_TM}"
} > "${EXP_DATA}/logs/launch_h1_N500_${TS}.manifest"
