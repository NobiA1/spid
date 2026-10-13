#!/usr/bin/env bash
# TM/SA fidelity: MSCI A50 ×50, N=500, seed=42, institution cache from canonical SUM.
set -euo pipefail

PHASE="${1:-january}"  # january | full
TS="${2:-$(date +%Y%m%d_%H%M%S)}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$SCRIPT_DIR/../../.."

CODE_DIR="$WORKSPACE/framework/run_code"
EXP_ROOT="$WORKSPACE/outputs/fidelity/full_year_msci_a50_2025"
EXP_DATA="$WORKSPACE/data/fidelity/full_year_msci_a50_2025"
INST_CACHE="${EXP_DATA}/runs/sum_msci_a50_2025_full_N1000_seed42_full_decisions_20260502_120345"
NEWS_ROOT="$WORKSPACE/framework/data_config/news"
UNIVERSE="../data_config/MSCI_A50.csv"
WORKERS=20

mkdir -p "${EXP_ROOT}/runs" "${EXP_DATA}/logs"

if [[ "$PHASE" == "january" ]]; then
  START=20250102
  END=20250127
  TAG=january
elif [[ "$PHASE" == "full" ]]; then
  START=20250102
  END=20251231
  TAG=full
else
  echo "Usage: $0 [january|full] [timestamp]"
  exit 1
fi

launch_one() {
  local mode="$1"
  local prefix="$2"
  local out="${EXP_ROOT}/runs/${prefix}_msci_a50_2025_${TAG}_N500_seed42_${TS}"
  local log="${EXP_DATA}/logs/${prefix}_msci_a50_2025_${TAG}_N500_seed42_${TS}.log"
  echo "${out}" > "/tmp/fidelity_${prefix}_${TAG}_out.txt"
  echo "${log}" > "/tmp/fidelity_${prefix}_${TAG}_log.txt"
  cd "${CODE_DIR}"
  nohup python3 market_sim.py \
    --start "${START}" --end "${END}" \
    --seed 42 --mock-llm False \
    --retail-mode "${mode}" \
    --N 500 --max-stocks 50 \
    --stock-universe-csv "${UNIVERSE}" \
    --news-root "${NEWS_ROOT}" \
    --output-dir "${out}" \
    --institution-cache-run "${INST_CACHE}" \
    --prompt-baseline-workers "${WORKERS}" \
    --record-prompts False \
    --full-decision-log False \
    > "${log}" 2>&1 &
  echo "Started ${prefix} PID=$! OUT=${out} LOG=${log}"
}

launch_one stockagent sa
launch_one twinmarket tm
echo "TS=${TS} PHASE=${TAG}" > "${EXP_DATA}/logs/launch_${TAG}_N500_${TS}.manifest"
