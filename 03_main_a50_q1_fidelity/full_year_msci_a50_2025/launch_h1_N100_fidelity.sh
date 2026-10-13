#!/usr/bin/env bash
# H1 2025 fidelity: MSCI A50 ×50, N=100, seed=42.
# SUM reuses semantic + institution cache from canonical N=1000 full-year run.
# SA/TM use high prompt-baseline concurrency.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$SCRIPT_DIR/../../.."
WORKERS_SA_TM="${WORKERS_SA_TM:-60}"
WORKERS_SUM="${WORKERS_SUM:-20}"
START=20250102
# Q1 only: last trading day 20250331 (57 sessions)
END="${END:-20250331}"
SEED=42
N=100

CODE_DIR="$WORKSPACE/framework/run_code"
EXP_ROOT="$WORKSPACE/outputs/fidelity/full_year_msci_a50_2025"
EXP_DATA="$WORKSPACE/data/fidelity/full_year_msci_a50_2025"
CACHE_RUN="${EXP_DATA}/runs/sum_msci_a50_2025_full_N1000_seed42_full_decisions_20260502_120345"
NEWS_ROOT="$WORKSPACE/framework/data_config/news"
UNIVERSE="../data_config/MSCI_A50.csv"
TS="${1:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${EXP_ROOT}/runs" "${EXP_DATA}/logs"

stop_old() {
  for pid in 99304 99305; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "Stopping old PID $pid"
      kill "$pid" 2>/dev/null || true
    fi
  done
}

launch_sum() {
  local out="${EXP_ROOT}/runs/sum_msci_a50_2025_h1_N${N}_seed42_cache_from_N1000_${TS}"
  local log="${EXP_DATA}/logs/sum_msci_a50_2025_h1_N${N}_seed42_cache_${TS}.log"
  cd "${CODE_DIR}"
  PYTHONUNBUFFERED=1 nohup python3 market_sim.py \
    --start "${START}" --end "${END}" \
    --seed "${SEED}" --mock-llm False \
    --retail-mode signal \
    --N "${N}" --max-stocks 50 \
    --stock-universe-csv "${UNIVERSE}" \
    --news-root "${NEWS_ROOT}" \
    --output-dir "${out}" \
    --semantic-cache-run "${CACHE_RUN}" \
    --institution-cache-run "${CACHE_RUN}" \
    --llm-workers "${WORKERS_SUM}" \
    --record-prompts False \
    --full-decision-log False \
    > "${log}" 2>&1 &
  echo "SUM pid=$! out=${out} log=${log}"
}

launch_baseline() {
  local mode="$1"
  local prefix="$2"
  local out="${EXP_ROOT}/runs/${prefix}_msci_a50_2025_h1_N${N}_seed42_w${WORKERS_SA_TM}_${TS}"
  local log="${EXP_DATA}/logs/${prefix}_msci_a50_2025_h1_N${N}_w${WORKERS_SA_TM}_${TS}.log"
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
    > "${log}" 2>&1 &
  echo "${prefix} pid=$! out=${out} log=${log}"
}

stop_old
launch_sum
launch_baseline stockagent sa
launch_baseline twinmarket tm
echo "TS=${TS} N=${N} H1=${START}-${END} WORKERS_SUM=${WORKERS_SUM} WORKERS_SA_TM=${WORKERS_SA_TM}" > "${EXP_DATA}/logs/launch_h1_N100_${TS}.manifest"
