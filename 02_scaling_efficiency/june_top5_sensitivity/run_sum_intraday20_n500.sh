#!/usr/bin/env bash
# SUM N=500 June 2025 top-5 turnover, 20 intraday LOB observations per trading day.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$SCRIPT_DIR/../../.."

WORKERS="${WORKERS:-100}"
INTRADAY_STEPS="${INTRADAY_STEPS:-20}"
LLM_MAX_ATTEMPTS="${LLM_MAX_ATTEMPTS:-8}"
OPENAI_CONFIG_SECTION="${OPENAI_CONFIG_SECTION:-high-threads}"
OPENAI_MODEL="${OPENAI_MODEL:-qwen3-8b}"

EXP_DATA="$WORKSPACE/data/scalability/scalability_2025_06_top5"
CODE_DIR="$WORKSPACE/framework/run_code"
LOG_DIR="${EXP_DATA}/logs_intraday"
OUT_DIR="${EXP_DATA}/runs"
STOCK_CSV="../data_config/scaling_top5_turnover_2025_06.csv"
NEWS_ROOT="${EXP_DATA}/news_root_for_runs"
START="20250601"
END="20250630"
SEED=42
CACHE_RUN="${CACHE_RUN:-${EXP_DATA}/runs/output_2025_06_top5_sum_500_main}"

ts="$(date +%Y%m%d_%H%M%S)"
tag="sum_N0500_june_top5_intraday${INTRADAY_STEPS}_cache_w${WORKERS}"
out="${OUT_DIR}/output_2025_06_top5_${tag}_${ts}"
log="${LOG_DIR}/${tag}_${ts}.log"
pid_file="${LOG_DIR}/${tag}_${ts}.pid"
manifest="${LOG_DIR}/${tag}_${ts}.manifest"

mkdir -p "${LOG_DIR}" "${OUT_DIR}"

cd "${CODE_DIR}"
PYTHONUNBUFFERED=1 \
OPENAI_CONFIG_SECTION="${OPENAI_CONFIG_SECTION}" \
OPENAI_MODEL="${OPENAI_MODEL}" \
OPENAI_API_KEY="" \
OPENAI_BASE_URL="" \
LLM_MAX_ATTEMPTS="${LLM_MAX_ATTEMPTS}" \
nohup python3 market_sim.py \
  --start "${START}" --end "${END}" \
  --seed "${SEED}" --mock-llm False \
  --retail-mode signal \
  --N 500 --max-stocks 5 \
  --stock-universe-csv "${STOCK_CSV}" \
  --news-root "${NEWS_ROOT}" \
  --output-dir "${out}" \
  --semantic-cache-run "${CACHE_RUN}" \
  --institution-cache-run "${CACHE_RUN}" \
  --llm-workers "${WORKERS}" \
  --record-prompts False \
  --full-decision-log False \
  --lob-snapshot True \
  --intraday-steps-per-day "${INTRADAY_STEPS}" \
  > "${log}" 2>&1 &

pid="$!"
echo "${pid}" > "${pid_file}"
{
  echo "pid=${pid}"
  echo "out=${out}"
  echo "log=${log}"
  echo "workers=${WORKERS}"
  echo "intraday_steps=${INTRADAY_STEPS}"
  echo "llm_max_attempts=${LLM_MAX_ATTEMPTS}"
  echo "openai_config_section=${OPENAI_CONFIG_SECTION}"
  echo "openai_model=${OPENAI_MODEL}"
  echo "period=${START}-${END}"
  echo "seed=${SEED}"
  echo "cache_run=${CACHE_RUN}"
} > "${manifest}"

echo "pid=${pid}"
echo "out=${out}"
echo "log=${log}"
echo "manifest=${manifest}"
