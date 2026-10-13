#!/usr/bin/env bash
# Scalability week1 SA/TM runs — institution cache + optional prompt cache + high concurrency.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$SCRIPT_DIR/../../.."

WORKERS="${WORKERS:-60}"
EXP_ROOT="$WORKSPACE/outputs/scalability/scalability_2025_06_top5"
EXP_DATA="$WORKSPACE/data/scalability/scalability_2025_06_top5"
CODE_DIR="$WORKSPACE/framework/run_code"
LOG_DIR="${EXP_DATA}/logs_baselines"
STOCK_CSV="../data_config/scaling_top5_turnover_2025_06.csv"
NEWS_ROOT="${EXP_ROOT}/news_root_for_runs"
START="20250601"
END="20250610"
SEED=42

inst_cache_for_n() {
  local n="$1"
  case "$n" in
    50)  echo "${EXP_DATA}/runs/output_2025_06_top5_sum_50_main" ;;
    100) echo "${EXP_DATA}/runs/output_2025_06_top5_sum_100_real_main" ;;
    500) echo "${EXP_DATA}/runs/output_2025_06_top5_sum_500_main" ;;
    1000) echo "${EXP_DATA}/runs/output_2025_06_top5_sum_1000_real_main" ;;
    *) echo "unknown N=$n" >&2; exit 1 ;;
  esac
}

run_one() {
  local mode="$1" n="$2" prompt_cache="${3:-}"
  local ts tag out log inst
  ts="$(date +%Y%m%d_%H%M%S)"
  tag="${mode}_N$(printf '%04d' "$n")"
  if [[ -n "$prompt_cache" ]]; then
    tag="${tag}_promptcache"
  else
    tag="${tag}_w${WORKERS}"
  fi
  out="${EXP_ROOT}/runs_baselines/output_2025_06_top5_${mode}_${n}_week1_${tag}_${ts}"
  log="${LOG_DIR}/${tag}_week1_${ts}.log"
  inst="$(inst_cache_for_n "$n")"
  mkdir -p "$LOG_DIR" "${EXP_ROOT}/runs_baselines"

  local extra=()
  if [[ -n "$prompt_cache" ]]; then
    extra=(--prompt-baseline-cache-run "$prompt_cache")
  fi

  echo "[launch] $tag -> $out"
  cd "$CODE_DIR"
  PYTHONUNBUFFERED=1 nohup python3 market_sim.py \
    --start "$START" --end "$END" \
    --seed "$SEED" --mock-llm False \
    --retail-mode "$mode" \
    --N "$n" --max-stocks 5 \
    --stock-universe-csv "$STOCK_CSV" \
    --news-root "$NEWS_ROOT" \
    --output-dir "$out" \
    --institution-cache-run "$inst" \
    --prompt-baseline-workers "$WORKERS" \
    --record-prompts False \
    --full-decision-log False \
    ${extra+"${extra[@]}"} \
    > "$log" 2>&1 &
  echo "$!" > "${LOG_DIR}/${tag}_week1_${ts}.pid"
  echo "  pid=$! log=$log"
}

case "${1:-help}" in
  # Same-framework replay only (SA<-SA, TM<-TM). Cross-baseline cache disabled in market_sim.py.
  # n50-cache)
  #   run_one stockagent 50 "${EXP_DATA}/runs_baselines/output_2025_06_top5_stockagent_50_week1"
  #   run_one twinmarket 50 "${EXP_DATA}/runs_baselines/output_2025_06_top5_twinmarket_50_week1"
  #   ;;
  n100)
    run_one stockagent 100 ""
    ;;
  n100-tm)
    run_one twinmarket 100 ""
    ;;
  n500)
    run_one stockagent 500 ""
    ;;
  n500-tm)
    run_one twinmarket 500 ""
    ;;
  *)
    echo "Usage: $0 {n100|n100-tm|n500|n500-tm}  (WORKERS=60 default)"
    exit 1
    ;;
esac
