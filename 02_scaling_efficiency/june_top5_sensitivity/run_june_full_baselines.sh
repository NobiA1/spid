#!/usr/bin/env bash
# Full June 2025 SA/TM scalability runs with unified prefilter.
# Reuses week1 retail LLM cache for the first 6 trading days; new LLM for days 7–20.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$SCRIPT_DIR/../../.."
set -euo pipefail

WORKERS="${WORKERS:-80}"
EXP_DATA="$WORKSPACE/data/scalability/scalability_2025_06_top5"
CODE_DIR="$WORKSPACE/framework/run_code"
LOG_DIR="${EXP_DATA}/logs_baselines"
OUT_DIR="${EXP_DATA}/runs_baselines"
STOCK_CSV="../data_config/scaling_top5_turnover_2025_06.csv"
NEWS_ROOT="${EXP_DATA}/news_root_for_runs"
START="20250601"
END="20250630"
SEED=42

inst_cache_for_n() {
  local n="$1"
  case "$n" in
    50)  echo "${EXP_DATA}/runs/output_2025_06_top5_sum_50_main" ;;
    100) echo "${EXP_DATA}/runs/output_2025_06_top5_sum_100_real_main" ;;
    500) echo "${EXP_DATA}/runs/output_2025_06_top5_sum_500_main" ;;
    *) echo "unknown N=$n" >&2; exit 1 ;;
  esac
}

prompt_cache_for() {
  local mode="$1" n="$2"
  case "${mode}:${n}" in
    stockagent:50)  echo "${OUT_DIR}/output_2025_06_top5_stockagent_50_week1" ;;
    stockagent:100) echo "${OUT_DIR}/output_2025_06_top5_stockagent_100_week1_stockagent_N0100_w60_20260526_172238" ;;
    stockagent:500) echo "${OUT_DIR}/output_2025_06_top5_stockagent_500_week1_stockagent_N0500_w60_20260526_174046" ;;
    twinmarket:50)  echo "${OUT_DIR}/output_2025_06_top5_twinmarket_50_week1" ;;
    twinmarket:100) echo "${OUT_DIR}/output_2025_06_top5_twinmarket_100_week1_twinmarket_N0100_w60_20260526_173026" ;;
    twinmarket:500) echo "${OUT_DIR}/output_2025_06_top5_twinmarket_500_week1_tm_N0500_w80_20260526_190849" ;;
    *) echo "unknown mode/n: ${mode} N=${n}" >&2; exit 1 ;;
  esac
}

run_one() {
  local mode="$1" n="$2"
  local ts tag out log inst cache
  ts="$(date +%Y%m%d_%H%M%S)"
  tag="${mode}_N$(printf '%04d' "$n")_june_full_w${WORKERS}"
  out="${OUT_DIR}/output_2025_06_top5_${mode}_${n}_june_full_${tag}_${ts}"
  log="${LOG_DIR}/${tag}_${ts}.log"
  inst="$(inst_cache_for_n "$n")"
  cache="$(prompt_cache_for "$mode" "$n")"
  mkdir -p "$LOG_DIR" "$OUT_DIR"

  echo "[launch] ${tag}"
  echo "  out=${out}"
  echo "  cache=${cache}"
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
    --prompt-baseline-cache-run "$cache" \
    --prompt-baseline-workers "$WORKERS" \
    --record-prompts False \
    --full-decision-log False \
    > "$log" 2>&1 &
  echo "$!" > "${LOG_DIR}/${tag}_${ts}.pid"
  echo "  pid=$! log=$log"
}

case "${1:-help}" in
  n50)
    run_one stockagent 50
    run_one twinmarket 50
    ;;
  n100)
    run_one stockagent 100
    run_one twinmarket 100
    ;;
  n500)
    run_one stockagent 500
    run_one twinmarket 500
    ;;
  all)
    run_one stockagent 50
    run_one twinmarket 50
    sleep 5
    run_one stockagent 100
    run_one twinmarket 100
    sleep 5
    run_one stockagent 500
    run_one twinmarket 500
    ;;
  *)
    echo "Usage: WORKERS=80 $0 {n50|n100|n500|all}"
    exit 1
    ;;
esac
