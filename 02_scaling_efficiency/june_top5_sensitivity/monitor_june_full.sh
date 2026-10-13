#!/usr/bin/env bash
# Monitor June full baseline runs; prints progress every interval.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$SCRIPT_DIR/../../.."

EXP_DATA="$WORKSPACE/data/scalability/scalability_2025_06_top5"
OUT_DIR="${EXP_DATA}/runs_baselines"
TOTAL_STEPS=20
INTERVAL="${1:-60}"

while true; do
  clear
  echo "=== June Full SA/TM Progress ($(date '+%Y-%m-%d %H:%M:%S')) ==="
  printf "%-8s %-6s %-6s %-8s %-10s %-12s %s\n" "MODE" "N" "STEPS" "CALLS" "CACHE_H" "ELAPSED" "DIR"
  running=0
  for d in "$OUT_DIR"/output_2025_06_top5_*_june_full_*; do
    [[ -d "$d" ]] || continue
    base="$(basename "$d")"
    mode="?"
    n="?"
    [[ "$base" == *stockagent* ]] && mode="SA"
    [[ "$base" == *twinmarket* ]] && mode="TM"
    if [[ "$base" =~ _([0-9]+)_june_full ]]; then
      n="${BASH_REMATCH[1]}"
    fi
    steps="0"
    calls="?"
    cache_h="?"
    elapsed="?"
    if [[ -f "$d/step_metrics.json" ]]; then
      steps="$(python3 -c "import json; print(len(json.load(open('$d/step_metrics.json'))))" 2>/dev/null || echo 0)"
    fi
    if [[ -f "$d/run_metadata.json" ]]; then
      read -r calls cache_h elapsed <<< "$(python3 -c "
import json
m=json.load(open('$d/run_metadata.json'))
calls=m.get('retail_llm_calls', m.get('total_retail_llm_calls','?'))
hits=m.get('prompt_baseline_cache_hits','?')
elapsed=m.get('elapsed_seconds', m.get('wall_time_seconds','?'))
print(calls, hits, elapsed)
" 2>/dev/null || echo '? ? ?')"
    fi
    pid_file="${EXP_DATA}/logs_baselines/"*"_june_full_"*".pid"
    alive="done"
    for pf in ${EXP_DATA}/logs_baselines/*june_full*.pid; do
      [[ -f "$pf" ]] || continue
      if grep -q "$(basename "$d" | sed 's/output_2025_06_top5_//')" "${pf%.pid}.log" 2>/dev/null; then
        pid="$(cat "$pf" 2>/dev/null || true)"
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
          if [[ "$d" == *"$(echo "$pf" | sed 's/.*output_2025_06_top5_//')"* ]]; then alive="run"; fi
        fi
      fi
    done
    # simpler alive check via log tail
    log_match="$(ls "${EXP_DATA}/logs_baselines/"*"june_full"* 2>/dev/null | while read -r lg; do
      if grep -q "$(basename "$d")" "$lg" 2>/dev/null; then echo "$lg"; fi
    done | head -1)"
    status="done"
    if [[ -n "$log_match" ]]; then
      if tail -1 "$log_match" 2>/dev/null | grep -qE 'Simulation complete|Saved run_metadata|Traceback|Error'; then
        status="done"
      elif [[ "$steps" -lt "$TOTAL_STEPS" ]]; then
        status="RUN"
        running=$((running+1))
      fi
    fi
    if [[ "$steps" -lt "$TOTAL_STEPS" && "$status" != "done" ]]; then running=$((running+1)); fi
    pct=$(( steps * 100 / TOTAL_STEPS ))
    printf "%-8s %-6s %2d/%-2d(%3d%%) %-8s %-10s %-12s %s [%s]\n" \
      "$mode" "$n" "$steps" "$TOTAL_STEPS" "$pct" "$calls" "$cache_h" "$elapsed" "${base:0:45}" "$status"
  done
  echo ""
  echo "Active-ish runs: $running | refresh every ${INTERVAL}s (Ctrl+C to stop monitor)"
  sleep "$INTERVAL"
done
