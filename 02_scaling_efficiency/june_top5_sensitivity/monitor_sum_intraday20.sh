#!/usr/bin/env bash
# Print one-shot progress for the latest/running SUM intraday20 run.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$SCRIPT_DIR/../../.."

EXP_DATA="$WORKSPACE/data/scalability/scalability_2025_06_top5"
LOG_DIR="${EXP_DATA}/logs_intraday"
TOTAL_STEPS="${TOTAL_STEPS:-400}"

manifest="${1:-}"
if [[ -z "${manifest}" ]]; then
  manifest="$(ls -t "${LOG_DIR}"/sum_N0500_june_top5_intraday*_w*.manifest 2>/dev/null | head -1)"
fi
if [[ -z "${manifest}" || ! -f "${manifest}" ]]; then
  echo "No intraday manifest found in ${LOG_DIR}"
  exit 1
fi

out="$(awk -F= '$1=="out"{print $2}' "${manifest}")"
log="$(awk -F= '$1=="log"{print $2}' "${manifest}")"
pid="$(awk -F= '$1=="pid"{print $2}' "${manifest}")"
status="DONE"
if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
  status="RUNNING"
fi

steps=0
calls="?"
fail="?"
tokens="?"
elapsed="?"
last_step="-"
if [[ -f "${out}/step_metrics.json" ]]; then
  read -r steps calls fail tokens last_step <<< "$(python3 - <<PY
import json
p = "${out}/step_metrics.json"
data = json.load(open(p))
steps = len(data)
calls = sum(int(x.get("llm_calls", 0) or 0) for x in data)
fail = sum(int(x.get("failed_llm_calls", 0) or 0) for x in data)
tokens = sum(int(x.get("total_llm_tokens", 0) or 0) for x in data)
if data:
    last = data[-1]
    last_step = f"{last.get('date')} {last.get('hour')} step={last.get('step')} I={last.get('intraday_step')}/{last.get('intraday_steps_per_day')}"
else:
    last_step = "-"
print(steps, calls, fail, tokens, last_step)
PY
)"
fi
if [[ -f "${out}/run_metadata.json" ]]; then
  elapsed="$(python3 - <<PY
import json
m=json.load(open("${out}/run_metadata.json"))
print(m.get("elapsed_seconds", "?"))
PY
)"
fi
pct=$(( steps * 100 / TOTAL_STEPS ))

echo "=== SUM N=500 June top5 intraday progress ($(date '+%Y-%m-%d %H:%M:%S')) ==="
echo "status=${status} pid=${pid}"
echo "steps=${steps}/${TOTAL_STEPS} (${pct}%)"
echo "llm_calls=${calls} failed=${fail} tokens=${tokens}"
echo "elapsed=${elapsed}"
echo "last=${last_step}"
echo "out=${out}"
echo "log=${log}"
if [[ -f "${log}" ]]; then
  echo "--- log tail ---"
  tail -n 12 "${log}"
fi
