#!/bin/bash
# Overnight queue (2026-10-04/05): capture-vs-goal rollouts (already running) ->
# net-vs-owner gaps for the newer games -> disagreement rollouts -> summary.
# Stops at the first failing step; everything resumes if re-run.
set -euo pipefail
cd "$(dirname "$0")"
LOGS="quahuru-games-apvo2h65-v2.jsonl quahuru-games-f7in6olg.jsonl quahuru-games-apvo2h65-2026-10-05.jsonl"
stamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "$(stamp) waiting for the capture-vs-goal rollouts"
while pgrep -f "rollout_probe.py cvg" >/dev/null; do sleep 60; done
echo "$(stamp) cvg rollouts done: $(wc -l < rollout_cvg.jsonl) positions"

echo "$(stamp) gaps for games not yet scored"
GAPS_ONLY=1 N_WORKERS=4 RESULTS=weakness_gaps.jsonl python3 weakness_probe.py $LOGS > weakness_gaps2.log 2>&1
echo "$(stamp) gaps done: $(wc -l < weakness_gaps.jsonl) games scored"

echo "$(stamp) disagreement rollouts"
N_ROLL=20 N_WORKERS=4 python3 rollout_probe.py disagree > rollout_disagree.log 2>&1
echo "$(stamp) disagreement rollouts done: $(wc -l < rollout_disagree.jsonl) positions"

{
  echo "=== capture vs goal ($(stamp)) ==="
  python3 rollout_probe.py analyze cvg
  echo
  echo "=== net vs owner disagreements ==="
  python3 rollout_probe.py analyze disagree
} > overnight_summary.txt 2>&1
echo "$(stamp) all done -> overnight_summary.txt"
