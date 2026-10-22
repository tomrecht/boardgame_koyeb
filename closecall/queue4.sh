#!/bin/zsh
# Close-call queue, part 4 (2026-10-22, owner): does a RANDOM roll per decision
# beat the fixed 3-4? Same as MR3_confirm (T=3 K=3, 1-roll cap, 1,000 pairs vs
# the plain agent) and on the SAME seeds (96M), so the two results are compared
# seed for seed against fixed-roll's +0.189. Waits for queue3's distillation to
# finish first, so the two never share the CPUs.
cd "${0:A:h}/.."
while pgrep -f distill_run.py > /dev/null; do sleep 60; done
echo "== MR3_random start $(date)"
CC_ROLL=random SEED_BASE=96000000 MAX_REPLIES=3 PYTHONHASHSEED=0 caffeinate -is .venv/bin/python -u \
    close_call.py match fv2C_it55.pt 3 3 1000 closecall/MR3_random.jsonl > closecall/MR3_random.log 2>&1
tail -1 closecall/MR3_random.log
.venv/bin/python closecall/compare_rolls.py closecall/MR3_confirm.jsonl closecall/MR3_random.jsonl
echo "QUEUE4 DONE $(date)"
