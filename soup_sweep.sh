#!/bin/zsh
# soup_sweep.sh -- fv2C soups vs the shipped fv2C_it55, 300 colour-swapped pairs
# each on ONE fresh seed block (82M) so they are comparable; app config. The
# winner then needs a confirmation on another block (winner's curse).
# Waits for a running confirm_match.py first; then resumes any processes listed
# in runs/fv2V1_paused_pids.txt (the paused fv2V1 run).
cd "${0:A:h}"
# [c]onfirm: the bracket stops a watcher that merely MENTIONS the name from matching
while pgrep -f "python -u [c]onfirm_match[.]py" > /dev/null; do sleep 60; done
for s in s3 s4 s4heavy s4span; do
    SEED_BASE=82000000 N_WORKERS=8 PYTHONHASHSEED=0 caffeinate -is .venv/bin/python -u \
        confirm_match.py soups/$s.pt fv2C_it55.pt 300 soups/sw_$s.jsonl > soups/sw_$s.log 2>&1
done
for s in s3 s4 s4heavy s4span; do
    .venv/bin/python confirm_match.py analyze soups/sw_$s.jsonl
done > soups/sweep_summary.txt 2>&1
[[ -f runs/fv2V1_paused_pids.txt ]] && kill -CONT $(cat runs/fv2V1_paused_pids.txt) && echo "fv2V1 resumed $(date)"
