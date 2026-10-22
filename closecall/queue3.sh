#!/bin/zsh
# Close-call queue, part 3 (2026-10-22): replaces queue2 after MR30. The cheap
# inner reply lost most of the gain (+0.045), so its MR variants are dropped;
# the 1-roll cap (MR3, 0.58 s/trigger, +0.090 on 300 pairs) gets a 1,000-pair
# confirmation, then T=12 at that cap, then distillation (track B).
cd "${0:A:h}/.."
while [ $(wc -l < closecall/MR30.jsonl 2>/dev/null || echo 0) -lt 600 ]; do sleep 60; done
sleep 60
.venv/bin/python close_call.py analyze closecall/MR30.jsonl
echo "== MR30 done $(date)"
run() {   # name seedbase T K pairs max_replies
    echo "== $1 start $(date)"
    SEED_BASE=$2 MAX_REPLIES=$6 PYTHONHASHSEED=0 caffeinate -is .venv/bin/python -u \
        close_call.py match fv2C_it55.pt $3 $4 $5 closecall/$1.jsonl > closecall/$1.log 2>&1
    tail -1 closecall/$1.log
    echo "== $1 done $(date)"
}
run MR3_confirm 96000000 3 3 1000 3
run T12_MR3     97000000 12 3 300 3
echo "== distill start $(date)"
mkdir -p distill
BOARDGAME_DEVICE=cpu PYTHONHASHSEED=0 N_WORKERS=8 caffeinate -is .venv/bin/python -u distill_run.py >> distill/run.log 2>&1
echo "QUEUE3 DONE $(date)"
