#!/bin/zsh
# Close-call queue, part 2 (2026-10-21 night): owner's Friday plan, track A
# (how much of the gain a phone-sized partial search keeps), then T=12, then
# track B (distillation). Waits for queue.sh's running T3_K3_cheap to finish.
cd "${0:A:h}/.."
while [ $(wc -l < closecall/T3_K3_cheap.jsonl 2>/dev/null || echo 0) -lt 600 ]; do sleep 60; done
sleep 60
.venv/bin/python close_call.py analyze closecall/T3_K3_cheap.jsonl
echo "== T3_K3_cheap done $(date)"
run() {   # name seedbase T K pairs inner_prefilter max_replies
    echo "== $1 start $(date)"
    SEED_BASE=$2 INNER_PREFILTER=$6 MAX_REPLIES=$7 PYTHONHASHSEED=0 caffeinate -is .venv/bin/python -u \
        close_call.py match fv2C_it55.pt $3 $4 $5 closecall/$1.jsonl > closecall/$1.log 2>&1
    tail -1 closecall/$1.log
    echo "== $1 done $(date)"
}
run MR3        91000000 3 3 300 ""     3
run MR9        92000000 3 3 300 ""     9
run MR30       93000000 3 3 300 ""     30
run MR9_cheap  94000000 3 3 300 6,10,3 9
run MR3_cheap  95000000 3 3 300 6,10,3 3
run T12_K3     88000000 12 3 300 ""    0
echo "== distill start $(date)"
mkdir -p distill
BOARDGAME_DEVICE=cpu PYTHONHASHSEED=0 N_WORKERS=8 caffeinate -is .venv/bin/python -u distill_run.py >> distill/run.log 2>&1
echo "QUEUE2 DONE $(date)"
