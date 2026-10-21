#!/bin/zsh
# Close-call experiments, run in order (2026-10-21, owner away till Friday).
# Each is resumable: rerunning this script skips finished games.
cd "${0:A:h}/.."
run() {   # name seedbase T K pairs [inner_prefilter]
    echo "== $1 start $(date)"
    SEED_BASE=$2 INNER_PREFILTER=${6:-} PYTHONHASHSEED=0 caffeinate -is .venv/bin/python -u \
        close_call.py match fv2C_it55.pt $3 $4 $5 closecall/$1.jsonl > closecall/$1.log 2>&1
    tail -1 closecall/$1.log
    echo "== $1 done $(date)"
}
run T3_K3_confirm 87000000 3 3 700
run T3_K3_cheap   89000000 3 3 300 6,10,3
run T12_K3        88000000 12 3 300
run T1_K2         90000000 1 2 300
echo "QUEUE DONE $(date)"
