#!/bin/zsh
# After queue4: does the close-call search still add on top of the distilled net?
# distill_cand3 + close-call (1 roll, T=3 K=3) vs distill_cand3 plain, 1,000 pairs.
cd "${0:A:h}/.."
until grep -q "QUEUE4 DONE" closecall/queue4.log; do sleep 60; done
SEED_BASE=99000000 MAX_REPLIES=3 PYTHONHASHSEED=0 caffeinate -is .venv/bin/python -u \
    close_call.py match distill_cand3.pt 3 3 1000 closecall/stack_distill_cc.jsonl > closecall/stack_distill_cc.log 2>&1
tail -1 closecall/stack_distill_cc.log
echo "QUEUE5 DONE $(date)"
