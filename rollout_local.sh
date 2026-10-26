#!/bin/zsh
# This Mac's share of the rollout relabelling (rollout_run.py): waits for the
# luck probe to free the cores, then labels decisions from LOCAL_FROM up,
# skipping any GitHub has already returned. Touches rollout/local.done.
cd "${0:a:h}"
LOCAL_FROM=${LOCAL_FROM:-3600}
while pgrep -f "luck_probe[.]py run" > /dev/null; do sleep 120; done
echo "$(date) local relabel from $LOCAL_FROM"
ROLL_H=4 ROLL_N=4 SEED_BASE=$LOCAL_FROM DONE_GLOB='rollout/relabel_*.jsonl' \
  BOARDGAME_DEVICE=cpu PYTHONHASHSEED=0 N_WORKERS=${N_WORKERS:-8} \
  .venv/bin/python -u rollout_label.py run distill/cand_13.pt rollout/decisions_A.jsonl.gz rollout/relabel_local.jsonl
touch rollout/local.done
echo "$(date) local done"
