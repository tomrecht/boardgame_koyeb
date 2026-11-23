#!/bin/zsh
# Two-turn (exact2) teacher distillation on sum7, from the shipped tdlow_rank net (owner, 2026-11-18).
cd /Users/tomrecht/quahuru-train
export PATH=/opt/homebrew/bin:$PATH
export DISTILL_DIR=distill_sum7_e2 START=rankexp/tdlow_rank.pt GOAL_LAYOUT=sum7 CC_DEEP=exact2 \
  REMOTE=1 REMOTE_SHARDS=125 REMOTE_QUEUE_MIN=300 REMOTE_TIMEOUT_MIN=1800 \
  ITERS=2 GAMES=500 T=3 K=3 REPLAY=2 CONFIRM_PAIRS=2000 BAR=0 \
  TRAIN_ARGS='--epochs 6 --lr 2e-5 --td-w 0.05 --rank-w 3e-4 --rank-tau 0.003' \
  BOARDGAME_DEVICE=cpu PYTHONHASHSEED=0 N_WORKERS=8
.venv/bin/python -u distill_run.py >> distill_sum7_e2/run.log 2>&1
# results to git so they survive and are visible remotely
grep -E 'iteration [0-9]+: cand|AFTER|BASELINE' distill_sum7_e2/run.log distill_sum7_e2/train_*.log > distill_sum7_e2/RESULT.txt 2>/dev/null
git add -f distill_sum7_e2/RESULT.txt distill_sum7_e2/state.json distill_sum7_e2/run.sh distill_sum7_e2/confirm_*.jsonl
git commit -qm "exact2-teacher distillation on sum7: results

$(cat distill_sum7_e2/RESULT.txt | tail -12)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push -q
