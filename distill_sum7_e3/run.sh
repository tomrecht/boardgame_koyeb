#!/bin/zsh
# Two-turn (exact2) teacher, SCALED: 1,000 games an iteration from the shipped cand_1 (owner, 2026-11-24).
cd /Users/tomrecht/quahuru-train
export PATH=/opt/homebrew/bin:$PATH
export DISTILL_DIR=distill_sum7_e3 START=distill_sum7_e2/cand_1.pt SEED_OFF=300000000 GOAL_LAYOUT=sum7 CC_DEEP=exact2 \
  REMOTE=1 REMOTE_SHARDS=250 REMOTE_QUEUE_MIN=300 REMOTE_TIMEOUT_MIN=1800 \
  ITERS=2 GAMES=1000 T=3 K=3 REPLAY=2 CONFIRM_PAIRS=3000 BAR=0 \
  TRAIN_ARGS='--epochs 6 --lr 2e-5 --td-w 0.05 --rank-w 3e-4 --rank-tau 0.003' \
  BOARDGAME_DEVICE=cpu PYTHONHASHSEED=0 N_WORKERS=8
.venv/bin/python -u distill_run.py >> distill_sum7_e3/run.log 2>&1
# results to git so they survive and are visible remotely
grep -E 'iteration [0-9]+: cand|AFTER|BASELINE' distill_sum7_e3/run.log distill_sum7_e3/train_*.log > distill_sum7_e3/RESULT.txt 2>/dev/null
git add -f distill_sum7_e3/RESULT.txt distill_sum7_e3/state.json distill_sum7_e3/run.sh distill_sum7_e3/confirm_*.jsonl
git commit -qm "exact2-teacher distillation on sum7, scaled (e3): results

$(cat distill_sum7_e3/RESULT.txt | tail -12)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push -q
