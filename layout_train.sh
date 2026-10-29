#!/bin/zsh
# Distillation on an alternative goal layout (goal_layout.py), owner 2026-10-29.
#   zsh layout_train.sh sum7      (or B)   -- run from the repo, work iMac
# Warm start = the live net (distill_cand13.pt); close-call labels + confirms on
# GitHub (10 shards, so sum7 and B can run AT THE SAME TIME within GitHub's 20
# concurrent jobs), training here. Run dir distill_<layout>/ (state.json,
# run.log); rerun the same command to resume. Iteration 0's confirm = the first
# layout-trained net vs cand_13 on that layout.
set -e
L=${1:?layout: sum7 or B}
mkdir -p distill_$L
export PATH=/opt/homebrew/bin:$PATH          # gh, for REMOTE=1
GOAL_LAYOUT=$L DISTILL_DIR=distill_$L START=${START:-distill_cand13.pt} \
REMOTE=${REMOTE:-1} REMOTE_SHARDS=${REMOTE_SHARDS:-10} \
ITERS=${ITERS:-8} GAMES=${GAMES:-600} REPLAY=${REPLAY:-3} CONFIRM_PAIRS=${CONFIRM_PAIRS:-500} \
T=${T:-3} K=${K:-3} TRAIN_ARGS=${TRAIN_ARGS:-'--epochs 8 --lr 2e-5 --td-w 0.5 --rank-w 0'} \
BOARDGAME_DEVICE=cpu PYTHONHASHSEED=0 N_WORKERS=${N_WORKERS:-4} \
  caffeinate -is nohup .venv/bin/python -u distill_run.py >> distill_$L/run.log 2>&1 &
echo "started $L: tail -f distill_$L/run.log"
