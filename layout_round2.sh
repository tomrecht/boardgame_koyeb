#!/bin/zsh
# Round 2 for a goal layout (owner 2026-11-02): continue distill_run to ITERS
# (default 16), then the final check vs cand13 on GitHub (2,000 pairs), all in
# one shell chain so it does not depend on a Claude session staying open.
#   zsh layout_round2.sh sum7 [iters] [seed_base]
set -e
L=${1:?layout}; N=${2:-16}; SB=${3:-204000000}
cd "$(dirname "$0")"
# L=today: today's goal layout, run dir distill/ (GOAL_LAYOUT unset)
if [ "$L" = today ]; then DIR=distill; unset GOAL_LAYOUT; LENV="{}"
else DIR=distill_$L; export GOAL_LAYOUT=$L; LENV="{'GOAL_LAYOUT':'$L'}"; fi
TAG=${TAG:-r2}
export PATH=/opt/homebrew/bin:$PATH DISTILL_DIR=$DIR
export REMOTE=1 REMOTE_SHARDS=10 REMOTE_QUEUE_MIN=${REMOTE_QUEUE_MIN:-90}
export ITERS=$N GAMES=600 REPLAY=3 CONFIRM_PAIRS=500 T=3 K=3
export TRAIN_ARGS='--epochs 8 --lr 2e-5 --td-w 0.5 --rank-w 0'
export BOARDGAME_DEVICE=cpu PYTHONHASHSEED=0 N_WORKERS=4
.venv/bin/python -u distill_run.py >> $DIR/run.log 2>&1
CH=$(.venv/bin/python -c "import json;print(json.load(open('$DIR/state.json'))['champion'])")
mkdir -p layout_final
.venv/bin/python -u -c "
import sys; sys.argv=['x']; import remote_play as r, subprocess
out='layout_final/${L}_${TAG}.jsonl'
miss=r.remote_play('confirm',['$CH','distill_cand13.pt'],2000,$SB,out,env=$LENV,shards=10)
print('champion $CH; missing',miss,flush=True)
subprocess.call([sys.executable,'confirm_match.py','analyze',out])
" > layout_final/${L}_${TAG}.log 2>&1
