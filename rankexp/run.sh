#!/bin/zsh
# Loss-weighting test (2026-11-16): round 4's training (td-w 0.5) had TD loss ~50x the
# distill loss, so the noisy TD chain dominated. Same data/start/settings as round 4
# iteration 24 (labels_22-24, warm start distill_sum7/cand_13, 4 epochs lr 2e-5);
# baseline fit (r4, td-w 0.5): within-decision corr 0.219 -> 0.262, top-1 55.9 -> 55.7%.
cd /Users/tomrecht/quahuru-train
export GOAL_LAYOUT=sum7 BOARDGAME_DEVICE=cpu PYTHONHASHSEED=0 OMP_NUM_THREADS=3 N_WORKERS=3
D=distill_sum7; F=($D/labels_22.jsonl.gz $D/labels_23.jsonl.gz $D/labels_24.jsonl.gz)
arm() { .venv/bin/python -u distill_train.py --data $F --load $D/cand_13.pt --save rankexp/$1.pt \
          --epochs 4 --lr 2e-5 ${@:2} > rankexp/$1.log 2>&1; }
arm tdlow  --td-w 0.05 &
arm tdlow_rank --td-w 0.05 --rank-w 3e-4 --rank-tau 0.003 &
arm rank   --td-w 0.5  --rank-w 3e-4 --rank-tau 0.003 &
wait
grep -H "AFTER" rankexp/*.log
