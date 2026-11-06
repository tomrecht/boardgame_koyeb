#!/bin/zsh
# Layout comparison (owner, 2026-11-06): today / sum7 / B, each with its own net.
#   self  = net vs itself, full strength (same seed + same agent => the colour-swapped
#           twin game is identical, so 1,500 "pairs" = 1,500 distinct games)
#   hcap  = net vs the same net playing select_move_pair_fast (WEAK_B=1), 1,000 pairs
# Every game carries its trajectory (LAYOUT_STATS=1). Analysis: layout_cmp/analyze.py
cd "$(dirname "$0")/.."
export PATH=/opt/homebrew/bin:$PATH REMOTE_QUEUE_MIN=240 REMOTE_TIMEOUT_MIN=480
run() {  # layout net mode pairs seed
  local L=$1 NET=$2 M=$3 P=$4 S=$5 W=0; [[ $M == hcap ]] && W=1
  .venv/bin/python -u -c "
import remote_play as r, subprocess, sys
out='layout_cmp/${L}_${M}.jsonl'
env={'LAYOUT_STATS':'1','WEAK_B':'$W'}
if '$L'!='today': env['GOAL_LAYOUT']='$L'
miss=r.remote_play('confirm',['$NET','$NET'],$P,$S,out,env=env,shards=10)
print('missing',miss)
subprocess.call([sys.executable,'confirm_match.py','analyze',out])
" > layout_cmp/${L}_${M}.log 2>&1
}
run today distill_cand13.pt self 1500 220000000 &
run sum7 distill_sum7/cand_13.pt self 1500 222000000 &
run B distill_cand13.pt self 1500 224000000 &
run today distill_cand13.pt hcap 1000 226000000 &
run sum7 distill_sum7/cand_13.pt hcap 1000 228000000 &
run B distill_cand13.pt hcap 1000 230000000 &
wait
echo ALL DONE > layout_cmp/done.txt
