#!/bin/zsh
# Layout comparison PASS 2 (owner, 2026-11-06; runs unattended for a week):
# layout_play.py records (per turn: 21-roll values for luck, move gaps, captures,
# walls, seals). Per layout: self = 1,500 full-strength self-play games (SELF_ONLY),
# hcap = 1,000 pairs full vs handicapped. GitHub first (20 shards), then the local
# script fills whatever is missing (resumes by seed). Then the reports, committed.
cd "$(dirname "$0")/.."
export PATH=/opt/homebrew/bin:$PATH REMOTE_QUEUE_MIN=360 REMOTE_TIMEOUT_MIN=720
run() {  # layout net mode pairs seed
  local L=$1 NET=$2 M=$3 P=$4 S=$5 W=0 SO=0; [[ $M == hcap ]] && W=1; [[ $M == self ]] && SO=1
  local OUT=layout_cmp/p2_${L}_${M}.jsonl LOG=layout_cmp/p2_${L}_${M}.log
  local GL=$L; [[ $L == today ]] && GL=
  .venv/bin/python -u -c "
import remote_play as r
env={'LAYOUT_STATS':'2','WEAK_B':'$W','SELF_ONLY':'$SO'}
if '$GL': env['GOAL_LAYOUT']='$GL'
try:
    print('missing', r.remote_play('confirm',['$NET','$NET'],$P,$S,'$OUT',env=env,shards=20))
except Exception as e:
    print('remote failed:', e)
" >> $LOG 2>&1
  # local fill (plays only what is missing)
  GOAL_LAYOUT=$GL LAYOUT_STATS=2 WEAK_B=$W SELF_ONLY=$SO SEED_BASE=$S N_WORKERS=8 BOARDGAME_DEVICE=cpu \
    PYTHONHASHSEED=0 nice -n 10 .venv/bin/python -u confirm_match.py $NET $NET $P $OUT >> $LOG 2>&1
  echo "done $L $M $(wc -l < $OUT) records" >> $LOG
}
for spec in "today distill_cand13.pt 240" "sum7 distill_sum7/cand_13.pt 244" "B distill_cand13.pt 248"; do
  set -- ${=spec}
  run $1 $2 self 1500 ${3}000000 &
  run $1 $2 hcap 1000 $(( $3 + 2 ))000000 &
  wait
  echo "$(date) layout $1 done" >> layout_cmp/run2_driver.log
done
.venv/bin/python layout_cmp/analyze2.py > layout_cmp/REPORT2.txt 2>&1
.venv/bin/python layout_cmp/analyze.py > layout_cmp/REPORT1.txt 2>&1
for f in layout_cmp/p2_*.jsonl layout_cmp/today_*.jsonl layout_cmp/sum7_*.jsonl layout_cmp/B_*.jsonl; do
  [ -f $f ] && gzip -kf $f
done
git add -f layout_cmp/REPORT*.txt layout_cmp/*.jsonl.gz layout_cmp/*.log
git commit -qm "layout comparison: pass 1 + 2 results (automatic, run2.sh)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin train-features-v2
echo "$(date) ALL DONE" >> layout_cmp/run2_driver.log
