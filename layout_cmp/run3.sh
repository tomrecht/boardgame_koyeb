#!/bin/zsh
# Seal evenness, sum7 vs B (pass 2 left it unresolved: max-pair share -0.018, CI -0.043..+0.006).
# 4,500 more self-play games each (LAYOUT_STATS=1, white only), fresh seeds. Then seal_boot.py.
cd "$(dirname "$0")/.."
export PATH=/opt/homebrew/bin:$PATH REMOTE_QUEUE_MIN=360 REMOTE_TIMEOUT_MIN=720
run() {  # layout net seed
  .venv/bin/python -u -c "
import remote_play as r
env={'LAYOUT_STATS':'1','WEAK_B':'0','SELF_ONLY':'1','GOAL_LAYOUT':'$1'}
print('missing',r.remote_play('confirm',['$2','$2'],4500,$3,'layout_cmp/p3_$1_self.jsonl',env=env,shards=20))
" > layout_cmp/p3_$1_self.log 2>&1
  SEED_BASE=$3 PYTHONHASHSEED=0 N_WORKERS=8 LAYOUT_STATS=1 WEAK_B=0 SELF_ONLY=1 GOAL_LAYOUT=$1 nice -n 10 \
    .venv/bin/python -u confirm_match.py $2 $2 4500 layout_cmp/p3_$1_self.jsonl >> layout_cmp/p3_$1_self.log 2>&1
}
run sum7 distill_sum7/cand_13.pt 260000000 &
run B distill_cand13.pt 262000000 &
wait
.venv/bin/python layout_cmp/seal_boot.py > layout_cmp/REPORT3.txt 2>&1
gzip -kf layout_cmp/p3_*_self.jsonl
git add -f layout_cmp/run3.sh layout_cmp/seal_boot.py layout_cmp/REPORT3.txt layout_cmp/p3_*_self.jsonl.gz layout_cmp/p3_*_self.log
git commit -qm "layout comparison pass 3: seal evenness sum7 vs B (automatic, run3.sh)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin HEAD
date >> layout_cmp/run3_driver.log; echo ALL DONE >> layout_cmp/run3_driver.log
