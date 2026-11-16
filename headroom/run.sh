#!/bin/zsh
# Headroom oracle (owner, 2026-11-16): how much does a deeper, noise-free search
# gain over the shipped sum7 net? close-call T=3 K=3 on distill_sum7/cand_13 vs
# the same net plain, GOAL_LAYOUT=sum7, seeds 290M.. (1,000 pairs):
#   cc1      candidates rescored by 1 opponent turn, all 21 rolls  (~20-40 min)
#   e2_0..4  rescored by TWO turns, all 21 x 21 rolls (CC_DEEP=exact2), 200 pairs
#            per dispatch on 40 shards (5 pairs, ~3 h on GitHub's 2x-slower cores;
#            20 shards x 10 pairs overran the 6 h job limit), two waves of 20
# Then headroom/RESULT.txt, committed. Resumable: finished files are skipped.
cd /Users/tomrecht/quahuru-train
export PATH=/opt/homebrew/bin:/usr/local/bin:$PATH REMOTE_TIMEOUT_MIN=900 REMOTE_QUEUE_MIN=120
PY=.venv/bin/python NET=distill_sum7/cand_13.pt
rp() {  # mode-out pairs seed env-json
  $PY - "$@" <<'PYEOF'
import sys, json; sys.argv = [sys.argv[0]] + sys.argv[1:]
import remote_play as r
out, n, seed, env = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), json.loads(sys.argv[4])
miss = r.remote_play('ccmatch', ['distill_sum7/cand_13.pt'], n, seed, out, env=env)
print('missing', miss, flush=True)
PYEOF
}
[ -s headroom/cc1.jsonl ] || rp headroom/cc1.jsonl 1000 290000000 '{"T":"3","K":"3","GOAL_LAYOUT":"sum7"}'
$PY close_call.py analyze headroom/cc1.jsonl
for i in 0 1 2 3 4; do
  [ -s headroom/e2_$i.jsonl ] && continue
  REMOTE_SHARDS=40 rp headroom/e2_$i.jsonl 200 $((290000000 + 200 * i)) '{"T":"3","K":"3","GOAL_LAYOUT":"sum7","CC_DEEP":"exact2"}'
  $PY close_call.py analyze headroom/e2_$i.jsonl
  $PY headroom/analyze.py
done
{ date; $PY headroom/analyze.py; for f in headroom/cc1.jsonl headroom/e2_*.jsonl; do $PY close_call.py analyze $f; done; } > headroom/RESULT.txt 2>&1
cat headroom/RESULT.txt
git add -f headroom/RESULT.txt headroom/*.jsonl && git commit -m "Headroom oracle result (exact 2-turn search vs 1-ply vs plain, sum7 net)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push
