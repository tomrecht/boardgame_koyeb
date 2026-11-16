#!/bin/zsh
# tdlow_rank (td-w 0.05, rank-w 3e-4, tau 0.003) vs the shipped sum7 net, 2,000 pairs, sum7, seeds 300M
cd /Users/tomrecht/quahuru-train
export PATH=/opt/homebrew/bin:$PATH REMOTE_QUEUE_MIN=240 REMOTE_TIMEOUT_MIN=600
.venv/bin/python - <<'PY'
import remote_play as r
r.remote_play('confirm', ['rankexp/tdlow_rank.pt', 'distill_sum7/cand_13.pt'], 2000, 300000000,
              'rankexp/match_tdlow_rank.jsonl', env={'GOAL_LAYOUT': 'sum7'})
PY
.venv/bin/python confirm_match.py analyze rankexp/match_tdlow_rank.jsonl | tee rankexp/MATCH.txt
