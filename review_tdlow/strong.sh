#!/bin/zsh
cd /Users/tomrecht/quahuru-train
export PATH=/opt/homebrew/bin:$PATH REMOTE_QUEUE_MIN=240 REMOTE_TIMEOUT_MIN=600
.venv/bin/python -u -c "
import remote_play as R
R.remote_play('difficulty', ['rankexp/tdlow_rank.pt'], 600, 360000000, 'review_tdlow/strong.jsonl',
              env={'LEVELS': '0.84,0.86,0.88,0.90', 'GOAL_LAYOUT': 'sum7'})"
LEVELS=0.84,0.86,0.88,0.90 GOAL_LAYOUT=sum7 .venv/bin/python difficulty_fine.py analyze review_tdlow/strong.jsonl > review_tdlow/STRONG.txt 2>&1
git add -f review_tdlow/strong.jsonl review_tdlow/STRONG.txt review_tdlow/strong.sh
git commit -qm "tdlow_rank difficulty sweep 0.84-0.90 (Strong knot re-spacing)

$(cat review_tdlow/STRONG.txt | tail -12)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push -q
