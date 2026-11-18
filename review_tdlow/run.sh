#!/bin/zsh
# tdlow_rank ship checklist steps 2-3 on the work iMac (sum7 games only).
cd /Users/tomrecht/quahuru-train
export GOAL_LAYOUT=sum7 BOARDGAME_DEVICE=cpu PYTHONHASHSEED=0 N_WORKERS=4
NET=rankexp/tdlow_rank.pt
.venv/bin/python -u filter_probe.py $NET sum7_r2_cand_13.pt,sum7_cand_7.pt review_tdlow/filter.jsonl 300 > review_tdlow/filter.log 2>&1 &
F=$!
.venv/bin/python -u rule_probe.py $NET review_tdlow/rule.jsonl > review_tdlow/rule.log 2>&1 &
R=$!
wait $F $R
.venv/bin/python review_tdlow/analyze.py review_tdlow/filter.jsonl review_tdlow/rule.jsonl > review_tdlow/RESULT.txt 2>&1
git add -f review_tdlow/RESULT.txt review_tdlow/*.jsonl review_tdlow/*.py review_tdlow/run.sh
git commit -qm "tdlow_rank checklist: filter_probe + rule_probe on sum7 games

$(cat review_tdlow/RESULT.txt)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push -q
