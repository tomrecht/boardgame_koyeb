#!/bin/bash
# Overnight phase 2 (2026-10-05/06): after the blend sweep -> confirm the winner
# on FRESH seeds vs the champion (1,000 pairs) -> winner vs b50_50 head to head
# if different (500 fresh pairs) -> play-time ensemble champion+iter10 vs the
# champion (300 pairs). Every match appends and resumes.
cd "$(dirname "$0")"
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
echo "$(stamp) waiting for the sweep"
while pgrep -f "sweep.sh" >/dev/null; do sleep 60; done
WIN=$(python3 - <<'PY'
import json, statistics as st
best=None
for b in ['b75_25','b50_50','b25_75','b3way','b4way']:
    try: rows=[json.loads(l) for l in open(f'match_sw_{b}.jsonl')]
    except FileNotFoundError: continue
    by={}
    for r in rows: by.setdefault(r['seed'],[]).append(r['a_margin'])
    p=[st.fmean(v) for v in by.values() if len(v)==2]
    m=st.fmean(p)
    if best is None or m>best[1]: best=(b,m)
print(best[0])
PY
)
echo "$(stamp) sweep winner: $WIN"
echo "$WIN" > blends/winner.txt
echo "$(stamp) confirmation: $WIN vs champion, 1000 fresh pairs"
MODEL_A=blends/$WIN.onnx MODEL_B=model.onnx TAG=confirm_$WIN SEED_BASE=7300000 N_WORKERS=4 \
  python3 match_models.py 1000 > blends/confirm_$WIN.log 2>&1
if [ "$WIN" != "b50_50" ]; then
  echo "$(stamp) head to head: $WIN vs b50_50, 500 fresh pairs"
  MODEL_A=blends/$WIN.onnx MODEL_B=blends/b50_50.onnx TAG=h2h_${WIN}_b50 SEED_BASE=7400000 N_WORKERS=4 \
    python3 match_models.py 500 > blends/h2h_$WIN.log 2>&1
fi
echo "$(stamp) ensemble champion+iter10 vs champion, 300 pairs"
MODEL_A="ens:symaug_iter6.pt+td_champion_July18_iter10.pt" MODEL_B=model.onnx TAG=ens_c_i10 SEED_BASE=7500000 N_WORKERS=4 \
  python3 match_models.py 300 > blends/ens_c_i10.log 2>&1
echo "$(stamp) phase 2 done"
