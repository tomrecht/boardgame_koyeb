#!/bin/zsh
# Weekend 2026-10-23..26: continue distill_run (state iter 6, champion cand_3) for
# 8 more iterations with more data and a stronger gate, then check the final
# champion against the LIVE net (distill_cand3) on 1,000 fresh pairs.
cd "${0:A:h}/.."
ITERS=14 GAMES=600 REPLAY=3 CONFIRM_PAIRS=500 BOARDGAME_DEVICE=cpu PYTHONHASHSEED=0 N_WORKERS=8 \
    caffeinate -is .venv/bin/python -u distill_run.py >> distill/run.log 2>&1
echo "== distill done $(date)"
CH=$(.venv/bin/python -c "import json; print(json.load(open('distill/state.json'))['champion'])")
echo "final champion: $CH"
cp "$CH" distill/weekend_champion.pt
SEED_BASE=100000000 N_WORKERS=8 PYTHONHASHSEED=0 caffeinate -is .venv/bin/python -u \
    confirm_match.py distill/weekend_champion.pt distill_cand3.pt 1000 closecall/weekend_vs_live.jsonl \
    > closecall/weekend_vs_live.log 2>&1
tail -1 closecall/weekend_vs_live.log
echo "WEEKEND DONE $(date)"
