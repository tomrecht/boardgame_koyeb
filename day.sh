#!/bin/bash
# Day queue (2026-10-05): after the prefilter arena -> difficulty fine sweep ->
# calibration of the older champions -> training-loop smoke run (features v2,
# with the rebuilt start pool and the calibration benchmark). Each step logs;
# a failing step does not stop the later ones (they are independent).
cd "$(dirname "$0")"
stamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "$(stamp) waiting for the prefilter arena"
while pgrep -f "match_scales.py 1500" >/dev/null; do sleep 60; done
echo "$(stamp) arena done"

echo "$(stamp) difficulty fine sweep"
N_WORKERS=4 python3 difficulty_fine.py 60 > difficulty_fine.log 2>&1 || echo "$(stamp) difficulty sweep FAILED"
echo "$(stamp) difficulty sweep done"

echo "$(stamp) calibration of older nets"
python3 calib_bench.py symaug_iter6.pt symaug_almostchamp_July27_iter11.pt \
  td_champion_July19_iter14.pt td_champion_July18_iter10.pt td_champion_July17_iter4.pt \
  td_champion_July21_aux_iter14.pt best_iter5_m46.pt symaug6_AB.pt > calib_older.log 2>&1 \
  || echo "$(stamp) calibration FAILED"
echo "$(stamp) calibration done"

echo "$(stamp) training-loop smoke run (AB)"
rm -rf smoke_ab2 && mkdir -p smoke_ab2
BOARDGAME_DEVICE=cpu PYTHONHASHSEED=0 SMOKE=1 PREFIX=smoke_ab2/ab WARM_START=symaug6_AB.pt \
  N_WORKERS=4 PRESCREEN_BAR=-12 python3 -u league_run.py > smoke_ab2/smoke.log 2>&1 \
  || echo "$(stamp) smoke run FAILED"
echo "$(stamp) all done"
