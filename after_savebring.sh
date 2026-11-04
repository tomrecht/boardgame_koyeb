#!/bin/zsh
# Owner 2026-11-04: once the save-vs-bring playouts finish, run distillation on
# today's layout (iterations 14-21 from cand_13, final check vs the live net)
# and sum7 round 3 (iterations 16-23), side by side.
cd "$(dirname "$0")"
until grep -q "play done" probe_savebring/run.log; do sleep 60; done
TAG=r1 caffeinate -is zsh layout_round2.sh today 22 210000000 > layout_final/today_r1_driver.log 2>&1 &
sleep 120
TAG=r3 caffeinate -is zsh layout_round2.sh sum7 24 208000000 > layout_final/sum7_r3_driver.log 2>&1 &
wait
