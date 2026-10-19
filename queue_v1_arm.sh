#!/bin/zsh
# queue_v1_arm.sh -- start the v1 AFTERSTATE control arm (fv2V1) once fv2C ends.
#
# The control for fv2C: identical settings and seeds, but warm-started from the
# plain v1 blend (no new inputs), so fv2V1 vs fv2C separates the new inputs from
# afterstate training + the rest of the pipeline (CLAUDE.md, "fv2C RESULT").
# fv2C promoted at iteration 55 under LR 5e-5 / 8 epochs, so this arm uses those
# and runs 80 iterations.
#
# Waits for the fv2C process to exit. To end fv2C early (and so start this
# sooner):  pkill -f "league_run.py"  -- the queued launch follows within a minute.
#
#     nohup ./queue_v1_arm.sh > runs/queue_v1_arm.log 2>&1 &
cd "${0:A:h}"
# STOP_AT (default 92): end fv2C once that iteration is done (the smaller-step
# restart at iter 80 has had a fair test by then); 0 = wait for it to end itself.
STOP_AT=${STOP_AT:-92}
while pgrep -f "PREFIX=runs/fv2C " > /dev/null; do
    if [[ $STOP_AT -gt 0 ]] && grep -q "^  iter $STOP_AT done" runs/fv2C.log; then
        echo "fv2C iter $STOP_AT done $(date); stopping it"
        pkill -f "league_run.py"
        sleep 60
        continue
    fi
    sleep 60
done
echo "fv2C ended $(date); starting fv2V1"
sleep 30   # let the old pool's workers exit
exec caffeinate -is env PATH="$PWD/.venv/bin:$PATH" BOARDGAME_DEVICE=mps PYTHONHASHSEED=0 \
    ITERS=80 LR=5e-5 EPOCHS=8 PREFIX=runs/fv2V1 WARM_START=blend.pt \
    DUMP_RECORDS=runs/fv2V1_records.jsonl.gz \
    python -u league_run.py >> runs/fv2V1.log 2>&1
