#!/bin/zsh
# CAPACITY TEST (2026-10-23): is the 64-wide student short of capacity?
# Same distillation labels (all of distill/labels_*.jsonl.gz: round 1 + the
# weekend), same held-out games (split by game, --seed 0), three nets trained
# FROM SCRATCH: 64x6 (today's shape), 128x6 (wider), 128x9 (wider + deeper),
# plus today's net warm-started as the real loop does. Reads: held-out MSE to
# the search labels, within-decision corr, top-1 agreement (distill_train's
# AFTER line). If 128 fits clearly better than 64 -> capacity-limited: a bigger
# teacher (or phone net) is worth it. If level -> the labels/data are the limit.
# Waits for the weekend run first. Launch on the work Mac:
#   nohup zsh closecall/fit_test.sh > closecall/fit_test.log 2>&1 &
cd "${0:A:h}/.."
while pgrep -f closecall/weekend.sh > /dev/null; do sleep 300; done
echo "== start $(date)"
DATA=(distill/labels_*.jsonl.gz)
echo "data: ${#DATA} files"
# Training costs ~4 ms per record per epoch on CPU (64 wide; 128 wide ~4x), so
# the test uses a game SAMPLE (same games for every arm): ~2,000 games ~250k
# records x 8 epochs -- an evening on the work Mac, not days.
NW=${N_WORKERS:-8}
COMMON=(--data $DATA --max-games ${GAMES:-2000} --epochs ${EPOCHS:-8} --lr ${LR:-5e-4} --batch-size 256
        --td-w 0.5 --seed 0 --workers $NW --log closecall/fit_test.jsonl)
export BOARDGAME_DEVICE=cpu PYTHONHASHSEED=0
for HL in "64 6" "128 6" "128 9"; do
  set -- ${=HL}
  echo "== scratch $1 x $2 $(date)"
  .venv/bin/python -u distill_train.py --scratch AB2 --hidden $1 --layers $2 \
      --save closecall/fit_$1x$2.pt $COMMON
done
echo "== warm start from distill_cand3 (the real loop's step) $(date)"
.venv/bin/python -u distill_train.py --load distill_cand3.pt --save closecall/fit_warm.pt \
    --data $DATA --max-games ${GAMES:-2000} --epochs 8 --lr 2e-5 --batch-size 256 --td-w 0.5 --seed 0 \
    --workers $NW --log closecall/fit_test.jsonl
.venv/bin/python - <<'PY'
import json
print('\narm                         held-out: mse      corr-within  top-1')
for l in open('closecall/fit_test.jsonl'):
    r = json.loads(l); a = r['after']; arg = r['args']
    name = (f"scratch {arg['hidden']}x{arg['layers']}" if arg.get('scratch') else 'warm (distill_cand3)')
    print(f"{name:27s} {a.get('mse', float('nan')):.5f}   {a.get('within_corr', float('nan')):+.3f}       {a.get('top_agree', float('nan')):.3f}")
PY
echo "== FIT TEST DONE $(date)"
