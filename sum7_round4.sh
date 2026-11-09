#!/bin/zsh
# sum7 ROUND 4 (owner, 2026-11-09): ONE big distillation round on the sum7 goal
# layout, from the SHIPPED sum7 net (distill_sum7/cand_13.pt = sum7_r2_cand_13,
# model-dffd8d8f.onnx). Why: the teacher gauge on that net (2026-11-09, 2,000
# pairs) gave close-call +0.118 (95% CI +0.054..+0.182) -- the search still has
# something to teach -- while round 3's eight 600-game iterations came out level
# (per-iteration gains below what a 500-pair confirm can see). So: one iteration,
# 2,000 labelled games, judged by a 2,000-pair match against the shipped net.
#
#   cd <repo> && git pull && caffeinate -is zsh sum7_round4.sh
#
# Resumable: rerun the same command after an interruption (distill_run resumes by
# state.json; labels and confirms resume by seed). Needs gh on PATH and logged in
# (GitHub does the labelling and the match); any GitHub failure falls back to local.
# Writes sum7_r4_RESULT.txt, exports the candidate if it is clearly better, and
# commits + pushes the result (not the checkpoint) to train-features-v2.
set -e
cd "$(dirname "$0")"
export PATH=/opt/homebrew/bin:$PATH
export GOAL_LAYOUT=sum7 DISTILL_DIR=distill_sum7
D=distill_sum7
[ -f $D/state.json ] || { echo "no $D/state.json -- run this on the work iMac (the sum7 run lives there)"; exit 1; }
[ -f $D/cand_13.pt ] || { echo "no $D/cand_13.pt (the shipped sum7 net)"; exit 1; }

# Start the round from the SHIPPED net. The run's state ended round 3 on cand_22
# (level with cand_13: -0.014, 95% CI -0.082..+0.054), but the gauge was measured on
# cand_13 and cand_13 is what players have. Done once (marker 'r4'); iteration 24
# keys fresh seeds (labels 91.24M, confirm 93.24M).
.venv/bin/python - <<'PY'
import json, os
p = 'distill_sum7/state.json'
st = json.load(open(p))
if not st.get('r4'):
    assert st['iter'] == 24, f"expected the run at iteration 24 (after round 3), found {st['iter']}"
    assert st.get('goal_layout') == 'sum7', st.get('goal_layout')
    st['r4'] = {'from_champion': st['champion']}
    st['champion'] = os.path.abspath('distill_sum7/cand_13.pt')
    json.dump(st, open(p, 'w'), indent=1)
    print('round 4: champion set to', st['champion'], '(was', st['r4']['from_champion'] + ')')
else:
    print('round 4: resuming at iteration', st['iter'])
PY

# One iteration. REPLAY=3 trains on labels_22 + labels_23 (round 3, 1,200 games)
# plus the new 2,000: 3,200 games, so 4 epochs, not 8 (CLAUDE.md: big data, 3-4).
export REMOTE=1 REMOTE_SHARDS=20 REMOTE_QUEUE_MIN=${REMOTE_QUEUE_MIN:-90}
export ITERS=25 GAMES=2000 REPLAY=3 CONFIRM_PAIRS=2000 T=3 K=3 BAR=0.0
export TRAIN_ARGS='--epochs 4 --lr 2e-5 --td-w 0.5 --rank-w 0'
export BOARDGAME_DEVICE=cpu PYTHONHASHSEED=0 N_WORKERS=${N_WORKERS:-8}
.venv/bin/python -u distill_run.py >> $D/run.log 2>&1

# Verdict: the iteration's own confirm IS the decision match (cand_24 vs cand_13,
# sum7, 2,000 pairs, seeds 93.24M). Ship only if the 95% CI clears 0.
.venv/bin/python - <<'PY' > sum7_r4_RESULT.txt
import json
st = json.load(open('distill_sum7/state.json'))
h = [x for x in st['history'] if x['iter'] == 24][-1]
lo, hi = h['mean'] - 1.96 * h['se'], h['mean'] + 1.96 * h['se']
ship = lo > 0
print(f"sum7 round 4: cand_24 vs shipped cand_13 on sum7, {h['pairs']} pairs: "
      f"{h['mean']:+.3f} (95% CI {lo:+.3f}..{hi:+.3f})")
print('VERDICT:', 'SHIP (CI clear of 0) -- exported to export/sum7_r4/' if ship
      else 'DO NOT SHIP (CI includes 0) -- the shipped net stays')
open('/tmp/sum7_r4_ship', 'w').write('1' if ship else '0')
PY
cat sum7_r4_RESULT.txt
if [ "$(cat /tmp/sum7_r4_ship)" = 1 ]; then
    mkdir -p export/sum7_r4
    GOAL_LAYOUT=sum7 BOARDGAME_DEVICE=cpu .venv/bin/python onnx_export.py $D/cand_24.pt export/sum7_r4/model_sum7.onnx \
        >> sum7_r4_RESULT.txt 2>&1
    ls export/sum7_r4 >> sum7_r4_RESULT.txt
fi
git add -f sum7_r4_RESULT.txt $D/run.log
git commit -m "sum7 round 4 result: $(head -1 sum7_r4_RESULT.txt | cut -c1-120)" || true
git push origin train-features-v2 || echo "push failed -- commit is local"
