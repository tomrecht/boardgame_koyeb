#!/bin/zsh
# sum7 ROUND 4 (owner, 2026-11-09): big distillation iterations on the sum7 goal
# layout, from the SHIPPED sum7 net (distill_sum7/cand_13.pt = sum7_r2_cand_13,
# model-dffd8d8f.onnx), CONTINUING WHILE EACH ONE CLEARLY WINS.
# Why: the teacher gauge on that net (2026-11-09, 2,000 pairs) gave close-call
# +0.118 (95% CI +0.054..+0.182) -- the search still has something to teach --
# while round 3's eight 600-game iterations came out level (their gains were below
# what a 500-pair confirm can see, and BAR=0 promotions drifted).
#
# Each iteration: 2,000 games labelled on GitHub, training here (last 3 label
# files, 4 epochs), then a 2,000-pair match vs its parent on GitHub. It is kept
# only if that match's 95% CI clears 0; otherwise the parent stays champion and
# the run STOPS. At most MAX_ITERS (default 5) iterations. If anything was kept,
# a final 2,000-pair head to head (final champion vs the shipped cand_13, fresh
# seeds) decides shipping, and close-call's gauge on the final champion says
# whether another round is worth it.
#
#   cd <repo> && git pull && caffeinate -is zsh sum7_round4.sh
#
# Resumable: rerun the same command (distill_run resumes from state.json; labels
# and matches resume by seed; finished iterations are not redone). Needs gh on
# PATH and logged in; any GitHub failure falls back to local. Writes
# sum7_r4_RESULT.txt, exports the final champion if it is clearly better than
# the shipped net, and commits + pushes the result (not checkpoints).
set -e
cd "$(dirname "$0")"
export PATH=/opt/homebrew/bin:$PATH
export GOAL_LAYOUT=sum7 DISTILL_DIR=distill_sum7
D=distill_sum7
PY=.venv/bin/python
MAX_ITERS=${MAX_ITERS:-5}
FIRST=24                        # round 3 ended after iteration 23
[ -f $D/state.json ] || { echo "no $D/state.json -- run this on the work iMac (the sum7 run lives there)"; exit 1; }
[ -f $D/cand_13.pt ] || { echo "no $D/cand_13.pt (the shipped sum7 net)"; exit 1; }

# Start from the SHIPPED net. Round 3 ended on cand_22 (level with cand_13: -0.014,
# 95% CI -0.082..+0.054), but the gauge was measured on cand_13 and cand_13 is what
# players have. Done once (marker 'r4').
$PY - <<'PY'
import json, os
p = 'distill_sum7/state.json'
st = json.load(open(p))
if not st.get('r4'):
    assert st['iter'] == 24, f"expected the run at iteration 24 (after round 3), found {st['iter']}"
    assert st.get('goal_layout') == 'sum7', st.get('goal_layout')
    st['r4'] = {'from_champion': st['champion'], 'start': os.path.abspath('distill_sum7/cand_13.pt'),
                'stopped': False}
    st['champion'] = st['r4']['start']
    json.dump(st, open(p, 'w'), indent=1)
    print('round 4: champion set to', st['champion'], '(was', st['r4']['from_champion'] + ')')
else:
    print('round 4: resuming at iteration', st['iter'], '-- stopped' if st['r4'].get('stopped') else '')
PY

export REMOTE=1 REMOTE_SHARDS=20 REMOTE_QUEUE_MIN=${REMOTE_QUEUE_MIN:-90}
export GAMES=2000 REPLAY=3 CONFIRM_PAIRS=2000 T=3 K=3 BAR=0.0
export TRAIN_ARGS='--epochs 4 --lr 2e-5 --td-w 0.5 --rank-w 0'
export BOARDGAME_DEVICE=cpu PYTHONHASHSEED=0 N_WORKERS=${N_WORKERS:-8}

for (( k = FIRST; k < FIRST + MAX_ITERS; k++ )); do
    STOP=$($PY -c "import json; print(int(json.load(open('$D/state.json'))['r4'].get('stopped', False)))")
    [ "$STOP" = 1 ] && break
    ITERS=$((k + 1)) $PY -u distill_run.py >> $D/run.log 2>&1
    # distill_run promotes on mean > 0; round 4 keeps an iteration only if its
    # 95% CI clears 0, else reverts to the parent and stops.
    $PY - "$k" <<'PY'
import json, sys
k = int(sys.argv[1]); p = 'distill_sum7/state.json'
st = json.load(open(p))
h = [x for x in st['history'] if x['iter'] == k][-1]
lo = h['mean'] - 1.96 * h['se']
kept = lo > 0
if not kept:
    st['champion'] = h['champion']          # the parent
    st['r4']['stopped'] = True
st['r4'].setdefault('iters', {})[str(k)] = {'mean': h['mean'], 'lo': lo,
                                            'hi': h['mean'] + 1.96 * h['se'], 'kept': kept}
json.dump(st, open(p, 'w'), indent=1)
print(f"round 4 iteration {k}: {h['mean']:+.3f} (95% CI {lo:+.3f}..{h['mean'] + 1.96 * h['se']:+.3f}) "
      f"-> {'KEPT, continuing' if kept else 'not clear of 0: parent stays champion, STOP'}", flush=True)
PY
done

# Final: if anything was kept, the final champion vs the shipped net directly
# (chained parent-vs-child wins can overstate the total), then the teacher gauge.
CH=$($PY -c "import json; print(json.load(open('$D/state.json'))['champion'])")
START=$($PY -c "import json; print(json.load(open('$D/state.json'))['r4']['start'])")
mkdir -p layout_final
if [ "$CH" != "$START" ]; then
    $PY -u -c "
import sys; sys.argv=['x']; import remote_play as r, subprocess
out='layout_final/sum7_r4_final.jsonl'
r.remote_play('confirm',['$CH','$START'],2000,274000000,out,env={'GOAL_LAYOUT':'sum7'})
subprocess.call([sys.executable,'confirm_match.py','analyze',out])
" > layout_final/sum7_r4_final.log 2>&1 || true
    $PY -u -c "
import sys; sys.argv=['x']; import remote_play as r, subprocess
out='layout_final/sum7_r4_gauge.jsonl'
r.remote_play('ccmatch',['$CH'],2000,276000000,out,env={'T':'3','K':'3','MAX_REPLIES':'3','GOAL_LAYOUT':'sum7'})
subprocess.call([sys.executable,'close_call.py','analyze',out])
" > layout_final/sum7_r4_gauge.log 2>&1 || true
fi

$PY - "$CH" "$START" <<'PY' > sum7_r4_RESULT.txt
import json, os, sys
ch, start = sys.argv[1], sys.argv[2]
st = json.load(open('distill_sum7/state.json'))
print('sum7 round 4 (2,000-game iterations, kept only if 95% CI vs parent clears 0):')
for k, v in sorted(st['r4'].get('iters', {}).items(), key=lambda kv: int(kv[0])):
    print(f"  iteration {k}: {v['mean']:+.3f} (95% CI {v['lo']:+.3f}..{v['hi']:+.3f}) {'KEPT' if v['kept'] else 'not kept'}")
ship = False
if ch == start:
    print('VERDICT: nothing kept -- the shipped net (cand_13) stays; sum7 distillation with this teacher is done.')
else:
    fin = 'layout_final/sum7_r4_final.log'
    last = [l for l in open(fin)] if os.path.exists(fin) else []
    line = next((l.strip() for l in reversed(last) if 'paired' in l or 'CI' in l), 'final match: see ' + fin)
    print('final champion', os.path.basename(ch), 'vs shipped cand_13 (2,000 pairs, seeds 274M):', line)
    import re
    m = re.search(r'95% CI ([+-][\d.]+)', line)
    ship = bool(m and float(m.group(1)) > 0)
    print('VERDICT:', 'SHIP -- exported to export/sum7_r4/' if ship
          else 'DO NOT SHIP (final match not clear of 0) -- the shipped net stays')
    g = 'layout_final/sum7_r4_gauge.log'
    if os.path.exists(g):
        gl = [l.strip() for l in open(g) if 'close-call agent' in l]
        print('teacher gauge on the final champion:', gl[-1] if gl else 'see ' + g)
open('/tmp/sum7_r4_ship', 'w').write(ch if ship else '')
PY
cat sum7_r4_RESULT.txt
SHIP=$(cat /tmp/sum7_r4_ship)
if [ -n "$SHIP" ]; then
    mkdir -p export/sum7_r4
    GOAL_LAYOUT=sum7 BOARDGAME_DEVICE=cpu $PY onnx_export.py "$SHIP" export/sum7_r4/model_sum7.onnx \
        >> sum7_r4_RESULT.txt 2>&1
    ls export/sum7_r4 >> sum7_r4_RESULT.txt
fi
git add -f sum7_r4_RESULT.txt $D/run.log layout_final/sum7_r4_*.log 2>/dev/null || git add -f sum7_r4_RESULT.txt $D/run.log
git commit -m "sum7 round 4 result: $(grep VERDICT sum7_r4_RESULT.txt | cut -c1-110)" || true
git push origin train-features-v2 || echo "push failed -- commit is local"
