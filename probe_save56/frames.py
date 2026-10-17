"""Net's bring - save (B - A) in the search frame vs the trained frame, on the
36 positions with playout results; correlate each with the playouts."""
import sys, os, json, statistics as st
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
os.environ.setdefault('NET', os.path.join(os.getcwd(), 'blend4way_Oct6.pt'))
import frame_probe as F, s56, sgen
D = os.path.dirname(os.path.abspath(__file__))
play = {('3c25f709', 1): -0.24, ('506cc169', 1): -1.26, ('48c9e1c7', 1): -0.01}
Bfirst = {k: "((('%s', 1), (5, 10), 5), (('%s', 1), (7, 12), 6))" for k in play}
runs = defaultdict(dict)
for l in open(f'{D}/rungen.jsonl'):
    r = json.loads(l); runs[(r['game'], r['n'])].setdefault(r['i'], {})[r['opt']] = r['margin']
for k, v in runs.items():
    play[k] = st.mean(x['B'] - x['A'] for x in v.values())
P = {(p['game'], p['n']): p for p in json.load(open(f'{D}/gen_prefs.json'))}
ag = F.agent()
rows = []
for r, n, A, B in sgen.positions():
    k = (r['game'], n)
    if k not in play: continue
    b = s56.setup(r); us = b.current_player
    Bp = eval(P[k]['B']) if k in P else eval(Bfirst[k] % (us, us))
    vals = {}
    for tag, pair in (('A', A), ('B', Bp)):
        st0 = dict(b.game_stages); n0 = len(b.moves)
        for m in pair: b.apply_move(m, switch_turn=False)
        cur = F.current_raw(ag, b, us) * 12
        tr = F.trained_values(ag, [F.trained_encs(ag, b, us)])[0] * 12
        while len(b.moves) > n0: b.undo_last_move()
        b.game_stages.update(st0)
        vals[tag] = (cur, tr)
    rows.append({'k': k, 'play': play[k], 'cur': vals['B'][0] - vals['A'][0], 'tr': vals['B'][1] - vals['A'][1]})
json.dump(rows, open(f'{D}/frames.json', 'w'))
print(len(rows), 'positions')
for f in ('cur', 'tr'):
    xs = [r[f] for r in rows]; ys = [r['play'] for r in rows]
    print(f"{f:>4}: mean B-A {st.mean(xs):+.3f}  corr with playouts {st.correlation(xs, ys):+.2f}  "
          f"sign agrees {sum((x > 0) == (y > 0) for x, y in zip(xs, ys))}/{len(rows)}")
print(f"playouts: mean B-A {st.mean(r['play'] for r in rows):+.3f}")
