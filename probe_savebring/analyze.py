"""Summarise probe_savebring: B - A (bring to goal minus max save), mover's margin
points, by stratum, CIs bootstrapped over GAMES; the net's own B - A beside it;
then a shallow tree on position features, scored on held-out games."""
import json, os, random, statistics as st, sys
from collections import defaultdict
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
from sb import kind, POOL

pick = {(p['game'], p['turn'], tuple(p['roll'])): p for p in json.load(open(f'{D}/play_pick.json'))}
runs = defaultdict(dict)
for l in open(f'{D}/play.jsonl'):
    r = json.loads(l); runs[(r['game'], r['turn'], tuple(r['roll']))].setdefault(r['i'], {})[r['opt']] = r['margin']
rows = []
for k, v in runs.items():
    d = [x['B'] - x['A'] for x in v.values() if 'A' in x and 'B' in x]
    if len(d) < 10: continue
    p = pick[k]; rows.append(dict(p, play=st.mean(d), n=len(d), net=p['valB'] - p['valA'], kind=kind(p)))


def boot(rs, f=lambda r: r['play'], B=2000):
    by = defaultdict(list)
    for r in rs: by[r['game']].append(f(r))
    gs = list(by); rng = random.Random(0); ms = []
    for _ in range(B):
        xs = [x for g in (rng.choice(gs) for _ in gs) for x in by[g]]
        ms.append(st.mean(xs))
    ms.sort(); return st.mean(f(r) for r in rs), ms[int(.025 * B)], ms[int(.975 * B)]


def line(name, rs):
    if not rs: return
    m, lo, hi = boot(rs)
    print(f"{name:20s} {len(rs):4d} pos {len({r['game'] for r in rs}):4d} games  playouts B-A {m:+.3f} "
          f"({lo:+.3f}..{hi:+.3f})  net B-A {st.mean(r['net'] for r in rs):+.3f}  net top=B {sum(r['top']=='B' for r in rs)}")


print('B - A = bring to goal minus save the most (negative: saving is better)')
line('all', rows)
for k in sorted({r['kind'] for r in rows}):
    line(k, [r for r in rows if r['kind'] == k])
for nm, f in (('B numbered', lambda r: r['kind'].startswith('num')), ('B blank', lambda r: r['kind'].startswith('blank')),
              ('late', lambda r: r['kind'].endswith('late')), ('early', lambda r: r['kind'].endswith('early'))):
    line(nm, [r for r in rows if f(r)])
if len(rows) > 3:
    print(f"per-position corr net vs playouts {st.correlation([r['net'] for r in rows], [r['play'] for r in rows]):+.3f}")


# ---- features and a shallow tree, held out by game
def feats(r):
    b_num = [n for n in r['B_arr'] if n <= 6]
    a_num = [n for n in r['A_saves'] if n <= 6]
    return {
        'B_numbered': int(bool(b_num)), 'B_num_max': max(b_num, default=0),
        'B_arrivals': len(r['B_arr']), 'saves_forgone': len(r['A_saves']) - len(r['B_saves']),
        'A_saves_numbered': len(a_num), 'A_num_max': max(a_num, default=0),
        'unsaved_us': r['unsaved_us'], 'unsaved_opp': r['unsaved_opp'],
        'lead': r['unsaved_opp'] - r['unsaved_us'],
        'us_endgame': int(r['stage_us'] == 'endgame'), 'opp_endgame': int(r['stage_opp'] == 'endgame'),
        'roll_hi': max(r['roll']), 'roll_lo': min(r['roll']), 'doubles': int(r['roll'][0] == r['roll'][1]),
    }

if len(rows) >= 60 and '--tree' in sys.argv:
    import numpy as np
    from sklearn.tree import DecisionTreeRegressor, export_text
    from sklearn.model_selection import GroupKFold
    names = list(feats(rows[0]))
    X = np.array([[feats(r)[n] for n in names] for r in rows], float)
    y = np.array([r['play'] for r in rows]); g = np.array([r['game'] for r in rows])
    w = np.array([r['n'] for r in rows], float)
    for depth in (1, 2, 3):
        pred = np.zeros_like(y); base = np.zeros_like(y)
        for tr, te in GroupKFold(5).split(X, y, g):
            t = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=25).fit(X[tr], y[tr], sample_weight=w[tr])
            pred[te] = t.predict(X[te]); base[te] = np.average(y[tr], weights=w[tr])
        # sign accuracy where it matters: does the tree's sign match the playouts'?
        mse, mse0 = np.average((y - pred) ** 2, weights=w), np.average((y - base) ** 2, weights=w)
        print(f"tree depth {depth}: held-out MSE {mse:.4f} vs constant {mse0:.4f} ({100*(1-mse/mse0):+.1f}%)")
    t = DecisionTreeRegressor(max_depth=2, min_samples_leaf=25).fit(X, y, sample_weight=w)
    print(export_text(t, feature_names=names, decimals=2))
    # net as a predictor, same folds, for comparison
    net = np.array([r['net'] for r in rows])
    print(f"net B-A as predictor: corr {np.corrcoef(net, y)[0,1]:+.3f}; tree depth-2 in-sample corr "
          f"{np.corrcoef(t.predict(X), y)[0,1]:+.3f}")
