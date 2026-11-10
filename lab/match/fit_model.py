"""Fit match.js's outcome model from self-play (selfplay_values.py output):
  prior   P(final margin from the STARTER's view), m = -12..12 (0 = draw / unfinished)
  cond    P(final margin from the MOVER's view | the net's value v of the chosen
          position, phase = total unsaved pieces after the move), on a grid of v,
          by a Gaussian kernel in v within each phase band, shrunk toward the
          band's pooled distribution where data are thin.
    python3 fit_model.py sp.jsonl [...] -> match_model_<layout>.json (argv -o)
Also prints held-out log-loss (by game) of cond vs the prior-shifted baseline."""
import json, math, sys, random

PHASES = [0, 6, 12, 18]                     # lower bounds of total-unsaved bands
VGRID = [x / 2 for x in range(-24, 25)]     # -12 .. 12 points, step 0.5
BW = 0.5                                    # kernel bandwidth in points

def load(paths):
    games = []
    for p in paths:
        for l in open(p):
            g = json.loads(l)
            if g['winner'] in ('white', 'black'):
                games.append(g)
    return games

def rows_of(games):
    out = []
    for gi, g in enumerate(games):
        wm = g['margin'] if g['winner'] == 'white' else -g['margin']
        for mover, v, mine, theirs in g['turns']:
            m = wm if mover == 'w' else -wm
            out.append((gi, v, mine + theirs, m))
    return out

def band(ph):
    b = 0
    for i, lo in enumerate(PHASES):
        if ph >= lo: b = i
    return b

def fit(rows):
    dist = []
    for b in range(len(PHASES)):
        rs = [(v, m) for _, v, ph, m in rows if band(ph) == b]
        pooled = [1e-3] * 25
        for _, m in rs: pooled[m + 12] += 1
        z = sum(pooled); pooled = [x / z for x in pooled]
        row = []
        for x in VGRID:
            h = [0.0] * 25; wsum = 0.0
            for v, m in rs:
                w = math.exp(-0.5 * ((v - x) / BW) ** 2)
                if w > 1e-4:
                    h[m + 12] += w; wsum += w
            k = 20.0                               # pseudo-count toward the band's pooled shape
            p = [(h[i] + k * pooled[i]) / (wsum + k) for i in range(25)]
            row.append([round(q, 6) for q in p])
        dist.append(row)
    return dist

def predict(dist, v, ph):
    b = band(ph)
    x = max(VGRID[0], min(VGRID[-1], v))
    j = min(int((x - VGRID[0]) / 0.5), len(VGRID) - 2)
    t = (x - VGRID[j]) / 0.5
    return [(1 - t) * a + t * c for a, c in zip(dist[b][j], dist[b][j + 1])]

def main(paths, out):
    games = load(paths)
    prior = [0] * 25
    for g in games:
        m = g['margin'] if g['winner'] == g['starter'] else -g['margin']
        prior[m + 12] += 1
    n = sum(prior)
    rows = rows_of(games)
    # Held-out check, 5 folds by game.
    ids = list(range(len(games))); random.Random(0).shuffle(ids)
    fold = {gi: i % 5 for i, gi in enumerate(ids)}
    ll_c = ll_b = cnt = 0
    for f in range(5):
        tr = [r for r in rows if fold[r[0]] != f]; te = [r for r in rows if fold[r[0]] == f]
        d = fit(tr)
        pooled = [1e-3] * 25
        for r in tr: pooled[r[3] + 12] += 1
        z = sum(pooled); pooled = [x / z for x in pooled]
        for _, v, ph, m in te:
            ll_c -= math.log(max(1e-9, predict(d, v, ph)[m + 12])); ll_b -= math.log(pooled[m + 12]); cnt += 1
    print(f'{len(games)} games, {len(rows)} positions; starter wins {sum(prior[13:]) / n:.3f}, '
          f'mean margin (starter) {sum((i - 12) * c for i, c in enumerate(prior)) / n:+.3f}')
    print(f'held-out log-loss per position: cond {ll_c / cnt:.3f} vs no-information {ll_b / cnt:.3f}')
    d = fit(rows)
    # Calibration: mean of the conditional distribution vs v, per band.
    for b in range(len(PHASES)):
        rs = [(v, m) for _, v, ph, m in rows if band(ph) == b]
        print(f'phase >= {PHASES[b]:2d}: {len(rs):6d} positions, mean v {sum(v for v, _ in rs) / len(rs):+.2f}, '
              f'mean final {sum(m for _, m in rs) / len(rs):+.2f}')
    model = {'source': ','.join(paths), 'games': len(games),
             'prior': {str(i - 12): round(c / n, 6) for i, c in enumerate(prior) if c},
             'cond': {'phases': PHASES, 'vgrid': VGRID, 'dist': d}}
    json.dump(model, open(out, 'w'), separators=(',', ':'))
    print('wrote', out)

if __name__ == '__main__':
    a = sys.argv[1:]
    out = 'match_model.json'
    if '-o' in a:
        i = a.index('-o'); out = a[i + 1]; a = a[:i] + a[i + 2:]
    main(a, out)
