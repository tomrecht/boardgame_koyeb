"""Fit and judge a margin-SPREAD output on the net's hidden layer (spread_data.py output).
    python3 train_spread.py data.jsonl.gz [...] [-o spread_head_sum7.json]
Model: 64 -> 32 (ReLU) -> 25 softmax over the final margin (mover's view, -12..12),
trained by cross-entropy; the net's value output is untouched. Judged against the
current lookup (fit_model.py: shape by value and phase) by:
  1. held-out log-loss, 5 folds by game, raw and TILTED to the net's value (as used);
  2. spread calibration: positions binned by predicted SD -> actual RMS of (margin - v);
  3. separation: for near-equal candidates (|dv| < 0.15 pts) of one position, how
     different the two tilted distributions are (SD gap, total variation)."""
import base64, gzip, json, math, random, sys
import numpy as np

M = np.arange(-12, 13)
PHASES = [0, 6, 12, 18]
VGRID = np.arange(-12, 12.01, 0.5)
BW, K = 0.5, 20.0


def dec(h):
    return np.frombuffer(base64.b64decode(h), dtype=np.float16).astype(np.float32)


def load(paths):
    X, V, PH, Y, G, C = [], [], [], [], [], []
    gi = 0
    for p in paths:
        op = gzip.open if p.endswith('.gz') else open
        with op(p, 'rt') as f:
            for l in f:
                g = json.loads(l)
                if g['winner'] not in ('white', 'black'):
                    continue
                wm = g['margin'] if g['winner'] == 'white' else -g['margin']
                for mover, h, v, ph, cands in g['turns']:
                    X.append(dec(h)); V.append(v); PH.append(ph); Y.append((wm if mover == 'w' else -wm) + 12); G.append(gi)
                    if cands and len(cands) > 1:
                        C.append((gi, ph, [(dec(a), b) for a, b in cands]))
                gi += 1
    return np.array(X), np.array(V), np.array(PH), np.array(Y), np.array(G), C, gi


def band(ph):
    return np.searchsorted(PHASES, ph, side='right') - 1


# ---- the current lookup (fit_model.py), vectorised ----
def lookup_fit(V, PH, Y):
    b = band(PH); dist = np.zeros((len(PHASES), len(VGRID), 25))
    for k in range(len(PHASES)):
        s = b == k
        pooled = np.bincount(Y[s], minlength=25) + 1e-3; pooled /= pooled.sum()
        w = np.exp(-0.5 * ((V[s][None, :] - VGRID[:, None]) / BW) ** 2)
        w[w < 1e-4] = 0
        h = w @ np.eye(25)[Y[s]]
        dist[k] = (h + K * pooled) / (w.sum(1, keepdims=True) + K)
    return dist


def lookup_pred(dist, V, PH):
    b = band(PH); x = np.clip(V, VGRID[0], VGRID[-1])
    j = np.minimum(((x - VGRID[0]) / 0.5).astype(int), len(VGRID) - 2)
    t = ((x - VGRID[j]) / 0.5)[:, None]
    return (1 - t) * dist[b, j] + t * dist[b, j + 1]


def tilt(P, v):
    """Exponential tilt of each row so its mean equals v (as match.js does)."""
    v = np.clip(v, -11.95, 11.95)
    lo, hi = np.full(len(v), -30.0), np.full(len(v), 30.0)
    for _ in range(50):
        l = (lo + hi) / 2
        w = P * np.exp(np.outer(l, M) - np.abs(l)[:, None] * 12)
        mu = (w * M).sum(1) / w.sum(1)
        lo = np.where(mu < v, l, lo); hi = np.where(mu < v, hi, l)
    l = (lo + hi) / 2
    w = P * np.exp(np.outer(l, M) - np.abs(l)[:, None] * 12)
    return w / w.sum(1, keepdims=True)


# ---- the head ----
class Head:
    def __init__(self, d, hid=32, seed=0):
        r = np.random.RandomState(seed)
        self.p = {'W1': r.randn(d, hid) * math.sqrt(2 / d), 'b1': np.zeros(hid),
                  'W2': r.randn(hid, 25) * 0.01, 'b2': np.zeros(25)}

    def fwd(self, X):
        a = X @ self.p['W1'] + self.p['b1']; h = np.maximum(a, 0)
        z = h @ self.p['W2'] + self.p['b2']; z -= z.max(1, keepdims=True)
        e = np.exp(z); return a, h, e / e.sum(1, keepdims=True)

    def fit(self, X, Y, Xv, Yv, l2=1e-4, lr=2e-3, bs=512, epochs=40):
        m = {k: np.zeros_like(v) for k, v in self.p.items()}; s = {k: np.zeros_like(v) for k, v in self.p.items()}
        best, bestp, t, bad = 1e9, None, 0, 0
        n = len(X); r = np.random.RandomState(1)
        for ep in range(epochs):
            idx = r.permutation(n)
            for i in range(0, n, bs):
                b = idx[i:i + bs]; x, y = X[b], Y[b]
                a, h, P = self.fwd(x)
                dz = P.copy(); dz[np.arange(len(b)), y] -= 1; dz /= len(b)
                g = {'W2': h.T @ dz + l2 * self.p['W2'], 'b2': dz.sum(0)}
                dh = dz @ self.p['W2'].T; dh[a <= 0] = 0
                g['W1'] = x.T @ dh + l2 * self.p['W1']; g['b1'] = dh.sum(0)
                t += 1
                for k in self.p:
                    m[k] = 0.9 * m[k] + 0.1 * g[k]; s[k] = 0.999 * s[k] + 0.001 * g[k] ** 2
                    self.p[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(s[k] / (1 - 0.999 ** t)) + 1e-8)
            ll = nll(self.fwd(Xv)[2], Yv)
            if ll < best - 1e-4: best, bestp, bad = ll, {k: v.copy() for k, v in self.p.items()}, 0
            else:
                bad += 1
                if bad >= 4: break
        self.p = bestp
        return best


def nll(P, Y):
    return float(-np.log(np.maximum(P[np.arange(len(Y)), Y], 1e-9)).mean())


def feats(X, PH):
    return np.hstack([X, (PH / 24.0)[:, None]])


def sd(P):
    mu = (P * M).sum(1); return np.sqrt((P * M ** 2).sum(1) - mu ** 2)


def main(paths, out):
    X, V, PH, Y, G, C, ng = load(paths)
    print(f'{ng} games, {len(Y)} positions, {len(C)} candidate sets')
    F = feats(X, PH)
    ids = list(range(ng)); random.Random(0).shuffle(ids); fold = np.zeros(ng, int)
    for i, g in enumerate(ids): fold[g] = i % 5
    fp = fold[G]
    res = {k: [] for k in ('base', 'look', 'look_t', 'head', 'head_t')}
    predH = np.zeros((len(Y), 25)); predL = np.zeros((len(Y), 25))
    for f in range(5):
        tr, te = fp != f, fp == f
        trg = np.where(tr)[0]; vmask = (G[trg] % 10) == 0
        pooled = np.bincount(Y[tr], minlength=25) + 1e-3; pooled /= pooled.sum()
        d = lookup_fit(V[tr], PH[tr], Y[tr])
        PL = lookup_pred(d, V[te], PH[te]); predL[te] = PL
        hd = Head(F.shape[1], seed=f)
        hd.fit(F[trg[~vmask]], Y[trg[~vmask]], F[trg[vmask]], Y[trg[vmask]])
        PHd = hd.fwd(F[te])[2]; predH[te] = PHd
        res['base'].append(nll(np.tile(pooled, (te.sum(), 1)), Y[te]))
        res['look'].append(nll(PL, Y[te])); res['look_t'].append(nll(tilt(PL, V[te]), Y[te]))
        res['head'].append(nll(PHd, Y[te])); res['head_t'].append(nll(tilt(PHd, V[te]), Y[te]))
        print(f'fold {f}: ' + ' '.join(f'{k} {v[-1]:.4f}' for k, v in res.items()), flush=True)
    print('held-out log-loss: ' + ', '.join(f'{k} {np.mean(v):.4f}' for k, v in res.items()))

    # Spread calibration: does the predicted SD (tilted head) match the realised error?
    ST = sd(tilt(predH, V)); SL = sd(tilt(predL, V)); err = (Y - 12) - V
    for name, S in (('head', ST), ('lookup', SL)):
        qs = np.quantile(S, [0, .1, .25, .5, .75, .9, 1])
        print(f'{name}: predicted SD by decile band -> realised RMS(margin - v)')
        for a, b in zip(qs[:-1], qs[1:]):
            s = (S >= a) & (S <= b)
            print(f'   SD {a:5.2f}-{b:5.2f}  n {s.sum():6d}  mean pred {S[s].mean():5.2f}  realised {np.sqrt((err[s] ** 2).mean()):5.2f}')

    # Final head on all data.
    vmask = (G % 10) == 0
    hd = Head(F.shape[1], seed=9); hd.fit(F[~vmask], Y[~vmask], F[vmask], Y[vmask])
    d = lookup_fit(V, PH, Y)

    # Separation among near-equal candidates (in-sample head; the cands are not training rows).
    gaps_h, gaps_l, tv_h, tv_l, pairs = [], [], [], [], 0
    for gi, ph, cs in C:
        xs = np.array([c[0] for c in cs]); vs = np.array([c[1] for c in cs])
        phs = np.full(len(cs), ph)
        Ph = tilt(hd.fwd(feats(xs, phs))[2], vs); Pl = tilt(lookup_pred(d, vs, phs), vs)
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                if abs(vs[i] - vs[j]) < 0.15:
                    pairs += 1
                    gaps_h.append(abs(sd(Ph[[i]])[0] - sd(Ph[[j]])[0])); gaps_l.append(abs(sd(Pl[[i]])[0] - sd(Pl[[j]])[0]))
                    tv_h.append(0.5 * np.abs(Ph[i] - Ph[j]).sum()); tv_l.append(0.5 * np.abs(Pl[i] - Pl[j]).sum())
    if pairs:
        print(f'near-equal candidate pairs: {pairs}; |SD gap| head mean {np.mean(gaps_h):.3f} '
              f'(90th pct {np.quantile(gaps_h, .9):.3f}) vs lookup {np.mean(gaps_l):.3f}; '
              f'total variation head {np.mean(tv_h):.3f} vs lookup {np.mean(tv_l):.3f}')
    json.dump({'source': ','.join(paths), 'positions': int(len(Y)), 'input': 'readout[1] hidden (64) + phase/24',
               **{k: v.round(6).tolist() for k, v in hd.p.items()}}, open(out, 'w'), separators=(',', ':'))
    print('wrote', out)


if __name__ == '__main__':
    a = sys.argv[1:]; out = 'spread_head_sum7.json'
    if '-o' in a:
        i = a.index('-o'); out = a[i + 1]; a = a[:i] + a[i + 2:]
    main(a, out)
