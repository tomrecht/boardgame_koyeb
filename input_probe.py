"""input_probe.py -- do the new (A2/AB2) inputs carry information the net lacks?

fv2C (2026-10-19): the new input columns stayed at ~5-9% of an old column's
usage, and the burn-in iterations that used them more (ratio up to 0.37) were no
stronger. Two readings: (1) the inputs add nothing the net does not already
compute; (2) they do, but warm-started training cannot teach the net to use them.
This separates the two offline, with no self-play:

  target   z = final_score / 12 (clipped), the game's result from the record's
           mover -- the Monte-Carlo end of training's own TD(lambda) target
  base     the net's own value V (cubic in V, so calibration is not credited
           to the features)
  NEW      base + the new inputs: A2/AB2's 88 globals, per side sum and max of
           the 5 new piece columns, the opponent-wall tile count (109)
  OLD      base + the v1 globals and per side sum/max of the 24 v1 piece
           columns (107) -- the CONTROL: the same aggregation of inputs the net
           already sees. Whatever OLD gains is readout slack, not information.
  SHUF     base + NEW's columns permuted across games -- the overfitting null.

Each is fitted (ridge, and gradient boosting) with 5-fold CV grouped by game
and scored as % reduction of the base's held-out MSE, with a bootstrap-by-game
95% CI. Reading: NEW clearly above OLD and SHUF -> reading (2): the information
is there and the training is the bottleneck (try training only the new columns).
NEW ~ OLD ~ SHUF ~ 0 -> reading (1): stop spending self-play on these inputs.
Broken down by game stage too, as the endgame inputs may matter only late.

Records come from a run with DUMP_RECORDS=<path> (td_selfplay_loop), i.e. the
afterstates training sees, encoded exactly as training encodes them.

    python3 input_probe.py <records.jsonl.gz> <checkpoint.pt> [max_games=2000]
"""
import gzip, json, random, sys, time
from collections import defaultdict

import numpy as np
import torch

import network
from network import BoardEncoder, collate_batch
from encoder import feature_dims
from game import Board

NUM_PIECES = 12


def load_records(path, max_games, seed=0):
    by_game = defaultdict(list)
    with gzip.open(path, 'rt') as f:
        for line in f:
            r = json.loads(line)
            by_game[r['game_id']].append(r)
    games = sorted(by_game)
    random.Random(seed).shuffle(games)
    games = games[:max_games]
    return [r for g in games for r in by_game[g]], len(by_game)


def featurize(recs, model, enc, batch=256):
    """V, NEW, OLD, stage, game, z for each record."""
    v1t, v1p, v1g = feature_dims('v1')
    board = Board()
    V, NEW, OLD, keep = [], [], [], []
    buf, bufi = [], []

    def flush():
        if not buf:
            return
        with torch.no_grad():
            out = model(collate_batch(buf)).detach().cpu().numpy().reshape(-1)
        V.extend(out.tolist())
        buf.clear()

    for i, r in enumerate(recs):
        try:
            board.update_state(r['raw_state'])
            e = enc.encode(board, r['player'])
        except Exception:
            continue
        t, p, g = (np.asarray(e[k]) for k in ('tile_feats', 'piece_feats', 'global_feats'))
        mine, opp = p[p[:, 0] == 0], p[p[:, 0] == 1]

        def agg(cols):
            return np.concatenate([mine[:, cols].sum(0), mine[:, cols].max(0),
                                   opp[:, cols].sum(0), opp[:, cols].max(0)])
        new_p, old_p = list(range(v1p, p.shape[1])), list(range(v1p))
        NEW.append(np.concatenate([g[v1g:], agg(new_p), [t[:, v1t:].sum()]]))
        OLD.append(np.concatenate([g[:v1g], agg(old_p)]))
        keep.append(i)
        buf.append(e)
        if len(buf) >= batch:
            flush()
    flush()
    recs = [recs[i] for i in keep]
    z = np.array([max(-1.0, min(1.0, r['final_score'] / NUM_PIECES)) for r in recs])
    stage = np.array([r.get('game_stage', '?') for r in recs])
    game = np.array([r['game_id'] for r in recs])
    return np.array(V), np.array(NEW), np.array(OLD), stage, game, z


def cv_preds(X, y, groups, kind, k=5, seed=0):
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import RidgeCV
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        if kind == 'ridge':
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))
        else:
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                              max_leaf_nodes=15, min_samples_leaf=100,
                                              l2_regularization=1.0, random_state=seed)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def gain(base_err, err, game, mask=None, n_boot=1000, seed=0):
    """% reduction of held-out MSE vs base, with a bootstrap-by-game 95% CI."""
    if mask is not None:
        base_err, err, game = base_err[mask], err[mask], game[mask]
    ug, inv = np.unique(game, return_inverse=True)
    b = np.bincount(inv, base_err, len(ug))
    e = np.bincount(inv, err, len(ug))
    pt = 100 * (1 - e.sum() / b.sum())
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        s = rng.integers(0, len(ug), len(ug))
        bs.append(100 * (1 - e[s].sum() / b[s].sum()))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return pt, lo, hi, len(ug), int(mask.sum()) if mask is not None else len(err)


def main():
    path, ckpt = sys.argv[1], sys.argv[2]
    max_games = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    network.DEVICE = torch.device('cpu')
    model = network.model_from_state(torch.load(ckpt, map_location='cpu'))
    model.eval()
    if model.features in ('v1',):
        raise SystemExit('a v1 net has no new inputs to probe')
    enc = BoardEncoder(features=model.features, afterstate=bool(model.afterstate))
    t0 = time.time()
    recs, n_all = load_records(path, max_games)
    print(f'{len(recs)} records from {min(n_all, max_games)} of {n_all} games; '
          f'net {ckpt} ({model.features}, afterstate {bool(model.afterstate)})')
    V, NEW, OLD, stage, game, z = featurize(recs, model, enc)
    print(f'encoded {len(z)} in {time.time() - t0:.0f}s; NEW {NEW.shape[1]} cols, '
          f'OLD {OLD.shape[1]} cols')
    print(f'net vs result: corr {np.corrcoef(V, z)[0, 1]:+.3f}, '
          f'MSE raw {np.mean((V - z) ** 2):.4f}, var(z) {z.var():.4f}')

    base = np.stack([V, V ** 2, V ** 3], 1)
    perm = np.random.default_rng(1).permutation(len(z))
    sets = {'NEW': np.hstack([base, NEW]), 'OLD': np.hstack([base, OLD]),
            'SHUF': np.hstack([base, NEW[perm]])}
    stages = ['opening', 'midgame', 'endgame']
    for kind in ('ridge', 'gbm'):
        pb = cv_preds(base, z, game, kind)
        be = (pb - z) ** 2
        print(f'\n[{kind}] base held-out MSE {be.mean():.4f} '
              f'(explains {100 * (1 - be.mean() / z.var()):.1f}% of var z)')
        print(f'  {"set":5s} {"all":>24s}' + ''.join(f'{s:>24s}' for s in stages))
        for name, X in sets.items():
            e = (cv_preds(X, z, game, kind) - z) ** 2
            cells = [gain(be, e, game)] + [gain(be, e, game, stage == s) for s in stages]
            print(f'  {name:5s} ' + ''.join(
                f'{c[0]:+6.2f}% [{c[1]:+5.2f},{c[2]:+5.2f}] n{c[4]:<5d}' for c in cells))
    print(f'\ntotal {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
