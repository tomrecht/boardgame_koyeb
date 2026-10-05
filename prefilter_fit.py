"""Refit the prefilter heuristic for RECALL of the net's top move.

The heuristic (agent.Agent) was grid-searched to PLAY well. As the agent's
prefilter its only job is to keep the net's favourite move among what it passes
on: stage 1 keeps the top F=12 first moves by heuristic score (plus every first
move that saves or enables a save), stage 2 keeps the top K=40 non-save pairs
(min 5) of the kept first moves, plus every save pair; only those reach the net.

The heuristic score is a sum of 23 components, each (mover minus opponent). We
fit one SCALE per component (agent weights 'component_scale', default 1 = today)
so the net's best move ranks high at both stages.

  extract : for turn-start positions from owner's games, store every first move's
            and every pair's 23-component vector, the exemption flags, and which
            pairs reach the PURE net's best position (blanks interchangeable).
  fit     : listwise softmax surrogate on both stages, then a local search on the
            simulated keep-rate itself; train/test split by game.
  eval    : simulated keep-rate of today's heuristic vs the fitted scales.

    nice -n 19 python3 prefilter_fit.py extract [every_kth_turn=2]   -> prefilter_data/
    python3 prefilter_fit.py fit                                     -> prefilter_scales.json
"""
import glob, json, os, random, sys, time

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(REPO, os.environ.get('PREFILTER_DATA', 'prefilter_data'))
LOGS = ['quahuru-games-apvo2h65-v2.jsonl', 'quahuru-games-f7in6olg.jsonl',
        'quahuru-games-apvo2h65-2026-10-05.jsonl']
F_FIRST, K_PAIRS, MIN_K = 12, 40, 5
COMPONENTS = ['saved_pieces', 'saved_bonus', 'goal_pieces', 'goal_bonus', 'captured_pieces',
              'captured_bonus', 'pieces_near_goal', 'pieces_nearer_goal', 'near_goal_bonus',
              'blocked_pieces', 'blocked_piece_bonus', 'loose_pieces', 'loose_piece_bonus',
              'total_distance', 'unentered_pieces', 'off_goal_penalty', 'far_from_goal_penalty',
              'high_goal_penalty', 'high_goal_proximity_penalty', 'enemy_blot_penalty',
              'game_stage_bonus', 'dice_spread_bonus', 'permanent_block_bonus']
# A 24th, 'goal_layout_cost' (exact endgame-table banking turns of the pieces on
# goals), was tried 2026-10-05 and dropped: 5-fold CV keep-rate 0.957 with or
# without it. Its extraction lives in prefilter_data2/ (column 23).
NOOP = ((0, 0, 0), (1, 1, 1))


# ---------------------------------------------------------------- extraction
_H = None


def _heur():
    global _H
    if _H is None:
        from agent import Agent
        # weights=None loads best_weights.json -- what GNNAgent's prefilter
        # uses (and what heuristic_weights.json ships). Agent() with no
        # argument loads INITIAL_WEIGHTS, a different, untuned set.
        _H = Agent(weights=None)
    return _H


def comp_vec(board, player):
    """23 component values, mover minus opponent (their sum = today's score)."""
    s, c = _heur().evaluate(board, player)
    if not c:                                    # game over: no components
        return None
    return [c['player'][k] - c['opponent'][k] for k in COMPONENTS]


def extract_position(b, us):
    import weakness_probe as W
    from agent_gnn import _move_sort_key
    ag = W.agent()
    stages0 = dict(b.game_stages)
    moves = list(b.get_valid_moves())
    if (1, 1, 1) in moves:                       # draw callable: different decision
        return None
    scored = ag.select_move_pair(moves, b, us, return_scores=True)
    b.game_stages.update(stages0)
    if not isinstance(scored, list) or not scored or scored[0][0] == float('inf'):
        return None
    best = scored[0][0]

    def key_after(pair):
        n0 = len(b.moves)
        for m in pair:
            if m not in NOOP:
                b.apply_move(m, switch_turn=False)
        k = W._piece_locs(b)
        while len(b.moves) > n0:
            b.undo_last_move()
        b.game_stages.update(stages0)
        return k
    best_keys = {key_after(p) for s, p in scored if abs(s - best) < 1e-9}

    first = sorted((m for m in moves if m not in NOOP), key=_move_sort_key)
    F1, save1, en1 = [], [], []
    for m in first:
        n0 = len(b.moves)
        b.apply_move(m, switch_turn=False)
        w, _ = b.check_game_over()
        if w == us:
            while len(b.moves) > n0:
                b.undo_last_move()
            b.game_stages.update(stages0)
            return None                          # a winning move: the agent takes it
        F1.append(comp_vec(b, us))
        prev = b.game_stages[us]
        b.game_stages[us] = b.get_game_stage(us)
        en1.append(any(b.get_saving_die(p) for p in b.pieces if p.player == us))
        b.game_stages[us] = prev
        save1.append(isinstance(m, tuple) and m[1] == 'save')
        while len(b.moves) > n0:
            b.undo_last_move()
        b.game_stages.update(stages0)

    F2, idx2, save2, best2 = [], [], [], []

    def add_pair(i, pair):
        F2.append(comp_vec(b, us))
        idx2.append(i)
        save2.append(any(isinstance(x, tuple) and len(x) == 3 and x[1] == 'save' for x in pair))
        best2.append(W._piece_locs(b) in best_keys)

    if (0, 0, 0) in moves:
        add_pair(-1, ((0, 0, 0), (0, 0, 0)))
    for i, m in enumerate(first):
        n0 = len(b.moves)
        b.apply_move(m, switch_turn=False)
        if not [p for p in b.home_tile.pieces if p.player == b.current_player]:
            add_pair(i, (m, (0, 0, 0)))
        if not all(d.used for d in b.dice):
            nxt = sorted((x for x in set(b.get_valid_moves()) if x not in NOOP), key=_move_sort_key)
            for x in nxt:
                b.apply_move(x, switch_turn=False)
                w, _ = b.check_game_over()
                if w == us:
                    while len(b.moves) > n0:
                        b.undo_last_move()
                    b.game_stages.update(stages0)
                    return None
                add_pair(i, (m, x))
                b.undo_last_move()
        while len(b.moves) > n0:
            b.undo_last_move()
        b.game_stages.update(stages0)
    if not any(best2) or any(v is None for v in F1) or any(v is None for v in F2):
        return None
    return dict(F1=np.asarray(F1, np.float32), save1=np.asarray(save1), en1=np.asarray(en1),
                F2=np.asarray(F2, np.float32), idx2=np.asarray(idx2, np.int16),
                save2=np.asarray(save2), best2=np.asarray(best2))


def _extract_game(task):
    rec, every = task
    import weakness_probe as W
    out = os.path.join(DATA, f"{rec['id'][:8]}.npz")
    if os.path.exists(out):
        return rec['id'][:8], 0
    arrays, meta = {}, []
    for t, b, played, b2 in W.turn_states(rec):
        if t % every:
            continue
        d = extract_position(b, b.current_player)
        if d is None:
            continue
        j = len(meta)
        for k, v in d.items():
            arrays[f'{j}_{k}'] = v
        meta.append(t)
    np.savez_compressed(out, turns=np.asarray(meta), **arrays)
    return rec['id'][:8], len(meta)


def extract(every=2, workers=2):
    os.makedirs(DATA, exist_ok=True)
    recs, seen = [], set()
    for f in LOGS:
        for line in open(os.path.join(REPO, f)):
            r = json.loads(line)
            if r.get('completed') and r['id'] not in seen:
                seen.add(r['id'])
                recs.append(r)
    print(f'{len(recs)} games', flush=True)
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(workers) as pool:
        for i, (gid, n) in enumerate(pool.imap_unordered(_extract_game, [(r, every) for r in recs])):
            print(f'{i + 1}/{len(recs)} {gid}: {n} positions ({time.time() - t0:.0f}s)', flush=True)


# ---------------------------------------------------------------- simulation / fit
def load():
    games = {}
    for path in sorted(glob.glob(os.path.join(DATA, '*.npz'))):
        z = np.load(path)
        pos = []
        for j in range(len(z['turns'])):
            pos.append({k: z[f'{j}_{k}'] for k in ('F1', 'save1', 'en1', 'F2', 'idx2', 'save2', 'best2')})
        games[os.path.basename(path)[:8]] = pos
    return games


def kept(p, w):
    """Simulate the shipped prefilter with component scales w: does any kept
    pair reach the net's best position?"""
    n1 = len(p['F1'])
    if n1 > F_FIRST:
        s1 = p['F1'] @ w
        order = np.argsort(-s1, kind='stable')
        keep1 = np.zeros(n1, bool)
        keep1[order[:F_FIRST]] = True
        keep1 |= p['save1'] | p['en1']
    else:
        keep1 = np.ones(n1, bool)
    idx = p['idx2'].astype(int)
    cand = np.where(idx < 0, True, keep1[np.maximum(idx, 0)])
    sv = cand & p['save2']
    other = np.where(cand & ~p['save2'])[0]
    if len(other):
        s2 = p['F2'][other] @ w
        k = max(min(MIN_K, len(other)), min(K_PAIRS, len(other)))
        top = other[np.argsort(-s2, kind='stable')[:k]]
    else:
        top = other
    return bool(p['best2'][sv].any() or p['best2'][top].any())


def keep_rate(positions, w):
    return float(np.mean([kept(p, w) for p in positions]))


def surrogate_grad(positions, w, tau=1.0):
    """Listwise softmax log-likelihood of the best candidates at both stages
    (scores scaled by tau), and its gradient in w."""
    ll, g = 0.0, np.zeros_like(w)
    for p in positions:
        for F, lab in ((p['F1'], _first_labels(p)), (p['F2'], p['best2'])):
            if not lab.any() or len(F) < 2:
                continue
            s = (F @ w) / tau
            s -= s.max()
            e = np.exp(s)
            Z = e.sum()
            pz = e / Z
            pl = e[lab].sum() / Z
            ll += np.log(max(pl, 1e-300))
            # d log pl / dw = E_{lab}[F] - E_all[F]   (over tau)
            el = (e[lab, None] * F[lab]).sum(0) / e[lab].sum()
            ea = (pz[:, None] * F).sum(0)
            g += (el - ea) / tau
    return ll, g


def _first_labels(p):
    lab = np.zeros(len(p['F1']), bool)
    idx = p['idx2'].astype(int)
    for i in idx[p['best2'] & (idx >= 0)]:
        lab[i] = True
    return lab


def fit():
    games = load()
    ids = sorted(games)
    random.Random(1).shuffle(ids)
    tr_ids, te_ids = ids[:len(ids) // 2], ids[len(ids) // 2:]
    tr = [p for g in tr_ids for p in games[g]]
    te = [p for g in te_ids for p in games[g]]
    w0 = np.ones(len(COMPONENTS))
    if 'goal_layout_cost' in COMPONENTS:
        w0[COMPONENTS.index('goal_layout_cost')] = 0.0     # today's heuristic
    print(f'{len(tr)} train / {len(te)} test positions ({len(tr_ids)} / {len(te_ids)} games)')
    print(f'today: keep-rate train {keep_rate(tr, w0):.3f}, test {keep_rate(te, w0):.3f}')
    # scale so typical score gaps are O(1) for the softmax
    sd = np.std(np.concatenate([p['F2'] @ w0 for p in tr[:200]]))
    tau = max(sd, 1e-6)
    w = w0.copy()
    lr = 0.05
    for it in range(60):
        ll, g = surrogate_grad(tr, w, tau)
        w = np.clip(w + lr * g / (np.linalg.norm(g) + 1e-12) * np.linalg.norm(w0), 0.0, 5.0)
        if it % 10 == 9:
            print(f'  surrogate it{it + 1}: ll {ll:.1f}  keep train {keep_rate(tr, w):.3f}', flush=True)
    # local search on the keep-rate itself (coordinate-wise multiplicative steps)
    best_w, best_k = w.copy(), keep_rate(tr, w)
    # additive step sizes per component: the score spread a unit of it buys,
    # so a component that starts at 0 (multiplicative steps cannot leave 0)
    # can still be switched on, with either sign
    allF = np.concatenate([p['F2'] for p in tr[:300]])
    score_sd = np.std(allF @ w0) or 1.0
    unit = np.array([score_sd / (np.std(allF[:, i]) or 1.0) for i in range(len(w0))])
    for sweep in range(3):
        for i in range(len(w)):
            cands = [best_w[i] * f for f in (0.0, 0.5, 0.8, 1.25, 2.0)]
            cands += [best_w[i] + c * unit[i] for c in (-1.0, -0.3, -0.1, 0.1, 0.3, 1.0)]
            for v in cands:
                cand = best_w.copy()
                cand[i] = v
                k = keep_rate(tr, cand)
                if k > best_k + 1e-9:
                    best_w, best_k = cand, k
        print(f'  local sweep {sweep + 1}: keep train {best_k:.3f}', flush=True)
    print(f'fitted: keep-rate train {keep_rate(tr, best_w):.3f}, TEST {keep_rate(te, best_w):.3f} '
          f'(today test {keep_rate(te, w0):.3f})')
    scales = {c: round(float(v), 4) for c, v in zip(COMPONENTS, best_w)}
    for c, v in scales.items():
        if abs(v - 1) > 1e-6:
            print(f'    {c:<30} x{v}')
    json.dump(scales, open(os.path.join(REPO, 'prefilter_scales.json'), 'w'), indent=1)
    print('saved prefilter_scales.json')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'fit'
    if cmd == 'extract':
        extract(int(sys.argv[2]) if len(sys.argv) > 2 else 2,
                int(os.environ.get('N_WORKERS', '2')))
    elif cmd == 'fit':
        fit()


# ---------------------------------------------------------------- cross-validation
def _fit_on(tr, w0, restrained=False, sweeps=3):
    """The fit above as a function: surrogate ascent, then local search on the
    keep-rate. restrained=True keeps every scale within [0.25, 4] x today's and
    uses multiplicative steps only (no switching components on or off)."""
    sd = np.std(np.concatenate([p['F2'] @ w0 for p in tr[:200]]))
    tau = max(sd, 1e-6)
    lo, hi = (0.25 * w0, 4.0 * w0) if restrained else (np.full_like(w0, -1e9), np.full_like(w0, 1e9))
    w = w0.copy()
    for it in range(60):
        ll, g = surrogate_grad(tr, w, tau)
        w = w + 0.05 * g / (np.linalg.norm(g) + 1e-12) * np.linalg.norm(w0)
        w = np.clip(w, np.minimum(lo, hi), np.maximum(lo, hi)) if restrained else np.clip(w, -1e6, 1e6)
    allF = np.concatenate([p['F2'] for p in tr[:300]])
    score_sd = np.std(allF @ w0) or 1.0
    unit = np.array([score_sd / (np.std(allF[:, i]) or 1.0) for i in range(len(w0))])
    best_w, best_k = w.copy(), keep_rate(tr, w)
    for sweep in range(sweeps):
        for i in range(len(w)):
            cands = [best_w[i] * f for f in (0.5, 0.8, 1.25, 2.0)]
            if not restrained:
                cands += [0.0] + [best_w[i] + c * unit[i] for c in (-1.0, -0.3, -0.1, 0.1, 0.3, 1.0)]
            for v in cands:
                if restrained and not (min(lo[i], hi[i]) - 1e-12 <= v <= max(lo[i], hi[i]) + 1e-12):
                    continue
                cand = best_w.copy()
                cand[i] = v
                k = keep_rate(tr, cand)
                if k > best_k + 1e-9:
                    best_w, best_k = cand, k
    return best_w


def crossval(k=5):
    games = load()
    ids = sorted(games)
    random.Random(7).shuffle(ids)
    folds = [ids[i::k] for i in range(k)]
    names = COMPONENTS
    lay = names.index('goal_layout_cost') if 'goal_layout_cost' in names else None
    variants = {
        'today': None,
        '23 components': dict(drop_layout=True, restrained=False),
        '24 components': dict(drop_layout=False, restrained=False),
        '23 restrained (0.25-4x)': dict(drop_layout=True, restrained=True),
    }
    res = {v: [] for v in variants}
    for fi in range(k):
        te = [p for g in folds[fi] for p in games[g]]
        tr = [p for j in range(k) if j != fi for g in folds[j] for p in games[g]]
        for v, cfg in variants.items():
            w0 = np.ones(len(names))
            if lay is not None:
                w0[lay] = 0.0
            if cfg is None:
                w = w0
            else:
                trv = tr
                if cfg['drop_layout'] and lay is not None:
                    # fit without the layout column (keep it at 0)
                    w = _fit_on([dict(p, F1=np.delete(p['F1'], lay, 1), F2=np.delete(p['F2'], lay, 1)) for p in trv],
                                np.delete(w0, lay), restrained=cfg['restrained'])
                    w = np.insert(w, lay, 0.0)
                else:
                    if lay is not None:
                        w0[lay] = 1e-9          # let multiplicative steps work from ~0
                    w = _fit_on(trv, w0, restrained=cfg['restrained'])
            res[v].append(keep_rate(te, w))
        print(f'fold {fi + 1}/{k}: ' + ', '.join(f'{v} {res[v][-1]:.3f}' for v in variants), flush=True)
    print('\nheld-out keep-rate, mean over folds (min..max):')
    for v in variants:
        r = res[v]
        print(f'  {v:<26} {np.mean(r):.3f}  ({min(r):.3f}..{max(r):.3f})')


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'cv':
    crossval()


# ---------------------------------------------------------------- final fit + out-of-sample check
def final():
    games = load()
    allp = [p for g in games.values() for p in g]
    w0 = np.ones(len(COMPONENTS))
    w = _fit_on(allp, w0)
    print(f'final fit on {len(allp)} positions ({len(games)} games): keep-rate {keep_rate(allp, w):.3f} '
          f'(today {keep_rate(allp, w0):.3f}, in-sample)')
    scales = {c: round(float(v), 5) for c, v in zip(COMPONENTS, w)}
    json.dump(scales, open(os.path.join(REPO, 'prefilter_scales.json'), 'w'), indent=1)
    for c, v in scales.items():
        print(f'    {c:<30} x{v}')


def _selfplay_game(seed):
    """One computer-vs-computer game with the SHIPPED agent; returns the
    serialized turn-start positions (every second turn)."""
    import random as _r
    import weakness_probe as W
    from agent_gnn import GNNAgent
    from game import Board
    W.agent()
    dep = GNNAgent(model=W._MODEL, use_prefilter=True, first_move_prefilter=12,
                   prefilter_top_k=40, prefilter_min_k=5)
    _r.seed(seed)
    b = Board()
    out = []
    for turn in range(200):
        w, _ = b.check_game_over()
        if w or b.draw_callable:
            break
        us = b.current_player
        moves = list(b.get_valid_moves())
        if turn % 2 == 0 and len(moves) > 1:
            st = dict(b.game_stages)
            d = extract_position(b, us)
            b.game_stages.update(st)
            if d is not None:
                out.append(d)
        pair = dep.select_move_pair(moves, b, us, difficulty=1.0)
        if isinstance(pair, tuple) and len(pair) == 3:
            pair = (pair, (0, 0, 0))
        for m in pair:
            if m not in NOOP:
                b.apply_move(m, switch_turn=False)
        b.switch_turn()
    return out


def ood(n_games=20, workers=4):
    from multiprocessing import Pool
    with Pool(workers) as pool:
        positions = [p for g in pool.map(_selfplay_game, range(31_000, 31_000 + n_games)) for p in g]
    w0 = np.ones(len(COMPONENTS))
    sc = json.load(open(os.path.join(REPO, 'prefilter_scales.json')))
    w = np.array([sc[c] for c in COMPONENTS])
    print(f'OUT-OF-SAMPLE (computer vs computer, {n_games} games, {len(positions)} positions): '
          f'keep-rate today {keep_rate(positions, w0):.3f}, refit {keep_rate(positions, w):.3f}')


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'final':
    final()
if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'ood':
    ood()
