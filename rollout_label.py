"""rollout_label.py -- a DEEPER distillation teacher: luck-adjusted truncated
rollouts on the close-call decisions (owner + Claude, 2026-10-26).

luck_probe.py measured it on the calibration positions: luck-adjusted rollouts
4-6 turns deep carry ~+-0.1 pts of noise per position at 8 playouts (October's
plain full playouts: +-0.6), and agree with the full playouts a little better
than the 1-turn search (corr +0.31 -> ~+0.38, CI crossing 0). The decisive test
is the one that worked for the close-call teacher: distil and play.

For each decision (from close_call.py label files) and each searched candidate:
  ply 1     the opponent's 21 rolls from the candidate's afterstate, each with
            the app's own search (prefilter 12/40/5, hand rules) -- E_1 is
            EXACTLY close_call's label; computed once per candidate
  playout i rolls drawn from crc32(decision, i): the SAME dice for every
            candidate of the decision (common random numbers); plies 2..H the
            same search for whoever moves, all 21 rolls evaluated per ply
  estimate  est_h = E_h - sum_{t<h} (v_t(thrown) - E_t)   (mover frame, raw units)
            unbiased for the h-ply greedy expectimax value; est_1 = E_1 (no noise)
Every depth h = 1..H is stored (mean over the N playouts), so students trained on
h=1 (today's teacher, re-run with the CURRENT net), h=2 and h=H use identical
positions.

    python3 rollout_label.py prep  <labels.jsonl.gz> <decisions.jsonl.gz> [games=400] [seed=0]
    python3 rollout_label.py run   <net.pt> <decisions.jsonl.gz> <out.jsonl> [count]
            (SEED_BASE = first decision index, ROLL_H=4, ROLL_N=4, N_WORKERS)
    python3 rollout_label.py build <decisions.jsonl.gz> <out_labels.jsonl.gz> <h> <relabel.jsonl>...
    python3 rollout_label.py stats <relabel.jsonl>...
"""
import gzip, json, os, random, sys, time, zlib
import multiprocessing as mp

import numpy as np

os.environ.setdefault('GATE_PREFILTER', '12,40,5')
ROLLS = [(a, b, (1 if a == b else 2) / 36.0) for a in range(1, 7) for b in range(a, 7)]
H = int(os.environ.get('ROLL_H', '4'))
N = int(os.environ.get('ROLL_N', '4'))
N_WORKERS = int(os.environ.get('N_WORKERS', '8'))
DRAW_PAIR = ((1, 1, 1), (0, 0, 0))
_AG = {}
_DEC = None


def _agent(path):
    if path not in _AG:
        import torch
        from network import model_from_state
        from agent_gnn import GNNAgent, prefilter_kwargs, SCORE_SCALE
        m = model_from_state(torch.load(path, map_location='cpu'), strict=False)
        ag = GNNAgent(model=m, hand_rules=True, **prefilter_kwargs(os.environ['GATE_PREFILTER']))
        assert getattr(ag, 'afterstate', False), 'rollout labels need an afterstate net'
        _AG[path] = (ag, SCORE_SCALE)
    return _AG[path]


def _set_dice(b, d1, d2):
    b.dice[0].number, b.dice[0].used = d1, False
    b.dice[1].number, b.dice[1].used = d2, False
    b._reachable_cache.clear()


def _apply(b, pair):
    for m in pair:
        if m != (0, 0, 0):
            b.apply_move(m, switch_turn=False)


def _best(ag, scale, b, player, d1, d2):
    """(value in `player`'s frame, raw units; pair; terminal) of the best pair
    on roll d1-d2. A winning pair is scored by its exact margin."""
    from agent_gnn import NUM_PIECES
    st = dict(b.game_stages)
    _set_dice(b, d1, d2)
    r = ag.select_move_pair(b.get_valid_moves(), b, player, True)
    b.game_stages.update(st)
    if not isinstance(r, list) or not r:
        return 0.0, ((0, 0, 0), (0, 0, 0)), False
    v, pair = max(r, key=lambda x: x[0])
    if v == float('inf'):
        n0 = len(b.moves)
        _apply(b, pair)
        w, s = b.check_game_over()
        while len(b.moves) > n0:
            b.undo_last_move()
        b.game_stages.update(st)
        return (s / NUM_PIECES if w == player else -s / NUM_PIECES), pair, True
    if pair == DRAW_PAIR:
        return 0.0, pair, True
    return v / scale, pair, False


def _ply(ag, scale, b):
    """Enter the next player's turn; evaluate all 21 rolls. -> (player, vals)."""
    b.switch_turn()
    p = b.current_player
    return p, {(d1, d2): _best(ag, scale, b, p, d1, d2) for d1, d2, _ in ROLLS}


def _expect(vals):
    return sum(pr * vals[(d1, d2)][0] for d1, d2, pr in ROLLS)


def _throw(rng):
    return tuple(sorted((rng.randint(1, 6), rng.randint(1, 6))))


def _cand(path, raw_state, mover, dec_key):
    """est[h-1] for h = 1..H (mean over N playouts), plus the per-playout
    estimates at depth H (for a noise check)."""
    from game import Board
    ag, scale = _agent(path)

    def fresh():
        b = Board()
        b.update_state(raw_state)
        return b

    b = fresh()
    p1, vals1 = _ply(ag, scale, b)
    s1 = 1.0 if p1 == mover else -1.0
    e1 = s1 * _expect(vals1)
    per = []                                   # per playout: [est_1..est_H]
    for i in range(N):
        rng = random.Random(zlib.crc32(f'{dec_key}#{i}'.encode()))
        b = fresh()
        b.switch_turn()
        ests, luck = [e1], 0.0
        r = _throw(rng)
        v, pair, term = vals1[r]
        luck += s1 * v - e1
        if pair != DRAW_PAIR and not term:
            _set_dice(b, *r)
            b.get_valid_moves()
            _apply(b, pair)
        end = (s1 * v) if term else None
        for h in range(2, H + 1):
            if end is not None:
                ests.append(end - luck)     # = the estimate at the ending ply
                continue
            p, vals = _ply(ag, scale, b)
            s = 1.0 if p == mover else -1.0
            e = s * _expect(vals)
            ests.append(e - luck)
            r = _throw(rng)
            v, pair, term = vals[r]
            luck += s * v - e
            if term:
                end = s * v
                continue
            _set_dice(b, *r)
            b.get_valid_moves()
            _apply(b, pair)
        per.append(ests)
    per = np.array(per)
    return per.mean(0).tolist(), per[:, -1].tolist()


def _decision(task):
    path, rec = task
    t0 = time.time()
    out = []
    for c in rec['cands']:
        m, last = _cand(path, c['raw_state'], rec['player'], rec['decision'])
        out.append({'est': m, 'lastH': last})
    return {'idx': rec['idx'], 'decision': rec['decision'], 'H': H, 'N': N,
            'cands': out, 'secs': round(time.time() - t0, 1)}


def prep(src, out, games=400, seed=0):
    """Pick `games` whole games from a label file; write their decisions,
    shuffled, with a global index, plus their turn records (kind 'turn')."""
    by_game = {}
    with gzip.open(src, 'rt') as fh:
        for l in fh:
            r = json.loads(l)
            if r.get('kind') in ('cand', 'turn'):
                by_game.setdefault(r['game_id'], []).append(r)
    gids = sorted(by_game)
    random.Random(seed).shuffle(gids)
    gids = gids[:games]
    decs, turns = {}, []
    for g in gids:
        for r in by_game[g]:
            if r['kind'] == 'turn':
                turns.append(r)
            else:
                d = decs.setdefault(r['decision'], {'decision': r['decision'], 'game_id': g,
                                                    'player': r['player'], 'cands': []})
                d['cands'].append(r)
    order = sorted(decs)
    random.Random(seed + 1).shuffle(order)
    with gzip.open(out, 'wt') as fo:
        for i, k in enumerate(order):
            fo.write(json.dumps({'kind': 'decision', 'idx': i, **decs[k]}) + '\n')
        for r in turns:
            fo.write(json.dumps(r) + '\n')
    nc = sum(len(d['cands']) for d in decs.values())
    print(f'{out}: {len(gids)} games, {len(order)} decisions, {nc} cands, {len(turns)} turns')


def _load_decisions(path):
    decs, turns = [], []
    with gzip.open(path, 'rt') as fh:
        for l in fh:
            r = json.loads(l)
            (decs if r['kind'] == 'decision' else turns).append(r)
    decs.sort(key=lambda d: d['idx'])
    return decs, turns


def run(path, dpath, out, count=None):
    path = os.path.abspath(path)
    decs, _ = _load_decisions(dpath)
    lo = int(os.environ.get('SEED_BASE', '0'))
    hi = len(decs) if count is None else min(len(decs), lo + int(count))
    import glob
    done = set()
    for p in {out, *glob.glob(os.environ.get('DONE_GLOB', '') or '/nonexistent')}:
        if os.path.exists(p):
            for l in open(p):
                try:
                    done.add(json.loads(l)['idx'])
                except ValueError:
                    pass
    tasks = [(path, d) for d in decs[lo:hi] if d['idx'] not in done]
    print(f'{len(tasks)} decisions to relabel ({len(done)} done), idx {lo}..{hi - 1}, H={H} N={N}',
          flush=True)
    from arena import _worker_init
    t0 = time.time()
    with mp.Pool(N_WORKERS, initializer=_worker_init) as pool, open(out, 'a') as fo:
        for j, r in enumerate(pool.imap_unordered(_decision, tasks, chunksize=1), 1):
            fo.write(json.dumps(r) + '\n'); fo.flush()
            if j % 20 == 0 or j == len(tasks):
                print(f'  {j}/{len(tasks)}  {(time.time() - t0) / j:.1f} s/decision wall', flush=True)


def _relabels(paths):
    got = {}
    for p in paths:
        for l in open(p):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            got[r['idx']] = r
    return got


def build(dpath, out, h, paths):
    """A distill_train file: every relabelled decision's candidates with label =
    est_h, plus the turn records of the games those decisions came from."""
    h = int(h)
    decs, turns = _load_decisions(dpath)
    got = _relabels(paths)
    games = set()
    nc = 0
    with gzip.open(out, 'wt') as fo:
        for d in decs:
            r = got.get(d['idx'])
            if r is None:
                continue
            games.add(d['game_id'])
            for c, e in zip(d['cands'], r['cands']):
                fo.write(json.dumps(dict(c, label=e['est'][h - 1], label_cc=c['label'])) + '\n')
                nc += 1
        nt = 0
        for t in turns:
            if t['game_id'] in games:
                fo.write(json.dumps(t) + '\n'); nt += 1
    print(f'{out}: h={h}, {len(got)} decisions, {nc} cands, {nt} turns from {len(games)} games')


def stats(paths):
    got = _relabels(paths)
    rs = list(got.values())
    Hh = rs[0]['H']
    print(f'{len(rs)} decisions, H={Hh} N={rs[0]["N"]}, {np.mean([r["secs"] for r in rs]):.0f} s/decision '
          f'(one core), {np.mean([len(r["cands"]) for r in rs]):.2f} cands')
    # how often each depth changes the pick vs depth 1; spread of within-decision differences
    for h in range(1, Hh + 1):
        ch = sum(int(np.argmax([c['est'][h - 1] for c in r['cands']]) !=
                     np.argmax([c['est'][0] for c in r['cands']])) for r in rs if len(r['cands']) > 1)
        sd = [np.std([c['est'][h - 1] for c in r['cands']]) * 12 for r in rs if len(r['cands']) > 1]
        print(f'  h={h}: pick differs from h=1 in {ch / len(rs):.1%}, '
              f'mean within-decision SD {np.mean(sd):.3f} pts')
    # noise of the depth-H difference between the first two candidates
    d = [np.std(np.array(r['cands'][0]['lastH']) - np.array(r['cands'][1]['lastH']), ddof=1) * 12
         for r in rs if len(r['cands']) > 1 and len(r['cands'][0]['lastH']) > 1]
    print(f'  depth-H per-playout SD of a candidate difference {np.sqrt(np.mean(np.square(d))):.3f} pts '
          f'-> SE at N={rs[0]["N"]} {np.sqrt(np.mean(np.square(d))) / np.sqrt(rs[0]["N"]):.3f}')


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'prep':
        prep(a[1], a[2], int(a[3]) if len(a) > 3 else 400, int(a[4]) if len(a) > 4 else 0)
    elif a[0] == 'run':
        run(a[1], a[2], a[3], a[4] if len(a) > 4 else None)
    elif a[0] == 'build':
        build(a[1], a[2], a[3], a[4:])
    elif a[0] == 'stats':
        stats(a[1:])
