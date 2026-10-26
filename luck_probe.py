"""luck_probe.py -- can LUCK-ADJUSTED, TRUNCATED playouts be a low-noise teacher?

October's plain playouts (20 per move, calib_bench.json) carried ~+-0.6 pts of
noise per position against real move differences of tenths -- too noisy to train
on. This measures the fix, on the same positions, with the shipped afterstate net:

  playout   from a calibration position's afterstate (owner's move A, or the
            net's B), the sides alternate with the app's own search (prefilter,
            hand rules off); rollout i of A and of B share the dice (seed i)
  luck      at every ply, the mover's best value is also computed for all 21
            rolls; luck_t = value(best | roll thrown) - E_rolls value(best),
            in the ORIGINAL mover's frame
  estimate  after H plies (or the end): V_H - sum_{t<=H} luck_t, where V_H is the
            net's value of the position reached (exact margin if the game ended)
            -- unbiased for the H-ply greedy expectimax value; at H=1 it IS the
            21-roll close-call value (no noise). Also the plain V_H for contrast.

Reported per H: the SD of B-A across playouts (-> SE per position at N), its
agreement with October's 20-playout difference, and the mean shift between the
adjusted and plain estimates (a check on the luck terms: ~0 expected).

    python3 luck_probe.py run <net.pt> [positions=120] [N=6] [H=6] [out.jsonl]
    python3 luck_probe.py analyze <out.jsonl>
"""
import ast, json, math, os, random, sys, time
import multiprocessing as mp

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(REPO, 'calib_bench.json')
ROLLS = [(a, b, (1 if a == b else 2) / 36.0) for a in range(1, 7) for b in range(a, 7)]
N_WORKERS = int(os.environ.get('N_WORKERS', '8'))
PTS = 12.0
_AG = None


def _agent(path):
    global _AG
    if _AG is None:
        import torch
        torch.set_num_threads(1)
        torch.set_grad_enabled(False)
        import network
        network.DEVICE = torch.device('cpu')
        from agent_gnn import GNNAgent, prefilter_kwargs, SCORE_SCALE
        m = network.model_from_state(torch.load(path, map_location='cpu'))
        _AG = (GNNAgent(model=m, hand_rules=False, **prefilter_kwargs('12,40,5')), SCORE_SCALE)
    return _AG


def _best(ag, scale, b, player, d1, d2):
    """(value in player's frame, raw units; pair) of the best reply on roll d1-d2.
    Board unchanged afterwards."""
    st = dict(b.game_stages)
    b.dice[0].number, b.dice[0].used = d1, False
    b.dice[1].number, b.dice[1].used = d2, False
    b._reachable_cache.clear()
    moves = b.get_valid_moves()
    r = ag.select_move_pair(moves, b, player, True)
    b.game_stages.update(st)
    if not isinstance(r, list) or not r:
        return 0.0, ((0, 0, 0), (0, 0, 0))
    v, pair = r[0]
    if v == float('inf'):
        # a winning pair: apply it to read the exact margin
        n0 = len(b.moves)
        for m in pair:
            if m not in ((0, 0, 0), (1, 1, 1)):
                b.apply_move(m, switch_turn=False)
        w, s = b.check_game_over()
        while len(b.moves) > n0:
            b.undo_last_move()
        b.game_stages.update(st)
        return (s / PTS if w == player else -s / PTS), pair
    if pair == ((1, 1, 1), (0, 0, 0)):
        return 0.0, pair
    return v / scale, pair


def _playout(path, state, mover, seed, H):
    """Per ply: (E over rolls, value of the roll thrown), both in MOVER's frame,
    raw units; plus how it ended."""
    from game import Board
    ag, scale = _agent(path)
    b = Board()
    b.update_state(state)
    rng = random.Random(seed)
    plies = []
    for t in range(H):
        w, s = b.check_game_over()
        if w:
            return plies, (s / PTS if w == mover else -s / PTS)
        b.switch_turn()
        p = b.current_player
        sign = 1.0 if p == mover else -1.0
        vals = {}
        for d1, d2, _ in ROLLS:
            vals[(d1, d2)] = _best(ag, scale, b, p, d1, d2)
        e = sum(pr * vals[(d1, d2)][0] for d1, d2, pr in ROLLS)
        r = tuple(sorted((rng.randint(1, 6), rng.randint(1, 6))))
        v_r, pair = vals[r]
        plies.append((sign * e, sign * v_r))
        b.dice[0].number, b.dice[0].used = r[0], False
        b.dice[1].number, b.dice[1].used = r[1], False
        b._reachable_cache.clear()
        b.get_valid_moves()
        if pair == ((1, 1, 1), (0, 0, 0)):
            return plies, 0.0
        for m in pair:
            if m != (0, 0, 0):
                b.apply_move(m, switch_turn=False)
    w, s = b.check_game_over()
    return plies, ((s / PTS if w == mover else -s / PTS) if w else None)


def _job(task):
    path, idx, side, i, H = task
    it = json.load(open(BENCH))[idx]
    st = it['state_' + side]
    if isinstance(st, str):
        st = ast.literal_eval(st)
    t0 = time.time()
    plies, end = _playout(path, st, it['mover'], 1_000_003 * idx + i, H)
    return {'idx': idx, 'side': side, 'i': i, 'plies': plies, 'end': end,
            'secs': round(time.time() - t0, 1)}


def _init():
    os.environ['BOARDGAME_DEVICE'] = 'cpu'


def run(path, n_pos=120, N=6, H=6, out='luck_probe.jsonl'):
    path = os.path.abspath(path)
    items = json.load(open(BENCH))
    idxs = sorted(random.Random(0).sample(range(len(items)), n_pos))
    done = set()
    if os.path.exists(out):
        done = {(r['idx'], r['side'], r['i']) for r in map(json.loads, open(out))}
    tasks = [(path, k, s, i, H) for k in idxs for i in range(N) for s in 'AB'
             if (k, s, i) not in done]
    print(f'{len(tasks)} playouts to run ({len(done)} done), H={H}', flush=True)
    with mp.Pool(N_WORKERS, initializer=_init) as pool, open(out, 'a') as f:
        for j, r in enumerate(pool.imap_unordered(_job, tasks, chunksize=1), 1):
            f.write(json.dumps(r) + '\n'); f.flush()
            if j % 50 == 0:
                print(f'  {j}/{len(tasks)}', flush=True)
    analyze(out)


def _estimates(r, h):
    """(adjusted, plain) estimate at depth h, mover frame, POINTS."""
    plies = r['plies']
    if len(plies) < h:                       # the game ended earlier: exact
        v = r['end'] if r['end'] is not None else (plies[-1][1] if plies else 0.0)
        luck = sum(vr - e for e, vr in plies)
        return PTS * (v - luck), PTS * v
    v = plies[h - 1][1]
    luck = sum(vr - e for e, vr in plies[:h])
    return PTS * (v - luck), PTS * v


def analyze(out):
    rows = [json.loads(l) for l in open(out)]
    items = json.load(open(BENCH))
    by = {}
    for r in rows:
        by.setdefault(r['idx'], {}).setdefault(r['i'], {})[r['side']] = r
    H = max(len(r['plies']) for r in rows)
    print(f'{out}: {len(rows)} playouts, {len(by)} positions, '
          f'{np.mean([r["secs"] for r in rows]):.0f} s per playout (single core)')
    oct_sd = np.median([it['playout_se'] * math.sqrt(20) for it in items])
    print(f'October plain full playouts: per-playout SD of B-A ~{oct_sd:.2f} pts '
          f'(median), SE at 20 = {oct_sd / math.sqrt(20):.2f}')
    print(f'{"H":>2} {"est":6s} {"SD(B-A)":>8s} {"SE@6":>6s} {"SE@20":>6s} '
          f'{"corr vs Oct":>11s} {"adj-plain mean":>14s}')
    for h in range(1, H + 1):
        for kind in (0, 1):
            sds, means, octs, shifts = [], [], [], []
            for idx, runs in by.items():
                d = [_estimates(v['B'], h)[kind] - _estimates(v['A'], h)[kind]
                     for v in runs.values() if 'A' in v and 'B' in v]
                if len(d) < 2:
                    continue
                sds.append(np.std(d, ddof=1))
                means.append(np.mean(d))
                octs.append(items[idx]['playout_diff'])
                if kind == 0:
                    shifts.append(np.mean([_estimates(v[s], h)[0] - _estimates(v[s], h)[1]
                                           for v in runs.values() for s in 'AB' if s in v]))
            sd = float(np.sqrt(np.mean(np.square(sds))))
            c = float(np.corrcoef(means, octs)[0, 1]) if np.std(means) > 0 else float('nan')
            print(f'{h:>2} {"luck" if kind == 0 else "plain":6s} {sd:8.3f} {sd / math.sqrt(6):6.3f} '
                  f'{sd / math.sqrt(20):6.3f} {c:+11.3f} '
                  + (f'{np.mean(shifts):+14.3f}' if kind == 0 else ''))


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        a = sys.argv[2:]
        run(a[0], int(a[1]) if len(a) > 1 else 120, int(a[2]) if len(a) > 2 else 6,
            int(a[3]) if len(a) > 3 else 6, a[4] if len(a) > 4 else 'luck_probe.jsonl')
    else:
        analyze(sys.argv[2])
