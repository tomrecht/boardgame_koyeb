"""quiz_playout.py -- check the quiz's confirmed mistakes by playing them out.

For each quiz position, owner's game move (A) and the champion's move (B) are
each applied and the game is played to the END N times, both sides moved by the
app's own search on the given net (prefilter 12/40/5, hand rules). Playout i of
A and playout i of B throw the SAME dice turn by turn (common random numbers),
so luck mostly cancels in the paired difference B - A (final margin, mover's
side, points). No value estimate is involved at the end: the result is the real
final score, though both sides still play by the net's judgement.

    python3 quiz_playout.py prep <tasks.jsonl.gz> [min_gap=0.3]   (needs the games; local)
    python3 quiz_playout.py run <net.pt> <tasks.jsonl.gz> <out.jsonl> [count]
            SEED_BASE = first task index, QP_N playouts per arm (default 24), N_WORKERS
    python3 quiz_playout.py stats <out.jsonl>...

Tasks carry the position as serialize_board() (positions only, no game records).
"""
import copy, glob, gzip, json, os, random, sys, time, zlib

os.environ.setdefault('GATE_PREFILTER', '12,40,5')
N = int(os.environ.get('QP_N', '24'))
N_WORKERS = int(os.environ.get('N_WORKERS', '4'))
MAX_TURNS = 300
_AG = {}
_TASKS = None


def prep(out, min_gap=0.3):
    import ast
    import review_games as RG
    from game_worker import serialize_board
    idx = json.load(open('review/quiz/index.json'))['items']
    rows = {('%s_%d' % (r['game'][:8], r['turn'])): r
            for r in map(json.loads, open('review/verify_cand13.jsonl')) if 'deep_gap' in r}
    games = {g['id']: g for g in RG.load_games()}
    ids = sorted((i for i in idx if rows[i]['deep_gap'] >= min_gap),
                 key=lambda i: zlib.crc32(i.encode()))          # mixed order: shards get a spread
    n = 0
    with gzip.open(out, 'wt') as fo:
        for i in ids:
            r = rows[i]
            b = RG._state_at(games[r['game']], r['turn'])
            fo.write(json.dumps({'idx': n, 'id': i, 'g': r['deep_gap'], 'gap': r['gap'],
                                 'mover': b.current_player, 'state': serialize_board(b),
                                 'own': r['own_pair'], 'best': r['best_pair']}) + '\n')
            n += 1
    print('tasks', n)


def _agent(path):
    if path not in _AG:
        import torch
        from network import model_from_state
        from agent_gnn import GNNAgent, prefilter_kwargs
        m = model_from_state(torch.load(path, map_location='cpu'), strict=False)
        _AG[path] = GNNAgent(model=m, hand_rules=True, **prefilter_kwargs(os.environ['GATE_PREFILTER']))
    return _AG[path]


def _board(state):
    from game import Board
    b = Board()
    b.update_state(state)
    b._reachable_cache.clear()
    b.get_valid_moves()
    return b


def _switch(c, key, k):
    random.seed(zlib.crc32(f'{key}:{k}'.encode()))       # same dice for both arms, turn by turn
    c.switch_turn()


def _playout(ag, b, us, key):
    c = copy.deepcopy(b)
    _switch(c, key, 0)
    for k in range(1, MAX_TURNS):
        w, s = c.check_game_over()
        if w:
            return s if w == us else -s
        if c.draw_callable:
            return 0
        mv = list(c.get_valid_moves())
        st = dict(c.game_stages)
        pr = ag.select_move_pair(mv, c, c.current_player)
        c.game_stages.update(st)
        if isinstance(pr, tuple) and len(pr) == 3:
            pr = (pr, (0, 0, 0))
        if pr and pr[0] == (1, 1, 1):
            return 0
        for m in pr:
            if m not in ((0, 0, 0), (1, 1, 1)):
                c.apply_move(m, switch_turn=False)
        _switch(c, key, k)
    return 0


def _arm(ag, b, us, pair, key):
    import ast
    c = copy.deepcopy(b)
    for m in ast.literal_eval(pair):
        if m == (1, 1, 1):
            return [0] * N                                # a called draw ends the game at 0
        if m != (0, 0, 0):
            c.apply_move(m, switch_turn=False)
    return [_playout(ag, c, us, f'{key}:{i}') for i in range(N)]


def _job(args):
    path, t = args
    ag = _agent(path)
    t0 = time.time()
    b = _board(t['state'])
    a = _arm(ag, b, t['mover'], t['own'], t['id'])
    bb = _arm(ag, b, t['mover'], t['best'], t['id'])
    return {'idx': t['idx'], 'id': t['id'], 'g': t['g'], 'A': a, 'B': bb,
            'sec': round(time.time() - t0, 1)}


def run(path, tpath, out, count=None):
    from multiprocessing import Pool
    tasks = [json.loads(l) for l in gzip.open(tpath, 'rt')]
    lo = int(os.environ.get('SEED_BASE', '0'))
    hi = len(tasks) if count is None else min(len(tasks), lo + int(count))
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)['idx'] for l in open(out) if l.strip()}
    todo = [(path, t) for t in tasks[lo:hi] if t['idx'] not in done]
    print(f'{len(todo)} positions, {N} playouts per arm, {N_WORKERS} workers', flush=True)
    with Pool(N_WORKERS) as pool, open(out, 'a') as fo:
        for r in pool.imap_unordered(_job, todo):
            fo.write(json.dumps(r) + '\n'); fo.flush()
            d = sum(r['B']) / N - sum(r['A']) / N
            print(f"{r['id']} claim {r['g']:+.2f} playouts {d:+.2f} ({r['sec']}s)", flush=True)


def stats(paths):
    import numpy as np
    rs = [json.loads(l) for p in paths for l in open(p) if l.strip()]
    g = np.array([r['g'] for r in rs])
    d = np.array([np.mean(r['B']) - np.mean(r['A']) for r in rs])
    se = np.array([np.std(np.array(r['B']) - np.array(r['A']), ddof=1) / np.sqrt(len(r['A'])) for r in rs])
    print(f'{len(rs)} positions; claimed mean {g.mean():+.3f}, playouts mean {d.mean():+.3f} '
          f'(+-{1.96 * d.std(ddof=1) / np.sqrt(len(d)):.3f}); corr {np.corrcoef(g, d)[0, 1]:+.3f}; '
          f'median per-position SE {np.median(se):.2f}')
    for lo, hi in ((0.3, 0.5), (0.5, 1.0), (1.0, 99)):
        m = (g >= lo) & (g < hi)
        if m.sum():
            dm = d[m]
            print(f'  claim {lo}-{hi}: n {m.sum():4d}  claimed {g[m].mean():+.2f}  playouts {dm.mean():+.2f} '
                  f'(+-{1.96 * dm.std(ddof=1) / np.sqrt(m.sum()):.2f})  champion ahead {np.mean(dm > 0):.0%}  '
                  f'clearly ahead {np.mean(d[m] > 1.96 * se[m]):.0%}  clearly behind {np.mean(d[m] < -1.96 * se[m]):.0%}')


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'prep':
        prep(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.3)
    elif cmd == 'run':
        run(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else None)
    elif cmd == 'stats':
        stats(sys.argv[2:])
