"""calib_search.py -- which close-call search makes the best TEACHER?

For distillation the search's VALUES matter, not just its pick, so judge each
search by how well its move gap tracks reality on the calibration benchmark
(calib_bench.json: 559 of owner's positions, his move A vs the old champion's
best B, each with a 20-playout difference B - A).

For each position and arm the net scores the end-of-turn position (PLAIN, the
shallow search's own number) and, per opponent roll, minus the opponent's best
reply (the app's search, as close_call._reply_value). All 21 rolls are stored,
so one run gives every teacher:

    plain      the net's afterstate value
    fixed34    one roll, always 3-4 (what the app and MAX_REPLIES=3 use)
    random1    one roll drawn by its odds (CC_ROLL=random); the score is the
               average over 2,000 redraws, with its spread
    full21     the probability-weighted average of all 21 (distill_run's labels)

Score = corr and slope of the gap (margin points) vs the playout difference;
paired bootstrap over positions for the differences between teachers.

    python3 calib_search.py run <net.pt> [out.jsonl]     resumable, N_WORKERS
    python3 calib_search.py analyze <out.jsonl>

Cost: ~28 s per position single core (smoke, 3 positions), so ~4.3 h of CPU,
~35 min on 8 workers.

Caveats: the playout labels are noisy (median SE ~0.6 per position) and came
from the old blend's 1-ply policy, so this separates big differences only.
calib_bench states carry no no-save counter (both arms read 0).
"""
import json, math, os, random, sys, time
import multiprocessing as mp

os.environ.setdefault('GATE_PREFILTER', '12,40,5')
N_WORKERS = int(os.environ.get('N_WORKERS', '8'))
REPO = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(REPO, 'calib_bench.json')
PTS = 12.0            # raw net units -> margin points (NUM_PIECES), as calib_bench

_ITEMS = None


def _items():
    global _ITEMS
    if _ITEMS is None:
        _ITEMS = json.load(open(BENCH))
    return _ITEMS


def _arm_values(agent, board, state, mover):
    """(plain, {roll: value}) in margin points, mover's frame."""
    from agent_gnn import NUM_PIECES, SCORE_SCALE
    from close_call import ROLLS
    board.update_state(state)
    board.current_player = mover
    assert len(board.moves) == 0
    plain = float(agent.model(agent._encode_after(board, mover))) * PTS
    opp = 'white' if mover == 'black' else 'black'
    mine = len(board.white_saved if mover == 'white' else board.black_saved)
    win_value = -(NUM_PIECES - mine) / NUM_PIECES * SCORE_SCALE
    snap = agent._enter_opponent_turn_deterministic(board)
    rolls = {}
    agent._inner = True
    try:
        for d1, d2, _ in ROLLS:
            v = agent._reply_value(board, opp, d1, d2, win_value)
            rolls[f'{d1}{d2}'] = v / SCORE_SCALE * PTS
    finally:
        agent._inner = False
        agent._restore_turn_state(board, snap)
    return plain, rolls


def _job(task):
    path, i = task
    from close_call import _agents
    from game import Board
    agent, _ = _agents(path, 3.0, 3)
    it = _items()[i]
    board = Board()
    t0 = time.time()
    out = {'i': i, 'game': it['game'], 'turn': it['turn'],
           'playout_diff': it['playout_diff'], 'playout_se': it['playout_se']}
    for arm in ('A', 'B'):
        p, r = _arm_values(agent, board, it['state_' + arm], it['mover'])
        out['plain_' + arm], out['rolls_' + arm] = p, r
    out['secs'] = time.time() - t0
    return out


def run(path, out=None):
    path = os.path.abspath(path)
    out = out or 'calib_search.jsonl'
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)['i'] for l in open(out) if l.strip()}
    tasks = [(path, i) for i in range(len(_items())) if i not in done]
    print(f'{len(tasks)} positions to score ({len(done)} done), net {os.path.basename(path)}', flush=True)
    from arena import _worker_init
    t0 = time.time()
    with mp.Pool(N_WORKERS, initializer=_worker_init) as pool, open(out, 'a') as f:
        for k, r in enumerate(pool.imap_unordered(_job, tasks, chunksize=1), 1):
            f.write(json.dumps(r) + '\n'); f.flush()
            if k % 50 == 0 or k == len(tasks):
                print(f'  {k}/{len(tasks)}  {(time.time() - t0) / k:.2f} s/position wall', flush=True)
    analyze(out)


def _corr_slope(g, y):
    n = len(g)
    mg, my = sum(g) / n, sum(y) / n
    cov = sum((a - mg) * (b - my) for a, b in zip(g, y))
    vg = sum((a - mg) ** 2 for a in g)
    vy = sum((b - my) ** 2 for b in y)
    if vg == 0 or vy == 0:
        return float('nan'), float('nan')
    return cov / math.sqrt(vg * vy), cov / vg


def analyze(path):
    from close_call import ROLLS
    rows = [json.loads(l) for l in open(path) if l.strip()]
    n = len(rows)
    y = [r['playout_diff'] for r in rows]
    keys = [f'{a}{b}' for a, b, _ in ROLLS]
    p = {f'{a}{b}': pr for a, b, pr in ROLLS}
    roll_gap = [{k: r['rolls_B'][k] - r['rolls_A'][k] for k in keys} for r in rows]
    gaps = {
        'plain': [r['plain_B'] - r['plain_A'] for r in rows],
        'fixed34': [g['34'] for g in roll_gap],
        'full21': [sum(p[k] * g[k] for k in keys) for g in roll_gap],
    }
    rng = random.Random(0)
    weights = [p[k] for k in keys]

    def draw():
        return [g[rng.choices(keys, weights)[0]] for g in roll_gap]

    print(f'{path}: {n} positions')
    res = {name: _corr_slope(g, y) for name, g in gaps.items()}
    rc = [_corr_slope(draw(), y) for _ in range(2000)]
    rc = [(c, s) for c, s in rc if c == c]
    rcs = sorted(c for c, _ in rc)
    res['random1'] = (sum(rcs) / len(rcs), sum(s for _, s in rc) / len(rc))
    for name in ('plain', 'fixed34', 'random1', 'full21'):
        c, s = res[name]
        extra = (f'   (redraws 5-95%: {rcs[len(rcs) // 20]:+.3f} .. {rcs[-len(rcs) // 20]:+.3f})'
                 if name == 'random1' else '')
        print(f'  {name:<8} corr {c:+.3f}  slope {s:+.3f}{extra}')
    # paired bootstrap over positions: full21 vs each, fixed34 vs plain
    B = 2000
    pairs = [('full21', 'fixed34'), ('full21', 'plain'), ('fixed34', 'plain')]
    diffs = {pr: [] for pr in pairs}
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        yy = [y[i] for i in idx]
        cs = {name: _corr_slope([gaps[name][i] for i in idx], yy)[0] for name in gaps}
        for a, b in pairs:
            diffs[(a, b)].append(cs[a] - cs[b])
    for (a, b), d in diffs.items():
        d = sorted(x for x in d if x == x)          # drop NaN (no spread in a resample)
        m = len(d)
        print(f'  corr {a} - {b}: {res[a][0] - res[b][0]:+.3f} '
              f'(95% bootstrap {d[int(.025 * m)]:+.3f} .. {d[min(m - 1, int(.975 * m))]:+.3f})')
    print(f'  {sum(r["secs"] for r in rows) / n:.1f} s per position (single core)')


if __name__ == '__main__':
    if sys.argv[1:2] == ['run']:
        run(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    elif sys.argv[1:2] == ['analyze']:
        analyze(sys.argv[2])
    else:
        sys.exit(__doc__)
