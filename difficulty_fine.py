"""difficulty_fine.py -- the fine sweep of the difficulty slider's band.

CLAUDE.md, "THE DIFFICULTY SLIDER IS REMAPPED ONTO 0.8..1.0": only the endpoints
were measured (1.0 = 50% against itself, 0.8 = 0 of 94), so the slider's labels
carry no numbers. This measures the band in the configuration the APP ships --
ONNX net, prefilter F=12 / K=40 / min 5, hand rules on -- at ~22 s a game, ten
times faster than difficulty_arena.py's torch path.

Each game: one side at effective difficulty d, the other at 1.0, paired seeds
with colours swapped. Both generators seeded per game (dice: random; the
difficulty sampling: numpy). Appends to difficulty_fine.jsonl; resumes.

    python3 difficulty_fine.py [pairs_per_level=60]
    python3 difficulty_fine.py analyze
"""
import json, math, os, random, statistics, sys
import multiprocessing as mp

REPO = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get('MODEL', f'{REPO}/model.onnx')
OUT = os.path.join(REPO, 'difficulty_fine.jsonl')
LEVELS = [float(x) for x in os.environ.get('LEVELS', '0.99,0.97,0.95,0.92,0.90,0.85').split(',')]
N_WORKERS = int(os.environ.get('N_WORKERS', '4'))
SEED_BASE = 9_900_000
MAX_TURNS, STUCK_LIMIT = 200, 60
_AG = None


def _agent():
    global _AG
    if _AG is None:
        from agent_gnn import GNNAgent
        _AG = GNNAgent(weights_path=MODEL, use_prefilter=True, prefilter_top_k=40,
                       prefilter_min_k=5, first_move_prefilter=12)
    return _AG


def _play(seed, d_white, d_black):
    import numpy as np
    from game import Board
    random.seed(seed)
    np.random.seed(seed % (2 ** 31 - 1))
    ag = _agent()
    board = Board()
    last_saved, since = 0, 0
    for turns in range(MAX_TURNS):
        winner, score = board.check_game_over()
        if winner:
            return winner, score, turns
        if board.draw_callable:
            return None, 0, turns
        cur = len(board.white_saved) + len(board.black_saved)
        if cur > last_saved:
            last_saved, since = cur, 0
        elif last_saved > 0:
            since += 1
        if last_saved > 0 and since >= STUCK_LIMIT:
            return None, 0, turns
        p = board.current_player
        chosen = ag.select_move_pair(list(board.get_valid_moves()), board, p,
                                     difficulty=d_white if p == 'white' else d_black)
        if isinstance(chosen, tuple) and len(chosen) == 3:
            chosen = (chosen, (0, 0, 0))
        for m in chosen:
            if m != (0, 0, 0):
                board.apply_move(m, switch_turn=False)
        board.switch_turn()
    return None, 0, MAX_TURNS


def _worker(task):
    d, seed, weak_white = task
    winner, score, turns = _play(seed, d if weak_white else 1.0, 1.0 if weak_white else d)
    m = 0 if winner is None else (score if (winner == 'white') == weak_white else -score)
    return {'d': d, 'seed': seed, 'weak_white': weak_white, 'winner': winner,
            'weak_margin': m, 'turns': turns}


def analyze():
    rows = [json.loads(l) for l in open(OUT)]
    print(f'{"d":>5} {"slider":>7} {"games":>6} {"win%":>6} {"95% CI":>13} {"margin":>8} {"95% CI":>16}')
    for d in sorted({r['d'] for r in rows}, reverse=True):
        rs = [r for r in rows if r['d'] == d]
        n = len(rs)
        w = sum(r['weak_margin'] > 0 for r in rs)
        p = w / n
        z = 1.96
        den = 1 + z * z / n
        c = (p + z * z / (2 * n)) / den
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        by = {}
        for r in rs:
            by.setdefault(r['seed'], []).append(r['weak_margin'])
        paired = [statistics.fmean(v) for v in by.values() if len(v) == 2]
        m = statistics.fmean(paired)
        se = statistics.stdev(paired) / math.sqrt(len(paired)) if len(paired) > 1 else float('nan')
        pos = (d - 0.8) / 0.2
        print(f'{d:>5.2f} {pos:>6.0%} {n:>6} {100 * p:>5.1f}% {100 * (c - h):>5.1f}-{100 * (c + h):>4.1f}% '
              f'{m:>+8.2f} {m - 1.96 * se:>+7.2f}..{m + 1.96 * se:>+5.2f}')


def main():
    if sys.argv[1:2] == ['analyze']:
        return analyze()
    pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    done = set()
    if os.path.exists(OUT):
        done = {(r['d'], r['seed'], r['weak_white']) for r in map(json.loads, open(OUT))}
    tasks = [(d, SEED_BASE + k, w) for k in range(pairs) for d in LEVELS for w in (True, False)
             if (d, SEED_BASE + k, w) not in done]
    print(f'{len(tasks)} games ({pairs} pairs x {len(LEVELS)} levels), {N_WORKERS} workers', flush=True)
    ctx = mp.get_context('spawn')
    with ctx.Pool(N_WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, tasks, chunksize=1), 1):
            with open(OUT, 'a') as f:
                f.write(json.dumps(r) + '\n')
            if i % 40 == 0:
                print(f'  {i}/{len(tasks)}', flush=True)
    analyze()


if __name__ == '__main__':
    main()
