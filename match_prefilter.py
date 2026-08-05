"""
match_prefilter.py — does the two-stage prefilter cost any strength?

Same weights on both sides; the only difference is how candidates are filtered
before the net scores them:

    A  first_move_prefilter=0   every pair scored by the heuristic (current)
    B  first_move_prefilter=F   first moves ranked first, only the best F expanded

Paired seeds with colours swapped (common random numbers), so dice luck cancels
across each pair and the comparison is far sharper per game than it looks.

    python match_prefilter.py [F] [pairs]        # default F=12, 120 pairs
"""

import json
import math
import os
import random
import statistics
import sys
import multiprocessing as mp

REPO = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get('MODEL', f'{REPO}/model.onnx')
N_WORKERS = int(os.environ.get('N_WORKERS', '8'))
SEED_BASE = 7_700_000
MAX_TURNS, STUCK_LIMIT = 200, 60

_CACHE = {}


def _agent(first_move_prefilter):
    """Both agents share one loaded net and one encoder -- same weights, and it
    halves the memory per worker."""
    if first_move_prefilter not in _CACHE:
        from agent_gnn import GNNAgent
        a = GNNAgent(weights_path=MODEL, use_prefilter=True, prefilter_top_k=40,
                     first_move_prefilter=first_move_prefilter)
        base = next(iter(_CACHE.values()), None)
        if base is not None:
            a.backend = base.backend
            a.model = base.backend
            a.encoder = base.encoder
        _CACHE[first_move_prefilter] = a
    return _CACHE[first_move_prefilter]


def _worker_init():
    os.environ['BOARDGAME_DEVICE'] = 'cpu'


def _play(seed, white_agent, black_agent):
    from game import Board
    random.seed(seed)
    board = Board()
    agents = {'white': white_agent, 'black': black_agent}
    last_saved, since = 0, 0
    for _ in range(MAX_TURNS):
        winner, score = board.check_game_over()
        if winner:
            return winner, score
        if board.draw_callable:
            return None, 0
        cur = len(board.white_saved) + len(board.black_saved)
        if cur > last_saved:
            last_saved, since = cur, 0
        elif last_saved > 0:
            since += 1
        if last_saved > 0 and since >= STUCK_LIMIT:
            return None, 0
        player = board.current_player
        chosen = agents[player].select_move_pair(
            list(board.get_valid_moves()), board, player)
        if isinstance(chosen, tuple) and len(chosen) == 3:
            chosen = (chosen, (0, 0, 0))
        for m in chosen:
            if m != (0, 0, 0):
                board.apply_move(m, switch_turn=False)
        board.switch_turn()
    return None, 0


def _worker(task):
    seed, fast_white, F = task
    fast, slow = _agent(F), _agent(0)
    winner, score = _play(seed, fast if fast_white else slow,
                          slow if fast_white else fast)
    if winner is None:
        fast_margin = 0
    else:
        fast_won = (winner == 'white') == fast_white
        fast_margin = score if fast_won else -score
    return {'seed': seed, 'fast_white': fast_white,
            'winner': winner, 'fast_margin': fast_margin}


def main():
    F = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    pairs = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    tasks = [(SEED_BASE + k, w, F) for k in range(pairs) for w in (True, False)]
    print(f'{pairs} paired games ({len(tasks)} total), first_move_prefilter={F} '
          f'vs the current all-pairs prefilter, {N_WORKERS} workers', flush=True)

    ctx = mp.get_context('spawn')
    rows = []
    with ctx.Pool(N_WORKERS, initializer=_worker_init) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, tasks, chunksize=1), 1):
            rows.append(r)
            if i % 40 == 0:
                print(f'  {i}/{len(tasks)} games', flush=True)

    with open(f'{REPO}/match_prefilter.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')

    decisive = [r for r in rows if r['winner'] is not None]
    wins = sum(1 for r in decisive if r['fast_margin'] > 0)
    n = len(decisive)
    # paired unit = one seed, both colours: dice luck cancels within the pair
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r['seed'], []).append(r['fast_margin'])
    paired = [statistics.fmean(v) for v in by_seed.values() if len(v) == 2]
    m = statistics.fmean(paired)
    sd = statistics.stdev(paired) if len(paired) > 1 else float('nan')
    se = sd / math.sqrt(len(paired))
    print(f'\nfast wins {wins}/{n} decisive games ({100*wins/n:.1f}%), '
          f'{len(rows)-n} draws/stuck')
    print(f'mean paired margin for the fast agent: {m:+.3f} '
          f'(95% CI {m-1.96*se:+.3f} .. {m+1.96*se:+.3f}, n={len(paired)} pairs)')
    if abs(m) < 1.96 * se:
        print('-> no measurable strength difference at this sample size')
    else:
        print('-> a real difference at this sample size')
    print(f'   (a drop of {1.96*se:.2f} margin or more would have shown up here)')


if __name__ == '__main__':
    main()
