"""
match_topk.py — how much strength does a smaller prefilter cost?

Motivation: in-browser move latency is median 1.25s / p90 2.97s at 4x CPU
throttle (a mid-range phone), and the agent scores 40 candidates per move --
which is exactly `prefilter_top_k`. Cutting candidates should cut most of that
time, because the encoder's per-candidate BFS dominates, not the forward pass.
The question this answers is what it costs in play.

NOT a knob to turn blind: these settings decide which moves the value net ever
sees. F=12 was adopted only after 120 paired games showed no measurable loss.

Two modes:

    python match_topk.py probe [positions]
        Cheap and fast. No games: walks recorded positions and reports, per
        setting, the candidate count, the seconds per move, and -- the one that
        matters for correctness rather than speed -- how often a save was legal
        but NO save pair survived the cull to reach the net. Save pairs are
        exempt from the top-K cull precisely because the heuristic undervalues
        them; at k=40 that exemption rarely binds, at k=10 it is load-bearing,
        and pass-over-save is already a known rough edge. Run this FIRST: if a
        setting drops saves, no amount of match data makes it acceptable.

    python match_topk.py match [pairs] [top_k[,top_k...]] [F]
        The real measurement. Each candidate setting plays the served baseline
        (top_k=40, F=12) over paired seeds with colours swapped, so dice luck
        cancels within a pair.

Budget the sample BEFORE starting. Paired-margin SD is ~1.74/pair, so ~100 pairs
resolves a 0.5 margin and ~290 resolves 0.3; the arena work is the cautionary
tale here -- the top four champions were statistically inseparable at 9-16 games
per matchup, and three of five gate promotions reversed in open play.

    N_WORKERS=8 MODEL=model.onnx python match_topk.py match 120 24,16,10 12
"""

import math
import os
import random
import statistics
import sys
import time
import multiprocessing as mp

REPO = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get('MODEL', f'{REPO}/model.onnx')
N_WORKERS = int(os.environ.get('N_WORKERS', '8'))
SEED_BASE = 7_900_000            # disjoint from match_prefilter.py's 7_700_000
MAX_TURNS, STUCK_LIMIT = 200, 60

BASELINE = (40, 12)              # what app.py serves: prefilter_top_k, F

_CACHE = {}


def _agent(top_k, F):
    """Agents share one loaded net and encoder: same weights either way, and it
    keeps the memory per worker flat as the sweep widens."""
    key = (top_k, F)
    if key not in _CACHE:
        from agent_gnn import GNNAgent
        a = GNNAgent(weights_path=MODEL, use_prefilter=True,
                     prefilter_top_k=top_k, first_move_prefilter=F)
        base = next(iter(_CACHE.values()), None)
        if base is not None:
            a.backend = base.backend
            a.model = base.backend
            a.encoder = base.encoder
        _CACHE[key] = a
    return _CACHE[key]


def _worker_init():
    os.environ['BOARDGAME_DEVICE'] = 'cpu'


def _has_save(move):
    return isinstance(move, tuple) and len(move) == 3 and move[1] == 'save'


# ---------------------------------------------------------------- probe mode

def _walk_positions(n, seed0=0):
    """Seeded positions spanning opening/midgame/endgame. Canonicalised, both
    because get_valid_moves' order is not reproducible and because the served
    path always rebuilds via update_state -> assign_piece_indices."""
    from game import Board
    from dump_engine_fixture import canon
    out = []
    seed = seed0
    while len(out) < n:
        random.seed(seed)
        seed += 1
        b = Board()
        for _ in range(random.randint(2, 70)):
            if b.check_game_over()[0]:
                break
            for _ in range(2):
                ms = sorted((m for m in b.get_valid_moves() if isinstance(m[0], tuple)),
                            key=canon)
                if not ms:
                    break
                saves = [m for m in ms if m[1] == 'save']
                pool = saves if (saves and random.random() < 0.75) else ms
                b.apply_move(random.choice(pool), switch_turn=False)
            b.switch_turn()
        if b.check_game_over()[0] or not b.get_valid_moves():
            continue
        b.assign_piece_indices()
        out.append(b)
    return out


def probe(n_positions, settings):
    boards = _walk_positions(n_positions)
    print(f'{len(boards)} positions\n')
    print(f'{"top_k":>6} {"F":>3} {"cand med":>9} {"cand max":>9} '
          f'{"s/move med":>11} {"s/move p90":>11} {"save dropped":>13}')
    for top_k, F in settings:
        a = _agent(top_k, F)
        counts, times, save_legal, save_dropped = [], [], 0, 0
        for b in boards:
            moves = list(b.get_valid_moves())
            player = b.current_player
            t = time.perf_counter()
            ranked = a.select_move_pair(moves, b, player, return_scores=True)
            times.append(time.perf_counter() - t)
            counts.append(len(ranked))
            if any(_has_save(m) for m in moves):
                save_legal += 1
                # Did any pair carrying a save survive to be scored?
                if not any(any(_has_save(m) for m in pair) for _, pair in ranked):
                    save_dropped += 1
        q = lambda a_, p: sorted(a_)[min(len(a_) - 1, int(len(a_) * p))]
        drop = f'{save_dropped}/{save_legal}' if save_legal else 'n/a'
        print(f'{top_k:>6} {F:>3} {q(counts, .5):>9} {max(counts):>9} '
              f'{q(times, .5):>11.3f} {q(times, .9):>11.3f} {drop:>13}')
    print('\n"save dropped" = positions where a save was legal but no save pair '
          'reached the net.\nAnything above zero is disqualifying on its own.')


# ---------------------------------------------------------------- match mode

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
    seed, small_white, top_k, F = task
    small, base = _agent(top_k, F), _agent(*BASELINE)
    winner, score = _play(seed, small if small_white else base,
                          base if small_white else small)
    if winner is None:
        margin = 0
    else:
        small_won = (winner == 'white') == small_white
        margin = score if small_won else -score
    return {'seed': seed, 'top_k': top_k, 'F': F,
            'winner': winner, 'margin': margin}


def match(pairs, settings):
    tasks = []
    for top_k, F in settings:
        for i in range(pairs):
            s = SEED_BASE + i
            # The SAME dice seed for both colour assignments: that pairing is
            # what cancels dice luck, and it is why this resolves far more per
            # game than an unpaired match would.
            tasks.append((s, True, top_k, F))
            tasks.append((s, False, top_k, F))
    print(f'{len(tasks)} games: {pairs} pairs x {len(settings)} settings, '
          f'against baseline top_k={BASELINE[0]} F={BASELINE[1]}')

    ctx = mp.get_context('spawn')      # fork deadlocked on CUDA contexts before
    with ctx.Pool(N_WORKERS, initializer=_worker_init) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(_worker, tasks), 1):
            results.append(r)
            if i % 20 == 0:
                print(f'  {i}/{len(tasks)}', flush=True)

    print(f'\n{"top_k":>6} {"F":>3} {"win%":>7} {"margin":>8} {"95% CI":>18} {"pairs":>6}')
    for top_k, F in settings:
        rs = [r for r in results if (r['top_k'], r['F']) == (top_k, F)]
        decisive = [r for r in rs if r['winner'] is not None]
        wins = sum(1 for r in decisive if r['margin'] > 0)
        by_seed = {}
        for r in rs:
            by_seed.setdefault(r['seed'], []).append(r['margin'])
        paired = [statistics.fmean(v) for v in by_seed.values() if len(v) == 2]
        m = statistics.fmean(paired) if paired else float('nan')
        sd = statistics.stdev(paired) if len(paired) > 1 else float('nan')
        se = sd / math.sqrt(len(paired)) if paired else float('nan')
        wr = 100 * wins / len(decisive) if decisive else float('nan')
        print(f'{top_k:>6} {F:>3} {wr:>6.1f}% {m:>+8.3f} '
              f'{m - 1.96 * se:>+8.3f}..{m + 1.96 * se:>+8.3f} {len(paired):>6}')
    print('\nMargin is from the SMALLER setting\'s point of view: negative means '
          'it plays worse.\nA CI straddling 0 means this sample could not '
          'separate them -- which is\nthe answer only if the CI is also narrow '
          'enough to rule out a drop you care about.')


def _settings(arg, F):
    return [(int(k), F) for k in arg.split(',')]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'probe'
    if mode == 'probe':
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        ks = sys.argv[3] if len(sys.argv) > 3 else '40,24,16,10'
        Fs = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else '12,8,6').split(',')]
        settings = [(int(k), F) for F in Fs for k in ks.split(',')]
        probe(n, settings)
    elif mode == 'match':
        pairs = int(sys.argv[2]) if len(sys.argv) > 2 else 120
        ks = sys.argv[3] if len(sys.argv) > 3 else '24,16,10'
        F = int(sys.argv[4]) if len(sys.argv) > 4 else 12
        match(pairs, _settings(ks, F))
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == '__main__':
    main()
