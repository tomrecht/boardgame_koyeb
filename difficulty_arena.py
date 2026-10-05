"""What does the difficulty slider actually PLAY like?

Settings > Difficulty is a number in [0,1] that game.js hands the agent, where
1.0 = argmax / full strength and lower means top-p sampling over a z-scored
softmax (`_pick_move_index` in agent_gnn.py; `pickMoveIndex` in agent.js, which
is the same formulas -- same temp/top_p ramps, same population std -- so a
Python measurement labels the SHIPPED slider honestly).

Nobody had ever measured what a given setting plays like, so the slider's ends
said "Max" and "Easy" and its middle said "55%", which tells a player nothing.
This measures the one number a label can be built from: **the win rate of a
weakened agent against the SAME net at full strength.**

Method: the deployed champion (symaug_champ_July27_iter6.pt) on both sides, one
side at difficulty d and the other at 1.0, over paired colour-swapped seeds
(CRN -- the same dice sequence played from both colours, which removes most of
the luck). Plain 2-ply shallow, the deployed policy. One model instance serves
both sides, since difficulty is a per-call argument.

Resumable: appends to difficulty_arena.jsonl and skips games already recorded,
so it can be stopped and extended.

Usage:
  python -u difficulty_arena.py                 # play rounds until stopped
  python difficulty_arena.py analyze            # table from the jsonl
Env: N_WORKERS (default 4), N_SEEDS (seed pairs per setting per pass, default 25),
     LEVELS="1.0,0.9,..." to override the sweep, NET=<path>.
"""
import os, sys, json, time, random
import multiprocessing as mp

REPO = os.path.dirname(os.path.abspath(__file__))
NET = os.environ.get('NET', f'{REPO}/symaug_champ_July27_iter6.pt')
# The slider is min=0 max=100 step=5, so every value here is one the player can
# actually select. d=1.0 is the CONTROL: argmax against argmax, which must come
# out near 50% or the harness itself is asymmetric.
# Six points, not twenty-one: the slider needs BANDS with honest labels, and a
# win rate good to a few percent at six settings costs a fifth of what sweeping
# every step would. 1.0 is the control -- argmax against argmax, which must land
# near 50% or the harness itself is asymmetric.
LEVELS = [float(x) for x in os.environ.get(
    'LEVELS', '1.0,0.8,0.6,0.4,0.2,0.0').split(',')]
N_SEEDS = int(os.environ.get('N_SEEDS', '25'))
N_WORKERS = int(os.environ.get('N_WORKERS', '4'))
RESULTS = os.environ.get('RESULTS', f'{REPO}/difficulty_arena.jsonl')
SEED_BASE = 7_700_000
MAX_TURNS, STUCK_LIMIT = 200, 60

_CACHE = {}


def _agent(path):
    if path not in _CACHE:
        import torch
        from network import BoardGNN
        from agent_gnn import GNNAgent
        m = BoardGNN()
        m.load_state_dict(torch.load(path, map_location='cpu'), strict=False)
        m.eval()
        _CACHE[path] = GNNAgent(model=m)
    return _CACHE[path]


def _worker_init():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['BOARDGAME_DEVICE'] = 'cpu'
    import torch
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    import network
    network.DEVICE = torch.device('cpu')


def _play(seed, agent, d_white, d_black):
    import numpy as np
    from game import Board
    # BOTH generators: the dice come from `random`, the difficulty sampling from
    # numpy. Seeding only one makes a "paired" seed unpaired on the sampled side.
    random.seed(seed)
    np.random.seed(seed % (2 ** 31 - 1))
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
        player = board.current_player
        chosen = agent.select_move_pair(
            list(board.get_valid_moves()), board, player,
            difficulty=(d_white if player == 'white' else d_black))
        if isinstance(chosen, tuple) and len(chosen) == 3:
            chosen = (chosen, (0, 0, 0))
        for m in chosen:
            if m != (0, 0, 0):
                board.apply_move(m, switch_turn=False)
        board.switch_turn()
    return None, 0, MAX_TURNS


def worker(task):
    d, seed, weak_white = task
    ag = _agent(NET)
    t0 = time.time()
    # `weak` is the side at difficulty d; the other is always full strength.
    winner, score, turns = _play(seed, ag, d if weak_white else 1.0, 1.0 if weak_white else d)
    if winner is None:
        margin = 0
    else:
        weak_won = (winner == 'white') == weak_white
        margin = score if weak_won else -score
    return {'d': d, 'seed': seed, 'weak_white': weak_white, 'winner': winner,
            'weak_margin': margin, 'turns': turns, 'secs': round(time.time() - t0, 1)}


def load_rows():
    if not os.path.exists(RESULTS):
        return []
    with open(RESULTS) as f:
        return [json.loads(l) for l in f if l.strip()]


def _wilson(k, n):
    """95% CI for a proportion. The normal approximation is wrong at the ends,
    and d=0.0 is expected to sit near one of them."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.959964
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return (max(0.0, c - h), min(1.0, c + h))


def analyze(rows=None):
    rows = rows if rows is not None else load_rows()
    if not rows:
        print('no games recorded yet')
        return
    by = {}
    for r in rows:
        by.setdefault(r['d'], []).append(r)
    print(f'\n{len(rows)} games, net = {os.path.basename(NET)}')
    print('The "weak" side plays at difficulty d; the other side is always d=1.0.\n')
    print('   d   slider  games   win%  (95% CI)        draw%   avg margin   pts/game   turns')
    print('  ' + '-' * 81)
    for d in sorted(by, reverse=True):
        g = by[d]
        n = len(g)
        w = sum(1 for r in g if r['weak_margin'] > 0)
        l = sum(1 for r in g if r['weak_margin'] < 0)
        dr = n - w - l
        lo, hi = _wilson(w, n)
        avg = sum(r['weak_margin'] for r in g) / n
        # What a MATCH would pay it: a match is decided on total score, so the
        # weak side's points per game is the figure that matters over a match,
        # not the win rate.
        pts = sum(r['weak_margin'] for r in g if r['weak_margin'] > 0) / n
        label = 'Max' if d >= 0.99 else ('Easy' if d <= 0.01 else f'{round(d*100)}%')
        tn = [r['turns'] for r in g if 'turns' in r]
        tstr = f'{sum(tn)/len(tn):5.0f}' if tn else '    -'
        print(f'  {d:4.2f}  {label:>5}  {n:5d}  {100*w/n:5.1f}  '
              f'({100*lo:4.1f}-{100*hi:4.1f})  {100*dr/n:6.1f}   {avg:+7.2f}    {pts:5.2f}   {tstr}')
    print('\n  win% counts a draw as neither. avg margin is from the weak side, in')
    print('  pieces (the same unit as a game score). pts/game is what it would')
    print('  actually score in a match, where the total is what decides it. turns is')
    print('  the game LENGTH, which is a cost of a lower setting in its own right: a')
    print('  gentler opponent that makes the game drag is its own bad first game.')


def main():
    rows = load_rows()
    done = {(r['d'], r['seed'], r['weak_white']) for r in rows}
    print(f'{len(rows)} games already recorded in {os.path.basename(RESULTS)}')
    rnd = 0
    while True:
        rnd += 1
        tasks = []
        for i in range(N_SEEDS):
            seed = SEED_BASE + (rnd - 1) * N_SEEDS + i
            for d in LEVELS:
                for weak_white in (True, False):      # colour-swapped pair, same seed
                    key = (d, seed, weak_white)
                    if key not in done:
                        tasks.append((d, seed, weak_white))
        if not tasks:
            continue
        print(f'\n--- pass {rnd}: {len(tasks)} games on {N_WORKERS} workers '
              f'({time.strftime("%H:%M:%S")})', flush=True)
        t0 = time.time()
        with mp.Pool(N_WORKERS, initializer=_worker_init) as pool, open(RESULTS, 'a') as out:
            n = 0
            for rec in pool.imap_unordered(worker, tasks, chunksize=1):
                out.write(json.dumps(rec) + '\n')
                out.flush()
                rows.append(rec)
                done.add((rec['d'], rec['seed'], rec['weak_white']))
                n += 1
                if n % 12 == 0:
                    el = time.time() - t0
                    print(f'    {n}/{len(tasks)}  {el/n:.1f}s per game  '
                          f'eta {(len(tasks)-n)*el/n/60:.1f} min', flush=True)
        analyze(rows)
        sys.stdout.flush()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'analyze':
        analyze()
    else:
        main()
