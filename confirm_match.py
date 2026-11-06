"""confirm_match.py -- fresh-seed head-to-head of two checkpoints, app config.

A gate that replays the same seeds every iteration selects for luck on those
seeds (fv2B: one promotion out of ~55 panel-tested candidates). This plays A vs
B on a seed block no gate has seen, colours swapped per seed (common random
numbers), in the APP's config: prefilter 12/40/5, hand rules ON. Each game is
appended to the output file as it finishes; re-running resumes.

    python3 confirm_match.py A.pt B.pt [pairs=1000] [out=confirm_<A>.jsonl]
    python3 confirm_match.py analyze out.jsonl
"""
import json, math, os, sys
import multiprocessing as mp

SEED_BASE = int(os.environ.get('SEED_BASE', 81_000_000))         # disjoint from gen (11M), panel (77M), pre-screen (78M), confirm gate (79M)
N_WORKERS = int(os.environ.get('N_WORKERS', '3'))
os.environ.setdefault('GATE_PREFILTER', '12,40,5')


_SIDES = {}


def _side(path, tag):
    """The app agent for one side. SCALES_A / SCALES_B (a prefilter scales file)
    give that side its own prefilter ranking -- same net, two filters."""
    f = os.environ.get('SCALES_' + tag)
    if not f:
        from arena import _agent
        return _agent(path, hand_rules=True)
    if tag not in _SIDES:
        import torch
        from network import model_from_state
        from agent_gnn import GNNAgent, prefilter_kwargs
        os.environ['PREFILTER_SCALES_FILE'] = os.path.abspath(f)
        m = model_from_state(torch.load(path, map_location='cpu'), strict=False)
        _SIDES[tag] = GNNAgent(model=m, hand_rules=True, **prefilter_kwargs(os.environ['GATE_PREFILTER']))
        del os.environ['PREFILTER_SCALES_FILE']
    return _SIDES[tag]


# LAYOUT_STATS=1 (layout comparison, owner 2026-11-06): each record also carries
# the game's trajectory, one entry per turn end: [white unsaved, black unsaved,
# goals cut off from home by WHITE's walls, by BLACK's] -- the last two as 6-bit
# masks over the goal tiles in ring order (bit i = ring 7 sector 2i+2), so seal
# statistics are by PHYSICAL tile whatever GOAL_LAYOUT numbers them.
# WEAK_B=1: side B plays select_move_pair_fast (first move by the net, second
# greedy after it) -- a handicap that does not depend on the layout.
LAYOUT_STATS = os.environ.get('LAYOUT_STATS', '0')     # '1' trajectory, '2' + roll values (layout_play.py)
SELF_ONLY = os.environ.get('SELF_ONLY') == '1'         # A white only: for A = B, the twin game is identical
WEAK_B = os.environ.get('WEAK_B') == '1'


class _Fast:
    def __init__(self, ag): self.ag = ag
    def select_move_pair(self, moves, board, player):
        return self.ag.select_move_pair_fast(moves, board, player)


def _cut_mask(board, sealer):
    from collections import deque
    walls = {t for t in board.tiles if t.type == 'field'
             and sum(1 for p in t.pieces if p.player == sealer) >= 2}
    seen = {board.home_tile}; q = deque([board.home_tile])
    while q:
        t = q.popleft()
        for n in t.neighbors:
            if n in seen or n.type in ('nogo', 'home') or n in walls:
                continue
            seen.add(n); q.append(n)
    m = 0
    for t in board.tiles:
        if t.type == 'save' and t not in seen:
            m |= 1 << ((t.pos - 2) // 2)
    return m


def _play_traj(seed, wag, bag):
    """arena._play, recording the trajectory."""
    import random
    from game import Board
    from arena import MAX_TURNS, STUCK_LIMIT
    random.seed(seed)
    board = Board(); agents = {'white': wag, 'black': bag}
    last_saved, since, traj = 0, 0, []
    for t in range(MAX_TURNS):
        winner, score = board.check_game_over()
        if winner:
            return winner, score, traj
        if board.draw_callable:
            return None, 0, traj
        cur = len(board.white_saved) + len(board.black_saved)
        if cur > last_saved:
            last_saved, since = cur, 0
        elif last_saved > 0:
            since += 1
        if last_saved > 0 and since >= STUCK_LIMIT:
            return None, 0, traj
        player = board.current_player
        chosen = agents[player].select_move_pair(list(board.get_valid_moves()), board, player)
        if isinstance(chosen, tuple) and len(chosen) == 3:
            chosen = (chosen, (0, 0, 0))
        for m in chosen:
            if m != (0, 0, 0):
                board.apply_move(m, switch_turn=False)
        traj.append([12 - len(board.white_saved), 12 - len(board.black_saved),
                     _cut_mask(board, 'white'), _cut_mask(board, 'black')])
        board.switch_turn()
    return None, 0, traj


def _game(task):
    a, b, seed, a_white = task
    from arena import _play
    wa, wb = _side(a, 'A'), _side(b, 'B')
    if WEAK_B:
        wb = _Fast(wb)
    traj = None
    if LAYOUT_STATS == '2':
        import layout_play
        winner, score, traj = layout_play.play(seed, wa if a_white else wb, wb if a_white else wa,
                                               judge=_side(a, 'A'))
    elif LAYOUT_STATS == '1':
        winner, score, traj = _play_traj(seed, wa if a_white else wb, wb if a_white else wa)
    else:
        winner, score = _play(seed, wa if a_white else wb, wb if a_white else wa)
    if winner is None:
        m = 0
    else:
        m = score if (winner == 'white') == a_white else -score
    r = {'seed': seed, 'a_white': a_white, 'a_margin': m}
    if traj is not None:
        r['traj'] = traj
    return r


def analyze(path):
    per = {}
    for l in open(path):
        r = json.loads(l)
        per.setdefault(r['seed'], {})[r['a_white']] = r['a_margin']
    units = [0.5 * (v[True] + v[False]) for v in per.values() if len(v) == 2]
    games = [m for v in per.values() for m in v.values()]
    n = len(units)
    if n < 2:
        print(f'{path}: {len(games)} games, too few pairs'); return
    mean = sum(units) / n
    sd = math.sqrt(sum((u - mean) ** 2 for u in units) / (n - 1))
    se = sd / math.sqrt(n)
    wins = sum(m > 0 for m in games)
    print(f'{path}: {len(games)} games, {n} pairs: A wins {wins}/{len(games)} '
          f'({100 * wins / len(games):.1f}%), mean paired margin {mean:+.3f} '
          f'(95% CI {mean - 1.96 * se:+.3f} .. {mean + 1.96 * se:+.3f})')


def main():
    if sys.argv[1] == 'analyze':
        analyze(sys.argv[2]); return
    a, b = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    pairs = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    out = sys.argv[4] if len(sys.argv) > 4 else \
        f'confirm_{os.path.splitext(os.path.basename(a))[0]}.jsonl'
    done = set()
    if os.path.exists(out):
        for l in open(out):
            r = json.loads(l)
            done.add((r['seed'], r['a_white']))
    tasks = [(a, b, SEED_BASE + k, w) for k in range(pairs) for w in ((True,) if SELF_ONLY else (True, False))
             if (SEED_BASE + k, w) not in done]
    print(f'{len(tasks)} games to play ({len(done)} done), A={a} B={b}, '
          f'prefilter {os.environ["GATE_PREFILTER"]}, hand rules on', flush=True)
    from arena import _worker_init
    with mp.Pool(N_WORKERS, initializer=_worker_init) as pool, open(out, 'a') as f:
        for i, r in enumerate(pool.imap_unordered(_game, tasks, chunksize=1), 1):
            f.write(json.dumps(r) + '\n'); f.flush()
            if i % 50 == 0:
                print(f'  {i}/{len(tasks)}', flush=True)
                analyze(out)
    analyze(out)


if __name__ == '__main__':
    main()
