"""close_call.py -- deeper search on CLOSE decisions only, for an afterstate net.

The net's weakness is value noise between near-equal moves (calibration corr
~0.2 vs playouts; CLAUDE.md). Searching one opponent turn deeper is exact -- real
rolls, real replies -- and only matters where the shallow search is unsure. So:

  shallow   the app's search (prefilter, hand rules) ranks the candidates
  trigger   best - second best < T  (score units: raw x SCORE_SCALE)
  rescore   the top K candidates within T of the best: apply it, enter the
            opponent's turn (agent_gnn's reversible turn switch), and for each
            of the 21 rolls (doubles 1/36, others 2/36) let the opponent pick
            its reply with the SAME shallow search; the candidate's value is
            minus the opponent's expected best score. An afterstate net scores
            exactly the opponent's end-of-turn position from the opponent's
            frame, the frame it was trained in. A winning reply is exact.
  choose    the best rescored candidate

The existing select_move_pair_deep refuses afterstate nets (it scores turn
starts); this is the afterstate version, restricted to close calls.

    python3 close_call.py gaps  <net.pt> [games=6]              gap distribution -> pick T
    python3 close_call.py match <net.pt> <T> [K=3] [pairs=300] [out.jsonl]
    python3 close_call.py analyze <out.jsonl>
"""
import json, math, os, sys, time
import multiprocessing as mp

import numpy as np

os.environ.setdefault('GATE_PREFILTER', '12,40,5')
SEED_BASE = int(os.environ.get('SEED_BASE', 84_000_000))
N_WORKERS = int(os.environ.get('N_WORKERS', '8'))
ROLLS = [(a, b, (1 if a == b else 2) / 36.0) for a in range(1, 7) for b in range(a, 7)]
DRAW_PAIR = ((1, 1, 1), (0, 0, 0))


def make_agent_class():
    from agent_gnn import GNNAgent, SCORE_SCALE, NUM_PIECES

    class CloseCallAgent(GNNAgent):
        def __init__(self, *a, T=50.0, K=3, inner=None, **kw):
            super().__init__(*a, **kw)
            self.T, self.K = float(T), int(K)
            self._inner = False
            # the opponent-reply search: this agent's own shallow search, or a
            # cheaper agent (e.g. a narrower prefilter) on the same net
            self.inner = inner
            self.stats = {'moves': 0, 'triggered': 0, 'changed': 0, 'deep_s': 0.0}
            self.gap_log = None          # list -> record every top-2 gap

        def select_move_pair(self, moves, board, player, return_scores=False,
                             difficulty=None, deadline=None):
            if self._inner or return_scores:
                return super().select_move_pair(moves, board, player, return_scores,
                                                difficulty, deadline)
            self.stats['moves'] += 1
            ranked = super().select_move_pair(moves, board, player, True, difficulty, deadline)
            plain = super().select_move_pair(moves, board, player, False, difficulty, deadline)
            if not isinstance(ranked, list):
                return plain
            if any(v == float('inf') for v, _ in ranked):
                return plain
            # one entry per distinct resulting POSITION (move orders transpose,
            # and same-colour blanks are interchangeable): a zero gap between
            # two orders of the same moves is not a close call
            cands, seen = [], set()
            for v, p in ranked:
                if p == DRAW_PAIR or not np.isfinite(v):
                    continue
                k = self._outcome(board, p)
                if k not in seen:
                    seen.add(k)
                    cands.append((v, p))
            if len(cands) < 2:
                return plain
            gap = cands[0][0] - cands[1][0]
            if self.gap_log is not None:
                self.gap_log.append(gap)
            if gap >= self.T:
                return plain
            top = [p for v, p in cands[:self.K] if cands[0][0] - v < self.T]
            self.stats['triggered'] += 1
            t0 = time.time()
            deep = [self._deep_value(board, player, p) for p in top]
            self.stats['deep_s'] += time.time() - t0
            pick = top[int(np.argmax(deep))]
            pick = self._dedupe_save_pair(pick)
            if pick != plain:
                self.stats['changed'] += 1
            return pick

        @staticmethod
        def _outcome(board, pair):
            root = len(board.moves)
            for m in pair:
                if m != (0, 0, 0):
                    board.apply_move(m, switch_turn=False)
            saved = (board.white_saved, board.black_saved)
            key = tuple(sorted(
                (p.player, p.number if p.number <= 6 else 0,
                 p.tile.index if p.tile is not None else (-2 if p.rack in saved else -1))
                for p in board.pieces))
            while len(board.moves) > root:
                board.undo_last_move()
            return key

        def _deep_value(self, board, player, pair):
            """Minus the opponent's expected best reply score after `pair`."""
            opp = 'white' if player == 'black' else 'black'
            root = len(board.moves)
            for m in pair:
                if m != (0, 0, 0):
                    board.apply_move(m, switch_turn=False)
            mine_saved = len(board.white_saved if player == 'white' else board.black_saved)
            win_value = -(NUM_PIECES - mine_saved) / NUM_PIECES * SCORE_SCALE
            snap = self._enter_opponent_turn_deterministic(board)
            total = 0.0
            self._inner = True
            try:
                for d1, d2, pr in ROLLS:
                    board.dice[0].number, board.dice[0].used = d1, False
                    board.dice[1].number, board.dice[1].used = d2, False
                    board._reachable_cache.clear()
                    moves = board.get_valid_moves()
                    if self.inner is not None:
                        r = self.inner.select_move_pair(moves, board, opp, True)
                    else:
                        r = super().select_move_pair(moves, board, opp, True)
                    if isinstance(r, list) and r:
                        best = max(v for v, _ in r)
                        val = win_value if best == float('inf') else -best
                    else:
                        val = 0.0
                    total += pr * val
            finally:
                self._inner = False
                self._restore_turn_state(board, snap)
                while len(board.moves) > root:
                    board.undo_last_move()
            return total

    return CloseCallAgent


_CACHE = {}


def _agents(path, T, K):
    key = (path, T, K)
    if key not in _CACHE:
        import torch
        from network import model_from_state
        from agent_gnn import GNNAgent, prefilter_kwargs
        m = model_from_state(torch.load(path, map_location='cpu'), strict=False)
        pf = prefilter_kwargs(os.environ['GATE_PREFILTER'])
        C = make_agent_class()
        ipf = os.environ.get('INNER_PREFILTER', '')      # e.g. '6,10,3'; '' = app's
        inner = GNNAgent(model=m, hand_rules=True, **prefilter_kwargs(ipf)) if ipf else None
        _CACHE[key] = (C(model=m, hand_rules=True, T=T, K=K, inner=inner, **pf),
                       GNNAgent(model=m, hand_rules=True, **pf))
    return _CACHE[key]


def _game(task):
    path, T, K, seed, deep_white = task
    from arena import _play
    deep, plain = _agents(path, T, K)
    before = dict(deep.stats)
    winner, score = _play(seed, deep if deep_white else plain, plain if deep_white else deep)
    m = 0 if winner is None else (score if (winner == 'white') == deep_white else -score)
    st = {k: deep.stats[k] - before[k] for k in before}
    return {'seed': seed, 'a_white': deep_white, 'a_margin': m, **st}


def _gap_game(task):
    path, seed = task
    from arena import _play
    deep, _ = _agents(path, 1e18, 1)      # never triggers; just logs gaps
    deep.gap_log = []
    _play(seed, deep, deep)
    return deep.gap_log


def gaps(path, n_games=6):
    from arena import _worker_init
    with mp.Pool(min(N_WORKERS, n_games), initializer=_worker_init) as pool:
        logs = pool.map(_gap_game, [(path, 85_000_000 + i) for i in range(n_games)])
    g = np.array([x for l in logs for x in l])
    print(f'{len(g)} decisions with 2+ candidates from {n_games} games (score units)')
    for q in (5, 10, 20, 30, 40, 50, 75, 90):
        print(f'  {q:3d}th percentile gap {np.percentile(g, q):9.2f}')


def analyze(path):
    rows = [json.loads(l) for l in open(path)]
    per = {}
    for r in rows:
        per.setdefault(r['seed'], {})[r['a_white']] = r['a_margin']
    units = [0.5 * (v[True] + v[False]) for v in per.values() if len(v) == 2]
    n = len(units)
    if n < 2:
        print(f'{path}: {len(rows)} games, too few pairs'); return
    mean = sum(units) / n
    se = math.sqrt(sum((u - mean) ** 2 for u in units) / (n - 1)) / math.sqrt(n)
    mv = sum(r['moves'] for r in rows)
    tr = sum(r['triggered'] for r in rows)
    ch = sum(r['changed'] for r in rows)
    wins = sum(r['a_margin'] > 0 for r in rows)
    print(f'{path}: {len(rows)} games, {n} pairs: close-call agent wins {wins}/{len(rows)} '
          f'({100 * wins / len(rows):.1f}%), mean paired margin {mean:+.3f} '
          f'(95% CI {mean - 1.96 * se:+.3f} .. {mean + 1.96 * se:+.3f}); '
          f'triggered {tr}/{mv} moves ({100 * tr / max(mv, 1):.1f}%), changed {ch} '
          f'({100 * ch / max(tr, 1):.1f}% of triggers, {ch / max(len(rows), 1):.2f}/game); '
          f'deep search {sum(r.get("deep_s", 0) for r in rows) / max(tr, 1):.2f} s per trigger')


def match(path, T, K=3, pairs=300, out=None):
    path = os.path.abspath(path)
    out = out or f'close_T{T:g}_K{K}.jsonl'
    done = set()
    if os.path.exists(out):
        done = {(r['seed'], r['a_white']) for r in map(json.loads, open(out))}
    tasks = [(path, T, K, SEED_BASE + k, w) for k in range(pairs) for w in (True, False)
             if (SEED_BASE + k, w) not in done]
    print(f'{len(tasks)} games to play ({len(done)} done), T={T} K={K}, seeds {SEED_BASE}', flush=True)
    from arena import _worker_init
    with mp.Pool(N_WORKERS, initializer=_worker_init) as pool, open(out, 'a') as f:
        for i, r in enumerate(pool.imap_unordered(_game, tasks, chunksize=1), 1):
            f.write(json.dumps(r) + '\n'); f.flush()
            if i % 40 == 0:
                print(f'  {i}/{len(tasks)}', flush=True)
                analyze(out)
    analyze(out)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'gaps':
        gaps(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 6)
    elif cmd == 'match':
        match(sys.argv[2], float(sys.argv[3]),
              int(sys.argv[4]) if len(sys.argv) > 4 else 3,
              int(sys.argv[5]) if len(sys.argv) > 5 else 300,
              sys.argv[6] if len(sys.argv) > 6 else None)
    elif cmd == 'analyze':
        analyze(sys.argv[2])
