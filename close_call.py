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
    python3 close_call.py label <net.pt> <T> [K=3] [games=300] [out.jsonl.gz]

REPLY CAP (MAX_REPLIES=n, 0 = off): the phone cannot afford 21 x K reply
searches on a move. With a cap the rolls are searched ROLL-MAJOR in ROLL_ORDER
-- every candidate on one roll, then the next roll -- and only COMPLETE rounds
count, so all candidates are compared on the same rolls (a paired comparison);
the value is the expectation over the rolls done, renormalised by their
probability. Fewer than one round -> the plain pick. agent.js's close-call search
does exactly this with a TIME budget; the cap is its device-independent twin.

LABEL (distillation, 2026-10-21): self-play with this agent on BOTH sides; at
every triggered decision each searched candidate's end-of-turn position is
written with its search value as the target (kind 'cand'), and every turn's
chosen afterstate is written as an ordinary TD record (kind 'turn'), so
distill_train.py can mix the two. START_POOL / START_FRAC as in league_run.
"""
import json, math, os, sys, time
import multiprocessing as mp

import numpy as np

os.environ.setdefault('GATE_PREFILTER', '12,40,5')
SEED_BASE = int(os.environ.get('SEED_BASE', 84_000_000))
N_WORKERS = int(os.environ.get('N_WORKERS', '8'))
ROLLS = [(a, b, (1 if a == b else 2) / 36.0) for a in range(1, 7) for b in range(a, 7)]
DRAW_PAIR = ((1, 1, 1), (0, 0, 0))
# Search order under a reply cap: a fixed spread, so a partial set of rolls is
# not all low or all doubles. Every roll appears once.
ROLL_ORDER = [(3, 4), (1, 6), (2, 5), (5, 6), (1, 2), (3, 3), (2, 6), (1, 4), (4, 5),
              (6, 6), (1, 3), (2, 4), (3, 6), (1, 1), (4, 6), (2, 3), (5, 5), (1, 5),
              (3, 5), (2, 2), (4, 4)]
ROLL_P = {(a, b): p for a, b, p in ROLLS}
assert sorted(ROLL_ORDER) == sorted(ROLL_P) and len(ROLL_ORDER) == 21
MAX_REPLIES = int(os.environ.get('MAX_REPLIES', '0'))
# CC_ROLL=random (2026-10-22, owner): under a reply cap, draw the rolls at random
# for EACH decision (by their odds, without replacement; all candidates still
# share them) instead of always starting ROLL_ORDER at 3-4. A fixed first roll
# judges every close call by the opponent's best 3-4, which may favour moves
# that are good against that roll in particular. With one round the drawn roll's
# value is an unbiased sample of the full 21-roll expectation. Own RNG stream,
# reseeded per game, so the game's dice are untouched.
CC_ROLL = os.environ.get('CC_ROLL', 'fixed')
assert CC_ROLL in ('fixed', 'random'), CC_ROLL


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
            self.label_log = None        # list -> record searched candidates (label)
            self.max_replies = MAX_REPLIES
            import random
            self.roll_rng = random.Random(0)    # CC_ROLL=random; reseeded per game

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
            if self.max_replies > 0:
                deep = self._deep_values_capped(board, player, top, self.max_replies)
            else:
                deep = [self._deep_value(board, player, p) for p in top]
            self.stats['deep_s'] += time.time() - t0
            if deep is None:                 # the cap allowed less than one round
                return plain
            pick = top[int(np.argmax(deep))]
            if self.label_log is not None:
                shallow = {p: v for v, p in cands}
                self.label_log.append({
                    'cands': [{'raw_state': self._afterstate_of(board, p),
                               'shallow': shallow[p] / SCORE_SCALE,
                               'label': d / SCORE_SCALE,
                               'pair': [list(m) if isinstance(m, tuple) else m for m in p]}
                              for p, d in zip(top, deep)],
                    'gap': gap / SCORE_SCALE})
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

        @staticmethod
        def _afterstate_of(board, pair):
            """serialize_board of the end of the mover's turn after `pair` -- the
            same position game_worker records for an afterstate net."""
            from game_worker import serialize_board
            root = len(board.moves)
            for m in pair:
                if m != (0, 0, 0):
                    board.apply_move(m, switch_turn=False)
            rs = serialize_board(board)
            while len(board.moves) > root:
                board.undo_last_move()
            return rs

        def _reply_value(self, board, opp, d1, d2, win_value):
            """Mover-frame value of one opponent roll, the opponent's turn
            already entered: minus its best reply score."""
            board.dice[0].number, board.dice[0].used = d1, False
            board.dice[1].number, board.dice[1].used = d2, False
            board._reachable_cache.clear()
            moves = board.get_valid_moves()
            if self.inner is not None:
                r = self.inner.select_move_pair(moves, board, opp, True)
            else:
                r = GNNAgent.select_move_pair(self, moves, board, opp, True)
            if isinstance(r, list) and r:
                best = max(v for v, _ in r)
                return win_value if best == float('inf') else -best
            return 0.0

        def _deep_values_capped(self, board, player, top, max_replies):
            """ROLL-MAJOR deep values under a cap of `max_replies` reply
            searches, complete rounds only (see the module docstring). Returns
            None when not even one round fits."""
            rounds = min(21, max_replies // len(top))
            if rounds < 1:
                return None
            opp = 'white' if player == 'black' else 'black'
            totals, mass = [0.0] * len(top), 0.0
            self._inner = True
            try:
                for d1, d2 in self._roll_sequence(rounds):
                    pr = ROLL_P[(d1, d2)]
                    for i, pair in enumerate(top):
                        root = len(board.moves)
                        for m in pair:
                            if m != (0, 0, 0):
                                board.apply_move(m, switch_turn=False)
                        mine = len(board.white_saved if player == 'white' else board.black_saved)
                        win_value = -(NUM_PIECES - mine) / NUM_PIECES * SCORE_SCALE
                        snap = self._enter_opponent_turn_deterministic(board)
                        try:
                            totals[i] += pr * self._reply_value(board, opp, d1, d2, win_value)
                        finally:
                            self._restore_turn_state(board, snap)
                            while len(board.moves) > root:
                                board.undo_last_move()
                    mass += pr
            finally:
                self._inner = False
            return [t / mass for t in totals]

        def _roll_sequence(self, rounds):
            """The rolls a capped search uses: ROLL_ORDER's first `rounds`, or
            under CC_ROLL=random a fresh draw by the rolls' odds, without
            replacement."""
            if CC_ROLL == 'fixed':
                return ROLL_ORDER[:rounds]
            pool, out = list(ROLL_ORDER), []
            for _ in range(rounds):
                r = self.roll_rng.random() * sum(ROLL_P[x] for x in pool)
                for j, x in enumerate(pool):
                    r -= ROLL_P[x]
                    if r < 0 or j == len(pool) - 1:
                        out.append(pool.pop(j))
                        break
            return out

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
    deep.roll_rng.seed((seed << 1 | deep_white) ^ 0xC0C0)
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
    print(f'{len(tasks)} games to play ({len(done)} done), T={T} K={K} '
          f'max_replies={MAX_REPLIES or "off"} rolls={CC_ROLL}, seeds {SEED_BASE}', flush=True)
    from arena import _worker_init
    with mp.Pool(N_WORKERS, initializer=_worker_init) as pool, open(out, 'a') as f:
        for i, r in enumerate(pool.imap_unordered(_game, tasks, chunksize=1), 1):
            f.write(json.dumps(r) + '\n'); f.flush()
            if i % 40 == 0:
                print(f'  {i}/{len(tasks)}', flush=True)
                analyze(out)
    analyze(out)


_POOL_STATES = None


def _start_state(seed):
    """A START_POOL position for this game, or None (START_FRAC of games, own
    RNG stream, rotated by a random symmetric image as league_run does)."""
    global _POOL_STATES
    pool_path = os.environ.get('START_POOL', '')
    frac = float(os.environ.get('START_FRAC', '0.3'))
    if not pool_path or frac <= 0:
        return None
    if _POOL_STATES is None:
        with open(pool_path) as fh:
            import goal_layout
            _POOL_STATES = [goal_layout.relabel_state(json.loads(l)['state']) for l in fh if l.strip()]
    import random
    rng = random.Random((seed << 8) ^ 0x5157)
    if rng.random() >= frac:
        return None
    rs = rng.choice(_POOL_STATES)
    from symmetry import Symmetry
    return Symmetry().transform(rs, rng.randint(0, 2))


def _label_game(task):
    """One self-play game, close-call agent on both sides. Returns its records:
    'turn' = every turn's chosen afterstate (a TD chain, as game_worker writes
    for an afterstate net), 'cand' = every searched candidate with its search
    value as 'label' (mover frame, raw units), grouped by 'decision'."""
    path, T, K, seed = task
    import random
    from arena import MAX_TURNS, STUCK_LIMIT
    from game import Board
    from game_worker import serialize_board
    deep, _ = _agents(path, T, K)
    assert getattr(deep, 'afterstate', False), 'labelling needs an afterstate net'
    deep.label_log = []
    deep.roll_rng.seed(seed ^ 0xC0C0)
    before = dict(deep.stats)
    t0 = time.time()
    random.seed(seed)
    board = Board()
    start = _start_state(seed)
    if start is not None:
        board.update_state(start)
        board.firstMove = None
        for d in board.dice:
            d.roll()
    gid = f'cc_{seed}' + ('_st' if start is not None else '')
    turns, cands = [], []
    winner, score = None, 0
    last_saved, since = 0, 0
    for t in range(MAX_TURNS):
        w, sc = board.check_game_over()
        if w:
            winner, score = w, sc
            break
        if board.draw_callable:
            break
        cur = len(board.white_saved) + len(board.black_saved)
        if cur > last_saved:
            last_saved, since = cur, 0
        elif last_saved > 0:
            since += 1
        if last_saved > 0 and since >= STUCK_LIMIT:
            break
        player = board.current_player
        n_log = len(deep.label_log)
        chosen = deep.select_move_pair(list(board.get_valid_moves()), board, player)
        if isinstance(chosen, tuple) and len(chosen) == 3:
            chosen = (chosen, (0, 0, 0))
        if len(deep.label_log) > n_log:
            entry = deep.label_log[-1]
            dec = f'{gid}_{t}'
            for c in entry['cands']:
                cands.append({'kind': 'cand', 'game_id': gid, 'decision': dec,
                              'player': player, 'move_index': t, 'afterstate': True,
                              'raw_state': c['raw_state'], 'label': c['label'],
                              'shallow': c['shallow'], 'gap': entry['gap'],
                              'chosen': [list(m) for m in chosen] == c['pair']})
        for m in chosen:
            if m != (0, 0, 0):
                board.apply_move(m, switch_turn=False)
        turns.append({'kind': 'turn', 'game_id': gid, 'player': player, 'move_index': t,
                      'afterstate': True, 'raw_state': serialize_board(board),
                      'terminal': board.check_game_over()[0] is not None})
        board.switch_turn()
    for r in turns + cands:
        r['final_score'] = 0 if winner is None else (score if r['player'] == winner else -score)
    st = {k: deep.stats[k] - before[k] for k in before}
    return {'seed': seed, 'secs': time.time() - t0, 'stats': st, 'records': turns + cands}


def label(path, T, K=3, games=300, out=None):
    """Write labelled self-play games to `out` (gzip JSONL, one record per line,
    plus one {'kind': 'game'} line per game). Resumable by seed."""
    import gzip
    path = os.path.abspath(path)
    out = out or f'cc_labels_T{T:g}_K{K}.jsonl.gz'
    done = set()
    if os.path.exists(out):
        with gzip.open(out, 'rt') as fh:
            for l in fh:
                r = json.loads(l)
                if r.get('kind') == 'game':
                    done.add(r['seed'])
    tasks = [(path, T, K, SEED_BASE + k) for k in range(games) if SEED_BASE + k not in done]
    print(f'{len(tasks)} games to label ({len(done)} done), T={T} K={K} '
          f'max_replies={MAX_REPLIES or "off"} rolls={CC_ROLL}, seeds {SEED_BASE}', flush=True)
    from arena import _worker_init
    n_c = n_t = 0
    t0 = time.time()
    with mp.Pool(N_WORKERS, initializer=_worker_init) as pool:
        for i, g in enumerate(pool.imap_unordered(_label_game, tasks, chunksize=1), 1):
            with gzip.open(out, 'at') as fh:      # one gzip member per game: a kill loses at most one
                for r in g['records']:
                    fh.write(json.dumps(r) + '\n')
                fh.write(json.dumps({'kind': 'game', 'seed': g['seed'], 'secs': g['secs'],
                                     **g['stats']}) + '\n')
            n_c += sum(r['kind'] == 'cand' for r in g['records'])
            n_t += sum(r['kind'] == 'turn' for r in g['records'])
            if i % 10 == 0 or i == len(tasks):
                print(f'  {i}/{len(tasks)} games, {n_t} turn + {n_c} cand records, '
                      f'{(time.time() - t0) / i:.1f} s/game wall', flush=True)


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
    elif cmd == 'label':
        label(sys.argv[2], float(sys.argv[3]),
              int(sys.argv[4]) if len(sys.argv) > 4 else 3,
              int(sys.argv[5]) if len(sys.argv) > 5 else 300,
              sys.argv[6] if len(sys.argv) > 6 else None)
