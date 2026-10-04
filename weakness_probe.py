"""Owner's three suspected weaknesses, measured on his recorded games.

For every replayable turn of every recorded game (both the computer's and
owner's), from the position at the start of the turn with the recorded dice:

  EXPOSURE of the mover's NUMBERED pieces after the move, computed exactly with
  the engine: every legal opponent reply over all 21 rolls, weighted by roll
  probability.
    cap   = sum over the mover's numbered pieces of P(the reply captures it)
    blk   = sum over the mover's numbered pieces of P(the reply lengthens its
            shortest route to its own goal, or cuts it, by building a wall)
  computed for the move actually played and for the net's top-K candidates, so
  "needless" exposure = played minus the least exposed candidate the net itself
  scores within DELTA margin points of its best.

  CAPTURE vs GOAL: positions where some candidate captures an opponent numbered
  piece and another puts one of the mover's numbered pieces on its own goal, and
  none does both. Records which the mover chose.

Usage: python3 weakness_probe.py <log.jsonl>...   (results -> weakness_probe.jsonl)
       python3 weakness_probe.py analyze
"""
import json, os, sys, time
from collections import deque

REPO = os.path.dirname(os.path.abspath(__file__))
NET = os.environ.get('NET', f'{REPO}/symaug_champ_July27_iter6.pt')
RESULTS = os.environ.get('RESULTS', f'{REPO}/weakness_probe.jsonl')
TOP_K = 6
DELTA = 0.25      # margin points: candidates the net scores this close count as alternatives

M = 12 / 1000.0   # select_move_pair scores are raw*1000; raw*12 = margin points

_AG = None
_DEP = None
_MODEL = None


def deployed():
    """The agent as the app configures it (agent.js header): heuristic
    prefilter on, 12 first moves, 40 pairs, min 5."""
    global _DEP
    if _DEP is None:
        from agent_gnn import GNNAgent
        agent()
        _DEP = GNNAgent(model=_MODEL, use_prefilter=True, first_move_prefilter=12,
                        prefilter_top_k=40, prefilter_min_k=5)
    return _DEP


def agent():
    """The pure net over ALL legal pairs (no prefilter)."""
    global _AG, _MODEL
    if _AG is None:
        os.environ['BOARDGAME_DEVICE'] = 'cpu'
        import torch
        torch.set_num_threads(1)
        torch.set_grad_enabled(False)
        import network
        network.DEVICE = torch.device('cpu')
        from network import BoardGNN
        from agent_gnn import GNNAgent
        m = BoardGNN()
        m.load_state_dict(torch.load(NET, map_location='cpu'), strict=False)
        m.eval()
        _MODEL = m
        _AG = GNNAgent(model=m)
    return _AG


def route_len(board, piece):
    """Shortest route from the piece's tile to its own goal for its owner, walls
    (2+ opponent pieces on a field tile) impassable. None = no route."""
    start = piece.tile
    if start is None or start.type != 'field':
        return None
    goal = next((t for t in board.tiles if t.type == 'save' and t.number == piece.number), None)
    seen, q = {start}, deque([(start, 0)])
    while q:
        t, d = q.popleft()
        if t is goal:
            return d
        for n in t.neighbors:
            if n in seen or n.type in ('nogo', 'home'):
                continue
            if n.type == 'field' and len(n.pieces) > 1 and n.pieces[0].player != piece.player:
                continue
            seen.add(n)
            q.append((n, d + 1))
    return None


def threats(board, us):
    """Opponent to move (turn already entered, dice free to overwrite). Returns
    {piece_number: (P_captured, P_blocked)} for `us`'s numbered field pieces."""
    from agent_gnn import GNNAgent
    targets = [p for p in board.pieces if p.player == us and p.number <= 6
               and p.tile is not None and p.tile.type == 'field']
    if not targets:
        return {}
    base = {p.number: route_len(board, p) for p in targets}
    out = {p.number: [0.0, 0.0] for p in targets}
    root = len(board.moves)
    fm0 = board.firstMove
    stages0 = dict(board.game_stages)

    def check(hit):
        for p in targets:
            n = p.number
            if p.tile is None:            # block-saved into OUR rack: not a threat
                continue
            if p.tile is board.home_tile:
                hit[n][0] = True
            elif not hit[n][1]:
                b, a = base[n], route_len(board, p)
                if b is not None and (a is None or a > b):
                    hit[n][1] = True

    for d1, d2, w in GNNAgent._DICE_ROLLS_21:
        board.dice[0].number, board.dice[1].number = d1, d2
        board.dice[0].used = board.dice[1].used = False
        board.firstMove = fm0
        hit = {p.number: [False, False] for p in targets}
        first = [m for m in board.get_valid_moves() if m not in ((0, 0, 0), (1, 1, 1))]
        for m1 in first:
            board.apply_move(m1, switch_turn=False)
            check(hit)
            second = [m for m in board.get_valid_moves() if m not in ((0, 0, 0), (1, 1, 1))]
            for m2 in second:
                board.apply_move(m2, switch_turn=False)
                check(hit)
                board.undo_last_move()
            board.undo_last_move()
            board.firstMove = fm0
            board.game_stages.update(stages0)
        assert len(board.moves) == root
        for n, (c, b) in hit.items():
            out[n][0] += w * c
            out[n][1] += w * b
    board.game_stages.update(stages0)
    board.firstMove = fm0
    return {n: (round(c, 4), round(b, 4)) for n, (c, b) in out.items()}


def summed(th):
    return (round(sum(c for c, _ in th.values()), 4), round(sum(b for _, b in th.values()), 4))


def threats_after_pair(board, pair, us):
    ag = agent()
    n0 = len(board.moves)
    for m in pair:
        if m not in ((0, 0, 0), (1, 1, 1)):
            board.apply_move(m, switch_turn=False)
    saved = ag._enter_opponent_turn_deterministic(board)
    th = threats(board, us)
    ag._restore_turn_state(board, saved)
    while len(board.moves) > n0:
        board.undo_last_move()
    return th


def pair_effects(board, pair, us):
    """(captures an opponent numbered piece, puts own numbered piece on its goal)."""
    opp = 'black' if us == 'white' else 'white'
    n0 = len(board.moves)
    home_before = {id(p) for p in board.home_tile.pieces}
    goal = False
    for m in pair:
        if m in ((0, 0, 0), (1, 1, 1)):
            continue
        board.apply_move(m, switch_turn=False)
        pid, dest, _ = m
        if isinstance(dest, tuple) and pid[0] == us and pid[1] <= 6:
            t = board.get_tile(*dest)
            if t.type == 'save' and t.number == pid[1]:
                goal = True
    cap = any(p.player == opp and p.number <= 6 and id(p) not in home_before
              for p in board.home_tile.pieces)
    while len(board.moves) > n0:
        board.undo_last_move()
    return cap, goal


def turn_states(rec):
    """Yield (turn_index, board at turn start with recorded dice, played pair as
    engine tuples). Replays with replay_games' exact-piece resolution."""
    import replay_games as R
    holder = {}

    class B(R.Board):
        def __init__(s, *a, **k):
            super().__init__(*a, **k)
            holder['b'] = s
    orig = R.Board
    R.Board = B
    try:
        for t in range(len(rec['turns'])):
            r1 = dict(rec); r1['turns'] = rec['turns'][:t]
            ok, _ = R.replay(r1)
            if not ok:
                return
            b = holder['b']
            for die, v in zip(b.dice, rec['turns'][t]['d']):
                die.number, die.used = v, False
            # the played pair: replay one more turn on a fresh board, record tuples
            r2 = dict(rec); r2['turns'] = rec['turns'][:t + 1]
            ok, _ = R.replay(r2)
            if not ok:
                return
            b2 = holder['b']
            played = []
            for mv in b2.moves[len(b.moves):]:
                # board.pieces gets REORDERED during play, so match by player and
                # number. A number can only have changed since via the last-piece
                # rule (-> 13), in which case it is that player's one numbered piece.
                q = mv['piece']
                key = (q.player, q.number)
                if key not in b.piece_lookup:
                    key = next((pp.player, pp.number) for pp in b.pieces
                               if pp.player == q.player and pp.number <= 6
                               and pp.rack is not b.get_save_rack(q.player))
                played.append((key, mv['destination'], mv['roll']))
            yield t, b, tuple(played), b2
    finally:
        R.Board = orig


def _piece_locs(board):
    """Position key with BLANKS ANONYMOUS: same-colour blanks are
    interchangeable, and the engine's dedup may name a different blank than the
    one the game moved (agent_gnn._piece_locs names them, which made 103 of 2098
    computer moves look unreproduced)."""
    saved = (board.white_saved, board.black_saved)
    out = []
    for p in board.pieces:
        loc = p.tile.index if p.tile is not None else (
            -2 if (p.rack is saved[0] or p.rack is saved[1]) else -1)
        out.append((p.player, p.number if p.number <= 6 else 99, loc))
    return tuple(sorted(out))


def analyse_game(rec, gaps_only=False):
    ag = agent()
    human = 'white' if not rec['whiteIsAI'] else 'black'
    rows = []
    for t, b, played, b2 in turn_states(rec):
        us = b.current_player
        moves = list(b.get_valid_moves())
        if len(moves) <= 1:
            continue
        stages = dict(b.game_stages)
        scored = ag.select_move_pair(moves, b, us, return_scores=True)
        b.game_stages.update(stages)
        if not isinstance(scored, list) or not scored:
            continue
        best = scored[0][0]
        def key(pair):
            n0 = len(b.moves)
            for m in pair:
                if m not in ((0, 0, 0), (1, 1, 1)):
                    b.apply_move(m, switch_turn=False)
            k = _piece_locs(b)
            while len(b.moves) > n0:
                b.undo_last_move()
            b.game_stages.update(stages)
            return k
        keys = [key(p) for _, p in scored]
        score_of = {}
        for k, (sc, _) in zip(keys, scored):
            score_of.setdefault(k, sc)
        pk = key(played)
        p_score = score_of.get(pk)
        # what the shipped agent would play here (prefilter on, full strength)
        dep = deployed().select_move_pair(list(b.get_valid_moves()), b, us, difficulty=1.0)
        b.game_stages.update(stages)
        if isinstance(dep, tuple) and len(dep) == 3:
            dep = (dep, (0, 0, 0))
        dk = key(dep)
        d_score = score_of.get(dk)
        if gaps_only:
            rows.append({'turn': t, 'who': 'human' if us == human else 'ai',
                         'gap': None if p_score is None else round((best - p_score) * M, 3),
                         'dep_is_played': dk == pk,
                         'dep_gap': None if d_score is None else round((best - d_score) * M, 3)})
            continue
        # exposure of the played move, and of the net's close alternatives
        # (distinct resulting positions only)
        th_played = threats(b2, us)
        alts, seen = [], set()
        for (sc, pair), k in zip(scored, keys):
            if len(alts) >= TOP_K or (best - sc) * M > DELTA:
                break
            if k in seen:
                continue
            seen.add(k)
            th = threats_after_pair(b, pair, us)
            b.game_stages.update(stages)
            alts.append({'gap': round((best - sc) * M, 3), 'exp': summed(th),
                         'is_played': k == pk})
        # capture vs goal
        effects = []
        for _, pair in scored:
            effects.append(pair_effects(b, pair, us))
            b.game_stages.update(stages)
        has_c = any(c and not g for c, g in effects)
        has_g = any(g and not c for c, g in effects)
        both = any(c and g for c, g in effects)
        cvg = None
        if has_c and has_g and not both:
            pc, pg = pair_effects(b, played, us)
            b.game_stages.update(stages)
            cvg = 'capture' if pc else ('goal' if pg else 'neither')
            # the net's own preference between the best of each class
            bc = max(sc for (sc, _), (c, g) in zip(scored, effects) if c and not g)
            bg = max(sc for (sc, _), (c, g) in zip(scored, effects) if g and not c)
            cvg_net = round((bc - bg) * M, 3)
        else:
            cvg_net = None
        opp = 'black' if us == 'white' else 'white'
        rows.append({
            'game': rec['id'][:8], 'turn': t, 'who': 'human' if us == human else 'ai',
            'my_rack': len(b.white_unentered if us == 'white' else b.black_unentered),
            'opp_rack': len(b.white_unentered if opp == 'white' else b.black_unentered),
            'stage': stages.get(us), 'n_cands': len(scored),
            'gap': None if p_score is None else round((best - p_score) * M, 3),
            'dep_is_played': dk == pk,
            'dep_gap': None if d_score is None else round((best - d_score) * M, 3),
            'played': repr(played), 'exp': summed(th_played), 'exp_by_piece': th_played,
            'alts': alts, 'cvg': cvg, 'cvg_net': cvg_net})
    return rows


def main():
    if sys.argv[1:2] == ['analyze']:
        return analyze()
    done = set()
    if os.path.exists(RESULTS):
        done = {json.loads(l)['game'] for l in open(RESULTS)}
    recs = []
    for f in sys.argv[1:]:
        for line in open(f):
            r = json.loads(line)
            if r.get('completed') and r['whiteIsAI'] != r['blackIsAI'] and r['id'][:8] not in done:
                recs.append(r)
    seen = set()
    recs = [r for r in recs if not (r['id'] in seen or seen.add(r['id']))]
    print(f'{len(recs)} games to analyse', flush=True)
    from multiprocessing import Pool
    with Pool(int(os.environ.get('N_WORKERS', '1'))) as pool:
        for gid, rows, secs in pool.imap_unordered(_job, recs):
            with open(RESULTS, 'a') as fh:
                fh.write(json.dumps({'game': gid, 'rows': rows}) + '\n')
            print(f"{gid}: {len(rows)} turns, {secs:.0f}s", flush=True)


def _job(r):
    t0 = time.time()
    return r['id'][:8], analyse_game(r, gaps_only=bool(os.environ.get('GAPS_ONLY'))), time.time() - t0


def analyze():
    rows = []
    for line in open(RESULTS):
        rows.extend(json.loads(line)['rows'])
    games = len({r['game'] for r in rows})
    print(f'{games} games, {len(rows)} turns\n')

    def phase(r):
        if r['my_rack'] > 0:
            return 'opening'
        return 'endgame' if r['stage'] == 'endgame' else 'midgame'

    def mean(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else float('nan')

    print('EXPOSURE OF THE MOVER\'S NUMBERED FIELD PIECES AFTER ITS MOVE')
    print('  per numbered field piece: P(captured next turn), P(route lengthened by a wall)')
    print('  needless = played minus least exposed alternative the net scores within '
          f'{DELTA} pts of its best\n')
    print(f'{"who":>6} {"phase":>8} {"turns":>6} {"pieces":>7} {"P(cap)":>7} {"P(blk)":>7} '
          f'{"needless cap>.25":>17} {"needless blk>.25":>17}')
    for who in ('ai', 'human'):
        for ph in ('opening', 'midgame', 'endgame', 'all'):
            rs = [r for r in rows if r['who'] == who and (ph == 'all' or phase(r) == ph)]
            if not rs:
                continue
            per = [v for r in rs for v in r['exp_by_piece'].values()]
            nc = nb = 0
            for r in rs:
                if r['alts']:
                    nc += r['exp'][0] - min(a['exp'][0] for a in r['alts']) > 0.25
                    nb += r['exp'][1] - min(a['exp'][1] for a in r['alts']) > 0.25
            print(f'{who:>6} {ph:>8} {len(rs):>6} {len(per):>7} {mean(v[0] for v in per):>7.3f} '
                  f'{mean(v[1] for v in per):>7.3f} {nc/len(rs):>16.1%} {nb/len(rs):>16.1%}')
    print()

    print('CAPTURE AN OPPONENT NUMBERED PIECE vs PUT OWN NUMBERED PIECE ON ITS GOAL')
    print('  (positions where both were available and no move did both; net pref = '
          'best capture minus best goal, margin pts)\n')
    for who in ('ai', 'human'):
        rs = [r for r in rows if r['who'] == who and r['cvg']]
        if not rs:
            continue
        c = sum(r['cvg'] == 'capture' for r in rs)
        g = sum(r['cvg'] == 'goal' for r in rs)
        n = sum(r['cvg'] == 'neither' for r in rs)
        print(f'  {who:>6}: n={len(rs):>4}  capture {c:>4}  goal {g:>4}  neither {n:>4}   '
              f'net prefers capture in {sum(r["cvg_net"] > 0 for r in rs)}/{len(rs)}, '
              f'mean pref {mean(r["cvg_net"] for r in rs):+.3f}')
    print()

    ai = [r for r in rows if r['who'] == 'ai']
    print('REPRODUCTION AND PREFILTER COST (computer turns)')
    print(f'  shipped-agent config reproduces the recorded move: '
          f'{sum(r["dep_is_played"] for r in ai)}/{len(ai)}')
    dg = [r['dep_gap'] for r in ai if r['dep_gap'] is not None]
    print(f'  prefilter misses the net\'s best: {sum(x > 0.001 for x in dg)}/{len(dg)} turns, '
          f'mean cost {mean(dg):.3f} pts/turn, total {sum(dg):.1f} pts over {games} games')
    hu = [r['gap'] for r in rows if r['who'] == 'human' and r['gap'] is not None]
    print(f'  owner\'s moves vs the net\'s best (net\'s own view): mean {mean(hu):.3f} pts/turn, '
          f'differ in {sum(x > 0.001 for x in hu)}/{len(hu)}')



if __name__ == '__main__':
    main()
