"""layout_play.py -- instrumented games for the goal-layout comparison, pass 2
(owner, 2026-11-06). Called from confirm_match._game when LAYOUT_STATS=2.

Each turn, BEFORE the dice are used, the mover's best value is computed for all
21 rolls with the FULL-strength agent (side A's agent, hand rules on, app
prefilter) -- in points, mover's frame:
    e    = E over rolls of the best value          (the position's value)
    vr   = best value on the roll thrown            (luck_t = vr - e, mean zero)
then the side to move plays its own choice (B may be the WEAK_B handicapped
agent). Recorded per turn end:
    [mover 0/1 (white/black), e, vr, gap1, gap2, n_close, caps,
     white unsaved, black unsaved, cut mask by white, cut mask by black,
     walls white, walls black]
gap1/gap2 = points between the best distinct score and the 2nd / 3rd distinct
score on the thrown roll (distinct = differs by > 1e-6 raw, so transpositions
collapse); n_close = distinct scores within 0.1 pts of the best (incl. best);
caps = opponent pieces sent home this turn.
Luck-adjusted result (white's frame): margin - sum over turns of luck in white's
frame. Unbiased, since each luck term has conditional mean zero.
"""
import random
from collections import deque

PTS = 12.0
ROLLS = [(a, b, (1 if a == b else 2) / 36.0) for a in range(1, 7) for b in range(a, 7)]


def _ranked(ag, b, player, d1, d2):
    st = dict(b.game_stages)
    b.dice[0].number, b.dice[0].used = d1, False
    b.dice[1].number, b.dice[1].used = d2, False
    b._reachable_cache.clear()
    r = ag.select_move_pair(b.get_valid_moves(), b, player, return_scores=True)
    b.game_stages.update(st)
    return r if isinstance(r, list) else []


def _value(ag, scale, b, player, ranked):
    """Best value in points (mover's frame) from a ranked list."""
    if not ranked:
        return 0.0
    v, pair = ranked[0]
    if v == float('inf'):
        st = dict(b.game_stages); n0 = len(b.moves)
        for m in pair:
            if m not in ((0, 0, 0), (1, 1, 1)):
                b.apply_move(m, switch_turn=False)
        w, s = b.check_game_over()
        while len(b.moves) > n0:
            b.undo_last_move()
        b.game_stages.update(st)
        return float(s if w == player else -s)
    if pair == ((1, 1, 1), (0, 0, 0)):
        return 0.0
    return v / scale * PTS


def _gaps(ranked, scale):
    vals = [v for v, _ in ranked if v != float('inf')]
    if not vals:
        return 0.0, 0.0, 1
    d = [vals[0]]
    for v in vals[1:]:
        if abs(v - d[-1]) > 1e-6:
            d.append(v)
    to = lambda x: round((d[0] - x) / scale * PTS, 4)
    g1 = to(d[1]) if len(d) > 1 else 99.0
    g2 = to(d[2]) if len(d) > 2 else 99.0
    close = sum(1 for x in d if (d[0] - x) / scale * PTS <= 0.1)
    return g1, g2, close


def _cut_mask(board, sealer):
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
    return m, len(walls)


def _home_count(b, player):
    return sum(1 for p in b.pieces if p.player == player and p.tile is b.home_tile) + \
        len(b.white_unentered if player == 'white' else b.black_unentered)


def play(seed, wag, bag, judge):
    """Like arena._play; `judge` is the full-strength agent that values the rolls.
    Returns (winner, score, traj)."""
    from game import Board
    from arena import MAX_TURNS, STUCK_LIMIT
    from agent_gnn import SCORE_SCALE as scale
    random.seed(seed)
    b = Board(); agents = {'white': wag, 'black': bag}
    last_saved, since, traj = 0, 0, []
    for t in range(MAX_TURNS):
        winner, score = b.check_game_over()
        if winner:
            return winner, score, traj
        if b.draw_callable:
            return None, 0, traj
        cur = len(b.white_saved) + len(b.black_saved)
        if cur > last_saved:
            last_saved, since = cur, 0
        elif last_saved > 0:
            since += 1
        if last_saved > 0 and since >= STUCK_LIMIT:
            return None, 0, traj
        p = b.current_player; opp = 'black' if p == 'white' else 'white'
        thrown = (b.dice[0].number, b.dice[1].number)
        key = tuple(sorted(thrown))
        e = vr = 0.0; g = (0.0, 0.0, 1)
        for d1, d2, pr in ROLLS:
            rk = _ranked(judge, b, p, d1, d2)
            v = _value(judge, scale, b, p, rk)
            e += pr * v
            if (d1, d2) == key:
                vr, g = v, _gaps(rk, scale)
        # restore the thrown dice exactly as the game rolled them
        b.dice[0].number, b.dice[0].used = thrown[0], False
        b.dice[1].number, b.dice[1].used = thrown[1], False
        b._reachable_cache.clear()
        home0 = _home_count(b, opp)
        chosen = agents[p].select_move_pair(list(b.get_valid_moves()), b, p)
        if isinstance(chosen, tuple) and len(chosen) == 3:
            chosen = (chosen, (0, 0, 0))
        for m in chosen:
            if m != (0, 0, 0):
                b.apply_move(m, switch_turn=False)
        caps = max(0, _home_count(b, opp) - home0)
        cw, ww = _cut_mask(b, 'white'); cb, wb = _cut_mask(b, 'black')
        traj.append([0 if p == 'white' else 1, round(e, 4), round(vr, 4), g[0], g[1], g[2], caps,
                     12 - len(b.white_saved), 12 - len(b.black_saved), cw, cb, ww, wb])
        b.switch_turn()
    return None, 0, traj
