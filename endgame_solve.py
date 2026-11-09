"""endgame_solve.py -- exact search for small endings (owner, 2026-11-09).

Owner's game f4252fca t50 (sum7): black 3@G3 x@G1 vs white's last piece, a
blank on G4; roll 2-2 -> black stepped the 3 to G4. Staying finished next turn
on 11/36 rolls, the move on 0/36. The endgame lookahead could not see it: it
counts only the OPPONENT's chance to finish, and both moves had the same one.

This solver answers such positions exactly, without the net. From a position
with dice rolled it searches every end-of-turn position the mover can reach,
then every opponent roll and reply, and so on for DEPTH turns, scoring finished
games by their exact margin. Where the game is still running at the horizon it
uses BOUNDS, not an estimate: the final margin from the mover's view lies in
[-(mover's unsaved), +(opponent's unsaved)] because unsaved counts never rise.
Bounds pass exactly through max and expectation, so every value comes back as
an interval [lo, hi] that contains the true game value under best play. Move A
is PROVEN better than B when lo(A) > hi(B); with both sides down to 1-2 pieces
the game is nearly always over within a few turns and the intervals are narrow.

Values are in POINTS (final margin), from the mover's view.

    python3 endgame_solve.py demo            # owner's t50 position
"""
import os
import sys
import time

from game import Board

ROLLS_21 = [(a, b, (1 if a == b else 2) / 36.0) for a in range(1, 7) for b in range(a, 7)]
PASS, DRAW = (0, 0, 0), (1, 1, 1)
NUM_PIECES = 12


def other(side):
    return 'white' if side == 'black' else 'black'


def unsaved(board, side):
    return NUM_PIECES - len(board.white_saved if side == 'white' else board.black_saved)


def state_key(board):
    """Everything that decides the future: where every unsaved piece is (blanks
    anonymous), rack order, the draw counter. Not whose turn -- callers add it."""
    on = []
    for p in board.pieces:
        if p.tile is not None:
            loc = p.tile.index if p.tile.index is not None else (p.tile.ring, p.tile.pos)
            on.append((p.player, p.number if p.number <= 6 else 0, loc))
    on.sort()
    racks = (tuple(p.number for p in board.white_unentered),
             tuple(p.number for p in board.black_unentered))
    return (tuple(on), racks, len(board.white_saved), len(board.black_saved),
            board.no_save_turns, board._half_turns_since_round, board.draw_callable)


# -- turn switching, reversible (mirrors agent_gnn._enter_opponent_turn_deterministic)

def enter_turn(board):
    saved = {
        'player': board.current_player,
        'dice': [(d.number, d.used) for d in board.dice],
        'firstMove': board.firstMove,
        'no_save_turns': board.no_save_turns,
        'half_turns': board._half_turns_since_round,
        'last_total_saved': board._last_total_saved,
        'draw_callable': board.draw_callable,
        'game_stages': dict(board.game_stages),
        'moves_len': len(board.moves),
    }
    _clear(board)
    board.update_no_save_counter()
    board.firstMove = None
    board.current_player = other(board.current_player)
    before = {id(p): p.number for p in board.pieces if p.number <= 6}
    board.apply_last_piece_rule()
    saved['renumbered'] = [(p, before[id(p)]) for p in board.pieces
                           if id(p) in before and p.number != before[id(p)]]
    return saved


def leave_turn(board, saved):
    assert len(board.moves) == saved['moves_len']
    for p, old in saved['renumbered']:
        k = (p.player, p.number)
        if board.piece_lookup.get(k) is p:
            del board.piece_lookup[k]
        p.number = old
        board.piece_lookup[(p.player, old)] = p
    board.current_player = saved['player']
    for d, (n, u) in zip(board.dice, saved['dice']):
        d.number, d.used = n, u
    board.firstMove = saved['firstMove']
    board.no_save_turns = saved['no_save_turns']
    board._half_turns_since_round = saved['half_turns']
    board._last_total_saved = saved['last_total_saved']
    board.draw_callable = saved['draw_callable']
    board.game_stages.update(saved['game_stages'])
    _clear(board)


def _clear(board):
    board._distance_cache.clear()
    board._blot_cache.clear()
    board._reachable_cache.clear()
    board._blocked_key_cache.clear()


def set_dice(board, a, b):
    board.dice[0].number, board.dice[0].used = a, False
    board.dice[1].number, board.dice[1].used = b, False
    board._reachable_cache.clear()


def undo_to(board, n):
    while len(board.moves) > n:
        board.undo_last_move()


def is_move(m):
    return isinstance(m, tuple) and len(m) == 3 and m not in (PASS, DRAW)


class Solver:
    """Interval expectiminimax. `depth` = opponent turns searched after the
    mover's own (depth 1: our move, every opponent roll + reply, then bounds)."""

    def __init__(self, depth=2, max_nodes=None):
        self.depth = depth
        self.memo = {}
        self.nodes = 0
        self.max_nodes = max_nodes

    # every distinct end-of-turn position for the side to move (dice set)
    def end_positions(self, board, cb):
        """Calls cb(pair) with the board AT that end position (turn not
        switched). Returns False if cb asked to stop. A draw call is reported as
        pair (DRAW,) with the board unchanged."""
        side = board.current_player
        root = len(board.moves)
        seen = set()

        def emit(pair):
            k = state_key(board)
            if k in seen:
                return True
            seen.add(k)
            return cb(pair)

        moves1 = board.get_valid_moves()
        if DRAW in moves1:
            cb((DRAW,))
        if not any(is_move(m) for m in moves1):
            return emit(())                      # nothing to do: the turn just ends
        if PASS in moves1:
            emit(())
        for m1 in moves1:
            if not is_move(m1):
                continue
            board.apply_move(m1, switch_turn=False)
            mid = len(board.moves)
            if board.check_game_over()[0] == side:
                emit((m1,))
            else:
                moves2 = board.get_valid_moves()
                real2 = [m for m in moves2 if is_move(m)]
                if not real2 or PASS in moves2:
                    emit((m1,))
                for m2 in real2:
                    board.apply_move(m2, switch_turn=False)
                    emit((m1, m2))
                    undo_to(board, mid)
            undo_to(board, root)
        return True

    def after_value(self, board, mover, depth):
        """Value (lo, hi) for `mover`, whose turn has just ended on this board."""
        w, s = board.check_game_over()
        if w:
            v = 0.0 if w == 'draw' else (float(s) if w == mover else -float(s))
            return (v, v)
        opp = other(mover)
        if depth <= 0:
            return self.leaf(board, mover)
        key = (state_key(board), mover, depth)
        hit = self.memo.get(key)
        if hit is not None:
            return hit
        self.nodes += 1
        lo = hi = 0.0
        saved = enter_turn(board)
        try:
            for a, b, w_ in ROLLS_21:
                set_dice(board, a, b)
                olo, ohi = self.turn_value(board, opp, depth - 1)
                lo += w_ * -ohi
                hi += w_ * -olo
        finally:
            undo_to(board, saved['moves_len'])
            leave_turn(board, saved)
        self.memo[key] = (lo, hi)
        return (lo, hi)

    def leaf(self, board, mover):
        """Bounds at the horizon, `mover`'s turn just ended. The opponent banks
        out next turn with P_o (exact): margin -my. If they miss they still have
        >= 1 piece, and `mover` then wins with probability >= q, the share of
        rolls on which it banks out by saves alone from goals -- opponent moves
        cannot touch pieces on goals or the banking rule (own pieces only), so q
        holds whatever they do. Otherwise the margin lies in [-my, +op]."""
        opp = other(mover)
        my, op = float(unsaved(board, mover)), float(unsaved(board, opp))
        key = ('leaf', state_key(board), mover)
        hit = self.memo.get(key)
        if hit is not None:
            return hit
        p_o = q = 0.0
        s1 = enter_turn(board)
        try:
            if op <= 2:
                p_o = finish_prob(board, opp)
            if my <= 2 and all_on_goals(board, mover):
                s2 = enter_turn(board)
                try:
                    q = finish_prob(board, mover, saves_only=True)
                finally:
                    undo_to(board, s2['moves_len'])
                    leave_turn(board, s2)
        finally:
            undo_to(board, s1['moves_len'])
            leave_turn(board, s1)
        lo = p_o * -my + (1 - p_o) * (q * 1.0 + (1 - q) * -my)
        hi = p_o * -my + (1 - p_o) * op
        self.memo[key] = (lo, hi)
        return (lo, hi)

    def turn_value(self, board, side, depth):
        """Best (lo, hi) for `side`, to move with the dice as set."""
        best = [float('-inf'), float('-inf')]

        def cb(pair):
            if pair == (DRAW,):
                v = (0.0, 0.0)
            else:
                v = self.after_value(board, side, depth)
            best[0] = max(best[0], v[0])
            best[1] = max(best[1], v[1])
            return True
        self.end_positions(board, cb)
        return (best[0], best[1])

    def rank(self, board):
        """Every distinct end position for the side to move: list of
        (lo, hi, pair, key), best lower bound first."""
        side = board.current_player
        out = []

        def cb(pair):
            if pair == (DRAW,):
                out.append((0.0, 0.0, pair, 'draw'))
            else:
                lo, hi = self.after_value(board, side, self.depth)
                out.append((lo, hi, pair, state_key(board)))
            return True
        self.end_positions(board, cb)
        out.sort(key=lambda r: (-r[0], -r[1]))
        return out


def can_finish(board, side, saves_only=False):
    """Can `side`, to move with the dice as set, bank everything this turn?"""
    root = len(board.moves)
    for m1 in board.get_valid_moves():
        if not is_move(m1) or (saves_only and m1[1] != 'save'):
            continue
        board.apply_move(m1, switch_turn=False)
        won = board.check_game_over()[0] == side
        if not won:
            mid = len(board.moves)
            for m2 in board.get_valid_moves():
                if not is_move(m2) or (saves_only and m2[1] != 'save'):
                    continue
                board.apply_move(m2, switch_turn=False)
                won = board.check_game_over()[0] == side
                undo_to(board, mid)
                if won:
                    break
        undo_to(board, root)
        if won:
            return True
    return False


def finish_prob(board, side, saves_only=False):
    """P(side, whose turn has been entered, banks out on the roll it gets).
    Dice are left set to the last roll; callers restore them."""
    p = 0.0
    for a, b, w_ in ROLLS_21:
        set_dice(board, a, b)
        if can_finish(board, side, saves_only):
            p += w_
    return p


def all_on_goals(board, side):
    rack = board.white_unentered if side == 'white' else board.black_unentered
    if rack:
        return False
    return all(p.tile is not None and p.tile.type == 'save'
               for p in board.pieces if p.player == side and p.tile is not None) and \
        not any(p.player == side and p.tile is not None and p.tile.type == 'home' for p in board.pieces)


def small_endgame(board, limit=2):
    return unsaved(board, 'white') <= limit and unsaved(board, 'black') <= limit


def _demo():
    import json
    import review_games as RG
    path = sys.argv[2] if len(sys.argv) > 2 else os.environ.get('GAMES')
    games = [json.loads(l) for l in open(path)]
    g = next(x for x in games if x['id'].startswith('f4252fca'))
    b = RG._state_at(g, 50)
    print(RG.notation(b))
    for d in (1, 2, 3):
        s = Solver(depth=d)
        t = time.time()
        r = s.rank(b)
        print('depth', d, 'nodes', s.nodes, '%.1fs' % (time.time() - t))
        for lo, hi, pair, k in r[:6]:
            print('   [%+.3f, %+.3f]' % (lo, hi), pair)
        played = [x for x in r if any(m[1] == (7, 12) for m in x[2] if is_move(m))]
        for lo, hi, pair, k in played[:1]:
            print('   played-like [%+.3f, %+.3f]' % (lo, hi), pair)


if __name__ == '__main__':
    if sys.argv[1:2] == ['demo']:
        _demo()
