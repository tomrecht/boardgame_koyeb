"""features_v2.py -- the extra encoder inputs for the next training run.

Feature sets (CLAUDE.md, "Next training run"):
  v1  the deployed champion's encoding, unchanged.
  A   v1 + exact bookkeeping:
        tile   +1  opponent permanent wall (2+ opponent BLANKS on a field tile;
                   the mirror of v1's flag, which marks the mover's walls only)
        piece  +1  raw distance to own goal, NUMBERED pieces only (0 for blanks,
                   rack and saved pieces; 1.0 = walled off). Near-redundant: for
                   a numbered piece game.py marks the five other goals
                   unreachable, so the min over v1's six slots already is this.
        global +12 no-save counter, opponent saveable count, per side
                   unentered / on field / saved, per side race count and
                   estimated turns to finish
  AB  A + per-piece THREATS, distance-based approximations of what
      weakness_probe.threats() enumerates exactly:
        piece  +2  P(captured if its enemy rolled now), P(its route to its own
                   goal is lengthened by a wall the enemy can build with that
                   roll). Both side-agnostic: "if the other side rolled next".

Everything here is pure Python over game.py's Board, so encoder.js needs a twin
before any AB net can ship.
"""
from collections import deque

FEATURE_SETS = ('v1', 'A', 'AB')
EXTRA_TILE = {'v1': 0, 'A': 1, 'AB': 1}
EXTRA_PIECE = {'v1': 0, 'A': 1, 'AB': 3}
EXTRA_GLOBAL = {'v1': 0, 'A': 12, 'AB': 12}

MAX_DIST = 14.0
NO_SAVE_TURNS_FOR_DRAW = 10
# Expected turns to bank a LONE piece from a goal (CLAUDE.md, domain facts):
# a blank by goal 1-6, a numbered piece on its own goal (exact match needed).
BLANK_BANK = {1: 1.000, 2: 1.029, 3: 1.125, 4: 1.303, 5: 1.462, 6: 1.644}
NUMBERED_BANK = 3.273
BLANK_BANK_MEAN = sum(BLANK_BANK.values()) / 6
PIPS_PER_TURN = 7.0          # mean of two dice
HOME_TO_GOAL = 7             # every goal is exactly 7 from home

# The 36 ordered rolls as 21 distinct ones with weights.
ROLLS = [(a, b, (1 if a == b else 2) / 36.0) for a in range(1, 7) for b in range(a, 7)]


def _other(p):
    return 'black' if p == 'white' else 'white'


# --------------------------------------------------------------------------
# A: bookkeeping
# --------------------------------------------------------------------------

def opp_wall_flags(board, current_player, tile_index, n_tiles):
    """[n_tiles] 1.0 where the OPPONENT has 2+ blanks on a field tile."""
    out = [0.0] * n_tiles
    opp = _other(current_player)
    for t in board.tiles:
        if t.type != 'field' or len(t.pieces) < 2:
            continue
        if sum(1 for p in t.pieces if p.player == opp and p.number > 6) >= 2:
            idx = tile_index.get((t.ring, t.pos))
            if idx is not None:
                out[idx] = 1.0
    return out


def own_goal_dist(board, piece):
    """Raw own-goal distance for a numbered piece on the board, normalised."""
    if piece.number > 6 or piece.tile is None or piece.tile.type == 'home':
        return 0.0
    d = board.shortest_route_to_goal(piece)
    if d == float('inf'):
        return 1.0
    return min(d, MAX_DIST) / MAX_DIST


def _piece_progress(board, p):
    """(remaining pips, expected bank turns) for one unsaved piece."""
    saved = (board.white_saved, board.black_saved)
    if p.rack is saved[0] or p.rack is saved[1]:
        return 0.0, 0.0
    bank = NUMBERED_BANK if p.number <= 6 else BLANK_BANK_MEAN
    if p.tile is None or p.tile.type == 'home':        # rack or captured
        return float(HOME_TO_GOAL), bank
    if p.can_be_saved():                               # standing on a goal it can bank from
        return 0.0, (NUMBERED_BANK if p.number <= 6 else BLANK_BANK.get(p.tile.number, BLANK_BANK_MEAN))
    if p.number <= 6:
        d = board.shortest_route_to_goal(p)
    else:
        ds = board.all_goal_distances(p)
        d = min(ds.values()) if ds else float('inf')
    if d == float('inf'):
        d = MAX_DIST
    return float(min(d, MAX_DIST)), bank


def side_summary(board, player):
    """(unentered, on field, saved, race pips, est. turns to finish)."""
    unentered = len(board.white_unentered if player == 'white' else board.black_unentered)
    saved_rack = board.white_saved if player == 'white' else board.black_saved
    saved = len(saved_rack)
    field = sum(1 for p in board.pieces
                if p.player == player and p.tile is not None and p.tile.type == 'field')
    pips, per_piece = 0.0, []
    for p in board.pieces:
        if p.player != player or p.rack is saved_rack:
            continue
        d, bank = _piece_progress(board, p)
        pips += d
        per_piece.append(d / PIPS_PER_TURN + bank)
    # CRUDE: pieces share two dice a turn, so the side is limited both by its
    # slowest piece and by the total work spread over two dice. A hint for the
    # net, not a rule; validate_features_v2 checks it against real turns left.
    turns = max(max(per_piece, default=0.0), sum(per_piece) / 2.0)
    return unentered, field, saved, pips, turns


def global_extras(board, current_player):
    opp = _other(current_player)
    me = side_summary(board, current_player)
    them = side_summary(board, opp)
    opp_saveable = sum(1 for p in board.pieces if p.player == opp and p.can_be_saved())
    return [
        min(board.no_save_turns, NO_SAVE_TURNS_FOR_DRAW) / NO_SAVE_TURNS_FOR_DRAW,
        opp_saveable / 12.0,
        me[0] / 12.0, me[1] / 12.0, me[2] / 12.0,
        them[0] / 12.0, them[1] / 12.0, them[2] / 12.0,
        me[3] / 84.0, them[3] / 84.0,
        min(me[4], 30.0) / 30.0, min(them[4], 30.0) / 30.0,
    ]


# --------------------------------------------------------------------------
# AB: threat approximations
# --------------------------------------------------------------------------

def _walled_for(board, mover):
    """Tiles `mover` cannot enter or pass: field tiles with 2+ enemy pieces."""
    return {t for t in board.tiles if t.type == 'field' and len(t.pieces) > 1
            and t.pieces[0].player != mover}


def _dist_from(board, target, blocked):
    """BFS distances from `target` to every tile, through tiles a mover may
    pass (not nogo, not home, not in `blocked`). Movement is along an
    undirected graph with node blocking, so this equals each tile's distance
    TO the target. Home is never entered (see _enemy_movers for its distance)."""
    dist = {target: 0}
    q = deque([target])
    while q:
        t = q.popleft()
        for n in t.neighbors:
            if n in dist or n.type == 'nogo':
                continue
            if n.type == 'home':
                continue
            if n in blocked:
                continue
            dist[n] = dist[t] + 1
            q.append(n)
    return dist


def _enemy_movers(board, enemy, target, blocked):
    """Distances to `target` for the enemy's next-turn movers, split by origin,
    plus its turn obligation:
      others : [dist] for each enemy piece on the board (not on `target`)
      dh     : home -> target distance (None if unreachable)
      n_home : pieces that can start from home (captured ones, else up to two
               rack entries)
      oblig  : 'two' if 2+ captured pieces must re-enter (both dice), 'one' if
               one captured piece or a rack entry is owed (one die), else 'none'
    """
    dist = _dist_from(board, target, blocked)
    others = [dist[p.tile] for p in board.pieces
              if p.player == enemy and p.tile is not None and p.tile.type != 'home'
              and p.tile is not target and p.tile in dist]
    captured = sum(1 for p in board.home_tile.pieces if p.player == enemy)
    rack = len(board.white_unentered if enemy == 'white' else board.black_unentered)
    # Adjacency runs home -> ring 1 but not back, so the BFS from the target
    # never reaches home: derive it from home's own neighbours.
    hn = [dist[n] for n in board.home_tile.neighbors if n in dist]
    dh = 1 + min(hn) if hn else None
    if captured >= 2:
        oblig, n_home = 'two', captured
    elif captured == 1:
        oblig, n_home = 'one', 1
    elif rack:
        oblig, n_home = 'one', min(rack, 2)
    else:
        oblig, n_home = 'none', 0
    return others, dh, n_home, oblig


def _lands(movers, a, b, need):
    """Can the roll (a, b) put `need` (1 or 2) enemy pieces on the target?
    Honours the obligation: owed home pieces must use their dice first."""
    others, dh, n_home, oblig = movers
    if need == 1:
        if oblig == 'none':
            return any(d in (a, b, a + b) for d in others)
        if oblig == 'one':
            return (dh is not None and dh in (a, b, a + b)) or any(d in (a, b) for d in others)
        return dh is not None and dh in (a, b)
    # need == 2: two different pieces, one per die
    if oblig == 'none':
        ia = [i for i, d in enumerate(others) if d == a]
        ib = [i for i, d in enumerate(others) if d == b]
        return any(i != j for i in ia for j in ib)
    home_both = dh is not None and n_home >= 2 and dh == a == b
    if oblig == 'one':
        if home_both:
            return True
        return dh is not None and ((dh == a and b in others) or (dh == b and a in others))
    return home_both


def capture_prob(board, piece):
    """Approx P(the enemy's next roll can capture `piece`). Only a lone piece
    on a field tile can be captured. A die of d, or two dice summing to d,
    reach a tile at route distance d (moves follow shortest routes, and a
    shortest route of length d1+d2 always passes a tile at distance d1), subject
    to the enemy's turn obligation (_enemy_movers)."""
    t = piece.tile
    if t is None or t.type != 'field' or len(t.pieces) != 1:
        return 0.0
    enemy = _other(piece.player)
    movers = _enemy_movers(board, enemy, t, _walled_for(board, enemy))
    return sum(w for a, b, w in ROLLS if _lands(movers, a, b, 1))


def _route_len(board, piece, extra_block=None):
    """Own-goal route length for a numbered piece under its own blocking
    (enemy walls), optionally with one more tile walled. None = no route."""
    goal = next((g for g in board.tiles if g.type == 'save' and g.number == piece.number), None)
    blocked = _walled_for(board, piece.player)
    if extra_block is not None:
        blocked = blocked | {extra_block}
    seen, q = {piece.tile: 0}, deque([piece.tile])
    while q:
        t = q.popleft()
        if t is goal:
            return seen[t]
        for n in t.neighbors:
            if n in seen or n.type in ('nogo', 'home') or n in blocked:
                continue
            seen[n] = seen[t] + 1
            q.append(n)
    return None


def wall_prob(board, piece):
    """Approx P(the enemy's next roll builds a wall that lengthens (or cuts)
    the route of numbered `piece` to its own goal). A wall on tile W matters
    iff W lies on EVERY shortest route (blocking it raises the length). The
    enemy walls W with one more piece if one already stands there, else with
    two different pieces, one per die. Tiles holding the piece's own side are
    skipped (landing there captures rather than walls)."""
    if piece.number > 6 or piece.tile is None or piece.tile.type != 'field':
        return 0.0
    base = _route_len(board, piece)
    if base is None:
        return 0.0
    enemy = _other(piece.player)
    # tiles on some shortest route: dist_from_piece + dist_to_goal == base
    goal = next(g for g in board.tiles if g.type == 'save' and g.number == piece.number)
    own_block = _walled_for(board, piece.player)
    d_from = _dist_from(board, piece.tile, own_block)
    d_to = _dist_from(board, goal, own_block)
    chokes = []
    for t in board.tiles:
        if t.type != 'field' or t is piece.tile or t not in d_from or t not in d_to:
            continue
        if d_from[t] + d_to[t] != base:
            continue
        if any(p.player == piece.player for p in t.pieces):
            continue
        after = _route_len(board, piece, extra_block=t)
        if after is None or after > base:
            chokes.append(t)
    if not chokes:
        return 0.0
    enemy_block = _walled_for(board, enemy)
    per_choke = []
    for w in chokes:
        already = sum(1 for p in w.pieces if p.player == enemy)
        per_choke.append((1 if already >= 1 else 2, _enemy_movers(board, enemy, w, enemy_block)))

    def walls(a, b):
        return any(_lands(movers, a, b, need) for need, movers in per_choke)
    return sum(w for a, b, w in ROLLS if walls(a, b))
