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
      and ROLL OPPORTUNITIES (owner, 2026-10-04): for each side and each die
      value 1-6, how many pieces that single die would save, bring onto a goal
      they can bank from, or use to capture -- numbered and blank separately:
        global +72 (2 sides x 6 dice x 6 counts, each /6 capped at 1)

Everything here is pure Python over game.py's Board, so encoder.js needs a twin
before any AB net can ship.
"""
from collections import deque

FEATURE_SETS = ('v1', 'A', 'AB')
EXTRA_TILE = {'v1': 0, 'A': 1, 'AB': 1}
EXTRA_PIECE = {'v1': 0, 'A': 1, 'AB': 3}
EXTRA_GLOBAL = {'v1': 0, 'A': 12, 'AB': 12 + 72}

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


def _piece_progress(board, p, endgame_rule=None):
    """(remaining pips, expected bank turns) for one unsaved piece. A blank
    banks on a higher die only under the ENDGAME rule (owner): outside it, it
    needs an exact die like a numbered piece."""
    saved = (board.white_saved, board.black_saved)
    if p.rack is saved[0] or p.rack is saved[1]:
        return 0.0, 0.0
    if endgame_rule is None:
        endgame_rule = board.get_game_stage(p.player) == 'endgame'
    if p.number <= 6 or not endgame_rule:
        bank = NUMBERED_BANK
    else:
        bank = BLANK_BANK.get(p.tile.number, BLANK_BANK_MEAN) if (
            p.tile is not None and p.tile.type == 'save') else BLANK_BANK_MEAN
    if p.tile is None or p.tile.type == 'home':        # rack or captured
        return float(HOME_TO_GOAL), bank
    if p.can_be_saved():                               # standing on a goal it can bank from
        return 0.0, bank
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
    eg = board.get_game_stage(player) == 'endgame'
    for p in board.pieces:
        if p.player != player or p.rack is saved_rack:
            continue
        d, bank = _piece_progress(board, p, eg)
        pips += d
        per_piece.append(d / PIPS_PER_TURN + bank)
    exact = endgame_turns(board, player)
    if exact is not None:
        turns = exact
    else:
        # CRUDE while any piece is still travelling: pieces share two dice a
        # turn, so the side is limited both by its slowest piece and by the
        # total work spread over two dice. validate_features_v2 checks it
        # against the turns the winner really took.
        turns = max(max(per_piece, default=0.0), sum(per_piece) / 2.0)
    return unentered, field, saved, pips, turns


_TABLE = None


def endgame_turns(board, player):
    """EXACT expected turns to bank everything, from endgame_table.json, when
    every unsaved piece of `player` stands on a goal it can bank from (else
    None). The table is per side: the opponent cannot block a goal or capture
    on one, so a side's banking race is independent of the other's."""
    global _TABLE
    rack = board.white_unentered if player == 'white' else board.black_unentered
    if rack:
        return None
    saved = board.white_saved if player == 'white' else board.black_saved
    mask, b = 0, [0] * 6
    for p in board.pieces:
        if p.player != player or p.rack is saved:
            continue
        t = p.tile
        if t is None or t.type != 'save' or not p.can_be_saved():
            return None
        if p.number <= 6:
            mask |= 1 << (p.number - 1)
        else:
            b[t.number - 1] += 1
    if mask == 0 and not any(b):
        return 0.0
    if sum(b) == 0 and mask & (mask - 1) == 0:          # last-piece rule
        b[mask.bit_length() - 1] = 1
        mask = 0
    if _TABLE is None:
        import json, os
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'endgame_table.json')
        _TABLE = json.load(open(path))
    return _TABLE.get(f'{mask}:' + ''.join(map(str, b)))


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
    return frozenset(t for t in board.tiles if t.type == 'field' and len(t.pieces) > 1
                     and t.pieces[0].player != mover)


# Distances depend only on (target, walls), and walls rarely change between the
# search's candidate positions, so cache them. Keyed on tile objects, which are
# per-Board; the cap bounds memory across boards.
_DIST_CACHE = {}
_CHOKE_CACHE = {}
CACHE_MAX = 200000


_MISS = object()


def _cached(cache, key, fn):
    v = cache.get(key, _MISS)
    if v is _MISS:
        if len(cache) > CACHE_MAX:
            cache.clear()
        v = cache[key] = fn()
    return v


def _dist_from(board, target, blocked):
    return _cached(_DIST_CACHE, (target, blocked), lambda: _dist_from_raw(board, target, blocked))


def _dist_from_raw(board, target, blocked):
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
    cnt = {}
    for p in board.pieces:
        if (p.player == enemy and p.tile is not None and p.tile.type != 'home'
                and p.tile is not target):
            d = dist.get(p.tile)
            if d is not None:
                cnt[d] = cnt.get(d, 0) + 1
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
    return cnt, dh, n_home, oblig


def _lands(movers, a, b, need):
    """Can the roll (a, b) put `need` (1 or 2) enemy pieces on the target?
    Honours the obligation: owed home pieces must use their dice first.
    movers[0] counts board pieces by their distance to the target."""
    cnt, dh, n_home, oblig = movers
    if need == 1:
        if oblig == 'none':
            return a in cnt or b in cnt or (a + b) in cnt
        if oblig == 'one':
            return (dh is not None and dh in (a, b, a + b)) or a in cnt or b in cnt
        return dh is not None and dh in (a, b)
    # need == 2: two different pieces, one per die
    if oblig == 'none':
        return cnt.get(a, 0) >= 2 if a == b else (a in cnt and b in cnt)
    home_both = dh is not None and n_home >= 2 and dh == a == b
    if oblig == 'one':
        return home_both or (dh is not None and ((dh == a and b in cnt) or (dh == b and a in cnt)))
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
    return _cached(_DIST_CACHE, ('route', piece.tile, goal, blocked),
                   lambda: _route_len_raw(piece.tile, goal, blocked))


def _route_len_raw(start, goal, blocked):
    seen, q = {start: 0}, deque([start])
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


def _chokes(board, piece, goal, own_block, own_occ):
    """Field tiles on EVERY shortest route of `piece` to `goal` (walling one
    lengthens or cuts the route), excluding tiles its own side occupies."""
    base = _route_len_raw(piece.tile, goal, own_block)
    if base is None:
        return ()
    d_from = _dist_from(board, piece.tile, own_block)
    d_to = _dist_from(board, goal, own_block)
    out = []
    for t in board.tiles:
        if t.type != 'field' or t is piece.tile or t not in d_from or t not in d_to:
            continue
        if d_from[t] + d_to[t] != base or t in own_occ:
            continue
        after = _route_len_raw(piece.tile, goal, own_block | {t})
        if after is None or after > base:
            out.append(t)
    return tuple(out)


def wall_prob(board, piece):
    """Approx P(the enemy's next roll builds a wall that lengthens (or cuts)
    the route of numbered `piece` to its own goal). A wall on tile W matters
    iff W lies on EVERY shortest route (blocking it raises the length). The
    enemy walls W with one more piece if one already stands there, else with
    two different pieces, one per die. Tiles holding the piece's own side are
    skipped (landing there captures rather than walls)."""
    if piece.number > 6 or piece.tile is None or piece.tile.type != 'field':
        return 0.0
    own_block = _walled_for(board, piece.player)
    goal = next(g for g in board.tiles if g.type == 'save' and g.number == piece.number)
    # Choke tiles depend on the walls and on which tiles the piece's own side
    # occupies (those are skipped), so key on both.
    own_occ = frozenset(t for t in board.tiles if t.type == 'field'
                        and any(p.player == piece.player for p in t.pieces))
    chokes = _cached(_CHOKE_CACHE, (piece.tile, goal, own_block, own_occ),
                     lambda: _chokes(board, piece, goal, own_block, own_occ))
    enemy = _other(piece.player)
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


def threat_features(board):
    """{piece: (P_capture, P_wall)} for every piece on a field tile, both
    sides -- the AB per-piece inputs. Walls and occupancy computed once."""
    walled = {pl: _walled_for(board, pl) for pl in ('white', 'black')}
    occ = {pl: frozenset(t for t in board.tiles if t.type == 'field'
                         and any(p.player == pl for p in t.pieces))
           for pl in ('white', 'black')}
    goals = {t.number: t for t in board.tiles if t.type == 'save'}
    out = {}
    for p in board.pieces:
        t = p.tile
        if t is None or t.type != 'field':
            continue
        enemy = _other(p.player)
        cap = 0.0
        if len(t.pieces) == 1:
            mv = _enemy_movers(board, enemy, t, walled[enemy])
            cap = sum(w for a, b, w in ROLLS if _lands(mv, a, b, 1))
        wall = 0.0
        if p.number <= 6:
            goal = goals[p.number]
            chokes = _cached(_CHOKE_CACHE, (t, goal, walled[p.player], occ[p.player]),
                             lambda: _chokes(board, p, goal, walled[p.player], occ[p.player]))
            if chokes:
                per = []
                for w_t in chokes:
                    already = sum(1 for q in w_t.pieces if q.player == enemy)
                    per.append((1 if already else 2,
                                _enemy_movers(board, enemy, w_t, walled[enemy])))
                wall = sum(w for a, b, w in ROLLS
                           if any(_lands(m, a, b, need) for need, m in per))
        out[p] = (cap, wall)
    return out


# --------------------------------------------------------------------------
# AB: roll opportunities ("how many good rolls do I have", owner)
# --------------------------------------------------------------------------

def _dist_from_home(board, blocked):
    """Distance from the home tile to every tile (pieces enter from home)."""
    def run():
        dist, q = {}, deque()
        for n in board.home_tile.neighbors:
            if n.type in ('nogo', 'home') or n in blocked:
                continue
            dist[n] = 1
            q.append(n)
        while q:
            t = q.popleft()
            for n in t.neighbors:
                if n in dist or n.type in ('nogo', 'home') or n in blocked:
                    continue
                dist[n] = dist[t] + 1
                q.append(n)
        return dist
    # The key must name THIS board (its home tile): with no walls `blocked` is
    # an empty frozenset that every board shares, and a dict of another board's
    # tiles silently finds nothing (caught by validate_roll_opps).
    return _cached(_DIST_CACHE, ('home', board.home_tile, blocked), run)


def _saves_with(die, piece, stage, highest_goal):
    """Would a die of `die` save `piece`? game.py's get_saving_die for a
    hypothetical die: no saves in the opening; numbered pieces need an exact
    match; in the endgame a blank on the side's highest occupied goal also
    banks on any higher die."""
    t = piece.tile
    if stage == 'opening' or t is None or t.type != 'save':
        return False
    if piece.number <= 6:
        return piece.number == t.number and die == t.number
    if die == t.number:
        return True
    return stage == 'endgame' and die > t.number and t.number == highest_goal


def roll_opportunities(board, player):
    """72-wide block is two calls of this. For die d = 1..6 (in that order),
    six counts: saves numbered, saves blank, onto-goal numbered, onto-goal
    blank, captures of a numbered enemy, captures of a blank enemy.
    Single die only; ignores turn obligations."""
    enemy = _other(player)
    blocked = _walled_for(board, player)
    stage = board.get_game_stage(player)
    highest = max((t.number for t in board.tiles if t.type == 'save'
                   and any(p.player == player for p in t.pieces)), default=-1)
    # movers and where each can land, by distance
    movers = []
    for p in board.pieces:
        if p.player != player or p.tile is None:
            continue
        if p.tile.type == 'home':
            movers.append((p, _dist_from_home(board, blocked)))
        else:
            movers.append((p, _dist_from(board, p.tile, blocked)))
    rack = board.white_unentered if player == 'white' else board.black_unentered
    if rack and not any(p.player == player for p in board.home_tile.pieces):
        movers.append((rack[0], _dist_from_home(board, blocked)))
    # tiles of interest by distance, per mover
    lone_enemy = {t: t.pieces[0] for t in board.tiles
                  if t.type == 'field' and len(t.pieces) == 1 and t.pieces[0].player == enemy}
    goals = [t for t in board.tiles if t.type == 'save']
    out = []
    for d in range(1, 7):
        sv_n = sv_b = gl_n = gl_b = 0
        caps_n, caps_b = set(), set()
        for p, dist in movers:
            if p.tile is not None and _saves_with(d, p, stage, highest):
                if p.number <= 6:
                    sv_n += 1
                else:
                    sv_b += 1
            onto = False
            already_home = p.tile is not None and p.tile.type == 'save' and p.can_be_saved()
            for g in ([] if already_home else goals):
                if dist.get(g) == d and (p.number > 6 or g.number == p.number):
                    onto = True
                    break
            if onto:
                if p.number <= 6:
                    gl_n += 1
                else:
                    gl_b += 1
            for t, e in lone_enemy.items():
                if dist.get(t) == d:
                    (caps_n if e.number <= 6 else caps_b).add(e)
        out += [sv_n, sv_b, gl_n, gl_b, len(caps_n), len(caps_b)]
    return [min(c, 6) / 6.0 for c in out]
