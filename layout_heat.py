"""Lone-blank heat maps per goal layout (owner, 2026-10-29): exact expected
turns for a side's LAST piece, a blank, to bank from every tile, by value
iteration with game.py generating each roll's legal turn outcomes (real routes,
real endgame rule: on a goal the lone blank banks with die >= goal number).
No opponent pieces. Layouts as in layout_probe.py.

    python3 layout_heat.py        -> layout_heat.json (+ table to stdout)
"""
import json, sys
from game import Board
from layout_probe import LAYOUTS, GOAL_SECTORS

ROLLS = [(a, b, (1 if a == b else 2) / 36.0) for a in range(1, 7) for b in range(a, 7)]
ME, BLANK = 'white', 7


def base_state(tile):
    others = [n for n in range(1, 13) if n != BLANK]
    return {'currentTurn': ME, 'dice': [{'value': 1, 'used': False}, {'value': 1, 'used': False}],
            'racks': {'whiteUnentered': [], 'whiteSaved': [{'color': 'white', 'number': n} for n in others],
                      'blackUnentered': [{'color': 'black', 'number': n} for n in range(1, 13)], 'blackSaved': []},
            'boardPieces': [{'color': 'white', 'number': BLANK, 'tile': {'ring': tile[0], 'sector': tile[1]}}],
            'noSaveTurns': 0, 'drawCallable': False}


def board(L):
    b = Board()
    for t in b.tiles:
        if t.type == 'save':
            t.number = L[GOAL_SECTORS.index(t.pos)]
    return b


def outcomes(b, tile, roll):
    """Set of end-of-turn locations ('S' = banked) reachable with this roll."""
    def setup():
        st = base_state(tile)
        st['dice'] = [{'value': roll[0], 'used': False}, {'value': roll[1], 'used': False}]
        b.update_state(st); b.firstMove = None; b.current_player = ME
        for c in ('_distance_cache', '_reachable_cache', '_blocked_key_cache', '_blot_key_cache'):
            getattr(b, c).clear()
    def where():
        p = [p for p in b.pieces if p.player == ME and p.number == BLANK][0]
        return 'S' if p.tile is None else (p.tile.ring, p.tile.pos)
    setup()
    res = {tile}
    firsts = [m for m in b.get_valid_moves() if m not in ((0, 0, 0), (1, 1, 1)) and m[2] != 0]
    for m in firsts:
        setup(); b.apply_move(m, switch_turn=False)
        w = where(); res.add(w)
        if w == 'S':
            continue
        for m2 in [x for x in b.get_valid_moves() if x not in ((0, 0, 0), (1, 1, 1)) and x[2] != 0]:
            setup(); b.apply_move(m, switch_turn=False); b.apply_move(m2, switch_turn=False)
            res.add(where())
    return res


def solve(L):
    b = board(L)
    tiles = [(t.ring, t.pos) for t in b.tiles if t.type in ('field', 'save', 'home')]
    succ = {t: [outcomes(b, t, (r[0], r[1])) for r in ROLLS] for t in tiles}
    V = {t: 10.0 for t in tiles}; V['S'] = 0.0
    for _ in range(2000):
        d = 0
        for t in tiles:
            v = 1 + sum(w * min(V[x] for x in s) for (_, _, w), s in zip(ROLLS, succ[t]))
            d = max(d, abs(v - V[t])); V[t] = v
        if d < 1e-10:
            break
    V.pop('S')
    return V


if __name__ == '__main__':
    names = sys.argv[1:] or list(LAYOUTS)
    out = {}
    for n in names:
        V = solve(LAYOUTS[n])
        out[n] = {f'{r}.{p}': round(v, 4) for (r, p), v in V.items()}
        goals = [round(V[(7, s)], 3) for s in GOAL_SECTORS]
        print(n, LAYOUTS[n], 'goals', goals, 'home', round(V[(0, 0)], 3), flush=True)
    json.dump(out, open('layout_heat.json', 'w'))
