import json, sys, os, copy, zlib, time
sys.path.insert(0, os.getcwd())
from game import Board

def positions():
    seen = {}
    for line in open('start_pool.jsonl'):
        r = json.loads(line); st = r['state']
        b = Board(); b.update_state(st); b.firstMove = None
        side = st['currentTurn']
        mine = [p for p in b.pieces if p.player == side and p.tile]
        on = {p.number for p in mine if p.tile.type == 'save' and p.tile.number == p.number and p.number <= 6}
        one = [p for p in mine if p.number == 1 and p.tile.type == 'field']
        if {5, 6} <= on and one and b.shortest_route_to_goal(one[0]) == 11 and r['game'] not in seen:
            seen[r['game']] = r
    return list(seen.values())

def setup(r):
    b = Board(); b.update_state(r['state']); b.firstMove = None
    b.dice[0].number, b.dice[1].number = 5, 6
    for d in b.dice: d.used = False
    return b

def options(b):
    us = b.current_player
    mv = list(b.get_valid_moves())
    A = None
    s5 = [m for m in mv if m[0] == (us, 5) and m[1] == 'save']
    if s5:
        n0 = len(b.moves); b.apply_move(s5[0], switch_turn=False)
        s6 = [m for m in b.get_valid_moves() if m[0] == (us, 6) and m[1] == 'save']
        if s6: A = (s5[0], s6[0])
        while len(b.moves) > n0: b.undo_last_move()
    B = []
    for m in mv:
        if m[0] != (us, 1) or not isinstance(m[1], tuple): continue
        n0 = len(b.moves); b.apply_move(m, switch_turn=False)
        for m2 in b.get_valid_moves():
            if m2[0] == (us, 1) and isinstance(m2[1], tuple):
                t = b.get_tile(*m2[1])
                if t.type == 'save' and t.number == 1: B.append((m, m2))
        while len(b.moves) > n0: b.undo_last_move()
    return A, B

if __name__ == '__main__':
    for r in positions():
        b = setup(r); A, B = options(b)
        us = b.current_player
        print(r['game'], r['turn'], us, r['stage'], 'A' if A else '-', len(B), 'B routes',
              'saved', len(b.get_save_rack(us)), 'opp saved', len(b.get_save_rack('black' if us=='white' else 'white')))
