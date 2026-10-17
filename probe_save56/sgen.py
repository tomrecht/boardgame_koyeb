import json, sys, os
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from game import Board
import s56

def options(b, N):
    us = b.current_player
    A, _ = s56.options(b)
    B = []
    for m in b.get_valid_moves():
        if m[0] != (us, N) or not isinstance(m[1], tuple): continue
        n0 = len(b.moves); b.apply_move(m, switch_turn=False)
        for m2 in b.get_valid_moves():
            if m2[0] == (us, N) and isinstance(m2[1], tuple):
                t = b.get_tile(*m2[1])
                if t.type == 'save' and t.number == N: B.append((m, m2))
        while len(b.moves) > n0: b.undo_last_move()
    return A, B

def positions(Ns=(1, 2, 3, 4)):
    seen = {}
    for line in open('start_pool.jsonl'):
        r = json.loads(line); st = r['state']
        b = Board(); b.update_state(st); b.firstMove = None
        side = st['currentTurn']
        mine = [p for p in b.pieces if p.player == side and p.tile]
        on = {p.number for p in mine if p.tile.type == 'save' and p.tile.number == p.number and p.number <= 6}
        if not {5, 6} <= on: continue
        for p in mine:
            if p.number in Ns and p.tile.type == 'field' and b.shortest_route_to_goal(p) == 11 and (r['game'], p.number) not in seen:
                bb = s56.setup(r); A, B = options(bb, p.number)
                if A and B: seen[(r['game'], p.number)] = (r, p.number, A, B)
    return list(seen.values())

if __name__ == '__main__':
    from collections import Counter
    ps = positions()
    print(Counter(n for _, n, _, _ in ps))
    for r, n, A, B in ps: print(n, r['game'], r['turn'], r['stage'])
