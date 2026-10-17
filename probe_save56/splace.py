"""Real positions with a numbered N (1-4) exactly 11 from goal N; the mover's 5 and 6,
if on the board (field or goal), are PLACED on their goals."""
import json, sys, os, copy
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from game import Board
import s56, sgen

def place(st, side, n, b0):
    g = next(t for t in b0.tiles if t.type == 'save' and t.number == n)
    p = next(p for p in st['boardPieces'] if p['color'] == side and p['number'] == n)
    old = p['tile']; p['tile'] = {'ring': g.ring, 'sector': g.pos}
    return old

def positions(per_game=3):
    got = {}
    for line in open('start_pool.jsonl'):
        r = json.loads(line); st0 = r['state']; side = st0['currentTurn']
        nums = {p['number']: p for p in st0['boardPieces'] if p['color'] == side}
        if 5 not in nums or 6 not in nums: continue
        b0 = Board(); b0.update_state(st0)
        for N in (1, 2, 3, 4):
            if N not in nums: continue
            pN = next(p for p in b0.pieces if p.player == side and p.number == N)
            if pN.tile.type != 'field': continue
            k = (r['game'], N)
            if len(got.get(k, [])) >= per_game or (got.get(k) and r['turn'] - got[k][-1][0]['turn'] < 4): continue
            st = copy.deepcopy(st0); moved = 0
            for n in (5, 6):
                t = next(p for p in b0.pieces if p.player == side and p.number == n).tile
                if not (t.type == 'save' and t.number == n):
                    if t.type != 'field': break
                    place(st, side, n, b0); moved += 1
            else:
                b = Board(); b.update_state(st)
                q = next(p for p in b.pieces if p.player == side and p.number == N)
                if b.shortest_route_to_goal(q) != 11: continue
                rr = dict(r, state=st); bb = s56.setup(rr); A, Bp = sgen.options(bb, N)
                if A and Bp: got.setdefault(k, []).append((rr, N, A, Bp, moved))
    return [x for v in got.values() for x in v]

if __name__ == '__main__':
    from collections import Counter
    ps = positions()
    late = [p for p in ps if 12 - len(p[0]['state']['racks'][p[0]['state']['currentTurn'] + 'Saved']) <= 8]
    print(len(ps), 'positions;', len({p[0]['game'] for p in ps}), 'games; by N', Counter(p[1] for p in ps),
          'pieces moved', Counter(p[4] for p in ps), 'late', len(late), Counter(p[1] for p in late))
