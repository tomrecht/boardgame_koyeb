"""5,6 on own goals and N (2-4) exactly 11 from goal N, made by swapping identities
of the mover's own on-board pieces (blank <-> numbered) in real positions."""
import json, sys, os, copy
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from game import Board
import s56, sgen

def swap(st, side, a, b):
    """Swap the tiles of two of `side`'s board pieces, numbers a and b."""
    bp = st['boardPieces']
    pa = next(p for p in bp if p['color'] == side and p['number'] == a)
    pb = next(p for p in bp if p['color'] == side and p['number'] == b)
    pa['tile'], pb['tile'] = pb['tile'], pa['tile']

def candidates(r):
    st0 = r['state']; side = st0['currentTurn']
    out = []
    onb = lambda st: {p['number']: p['tile'] for p in st['boardPieces'] if p['color'] == side}
    b0 = Board(); b0.update_state(st0)
    goal = {n: next(t for t in b0.tiles if t.type == 'save' and t.number == n) for n in range(1, 7)}
    st = copy.deepcopy(st0); B = onb(st)
    for g in (5, 6):                       # get g onto its goal
        if g not in B: return out          # saved or unentered
        here = [n for n, t in B.items() if (t['ring'], t['sector']) == (goal[g].ring, goal[g].pos)]
        if g in here: continue
        bl = [n for n in here if n > 6]
        if not bl: return out
        swap(st, side, g, bl[0]); B = onb(st)
    for N in (1, 2, 3, 4):
        if N not in B: continue
        b = Board(); b.update_state(st)
        pN = next(p for p in b.pieces if p.player == side and p.number == N)
        if pN.tile.type != 'field': continue      # never pull N off its goal (or the rack/home)
        if b.shortest_route_to_goal(pN) == 11:
            out.append((st, N, False)); continue
        for p in b.pieces:                 # a blank on a tile 11 from goal N
            if p.player == side and p.number > 6 and p.tile and p.tile.type == 'field':
                s2 = copy.deepcopy(st); swap(s2, side, N, p.number)
                bb = Board(); bb.update_state(s2)
                q = next(x for x in bb.pieces if x.player == side and x.number == N)
                if bb.shortest_route_to_goal(q) == 11:
                    out.append((s2, N, True)); break
    return out

def positions(per_game=3):
    got = {}
    for line in open('start_pool.jsonl'):
        r = json.loads(line)
        for st, N, swapped in candidates(r):
            k = (r['game'], N)
            if len(got.get(k, [])) >= per_game: continue
            if got.get(k) and r['turn'] - got[k][-1][0]['turn'] < 4: continue   # space them out
            rr = dict(r, state=st)
            b = s56.setup(rr); A, Bp = sgen.options(b, N)
            if A and Bp: got.setdefault(k, []).append((rr, N, A, Bp, swapped))
    return [x for v in got.values() for x in v]

if __name__ == '__main__':
    from collections import Counter
    ps = positions()
    print(len(ps), 'positions;', len({p[0]['game'] for p in ps}), 'games')
    print('by N', Counter(p[1] for p in ps), 'swapped', Counter(p[4] for p in ps))
    print('unsaved mover <=8:', sum(1 for p in ps if 12 - len(p[0]['state']['racks'][p[0]['state']['currentTurn'] + 'Saved']) <= 8))
