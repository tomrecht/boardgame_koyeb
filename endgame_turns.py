"""Expected turns to bank a set of pieces already sitting on goals.

Exact value iteration. Rules are read off game.py and CROSS-CHECKED against the
engine before use (both checks run in __main__):
  * numbered n: saveable ONLY from goal n, ONLY with a die == n.
  * blank: from goal g with die == g; in the ENDGAME also with die > g, but
    only while g is the highest goal number that player still occupies.
  * endgame iff every unsaved piece can_be_saved (blank on any goal, numbered
    on its own goal).

Goal-to-goal distances are 4, 7, 11, 14 (partners 1&6, 2&4, 3&5), so for pieces
sitting on goals the only turn shapes that exist are
    pass | save | move(4) | save+save | move(4)+save | save+move(4) | sum move
A second MOVE in one turn could only return a piece to where it started (4+4
between partners), which the engine's sum rule forbids anyway.

SCOPE: pieces are confined to goal tiles. Stepping onto a FIELD tile makes
can_be_saved false, which drops the player out of 'endgame' and kills the
higher-die rule for every piece, so it is structurally bad with 2+ pieces. For
ONE piece the restriction is checkable: this reproduces the published
full-board figures exactly on goals 1-4 and reads 0.001/0.006 high on goals
5-6, the two the full-board DP sometimes leaves for a field tile. Treat results
as exact when no piece sits on goal 5 or 6, and as a tight upper bound if one
does.

    python3 endgame_turns.py
"""
import itertools
from collections import deque

GOALS = [1, 2, 3, 4, 5, 6]
ROLLS = [((a, b), (1 if a == b else 2) / 36.0)
         for a in range(1, 7) for b in range(a, 7)]
BIG = 1e6


def goal_distances():
    """BFS between goal tiles on the real board, with get_reachable_tiles' filter."""
    from game import Board
    b = Board()
    goals = {t.number: t for t in b.tiles if t.type == 'save'}
    out = {}
    for g, gt in goals.items():
        dist, q = {gt: 0}, deque([gt])
        while q:
            cur = q.popleft()
            for n in cur.neighbors:
                if n not in dist and n.type not in ('nogo', 'home') and not n.is_blocked():
                    dist[n] = dist[cur] + 1
                    q.append(n)
        out[g] = {h: dist.get(ht) for h, ht in goals.items() if h != g}
    return b, goals, out


_GD = {}
def GD(a, b):
    return 0 if a == b else _GD[a][b]


def endgame(st):  return all(k == 0 or k == g for (k, g) in st)
def highest(st):  return max((g for (_, g) in st), default=-1)


def can_save(st, i, d):
    k, g = st[i]
    if k != 0:
        return g == k and d == k
    if d == g:
        return True
    return endgame(st) and d > g and g == highest(st)


def _saves(st, d):
    return [tuple(sorted(st[:i] + st[i+1:]))
            for i in range(len(st)) if can_save(st, i, d)]


def _moves(st, d):
    out = []
    for i, (k, g) in enumerate(st):
        for h in GOALS:
            if h != g and GD(g, h) == d:
                out.append(tuple(sorted(st[:i] + ((k, h),) + st[i+1:])))
    return out


def _solve(kinds):
    V = {(): 0.0}
    for r in range(1, len(kinds) + 1):
        for sub in set(itertools.combinations(sorted(kinds), r)):
            for assign in itertools.product(GOALS, repeat=r):
                V.setdefault(tuple(sorted(zip(sub, assign))), BIG)
    states = [s for s in V if s]
    for _ in range(2000):
        delta = 0.0
        for st in states:
            tot = 0.0
            for (a, b), p in ROLLS:
                best = V[st]                               # pass
                for s in _moves(st, a + b):                # sum move
                    best = min(best, V[s])
                for d1, d2 in ([(a, b)] if a == b else [(a, b), (b, a)]):
                    for s1 in _saves(st, d1):
                        best = min(best, V[s1])
                        for s2 in _saves(s1, d2) + _moves(s1, d2):
                            best = min(best, V[s2])
                    for s1 in _moves(st, d1):              # step on, then bank
                        best = min(best, V[s1])
                        for s2 in _saves(s1, d2):
                            best = min(best, V[s2])
                tot += p * best
            new = 1.0 + tot
            if new < V[st] - 1e-14:
                delta = max(delta, V[st] - new)
                V[st] = new
        if delta < 1e-13:
            break
    return V


_cache = {}
def turns(pieces):
    """pieces: list of (kind, goal); kind 0 = blank, n = numbered n."""
    kinds = tuple(sorted(k for k, _ in pieces))
    if kinds not in _cache:
        _cache[kinds] = _solve(kinds)
    return _cache[kinds][tuple(sorted(pieces))]


def _check_engine(b, goals):
    """can_save() vs the engine's get_saving_die(), and the stage, exhaustively."""
    bad = checked = 0
    for kinds in [(0,), (2,), (0, 0), (2, 0), (0, 0, 0), (2, 0, 0), (4, 0), (5, 0, 0)]:
        for assign in itertools.product(GOALS, repeat=len(kinds)):
            state = tuple(sorted(zip(kinds, assign)))
            for t in b.tiles:
                t.pieces = []
            b.white_unentered = b.black_unentered = []
            b.white_saved, b.black_saved = [], []
            whites = [p for p in b.pieces if p.player == 'white']
            used = []
            for (kind, g) in state:
                want = (lambda p: p.number == kind) if kind else (lambda p: p.number > 6)
                p = next(q for q in whites if want(q) and q not in used)
                used.append(p)
                p.tile = goals[g]; p.rack = None; goals[g].pieces.append(p)
            for p in whites:
                if p not in used:
                    p.tile = None; p.rack = b.white_saved; b.white_saved.append(p)
            for p in b.pieces:
                if p.player == 'black':
                    p.tile = None; p.rack = b.black_saved; b.black_saved.append(p)
            b.game_stages['white'] = b.get_game_stage('white')
            if (b.game_stages['white'] == 'endgame') != endgame(state):
                bad += 1
            for d in range(1, 7):
                b.dice[0].number = d; b.dice[0].used = False
                b.dice[1].number = d; b.dice[1].used = True
                for i, p in enumerate(used):
                    checked += 1
                    if (d in b.get_saving_die(p)) != can_save(state, i, d):
                        bad += 1
    return bad, checked


if __name__ == '__main__':
    b, goals, dist = goal_distances()
    _GD.update(dist)

    # distances must agree with the engine's own reachability
    bad = sum(1 for g, gt in goals.items() for d in range(1, 15)
              for h, ht in goals.items() if h != g
              and (ht in set(b.get_reachable_tiles(gt, d))) != (dist[g][h] == d))
    print(f"goal distances vs engine reachability: {bad} disagreements out of 420")
    sbad, schecked = _check_engine(b, goals)
    print(f"save rule + stage vs engine:           {sbad} disagreements out of {schecked}")

    print("\nVALIDATION against the published full-board value iteration")
    print("  lone blank by goal :", [round(turns([(0, g)]), 3) for g in GOALS])
    print("  CLAUDE.md          : [1.0, 1.029, 1.125, 1.303, 1.462, 1.644]")
    print("  lone numbered      :", [round(turns([(n, n)]), 3) for n in GOALS],
          "(CLAUDE.md 3.273)")
    print(f"  two blanks goal 2  : {turns([(0,2),(0,2)]):.3f} "
          "(CLAUDE.md 1.320 MC / 1.323 closed form)")

    print("\nTURNS TO BANK")
    for label, pieces in [
        ("numbered-2 + 2 blanks, all on goal 2", [(2,2),(0,2),(0,2)]),
        ("numbered-2 goal 2 + blank goal 1",     [(2,2),(0,1)]),
        ("blanks on 2, 2, 1",                    [(0,2),(0,2),(0,1)]),
        ("blanks on 4, 2",                       [(0,4),(0,2)]),
        ("numbered-2 alone on goal 2",           [(2,2)]),
        ("numbered-2 + blank, both goal 2",      [(2,2),(0,2)]),
        ("blanks on 2, 2",                       [(0,2),(0,2)]),
        ("blanks on 2, 1",                       [(0,2),(0,1)]),
        ("blanks on 1, 1, 1",                    [(0,1),(0,1),(0,1)]),
        ("blanks on 4, 4",                       [(0,4),(0,4)]),
    ]:
        print(f"  {label:<38} {turns(pieces):6.3f}")
