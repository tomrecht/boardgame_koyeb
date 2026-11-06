"""Layout comparison, pass 2 (layout_play.py records). One table, today / sum7 / B.

    python3 layout_cmp/analyze2.py [dir=layout_cmp] [prefix=p2_]

Per layout (+- = 95% half-width; 'vs today' = difference with its 95% half-width):
  skill     full net vs the same net handicapped (1-move greedy), paired margin,
            raw and LUCK-ADJUSTED (margin minus the dice luck, judged by the full net)
  luck      share of the outcome's variance that is dice luck: Var(sum luck)/Var(margin)
  first     first player's (white's) margin, raw and luck-adjusted
  dead      turns at the end already decided (the eventual winner's position value,
            by the net, >= the threshold where 95% of such positions are won -- fitted
            on all layouts pooled), mean per game and as a share of the game
  leads     lead changes per game (sign of white's position value flips; 0.25 pt hysteresis)
  close     share of turns that are close calls (best two distinct moves <= 0.1 pt apart)
            and forced (<= 1 distinct option or the 2nd >= 1 pt behind)
  contact   captures per game, walls standing per turn end (both sides)
  seals     built per game, by goal pair (physical pair, labelled by its numbers), chi2
  margins   share of games won by <= 2, by >= 8; shutouts (12); draws
  length    turn ends per game
  comeback  side behind at the midpoint (12+ banked in total) goes on to win
"""
import json, math, os, sys
from collections import Counter

D = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
PRE = sys.argv[2] if len(sys.argv) > 2 else 'p2_'
LAY = {'today': (4, 2, 5, 3, 6, 1), 'sum7': (1, 6, 2, 5, 3, 4), 'B': (1, 4, 5, 2, 3, 6)}
# record fields
MV, E, VR, G1, G2, NC, CAPS, WU, BU, CW, CB, WW, WB = range(13)


def load(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def ms(xs):
    n = len(xs)
    if n < 2:
        return (xs[0] if xs else float('nan')), float('nan'), n
    m = sum(xs) / n
    return m, 1.96 * math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n), n


def wluck(r):
    """Sum of dice luck in WHITE's frame."""
    return sum((x[VR] - x[E]) * (1 if x[MV] == 0 else -1) for x in r['traj'])


def ew(x):
    """Position value before the roll, white's frame."""
    return x[E] if x[MV] == 0 else -x[E]


def white_margin(r):
    return r['a_margin'] if r['a_white'] else -r['a_margin']


data = {L: (load(f'{D}/{PRE}{L}_self.jsonl'), load(f'{D}/{PRE}{L}_hcap.jsonl')) for L in LAY}
selfg = {L: [r for r in s if r['a_white']] for L, (s, _) in data.items()}

# dead-time threshold: smallest t such that positions with |value| >= t are won by the
# side ahead >= 95% of the time (pooled, decided games, every turn)
pts = []
for L in LAY:
    for r in selfg[L]:
        wm = white_margin(r)
        if wm == 0:
            continue
        for x in r['traj']:
            pts.append((abs(ew(x)), (ew(x) > 0) == (wm > 0)))
THR = None
for t in [i / 20 for i in range(1, 121)]:
    sub = [ok for v, ok in pts if v >= t]
    if len(sub) >= 200 and sum(sub) / len(sub) >= 0.95:
        THR = t; break
print(f'dead-time threshold: |position value| >= {THR} pts (95% won by the side ahead; {len(pts)} positions)'
      if THR else 'dead-time threshold: not found (too little data)')

rows = {}
for L, nums in LAY.items():
    s, h = data[L]; g = selfg[L]
    R = {}
    if h:
        per = {}
        for r in h:
            per.setdefault(r['seed'], {})[r['a_white']] = r
        raw, adj = [], []
        for v in per.values():
            if len(v) != 2:
                continue
            raw.append(sum(r['a_margin'] for r in v.values()) / 2)
            # A's frame luck = white luck if A is white, else minus
            adj.append(sum(r['a_margin'] - (wluck(r) if r['a_white'] else -wluck(r)) for r in v.values()) / 2)
        R['skill'] = ms(raw); R['skill_adj'] = ms(adj)
    if g:
        wm = [white_margin(r) for r in g]; lk = [wluck(r) for r in g]
        n = len(wm); mw = sum(wm) / n; ml = sum(lk) / n
        vm = sum((x - mw) ** 2 for x in wm) / (n - 1); vl = sum((x - ml) ** 2 for x in lk) / (n - 1)
        R['luck'] = (vl / vm, None, n)
        R['first'] = ms(wm); R['first_adj'] = ms([a - b for a, b in zip(wm, lk)])
        dead, deadf, leads, T = [], [], [], []
        close = forced = turns = 0; caps = []; walls = []; built = Counter(); cb = ct = 0
        for r in g:
            t = r['traj']; m = white_margin(r); T.append(len(t))
            if THR and m != 0:
                k = 0
                for x in reversed(t):
                    if (ew(x) if m > 0 else -ew(x)) >= THR: k += 1
                    else: break
                dead.append(k); deadf.append(k / max(len(t), 1))
            sign, flips = 0, 0
            for x in t:
                v = ew(x)
                if v > 0.25 and sign <= 0:
                    flips += sign != 0; sign = 1
                elif v < -0.25 and sign >= 0:
                    flips += sign != 0; sign = -1
            leads.append(flips)
            for x in t:
                turns += 1
                close += x[G1] <= 0.1
                forced += x[G1] >= 1.0
            caps.append(sum(x[CAPS] for x in t))
            walls.extend(x[WW] + x[WB] for x in t)
            prev = {CW: 0, CB: 0}
            for x in t:
                for k in (CW, CB):
                    for pr in range(3):
                        mk = 3 << (2 * pr)
                        if x[k] & mk == mk and prev[k] & mk != mk:
                            built[pr] += 1
                    prev[k] = x[k]
            mid = next((x for x in t if x[WU] + x[BU] <= 12), None)
            if mid and mid[WU] != mid[BU] and m != 0:
                ct += 1; cb += (mid[WU] < mid[BU]) != (m > 0)
        R['dead'] = ms(dead); R['deadf'] = ms(deadf); R['leads'] = ms(leads)
        R['close'] = (close / turns, None, turns); R['forced'] = (forced / turns, None, turns)
        R['caps'] = ms(caps); R['walls'] = ms(walls); R['len'] = ms(T)
        tb = sum(built.values())
        R['seals'] = (tb / n, built, sum((built[p] - tb / 3) ** 2 / (tb / 3) for p in range(3)) if tb else 0)
        R['close2'] = sum(abs(x) <= 2 and x != 0 for x in wm) / n
        R['big'] = sum(abs(x) >= 8 for x in wm) / n
        R['shut'] = sum(abs(x) == 12 for x in wm) / n
        R['draw'] = sum(x == 0 for x in wm) / n
        R['comeback'] = (cb / ct if ct else float('nan'), ct)
    rows[L] = R


def f(v, d=3):
    m, e, n = v
    return f'{m:+.{d}f} +- {e:.{d}f}' if e == e and e is not None else f'{m:.{d}f}'


def diff(a, b):
    (m1, e1, _), (m2, e2, _) = a, b
    return f'{m1 - m2:+.3f} +- {math.sqrt(e1 ** 2 + e2 ** 2):.3f}'


for L, nums in LAY.items():
    R = rows[L]
    print(f'\n== {L} ({"-".join(map(str, nums))})  self-play games {len(selfg[L])}, handicap games {len(data[L][1])}')
    if not R:
        continue
    vs = (lambda k: f'   vs today {diff(R[k], rows["today"][k])}' if L != 'today' and k in rows['today'] else '')
    if 'skill' in R:
        print(f'  skill     raw {f(R["skill"])}   luck-adjusted {f(R["skill_adj"])}{vs("skill_adj")}')
    if 'first' not in R:
        continue
    print(f'  luck      {100 * R["luck"][0]:.0f}% of the outcome variance is dice luck')
    print(f'  first     raw {f(R["first"])}   luck-adjusted {f(R["first_adj"])}{vs("first_adj")}')
    print(f'  dead      {f(R["dead"], 2)} turns, {100 * R["deadf"][0]:.1f}% of the game{vs("dead")}')
    print(f'  leads     {f(R["leads"], 2)} lead changes per game{vs("leads")}')
    print(f'  close     {100 * R["close"][0]:.1f}% close calls, {100 * R["forced"][0]:.1f}% forced')
    print(f'  contact   captures {f(R["caps"], 2)}/game{vs("caps")}; walls standing {f(R["walls"], 2)}')
    pr, b, chi = R['seals']
    lab = [f'{nums[2 * p]}&{nums[2 * p + 1]}' for p in range(3)]
    print(f'  seals     {pr:.2f}/game: ' + ', '.join(f'{lab[p]} {b[p]}' for p in range(3)) + f'  (chi2 {chi:.1f}; 5.99 = 5%)')
    print(f'  margins   <=2: {100 * R["close2"]:.1f}%  >=8: {100 * R["big"]:.1f}%  shutouts {100 * R["shut"]:.1f}%  draws {100 * R["draw"]:.1f}%')
    print(f'  length    {f(R["len"], 1)} turn ends{vs("len")}')
    c, n = R['comeback']
    print(f'  comeback  {100 * c:.1f}% of {n} (+- {196 * math.sqrt(c * (1 - c) / max(n, 1)):.1f})')
