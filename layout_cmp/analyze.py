"""Layout comparison summary (layout_cmp/run.sh outputs).
  skill   : full-strength net vs the same net handicapped (1-move greedy), paired margin
  first   : first player's (white's) mean margin in full-strength self-play
  draws, game length (turn ends), comebacks (side behind at the midpoint -- first turn
  end with 12+ pieces banked in total -- goes on to win), seals by physical goal pair
  (positions 0/1, 2/3, 4/5 in ring order; built = a pair becomes cut off from home)."""
import json, math, os, random
from collections import Counter
D = os.path.dirname(os.path.abspath(__file__))
LAY = {'today': (4, 2, 5, 3, 6, 1), 'sum7': (1, 6, 2, 5, 3, 4), 'B': (1, 4, 5, 2, 3, 6)}


def ci(xs):
    n = len(xs); m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, 1.96 * sd / math.sqrt(n)


def load(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


for L, nums in LAY.items():
    print(f'== {L}  ({"-".join(map(str, nums))})')
    h = load(f'{D}/{L}_hcap.jsonl')
    if h:
        per = {}
        for r in h: per.setdefault(r['seed'], {})[r['a_white']] = r['a_margin']
        u = [(v[True] + v[False]) / 2 for v in per.values() if len(v) == 2]
        m, e = ci(u); w = sum(r['a_margin'] > 0 for r in h) / len(h)
        print(f'  skill: full vs handicapped {m:+.3f} +- {e:.3f} pts, full wins {100*w:.1f}%  ({len(u)} pairs)')
    s = load(f'{D}/{L}_self.jsonl')
    g = [r for r in s if r['a_white']]            # the a_white=False twin is the same game
    tw = {r['seed']: r for r in s if not r['a_white']}
    same = sum(1 for r in g if r['seed'] in tw and tw[r['seed']]['traj'] == r['traj'])
    if not g: continue
    wm = [r['a_margin'] for r in g]                # A is white in these
    m, e = ci(wm)
    print(f'  games {len(g)} (twins identical {same}/{sum(r["seed"] in tw for r in g)})')
    print(f'  first player (white) margin {m:+.3f} +- {e:.3f}, white wins {100*sum(x>0 for x in wm)/len(wm):.1f}%, '
          f'draws {100*sum(x==0 for x in wm)/len(wm):.1f}%')
    T = [len(r['traj']) for r in g]
    print(f'  length: mean {sum(T)/len(T):.1f} turn ends, median {sorted(T)[len(T)//2]}')
    cb = tot = 0
    for r in g:
        mid = next((x for x in r['traj'] if x[0] + x[1] <= 12), None)
        if mid is None or mid[0] == mid[1] or r['a_margin'] == 0: continue
        lead_white = mid[0] < mid[1]; tot += 1
        cb += lead_white != (r['a_margin'] > 0)
    print(f'  comebacks: behind at midpoint wins {cb}/{tot} = {100*cb/max(tot,1):.1f}%')
    built, ends, games_with = Counter(), Counter(), Counter()
    for r in g:
        prev = {2: 0, 3: 0}; had = set()
        for x in r['traj']:
            for k in (2, 3):
                for pr in range(3):
                    mask = 3 << (2 * pr)
                    if x[k] & mask == mask:
                        ends[pr] += 1; had.add(pr)
                        if prev[k] & mask != mask: built[pr] += 1
                prev[k] = x[k]
        for pr in had: games_with[pr] += 1
    tb = sum(built.values())
    chi = sum((built[p] - tb / 3) ** 2 / (tb / 3) for p in range(3)) if tb else 0
    lab = [f'{nums[2*p]}&{nums[2*p+1]}' for p in range(3)]
    print('  seals built: ' + ', '.join(f'{lab[p]} {built[p]}' for p in range(3)) +
          f'  (total {tb}, {tb/len(g):.2f}/game, chi2 vs even {chi:.1f}, df 2, 5% = 5.99)')
    print('  games with a seal: ' + ', '.join(f'{lab[p]} {games_with[p]}' for p in range(3)) +
          '; turn ends sealed: ' + ', '.join(f'{lab[p]} {ends[p]}' for p in range(3)))
