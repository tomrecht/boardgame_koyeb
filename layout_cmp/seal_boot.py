"""Seal evenness per layout, all passes pooled (pass 1 white-only, pass 2, pass 3), bootstrap by game.
Evenness = share of seals taken by the most-sealed goal pair (1/3 = perfectly even)."""
import json, os, random
D = os.path.dirname(os.path.abspath(__file__))
SRC = [('{L}_self.jsonl', 2, 3), ('p2_{L}_self.jsonl', 9, 10), ('p3_{L}_self.jsonl', 2, 3)]
LAB = {'today': ('4&2', '5&3', '6&1'), 'sum7': ('1&6', '2&5', '3&4'), 'B': ('1&4', '5&2', '3&6')}

def seals(traj, ia, ib):
    c = [0, 0, 0]; prev = {ia: 0, ib: 0}
    for x in traj:
        for k in (ia, ib):
            for p in range(3):
                mk = 3 << (2 * p)
                if x[k] & mk == mk and prev[k] & mk != mk: c[p] += 1
            prev[k] = x[k]
    return c

G = {}
for L in LAB:
    g = []
    for f, ia, ib in SRC:
        p = os.path.join(D, f.format(L=L))
        if os.path.exists(p):
            g += [seals(r['traj'], ia, ib) for r in map(json.loads, open(p)) if r['a_white']]
    G[L] = g
def share(g):
    t = [sum(x[p] for x in g) for p in range(3)]; return max(t) / sum(t)
rng = random.Random(0); B = 2000
bs = {L: [share([g[rng.randrange(len(g))] for _ in g]) for _ in range(B)] for L, g in G.items()}
for L, g in G.items():
    t = [sum(x[p] for x in g) for p in range(3)]
    s = sorted(bs[L])
    print(f'{L:6s} {len(g)} games, {sum(t)/len(g):.2f} seals/game: ' +
          ', '.join(f'{LAB[L][p]} {t[p]}' for p in range(3)) +
          f'   largest share {share(g):.3f} ({s[50]:.3f}..{s[1949]:.3f})')
for a, b in (('sum7', 'B'), ('sum7', 'today'), ('B', 'today')):
    d = sorted(x - y for x, y in zip(bs[a], bs[b]))
    print(f'{a} - {b}: {share(G[a]) - share(G[b]):+.4f} (95% {d[50]:+.4f}..{d[1949]:+.4f})')
