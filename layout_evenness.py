import sys, statistics as st, json
sys.path.insert(0, '/Users/tom/Game/BoardGame')
from layout_probe import LAYOUTS, GOAL_SECTORS
from layout_heat import solve
G = [lambda i, r=r: (i + 2 * r) % 6 for r in range(3)] + [lambda i, r=r: (1 - i + 2 * r) % 6 for r in range(3)]
# symmetry check: every image of today's layout gives the same multiset of tile values
base = sorted(round(v, 6) for v in solve(LAYOUTS['today']).values())
for g in G[3:]:
    Lg = tuple(LAYOUTS['today'][g(i)] for i in range(6))
    print('mirror multiset equal:', sorted(round(v, 6) for v in solve(Lg).values()) == base)
spoke = {5: [(5, s) for s in GOAL_SECTORS], 6: [(6, s) for s in GOAL_SECTORS]}
out = {}
for n, L in LAYOUTS.items():
    Vs = [solve(tuple(L[g(i)] for i in range(6))) for g in G]
    field = [k for k in Vs[0] if k[0] not in (0, 7)]
    orbit = st.mean(st.pstdev([V[k] for V in Vs]) for k in field)
    V = Vs[0]
    sp = {r: [V[k] for k in spoke[r]] for r in (5, 6)}
    both = sp[5] + sp[6]
    sp_sd = st.mean([st.pstdev(sp[5]), st.pstdev(sp[6])])
    out[n] = dict(orbit_sd=orbit, spoke5=sp[5], spoke6=sp[6], spoke_sd=sp_sd,
                  spoke_range=[max(sp[r]) - min(sp[r]) for r in (5, 6)])
    print(f"{n:8s} all field tiles: orbit SD {orbit:.3f} | spokes r5 {' '.join(f'{x:.2f}' for x in sp[5])} (range {max(sp[5])-min(sp[5]):.2f}) "
          f"r6 {' '.join(f'{x:.2f}' for x in sp[6])} (range {max(sp[6])-min(sp[6]):.2f}) SD {sp_sd:.3f}", flush=True)
json.dump(out, open('/Users/tom/Game/BoardGame/layout_evenness.json', 'w'), indent=1)
