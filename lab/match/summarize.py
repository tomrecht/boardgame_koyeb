"""Summarise match_play.js output: match wins for the match-aware side (A) vs
plain (B), paired by seed (A as white and as black on the same dice).
    python3 summarize.py out.jsonl"""
import json, math, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
by = collections.defaultdict(list)
for r in rows:
    by[r['seed']].append(r)
pts = []                                   # per seed: A's match wins out of its 2 (0, 0.5, 1)
for s, rs in by.items():
    if len(rs) == 2:
        pts.append(sum(1.0 if r['winner'] == 'A' else 0.5 if r['winner'] == 'draw' else 0.0 for r in rs) / 2)
n = len(pts)
if not n:
    print('no complete pairs'); sys.exit(0)
m = sum(pts) / n
sd = math.sqrt(sum((p - m) ** 2 for p in pts) / max(1, n - 1))
dec = sum(r['decisions'] for r in rows); ch = sum(r['changed'] for r in rows)
print(f'{len(rows)} matches, {n} pairs: match-aware wins {100 * m:.1f}% (95% CI {100 * (m - 1.96 * sd / math.sqrt(n)):.1f}..'
      f'{100 * (m + 1.96 * sd / math.sqrt(n)):.1f}); its pick differed from plain on {ch}/{dec} turns ({100 * ch / max(1, dec):.1f}%); '
      f'mean games per match {sum(r["games"] for r in rows) / len(rows):.1f}')
