"""Summarise filter_probe / rule_probe outputs for tdlow_rank (sum7)."""
import json, sys, math, random
def ci(xs, B=4000):
    if not xs: return (0, 0, 0)
    r = random.Random(0); n = len(xs); m = sum(xs)/n
    bs = sorted(sum(xs[r.randrange(n)] for _ in range(n))/n for _ in range(B))
    return m, bs[int(.025*B)], bs[int(.975*B)]
def filt(p):
    R = [json.loads(l) for l in open(p)]
    # exclude full passes (the all-moves set omits the pass)
    R = [r for r in R if r['pick'] not in ('-', 'pass')]
    diff = [r for r in R if not r['same']]
    cost = [(r['d_best'] - r['d_pick']) if not r['same'] else 0.0 for r in R]
    m, lo, hi = ci(cost)
    print(f"FILTER {p}: {len(R)} computer turns; keep-rate {1-len(diff)/max(1,len(R)):.1%} "
          f"({len(diff)} turns the filtered pick ends elsewhere); deep-check cost per turn "
          f"{m:+.4f} pts (95% CI {lo:+.4f}..{hi:+.4f}); x ~28 turns/game = {28*m:+.3f} pts/game")
def rule(p):
    R = [json.loads(l) for l in open(p)]
    for k in ('wastes', 'bank', 'strand'):
        rs = [r for r in R if r['rule'] == k]; ch = [r for r in rs if not r['same']]
        g = [r['d_on'] - r['d_off'] for r in ch]
        s = f"mean on-off {sum(g)/len(g):+.4f}, min {min(g):+.4f}, max {max(g):+.4f}" if g else ""
        print(f"RULE {k}: tried {len(rs)}, changed the move {len(ch)} {s}")
for p in sys.argv[1:]:
    try: (filt if 'filter' in p else rule)(p)
    except FileNotFoundError: print('missing', p)
