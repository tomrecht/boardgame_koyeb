"""headroom/analyze.py -- the headroom oracle's result (2026-11-16).
Pairs = colour-swapped games on one seed; unit = mean margin of the searching
agent over the pair. Reports exact2 vs plain, 1-ply (full 21 rolls) vs plain,
and exact2 - 1ply seed for seed (both played vs the same plain net)."""
import glob, json, math, sys


def units(paths):
    per = {}
    for p in paths:
        for l in open(p):
            r = json.loads(l)
            per.setdefault(r['seed'], {})[r['a_white']] = r['a_margin']
    return {s: 0.5 * (v[True] + v[False]) for s, v in per.items() if len(v) == 2}


def ci(xs):
    n = len(xs)
    m = sum(xs) / n
    se = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)
    return f'{m:+.3f} (95% CI {m - 1.96 * se:+.3f}..{m + 1.96 * se:+.3f}), {n} pairs'


e2 = units(sorted(glob.glob('headroom/e2_*.jsonl')))
c1 = units(['headroom/cc1.jsonl']) if glob.glob('headroom/cc1.jsonl') else {}
print('exact2 (2 turns) + net vs net plain:', ci(list(e2.values())) if len(e2) > 1 else '-')
print('1-ply (21 rolls) + net vs net plain:', ci(list(c1.values())) if len(c1) > 1 else '-')
both = sorted(set(e2) & set(c1))
if len(both) > 1:
    print('exact2 - 1ply, same seeds:          ', ci([e2[s] - c1[s] for s in both]))
