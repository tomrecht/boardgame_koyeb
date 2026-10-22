"""Fixed vs random first roll, seed for seed: both files are matches of the
close-call agent vs the plain agent on the same seeds (MR3_confirm and
MR3_random). Per seed, the difference of the colour-swapped pair means.

    python3 closecall/compare_rolls.py <fixed.jsonl> <random.jsonl>
"""
import json, math, sys


def pairs(path):
    per = {}
    for l in open(path):
        r = json.loads(l)
        per.setdefault(r['seed'], {})[r['a_white']] = r['a_margin']
    return {s: 0.5 * (v[True] + v[False]) for s, v in per.items() if len(v) == 2}


a, b = pairs(sys.argv[1]), pairs(sys.argv[2])
seeds = sorted(set(a) & set(b))
d = [b[s] - a[s] for s in seeds]
n = len(d)
if n < 2:
    sys.exit(f'only {n} common pairs')
m = sum(d) / n
se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n)
ma, mb = sum(a[s] for s in seeds) / n, sum(b[s] for s in seeds) / n
print(f'{n} common seeds: fixed {ma:+.3f}, random {mb:+.3f} vs plain; '
      f'random - fixed {m:+.3f} (95% CI {m - 1.96 * se:+.3f} .. {m + 1.96 * se:+.3f})')
