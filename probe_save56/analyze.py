"""Summarise runswap.jsonl: B - A (bring N minus save 5+6), mover's margin points.

Per position the mean paired difference; overall, per N and late/early the mean
over positions with a 95% CI from a bootstrap over GAMES (positions from one game
are correlated). Also the net's own B - A from swap_prefs.json.

    python3 probe_save56/analyze.py
"""
import json, os, random, statistics as st
from collections import defaultdict

D = os.path.dirname(os.path.abspath(__file__))
prefs = {(p['game'], p['turn'], p['n']): p for p in json.load(open(f'{D}/swap_prefs.json'))}
runs = defaultdict(dict)
for line in open(f'{D}/runswap.jsonl'):
    r = json.loads(line)
    runs[(r['game'], r['turn'], r['n'])].setdefault(r['i'], {})[r['opt']] = r['margin']

rows = []
for k, v in runs.items():
    d = [x['B'] - x['A'] for x in v.values() if len(x) == 2]
    if d:
        p = prefs[k]
        rows.append({'game': k[0], 'n': k[2], 'late': p['unsaved_us'] <= 8, 'swapped': bool(p['swapped']),
                     'play': st.mean(d), 'net': p['valB'] - p['valA'], 'pairs': len(d)})


def summary(label, rs, B=4000):
    if not rs:
        return
    by_game = defaultdict(list)
    for r in rs:
        by_game[r['game']].append(r)
    games = list(by_game)
    rng = random.Random(0)
    boots = []
    for _ in range(B):
        s = [r for g in (rng.choice(games) for _ in games) for r in by_game[g]]
        boots.append(st.mean(r['play'] for r in s))
    boots.sort()
    print(f"{label:<14} {len(rs):>4} pos {len(games):>4} games   playouts B-A {st.mean(r['play'] for r in rs):+.3f}"
          f"  (95% CI {boots[int(.025 * B)]:+.3f}..{boots[int(.975 * B)]:+.3f})   net B-A {st.mean(r['net'] for r in rs):+.3f}")


print('B - A = bring N to its goal minus save the 5 and 6 (negative: saving is better)')
summary('all', rows)
for n in (1, 2, 3, 4):
    summary(f'N={n}', [r for r in rows if r['n'] == n])
summary('late (<=8)', [r for r in rows if r['late']])
summary('early (>8)', [r for r in rows if not r['late']])
summary('unmodified', [r for r in rows if not r['swapped']])
summary('swapped', [r for r in rows if r['swapped']])
