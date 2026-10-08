"""Merge and summarise recorded game logs (`quahuru-games-*.jsonl`).

The recorder writes one JSON object per line, and each device keeps its own log
(localStorage is per browser, so desktop and phone produce separate files). Every
game carries a uuid, so merging is exact: concatenate, drop duplicates by id.
Re-exporting a device without clearing it is therefore harmless.

Usage:
  python3 merge_games.py logs/*.jsonl                  # summarise
  python3 merge_games.py logs/*.jsonl -o all.jsonl     # ...and write the merge
"""
import sys, json, argparse
from collections import Counter, defaultdict


def load(paths):
    games, dupes, bad = {}, 0, 0
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    g = json.loads(line)
                except Exception:
                    bad += 1
                    continue
                gid = g.get('id')
                if not gid:
                    bad += 1
                    continue
                if gid in games:
                    dupes += 1
                    # A resumed game (autosave) is written twice under one id:
                    # abandoned when the page went away, then again later. Keep
                    # the copy that got furthest.
                    old = games[gid]
                    if (len(g.get('turns', [])), bool(g.get('completed'))) > \
                       (len(old.get('turns', [])), bool(old.get('completed'))):
                        games[gid] = g
                    continue
                games[gid] = g
    return games, dupes, bad


def wilson(k, n, z=1.959964):
    if not n:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return (max(0.0, c - h) * 100, min(1.0, c + h) * 100)


def human_side(g):
    """Which colour the human played, or None for self-play / two humans."""
    w, b = g.get('whiteIsAI'), g.get('blackIsAI')
    if w and not b:
        return 'black'
    if b and not w:
        return 'white'
    return None


def summarise(games):
    print(f'\n{len(games)} unique games')
    by_dev = Counter(g.get('device', '?') for g in games.values())
    print('  by device: ' + ', '.join(f'{d} {n}' for d, n in by_dev.most_common()))
    by_model = Counter(g.get('model', '?') for g in games.values())
    print('  by model:  ' + ', '.join(f'{m} {n}' for m, n in by_model.most_common()))
    inc = sum(1 for g in games.values() if not g.get('completed'))
    print(f'  abandoned (not counted below): {inc}')

    # HUMAN GAMES ONLY, split by effective difficulty -- after the slider remap a
    # saved setting means something different, so mixing them would be meaningless.
    # Games where a hint was consulted are reported SEPARATELY rather than dropped:
    # they are not a measure of unaided play, and silently folding them in would
    # inflate the figure without saying so.
    buckets = defaultdict(lambda: {'w': 0, 'l': 0, 'd': 0, 'margin': 0, 'n': 0, 'hinted': 0})
    for g in games.values():
        if not g.get('completed'):
            continue
        side = human_side(g)
        if side is None:
            continue
        res = g.get('result') or {}
        win = res.get('winner')
        key = g.get('difficulty')
        b = buckets[key]
        if g.get('hints'):
            b['hinted'] += 1
            continue
        b['n'] += 1
        m = res.get('margin') or 0
        if win == side:
            b['w'] += 1; b['margin'] += m
        elif win in ('white', 'black'):
            b['l'] += 1; b['margin'] -= m
        else:
            b['d'] += 1
    if not buckets:
        print('\n  no completed human-vs-computer games yet')
        return
    print('\nYour record against the model (unaided games only)')
    print('  effective d   games   win%   (95% CI)      avg margin   hint-assisted')
    print('  ' + '-' * 68)
    for d in sorted(buckets, key=lambda x: (x is None, x), reverse=True):
        b = buckets[d]
        if not b['n']:
            print(f'  {str(d):>11}   {0:5d}      -                        -       {b["hinted"]:5d}')
            continue
        lo, hi = wilson(b['w'], b['n'])
        print(f'  {str(d):>11}   {b["n"]:5d}  {100*b["w"]/b["n"]:5.1f}  '
              f'({lo:4.1f}-{hi:4.1f})     {b["margin"]/b["n"]:+7.2f}       {b["hinted"]:5d}')
    print('\n  A 95% CI that still spans 50% means the sample cannot yet tell you')
    print('  whether you are ahead. Watch avg margin too: it is lower variance')
    print('  than win/loss (measured -- see the arena notes in CLAUDE.md).')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+')
    ap.add_argument('-o', '--out', help='write the merged JSONL here')
    a = ap.parse_args()
    games, dupes, bad = load(a.paths)
    if dupes:
        print(f'  ({dupes} duplicate ids skipped)')
    if bad:
        print(f'  ({bad} unparseable lines skipped)')
    summarise(games)
    if a.out:
        with open(a.out, 'w') as f:
            for g in sorted(games.values(), key=lambda x: x.get('at', '')):
                f.write(json.dumps(g) + '\n')
        print(f'\nwrote {len(games)} games to {a.out}')


if __name__ == '__main__':
    main()
