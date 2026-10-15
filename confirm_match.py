"""confirm_match.py -- fresh-seed head-to-head of two checkpoints, app config.

A gate that replays the same seeds every iteration selects for luck on those
seeds (fv2B: one promotion out of ~55 panel-tested candidates). This plays A vs
B on a seed block no gate has seen, colours swapped per seed (common random
numbers), in the APP's config: prefilter 12/40/5, hand rules ON. Each game is
appended to the output file as it finishes; re-running resumes.

    python3 confirm_match.py A.pt B.pt [pairs=1000] [out=confirm_<A>.jsonl]
    python3 confirm_match.py analyze out.jsonl
"""
import json, math, os, sys
import multiprocessing as mp

SEED_BASE = 81_000_000          # disjoint from gen (11M), panel (77M), pre-screen (78M), confirm gate (79M)
N_WORKERS = int(os.environ.get('N_WORKERS', '3'))
os.environ.setdefault('GATE_PREFILTER', '12,40,5')


def _game(task):
    a, b, seed, a_white = task
    from arena import _agent, _play
    wa, wb = _agent(a, hand_rules=True), _agent(b, hand_rules=True)
    winner, score = _play(seed, wa if a_white else wb, wb if a_white else wa)
    if winner is None:
        m = 0
    else:
        m = score if (winner == 'white') == a_white else -score
    return {'seed': seed, 'a_white': a_white, 'a_margin': m}


def analyze(path):
    per = {}
    for l in open(path):
        r = json.loads(l)
        per.setdefault(r['seed'], {})[r['a_white']] = r['a_margin']
    units = [0.5 * (v[True] + v[False]) for v in per.values() if len(v) == 2]
    games = [m for v in per.values() for m in v.values()]
    n = len(units)
    if n < 2:
        print(f'{path}: {len(games)} games, too few pairs'); return
    mean = sum(units) / n
    sd = math.sqrt(sum((u - mean) ** 2 for u in units) / (n - 1))
    se = sd / math.sqrt(n)
    wins = sum(m > 0 for m in games)
    print(f'{path}: {len(games)} games, {n} pairs: A wins {wins}/{len(games)} '
          f'({100 * wins / len(games):.1f}%), mean paired margin {mean:+.3f} '
          f'(95% CI {mean - 1.96 * se:+.3f} .. {mean + 1.96 * se:+.3f})')


def main():
    if sys.argv[1] == 'analyze':
        analyze(sys.argv[2]); return
    a, b = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    pairs = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    out = sys.argv[4] if len(sys.argv) > 4 else \
        f'confirm_{os.path.splitext(os.path.basename(a))[0]}.jsonl'
    done = set()
    if os.path.exists(out):
        for l in open(out):
            r = json.loads(l)
            done.add((r['seed'], r['a_white']))
    tasks = [(a, b, SEED_BASE + k, w) for k in range(pairs) for w in (True, False)
             if (SEED_BASE + k, w) not in done]
    print(f'{len(tasks)} games to play ({len(done)} done), A={a} B={b}, '
          f'prefilter {os.environ["GATE_PREFILTER"]}, hand rules on', flush=True)
    from arena import _worker_init
    with mp.Pool(N_WORKERS, initializer=_worker_init) as pool, open(out, 'a') as f:
        for i, r in enumerate(pool.imap_unordered(_game, tasks, chunksize=1), 1):
            f.write(json.dumps(r) + '\n'); f.flush()
            if i % 50 == 0:
                print(f'  {i}/{len(tasks)}', flush=True)
                analyze(out)
    analyze(out)


if __name__ == '__main__':
    main()
