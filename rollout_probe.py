"""Rollout arbitration of a choice between two moves in real recorded positions.

For each target position, two candidate pairs A and B are applied and the game is
played out N times from each with the 1-ply net policy for BOTH sides
(select_move_pair_fast). Rollout i of A and rollout i of B use the SAME dice seed
(common random numbers), so luck mostly cancels in the paired difference.
Result: mean final margin from the mover's side, B minus A.

Target sets:
  block : computer opening turns where the net scored an alternative within
          DELTA pts of its best whose numbered pieces were >0.25 less blockable
          (sum over pieces) than the move played. A = played, B = least blockable
          such alternative.

Usage: python3 rollout_probe.py block      (results -> rollout_block.jsonl, resumes)
       python3 rollout_probe.py analyze block
"""
import copy, json, os, random, sys, time, zlib
from collections import defaultdict

import weakness_probe as W

REPO = os.path.dirname(os.path.abspath(__file__))
N_ROLL = int(os.environ.get('N_ROLL', '24'))
N_WORKERS = int(os.environ.get('N_WORKERS', '4'))
MAX_TURNS = 200
LOGS = ['quahuru-games-apvo2h65.jsonl', 'quahuru-games-f7in6olg.jsonl']


def _records():
    recs = {}
    for f in LOGS:
        for line in open(os.path.join(REPO, f)):
            r = json.loads(line)
            recs[r['id'][:8]] = r
    return recs


def block_targets():
    out = []
    for line in open(os.path.join(REPO, 'weakness_probe.jsonl')):
        g = json.loads(line)
        for r in g['rows']:
            if r['who'] == 'ai' and r['my_rack'] > 0 and r['alts']:
                if r['exp'][1] - min(a['exp'][1] for a in r['alts']) > 0.25:
                    out.append((g['game'], r['turn']))
    return out


def rollout(board, seed):
    """Play out from `board` (mover's pair applied, turn NOT yet switched)."""
    ag = W.agent()
    c = copy.deepcopy(board)
    random.seed(seed)
    c.switch_turn()
    for _ in range(MAX_TURNS):
        w, s = c.check_game_over()
        if w:
            return w, s
        if c.draw_callable:
            return None, 0
        mv = list(c.get_valid_moves())
        pr = ag.select_move_pair_fast(mv, c, c.current_player)
        if isinstance(pr, tuple) and len(pr) == 3:
            pr = (pr, (0, 0, 0))
        for m in pr:
            if m != (0, 0, 0):
                c.apply_move(m, switch_turn=False)
        c.switch_turn()
    return None, 0


def pick_block_pair(b, us, played):
    """Recompute the net's close alternatives and return the least blockable."""
    ag = W.agent()
    st = dict(b.game_stages)
    scored = ag.select_move_pair(list(b.get_valid_moves()), b, us, return_scores=True)
    b.game_stages.update(st)
    best = scored[0][0]
    th_p = W.threats_after_pair(b, played, us)
    b.game_stages.update(st)
    choice, seen = None, set()
    for sc, pair in scored:
        if (best - sc) * W.M > W.DELTA or len(seen) >= W.TOP_K:
            break
        n0 = len(b.moves)
        for m in pair:
            if m not in ((0, 0, 0), (1, 1, 1)):
                b.apply_move(m, switch_turn=False)
        k = W._piece_locs(b)
        while len(b.moves) > n0:
            b.undo_last_move()
        b.game_stages.update(st)
        if k in seen:
            continue
        seen.add(k)
        th = W.threats_after_pair(b, pair, us)
        b.game_stages.update(st)
        e = W.summed(th)[1]
        if choice is None or e < choice[0]:
            choice = (e, pair, round((best - sc) * W.M, 3))
    return W.summed(th_p)[1], choice


def job(target):
    gid, turn = target
    rec = _records()[gid]
    for t, b, played, b2 in W.turn_states(rec):
        if t != turn:
            continue
        us = b.current_player
        exp_a, (exp_b, pair_b, gap_b) = pick_block_pair(b, us, played)
        res = {}
        for arm, pair in (('A', played), ('B', pair_b)):
            n0 = len(b.moves)
            for m in pair:
                if m not in ((0, 0, 0), (1, 1, 1)):
                    b.apply_move(m, switch_turn=False)
            margins = []
            for i in range(N_ROLL):
                w, s = rollout(b, seed=zlib.crc32(f"{gid}:{turn}:{i}".encode()))
                margins.append(0 if w is None else (s if w == us else -s))
            while len(b.moves) > n0:
                b.undo_last_move()
            res[arm] = margins
        return {'game': gid, 'turn': turn, 'exp_A': exp_a, 'exp_B': exp_b,
                'net_gap_B': gap_b, 'pair_A': repr(played), 'pair_B': repr(pair_b),
                'A': res['A'], 'B': res['B']}
    return {'game': gid, 'turn': turn, 'error': 'position not reached'}


def _init():
    W.agent()


def main():
    if sys.argv[1] == 'analyze':
        return analyze(sys.argv[2])
    kind = sys.argv[1]
    out = os.path.join(REPO, f'rollout_{kind}.jsonl')
    done = set()
    if os.path.exists(out):
        done = {(r['game'], r['turn']) for r in map(json.loads, open(out))}
    targets = [t for t in block_targets() if t not in done]
    print(f'{len(targets)} positions, {N_ROLL} rollouts per arm', flush=True)
    from multiprocessing import Pool
    with Pool(N_WORKERS, initializer=_init) as pool:
        for r in pool.imap_unordered(job, targets):
            with open(out, 'a') as fh:
                fh.write(json.dumps(r) + '\n')
            if 'error' in r:
                print(r, flush=True)
            else:
                d = sum(b - a for a, b in zip(r['A'], r['B'])) / len(r['A'])
                print(f"{r['game']} t{r['turn']}: exp {r['exp_A']:.2f}->{r['exp_B']:.2f}  "
                      f"B-A {d:+.2f}", flush=True)


def analyze(kind):
    import math
    rows = [r for r in map(json.loads, open(os.path.join(REPO, f'rollout_{kind}.jsonl')))
            if 'error' not in r]
    diffs = [sum(b - a for a, b in zip(r['A'], r['B'])) / len(r['A']) for r in rows]
    n = len(diffs)
    m = sum(diffs) / n
    sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / (n - 1))
    print(f'{n} positions x {len(rows[0]["A"])} paired rollouts')
    print(f'mean margin, less-blockable move minus move played: {m:+.3f} pts '
          f'(95% CI {m - 1.96 * sd / math.sqrt(n):+.3f} .. {m + 1.96 * sd / math.sqrt(n):+.3f})')
    print(f'B better in {sum(d > 0 for d in diffs)}, worse in {sum(d < 0 for d in diffs)}')
    big = [(r, d) for r, d in zip(rows, diffs) if r['exp_A'] - r['exp_B'] > 1.0]
    if big:
        mb = sum(d for _, d in big) / len(big)
        print(f'exposure reduced by >1.0: {len(big)} positions, mean {mb:+.3f}')


if __name__ == '__main__':
    main()
