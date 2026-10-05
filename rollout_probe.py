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

  cvg   : every capture-vs-goal position (cvg_breakdown.json: both a capture of
          an enemy numbered piece and an own numbered piece to its goal were
          available, no pair did both), owner's and the computer's turns: A = the net's best pair that captures an
          enemy numbered piece without putting an own numbered piece on its
          goal, B = its best pair that does the reverse. Also records the net's
          own preference (A minus B, margin points).

  disagree : owner's turns where the net's best pair differs from his
          (weakness_gaps.jsonl): EVERY turn the net scores >= TOP_GAP pts worse
          than its best, then CONTROL_N sampled from 0.01-0.1 and MID_N from
          0.1-TOP_GAP, in that order. A = owner's move, B = the net's best. The
          playouts double as rollout-labelled training targets.

Usage: python3 rollout_probe.py block|cvg|disagree   (results -> rollout_<kind>.jsonl, resumes)
       python3 rollout_probe.py analyze block|cvg|disagree
"""
import copy, json, os, random, sys, time, zlib
from collections import defaultdict

import weakness_probe as W

REPO = os.path.dirname(os.path.abspath(__file__))
N_ROLL = int(os.environ.get('N_ROLL', '24'))
N_WORKERS = int(os.environ.get('N_WORKERS', '4'))
MAX_TURNS = 200
LOGS = ['quahuru-games-apvo2h65-v2.jsonl', 'quahuru-games-f7in6olg.jsonl',
        'quahuru-games-apvo2h65-2026-10-05.jsonl']


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


def cvg_targets(max_progress=None):
    """All capture-vs-goal positions (owner's choice, 2026-10-04), shuffled
    with a fixed seed so partial results are a fair sample. max_progress
    restricts to positions whose capturable piece had come at most that far
    (progress = 7 minus its remaining route; a detour can make it <= 0)."""
    import random
    out = []
    for r in json.load(open(os.path.join(REPO, 'cvg_breakdown.json'))):
        ps = [v for v in r.get('cap_prog', {}).values() if v is not None]
        if max_progress is None or (ps and max(ps) <= max_progress):
            out.append((r['game'], r['turn']))
    random.Random(20261004).shuffle(out)
    return out


TOP_GAP = float(os.environ.get('TOP_GAP', '0.26'))
CONTROL_N = int(os.environ.get('CONTROL_N', '100'))
MID_N = int(os.environ.get('MID_N', '100'))


def disagree_targets():
    import random
    rows = []
    for line in open(os.path.join(REPO, 'weakness_gaps.jsonl')):
        g = json.loads(line)
        for r in g['rows']:
            if r['who'] == 'human' and r['gap'] is not None:
                rows.append((g['game'], r['turn'], r['gap']))
    rng = random.Random(20261005)
    top = sorted([x for x in rows if x[2] >= TOP_GAP], key=lambda x: -x[2])
    ctrl = [x for x in rows if 0.01 <= x[2] < 0.1]
    mid = [x for x in rows if 0.1 <= x[2] < TOP_GAP]
    rng.shuffle(ctrl)
    rng.shuffle(mid)
    out = top + ctrl[:CONTROL_N] + mid[:MID_N]
    return [(g, t) for g, t, _ in out]


def net_best_pair(b, us):
    ag = W.agent()
    st = dict(b.game_stages)
    scored = ag.select_move_pair(list(b.get_valid_moves()), b, us, return_scores=True)
    b.game_stages.update(st)
    return scored


def pick_cvg_pairs(b, us):
    """Best capture-only and best goal-only pair by the pure net's 2-ply score."""
    ag = W.agent()
    st = dict(b.game_stages)
    scored = ag.select_move_pair(list(b.get_valid_moves()), b, us, return_scores=True)
    b.game_stages.update(st)
    best_c = best_g = None
    for sc, pair in scored:                       # sorted best first
        c, g = W.pair_effects(b, pair, us)
        b.game_stages.update(st)
        if c and not g and best_c is None:
            best_c = (sc, pair)
        if g and not c and best_g is None:
            best_g = (sc, pair)
        if best_c and best_g:
            break
    return best_c, best_g


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


def _play_out(b, us, pair, gid, turn):
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
    return margins


def job(task):
    kind, gid, turn = task
    rec = _records()[gid]
    for t, b, played, b2 in W.turn_states(rec):
        if t != turn:
            continue
        us = b.current_player
        if kind == 'disagree':
            scored = net_best_pair(b, us)
            best_sc, best = scored[0]
            keys = {}
            for sc, pair in scored:
                n0 = len(b.moves)
                for m in pair:
                    if m not in ((0, 0, 0), (1, 1, 1)):
                        b.apply_move(m, switch_turn=False)
                keys.setdefault(W._piece_locs(b), sc)
                while len(b.moves) > n0:
                    b.undo_last_move()
            n0 = len(b.moves)
            for m in played:
                if m not in ((0, 0, 0), (1, 1, 1)):
                    b.apply_move(m, switch_turn=False)
            pk = W._piece_locs(b)
            while len(b.moves) > n0:
                b.undo_last_move()
            p_sc = keys.get(pk)
            return {'game': gid, 'turn': turn, 'mover': us,
                    'net_gap': None if p_sc is None else round((best_sc - p_sc) * W.M, 3),
                    'pair_A': repr(played), 'pair_B': repr(best),
                    'A': _play_out(b, us, played, gid, turn),
                    'B': _play_out(b, us, best, gid, turn)}
        if kind == 'cvg':
            best_c, best_g = pick_cvg_pairs(b, us)
            if not (best_c and best_g):
                return {'game': gid, 'turn': turn, 'error': 'class missing on recompute'}
            pc, pg = W.pair_effects(b, played, us)
            return {'game': gid, 'turn': turn, 'mover': us,
                    'played': 'capture' if pc and not pg else ('goal' if pg and not pc else 'other'),
                    'net_pref_A': round((best_c[0] - best_g[0]) * W.M, 3),
                    'pair_A': repr(best_c[1]), 'pair_B': repr(best_g[1]),
                    'A': _play_out(b, us, best_c[1], gid, turn),
                    'B': _play_out(b, us, best_g[1], gid, turn)}
        exp_a, (exp_b, pair_b, gap_b) = pick_block_pair(b, us, played)
        return {'game': gid, 'turn': turn, 'exp_A': exp_a, 'exp_B': exp_b,
                'net_gap_B': gap_b, 'pair_A': repr(played), 'pair_B': repr(pair_b),
                'A': _play_out(b, us, played, gid, turn),
                'B': _play_out(b, us, pair_b, gid, turn)}
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
    pool_targets = {'cvg': cvg_targets, 'disagree': disagree_targets,
                    'block': block_targets}[kind]()
    targets = [(kind, g, t) for g, t in pool_targets if (g, t) not in done]
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
                tag = (f"net pref capture {r['net_pref_A']:+.2f}" if kind == 'cvg'
                       else f"net gap {r['net_gap']}" if kind == 'disagree'
                       else f"exp {r['exp_A']:.2f}->{r['exp_B']:.2f}")
                print(f"{r['game']} t{r['turn']}: {tag}  B-A {d:+.2f}", flush=True)


def analyze(kind):
    import math
    rows = [r for r in map(json.loads, open(os.path.join(REPO, f'rollout_{kind}.jsonl')))
            if 'error' not in r]
    diffs = [sum(b - a for a, b in zip(r['A'], r['B'])) / len(r['A']) for r in rows]
    n = len(diffs)
    m = sum(diffs) / n
    sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / (n - 1))
    print(f'{n} positions x {len(rows[0]["A"])} paired rollouts')
    if kind == 'disagree':
        ci = lambda xs: (sum(xs) / len(xs), 1.96 * math.sqrt(
            sum((x - sum(xs) / len(xs)) ** 2 for x in xs) / max(1, len(xs) - 1)) / math.sqrt(len(xs)))
        print("net's best move minus owner's move (positive = the net was right):")
        for lab, lo, hi in (('net gap >= %.2f' % TOP_GAP, TOP_GAP, 99), ('0.1 - %.2f' % TOP_GAP, 0.1, TOP_GAP),
                            ('control 0.01 - 0.1', 0.01, 0.1), ('all', -99, 99)):
            sel = [(r, d) for r, d in zip(rows, diffs) if r['net_gap'] is not None and lo <= r['net_gap'] < hi]
            if not sel:
                continue
            a, h = ci([d for _, d in sel])
            g = sum(r['net_gap'] for r, _ in sel) / len(sel)
            print(f'  {lab:<22} {len(sel):>4} positions: net says {g:+.3f}, playouts say {a:+.3f} +- {h:.3f}; '
                  f'owner better in {sum(d < 0 for _, d in sel)}, net better in {sum(d > 0 for _, d in sel)}')
        return
    if kind == 'cvg':
        ci = lambda xs: (sum(xs) / len(xs), 1.96 * math.sqrt(
            sum((x - sum(xs) / len(xs)) ** 2 for x in xs) / max(1, len(xs) - 1)) / math.sqrt(len(xs)))
        m_, h_ = ci(diffs)
        print(f'goal move minus capture move: {m_:+.3f} +- {h_:.3f} pts; '
              f'goal better in {sum(d > 0 for d in diffs)}, capture better in {sum(d < 0 for d in diffs)}')
        prefs = [r['net_pref_A'] for r in rows]
        mp = sum(prefs) / n
        num = sum((p - mp) * (-d - (-m_)) for p, d in zip(prefs, diffs))
        den = math.sqrt(sum((p - mp) ** 2 for p in prefs) * sum((d - m_) ** 2 for d in diffs))
        print(f"net's preference for capture vs the playouts' (capture minus goal): corr {num / den:+.3f}")
        prog = {}
        for b in json.load(open(os.path.join(REPO, 'cvg_breakdown.json'))):
            ps = [v for v in b.get('cap_prog', {}).values() if v is not None]
            prog[(b['game'], b['turn'])] = (max(ps) if ps else None, b['who'])
        groups = [('net prefers capture', lambda r: r['net_pref_A'] > 0),
                  ('net prefers goal', lambda r: r['net_pref_A'] <= 0),
                  ("computer's turns", lambda r: prog[(r['game'], r['turn'])][1] == 'ai'),
                  ("owner's turns", lambda r: prog[(r['game'], r['turn'])][1] == 'human'),
                  ('capturable piece progress <= 1', lambda r: (prog[(r['game'], r['turn'])][0] or 0) <= 1),
                  ('progress 2-3', lambda r: 2 <= (prog[(r['game'], r['turn'])][0] or 0) <= 3),
                  ('progress 4+', lambda r: (prog[(r['game'], r['turn'])][0] or 0) >= 4)]
        for lab, f in groups:
            sel = [d for r, d in zip(rows, diffs) if f(r)]
            if sel:
                a, h = ci(sel)
                print(f'  {lab:<32} {len(sel):>4} positions: goal minus capture {a:+.3f} +- {h:.3f}')
        return
    print(f'mean margin, less-blockable move minus move played: {m:+.3f} pts '
          f'(95% CI {m - 1.96 * sd / math.sqrt(n):+.3f} .. {m + 1.96 * sd / math.sqrt(n):+.3f})')
    print(f'B better in {sum(d > 0 for d in diffs)}, worse in {sum(d < 0 for d in diffs)}')
    big = [(r, d) for r, d in zip(rows, diffs) if r['exp_A'] - r['exp_B'] > 1.0]
    if big:
        mb = sum(d for _, d in big) / len(big)
        print(f'exposure reduced by >1.0: {len(big)} positions, mean {mb:+.3f}')


if __name__ == '__main__':
    main()
