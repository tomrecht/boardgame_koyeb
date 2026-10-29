"""Goal-layout probe (owner, 2026-10-28): how much does a goal numbering make
the board's REGIONS unequal (trio balance) and its PAIR SEALS unequal (pair
balance) -- measured jointly, because both act through the endgame rule.

Solitaire: the side to move in a real midgame position (owner's start pool)
plays ALONE under game.py rules with the heuristic agent; the opponent's
board pieces are removed (they could never move, so their walls would stand
forever); only the seal walls, if any, are placed. Score = turns to bank all 12.

A layout = goal numbers at the six goal tiles in ring order (sectors 2..12 of
ring 7; partners are positions (0,1) (2,3) (4,5)). Positions are RELABELLED
so every numbered piece keeps its goal TILE (piece n -> the number its goal
tile carries in the new layout); blanks are untouched. So a layout changes
nothing about routes -- only which die banks where.

Per position x layout x rotation (the board's 240-degree symmetry moves the
labels two positions; a balanced layout should not care which way round the
position sits) x seal (none, or two walls sealing physical pair 0/1/2 for
SEAL_TURNS turns, then removed): R playouts with common dice per playout index.

    python3 layout_probe.py run N R [workers]   -> layout_probe.jsonl (resumable)
    python3 layout_probe.py report
"""
import sys, os, json, random, itertools, time
from multiprocessing import Pool

TODAY = (4, 2, 5, 3, 6, 1)
LAYOUTS = {
    'today':     (4, 2, 5, 3, 6, 1),
    'swap16':    (4, 2, 5, 3, 1, 6),
    'swapall':   (2, 4, 3, 5, 1, 6),
    'A':         (1, 6, 3, 2, 5, 4),
    'B':         (1, 4, 5, 2, 3, 6),
    'sum7':      (1, 6, 2, 5, 3, 4),
}
GOAL_SECTORS = (2, 4, 6, 8, 10, 12)                 # ring 7
SEAL_OPTIONS = lambda a, b: [((6, a), (6, b)), ((5, a), (5, b)), ((5, a), (6, b)), ((6, a), (5, b))]
SEAL_TURNS = int(os.environ.get('SEAL_TURNS', 4))
MAX_TURNS = 150
OUT = os.environ.get('OUT', 'layout_probe.jsonl')
POOL = os.environ.get('POOL', 'start_pool.jsonl')


def rot(L, r):
    return tuple(L[(i + 2 * r) % 6] for i in range(6))


def relabel_state(state, L):
    sig = {TODAY[i]: L[i] for i in range(6)}
    s = json.loads(json.dumps(state))
    f = lambda n: sig[n] if n <= 6 else n
    for k, rack in s['racks'].items():
        for p in rack:
            p['number'] = f(p['number'])
    for p in s['boardPieces']:
        p['number'] = f(p['number'])
        p.pop('reachableBySum', None)
    for d in s['dice']:
        d['used'] = False
    return s


def make_board(state, L):
    from game import Board
    b = Board()
    for t in b.tiles:
        if t.type == 'save':
            t.number = L[GOAL_SECTORS.index(t.pos)]
    b.update_state(relabel_state(state, L))
    b.firstMove = None
    return b


def place_seal(b, pair_idx, me):
    """Put two opponent walls on a sealing tile pair; return the 4 pieces or None."""
    opp = 'black' if me == 'white' else 'white'
    a, c = GOAL_SECTORS[2 * pair_idx], GOAL_SECTORS[2 * pair_idx + 1]
    tiles = None
    for opt in SEAL_OPTIONS(a, c):
        ts = [b.get_tile(*x) for x in opt]
        if all(not any(p.player == me for p in t.pieces) for t in ts):
            tiles = ts
            break
    if tiles is None:
        return None
    need = []
    for t in tiles:
        have = sum(1 for p in t.pieces if p.player == opp)
        need += [t] * max(0, 2 - have)
    pool = (list(b.get_unentered_rack(opp)) + list(b.get_save_rack(opp)) +
            [p for p in b.home_tile.pieces if p.player == opp] +
            [p for p in b.pieces if p.player == opp and p.tile and p.tile.type == 'field'
             and len(p.tile.pieces) == 1 and p.tile not in tiles])
    if len(pool) < len(need):
        return None
    moved = []
    for t, p in zip(need, pool):
        if p.rack is not None:
            p.rack.remove(p); p.rack = None
        if p.tile is not None:
            p.tile.pieces.remove(p)
        p.tile = t; t.pieces.append(p); moved.append(p)
    b._blocked_key_cache.clear(); b._blot_key_cache.clear()
    b._distance_cache.clear(); b._reachable_cache.clear()
    return moved


def strip_opponent(b, opp):
    """Solitaire: the opponent's board pieces go to its unentered rack (its saved rack would end the game) (they never
    move, so a wall of theirs would stand forever). Only seal walls remain."""
    for p in list(b.pieces):
        if p.player == opp and p.tile is not None:
            p.tile.pieces.remove(p); p.tile = None
            r = b.get_unentered_rack(opp); r.append(p); p.rack = r
    b._blocked_key_cache.clear(); b._blot_key_cache.clear()
    b._distance_cache.clear(); b._reachable_cache.clear()


def lift_seal(b, moved, opp):
    for p in moved:
        if p.tile is not None:
            p.tile.pieces.remove(p); p.tile = None
            r = b.get_unentered_rack(opp); r.append(p); p.rack = r
    b._blocked_key_cache.clear(); b._blot_key_cache.clear()
    b._distance_cache.clear(); b._reachable_cache.clear()


_AG = None


def agent():
    global _AG
    if _AG is None:
        from agent import Agent
        _AG = Agent(weights=None)
    return _AG


def solitaire(state, L, seal, seed):
    me = state['currentTurn']
    opp = 'black' if me == 'white' else 'white'
    b = make_board(state, L)
    strip_opponent(b, opp)
    moved = None
    if seal is not None:
        moved = place_seal(b, seal, me)
        if moved is None:
            return None
    rng = random.Random(seed)
    saved = b.get_save_rack(me)
    for turn in range(MAX_TURNS):
        if len(saved) == 12:
            return turn
        if moved is not None and turn == SEAL_TURNS:
            lift_seal(b, moved, opp); moved = None
        b.current_player = me
        for d in b.dice:
            d.number = rng.randint(1, 6); d.used = False
        b.firstMove = None
        b.draw_callable = False
        b.apply_last_piece_rule()
        b._distance_cache.clear(); b._reachable_cache.clear()
        b._blocked_key_cache.clear(); b._blot_key_cache.clear()
        mv = [m for m in b.get_valid_moves() if m != (1, 1, 1)]
        state_before = random.getstate()
        pr = agent().select_move_pair(mv, b, me)
        random.setstate(state_before)
        if isinstance(pr, tuple) and len(pr) == 3:
            pr = (pr, (0, 0, 0))
        for m in pr:
            if m not in ((0, 0, 0), (1, 1, 1)):
                b.apply_move(m, switch_turn=False)
    return MAX_TURNS


def task(t):
    pid, state, name, r, seal, k = t
    L = rot(LAYOUTS[name], r)
    t0 = time.time()
    T = solitaire(state, L, seal, seed=1_000_003 * pid + 7919 * k)
    return {'pid': pid, 'layout': name, 'rot': r, 'seal': seal, 'k': k, 'T': T,
            'labels': list(L), 'secs': round(time.time() - t0, 2)}


def load_positions(n):
    rows = []
    with open(POOL) as f:
        for line in f:
            d = json.loads(line)
            if d.get('stage') == 'midgame':
                rows.append(d)
    rng = random.Random(20261028)
    rng.shuffle(rows)
    return rows[:n]


def run(n, R, workers):
    pos = load_positions(n)
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            done.add((d['pid'], d['layout'], d['rot'], d['seal'], d['k']))
    tasks = []
    for k in range(R):                       # playout index outermost: early stop is balanced
        for pid, d in enumerate(pos):
            for name in LAYOUTS:
                for r in range(3):
                    for seal in (None, 0, 1, 2):
                        if (pid, name, r, seal, k) not in done:
                            tasks.append((pid, d['state'], name, r, seal, k))
    print(f'{len(pos)} positions, {len(tasks)} playouts to go', flush=True)
    t0 = time.time()
    with Pool(workers) as p, open(OUT, 'a') as f:
        for i, res in enumerate(p.imap_unordered(task, tasks, chunksize=4)):
            f.write(json.dumps(res) + '\n'); f.flush()
            if (i + 1) % 200 == 0:
                el = time.time() - t0
                print(f'{i + 1}/{len(tasks)}  {el / 60:.1f} min, eta {el / (i + 1) * (len(tasks) - i - 1) / 3600:.1f} h', flush=True)


def report():
    import statistics as st
    rows = [json.loads(l) for l in open(OUT)]
    rows = [r for r in rows if r['T'] is not None]
    by = {}
    for r in rows:
        by.setdefault((r['pid'], r['layout'], r['rot'], r['seal']), []).append(r['T'])
    mean = {k: sum(v) / len(v) for k, v in by.items()}
    print(f'{len(rows)} playouts; SEAL_TURNS {SEAL_TURNS}')
    print('layout   mean T   region spread (SD over 3 rotations, no seal)   seal cost by labelled pair   seal-cost spread')
    out = {}
    for name, L in LAYOUTS.items():
        pids = sorted({k[0] for k in mean if k[1] == name})
        # region: per position SD of the 3 rotations' means (no seal)
        sds, base = [], []
        for pid in pids:
            v = [mean.get((pid, name, r, None)) for r in range(3)]
            if None in v:
                continue
            sds.append(st.pstdev(v)); base += v
        # seal cost per LABELLED pair: physical pair s under rotation r holds rot(L,r)[2s..2s+1]
        cost = {}
        for pid in pids:
            for r in range(3):
                b0 = mean.get((pid, name, r, None))
                Lr = rot(L, r)
                for s in range(3):
                    v = mean.get((pid, name, r, s))
                    if b0 is None or v is None:
                        continue
                    key = tuple(sorted((Lr[2 * s], Lr[2 * s + 1])))
                    cost.setdefault(key, []).append(v - b0)
        def ci(x):
            m = sum(x) / len(x); se = st.pstdev(x) / len(x) ** 0.5 if len(x) > 1 else 0
            return m, se
        cs = {k: ci(v) for k, v in cost.items()}
        spread = max(m for m, _ in cs.values()) - min(m for m, _ in cs.values()) if cs else float('nan')
        rs = ci(sds) if sds else (float('nan'), 0)
        print(f"{name:8s} {sum(base) / max(1, len(base)):6.2f}   {rs[0]:.3f} +- {rs[1]:.3f} (n={len(sds)})   " +
              '  '.join(f"{a}&{b}: {m:+.2f}+-{se:.2f}" for (a, b), (m, se) in sorted(cs.items())) +
              f"   {spread:.2f}")
        out[name] = {'meanT': sum(base) / max(1, len(base)), 'region_sd': rs, 'seal': {f'{a}&{b}': v for (a, b), v in cs.items()},
                     'seal_spread': spread}
    json.dump(out, open('layout_probe_report.json', 'w'), indent=1)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'run':
        run(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 3)
    elif cmd == 'report':
        report()
    elif cmd == 'time':
        pos = load_positions(3)
        for d in pos:
            for seal in (None, 0):
                t0 = time.time()
                T = solitaire(d['state'], LAYOUTS['today'], seal, 1)
                print('T', T, 'seal', seal, f'{time.time() - t0:.1f}s', flush=True)
