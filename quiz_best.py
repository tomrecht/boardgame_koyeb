"""quiz_best.py -- re-solve each quiz position over ALL legal moves (no prefilter).

The review's "champion's move" came from the app's search, which only scores the
~40 pairs the heuristic prefilter keeps; owner found a position where the net's
own favourite had been filtered out. For each task (quiz_playout.py prep format):
  1. every legal pair's afterstate scored by the net (quick value, points);
  2. distinct outcomes; the top TOP_DEEP by quick value, plus the old filtered
     best and owner's move, re-scored by the 21-roll deep check;
  3. best = highest deep value; a table of the top outcomes (quick value within
     TABLE_SPAN of the best, at most TABLE_MAX) as canonical position keys with
     their scores relative to the best -- the quiz page looks up whatever
     position the player makes in it.

    python3 quiz_best.py run <net.pt> <tasks.jsonl.gz> <out.jsonl> [count]   (SEED_BASE = first index)

Canonical key (must match the page): per side, sorted "<n|x>@<loc>" of pieces on
the board (loc ring.sector / G1-6 / H), then saved count and rack count:
"W:1@G1,x@3.6;s5;r0|B:...".
"""
import ast, gzip, json, os, sys, time

os.environ.setdefault('GATE_PREFILTER', '12,40,5')
TOP_DEEP = int(os.environ.get('TOP_DEEP', '4'))
TABLE_SPAN = float(os.environ.get('TABLE_SPAN', '0.6'))
TABLE_MAX = int(os.environ.get('TABLE_MAX', '60'))
N_WORKERS = int(os.environ.get('N_WORKERS', '4'))
_AG = {}


def _agent(path):
    if path not in _AG:
        import review_games as RG
        _AG[path] = RG._agent(path)
    return _AG[path]


def canon(b):
    parts = []
    for side in ('white', 'black'):
        on = []
        for p in b.pieces:
            if p.player == side and p.tile is not None:
                t = p.tile
                loc = ('G%d' % t.number if t.type == 'save' else
                       'H' if t is b.home_tile else '%d.%d' % (t.ring, t.pos))
                on.append('%s@%s' % (p.number if p.number <= 6 else 'x', loc))
        saved = len(b.white_saved if side == 'white' else b.black_saved)
        rack = len(b.white_unentered if side == 'white' else b.black_unentered)
        parts.append('%s:%s;s%d;r%d' % (side[0].upper(), ','.join(sorted(on)), saved, rack))
    return '|'.join(parts)


def _pairs(b):
    out = set()
    for m1 in list(b.get_valid_moves()):
        if m1 in ((0, 0, 0), (1, 1, 1)):
            continue
        n0 = len(b.moves); st = dict(b.game_stages)
        b.apply_move(m1, switch_turn=False)
        nxt = [m for m in b.get_valid_moves() if m != (1, 1, 1)]
        for m2 in nxt or [(0, 0, 0)]:
            out.add((m1, m2))
        while len(b.moves) > n0:
            b.undo_last_move()
        b.game_stages.update(st)
    if (0, 0, 0) in b.get_valid_moves() or not out:
        out.add(((0, 0, 0), (0, 0, 0)))
    return out


def _key(b, pair):
    n0 = len(b.moves); st = dict(b.game_stages)
    for m in pair:
        if m not in ((0, 0, 0), (1, 1, 1)):
            b.apply_move(m, switch_turn=False)
    k = canon(b)
    while len(b.moves) > n0:
        b.undo_last_move()
    b.game_stages.update(st)
    return k


def _job(args):
    path, t = args
    import review_games as RG
    import quiz_build as QB
    from game import Board
    from agent_gnn import SCORE_SCALE, NUM_PIECES
    ag = _agent(path)
    t0 = time.time()
    b = Board(); b.update_state(t['state']); b._reachable_cache.clear(); b.get_valid_moves()
    side = t['mover']
    own, old = ast.literal_eval(t['own']), ast.literal_eval(t['best'])
    if (1, 1, 1) in own or (1, 1, 1) in old:
        return {'idx': t['idx'], 'id': t['id'], 'skip': 'draw call'}
    pairs = _pairs(b) | {own, old}
    quick, rep = {}, {}
    for p in pairs:
        k = _key(b, p)
        q = RG._shallow(ag, b, side, p)
        if k not in quick or q > quick[k]:
            quick[k], rep[k] = q, p
    order = sorted(quick, key=lambda k: -quick[k])
    k_own, k_old = _key(b, own), _key(b, old)
    deep = {}
    for k in dict.fromkeys(order[:TOP_DEEP] + [k_old, k_own]):
        deep[k] = ag._deep_value(b, side, rep[k]) / SCORE_SCALE * NUM_PIECES
    k_best = max(deep, key=deep.get)
    q0, d0 = quick[k_best], deep[k_best]
    table = []
    for k in order:
        if quick[k] < quick[order[0]] - TABLE_SPAN or len(table) >= TABLE_MAX:
            break
        e = {'k': k, 'q': round(quick[k] - q0, 3)}
        if k in deep:
            e['d'] = round(deep[k] - d0, 3)
        table.append(e)
    for k in (k_own, k_old):
        if not any(e['k'] == k for e in table):
            table.append({'k': k, 'q': round(quick[k] - q0, 3), 'd': round(deep[k] - d0, 3)})
    bp = rep[k_best]
    return {'idx': t['idx'], 'id': t['id'], 'n_pairs': len(pairs), 'n_outcomes': len(quick),
            'best': RG.pair_str(bp), 'best_pair': repr(bp), 'bm': QB.moves(b, repr(bp)),
            'k_best': k_best, 'k_own': k_own, 'k_old': k_old,
            'gap': round(d0 - deep[k_own], 3), 'old_gap': round(deep[k_old] - deep[k_own], 3),
            'table': table, 'sec': round(time.time() - t0, 1)}


def run(path, tpath, out, count=None):
    from multiprocessing import Pool
    tasks = [json.loads(l) for l in gzip.open(tpath, 'rt')]
    lo = int(os.environ.get('SEED_BASE', '0'))
    hi = len(tasks) if count is None else min(len(tasks), lo + int(count))
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)['idx'] for l in open(out) if l.strip()}
    todo = [(path, t) for t in tasks[lo:hi] if t['idx'] not in done]
    print(f'{len(todo)} positions, {N_WORKERS} workers', flush=True)
    with Pool(N_WORKERS) as pool, open(out, 'a') as fo:
        for r in pool.imap_unordered(_job, todo):
            fo.write(json.dumps(r) + '\n'); fo.flush()
            print(r['id'], r.get('best'), r.get('gap'), r.get('old_gap'), r.get('sec'), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        run(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else None)
