"""review_games.py -- find owner's CONFIRMED mistakes in his recorded games and
group them by theme (owner, 2026-10-27: "how can I systematically learn from the
champion?").

Pipeline (each stage writes a jsonl the next one reads):

  scan    every turn the HUMAN played against the computer: the champion's
          shallow search (app config: prefilter 12/40/5, hand rules) ranks its
          candidates; owner's own pair is scored the same way (net value of the
          afterstate). Gap = best - owner's, in POINTS (raw * 12).
  verify  for each scanned turn with gap >= VERIFY_MIN: the exact one-ply value
          of both moves -- every one of the opponent's 21 rolls answered by the
          app's own search (close_call's _deep_value). A mistake is CONFIRMED
          when the deep gap is >= CONFIRM_MIN too: two independent looks agree.
  report  themes (what the better move did that owner's did not), points lost
          per game by theme, and the worst examples with position notation for
          pos_image.mjs.

    python3 review_games.py scan   <net.pt> <out_scan.jsonl> [games.jsonl ...]
    python3 review_games.py verify <net.pt> <scan.jsonl> <out_verify.jsonl>
    python3 review_games.py report <verify.jsonl> [--top 40]

    env: N_WORKERS (default 3), SHARD=i/n (split games/turns across machines),
         VERIFY_MIN (0.15 pts), CONFIRM_MIN (0.15 pts)

Default games = every quahuru-games-*.jsonl in the repo, deduped by id (the copy
that got furthest wins). Only games with exactly one human side are used; every
net's games count, since it is owner's moves being judged.
"""
import glob, json, os, sys, time
import multiprocessing as mp

os.environ.setdefault('GATE_PREFILTER', '12,40,5')
os.environ.setdefault('BOARDGAME_DEVICE', 'cpu')
REPO = os.path.dirname(os.path.abspath(__file__))
N_WORKERS = int(os.environ.get('N_WORKERS', '3'))
VERIFY_MIN = float(os.environ.get('VERIFY_MIN', '0.15'))
CONFIRM_MIN = float(os.environ.get('CONFIRM_MIN', '0.15'))
_AG = {}


# ---------------------------------------------------------------- games

def load_games(paths=None):
    # local exports + the phone's GitHub-synced games (clone of the private
    # tom-recht/quahuru-games repo at ./games_sync, one JSON file per game)
    paths = paths or (sorted(glob.glob(os.path.join(REPO, 'quahuru-games-*.jsonl'))) +
                      sorted(glob.glob(os.path.join(REPO, 'games_sync', 'games', '*', '*.json'))))
    games = {}
    for p in paths:
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            try:
                g = json.loads(line)
            except Exception:
                continue
            if not g.get('id'):
                continue
            old = games.get(g['id'])
            if old is None or (len(g.get('turns', [])), bool(g.get('completed'))) > \
                              (len(old.get('turns', [])), bool(old.get('completed'))):
                games[g['id']] = g
    out = []
    for g in games.values():
        w, b = bool(g.get('whiteIsAI')), bool(g.get('blackIsAI'))
        if w != b:
            out.append(g)
    out.sort(key=lambda g: (g.get('t', g.get('ts', 0)) or 0, g['id']))
    return out


def _shard(items):
    s = os.environ.get('SHARD')
    if not s:
        return items
    i, n = map(int, s.split('/'))
    return [x for k, x in enumerate(items) if k % n == i]


# ---------------------------------------------------------------- agent

def _agent(path):
    if path not in _AG:
        import torch
        from network import model_from_state
        from agent_gnn import prefilter_kwargs
        from close_call import make_agent_class
        m = model_from_state(torch.load(path, map_location='cpu'), strict=False)
        C = make_agent_class()
        ag = C(model=m, hand_rules=True, **prefilter_kwargs(os.environ['GATE_PREFILTER']))
        assert ag.afterstate, 'review_games expects an afterstate net'
        _AG[path] = ag
    return _AG[path]


def _apply(board, pair):
    root = len(board.moves)
    for m in pair:
        if m != (0, 0, 0):
            board.apply_move(m, switch_turn=False)
    return root


def _undo(board, root):
    while len(board.moves) > root:
        board.undo_last_move()


def _shallow(ag, board, side, pair):
    """Net value (points, mover frame) of the afterstate of `pair`; a pair that
    ends the game gets its exact margin."""
    import torch
    from agent_gnn import NUM_PIECES
    root = _apply(board, pair)
    try:
        w, s = board.check_game_over()
        if w:
            return float((1 if w == side else -1) * s)
        with torch.no_grad():
            raw = float(ag.model(ag._encode_after(board, side)))
        return raw * NUM_PIECES
    finally:
        _undo(board, root)


def _outcome(board, pair):
    from agent_gnn import _piece_locs
    root = _apply(board, pair)
    try:
        return _canon(board)
    finally:
        _undo(board, root)


def _canon(board):
    saved = (board.white_saved, board.black_saved)
    return tuple(sorted(
        (p.player, p.number if p.number <= 6 else 0,
         p.tile.index if p.tile is not None else (-2 if p.rack in saved else -1))
        for p in board.pieces))


# ---------------------------------------------------------------- descriptors

def notation(board):
    """Position notation (game.js positionToNotation subset): side, dice, then
    per side pieces <n|x>@<ring.sector|G#|H>, rack r:, saved s:."""
    def loc(p):
        t = p.tile
        if t.type == 'save':
            return 'G%d' % t.number
        if t is board.home_tile:
            return 'H'
        return '%d.%d' % (t.ring, t.pos)
    parts = []
    for side in ('white', 'black'):
        on, rack, saved = [], [], []
        srack = board.get_save_rack(side)
        for p in board.pieces:
            if p.player != side:
                continue
            n = str(p.number) if p.number <= 6 else 'x'
            if p.tile is not None:
                on.append('%s@%s' % (n, loc(p)))
            elif p.rack is srack:
                saved.append(n)
        unent = board.white_unentered if side == 'white' else board.black_unentered
        rack = [str(p.number) if p.number <= 6 else 'x' for p in unent]
        parts.append('%s %s r:%s s:%s' % (side[0].upper(), ' '.join(sorted(on)),
                                          ','.join(rack) or '-', ','.join(sorted(saved)) or '-'))
    d = "%d%d" % (board.dice[0].number, board.dice[1].number)
    return '%s %s | %s | %s' % (board.current_player[0], d, parts[0], parts[1])


def describe(board, side, pair):
    """What a pair does: saves, captures, entries, numbered pieces brought to
    their own goal, pieces onto any goal, walls (own tiles with 2+), own lone
    numbered pieces on field tiles afterwards (exposed)."""
    opp = 'black' if side == 'white' else 'white'
    srack = board.get_save_rack(side)

    def snap():
        own_saved = len(board.white_saved if side == 'white' else board.black_saved)
        opp_home = sum(1 for p in board.pieces if p.player == opp and p.tile is None
                       and p.rack is not board.get_save_rack(opp))
        tiles = {}
        lone_num = 0
        on_goal_num = 0
        on_goal = 0
        for p in board.pieces:
            if p.player != side or p.tile is None:
                continue
            tiles.setdefault(p.tile.index, []).append(p)
        for idx, ps in tiles.items():
            t = ps[0].tile
            if t.type == 'save':
                on_goal += len(ps)
                on_goal_num += sum(1 for p in ps if p.number <= 6
                                   and t.number == p.number)
            elif t.type == 'field' and len(ps) == 1 and ps[0].number <= 6:
                lone_num += 1
        walls = sum(1 for idx, ps in tiles.items()
                    if ps[0].tile.type == 'field' and len(ps) >= 2)
        unentered = sum(1 for p in board.pieces if p.player == side and p.tile is None
                        and p.rack is not srack)
        return dict(saved=own_saved, opp_home=opp_home, walls=walls, lone_num=lone_num,
                    num_on_own_goal=on_goal_num, on_goal=on_goal, unentered=unentered)
    a = snap()
    root = _apply(board, pair)
    try:
        b = snap()
    finally:
        _undo(board, root)
    return {
        'saves': b['saved'] - a['saved'],
        'captures': max(0, b['opp_home'] - a['opp_home']),
        'entries': max(0, a['unentered'] - b['unentered']),
        'num_to_goal': b['num_on_own_goal'] - a['num_on_own_goal'],
        'onto_goal': b['on_goal'] - a['on_goal'],
        'walls': b['walls'],
        'exposed_num': b['lone_num'],
        'passes': sum(1 for m in pair if m == (0, 0, 0)),
    }


THEMES = [
    # (name, test on best-minus-owner differences)
    ('missed save', lambda d: d['saves'] > 0),
    ('saved instead of something better', lambda d: d['saves'] < 0),
    ('missed capture', lambda d: d['captures'] > 0),
    ('capture was wrong', lambda d: d['captures'] < 0),
    ('numbered piece to its goal', lambda d: d['num_to_goal'] > 0),
    ('numbered piece left its goal / not needed there', lambda d: d['num_to_goal'] < 0),
    ('wall: build one', lambda d: d['walls'] > 0),
    ('wall: break / do not build', lambda d: d['walls'] < 0),
    ('exposed numbered piece', lambda d: d['exposed_num'] < 0),
    ('left a die unused', lambda d: d['passes'] < 0),
    ('used a die where pass/hold was better', lambda d: d['passes'] > 0),
    ('goal placement', lambda d: d['onto_goal'] != 0),
]


def theme(diff, stage):
    for name, test in THEMES:
        if test(diff):
            return name
    return 'piece placement (%s)' % stage


GOAL_AT = {(7, 2): 4, (7, 4): 2, (7, 6): 5, (7, 8): 3, (7, 10): 6, (7, 12): 1}


def _where(dest):
    if dest == 'save':
        return 'bank'
    if isinstance(dest, tuple) and len(dest) == 2:
        if dest in GOAL_AT:
            return 'G%d' % GOAL_AT[dest]
        if dest == (0, 0):
            return 'H'
        return '%d.%d' % dest
    return str(dest)


def pair_str(pair):
    out = []
    for m in pair:
        if m == (0, 0, 0):
            out.append('pass')
        else:
            pid, dest, roll = m
            n = pid[1] if pid[1] <= 6 else 'x'
            out.append('%s>%s(%s)' % (n, _where(dest), roll))
    return ' '.join(out)


# ---------------------------------------------------------------- scan

def _scan_game(task):
    path, rec = task
    import weakness_probe as W
    from agent_gnn import SCORE_SCALE, NUM_PIECES
    ag = _agent(path)
    human = 'black' if rec.get('whiteIsAI') else 'white'
    rows = []
    for t, b, played, _b2 in W.turn_states(rec):
        if b.current_player != human:
            continue
        moves = b.get_valid_moves()
        if not moves or len(moves) <= 1 and (0, 0, 0) in moves:
            continue
        try:
            ranked = ag.select_move_pair(moves, b, human, return_scores=True)
        except Exception as e:
            rows.append({'game': rec['id'], 'turn': t, 'error': repr(e)[:200]})
            continue
        if not isinstance(ranked, list) or not ranked:
            continue
        played = tuple(played) + ((0, 0, 0),) * (2 - len(played))
        best = ranked[0][1]
        k_best, k_own = _outcome(b, best), _outcome(b, played)
        stage = b.get_game_stage(human)
        row = {'game': rec['id'], 'turn': t, 'side': human, 'stage': stage,
               'dice': [d.number for d in b.dice], 'pos': notation(b),
               'model': rec.get('model'), 'n_cands': len(ranked),
               'own': pair_str(played), 'best': pair_str(best),
               'own_pair': repr(played), 'best_pair': repr(best)}
        if k_best == k_own:
            row.update(gap=0.0, same=True)
        else:
            v_best = _shallow(ag, b, human, best)
            v_own = _shallow(ag, b, human, played)
            in_cands = any(_outcome(b, p) == k_own for _, p in ranked[:40])
            row.update(gap=round(v_best - v_own, 4), v_best=round(v_best, 4),
                       v_own=round(v_own, 4), same=False, own_in_cands=in_cands,
                       d_best=describe(b, human, best), d_own=describe(b, human, played))
        rows.append(row)
    return rows


def scan(net, out, paths):
    games = _shard(load_games(paths or None))
    done = set()
    if os.path.exists(out):
        for l in open(out):
            done.add(json.loads(l)['game'])
    todo = [(net, g) for g in games if g['id'] not in done]
    print('games %d, to scan %d' % (len(games), len(todo)), flush=True)
    t0 = time.time()
    with mp.Pool(N_WORKERS) as pool, open(out, 'a') as f:
        for i, rows in enumerate(pool.imap_unordered(_scan_game, todo)):
            for r in rows:
                f.write(json.dumps(r) + '\n')
            f.flush()
            print('  %d/%d games, %d turns, %.0fs' % (i + 1, len(todo), len(rows),
                                                      time.time() - t0), flush=True)


# ---------------------------------------------------------------- verify

def _verify_row(task):
    path, row = task
    import ast
    from agent_gnn import SCORE_SCALE, NUM_PIECES
    ag = _agent(path)
    rec = _GAMES[row['game']]
    b = _state_at(rec, row['turn'])
    if b is not None:
        own = ast.literal_eval(row['own_pair'])
        best = ast.literal_eval(row['best_pair'])
        d_best = ag._deep_value(b, row['side'], best) / SCORE_SCALE * NUM_PIECES
        d_own = ag._deep_value(b, row['side'], own) / SCORE_SCALE * NUM_PIECES
        r = dict(row)
        r.update(deep_best=round(d_best, 4), deep_own=round(d_own, 4),
                 deep_gap=round(d_best - d_own, 4))
        return r
    return dict(row, error='replay failed')


def _state_at(rec, t):
    """Board at the start of turn t with its recorded dice."""
    import replay_games as R
    holder = {}

    class B(R.Board):
        def __init__(s_, *a, **k):
            super().__init__(*a, **k)
            holder['b'] = s_
    orig = R.Board
    R.Board = B
    try:
        r1 = dict(rec); r1['turns'] = rec['turns'][:t]
        if not R.replay(r1)[0]:
            return None
        b = holder['b']
        for die, v in zip(b.dice, rec['turns'][t]['d']):
            die.number, die.used = v, False
        b._reachable_cache.clear()
        b.get_valid_moves()
        return b
    finally:
        R.Board = orig


_GAMES = {}


def _init_verify(paths):
    _GAMES.update({g['id']: g for g in load_games(paths or None)})


def verify(net, scan_path, out, paths=None):
    rows = [json.loads(l) for l in open(scan_path)]
    rows = [r for r in rows if not r.get('same') and r.get('gap', 0) >= VERIFY_MIN]
    rows = _shard(sorted(rows, key=lambda r: -r['gap']))    # biggest first
    done = set()
    if os.path.exists(out):
        for l in open(out):
            r = json.loads(l)
            done.add((r['game'], r['turn']))
    todo = [(net, r) for r in rows if (r['game'], r['turn']) not in done]
    print('to verify %d' % len(todo), flush=True)
    t0 = time.time()
    with mp.Pool(N_WORKERS, initializer=_init_verify, initargs=(paths,)) as pool, \
            open(out, 'a') as f:
        for i, r in enumerate(pool.imap_unordered(_verify_row, todo)):
            f.write(json.dumps(r) + '\n')
            f.flush()
            if i % 10 == 0:
                print('  %d/%d %.0fs' % (i + 1, len(todo), time.time() - t0), flush=True)


# ---------------------------------------------------------------- report

def report(paths, top=40):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p)]
    rows = [r for r in rows if 'deep_gap' in r]
    games = len({r['game'] for r in rows})
    conf = [r for r in rows if r['deep_gap'] >= CONFIRM_MIN]
    print('verified %d flagged turns from %d games; CONFIRMED %d (deep gap >= %.2f)'
          % (len(rows), games, len(conf), CONFIRM_MIN))
    by = {}
    for r in conf:
        diff = {k: r['d_best'][k] - r['d_own'][k] for k in r['d_best']}
        r['theme'] = theme(diff, r['stage'])
        by.setdefault(r['theme'], []).append(r)
    print('\n%-48s %5s %8s %8s' % ('theme', 'n', 'pts', 'pts/gm'))
    for name, rs in sorted(by.items(), key=lambda kv: -sum(r['deep_gap'] for r in kv[1])):
        s = sum(r['deep_gap'] for r in rs)
        print('%-48s %5d %8.2f %8.3f' % (name, len(rs), s, s / max(1, games)))
    print('\nworst %d:' % top)
    for r in sorted(conf, key=lambda r: -r['deep_gap'])[:top]:
        print('%.2f  %-30s %s t%d  you: %-22s  champ: %s\n      %s'
              % (r['deep_gap'], r['theme'], r['game'][:8], r['turn'], r['own'],
                 r['best'], r['pos']))


def main():
    cmd = sys.argv[1]
    if cmd == 'scan':
        scan(sys.argv[2], sys.argv[3], sys.argv[4:])
    elif cmd == 'verify':
        verify(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
    elif cmd == 'report':
        a = sys.argv[2:]
        top = 40
        if '--top' in a:
            i = a.index('--top'); top = int(a[i + 1]); a = a[:i] + a[i + 2:]
        report(a, top)


if __name__ == '__main__':
    main()
