"""frame_probe.py -- how much does the train/search FRAME MISMATCH cost? (2026-10-12)

Every net so far is trained on TURN-START positions (mover to move, dice rolled),
but the 2-ply search scores each candidate at the END of the mover's own turn,
still in the mover's frame -- which the net reads as "to move again with the
spent dice" (CLAUDE.md, the t49 entry). This compares that CURRENT scoring with
the TRAINED-FRAME scoring of the same candidates:

    V_trained(c) = - sum over the opponent's 21 rolls of w * V_opp(turn start)

i.e. the position after the candidate with the turn switched exactly as
switch_turn does (last-piece rule, no-save counter), opponent to move with each
roll -- the very input the net was trained on. Same net, same candidates.

    python3 frame_probe.py calib          559-position calibration benchmark:
                                          corr / slope of each frame's move gaps
                                          vs the playouts (calib_bench.json)
    python3 frame_probe.py moves [N]      owner's computer turns (blend4way games):
                                          how often the trained frame picks a
                                          different END POSITION than the
                                          current one (-> frame_moves.jsonl,
                                          resumes); N = turns to sample (1500)
    python3 frame_probe.py report         summarise frame_moves.jsonl
"""
import json, math, os, random, sys, time
os.environ.setdefault('BOARDGAME_DEVICE', 'cpu')

REPO = os.path.dirname(os.path.abspath(__file__))
NET = os.environ.get('NET', os.path.join(REPO, 'blend4way_Oct6.pt'))
MODEL_TAG = os.environ.get('MODEL_TAG', 'blend4way_Oct6')
FILES = ['quahuru-games-apvo2h66.jsonl', 'quahuru-games-f7in6olg.jsonl']
OUT = os.path.join(REPO, 'frame_moves.jsonl')
ROLLS = [(a, b, (1 if a == b else 2) / 36.0) for a in range(1, 7) for b in range(a, 7)]
_AG = []


def agent():
    if not _AG:
        from agent_gnn import GNNAgent
        ag = GNNAgent(weights_path=NET, use_prefilter=True, prefilter_top_k=40,
                      prefilter_min_k=5, first_move_prefilter=12)
        ag.endgame_lookahead = False      # compare the shallow scores only
        _AG.append(ag)
    return _AG[0]


def current_raw(ag, board, mover):
    """What the search does today: the end-of-turn board in the mover's frame."""
    return float(ag.model([ag.encoder.encode(board, mover)])[0])


def trained_encs(ag, board, mover):
    """21 turn-start encodings (opponent to move, one per roll) for the
    end-of-turn board, or a float if the game is already over (exact)."""
    w, s = board.check_game_over()
    if w:
        return (1.0 if w == mover else -1.0) * s / 12.0 if w in ('white', 'black') else 0.0
    opp = 'black' if mover == 'white' else 'white'
    saved = ag._enter_opponent_turn_deterministic(board)
    try:
        enc = ag.encoder.encode(board, opp)
    finally:
        ag._restore_turn_state(board, saved)
    return [ag._dice_variant(enc, a, b, 0.0, 0.0) for a, b, _ in ROLLS]


def trained_values(ag, encs_list):
    """Batch: mover-frame trained values for a list of trained_encs outputs."""
    flat = [e for x in encs_list if isinstance(x, list) for e in x]
    vals = list(ag.model(flat)) if flat else []
    out, i = [], 0
    for x in encs_list:
        if not isinstance(x, list):
            out.append(x)
            continue
        v = sum(w * float(vals[i + k]) for k, (_, _, w) in enumerate(ROLLS))
        out.append(-v)
        i += len(ROLLS)
    return out


def _stats(gaps, diffs):
    n = len(gaps)
    mg, md = sum(gaps) / n, sum(diffs) / n
    cov = sum((x - mg) * (y - md) for x, y in zip(gaps, diffs))
    vg = sum((x - mg) ** 2 for x in gaps)
    vd = sum((y - md) ** 2 for y in diffs)
    return cov / math.sqrt(vg * vd), cov / vg, n


def calib():
    from game import Board
    ag = agent()
    items = json.load(open(os.path.join(REPO, 'calib_bench.json')))
    board = Board()
    cur, trn, diffs = [], [], []
    t0 = time.time()
    for k, it in enumerate(items):
        v = {}
        for arm in ('A', 'B'):
            board.update_state(it['state_' + arm])
            mover = it['mover']
            c = current_raw(ag, board, mover)
            t = trained_values(ag, [trained_encs(ag, board, mover)])[0]
            v[arm] = (c, t)
        cur.append((v['B'][0] - v['A'][0]) * 12)
        trn.append((v['B'][1] - v['A'][1]) * 12)
        diffs.append(it['playout_diff'])
        if (k + 1) % 100 == 0:
            print(f'  {k + 1}/{len(items)}  {time.time() - t0:.0f}s', flush=True)
    c = _stats(cur, diffs)
    t = _stats(trn, diffs)
    # paired bootstrap of the corr difference
    rng = random.Random(0)
    n = len(diffs)
    deltas = []
    for _ in range(2000):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append(_stats([trn[i] for i in idx], [diffs[i] for i in idx])[0]
                      - _stats([cur[i] for i in idx], [diffs[i] for i in idx])[0])
    deltas.sort()
    print(f'CURRENT frame: corr {c[0]:+.3f} slope {c[1]:+.3f}  (n {c[2]})')
    print(f'TRAINED frame: corr {t[0]:+.3f} slope {t[1]:+.3f}')
    print(f'corr difference trained - current: {t[0] - c[0]:+.3f} '
          f'(95% bootstrap {deltas[50]:+.3f}..{deltas[1949]:+.3f})')
    agree = sum(1 for a, b in zip(cur, trn) if (a > 0) == (b > 0))
    print(f'the two frames agree on which arm is better in {agree}/{n}')
    json.dump({'current': cur, 'trained': trn, 'playout': diffs},
              open(os.path.join(REPO, 'frame_calib.json'), 'w'))


def _turns():
    import owner_lookahead as O
    recs = []
    for f in FILES:
        p = os.path.join(REPO, f)
        if not os.path.exists(p):
            continue
        for line in open(p):
            r = json.loads(line)
            if r.get('model') != MODEL_TAG or not r.get('completed'):
                continue
            ai = 'b' if r.get('blackIsAI') else ('w' if r.get('whiteIsAI') else None)
            if ai is None or (r.get('whiteIsAI') and r.get('blackIsAI')):
                continue
            for t, turn in enumerate(r['turns']):
                if turn['p'] == ai:
                    recs.append((r, t))
    return recs


def moves(n_sample=1500):
    import owner_lookahead as O
    from agent_gnn import _piece_locs
    ag = agent()
    turns = _turns()
    rng = random.Random(12345)
    rng.shuffle(turns)
    turns = turns[:n_sample]
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            d = json.loads(line)
            done.add((d['game'], d['turn']))
    print(f'{len(turns)} sampled computer turns, {len(done)} already done', flush=True)
    t0 = time.time()
    k = 0
    with open(OUT, 'a') as f:
        for rec, t in turns:
            key = (rec['id'][:8], t)
            if key in done:
                continue
            b, _ = O.state_at(rec, t)
            if b is None:
                f.write(json.dumps({'game': key[0], 'turn': t, 'skip': 'replay'}) + '\n')
                continue
            side = b.current_player
            st = dict(b.game_stages)
            ranked = ag.select_move_pair(list(b.get_valid_moves()), b, side, return_scores=True)
            b.game_stages.update(st)
            if not isinstance(ranked, list) or not ranked or ranked[0][0] == float('inf'):
                f.write(json.dumps({'game': key[0], 'turn': t, 'skip': 'forced/win'}) + '\n')
                continue
            cands = [(s, p) for s, p in ranked if p != ((1, 1, 1), (0, 0, 0))]
            encs, locs = [], []
            base = len(b.moves)
            for s, p in cands:
                for m in p:
                    if m != (0, 0, 0):
                        b.apply_move(m, switch_turn=False)
                encs.append(trained_encs(ag, b, side))
                locs.append(str(_piece_locs(b)))
                while len(b.moves) > base:
                    b.undo_last_move()
                b.game_stages.update(st)
            tv = trained_values(ag, encs)
            cur_scores = [s / 1000.0 for s, _ in cands]          # raw units
            ic = max(range(len(cands)), key=lambda i: cur_scores[i])
            it_ = max(range(len(cands)), key=lambda i: tv[i])
            changed = locs[ic] != locs[it_]
            row = {'game': key[0], 'turn': t, 'side': side, 'n': len(cands),
                   'changed': changed,
                   # how much each frame says the other's choice gives up (margin pts)
                   'cur_gap': (cur_scores[ic] - cur_scores[it_]) * 12,
                   'trn_gap': (tv[it_] - tv[ic]) * 12,
                   'cur_pick': str(cands[ic][1]), 'trn_pick': str(cands[it_][1]),
                   'unsaved_min': min(12 - len(b.white_saved), 12 - len(b.black_saved))}
            f.write(json.dumps(row) + '\n')
            f.flush()
            k += 1
            if k % 25 == 0:
                print(f'  {k} turns, {time.time() - t0:.0f}s', flush=True)
    report()


def report():
    rows = [json.loads(l) for l in open(OUT)]
    ok = [r for r in rows if 'skip' not in r]
    ch = [r for r in ok if r['changed']]
    n = len(ok)
    p = len(ch) / n
    se = math.sqrt(p * (1 - p) / n)
    print(f'{n} scored turns ({len(rows) - n} skipped): trained frame picks a different '
          f'end position in {len(ch)} ({100 * p:.1f}% +- {100 * 1.96 * se:.1f})')
    if ch:
        tg = sorted(r['trn_gap'] for r in ch)
        cg = sorted(r['cur_gap'] for r in ch)
        print(f'  where they differ: trained frame says current pick gives up '
              f'median {tg[len(tg) // 2]:.2f} pts (mean {sum(tg) / len(tg):.2f}); '
              f'current frame says the trained pick gives up median {cg[len(cg) // 2]:.2f}')
        print(f'  per game (~28 computer turns): trained-frame estimate of the cost '
              f'{28 * sum(tg) / n:.2f} pts -- the net judging itself, an UPPER-ish '
              f'bound to be shrunk by its calibration slope')
    for lab, f in (('opening/midgame (both > 4 unsaved)', lambda r: r['unsaved_min'] > 4),
                   ('late (a side <= 4 unsaved)', lambda r: r['unsaved_min'] <= 4)):
        sub = [r for r in ok if f(r)]
        if sub:
            print(f'  {lab}: {sum(r["changed"] for r in sub)}/{len(sub)} changed')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'report'
    if cmd == 'calib':
        calib()
    elif cmd == 'moves':
        moves(int(sys.argv[2]) if len(sys.argv) > 2 else 1500)
    else:
        report()
