"""lookahead_probe.py -- does the endgame lookahead improve play?

Games are played forward (1-ply net policy until either side is down to 4
unsaved pieces, then the shipped 2-ply agent, app config). At every turn where
the lookahead's trigger fires (the mover's opponent has <= 2 unsaved pieces),
the shipped agent's choice S and the lookahead's choice L are both computed on
the real dice. Where they lead to different positions, both are played out
N_ROLL times with common dice (rollout i of S and of L share a seed), both
sides playing the lookahead agent. The game itself continues with S, so it
stays the baseline.

Per divergence: mean final margin from the mover's side, L minus S.
Per game: trigger turns, divergences -- so the effect scales to pts/game.

    python3 lookahead_probe.py [games=200]   (-> lookahead_probe.jsonl, resumes)
    python3 lookahead_probe.py analyze
"""
import copy, json, math, os, random, statistics, sys, time, zlib

REPO = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get('MODEL', f'{REPO}/model.onnx')
N_ROLL = int(os.environ.get('N_ROLL', '16'))
N_WORKERS = int(os.environ.get('N_WORKERS', '4'))
OUT = os.path.join(REPO, 'lookahead_probe.jsonl')
SEED_BASE = 9_100_000
MAX_TURNS = 200
_AG = []


def agent():
    if not _AG:
        from agent_gnn import GNNAgent
        _AG.append(GNNAgent(weights_path=MODEL, use_prefilter=True, prefilter_top_k=40,
                            prefilter_min_k=5, first_move_prefilter=12))
    return _AG[0]


def unsaved(b, side):
    return 12 - len(b.white_saved if side == 'white' else b.black_saved)


def choose(b, side, lookahead):
    ag = agent()
    ag.endgame_lookahead = lookahead
    st = dict(b.game_stages)
    pr = ag.select_move_pair(list(b.get_valid_moves()), b, side, difficulty=1.0)
    b.game_stages.update(st)
    if isinstance(pr, tuple) and len(pr) == 3:
        pr = (pr, (0, 0, 0))
    return pr


def apply(b, pr):
    for m in pr:
        if m not in ((0, 0, 0),):
            b.apply_move(m, switch_turn=False)


def outcome(b, pr):
    from agent_gnn import _piece_locs
    if (1, 1, 1) in pr:
        return 'draw'          # applying it sets draw_called, which no undo clears
    n0 = len(b.moves)
    st = dict(b.game_stages)
    apply(b, pr)
    k = _piece_locs(b)
    while len(b.moves) > n0:
        b.undo_last_move()
    b.game_stages.update(st)
    return k


def play_out(b, side, pr, seed):
    c = copy.deepcopy(b)
    apply(c, pr)
    random.seed(seed)
    c.switch_turn()
    for _ in range(MAX_TURNS):
        w, s = c.check_game_over()
        if w:
            return 0 if w == 'draw' else (s if w == side else -s)
        if c.draw_callable:
            return 0
        apply(c, choose(c, c.current_player, True))
        c.switch_turn()
    return 0


def game(seed):
    from game import Board
    ag = agent()
    random.seed(seed)
    b = Board()
    triggers, divs, t_look = 0, [], []
    result = None
    for turn in range(MAX_TURNS):
        w, s = b.check_game_over()
        if w or b.draw_callable:
            result = (w, s)
            break
        side = b.current_player
        opp = 'black' if side == 'white' else 'white'
        if min(unsaved(b, 'white'), unsaved(b, 'black')) > 4:
            pr = ag.select_move_pair_fast(list(b.get_valid_moves()), b, side)
            if isinstance(pr, tuple) and len(pr) == 3:
                pr = (pr, (0, 0, 0))
        else:
            pr = choose(b, side, False)
            if unsaved(b, opp) <= 2:
                triggers += 1
                rs = random.getstate()
                t0 = time.time()
                pl = choose(b, side, True)
                t_look.append(time.time() - t0)
                if outcome(b, pl) != outcome(b, pr):
                    seeds = [zlib.crc32(f'{seed}:{turn}:{i}'.encode()) for i in range(N_ROLL)]
                    S = [play_out(b, side, pr, sd) for sd in seeds]
                    L = [play_out(b, side, pl, sd) for sd in seeds]
                    divs.append({'turn': turn, 'mover': side,
                                 'my_unsaved': unsaved(b, side), 'opp_unsaved': unsaved(b, opp),
                                 'dice': [d.number for d in b.dice],
                                 'S': repr(pr), 'L': repr(pl), 'S_m': S, 'L_m': L})
                random.setstate(rs)
        apply(b, pr)
        b.switch_turn()
    return {'seed': seed, 'triggers': triggers, 'divs': divs,
            't_look': [round(x, 3) for x in t_look],
            'result': None if result is None else list(result)}


def analyze():
    rows = [json.loads(l) for l in open(OUT)]
    n = len(rows)
    trig = sum(r['triggers'] for r in rows)
    diffs = [statistics.mean(d['L_m']) - statistics.mean(d['S_m']) for r in rows for d in r['divs']]
    per_game = [sum(statistics.mean(d['L_m']) - statistics.mean(d['S_m']) for d in r['divs'])
                for r in rows]
    tl = sorted(x for r in rows for x in r['t_look'])
    print(f'{n} games, {trig} trigger turns ({trig / n:.1f}/game), '
          f'{len(diffs)} divergences ({len(diffs) / max(trig, 1):.0%} of triggers)')
    if diffs:
        m = statistics.mean(diffs)
        se = statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else float('nan')
        print(f'per divergence, L - S: {m:+.3f} pts (95% CI {m - 1.96 * se:+.3f}..{m + 1.96 * se:+.3f}); '
              f'L better {sum(d > 0 for d in diffs)} / worse {sum(d < 0 for d in diffs)} / level {sum(d == 0 for d in diffs)}')
    gm = statistics.mean(per_game)
    gse = statistics.stdev(per_game) / math.sqrt(n) if n > 1 else float('nan')
    print(f'per game: {gm:+.3f} pts (95% CI {gm - 1.96 * gse:+.3f}..{gm + 1.96 * gse:+.3f})')
    if tl:
        print(f'lookahead time per trigger turn (Python): median {tl[len(tl) // 2]:.2f}s, '
              f'p90 {tl[int(len(tl) * .9)]:.2f}s, max {tl[-1]:.2f}s')


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'analyze':
        return analyze()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)['seed'] for l in open(OUT)}
    seeds = [SEED_BASE + i for i in range(n) if SEED_BASE + i not in done]
    print(f'{len(seeds)} games to play, {N_ROLL} playouts per arm', flush=True)
    from multiprocessing import Pool
    with Pool(N_WORKERS, initializer=agent) as pool:
        for r in pool.imap_unordered(game, seeds):
            with open(OUT, 'a') as fh:
                fh.write(json.dumps(r) + '\n')
            d = [statistics.mean(x['L_m']) - statistics.mean(x['S_m']) for x in r['divs']]
            print(f"seed {r['seed']}: {r['triggers']} triggers, {len(d)} divs "
                  f"{' '.join(f'{x:+.2f}' for x in d)}", flush=True)


if __name__ == '__main__':
    main()
