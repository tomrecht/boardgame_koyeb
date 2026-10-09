"""owner_lookahead.py -- the endgame lookahead on owner's recorded games.

Every computer turn in owner's games where owner was down to <= 2 unsaved
pieces: the shipped agent's choice S (current model.onnx) and the lookahead's
choice L on the recorded dice. Where they differ, N_ROLL paired playouts each
(common dice, both sides the lookahead agent), as in lookahead_probe.py.

    python3 owner_lookahead.py        (-> owner_lookahead.jsonl, resumes)
    python3 owner_lookahead.py analyze
"""
import json, math, os, statistics, sys, time, zlib

import lookahead_probe as P
import weakness_probe as W

REPO = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(REPO, 'owner_lookahead.jsonl')
N_WORKERS = int(os.environ.get('N_WORKERS', '2'))
FILES = os.environ.get('FILES', 'quahuru-games-apvo2h66.jsonl').split(',')
MODEL_TAG = os.environ.get('MODEL_TAG', 'blend4way_Oct6')
_RECS = {}


def _records():
    if not _RECS:
        for f in FILES:
            for line in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f)):
                r = json.loads(line)
                if r.get('model') == MODEL_TAG:
                    _RECS[r['id'][:8]] = r
    return _RECS


def state_at(rec, t):
    """Board at the start of turn t with its recorded dice, and the pair played
    -- weakness_probe.turn_states' body for ONE turn."""
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
            return None, None
        b = holder['b']
        for die, v in zip(b.dice, rec['turns'][t]['d']):
            die.number, die.used = v, False
        r2 = dict(rec); r2['turns'] = rec['turns'][:t + 1]
        if not R.replay(r2)[0]:
            return None, None
        b2 = holder['b']
        played = []
        for mv in b2.moves[len(b.moves):]:
            q = mv['piece']
            key = (q.player, q.number)
            if key not in b.piece_lookup:
                key = next((pp.player, pp.number) for pp in b.pieces
                           if pp.player == q.player and pp.number <= 6
                           and pp.rack is not b.get_save_rack(q.player))
            played.append((key, mv['destination'], mv['roll']))
        return b, tuple(played)
    finally:
        R.Board = orig


def job(gid):
    rec = _records()[gid]
    ai = {'white': rec.get('whiteIsAI'), 'black': rec.get('blackIsAI')}
    rows = []
    human = 'white' if ai['black'] else 'black'
    for t in range(len(rec['turns']) - 1, -1, -1):
        # The human's unsaved count only falls, so walk back from the end.
        b, played = state_at(rec, t)
        if b is None:
            continue
        if P.unsaved(b, human) > 2:
            break
        side = b.current_player
        opp = 'black' if side == 'white' else 'white'
        if not ai[side] or ai[opp]:
            continue
        b.get_valid_moves()
        t0 = time.perf_counter()
        pr = P.choose(b, side, False)
        t1 = time.perf_counter()
        pl = P.choose(b, side, True)
        t2 = time.perf_counter()
        row = {'game': gid, 'turn': t, 'mover': side, 'opp_unsaved': P.unsaved(b, opp),
               'my_unsaved': P.unsaved(b, side), 'dice': [d.number for d in b.dice],
               'played': repr(played), 'S': repr(pr), 'L': repr(pl),
               't_S': round(t1 - t0, 3), 't_L': round(t2 - t1, 3)}
        seeds = [zlib.crc32(f'{gid}:{t}:{i}'.encode()) for i in range(P.N_ROLL)]
        ko, kl, kp = P.outcome(b, played), P.outcome(b, pl), P.outcome(b, pr)
        if kl != kp or kl != ko:
            row['L_m'] = [P.play_out(b, side, pl, sd) for sd in seeds]
        if kl != kp:
            row['S_m'] = [P.play_out(b, side, pr, sd) for sd in seeds]
        if kl != ko:     # what the computer actually played (the old champion)
            row['P_m'] = [P.play_out(b, side, played, sd) for sd in seeds]
        rows.append(row)
    return {'game': gid, 'rows': rows}


def analyze():
    games = [json.loads(l) for l in open(OUT)]
    rows = [r for g in games for r in g['rows']]
    div = [r for r in rows if 'S_m' in r]
    print(f'{len(games)} games, {len(rows)} computer turns with owner at <= 2 pieces, '
          f'{len(div)} where the lookahead changes the move')
    pd = [r for r in rows if 'P_m' in r]
    print(f'{len(pd)} where the computer actually PLAYED something else (old champion)')
    if pd:
        d = [statistics.mean(r['L_m']) - statistics.mean(r['P_m']) for r in pd]
        m = statistics.mean(d)
        se = statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else float('nan')
        print(f'L - played: {m:+.3f} pts (95% CI {m - 1.96 * se:+.3f}..{m + 1.96 * se:+.3f}), '
              f'better {sum(x > 0 for x in d)} / worse {sum(x < 0 for x in d)} / level {sum(x == 0 for x in d)}')
        for r, x in zip(pd, d):
            print(f"  {r['game']} t{r['turn']} opp {r['opp_unsaved']} me {r['my_unsaved']} dice {r['dice']}: {x:+.2f}"
                  f"\n    played {r['played']}\n    L      {r['L']}")
    if div:
        d = [statistics.mean(r['L_m']) - statistics.mean(r['S_m']) for r in div]
        m = statistics.mean(d)
        se = statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else float('nan')
        print(f'L - S per changed move: {m:+.3f} pts (95% CI {m - 1.96 * se:+.3f}..{m + 1.96 * se:+.3f}), '
              f'better {sum(x > 0 for x in d)} / worse {sum(x < 0 for x in d)} / level {sum(x == 0 for x in d)}')
        for r in div:
            print(f"  {r['game']} t{r['turn']} opp {r['opp_unsaved']} me {r['my_unsaved']} dice {r['dice']}: "
                  f"{statistics.mean(r['L_m']) - statistics.mean(r['S_m']):+.2f}\n    S {r['S']}\n    L {r['L']}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'analyze':
        return analyze()
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)['game'] for l in open(OUT)}
    gids = [g for g in _records() if g not in done]
    print(f'{len(gids)} games', flush=True)
    from multiprocessing import Pool
    with Pool(N_WORKERS, initializer=P.agent) as pool:
        for r in pool.imap_unordered(job, gids):
            with open(OUT, 'a') as fh:
                fh.write(json.dumps(r) + '\n')
            nd = sum('S_m' in x for x in r['rows']); npd = sum('P_m' in x for x in r['rows'])
            print(f"{r['game']}: {len(r['rows'])} turns, {nd} changed vs net, {npd} vs played", flush=True)


if __name__ == '__main__':
    main()
