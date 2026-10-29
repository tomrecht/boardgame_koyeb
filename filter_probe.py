"""filter_probe.py -- how often the app's prefilter (12/40/5) keeps the net's own
favourite move, on the COMPUTER's turns of owner's recorded games vs a given net.

For each turn: the app agent's pick (prefilter + hand rules) vs the best of ALL
legal pairs by the same net's afterstate value (no filter). Where the two end in
different positions, both are re-scored by the 21-roll deep check
(close_call _deep_value). Output: one JSON line per turn.

    python3 filter_probe.py <net.pt> <model_tag> <out.jsonl> [max_turns=300]
"""
import json, os, random, sys, time
os.environ.setdefault('BOARDGAME_DEVICE', 'cpu'); os.environ.setdefault('GATE_PREFILTER', '12,40,5')
import review_games as RG

_A = {}


def _agents(path):
    if path not in _A:
        import torch
        from network import model_from_state
        from agent_gnn import GNNAgent, prefilter_kwargs
        m = model_from_state(torch.load(path, map_location='cpu'), strict=False)
        _A[path] = (GNNAgent(model=m, hand_rules=True, **prefilter_kwargs(os.environ['GATE_PREFILTER'])),
                    RG._agent(path))
    return _A[path]


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
    return out


def job(task):
    path, rec, t = task
    from agent_gnn import SCORE_SCALE, NUM_PIECES
    app, cc = _agents(path)
    b = RG._state_at(rec, t)
    if b is None:
        return None
    side = b.current_player
    mv = list(b.get_valid_moves())
    if len(mv) <= 1:
        return None
    st = dict(b.game_stages)
    pick = app.select_move_pair(mv, b, side)
    b.game_stages.update(st)
    if isinstance(pick, tuple) and len(pick) == 3:
        pick = (pick, (0, 0, 0))
    if pick and pick[0] == (1, 1, 1):
        return None
    pairs = _pairs(b)
    if not pairs:
        return None
    sc = {p: RG._shallow(cc, b, side, p) for p in pairs}
    best = max(sc, key=sc.get)
    r = {'game': rec['id'], 'turn': t, 'n_pairs': len(pairs),
         'pick': RG.pair_str(pick), 'best': RG.pair_str(best),
         'q_pick': round(RG._shallow(cc, b, side, pick), 4), 'q_best': round(sc[best], 4)}
    r['same'] = RG._outcome(b, pick) == RG._outcome(b, best)
    if not r['same']:
        r['d_pick'] = round(cc._deep_value(b, side, pick) / SCORE_SCALE * NUM_PIECES, 4)
        r['d_best'] = round(cc._deep_value(b, side, best) / SCORE_SCALE * NUM_PIECES, 4)
    return r


def main(path, tag, out, maxn=300):
    games = [g for g in RG.load_games() if g.get('model') == tag]
    tasks = []
    for g in games:
        ai = 'white' if g.get('whiteIsAI') else 'black'
        for t, tr in enumerate(g['turns']):
            if tr.get('p') == ai[0]:
                tasks.append((path, g, t))
    random.Random(0).shuffle(tasks)
    tasks = tasks[:maxn]
    print(len(games), 'games;', len(tasks), 'computer turns', flush=True)
    from multiprocessing import Pool
    with Pool(int(os.environ.get('N_WORKERS', '3'))) as pool, open(out, 'w') as fo:
        for k, r in enumerate(pool.imap_unordered(job, tasks)):
            if r:
                fo.write(json.dumps(r) + '\n'); fo.flush()
            if k % 20 == 0:
                print(k, flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 300)
