"""filter_ab.py -- old vs new prefilter scales, decided by the 21-roll deep check.

On turns from owner's recorded games (either side to move), the app agent
(prefilter 12/40/5, hand rules on, endgame lookahead on) picks once with the
OLD scales file and once with the NEW one. Where the two picks end in different
positions, both are scored by close_call's _deep_value (points). Output: one
line per turn: same / d_old / d_new.

    python3 filter_ab.py <net.pt> <old_scales.json> <new_scales.json> <out.jsonl> [max_turns=3000]
"""
import json, os, random, sys
os.environ.setdefault('BOARDGAME_DEVICE', 'cpu'); os.environ.setdefault('GATE_PREFILTER', '12,40,5')
import review_games as RG

_A = {}


def _agents(path, old, new):
    if not _A:
        import torch
        from network import model_from_state
        from agent_gnn import GNNAgent, prefilter_kwargs
        m = model_from_state(torch.load(path, map_location='cpu'), strict=False)
        for k, f in (('old', old), ('new', new)):
            os.environ['PREFILTER_SCALES_FILE'] = f
            a = GNNAgent(model=m, hand_rules=True, **prefilter_kwargs(os.environ['GATE_PREFILTER']))
            a.endgame_lookahead = True
            _A[k] = a
        _A['cc'] = RG._agent(path)
    return _A


def _pick(a, b, side):
    mv = list(b.get_valid_moves())
    st = dict(b.game_stages)
    p = a.select_move_pair(mv, b, side)
    b.game_stages.update(st)
    if isinstance(p, tuple) and len(p) == 3:
        p = (p, (0, 0, 0))
    return p


def job(task):
    path, old, new, rec, t = task
    from agent_gnn import SCORE_SCALE, NUM_PIECES
    A = _agents(path, old, new)
    b = RG._state_at(rec, t)
    if b is None:
        return None
    side = b.current_player
    if len(list(b.get_valid_moves())) <= 1:
        return None
    po, pn = _pick(A['old'], b, side), _pick(A['new'], b, side)
    r = {'game': rec['id'], 'turn': t, 'old': RG.pair_str(po), 'new': RG.pair_str(pn)}
    r['same'] = RG._outcome(b, po) == RG._outcome(b, pn)
    if not r['same']:
        cc = A['cc']
        r['d_old'] = round(cc._deep_value(b, side, po) / SCORE_SCALE * NUM_PIECES, 4)
        r['d_new'] = round(cc._deep_value(b, side, pn) / SCORE_SCALE * NUM_PIECES, 4)
    return r


def main(path, old, new, out, maxn=3000):
    games = RG.load_games()
    tasks = [(path, old, new, g, t) for g in games for t in range(len(g['turns']))]
    random.Random(3).shuffle(tasks)
    tasks = tasks[:maxn]
    print(len(games), 'games;', len(tasks), 'turns', flush=True)
    from multiprocessing import Pool
    with Pool(int(os.environ.get('N_WORKERS', '3'))) as pool, open(out, 'w') as fo:
        for k, r in enumerate(pool.imap_unordered(job, tasks, chunksize=4)):
            if r:
                fo.write(json.dumps(r) + '\n'); fo.flush()
            if k % 200 == 0:
                print(k, flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:5], *(int(x) for x in sys.argv[5:6]))
