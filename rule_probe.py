"""rule_probe.py -- does the current net still need the app's hand-coded play rules?

For turns from owner's recorded games (either side to move), the app agent
(prefilter 12/40/5, endgame lookahead on, as in agent.js) picks with all rules
ON; where a rule could have mattered, it picks again with that ONE rule OFF.
Where the two picks end in different positions, both are scored by the
21-roll deep check (close_call _deep_value). Rules:
  wastes   _wastes_save          never leave a die unused while it could save
  bank     bank-the-most         opponent's last 1-2 blanks on goal 1
  strand   _strands_last_piece   keep a last numbered piece on its own goal
A rule is tried off only when it can matter: wastes / bank when their branch
fired on the ON pick (rule_stats), strand when the mover has <= 2 unsaved pieces
with a numbered one on its own goal.

    python3 rule_probe.py <net.pt> <out.jsonl> [max_turns=6000]
Output: one line per (turn, rule) tried: same / d_on / d_off (points).
"""
import json, os, random, sys
os.environ.setdefault('BOARDGAME_DEVICE', 'cpu'); os.environ.setdefault('GATE_PREFILTER', '12,40,5')
import review_games as RG

_A = {}
ATTR = {'wastes': 'rule_no_wasted_save', 'bank': 'rule_bank_most_when_lost', 'strand': 'rule_keep_last_piece'}


def _agents(path):
    if path not in _A:
        import torch
        from network import model_from_state
        from agent_gnn import GNNAgent, prefilter_kwargs
        m = model_from_state(torch.load(path, map_location='cpu'), strict=False)
        app = GNNAgent(model=m, hand_rules=True, **prefilter_kwargs(os.environ['GATE_PREFILTER']))
        app.endgame_lookahead = True
        _A[path] = (app, RG._agent(path))
    return _A[path]


def _pick(app, b, side, off=None):
    for k, a in ATTR.items():
        setattr(app, a, k != off)
    mv = list(b.get_valid_moves())
    st = dict(b.game_stages)
    p = app.select_move_pair(mv, b, side)
    b.game_stages.update(st)
    if isinstance(p, tuple) and len(p) == 3:
        p = (p, (0, 0, 0))
    return p


def _strand_possible(b, side):
    left = [p for p in b.pieces if p.player == side and p.tile is not None]
    unentered = b.black_unentered if side == 'black' else b.white_unentered
    return (not unentered and len(left) <= 2 and
            any(p.number <= 6 and p.tile.type == 'save' and p.tile.number == p.number for p in left))


def job(task):
    path, rec, t = task
    from agent_gnn import SCORE_SCALE, NUM_PIECES
    app, cc = _agents(path)
    b = RG._state_at(rec, t)
    if b is None:
        return []
    side = b.current_player
    if len(list(b.get_valid_moves())) <= 1:
        return []
    s0 = dict(app.rule_stats)
    on = _pick(app, b, side)
    if not on or on[0] == (1, 1, 1):
        return []
    try_off = []
    if app.rule_stats['wastes_save_dropped'] > s0['wastes_save_dropped']:
        try_off.append('wastes')
    if app.rule_stats['bank_most_fired'] > s0['bank_most_fired']:
        try_off.append('bank')
    if _strand_possible(b, side):
        try_off.append('strand')
    out = []
    for rule in try_off:
        off = _pick(app, b, side, rule)
        r = {'game': rec['id'], 'turn': t, 'rule': rule, 'on': RG.pair_str(on), 'off': RG.pair_str(off)}
        r['same'] = RG._outcome(b, on) == RG._outcome(b, off)
        if not r['same']:
            r['d_on'] = round(cc._deep_value(b, side, on) / SCORE_SCALE * NUM_PIECES, 4)
            r['d_off'] = round(cc._deep_value(b, side, off) / SCORE_SCALE * NUM_PIECES, 4)
        out.append(r)
    return out


def main(path, out, maxn=6000):
    games = RG.load_games()
    tasks = [(path, g, t) for g in games for t in range(len(g['turns']))]
    random.Random(1).shuffle(tasks)
    tasks = tasks[:maxn]
    print(len(games), 'games;', len(tasks), 'turns', flush=True)
    from multiprocessing import Pool
    n = 0
    with Pool(int(os.environ.get('N_WORKERS', '3'))) as pool, open(out, 'w') as fo:
        for k, rs in enumerate(pool.imap_unordered(job, tasks, chunksize=4)):
            for r in rs:
                fo.write(json.dumps(r) + '\n'); fo.flush(); n += 1
            if k % 200 == 0:
                print(k, 'turns,', n, 'rule tries', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 6000)
