"""Fixture for agent.js's close-call search (selectWithCloseCall) against its
Python twin, close_call.py, with dump_agent_fixture's STUBBED net -- a hash of
the position, identical in both languages -- so a mismatch is a search
difference, not float noise. T is set wide so the stub's spread triggers often;
MAX_REPLIES (count cap, the JS budget's device-independent twin) is varied.

    PYTHONHASHSEED=0 python3 close_call_fixture.py [n=30]   (-> close_call_fixture.json)
    node close_call_test.js
"""
import json, random, sys

import close_call as CC
import dump_agent_fixture as D
from dump_engine_fixture import canon, jmove, state
from game import Board

T, K = 60.0, 3
CAPS = [0, 9, 3]          # 0 = full 21 rolls; 9 = 3 rounds of 3; 3 = 1 round


def build(cap):
    C = CC.make_agent_class()
    C._afterstate_of = staticmethod(lambda board, pair: None)   # labels only; not needed here
    a = C('model.onnx', use_prefilter=True, prefilter_top_k=40, first_move_prefilter=12, T=T, K=K)
    a.encoder = D.StubEncoder()
    a.model = D.StubModel()
    a.endgame_lookahead = False     # close_call.py's agents run without it
    a.max_replies = cap
    a.label_log = []
    return a


def main(n=30, out='close_call_fixture.json'):
    agents = {cap: build(cap) for cap in CAPS}
    cases, seed, trig = [], 0, 0
    while len(cases) < n:
        random.seed(seed)
        seed += 1
        b = Board()
        for _ in range(random.randint(4, 60)):
            if b.check_game_over()[0]:
                break
            for _ in range(2):
                ms = sorted((m for m in b.get_valid_moves() if isinstance(m[0], tuple)), key=canon)
                if not ms:
                    break
                saves = [m for m in ms if m[1] == 'save']
                pool = saves if (saves and random.random() < 0.75) else ms
                b.apply_move(random.choice(pool), switch_turn=False)
            b.switch_turn()
        if b.check_game_over()[0]:
            continue
        b.assign_piece_indices()
        moves = b.get_valid_moves()
        if not moves:
            continue
        st = state(b)
        st['half_turns'] = int(b._half_turns_since_round)
        st['last_total_saved'] = int(b._last_total_saved)
        player = b.current_player
        c = {'seed': seed - 1, 'state': st, 'player': player, 'T': T, 'K': K, 'runs': []}
        for cap, a in agents.items():
            before = dict(a.stats)
            a.label_log.clear()
            pair = a.select_move_pair(moves, b, player)
            d = {k: a.stats[k] - before[k] for k in ('triggered', 'changed')}
            deep = [x['label'] * 1000.0 for x in a.label_log[0]['cands']] if a.label_log else None
            c['runs'].append({'cap': cap, 'chosen': [jmove(m) for m in pair],
                              'triggered': d['triggered'], 'deep': deep})
        trig += c['runs'][0]['triggered']
        cases.append(c)
    json.dump({'cases': cases}, open(out, 'w'))
    print(f'wrote {out}: {len(cases)} positions, close-call triggered on {trig} (full search)')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
