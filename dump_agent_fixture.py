"""Step 6.3 of the JS port: the SEARCH fixture.

The value net is deliberately replaced by a STUB -- a hash of the resulting
position, mapped into [-1, 1]. Both languages compute it from their own state
with integer arithmetic, so it is bit-identical by construction, and what this
fixture then measures is the search itself: enumeration, the two-stage
prefilter, the top-K cull, the save exemptions, the terminal short-circuits,
draw handling and every tie-break.

That separation is the point. infer.js matches Python to 4.5e-08, not exactly,
so an end-to-end comparison would mix a real search bug with float noise and
give no clean pass/fail. Inference is verified separately (step 5) and put back
in step 6.4.

The stub also exercises the tie-breaks HARDER than the real net would: pairs
that transpose to the same position get the same hash, which is exactly where
enumeration order used to decide the answer.

    python dump_agent_fixture.py [n] [out.json]
"""
import json
import random
import sys

import numpy as np

from agent_gnn import GNNAgent
from dump_engine_fixture import canon, endgame_board, jmove, state
from game import Board


def position_key(board):
    """Must match agent.js's positionKey() character for character."""
    saved = (board.white_saved, board.black_saved)
    locs = []
    for p in board.pieces:
        if p.tile is not None:
            locs.append(str(p.tile.index))
        elif p.rack is saved[0] or p.rack is saved[1]:
            locs.append('-2')
        else:
            locs.append('-1')
    return ','.join(locs) + '|' + ('1' if board.dice[0].used else '0') \
                          + ('1' if board.dice[1].used else '0')


def stub_score(key):
    """FNV-1a over the key, then into [-1, 1). Integer arithmetic throughout so
    JS reproduces it exactly (Math.imul for the 32-bit multiply)."""
    h = 0x811c9dc5
    for ch in key.encode('ascii'):
        h ^= ch
        h = (h * 0x01000193) & 0xFFFFFFFF
    return (h / 0x100000000) * 2.0 - 1.0


class StubEncoder:
    """encode() is called exactly where the board sits in the candidate's
    resulting position, so it is the cheapest place to capture that position."""
    def encode(self, board, player):
        return position_key(board)


class StubModel:
    def __call__(self, keys):
        return np.array([stub_score(k) for k in keys], dtype=np.float64)


def build_agent():
    # The served configuration.
    a = GNNAgent('model.onnx', use_prefilter=True, prefilter_top_k=40,
                 first_move_prefilter=12)
    a.encoder = StubEncoder()
    a.model = StubModel()
    return a


def case(agent, b):
    b.assign_piece_indices()
    moves = b.get_valid_moves()
    if not moves:
        return None
    st = state(b)
    player = b.current_player
    pair = agent.select_move_pair(moves, b, player)
    ranked = agent.select_move_pair(moves, b, player, return_scores=True)
    return {
        'state': st,
        'player': player,
        'moves': [jmove(m) for m in sorted(moves, key=canon)],
        'chosen': [jmove(m) for m in pair],
        # The top of the ranking too: if the choice differs, this says whether
        # the port scored differently or merely broke a tie differently.
        # Python's json.dump emits a BARE `Infinity` for float('inf'), which is
        # not valid JSON and JSON.parse rejects outright (the winning-move
        # short-circuit returns exactly that). Send it as a string.
        'top': [{'score': ('inf' if s == float('inf') else s),
                 'pair': [jmove(m) for m in p]} for s, p in ranked[:5]],
        'n_scored': len(ranked),
    }


def main(n=40, out='agent_fixture.json'):
    agent = build_agent()
    cases = []
    seed = 0
    while len(cases) < n:
        random.seed(seed)
        seed += 1
        b = Board()
        for _ in range(random.randint(2, 70)):
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
        c = case(agent, b)
        if not c:
            continue
        c['seed'] = seed - 1
        c['kind'] = 'walk'
        cases.append(c)

    rng = random.Random(12345)
    built = 0
    while built < max(4, n // 4):
        b = endgame_board(rng)
        if b.check_game_over()[0]:
            continue
        c = case(agent, b)
        if not c:
            continue
        c['seed'] = -1 - built
        c['kind'] = 'endgame'
        cases.append(c)
        built += 1

    with open(out, 'w') as f:
        json.dump({'cases': cases}, f)
    print(f'wrote {out}: {len(cases)} positions')
    kinds = {}
    for c in cases:
        for m in c['chosen']:
            k = ('pass' if m['piece'] is None and m['lone'] == 0 else
                 'draw' if m['piece'] is None else
                 'save' if m['dest'] == 'save' else
                 'block-save' if m['dest'] == 0 else 'move')
            kinds[k] = kinds.get(k, 0) + 1
    print('chosen half-moves:', kinds)
    print('candidates scored: min %d, median %d, max %d'
          % (min(c['n_scored'] for c in cases),
             sorted(c['n_scored'] for c in cases)[len(cases) // 2],
             max(c['n_scored'] for c in cases)))


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40,
         sys.argv[2] if len(sys.argv) > 2 else 'agent_fixture.json')
