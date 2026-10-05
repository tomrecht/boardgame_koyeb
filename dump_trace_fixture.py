"""Step 6.4 of the JS port: the TRACE fixture -- the real proof.

Plays seeded games with the SERVED agent (real model.onnx, prefilter F=12) and
records, at every turn, the position and the pair it chose. The JS side replays
each recorded position through its own agent with real inference and must
choose the same pair.

Positions come from actual agent play rather than a random walk, so the
distribution is the one the port will really meet -- including the near-ties
that are the whole reason this has to be checked.

The scores are also recorded. The JS runtime matches Python to 4.5e-08 rather
than exactly, so a pair that ties in Python can be a 1e-8 gap in JS; the margin
between the chosen pair and the runner-up is what says whether a disagreement is
a real difference or a coin-flip that landed the other way.

    python dump_trace_fixture.py [games] [out.json]
"""
import json
import random
import sys

from agent_gnn import GNNAgent
from dump_engine_fixture import canon, jmove, state
from game import Board

MAX_TURNS = 120


def main(games=2, out='trace_fixture.json'):
    agent = GNNAgent('model.onnx', use_prefilter=True, prefilter_top_k=40,
                     first_move_prefilter=12)
    cases = []
    for g in range(games):
        random.seed(1000 + g)
        b = Board()
        for turn in range(MAX_TURNS):
            if b.check_game_over()[0]:
                break
            moves = b.get_valid_moves()
            if not moves:
                b.switch_turn()
                continue
            b.assign_piece_indices()          # as update_state always leaves it
            st = state(b)
            player = b.current_player
            ranked = agent.select_move_pair(moves, b, player, return_scores=True)
            pair = agent.select_move_pair(moves, b, player)
            top = [{'score': ('inf' if s == float('inf') else s),
                    'pair': [jmove(m) for m in p]} for s, p in ranked[:3]]
            cases.append({
                'game': g, 'turn': turn, 'state': st, 'player': player,
                'moves': [jmove(m) for m in sorted(moves, key=canon)],
                'chosen': [jmove(m) for m in pair],
                'top': top,
                # How far clear the winner was. A disagreement inside ~1e-5 is
                # the runtime's float noise, not a search difference.
                'margin': (None if len(ranked) < 2 or ranked[0][0] == float('inf')
                           else ranked[0][0] - ranked[1][0]),
            })
            for m in pair:
                if m != (0, 0, 0):
                    b.apply_move(m, switch_turn=False)
            b.switch_turn()

    with open(out, 'w') as f:
        json.dump({'cases': cases}, f)
    margins = sorted(c['margin'] for c in cases if c['margin'] is not None)
    print(f'wrote {out}: {len(cases)} turns over {games} games')
    if margins:
        print(f'winning margin: min {margins[0]:.3e}, '
              f'median {margins[len(margins) // 2]:.3f}, max {margins[-1]:.3f}')
        print(f'turns decided by less than 1e-5: {sum(1 for m in margins if m < 1e-5)}')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2,
         sys.argv[2] if len(sys.argv) > 2 else 'trace_fixture.json')
