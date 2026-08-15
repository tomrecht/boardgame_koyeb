"""Step 6.2 of the JS port: the HEURISTIC fixture.

Records agent.py's evaluate() for each position AND for the position after each
legal move -- the latter is what actually matters, because that is what the
prefilter ranks. With FIRST_MOVE_PREFILTER=12 the served agent keeps only twelve
first moves, so a scoring difference changes which candidates the value net
ever sees.

Per-component totals are recorded alongside the score: a single number tells you
THAT the port disagrees, the components tell you WHERE.

    python dump_heuristic_fixture.py [n] [out.json]
"""
import json
import random
import sys

from agent import Agent, get_weights
from dump_engine_fixture import canon, endgame_board, jmove, state
from game import Board

# Exactly the agent app.py serves.
HEUR = Agent(weights=get_weights(weights_file='best_weights.json'))

# Components are floats summed in a fixed order; keep the raw values.
SKIP = {'_total_score', '_player', '_goal_pieces'}


def evaluate(b, player):
    score, comps = HEUR.evaluate(b, player)
    out = {'score': score}
    if comps:
        out['player'] = {k: v for k, v in comps['player'].items() if k not in SKIP}
        out['opponent'] = {k: v for k, v in comps['opponent'].items() if k not in SKIP}
    return out


def case(b):
    b.assign_piece_indices()          # canonical order, as update_state gives it
    moves = sorted(b.get_valid_moves(), key=canon)
    st = state(b)
    player = b.current_player
    rows = []
    for m in moves:
        base = len(b.moves)
        b.apply_move(m, switch_turn=False)
        rows.append({'move': jmove(m), 'eval': evaluate(b, player)})
        while len(b.moves) > base:
            b.undo_last_move()
        if m == (1, 1, 1):
            b.draw_called = False
    return {'state': st, 'player': player,
            'eval': evaluate(b, player),
            # Both players, since evaluate() is asymmetric and the search scores
            # from one fixed player's point of view throughout.
            'eval_opponent': evaluate(b, 'white' if player == 'black' else 'black'),
            'after': rows}


def main(n=30, out='heuristic_fixture.json'):
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
        c = case(b)
        if not c['after']:
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
        c = case(b)
        if not c['after']:
            continue
        c['seed'] = -1 - built
        c['kind'] = 'endgame'
        cases.append(c)
        built += 1

    with open(out, 'w') as f:
        json.dump({'cases': cases}, f)
    eva = sum(1 + len(c['after']) for c in cases) + len(cases)
    print(f'wrote {out}: {len(cases)} positions, {eva} evaluations')
    stages = {}
    for c in cases:
        s = c['state']['stages'][c['player']]
        stages[s] = stages.get(s, 0) + 1
    print('stage coverage:', stages)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30,
         sys.argv[2] if len(sys.argv) > 2 else 'heuristic_fixture.json')
