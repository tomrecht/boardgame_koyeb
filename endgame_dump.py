"""endgame_dump.py -- small-ending turns from owner's logged games, for the JS
solver (endgame_lab.js). One JSON line per turn where both sides have <= LIMIT
unsaved pieces at the turn start: the engine state (dump_engine_fixture.state,
the format Engine.fromState takes), who moved, whether the computer moved, and
the moves played ("m", recorder format) plus the agent's own pair ("a").

    GOAL_LAYOUT=sum7 python3 endgame_dump.py games.jsonl out.jsonl [LIMIT=2]

Run once per layout: only games recorded on GOAL_LAYOUT are dumped (today's
games have no 'layout' key).
"""
import json
import sys

import goal_layout
import review_games as RG
from dump_engine_fixture import state as engine_state

src, out = sys.argv[1], sys.argv[2]
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 2
layout = goal_layout.NAME if hasattr(goal_layout, 'NAME') else None
want = layout or None

n_games = n = 0
with open(out, 'w') as f:
    for line in open(src):
        g = json.loads(line)
        if (g.get('layout') or None) != want:
            continue
        n_games += 1
        un = {'w': 12, 'b': 12}
        cand = []
        for t, turn in enumerate(g['turns']):
            if un['w'] <= limit and un['b'] <= limit:
                cand.append(t)
            for m in turn['m']:
                if m.endswith('>s'):
                    un[turn['p']] -= 1
                elif m.endswith('>b'):
                    un['b' if turn['p'] == 'w' else 'w'] -= 1
        for t in cand:
            b = RG._state_at(g, t)
            if b is None:
                break
            turn = g['turns'][t]
            side = 'white' if turn['p'] == 'w' else 'black'
            ai = g['whiteIsAI'] if side == 'white' else g['blackIsAI']
            f.write(json.dumps({'game': g['id'], 't': t, 'layout': g.get('layout'),
                                'model': g.get('model'), 'side': side, 'ai': bool(ai),
                                'm': turn['m'], 'a': turn.get('a'),
                                'notation': RG.notation(b), 'state': engine_state(b)}) + '\n')
            n += 1
print('games', n_games, 'turns', n, file=sys.stderr)
