"""Fixture for the TWO-SIDED endgame lookahead (owner, 2026-11-09): positions
where either side has <= 2 unsaved pieces, from the endgame lab's self-play
(lab/eg_selfplay.js output, engine states, today's layout only), searched by
the Python agent with the STUBBED net of dump_agent_fixture.py, so a JS mismatch
is a search difference, not float noise.

    python3 lookahead2_fixture.py sp.jsonl [n=150]   (-> lookahead2_fixture.json)
    node agent_test.js lookahead2_fixture.json
"""
import json
import sys

import dump_agent_fixture as D
from game import Board

STATIC = json.load(open('encoder_static.json'))


def board_from_state(st):
    coords = STATIC['tile_coords']
    rk = lambda k: [{'number': n} for n in st['racks'][k]]
    gs = {
        'currentTurn': st['current_player'],
        'dice': [{'value': v, 'used': u} for v, u in st['dice']],
        'racks': {'whiteUnentered': rk('white_unentered'), 'whiteSaved': rk('white_saved'),
                  'blackUnentered': rk('black_unentered'), 'blackSaved': rk('black_saved')},
        'boardPieces': [{'color': pl, 'number': n, 'tile': {'ring': coords[int(i)][0], 'sector': coords[int(i)][1]}}
                        for i, lst in st['occupancy'].items() for pl, n in lst],
        'noSaveTurns': st.get('no_save_turns', 0),
        'drawCallable': st.get('draw_callable', False),
    }
    b = Board()
    b.update_state(gs)
    return b


def main(src, n=150, out='lookahead2_fixture.json'):
    agent = D.build_agent()
    rows = [json.loads(l) for l in open(src)]
    rows = [r for r in rows if not r.get('layout')]
    # every row where the two versions disagreed (the new branch decides), then the rest
    rows.sort(key=lambda r: r['same'])
    cases = []
    for r in rows:
        if len(cases) >= n:
            break
        b = board_from_state(r['state'])
        c = D.case(agent, b)
        if c:
            c['seed'] = f"{r['game'][:8]}:{r['k']}:{r['turn']}"
            c['kind'] = 'lookahead2'
            cases.append(c)
    json.dump({'cases': cases}, open(out, 'w'))
    print(f'wrote {out}: {len(cases)} positions')


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 150)
