"""Fixture for agent.js's endgame lookahead: owner's trigger positions (computer
to move, owner at <= 2 unsaved pieces, owner_lookahead.jsonl) with the STUBBED
net of dump_agent_fixture.py, so a mismatch is a search difference, not float
noise. Every case goes through selectWithLookahead.

    python3 lookahead_fixture.py      (-> lookahead_fixture.json)
    node agent_test.js lookahead_fixture.json
"""
import json

import dump_agent_fixture as D
import owner_lookahead as O


def main(out='lookahead_fixture.json'):
    agent = D.build_agent()
    recs = O._records()
    cases = []
    for line in open(O.OUT):
        g = json.loads(line)
        for r in g['rows']:
            b, _ = O.state_at(recs[g['game']], r['turn'])
            b.get_valid_moves()
            c = D.case(agent, b)
            if c:
                c['seed'] = f"{g['game']}:{r['turn']}"
                c['kind'] = 'lookahead'
                cases.append(c)
    json.dump({'cases': cases}, open(out, 'w'))
    print(f'wrote {out}: {len(cases)} positions')


if __name__ == '__main__':
    main()
