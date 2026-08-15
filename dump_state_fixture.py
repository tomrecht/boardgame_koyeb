"""Step 7 of the JS port (see PORTING.md): the STATE-REBUILD fixture.

The trace-diff of step 6.4 proved the AGENT, but it fed that agent engine states
dumped by Python. The one piece that only exists in the app is turning game.js's
own board model into the engine's -- `LocalAgent.engineState`, a mirror of
`Board.update_state`. If that rebuild differs, the two agents are answering
different questions and every other comparison is meaningless.

This takes the states game.js actually POSTED to /select_moves (collected from a
real browser session) and records, for each, what update_state makes of it:

    fingerprint -- the position, comparable with engine.js's fingerprint()
    moves       -- get_valid_moves(), canonically sorted (Python's own order is
                   not reproducible; the SET is the contract)

It also emits VARIANTS carrying `reachableBySum`, which is how update_state
learns that a piece already moved this turn (it sets board.firstMove from the
mere PRESENCE of that key). Self-play never produces it -- game.js only attaches
it to a piece whose reachable tiles were computed for a human -- so without
these the whole firstMove branch, which changes which moves are legal, would go
untested. That is not hypothetical: it is exactly the branch where
Engine.fromState was found to leave firstMove.piece null.

Collect the input states with a browser session that records every payload
passed to LocalAgent.selectMoves (see PORTING.md §7); then:

    PYTHONHASHSEED=0 python dump_state_fixture.py states.json state_expected.json

Check it with state_test.html.
"""
import json
import sys

from game import Board
from dump_engine_fixture import fingerprint, jmove


def variants(st):
    """The state as posted, plus firstMove-carrying versions of it."""
    yield 'as-posted', st
    bp = st.get('boardPieces') or []
    if not bp:
        return
    # A piece of the CURRENT player that already moved: the mid-turn case.
    mine = [i for i, p in enumerate(bp) if p['color'] == st['currentTurn']]
    # A piece of the OTHER player carrying a stale key: what a human's last move
    # leaves behind when the agent is asked on the following turn.
    theirs = [i for i, p in enumerate(bp) if p['color'] != st['currentTurn']]
    for label, idxs in (('first-mover-mine', mine), ('stale-theirs', theirs)):
        if not idxs:
            continue
        v = json.loads(json.dumps(st))
        v['boardPieces'][idxs[0]]['reachableBySum'] = []
        yield label, v
    if len(mine) > 1:
        v = json.loads(json.dumps(st))
        # Two carriers: update_state keeps the LAST one it sees, and so must JS.
        v['boardPieces'][mine[0]]['reachableBySum'] = []
        v['boardPieces'][mine[-1]]['reachableBySum'] = []
        yield 'two-carriers', v


def main():
    states = json.load(open(sys.argv[1]))
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'state_expected.json'

    cases = []
    for i, st in enumerate(states):
        for label, v in variants(st):
            b = Board()
            b.update_state(v)
            moves = b.get_valid_moves()
            cases.append({
                'i': i,
                'label': label,
                'state': v,
                'fingerprint': fingerprint(b),
                'moves': sorted([jmove(m) for m in moves],
                                key=lambda m: json.dumps(m, sort_keys=True)),
            })

    json.dump(cases, open(out_path, 'w'))
    labels = {}
    for c in cases:
        labels[c['label']] = labels.get(c['label'], 0) + 1
    print('wrote', out_path, len(cases), 'cases:', labels)
    print('with a firstMove set:', sum(1 for c in cases if c['fingerprint']['first_move']))


if __name__ == '__main__':
    main()
