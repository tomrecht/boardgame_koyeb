"""offgoal_probe.py -- recheck a known near-tie on a new net (CLAUDE.md, "KNOWN,
LEFT FOR NOW: OFFGOALING IN A NEAR-CERTAIN LOSS").

Owner's game 48ff83c7 turn 46: black 3@G3 x@G3 5@G5 x@6.2, white's last piece a
blank on G3, black to play 6-2. Right: blank to G1, leave the 2 unused (exact
next-turn saves 1.14 vs 0.97 for stepping the 3 off). Prints the net's top
candidates in the app config (prefilter 12/40/5, hand rules, endgame lookahead).
Measured 2026-10-24: blend4way leaves the 2 (lead 0.035); fv2C_it55 and
distill_cand3 step the 3 off (lead 0.003).

    python3 offgoal_probe.py <net.pt> [...]
"""
import os, sys
os.environ.setdefault('GATE_PREFILTER', '12,40,5')
import torch
from game import Board
from network import model_from_state
from agent_gnn import GNNAgent, prefilter_kwargs, SCORE_SCALE

STATE = {'currentTurn': 'black', 'dice': [{'value': 6, 'used': False}, {'value': 2, 'used': False}], 'racks': {'whiteUnentered': [], 'whiteSaved': [{'color': 'white', 'number': 6}, {'color': 'white', 'number': 2}, {'color': 'white', 'number': 11}, {'color': 'white', 'number': 1}, {'color': 'white', 'number': 4}, {'color': 'white', 'number': 12}, {'color': 'white', 'number': 5}, {'color': 'white', 'number': 8}, {'color': 'white', 'number': 9}, {'color': 'white', 'number': 10}, {'color': 'white', 'number': 7}], 'blackUnentered': [], 'blackSaved': [{'color': 'black', 'number': 12}, {'color': 'black', 'number': 1}, {'color': 'black', 'number': 11}, {'color': 'black', 'number': 2}, {'color': 'black', 'number': 6}, {'color': 'black', 'number': 4}, {'color': 'black', 'number': 9}, {'color': 'black', 'number': 8}]}, 'boardPieces': [{'color': 'white', 'number': 13, 'tile': {'ring': 7, 'sector': 8}}, {'color': 'black', 'number': 7, 'tile': {'ring': 7, 'sector': 8}}, {'color': 'black', 'number': 5, 'tile': {'ring': 7, 'sector': 6}}, {'color': 'black', 'number': 10, 'tile': {'ring': 6, 'sector': 2}}, {'color': 'black', 'number': 3, 'tile': {'ring': 7, 'sector': 8}}], 'noSaveTurns': 0, 'drawCallable': False}
GOAL = {2: 4, 4: 2, 6: 5, 8: 3, 10: 6, 12: 1}
RIGHT = 'x>G1:6 pass'


def loc(t):
    return 'save' if t == 'save' else (f"G{GOAL[t[1]]}" if t[0] == 7 and t[1] in GOAL else f"{t[0]}.{t[1]}")


def desc(pair):
    out = []
    for mv in pair:
        if mv in ((0, 0, 0), (1, 1, 1)):
            out.append('pass'); continue
        (_, n), dst, d = mv
        out.append(f"{n if n <= 6 else 'x'}>{loc(dst)}:{d}")
    if len(out) == 1:
        out.append('pass')
    return ' '.join(out)


for path in sys.argv[1:]:
    m = model_from_state(torch.load(path, map_location='cpu'), strict=False)
    ag = GNNAgent(model=m, hand_rules=True, **prefilter_kwargs(os.environ['GATE_PREFILTER']))
    ag.endgame_lookahead = True
    b = Board(); b.update_state(STATE); b.firstMove = None
    r = sorted(ag.select_move_pair(list(b.get_valid_moves()), b, 'black', True), key=lambda x: -x[0])
    top = desc(r[0][1])
    print(f"== {os.path.basename(path)}: picks '{top}' ({'RIGHT' if top == RIGHT else 'wrong'}), "
          f"lead {(r[0][0] - r[1][0]) / SCORE_SCALE * 12:.3f} pts")
    for v, p in r[:5]:
        print(f"   {v / SCORE_SCALE * 12:+.3f}  {desc(p)}")
