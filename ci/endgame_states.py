"""Endgame positions for lookahead2_fixture.py, built without a browser or a net:
dump_engine_fixture.endgame_board (white 1-4 pieces left, black partly on the
field), either side to move. Writes the eg_selfplay.js row shape that
lookahead2_fixture.py reads.
    python3 ci/endgame_states.py out.jsonl [n=400]"""
import json, random, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dump_engine_fixture import endgame_board, state

def main(out, n):
    rng = random.Random(4242)
    rows = []
    while len(rows) < n:
        b = endgame_board(rng)
        if rng.random() < 0.5:
            b.switch_turn()
        if b.check_game_over()[0] or not b.get_valid_moves():
            continue
        b.assign_piece_indices()
        rows.append({'state': state(b), 'game': f'ci{len(rows):06d}', 'k': 0, 'turn': len(rows), 'same': True})
    with open(out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(f'wrote {out}: {len(rows)} endgame positions')

if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 400)
