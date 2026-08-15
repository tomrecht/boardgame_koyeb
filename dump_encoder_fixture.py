"""Step 2 of the JS port (see PORTING.md): the test fixture.

Writes N seeded positions as {snapshot, expected}, where `snapshot` is the
serialisable board state the JS encoder will take as its input, and `expected`
is what encoder.py produces for it. Every later step of the port is then
asserted array-equal against this file rather than eyeballed.

The snapshot deliberately carries only what the encoder reads -- pieces, racks,
dice, whose turn it is -- NOT a serialised Board. game.js already has its own
board model, so the JS side builds this shape from that; porting game.py's Board
as well would mean maintaining two of them.

    python dump_encoder_fixture.py [n] [out.json]
"""
import json
import random
import sys

import numpy as np

import encoder as E
from game import Board


def snapshot(b):
    """Exactly the state the encoder reads, in a JS-friendly shape."""
    def where(p):
        if p.tile is not None:
            return {'kind': 'tile', 'ring': p.tile.ring, 'pos': p.tile.pos}
        if p.rack is b.white_unentered or p.rack is b.black_unentered:
            return {'kind': 'unentered'}
        if p.rack is b.white_saved or p.rack is b.black_saved:
            return {'kind': 'saved'}
        return {'kind': 'none'}

    # Rack ORDER matters (entry order), so record it explicitly.
    rack_order = {
        'white_unentered': [p.number for p in b.white_unentered],
        'black_unentered': [p.number for p in b.black_unentered],
        'white_saved': [p.number for p in b.white_saved],
        'black_saved': [p.number for p in b.black_saved],
    }
    return {
        'current_player': b.current_player,
        'dice': [{'value': d.number, 'used': bool(d.used)} for d in b.dice],
        'game_stages': dict(b.game_stages),
        'pieces': [{'player': p.player, 'number': p.number, **where(p)} for p in b.pieces],
        'racks': rack_order,
    }


def expected(enc, b):
    out = enc.encode(b, b.current_player)
    # all_pieces ordering decides the row order of piece_feats and the edge
    # indices, so the JS port has to reproduce it -- record it.
    _, all_pieces = E.encode_piece_features(b, enc.tile_index, b.current_player, None)
    return {
        'tile_feats': np.asarray(out['tile_feats'], dtype=np.float32).tolist(),
        'piece_feats': np.asarray(out['piece_feats'], dtype=np.float32).tolist(),
        'global_feats': np.asarray(out['global_feats'], dtype=np.float32).ravel().tolist(),
        'piece_to_tile': np.asarray(out['piece_to_tile']).astype(int).tolist(),
        'tile_to_piece': np.asarray(out['tile_to_piece']).astype(int).tolist(),
        'piece_order': [[p.player, p.number] for p in all_pieces],
    }


def main(n=40, out='encoder_fixture.json'):
    enc = E.BoardEncoder()
    cases = []
    seed = 0
    while len(cases) < n:
        random.seed(seed)
        seed += 1
        b = Board()
        # Walk a few turns so the sample spans opening/midgame shapes rather
        # than 40 copies of the start position.
        for _ in range(random.randint(0, 14)):
            if b.check_game_over()[0]:
                break
            for _ in range(2):
                moves = [m for m in b.get_valid_moves() if isinstance(m[0], tuple)]
                if not moves:
                    break
                b.apply_move(random.choice(moves), switch_turn=False)
            b.switch_turn()
        if b.check_game_over()[0]:
            continue
        cases.append({'seed': seed - 1, 'snapshot': snapshot(b), 'expected': expected(enc, b)})

    with open(out, 'w') as f:
        json.dump({'cases': cases}, f)
    dims = (len(cases[0]['expected']['tile_feats']),
            len(cases[0]['expected']['piece_feats']),
            len(cases[0]['expected']['global_feats']))
    print(f'wrote {out}: {len(cases)} positions, '
          f'tile_feats {dims[0]}x{len(cases[0]["expected"]["tile_feats"][0])}, '
          f'piece_feats {dims[1]}x{len(cases[0]["expected"]["piece_feats"][0])}, '
          f'global {dims[2]}')

    # Self-check: re-encode one case from its own snapshot is out of scope here
    # (that is what the JS port will do); instead verify the file round-trips
    # and that nothing is NaN, which would silently pass an allclose later.
    back = json.load(open(out))
    for c in back['cases']:
        for k in ('tile_feats', 'piece_feats', 'global_feats'):
            a = np.asarray(c['expected'][k], dtype=np.float32)
            assert np.isfinite(a).all(), f'non-finite value in {k} (seed {c["seed"]})'
    print('self-check: round-trips, all features finite')


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    out = sys.argv[2] if len(sys.argv) > 2 else 'encoder_fixture.json'
    main(n, out)
