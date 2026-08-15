"""Export the encoder's STATIC half for the JavaScript port.

Everything here is a pure function of `tile_neighbors.json` and never changes
during a game: the tile ordering, the tile-graph edge list, and the base tile
features. Exporting it instead of re-deriving it in JS removes a real hazard --
`build_tile_index` collects edges in a `set`, so the column ORDER of
tile_edge_index comes out of Python's set iteration. A JS reimplementation would
produce a different order, and although sum-aggregation is order-independent in
exact arithmetic, float addition is not associative, so the scores would differ
in the last bits and could flip an argmax in a near-tie. The Python order is
therefore the source of truth and JS just loads it.

    python export_encoder_static.py            # writes encoder_static.json

Re-run whenever tile_neighbors.json or encode_tile_features changes.
"""
import json

import numpy as np

import encoder as E


def main(out='encoder_static.json'):
    tile_index, tile_info, tile_edge_index = E.build_tile_index()
    base = E.encode_tile_features(tile_index, tile_info)

    # index -> (ring, sector), so JS can rebuild the same lookup
    coords = [None] * len(tile_index)
    for (r, s), idx in tile_index.items():
        coords[idx] = [r, s]
    assert all(c is not None for c in coords), 'tile_index is not contiguous'

    # per-tile metadata the dynamic pass needs (type and save number)
    types = [tile_info[(r, s)]['type'] for r, s in coords]
    numbers = [tile_info[(r, s)].get('number', 0) for r, s in coords]

    # Neighbour lists in tile_neighbors.json's own order, which the ENGINE port
    # needs. tile_edge_index is not a substitute: its column order comes out of
    # a Python set, and while that is fine for anything order-independent (the
    # GNN's sum-aggregation, a shortest-path length), it is NOT fine for
    # count_enemy_blots_on_shortest_path -- a plain FIFO BFS where the first
    # predecessor to reach a tile fixes its blot count, so neighbour order
    # changes the answer. Measured: 946 evaluations differed on exactly that
    # component until this was exported.
    raw = json.load(open('tile_neighbors.json'))
    key_of = {}
    for k in raw:
        r, sec = (int(x) for x in k.replace('ring', '').replace('sector', '').split('_'))
        key_of[(r, sec)] = k
    neighbors = []
    for r, s in coords:
        nb = raw[key_of[(r, s)]]['neighbors']
        neighbors.append([tile_index[(d['ring'], d['sector'])] for d in nb
                          if (d['ring'], d['sector']) in tile_index])

    payload = {
        'num_tiles': len(coords),
        'tile_neighbors': neighbors,
        'tile_coords': coords,
        'tile_types': types,
        'tile_numbers': numbers,
        'tile_edge_index': tile_edge_index.astype(int).tolist(),
        'base_tile_feats': np.asarray(base, dtype=np.float32).tolist(),
        'dims': {
            'TILE_FEAT_DIM': E.TILE_FEAT_DIM,
            'PIECE_FEAT_DIM': E.PIECE_FEAT_DIM,
            'GLOBAL_FEAT_DIM': E.GLOBAL_FEAT_DIM,
            'NUM_PIECES': E.NUM_PIECES,
            'MAX_DIST': E.MAX_DIST,
        },
    }
    with open(out, 'w') as f:
        json.dump(payload, f)
    print(f'wrote {out}: {len(coords)} tiles, '
          f'{tile_edge_index.shape[1]} directed edges, '
          f'{E.TILE_FEAT_DIM} tile features')

    # Self-verify: reload and compare against the in-memory arrays exactly.
    back = json.load(open(out))
    assert np.array_equal(np.array(back['tile_edge_index'], dtype=np.int64), tile_edge_index)
    assert np.array_equal(np.array(back['base_tile_feats'], dtype=np.float32), base)
    assert back['tile_neighbors'] == neighbors
    # The exported adjacency must agree with the engine's own, as a SET.
    from game import Board
    b = Board()
    for t in b.tiles:
        want = sorted(tile_index[(n2.ring, n2.pos)] for n2 in t.neighbors
                      if (n2.ring, n2.pos) in tile_index)
        assert sorted(neighbors[tile_index[(t.ring, t.pos)]]) == want, t
    print('self-check: edge index, base features and neighbour lists round-trip '
          'exactly, and the neighbour lists match game.py\'s Board')


if __name__ == '__main__':
    main()
