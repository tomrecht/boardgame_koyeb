"""Fixture for features_v2.js / encoder.js (the JS twin of features_v2.py).

Each case: a JS `snapshot` (dump_encoder_fixture.snapshot + the no-save counter)
and what encoder.py produces for it under v1, A, AB, and AB in AFTERSTATE mode.

Two kinds of position:
  pool   owner's positions (start_pool.jsonl), rebuilt with update_state --
         varied, real, with walls, captures and endgames;
  live   boards from seeded random play, snapshotted at the END of a turn,
         before the switch -- so a last numbered piece on its goal is still
         numbered, as the search sees it. The AFTERSTATE expectation is made the
         way training makes it (update_state(serialize_board) then encode), which
         applies the last-piece rule; the JS side must reproduce that itself.

    python3 dump_v2_fixture.py [n_pool] [n_live] [out.json]
"""
import json, random, sys

import numpy as np

import encoder as E
from game import Board
from game_worker import serialize_board
from dump_encoder_fixture import snapshot


def snap(b):
    s = snapshot(b)
    s['no_save_turns'] = b.no_save_turns
    return s


def enc_arrays(enc, b, player):
    out = enc.encode(b, player)
    _, all_pieces = E.encode_piece_features(b, enc.tile_index, player, None)
    return {
        'tile_feats': np.asarray(out['tile_feats'], dtype=np.float32).tolist(),
        'piece_feats': np.asarray(out['piece_feats'], dtype=np.float32).tolist(),
        'global_feats': np.asarray(out['global_feats'], dtype=np.float32).ravel().tolist(),
        'piece_order': [[p.player, p.number] for p in all_pieces],
    }


def main(n_pool=300, n_live=300, out='features_v2_fixture.json'):
    encs = {k: E.BoardEncoder(features=k) for k in ('v1', 'A', 'AB')}
    enc_as = E.BoardEncoder(features='AB', afterstate=True)
    cases = []
    b = Board()
    pool = [json.loads(l) for l in open('start_pool.jsonl')]
    rng = random.Random(7)
    for rec in rng.sample(pool, min(n_pool, len(pool))):
        st = rec.get('raw_state', rec.get('state', rec))
        b.update_state(st)
        p = b.current_player
        cases.append({'kind': 'pool', 'player': p, 'snapshot': snap(b),
                      'expected': {k: enc_arrays(e, b, p) for k, e in encs.items()}})
    # live end-of-turn boards from random play (captures, walls, endgames)
    b2 = Board()
    seed = 0
    while sum(c['kind'] == 'live' for c in cases) < n_live:
        random.seed(seed); seed += 1
        b = Board()
        stop = random.randint(5, 140)
        for t in range(stop):
            if b.check_game_over()[0]:
                break
            for _ in range(2):
                moves = [m for m in b.get_valid_moves() if isinstance(m[0], tuple)]
                if not moves:
                    break
                b.apply_move(random.choice(moves), switch_turn=False)
            if t < stop - 1:
                b.switch_turn()
        if b.check_game_over()[0]:
            continue
        p = b.current_player
        exp = {k: enc_arrays(e, b, p) for k, e in encs.items()}
        b2.update_state(serialize_board(b))
        exp['AB_after'] = enc_arrays(enc_as, b2, p)
        cases.append({'kind': 'live', 'player': p, 'snapshot': snap(b), 'expected': exp})
    # LAST-PIECE cases, played into: a side holds numbered N on goal N plus one
    # blank on goal g; it banks the blank mid-turn, leaving N as its lone piece
    # on its goal -- numbered on the live board, blanked in the afterstate.
    n_last = 0
    for k, rec in enumerate(rng.sample(pool, 400)):
        if n_last >= 80:
            break
        st = json.loads(json.dumps(rec.get('raw_state', rec.get('state', rec))))
        me = st['currentTurn']
        N = 1 + k % 6
        g = 1 + (k // 6) % 6
        side = 'white' if me == 'white' else 'black'
        cap = side[0].upper() + side[1:]
        nums = [n for n in range(1, 13) if n not in (N, 12)]
        st['racks'][side + 'Unentered'] = []
        st['racks'][side + 'Saved'] = [{'color': side, 'number': n} for n in nums]
        st['boardPieces'] = [q for q in st['boardPieces'] if q['color'] != side]
        b.update_state(st)
        goal = {t.number: t for t in b.tiles if t.type == 'save'}
        st['boardPieces'] += [{'color': side, 'number': N, 'tile': {'ring': goal[N].ring, 'sector': goal[N].pos}},
                              {'color': side, 'number': 12, 'tile': {'ring': goal[g].ring, 'sector': goal[g].pos}}]
        st['dice'] = [{'value': g, 'used': False}, {'value': (g % 6) + 1, 'used': False}]
        b.update_state(st)
        moves = b.get_valid_moves()
        save = next((m for m in moves if m[0] == (side, 12) and m[1] == 'save'), None)
        if save is None:
            continue
        b.apply_move(save, switch_turn=False)
        left = [q for q in b.pieces if q.player == side and q.rack is not b.get_save_rack(side)]
        if not (len(left) == 1 and left[0].number == N):
            continue
        p = side
        exp = {kk: enc_arrays(e, b, p) for kk, e in encs.items()}
        b2.update_state(serialize_board(b))
        exp['AB_after'] = enc_arrays(enc_as, b2, p)
        cases.append({'kind': 'last', 'player': p, 'snapshot': snap(b), 'expected': exp})
        n_last += 1
    # pool cases get an afterstate expectation too (their board is already rebuilt)
    for c in cases:
        if c['kind'] == 'pool':
            b.update_state(_raw_from_snapshot(c['snapshot']))
            c['expected']['AB_after'] = enc_arrays(enc_as, b, c['player'])
    json.dump({'cases': cases}, open(out, 'w'))
    blanked = sum(1 for c in cases if c['kind'] in ('live', 'last') and
                  c['expected']['AB_after']['piece_order'] != c['expected']['AB']['piece_order'])
    print(f'wrote {out}: {len(cases)} cases ({n_pool} pool, {n_live} live); '
          f'last-piece rule changes the afterstate in {blanked} live/last cases '
          f'({sum(c["kind"] == "last" for c in cases)} last-piece cases)')


def _raw_from_snapshot(s):
    """snapshot -> the frontend state format update_state reads."""
    pieces = s['pieces']
    def rack(kind, side):
        return [{'color': side, 'number': n} for n in s['racks'][f'{side}_{kind}']]
    return {
        'currentTurn': s['current_player'],
        'dice': [{'value': d['value'], 'used': d['used']} for d in s['dice']],
        'racks': {'whiteUnentered': rack('unentered', 'white'), 'whiteSaved': rack('saved', 'white'),
                  'blackUnentered': rack('unentered', 'black'), 'blackSaved': rack('saved', 'black')},
        'boardPieces': [{'color': p['player'], 'number': p['number'],
                         'tile': {'ring': p['ring'], 'sector': p['pos']}}
                        for p in pieces if p['kind'] == 'tile'],
        'noSaveTurns': s['no_save_turns'],
    }


if __name__ == '__main__':
    a = sys.argv[1:]
    main(int(a[0]) if a else 300, int(a[1]) if len(a) > 1 else 300,
         a[2] if len(a) > 2 else 'features_v2_fixture.json')
