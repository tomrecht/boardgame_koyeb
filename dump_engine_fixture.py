"""Step 6.1 of the JS port (see PORTING.md): the ENGINE fixture.

The search runs on get_valid_moves / apply_move / undo_last_move, so those have
to be ported before any of agent_gnn can be. This writes, for N seeded
positions:

    state     -- everything needed to rebuild the position (a superset of the
                 encoder fixture's snapshot: it adds firstMove, the game stages,
                 the draw flags and the rack ORDER, all of which change which
                 moves are legal)
    moves     -- get_valid_moves(), CANONICALLY SORTED. Python's own order is
                 unstable run to run (it walks a set of Tile objects, whose
                 hashes are ids), so the set is the contract, not the order.
    after     -- for each move: the position fingerprint after apply_move, and
                 the fingerprint after undo_last_move, which must equal the
                 starting one. Undo is where the subtle bugs live -- block-save
                 clears both dice and hands the turn to the mover's OPPONENT,
                 and a rack piece must go back to its own slot.

    python dump_engine_fixture.py [n] [out.json]
"""
import json
import random
import sys

from game import Board


def _tile_index(board, tile):
    return -1 if tile is None else tile.index


def fingerprint(b):
    """Everything a move can change, in a form two engines can compare."""
    saved = (b.white_saved, b.black_saved)
    pieces = []
    for p in b.pieces:
        where = ('tile', _tile_index(b, p.tile)) if p.tile is not None else (
            ('saved', -1) if (p.rack is saved[0] or p.rack is saved[1]) else
            ('rack', -1) if p.rack is not None else ('none', -1))
        pieces.append([p.player, p.number, where[0], where[1]])
    # Occupancy as a MULTISET, deliberately. Within-tile order is not semantic:
    # every reader is either `pieces[0].player` on a field tile (which only ever
    # holds one player's pieces, since landing on a lone enemy captures it) or
    # `pieces.pop()` on a tile holding exactly one enemy. It is also not
    # preserved by the engine itself -- undo APPENDS the piece to origin_tile
    # rather than restoring its index, which reordered 125 of 739 apply/undo
    # pairs in a first pass. So the port must match the set, not the order.
    occ = {str(t.index): sorted([p.player, p.number] for p in t.pieces)
           for t in b.tiles if t.pieces}
    return {
        'pieces': sorted(pieces, key=lambda r: (r[0], r[1])),
        'occupancy': occ,
        'racks': {
            'white_unentered': [p.number for p in b.white_unentered],
            'black_unentered': [p.number for p in b.black_unentered],
            'white_saved': sorted(p.number for p in b.white_saved),
            'black_saved': sorted(p.number for p in b.black_saved),
        },
        'dice': [[d.number, bool(d.used)] for d in b.dice],
        'current_player': b.current_player,
        'stages': dict(b.game_stages),
        'first_move': None if not b.firstMove else [
            b.firstMove['piece'].player, b.firstMove['piece'].number,
            _tile_index(b, b.firstMove['origin_tile'])],
        'draw_called': bool(b.draw_called),
        'game_over': list(b.check_game_over()),
    }


def state(b):
    """The rebuildable position. fingerprint() is a superset already, so the
    state IS the fingerprint plus the draw availability (which get_valid_moves
    reads but no move changes)."""
    s = fingerprint(b)
    s['draw_callable'] = bool(b.draw_callable)
    s['no_save_turns'] = int(b.no_save_turns)
    return s


def canon(move):
    """Total order on moves, matching agent_gnn._move_sort_key's intent. Python's
    get_valid_moves order is not reproducible, so only the SET is compared."""
    piece_id, destination, roll = move
    if not isinstance(piece_id, tuple):
        return (0, '', int(piece_id), 0, int(roll))
    player, number = piece_id
    if destination == 'save':
        return (1, player, int(number), 0, int(roll))
    if destination == 0:
        return (2, player, int(number), 0, int(roll))
    ring, pos = destination
    return (3, player, int(number), int(ring) * 1000 + int(pos), int(roll))


def jmove(move):
    piece_id, destination, roll = move
    pid = None if not isinstance(piece_id, tuple) else [piece_id[0], int(piece_id[1])]
    if destination == 'save':
        dest = 'save'
    elif isinstance(destination, tuple):
        dest = [int(destination[0]), int(destination[1])]
    else:
        dest = int(destination)
    return {'piece': pid, 'lone': None if pid else int(piece_id),
            'dest': dest, 'roll': int(roll)}


def case(b):
    # Canonical piece order, as the SERVED path always has it: every request
    # goes through update_state, which calls assign_piece_indices (white first,
    # then by number). A Board built here and walked keeps initialize_pieces'
    # SHUFFLE instead, and that order is visible in the move list -- the dedupe
    # of interchangeable blanks on one tile keeps whichever comes first, so a
    # shuffled board attributes the same destinations to a different blank.
    b.assign_piece_indices()
    start = state(b)
    moves = sorted(b.get_valid_moves(), key=canon)
    # get_valid_moves has a SIDE EFFECT: it recomputes game_stages for the
    # current player. Re-read the state so `start` matches what the JS engine
    # will be handed (it must reproduce the same side effect).
    start = state(b)
    after = []
    for m in moves:
        base = len(b.moves)
        b.apply_move(m, switch_turn=False)
        res = fingerprint(b)
        while len(b.moves) > base:
            b.undo_last_move()
        if m == (1, 1, 1):          # calling a draw pushes no undo record
            b.draw_called = False
        back = fingerprint(b)
        after.append({'move': jmove(m), 'result': res,
                      'undo_ok': back == start_fp(start)})
    return {'state': start, 'moves': [jmove(m) for m in moves], 'after': after}


def start_fp(s):
    """The fingerprint half of a state dict (state adds two keys)."""
    return {k: v for k, v in s.items() if k not in ('draw_callable', 'no_save_turns')}


def endgame_board(rng):
    """A hand-built endgame. Random walking never gets here -- saving needs a
    piece parked on its own goal, which chance does not arrange -- yet this is
    where the rules are hardest: the endgame's higher-die save for blanks, the
    "no higher goal occupied" condition on it, and the last-piece renumbering.

    White is put fully in the endgame (every piece saved or on a goal it can be
    saved from); black is left partly on the field, so the two players are in
    different stages, which is itself a case worth covering.
    """
    b = Board()
    goals = {t.number: t for t in b.tiles if t.type == 'save'}
    field = [t for t in b.tiles if t.type == 'field']

    def place(piece, tile):
        if piece.rack is not None:
            piece.rack.remove(piece)
            piece.rack = None
        piece.tile = tile
        tile.pieces.append(piece)

    def to_saved(piece, rack):
        if piece.rack is not None:
            piece.rack.remove(piece)
        piece.tile = None
        piece.rack = rack
        rack.append(piece)

    white = [p for p in b.pieces if p.player == 'white']
    black = [p for p in b.pieces if p.player == 'black']
    # Leave between 1 and 4 white pieces on the board; the rest are already
    # saved. One left over exercises the last-piece rule.
    keep = rng.sample(white, rng.randint(1, 4))
    for p in white:
        if p in keep:
            place(p, goals[p.number] if p.number <= 6 else goals[rng.randint(1, 6)])
        else:
            to_saved(p, b.white_saved)
    for p in black:
        r = rng.random()
        if r < 0.45:
            to_saved(p, b.black_saved)
        elif r < 0.8:
            place(p, rng.choice(field))
        else:
            place(p, goals[p.number] if p.number <= 6 else goals[rng.randint(1, 6)])

    b.current_player = 'white'
    for d in b.dice:
        d.number = rng.randint(1, 6)
        d.used = False
    b.assign_piece_indices()
    b.piece_lookup = {(p.player, p.number): p for p in b.pieces}
    b.game_stages['white'] = b.get_game_stage('white')
    b.game_stages['black'] = b.get_game_stage('black')
    b.apply_last_piece_rule()
    b._blocked_key_cache.clear()
    b._blot_key_cache.clear()
    b._distance_cache.clear()
    return b


def main(n=40, out='engine_fixture.json'):
    cases = []
    seed = 0
    bad_undo = 0
    while len(cases) < n:
        random.seed(seed)
        seed += 1
        b = Board()
        # Walk a few turns so the sample spans opening/midgame/endgame shapes.
        # The walk MUST be canonicalised: get_valid_moves' order is unstable, so
        # an uncanonicalised walk gives different positions run to run.
        # Deep enough to leave the opening: entering twelve pieces takes a dozen
        # turns on its own, so a shallow walk samples nothing but openings (a
        # first pass got 38 opening / 2 midgame / 0 endgame out of 40).
        for _ in range(random.randint(2, 70)):
            if b.check_game_over()[0]:
                break
            for _ in range(2):
                ms = sorted((m for m in b.get_valid_moves() if isinstance(m[0], tuple)),
                            key=canon)
                if not ms:
                    break
                # Bias toward saving. A purely random walk almost never reaches
                # the endgame (every piece saveable), which is exactly where the
                # rules are hardest -- the higher-die save and the last-piece
                # renumbering both live there.
                saves = [m for m in ms if m[1] == 'save']
                pool = saves if (saves and random.random() < 0.75) else ms
                b.apply_move(random.choice(pool), switch_turn=False)
            b.switch_turn()
        if b.check_game_over()[0]:
            continue
        c = case(b)
        if not c['moves']:
            continue
        bad_undo += sum(0 if a['undo_ok'] else 1 for a in c['after'])
        c['seed'] = seed - 1
        c['kind'] = 'walk'
        cases.append(c)

    # A quarter again of hand-built endgames, for the rules the walk cannot reach.
    rng = random.Random(12345)
    built = 0
    while built < max(4, n // 4):
        b = endgame_board(rng)
        if b.check_game_over()[0]:
            continue
        c = case(b)
        if not c['moves']:
            continue
        bad_undo += sum(0 if a['undo_ok'] else 1 for a in c['after'])
        c['seed'] = -1 - built
        c['kind'] = 'endgame'
        cases.append(c)
        built += 1

    with open(out, 'w') as f:
        json.dump({'cases': cases}, f)
    total = sum(len(c['moves']) for c in cases)
    kinds = {}
    for c in cases:
        for m in c['moves']:
            k = ('pass/draw' if m['piece'] is None else
                 'save' if m['dest'] == 'save' else
                 'block-save' if m['dest'] == 0 else 'move')
            kinds[k] = kinds.get(k, 0) + 1
    print(f'wrote {out}: {len(cases)} positions, {total} moves {kinds}')
    print(f'apply/undo round-trips exactly: {total - bad_undo}/{total}')
    stages = {}
    for c in cases:
        s = c['state']['stages'][c['state']['current_player']]
        stages[s] = stages.get(s, 0) + 1
    print('stage coverage:', stages)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40,
         sys.argv[2] if len(sys.argv) > 2 else 'engine_fixture.json')
