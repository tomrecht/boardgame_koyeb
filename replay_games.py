"""Replay recorded games in game.py and check they reproduce the recorded board.

Every recorded turn carries a fingerprint of the position it produced (see
`_recPosString` / `_recHash` in game.js). This replays the recorded moves through
`game.py` and compares. It does two jobs at once:

  1. **Proves the recording is complete.** If a move path in game.js is not hooked
     by the recorder, the replay diverges and this says which game and turn.
     Without that, the whole log could be quietly unusable for analysis.
  2. **Conformance-tests game.js against game.py** over real play. The port was
     verified 110/110 on chosen agent pairs, but game.js is the FRONTEND's own
     engine and has never been checked against game.py on move application across
     arbitrary games. A divergence here is a genuine engine mismatch and worth
     knowing about on its own.

A recorded move is resolved against `get_valid_moves()` rather than rebuilt as a
tuple: that derives the die (which the log deliberately omits) and asserts the
move was legal in game.py's view at the same time.

Usage:
  python3 replay_games.py logs/*.jsonl              # summary
  python3 replay_games.py logs/*.jsonl -v           # first divergence per game
"""
import sys, json, argparse, os
os.environ.setdefault('BOARDGAME_DEVICE', 'cpu')
from game import Board


def fnv1a(s):
    """Must match _recHash in game.js exactly."""
    h = 0x811c9dc5
    for ch in s:
        h ^= ord(ch)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def pos_string(board, shown=None):
    """Must match _recPosString in game.js exactly: whose turn, every piece's
    tile, and the two saved counts. Sorted as strings -- JS's .sort() and Python's
    sorted() agree on ASCII. BLANKS ARE ANONYMOUS (`w*`) because the game treats
    same-colour blanks as interchangeable and game.py dedups them on a shared
    tile, so naming them would call two identical boards different."""
    # `shown` maps a piece to the number game.js still displays for it. The
    # LAST-PIECE RULE is applied at different moments: game.py blanks a player's
    # last piece at EVERY turn switch (both players), game.js only at the start
    # of that player's OWN turn -- so during the opponent's turn game.js still
    # shows the number. Equivalent for play (the piece is blank whenever its
    # owner moves it); only the fingerprint differs. Found 2026-10-03: it was
    # all 17 final-turn divergences in owner's first 83 recorded games.
    shown = shown or {}
    on = sorted(
        f'{p.tile.ring}.{p.tile.pos}:{p.player[0]}'
        + ('*' if shown.get(id(p), p.number) > 6 else str(shown.get(id(p), p.number)))
        for p in board.pieces if p.tile is not None
    )
    return (board.current_player[0] + '|' + ','.join(on) + '|'
            + f'{len(board.white_saved)},{len(board.black_saved)}')


def fingerprint(board, shown=None):
    """base36, because that is what game.js stores (shorter than decimal)."""
    return _b36(fnv1a(pos_string(board, shown)))


def _b36(n):
    if n == 0:
        return '0'
    d = '0123456789abcdefghijklmnopqrstuvwxyz'
    out = ''
    while n:
        n, r = divmod(n, 36)
        out = d[r] + out
    return out


def parse_move(s, mover):
    """'7>5.4' | '7>s' | 'o7>b'  ->  (colour, number, destination)."""
    left, right = s.split('>')
    colour = mover
    if left.startswith('o'):
        left = left[1:]
        colour = 'white' if mover == 'black' else 'black'
    number = int(left)
    if right == 's':
        dest = 'save'
    elif right == 'b':
        dest = 0
    else:
        r, p = right.split('.')
        dest = (int(r), int(p))
    return colour, number, dest


def replay(rec, verbose=False):
    """Returns (ok, message). Stops at the first divergence."""
    board = Board()
    # Force the recorded rack order. Every piece is still in its rack at this
    # point, so reordering the list IS setting the order.
    for colour, key in (('white', 'rackWhite'), ('black', 'rackBlack')):
        want = rec.get(key)
        rack = board.white_unentered if colour == 'white' else board.black_unentered
        if not want or len(want) != len(rack):
            return False, f'rack order missing or wrong length for {colour}'
        order = {n: i for i, n in enumerate(want)}
        rack.sort(key=lambda p: order.get(p.number, 99))
    board.current_player = rec['starter']
    orig = {id(p): p.number for p in board.pieces}
    blanked_seen = set()      # pieces whose owner has had a turn start since blanking

    def shown_numbers():
        return {id(p): orig[id(p)] for p in board.pieces
                if p.number > 6 and orig[id(p)] <= 6 and id(p) not in blanked_seen}

    for ti, turn in enumerate(rec['turns']):
        for p in board.pieces:            # game.js blanks at its owner's turn start
            if p.player == board.current_player and p.number > 6 and orig[id(p)] <= 6:
                blanked_seen.add(id(p))
        # The recorded dice, both unused -- switch_turn() rolls fresh ones, so
        # they have to be set at the START of each turn, after the switch.
        for die, val in zip(board.dice, turn['d']):
            die.number = val
            die.used = False
        for ms in turn['m']:
            colour, number, dest = parse_move(ms, board.current_player)
            # RESOLVE THE EXACT PIECE, not "a legal move that looks like this".
            #
            # get_valid_moves() DEDUPS interchangeable blanks on a shared tile
            # ("keep one"), so it may not offer the very blank the log names. An
            # earlier version accepted an equivalent blank instead -- and that is
            # a trap: substituting once makes the piece-number mapping drift
            # permanently, so every later move naming that blank fails. It looked
            # like an engine divergence and was not. Measured: 0 of 3 games
            # replayed that way, against 3 of 3 once identity is preserved.
            #
            # So ask the PIECE for its own reachable set, which is pre-dedup, and
            # derive the roll from it. get_valid_moves() is still called first for
            # its side effect of refreshing game_stages, which save legality reads.
            board.get_valid_moves()
            piece = board.piece_lookup.get((colour, number))
            if piece is None:
                return False, f'turn {ti+1}: no piece {colour} {number}'
            roll = None
            if dest == 0:
                roll = 0                      # a block-save costs both dice
            else:
                board.get_reachable_tiles_by_dice(piece)
                for r, dests in (piece.reachable_tiles or {}).items():
                    for d in dests:
                        if (d == 'save' and dest == 'save') or (
                                d != 'save' and dest != 'save'
                                and (d.ring, d.pos) == dest):
                            roll = r
                            break
                    if roll is not None:
                        break
            if roll is None and dest not in ('save', 0):
                # BACKWARD TOLERANCE for logs written before sum moves were
                # recorded as two halves. game.py has no single "sum move", so a
                # human's one-gesture move on the dice sum appears as a
                # destination no single die can reach. Walk it as two halves via
                # any valid intermediate. Only sound when NOTHING is captured en
                # route -- with a capture the intermediate decides what died, and
                # the old format did not record which one -- so bail out rather
                # than guess if an intermediate holds a lone enemy.
                d0, d1 = board.dice[0], board.dice[1]
                if not d0.used and not d1.used:
                    start_tile = board.home_tile if piece.tile is None else piece.tile
                    cands = []          # (intermediate, first_roll, second_roll)
                    if start_tile is not None:
                        for first, second in ((d0.number, d1.number), (d1.number, d0.number)):
                            for mid in board.get_reachable_tiles(start_tile, first):
                                outs = board.get_reachable_tiles(mid, second)
                                if any((t.ring, t.pos) == dest for t in outs):
                                    cands.append((mid, first, second))
                    # game.js's own en-route rule, mirrored: a route passing exactly
                    # ONE lone enemy TAKES it automatically, so the replay must use
                    # that intermediate; with none, every route ends in the same
                    # position and any will do; with two or more the destination is
                    # withheld and cannot have been played as one gesture at all.
                    # An earlier version SKIPPED capturing intermediates, which
                    # silently replayed a different position instead of failing.
                    def captures(t):
                        return (t.type != 'save' and len(t.pieces) == 1
                                and t.pieces[0].player != colour)
                    capturing = [c for c in cands if captures(c[0])]
                    chosen = None
                    if len(capturing) == 1:
                        chosen = capturing[0]
                    elif not capturing and cands:
                        chosen = cands[0]
                    if chosen is not None:
                        mid, first, second = chosen
                        board.apply_move(((colour, number), (mid.ring, mid.pos), first),
                                         switch_turn=False)
                        board.get_reachable_tiles_by_dice(piece)
                        roll = second
            if roll is None:
                return False, (f'turn {ti+1} ({board.current_player}): recorded move '
                               f'{ms!r} is not reachable for that piece in game.py '
                               f'(dice {[d.number for d in board.dice]}, '
                               f'used {[d.used for d in board.dice]})')
            board.apply_move(((colour, number), dest, roll), switch_turn=False)
        got = fingerprint(board, shown_numbers())
        if got != turn['h']:
            return False, (f'turn {ti+1} ({board.current_player}): position differs '
                           f'(recorded {turn["h"]}, replayed {got})'
                           + (f'\n      replayed: {pos_string(board, shown_numbers())}' if verbose else ''))
        board.switch_turn()
    return True, f'{len(rec["turns"])} turns'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+')
    ap.add_argument('-v', '--verbose', action='store_true')
    ap.add_argument('-n', '--limit', type=int, default=0, help='stop after N games')
    a = ap.parse_args()
    games = []
    for path in a.paths:
        with open(path) as f:
            for line in f:
                if line.strip():
                    games.append(json.loads(line))
    if a.limit:
        games = games[:a.limit]
    ok = bad = 0
    for g in games:
        good, msg = replay(g, a.verbose)
        if good:
            ok += 1
        else:
            bad += 1
            print(f'  DIVERGED  {g.get("id","?")[:8]}  {msg}')
    print(f'\n{ok}/{ok+bad} games replay to the recorded position'
          + (f'   ({bad} diverged)' if bad else '   (all clean)'))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
