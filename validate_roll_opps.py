"""features_v2.roll_opportunities against the engine's own per-piece
reachability (game.py get_reachable_tiles_by_dice: shortest routes, walls, the
save rule with stage and highest-goal check), for both sides of every turn-start
position in owner's games. Turn obligations are ignored on both sides by design
(the feature counts what each die CAN do).

    python3 validate_roll_opps.py [n_games=20]
"""
import json, sys

import features_v2 as F
import weakness_probe as W


def engine_counts(board, player):
    enemy = F._other(player)
    saved_cp = board.current_player
    stages0 = dict(board.game_stages)
    board.current_player = player              # is_blocked() reads the mover
    board.game_stages[player] = board.get_game_stage(player)
    dice0 = [(d.number, d.used) for d in board.dice]
    fm0 = board.firstMove
    board.firstMove = None
    movers = [p for p in board.pieces if p.player == player and p.tile is not None]
    rack = board.white_unentered if player == 'white' else board.black_unentered
    if rack and not any(p.player == player for p in board.home_tile.pieces):
        movers.append(rack[0])
    out = []
    try:
        for d in range(1, 7):
            for die in board.dice:
                die.number, die.used = d, False
            sv_n = sv_b = gl_n = gl_b = 0
            caps_n, caps_b = set(), set()
            for p in movers:
                board.get_reachable_tiles_by_dice(p)
                reach = (p.reachable_tiles or {}).get(d, [])
                if 'save' in reach:
                    if p.number <= 6:
                        sv_n += 1
                    else:
                        sv_b += 1
                tiles = [t for t in reach if t != 'save']
                home_already = p.tile is not None and p.tile.type == 'save' and p.can_be_saved()
                if not home_already and any(t.type == 'save' and (p.number > 6 or t.number == p.number)
                                            for t in tiles):
                    if p.number <= 6:
                        gl_n += 1
                    else:
                        gl_b += 1
                for t in tiles:
                    if t.type == 'field' and len(t.pieces) == 1 and t.pieces[0].player == enemy:
                        e = t.pieces[0]
                        (caps_n if e.number <= 6 else caps_b).add(e)
            out += [sv_n, sv_b, gl_n, gl_b, len(caps_n), len(caps_b)]
    finally:
        board.current_player = saved_cp
        board.game_stages.update(stages0)
        for die, (n, u) in zip(board.dice, dice0):
            die.number, die.used = n, u
        board.firstMove = fm0
    return [min(c, 6) / 6.0 for c in out]


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    recs = [json.loads(l) for l in open('quahuru-games-apvo2h65-v2.jsonl')]
    recs = [r for r in recs if r.get('completed')][:n_games]
    names = ['save numbered', 'save blank', 'onto goal numbered', 'onto goal blank',
             'capture numbered', 'capture blank']
    n_pos = n_bad = 0
    nonzero = [0] * 6
    bad_by = [0] * 6
    shown = 0
    for rec in recs:
        for t, b, played, b2 in W.turn_states(rec):
            for side in ('white', 'black'):
                a = F.roll_opportunities(b, side)
                e = engine_counts(b, side)
                n_pos += 1
                for i, (x, y) in enumerate(zip(a, e)):
                    if y:
                        nonzero[i % 6] += 1
                    if abs(x - y) > 1e-9:
                        bad_by[i % 6] += 1
                if a != e:
                    n_bad += 1
                    if shown < 3:
                        import replay_games as R
                        print('MISMATCH', rec['id'][:8], t, side, R.pos_string(b))
                        for d in range(6):
                            print('   die', d + 1, 'feature', [round(v * 6) for v in a[6*d:6*d+6]],
                                  'engine', [round(v * 6) for v in e[6*d:6*d+6]])
                        shown += 1
    print(f'{n_pos} (position, side) pairs from {len(recs)} games: {n_pos - n_bad} identical, {n_bad} differ')
    for i, nm in enumerate(names):
        print(f'  {nm:<20} nonzero in {nonzero[i]:>5} (die, side, position) cells; mismatched {bad_by[i]}')


if __name__ == '__main__':
    main()
