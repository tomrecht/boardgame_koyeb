"""test_hand_rules.py -- the agent's two hand-coded play rules are switchable,
and the switch gates the CODE PATH, not just the outcome.

Rules (agent_gnn, from main 2026-10-01/02):
  * _wastes_save: drop candidate pairs that leave a die unused while a save is
    legal for it;
  * bank-the-most: if the opponent's last two pieces are blanks on goal 1 (they
    win next turn whatever happens), keep only the pairs with the most own saves.
Training (generation + gating) runs with hand_rules=False.

Two constructed positions (built with Board.update_state, so derived state --
stages, piece indices, last-piece rule -- is recomputed by the engine, not
hand-mutated). For each, with the rules ON the branch is reached AND removes
candidates; with them OFF the helper functions are replaced by tripwires that
raise, and the agent still answers -- so the branch is provably not entered.

Run:  python test_hand_rules.py   (uses symaug_iter6.pt)
"""
import os
import sys

os.environ.setdefault('BOARDGAME_DEVICE', 'cpu')
import torch
torch.set_grad_enabled(False)
import network
network.DEVICE = torch.device('cpu')
from network import BoardGNN
import agent_gnn
from agent_gnn import GNNAgent
from game import Board

CKPT = os.environ.get('CKPT', 'symaug_iter6.pt')


def _p(color, n, ring, sector):
    return {'color': color, 'number': n, 'tile': {'ring': ring, 'sector': sector}}


def _rack(color, nums):
    return [{'color': color, 'number': n} for n in nums]


def position(black_last_two_on_goal1):
    """White to move, dice 2 and 3. White: 1-6 banked, blanks 7 on goal 2 and
    8 on goal 3 (each savable with its own die), blanks 9-12 on the field.
    Black: either its last two pieces are blanks on goal 1 (certain loss for
    white -> bank-most rule), or it has four pieces out on the field."""
    white = [_p('white', 7, 7, 4), _p('white', 8, 7, 8),
             _p('white', 9, 3, 1), _p('white', 10, 3, 3),
             _p('white', 11, 4, 4), _p('white', 12, 5, 8)]
    if black_last_two_on_goal1:
        black = [_p('black', 7, 7, 12), _p('black', 8, 7, 12)]
        black_saved = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12]
    else:
        black = [_p('black', 7, 2, 2), _p('black', 8, 2, 6),
                 _p('black', 9, 4, 8), _p('black', 10, 6, 10)]
        black_saved = [1, 2, 3, 4, 5, 6, 11, 12]
    return {'currentTurn': 'white',
            'dice': [{'value': 2, 'used': False}, {'value': 3, 'used': False}],
            'racks': {'whiteUnentered': [], 'whiteSaved': _rack('white', [1, 2, 3, 4, 5, 6]),
                      'blackUnentered': [], 'blackSaved': _rack('black', black_saved)},
            'boardPieces': white + black}


def agent(model, hand_rules):
    return GNNAgent(model=model, hand_rules=hand_rules)


def choose(ag, state):
    b = Board()
    b.update_state(state)
    moves = list(b.get_valid_moves())
    return ag.select_move_pair(moves, b, 'white'), b, moves


def main():
    m = BoardGNN()
    m.load_state_dict(torch.load(CKPT, map_location='cpu'))
    m.eval()
    fails = 0

    for loss_case in (False, True):
        st = position(loss_case)
        b = Board(); b.update_state(st)
        moves = list(b.get_valid_moves())
        saves = [mv for mv in moves if mv[1] == 'save']
        print(f"\nposition {'B: opponent wins next turn' if loss_case else 'A: ordinary midgame'}"
              f" | stage {b.game_stages['white']} | {len(moves)} first moves, saves legal: {saves}")
        assert saves, 'fixture does not offer a save -- it would test nothing'
        assert agent_gnn._opponent_wins_next_turn_regardless(b, 'white') == loss_case, \
            'fixture does not put the loss-certain condition where intended'

        on = agent(m, True)
        pair_on, _, _ = choose(on, st)
        print(f"  rules ON : pair {pair_on} | stats {on.rule_stats}")
        ok = on.rule_stats['wastes_save_checked'] > 0 and on.rule_stats['wastes_save_dropped'] > 0
        if loss_case:
            ok = ok and on.rule_stats['bank_most_fired'] == 1
        else:
            ok = ok and on.rule_stats['bank_most_fired'] == 0
        print('   ', 'ok: branch reached and removed candidates' if ok else 'FAIL')
        fails += not ok

        # OFF: tripwires -- if the gated branch were entered these would raise.
        saved = (agent_gnn._wastes_save, agent_gnn._opponent_wins_next_turn_regardless)

        def tripwire(*a, **k):
            raise RuntimeError('hand rule evaluated with hand_rules=False')
        agent_gnn._wastes_save = tripwire
        agent_gnn._opponent_wins_next_turn_regardless = tripwire
        try:
            off = agent(m, False)
            pair_off, _, _ = choose(off, st)
            ok = all(v == 0 for v in off.rule_stats.values())
            print(f"  rules OFF: pair {pair_off} | stats {off.rule_stats}")
            print('   ', 'ok: branch never entered (tripwires untouched)' if ok else 'FAIL')
            fails += not ok
        except RuntimeError as e:
            print('  rules OFF: FAIL --', e)
            fails += 1
        finally:
            agent_gnn._wastes_save, agent_gnn._opponent_wins_next_turn_regardless = saved

        if loss_case:
            n_on = agent_gnn._own_saves(pair_on, 'white')
            print(f"  own saves in chosen pair: ON {n_on}, OFF {agent_gnn._own_saves(pair_off, 'white')}")
            if n_on != 2:
                print('  FAIL: bank-most rule did not bank the maximum (2)')
                fails += 1

    print('\nALL OK' if not fails else f'\n{fails} FAILURE(S)')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
