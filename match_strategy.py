"""iter10 vs iter14 head-to-head with STRATEGIC instrumentation.

Motivation (owner playtest): human beats iter14 (20/32) but loses to iter10
(5/16) despite iter14 promoting OVER iter10 at 60% -- a non-transitivity.
Owner's hypothesis: blocking a goal-pair is a strong weapon; iter10 wields it
a LOT (concentrated on 2&4), iter14 spreads it evenly but fires it RARELY, so
iter10 is stronger vs a human. This match tests it directly.

Both agents play plain 2-ply shallow (the deployed policy). Paired/color-
swapped seeds (CRN). Per game we log, for each agent:
  - goal-pair blocks ACHIEVED against the opponent, per pair (2&4,1&6,3&5) and
    any -- detection is free (CLAUDE.md): goal g is denied to player P iff P's
    non-saved numbered piece g has shortest_route_to_goal == INF; a pair is
    blocked iff both its goals are denied. Recorded as "ever this game" plus
    the number of turns it held (transient vs sustained).
  - offgoal moves made (numbered piece moved OFF its own save tile).
  - win / margin / game length.

Usage:
  PYTHONHASHSEED=0 python -u match_strategy.py                 # run games
  N_SEEDS=150 python -u match_strategy.py                      # 300 games
  python match_strategy.py analyze                             # aggregate the jsonl
Resumable: appends to match_strategy.jsonl, skips recorded games.
"""
import os, sys, json, time, random
from collections import Counter
os.environ.setdefault('BOARDGAME_DEVICE', 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
torch.set_num_threads(1); torch.set_grad_enabled(False)
import network
network.DEVICE = torch.device('cpu')
from network import BoardGNN
from game import Board
from agent_gnn import GNNAgent

REPO = os.path.dirname(os.path.abspath(__file__))
A_TAG, A_NET = 'iter10', f'{REPO}/td_champion_July18_iter10.pt'
B_TAG, B_NET = 'iter14', f'{REPO}/td_champion_July19_iter14.pt'
N_SEEDS = int(os.environ.get('N_SEEDS', '150'))
RESULTS = os.environ.get('RESULTS', f'{REPO}/match_strategy.jsonl')
SEED_BASE = 3_300_000
MAX_TURNS, STUCK_LIMIT = 200, 60
INF = float('inf')
PAIRS = [(2, 4), (1, 6), (3, 5)]
PAIR_NAME = {(2, 4): '2&4', (1, 6): '1&6', (3, 5): '3&5'}


def denied_goals(board, player):
    saved = board.white_saved if player == 'white' else board.black_saved
    saved_nums = {p.number for p in saved}
    d = set()
    for p in board.pieces:
        if p.player == player and p.number <= 6 and p.number not in saved_nums:
            if board.shortest_route_to_goal(p) == INF:
                d.add(p.number)
    return d


def blocked_pairs(board, player):
    d = denied_goals(board, player)
    return {pair for pair in PAIRS if pair[0] in d and pair[1] in d}


def is_offgoal_move(board, m):
    if not (isinstance(m, tuple) and len(m) == 3) or m in ((0, 0, 0), (1, 1, 1)):
        return False
    pid, dest, _roll = m
    if not (isinstance(pid, tuple) and pid[1] <= 6 and dest != 'save'):
        return False
    p = board.piece_lookup.get(pid)
    return bool(p and p.tile is not None and p.tile.type == 'save'
               and p.tile.number == pid[1])


def play_game(seed, white_agent, black_agent):
    random.seed(seed)
    board = Board()
    agents = {'white': white_agent, 'black': black_agent}
    last_saved, since_save = 0, 0
    # per-color: pairs ever blocked AGAINST this color, turns held, offgoals made
    ever = {'white': set(), 'black': set()}
    held = {'white': Counter(), 'black': Counter()}
    offg = {'white': 0, 'black': 0}
    t0 = time.time()
    for _turn in range(MAX_TURNS):
        winner, score = board.check_game_over()
        if winner:
            break
        if board.draw_callable:
            winner, score = None, 0
            break
        cur = len(board.white_saved) + len(board.black_saved)
        if cur > last_saved:
            last_saved, since_save = cur, 0
        elif last_saved > 0:
            since_save += 1
        if last_saved > 0 and since_save >= STUCK_LIMIT:
            winner, score = None, 0
            break
        # strategic snapshot (both colors) at turn start
        for col in ('white', 'black'):
            bp = blocked_pairs(board, col)
            ever[col] |= bp
            for pr in bp:
                held[col][pr] += 1
        player = board.current_player
        moves = board.get_valid_moves()
        chosen = agents[player].select_move_pair(list(moves), board, player)
        if isinstance(chosen, tuple) and len(chosen) == 3:
            chosen = (chosen, (0, 0, 0))
        for m in chosen:
            if is_offgoal_move(board, m):
                offg[player] += 1
            if m != (0, 0, 0):
                board.apply_move(m, switch_turn=False)
        board.switch_turn()
    else:
        winner, score = None, 0
    # attribute: a pair blocked AGAINST 'black' was achieved by the WHITE agent
    return {
        'winner': winner, 'score': score, 'secs': round(time.time() - t0, 1),
        'white_blocked_achieved': sorted(PAIR_NAME[p] for p in ever['black']),
        'black_blocked_achieved': sorted(PAIR_NAME[p] for p in ever['white']),
        'white_block_turns': {PAIR_NAME[p]: held['black'][p] for p in ever['black']},
        'black_block_turns': {PAIR_NAME[p]: held['white'][p] for p in ever['white']},
        'white_offgoals': offg['white'], 'black_offgoals': offg['black'],
    }


def run():
    mA = BoardGNN(); mA.load_state_dict(torch.load(A_NET, map_location='cpu')); mA.eval()
    mB = BoardGNN(); mB.load_state_dict(torch.load(B_NET, map_location='cpu')); mB.eval()
    agA, agB = GNNAgent(model=mA), GNNAgent(model=mB)
    done = set()
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            for line in f:
                done.add(json.loads(line)['game'])
        print(f"resuming: {len(done)} games recorded", flush=True)
    print(f"A={A_TAG} B={B_TAG} N_SEEDS={N_SEEDS} (x2 colors)", flush=True)
    for k in range(N_SEEDS):
        seed = SEED_BASE + k
        # game 0: A=white,B=black ; game 1: B=white,A=black  (CRN, color-swap)
        for gi, (wag, wtag, bag, btag) in enumerate((
                (agA, A_TAG, agB, B_TAG), (agB, B_TAG, agA, A_TAG))):
            gid = f"{k}_{gi}"
            if gid in done:
                continue
            r = play_game(seed, wag, bag)
            r.update({'game': gid, 'seed': seed, 'white': wtag, 'black': btag})
            with open(RESULTS, 'a') as f:
                f.write(json.dumps(r) + '\n')
            wa = r['winner']
            print(f"{gid}: W={wtag} B={btag} winner={wa} score={r['score']} "
                  f"| W-blocks={r['white_blocked_achieved']} "
                  f"B-blocks={r['black_blocked_achieved']} {r['secs']}s", flush=True)
    print("RUN COMPLETE (or N_SEEDS exhausted)", flush=True)


def analyze():
    rows = [json.loads(l) for l in open(RESULTS)]
    decisive = [r for r in rows if r['winner'] is not None]
    print(f"{len(rows)} games ({len(decisive)} decisive)\n")
    # per-agent aggregation (agent may be white or black in a given row)
    agg = {A_TAG: {'games': 0, 'wins': 0, 'margin': 0, 'offg': 0,
                   'anyblock': 0, 'pair': Counter(), 'blkturns': 0},
           B_TAG: {'games': 0, 'wins': 0, 'margin': 0, 'offg': 0,
                   'anyblock': 0, 'pair': Counter(), 'blkturns': 0}}
    for r in rows:
        for col, other in (('white', 'black'), ('black', 'white')):
            tag = r[col]
            a = agg[tag]
            a['games'] += 1
            a['offg'] += r[f'{col}_offgoals']
            blocks = r[f'{col}_blocked_achieved']
            if blocks:
                a['anyblock'] += 1
            for pr in blocks:
                a['pair'][pr] += 1
            a['blkturns'] += sum(r[f'{col}_block_turns'].values())
            if r['winner'] is not None:
                won = (r['winner'] == col)
                a['wins'] += int(won)
                a['margin'] += r['score'] if won else -r['score']
    for tag in (A_TAG, B_TAG):
        a = agg[tag]; g = max(a['games'], 1)
        print(f"=== {tag} (n={a['games']}) ===")
        print(f"  win rate {100*a['wins']/g:.1f}%  avg margin {a['margin']/g:+.2f}"
              f"  offgoals/game {a['offg']/g:.3f}")
        print(f"  ANY-pair block achieved: {100*a['anyblock']/g:.1f}% of games "
              f"| avg block-turns/game {a['blkturns']/g:.1f}")
        for pr in ('2&4', '1&6', '3&5'):
            print(f"    {pr}: {100*a['pair'][pr]/g:.1f}% of games")
    print("\nHYPOTHESIS CHECK: expect iter10 ANY-block >> iter14, iter10 "
          "concentrated on 2&4, iter14 flat-but-low.")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'analyze':
        analyze()
    else:
        run()
