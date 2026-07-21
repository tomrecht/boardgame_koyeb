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


def is_blocksave(m):
    """A block-save move: (pid, 0, 0) with pid a real piece -- saving an
    opponent's 2+-piece field block (dissolves it, banks the pieces). Tests
    hypothesis 2b: does iter14 beat iter10's blocks by SAVING them?"""
    return (isinstance(m, tuple) and len(m) == 3
            and m not in ((0, 0, 0), (1, 1, 1)) and m[1] == 0 and m[2] == 0)


def is_capture_move(board, m, mover):
    """m lands on a field tile holding exactly one opponent piece -> a capture."""
    if not (isinstance(m, tuple) and len(m) == 3) or m in ((0, 0, 0), (1, 1, 1)):
        return False
    _pid, dest, _roll = m
    if not (isinstance(dest, (list, tuple)) and len(dest) == 2):
        return False                          # 'save' / 0(block-save) never capture
    tile = board.get_tile(dest[0], dest[1])
    return bool(tile and tile.type == 'field' and len(tile.pieces) == 1
               and tile.pieces[0].player != mover)


def _snapshot(board, col):
    """Per-turn style counters for one color: (blots, parked-on-goal,
    field-blocks). Cheap positional reads."""
    blots = parked = 0
    block_tiles = set()
    for p in board.pieces:
        if p.player != col or p.tile is None:
            continue
        t = p.tile
        if t.type == 'field':
            if len(t.pieces) == 1:
                blots += 1
            elif len(t.pieces) >= 2 and t.pieces[0].player == col:
                block_tiles.add(t.index)
        elif t.type == 'save':
            parked += 1                       # sitting on a goal, not yet saved
    return blots, parked, len(block_tiles)


def play_game(seed, white_agent, black_agent):
    random.seed(seed)
    board = Board()
    agents = {'white': white_agent, 'black': black_agent}
    last_saved, since_save = 0, 0
    ever = {'white': set(), 'black': set()}          # goal-pairs ever blocked against col
    held = {'white': Counter(), 'black': Counter()}  # turns each such block held
    offg = {'white': 0, 'black': 0}                  # offgoal moves made
    caps = {'white': 0, 'black': 0}                  # captures made
    lost = {'white': 0, 'black': 0}                  # own pieces captured
    bsav = {'white': 0, 'black': 0}                  # block-saves made (dissolve opp block)
    passes = {'white': 0, 'black': 0}                # turns a die was left unused
    sums = {'white': [0, 0, 0], 'black': [0, 0, 0]}  # running [blots,parked,blocks]
    snaps = {'white': 0, 'black': 0}
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
        # style + block snapshot (both colors) at turn start
        for col in ('white', 'black'):
            bp = blocked_pairs(board, col)
            ever[col] |= bp
            for pr in bp:
                held[col][pr] += 1
            b, pk, bl = _snapshot(board, col)
            sums[col][0] += b; sums[col][1] += pk; sums[col][2] += bl
            snaps[col] += 1
        player = board.current_player
        moves = board.get_valid_moves()
        chosen = agents[player].select_move_pair(list(moves), board, player)
        if isinstance(chosen, tuple) and len(chosen) == 3:
            chosen = (chosen, (0, 0, 0))
        if any(m == (0, 0, 0) for m in chosen):
            passes[player] += 1               # a die was left unused this turn
        for m in chosen:
            if is_offgoal_move(board, m):
                offg[player] += 1
            if is_blocksave(m):
                bsav[player] += 1
            if is_capture_move(board, m, player):
                caps[player] += 1
                lost['black' if player == 'white' else 'white'] += 1
            if m != (0, 0, 0):
                board.apply_move(m, switch_turn=False)
        board.switch_turn()
    else:
        winner, score = None, 0

    def style(col):
        n = max(snaps[col], 1)
        return {'blots_pt': round(sums[col][0] / n, 3),
                'parked_pt': round(sums[col][1] / n, 3),
                'fieldblocks_pt': round(sums[col][2] / n, 3)}
    # attribute: a pair blocked AGAINST 'black' was achieved by the WHITE agent
    out = {'winner': winner, 'score': score, 'secs': round(time.time() - t0, 1),
           'white_blocked_achieved': sorted(PAIR_NAME[p] for p in ever['black']),
           'black_blocked_achieved': sorted(PAIR_NAME[p] for p in ever['white']),
           'white_block_turns': {PAIR_NAME[p]: held['black'][p] for p in ever['black']},
           'black_block_turns': {PAIR_NAME[p]: held['white'][p] for p in ever['white']}}
    for col in ('white', 'black'):
        out[f'{col}_offgoals'] = offg[col]
        out[f'{col}_captures'] = caps[col]
        out[f'{col}_lost'] = lost[col]
        out[f'{col}_blocksaves'] = bsav[col]
        out[f'{col}_passes'] = passes[col]
        out[f'{col}_style'] = style(col)
    return out


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

    def blank():
        return {'games': 0, 'wins': 0, 'margin': 0, 'offg': 0, 'caps': 0,
                'lost': 0, 'bsav': 0, 'passes': 0, 'anyblock': 0,
                'pair': Counter(), 'blkturns': 0, 'blots': 0.0, 'parked': 0.0,
                'fblocks': 0.0}
    agg = {A_TAG: blank(), B_TAG: blank()}
    for r in rows:
        for col in ('white', 'black'):
            a = agg[r[col]]
            a['games'] += 1
            a['offg'] += r[f'{col}_offgoals']
            a['caps'] += r[f'{col}_captures']
            a['lost'] += r[f'{col}_lost']
            a['bsav'] += r.get(f'{col}_blocksaves', 0)
            a['passes'] += r[f'{col}_passes']
            st = r[f'{col}_style']
            a['blots'] += st['blots_pt']; a['parked'] += st['parked_pt']
            a['fblocks'] += st['fieldblocks_pt']
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
        print(f"  win {100*a['wins']/g:.1f}%  margin {a['margin']/g:+.2f}"
              f"  len? (see secs)")
        print(f"  GOAL-PAIR block (any): {100*a['anyblock']/g:.1f}% of games "
              f"| block-turns/game {a['blkturns']/g:.1f}"
              f"  [2&4 {100*a['pair']['2&4']/g:.0f}% / "
              f"1&6 {100*a['pair']['1&6']/g:.0f}% / 3&5 {100*a['pair']['3&5']/g:.0f}%]")
        print(f"  CAPTURING: captures/game {a['caps']/g:.2f}  "
              f"pieces-lost/game {a['lost']/g:.2f}  "
              f"net {(a['caps']-a['lost'])/g:+.2f}")
        print(f"  BLOCK-SAVES/game {a['bsav']/g:.3f}  "
              f"(dissolving the OTHER agent's blocks -- hypothesis 2b)")
        print(f"  EXPOSURE: blots/turn {a['blots']/g:.2f}  "
              f"parked-on-goal/turn {a['parked']/g:.2f}  "
              f"field-blocks/turn {a['fblocks']/g:.2f}")
        print(f"  offgoals/game {a['offg']/g:.3f}  passes(die-unused)/game "
              f"{a['passes']/g:.2f}")
    # --- BLOCK PAYOFF: among games where a pair was blocked, blocker win rate ---
    print("\n=== BLOCK PAYOFF (does achieving a block correlate with winning?) ===")
    for pr in ('2&4', '1&6', '3&5', 'ANY'):
        w = n = 0
        per_agent = {A_TAG: [0, 0], B_TAG: [0, 0]}   # [blocker_wins, instances]
        for r in rows:
            if r['winner'] is None:
                continue
            for col in ('white', 'black'):
                blocks = r[f'{col}_blocked_achieved']
                hit = bool(blocks) if pr == 'ANY' else (pr in blocks)
                if hit:
                    won = int(r['winner'] == col)
                    w += won; n += 1
                    pa = per_agent[r[col]]; pa[0] += won; pa[1] += 1
        if n:
            detail = '  '.join(f"{t} {pa[0]}/{pa[1]}" for t, pa in per_agent.items() if pa[1])
            print(f"  {pr}: blocker won {w}/{n} = {100*w/n:.0f}%   (by achiever: {detail})")
        else:
            print(f"  {pr}: no blocks observed")
    dec = sum(1 for r in rows if r['winner'] is not None)
    print(f"  baseline: {A_TAG} {100*agg[A_TAG]['wins']/max(dec,1):.0f}% / "
          f"{B_TAG} {100*agg[B_TAG]['wins']/max(dec,1):.0f}% unconditional "
          f"(compare blocker-win-by-achiever against these)")

    print("\nHYPOTHESIS CHECK: expect iter10 goal-pair-block >> iter14, iter10 "
          "concentrated on 2&4, iter14 flat-but-low. BLOCK PAYOFF tests whether\n"
          "blocking actually wins (blocker-win > that agent's baseline => it "
          "pays). Other axes profile HOW the styles differ beyond blocking.")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'analyze':
        analyze()
    else:
        run()
