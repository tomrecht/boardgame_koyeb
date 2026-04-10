"""
train_gnn_MC.py — Self-play RL training for the GNN using Monte Carlo (MC) returns.

Algorithm:
  - Monte Carlo: target for position t is the discounted sum of all future
    shaped rewards plus a terminal +/-1.0 win/loss signal.
  - No target network or bootstrapping.
  - Shaped rewards for saving pieces, captures, endgame entry (small relative
    to the +-1 terminal — they provide dense gradient signal, not the primary
    learning objective).
  - Auxiliary head predicts (my_saved - opp_saved) / 12, weight AUX_LOSS_WEIGHT.

Why MC instead of TD:
  - Only agent turns are recorded, so effective timestep spacing is ~2x,
    making TD(1) especially myopic.
  - MC propagates the terminal +-1 cleanly without bootstrap bias from a
    weak early value function.
  - Simpler: no target network, no update interval to tune.

Data generation:
  - 50% vs frozen champion pool, 50% vs heuristic agent
  - 1-ply search (select_move_pair_fast) for speed
  - epsilon-greedy exploration

Evaluation (every generation):
  - vs heuristic agent: progress indicator
  - vs frozen champion pool: promotion gate
  - vs distilled baseline: collapse floor
  - Metrics: win rate + average margin, rolling 3-generation averages

Promotion:
  - win rate >= PROMOTION_WINRATE AND avg margin > best_frozen_margin
  - sustained over rolling PROMOTION_ROLLING_GENS-generation average
  - on promotion: add to frozen pool (max FROZEN_POOL_SIZE), drop oldest

Collapse detection:
  - rolling avg margin vs distilled drops below COLLAPSE_MARGIN_THRESHOLD
  - for COLLAPSE_CONSECUTIVE consecutive evaluations
  - action: reload best selfplay (or distilled) weights, clear buffer,
    reset frozen pool

Usage:
    python3 train_gnn_MC.py --full

Output:
    gnn_selfplay.pt                    — current best weights
    checkpoints/gnn_s{SESSION}_g{N}.pt — periodic checkpoints
"""

import argparse
import collections
import copy
import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

from game import Board
from agent_gnn import GNNAgent
from agent import Agent as HeuristicAgent
from network import (BoardEncoder, BoardGNN, collate_batch,
                     save_model, load_model, DEVICE,
                     AUX_LOSS_WEIGHT, NUM_PIECES)


CHECKPOINT_DIR = os.environ.get('CHECKPOINT_DIR', 'checkpoints')
SESSION        = int(time.time())

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -------------------------
# CONFIG
# -------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--full', action='store_true')
args = parser.parse_args()

if not args.full:
    raise SystemExit("POC mode removed. Run with --full.")

GAMES_PER_GEN       = 20
EVAL_PAIRS          = 10      # paired games (2 games each) per opponent per eval
BUFFER_SIZE         = 30_000
MIN_BUFFER          = 1_000
BATCH_SIZE          = 256
TRAINING_STEPS      = 100
LR                  = 1e-4
CHECKPOINT_INTERVAL = 5
MAX_TURNS           = 300

GAMMA            = 0.99
EXPLORATION_RATE = 0.10
LR_DECAY         = 0.995   # multiplicative per generation
MIN_LR           = 1e-5

FROZEN_POOL_SIZE          = 3
PROMOTION_WINRATE         = 0.55   # slightly above 50% to require genuine improvement
PROMOTION_ROLLING_GENS    = 3
COLLAPSE_MARGIN_THRESHOLD = -1.5
COLLAPSE_CONSECUTIVE      = 3

DISTILL_WEIGHTS  = 'gnn_weights.pt'
SELFPLAY_WEIGHTS = 'gnn_selfplay.pt'

# Shaped reward constants — small relative to the +-1 terminal signal.
# Purpose: dense gradient signal during the game, not primary objective.
R_SAVE_BASE         = 0.02
R_SAVE_NUMBER_SCALE = 0.002
R_CAPTURE           = 0.01
R_ENDGAME           = 0.05

print(f"FULL mode | games={GAMES_PER_GEN} eval_pairs={EVAL_PAIRS} "
      f"buffer={BUFFER_SIZE} min_buffer={MIN_BUFFER} batch={BATCH_SIZE}")
print(f"Session {SESSION} | checkpoints -> {CHECKPOINT_DIR}/")


# -------------------------
# HELPERS
# -------------------------

def save_model_and_buffer(model, path, buffer):
    save_model(model, path)
    buffer_file = os.path.join(CHECKPOINT_DIR, f'buffer_s{SESSION}.pkl')
    with open(buffer_file, 'wb') as f:
        pickle.dump(list(buffer), f)
    print(f"Saved replay buffer ({len(buffer)} positions) to {buffer_file}")


# -------------------------
# SHAPED REWARDS
# -------------------------

def compute_shaped_reward(board_before, board_after, current_player):
    """
    Shaped reward for one move transition, from current_player's perspective.

      +R_SAVE_BASE + R_SAVE_NUMBER_SCALE * piece.number  per piece we save
      mirror negative                                     per opponent piece saved
      +R_CAPTURE                                          per opponent piece newly on home
      +/-R_ENDGAME                                        on entering/opponent entering endgame
    """
    reward   = 0.0
    opponent = 'black' if current_player == 'white' else 'white'

    my_saved_before  = len(board_before.white_saved if current_player == 'white'
                           else board_before.black_saved)
    my_saved_after   = len(board_after.white_saved  if current_player == 'white'
                           else board_after.black_saved)
    opp_saved_before = len(board_before.black_saved if current_player == 'white'
                           else board_before.white_saved)
    opp_saved_after  = len(board_after.black_saved  if current_player == 'white'
                           else board_after.white_saved)

    newly_saved = my_saved_after - my_saved_before
    if newly_saved > 0:
        reward += newly_saved * R_SAVE_BASE
        my_save_rack = (board_after.white_saved if current_player == 'white'
                        else board_after.black_saved)
        for piece in my_save_rack[-newly_saved:]:
            if piece.number <= 6:
                reward += R_SAVE_NUMBER_SCALE * piece.number

    opp_newly_saved = opp_saved_after - opp_saved_before
    if opp_newly_saved > 0:
        reward -= opp_newly_saved * R_SAVE_BASE
        opp_save_rack = (board_after.black_saved if current_player == 'white'
                         else board_after.white_saved)
        for piece in opp_save_rack[-opp_newly_saved:]:
            if piece.number <= 6:
                reward -= R_SAVE_NUMBER_SCALE * piece.number

    opp_on_home_before = sum(1 for p in board_before.home_tile.pieces
                             if p.player == opponent)
    opp_on_home_after  = sum(1 for p in board_after.home_tile.pieces
                             if p.player == opponent)
    if opp_on_home_after > opp_on_home_before:
        reward += (opp_on_home_after - opp_on_home_before) * R_CAPTURE

    if (board_before.game_stages[current_player] != 'endgame' and
            board_after.game_stages[current_player] == 'endgame'):
        reward += R_ENDGAME
    if (board_before.game_stages[opponent] != 'endgame' and
            board_after.game_stages[opponent] == 'endgame'):
        reward -= R_ENDGAME

    return reward


# -------------------------
# BOARD STATE SNAPSHOT
# -------------------------

def snapshot_board_state(board):
    """
    Lightweight snapshot for reward computation.
    Avoids deep-copying the full Board object.
    """
    class Snap:
        pass
    s = Snap()
    s.white_saved      = list(board.white_saved)
    s.black_saved      = list(board.black_saved)
    s.home_tile_pieces = list(board.home_tile.pieces)
    s.game_stages      = dict(board.game_stages)
    # Keep references to live racks for piece identity checks
    s.white_saved_obj  = board.white_saved
    s.black_saved_obj  = board.black_saved
    return s


# -------------------------
# TERMINAL VALUE
# -------------------------

def _finish_game(record, winner, margin, current_player_is_agent):
    """
    Determine terminal value from the training agent's perspective.

    Pure self-play (current_player_is_agent is None): 0.0 — no unambiguous
    perspective, shaped rewards carry the signal.
    Win:  +1.0
    Draw:  0.0
    Loss: -1.0
    """
    if current_player_is_agent is None:
        terminal_value = 0.0
    elif winner is None:
        terminal_value = 0.0
    elif winner == current_player_is_agent:
        terminal_value = 1.0
    else:
        terminal_value = -1.0
    return record, winner, margin, terminal_value


# -------------------------
# SELF-PLAY GAME
# -------------------------

def play_game(agent, encoder, opponent_agent, current_player_is_agent,
              seed, heuristic_agent=None):
    """
    Play one self-play game.

    Returns (record, winner, margin, terminal_value) where:
      record        — list of (encoded, shaped_reward, aux_target) for each
                      of the training agent's turns
      winner        — 'white', 'black', or None
      margin        — int piece difference
      terminal_value — +1.0 / 0.0 / -1.0 from training agent's perspective

    current_player_is_agent: color the training agent plays ('white'/'black'),
      or None for pure self-play (both sides).
    opponent_agent: frozen GNNAgent, or None when using heuristic.
    heuristic_agent: HeuristicAgent instance (used when opponent_agent is None).
    """
    random.seed(seed)
    board  = Board()
    record = []
    turns  = 0
    consecutive_passes = 0

    # Track endgame entry per player to emit the one-time bonus only once
    endgame_entered = {'white': False, 'black': False}

    while turns < MAX_TURNS:
        winner, margin = board.check_game_over()
        if winner:
            return _finish_game(record, winner, margin, current_player_is_agent)

        current_player = board.current_player
        is_training_agent_turn = (
            current_player_is_agent is None or
            current_player == current_player_is_agent
        )

        moves = board.get_valid_moves()
        if not moves:
            break

        # Snapshot board state before move (for shaped reward computation)
        snap_before = snapshot_board_state(board)

        # Update endgame_entered to reflect actual board state before this turn
        for p in ['white', 'black']:
            if board.game_stages[p] == 'endgame' and not endgame_entered[p]:
                endgame_entered[p] = True

        # Encode position for training agent's turns
        if is_training_agent_turn:
            encoded = encoder.encode(board, current_player)
            encoded_stored = {k: v.clone() for k, v in encoded.items()}
            aux_target = (
                len(board.white_saved if current_player == 'white' else board.black_saved) -
                len(board.black_saved if current_player == 'white' else board.white_saved)
            ) / float(NUM_PIECES)

        # Select move
        if is_training_agent_turn:
            if random.random() < EXPLORATION_RATE:
                chosen = _random_move_pair(moves, board)
            else:
                chosen = agent.select_move_pair_fast(moves, board, current_player)
        else:
            if opponent_agent is not None:
                chosen = opponent_agent.select_move_pair_fast(
                    moves, board, current_player)
            else:
                chosen = heuristic_agent.select_move_pair(
                    moves, board, current_player)

        # Handle double-pass (stalemate detection)
        if chosen == ((0, 0, 0), (0, 0, 0)):
            consecutive_passes += 1
            if consecutive_passes >= 6:
                ws = len(board.white_saved)
                bs = len(board.black_saved)
                if ws > bs: return _finish_game(record, 'white', ws - bs, current_player_is_agent)
                if bs > ws: return _finish_game(record, 'black', bs - ws, current_player_is_agent)
                return _finish_game(record, None, 0, current_player_is_agent)
        else:
            consecutive_passes = 0

        # Apply moves
        for move in chosen:
            if move != (0, 0, 0):
                board.apply_move(move, switch_turn=False)
        board.switch_turn()

        # Compute shaped reward and store record entry
        if is_training_agent_turn:
            snap_after = snapshot_board_state(board)

            # Patch snap_before to use live rack references for piece identity
            snap_before.white_saved = snap_before.white_saved_obj
            snap_before.black_saved = snap_before.black_saved_obj
            snap_after.white_saved  = board.white_saved
            snap_after.black_saved  = board.black_saved
            snap_after.home_tile    = board.home_tile
            snap_before.home_tile   = type('T', (), {
                'pieces': snap_before.home_tile_pieces})()
            snap_after.game_stages  = board.game_stages

            # Suppress endgame bonus if already awarded this game
            if endgame_entered[current_player]:
                snap_before.game_stages[current_player] = 'endgame'
            opp = 'white' if current_player == 'black' else 'black'
            if endgame_entered[opp]:
                snap_before.game_stages[opp] = 'endgame'

            reward = compute_shaped_reward(snap_before, snap_after, current_player)

            # Update endgame_entered after reward computed
            for p in ['white', 'black']:
                if board.game_stages[p] == 'endgame':
                    endgame_entered[p] = True

            record.append((encoded_stored, reward, aux_target))

        turns += 1

    # MAX_TURNS reached — resolve by piece count
    ws = len(board.white_saved)
    bs = len(board.black_saved)
    if ws > bs: return _finish_game(record, 'white', ws - bs, current_player_is_agent)
    if bs > ws: return _finish_game(record, 'black', bs - ws, current_player_is_agent)
    return _finish_game(record, None, 0, current_player_is_agent)


def _random_move_pair(moves, board):
    """
    Pick a random legal first move, then a random legal second move
    from the resulting position. Correctly handles the case where the
    first move exhausts all dice.
    """
    moves_list = [m for m in moves if m != (0, 0, 0)]
    if not moves_list:
        return ((0, 0, 0), (0, 0, 0))
    first = random.choice(moves_list)
    initial = len(board.moves)
    board.apply_move(first, switch_turn=False)
    if all(die.used for die in board.dice):
        while len(board.moves) > initial:
            board.undo_last_move()
        return (first, (0, 0, 0))
    next_moves = list(set(board.get_valid_moves()) - {(0, 0, 0)})
    while len(board.moves) > initial:
        board.undo_last_move()
    if not next_moves:
        return (first, (0, 0, 0))
    return (first, random.choice(next_moves))


# -------------------------
# MONTE CARLO TARGET LABELING
# -------------------------

def compute_mc_targets(record, terminal_value, gamma=GAMMA):
    """
    Convert a game record into (encoded, value_target, aux_target) training
    samples using Monte Carlo returns.

    For each position t:
        G_t = r_t + gamma*r_{t+1} + ... + gamma^(T-t-1)*r_{T-1}
                  + gamma^(T-t) * terminal_value

    where terminal_value is +1.0 (win), -1.0 (loss), or 0.0 (draw).

    Targets clipped to [-1, 1] — should already be in range given small
    shaped rewards and a +-1 terminal, but clip as a safeguard.
    """
    T = len(record)
    if T == 0:
        return []

    samples = []
    G = terminal_value
    for t in reversed(range(T)):
        encoded, reward, aux_target = record[t]
        G = reward + gamma * G
        G_clipped = max(-1.0, min(1.0, G))
        samples.append((encoded, G_clipped, aux_target))

    samples.reverse()
    return samples


# -------------------------
# EVALUATION
# -------------------------

def play_eval_game(agent_white, agent_black, seed, heuristic_white=False,
                   heuristic_black=False, heuristic_agent=None):
    """Play one evaluation game. Returns (winner, margin, turns)."""
    random.seed(seed)
    board  = Board()
    turns  = 0
    consecutive_passes = 0

    while turns < MAX_TURNS:
        winner, margin = board.check_game_over()
        if winner:
            return winner, margin, turns

        current_player = board.current_player
        moves = board.get_valid_moves()
        if not moves:
            break

        if current_player == 'white':
            if heuristic_white:
                chosen = heuristic_agent.select_move_pair(moves, board, 'white')
            else:
                chosen = agent_white.select_move_pair_fast(moves, board, 'white')
        else:
            if heuristic_black:
                chosen = heuristic_agent.select_move_pair(moves, board, 'black')
            else:
                chosen = agent_black.select_move_pair_fast(moves, board, 'black')

        if chosen == ((0, 0, 0), (0, 0, 0)):
            consecutive_passes += 1
            if consecutive_passes >= 6:
                ws = len(board.white_saved)
                bs = len(board.black_saved)
                if ws > bs: return 'white', ws - bs, turns
                if bs > ws: return 'black', bs - ws, turns
                return None, 0, turns
        else:
            consecutive_passes = 0

        for move in chosen:
            if move != (0, 0, 0):
                board.apply_move(move, switch_turn=False)
        board.switch_turn()
        turns += 1

    ws = len(board.white_saved)
    bs = len(board.black_saved)
    if ws > bs: return 'white', ws - bs, turns
    if bs > ws: return 'black', bs - ws, turns
    return None, 0, turns


def evaluate_vs_opponent(challenger, opponent_agent, num_pairs, seed_offset,
                         heuristic=False, heuristic_agent=None, label=''):
    """
    Run num_pairs paired evaluation games (2 games per pair, sides swapped).
    challenger always plays as GNN agent.
    Returns (wins, total, avg_margin).
    """
    wins       = 0
    total      = 0
    margin_sum = 0.0

    for i in range(num_pairs):
        seed = seed_offset + i * 2

        # Early exit if promotion is mathematically impossible
        remaining = (num_pairs - i) * 2
        if total > 0 and (wins + remaining) / (total + remaining) < PROMOTION_WINRATE - 0.05:
            print(f"    Early exit: max possible win rate "
                  f"{(wins + remaining)/(total + remaining):.1%} < threshold")
            break

        # Game 1: challenger = white
        winner, margin, turns = play_eval_game(
            challenger, opponent_agent, seed,
            heuristic_black=heuristic, heuristic_agent=heuristic_agent)
        total += 1
        if winner == 'white':
            wins += 1; margin_sum += margin; result = 'Won by'
        elif winner == 'black':
            margin_sum -= margin; result = 'Lost by'
        else:
            result = 'Draw'
        print(f"    Game {i*2+1}: {result} {margin}, turns={turns}")

        # Game 2: challenger = black
        winner, margin, turns = play_eval_game(
            opponent_agent, challenger, seed + 1,
            heuristic_white=heuristic, heuristic_agent=heuristic_agent)
        total += 1
        if winner == 'black':
            wins += 1; margin_sum += margin; result = 'Won by'
        elif winner == 'white':
            margin_sum -= margin; result = 'Lost by'
        else:
            result = 'Draw'
        print(f"    Game {i*2+2}: {result} {margin}, turns={turns}")

    avg_margin = margin_sum / total if total > 0 else 0.0
    win_rate   = wins / total if total > 0 else 0.0
    if label:
        print(f"  vs {label}: {wins}/{total} ({win_rate:.1%}) | "
              f"avg margin {avg_margin:+.2f}")
    return wins, total, avg_margin


def _eval_vs_pool(challenger, frozen_pool, num_pairs, seed_offset, label=''):
    """Evaluate challenger against a random opponent drawn from frozen_pool."""
    wins       = 0
    total      = 0
    margin_sum = 0.0

    for i in range(num_pairs):
        seed = seed_offset + i * 2

        remaining = (num_pairs - i) * 2
        if total > 0 and (wins + remaining) / (total + remaining) < PROMOTION_WINRATE - 0.05:
            print(f"    Early exit: max possible win rate "
                  f"{(wins + remaining)/(total + remaining):.1%} < threshold")
            break

        opp_model = random.choice(frozen_pool)
        opp_agent = GNNAgent(model=opp_model)

        winner, margin, turns = play_eval_game(challenger, opp_agent, seed)
        total += 1
        if winner == 'white':
            wins += 1; margin_sum += margin; result = 'Won by'
        elif winner == 'black':
            margin_sum -= margin; result = 'Lost by'
        else:
            result = 'Draw'
        print(f"    Game {i*2+1} vs pool: {result} {margin}, turns={turns}")

        winner, margin, turns = play_eval_game(opp_agent, challenger, seed + 1)
        total += 1
        if winner == 'black':
            wins += 1; margin_sum += margin; result = 'Won by'
        elif winner == 'white':
            margin_sum -= margin; result = 'Lost by'
        else:
            result = 'Draw'
        print(f"    Game {i*2+2} vs pool: {result} {margin}, turns={turns}")

    avg_margin = margin_sum / total if total else 0.0
    win_rate   = wins / total if total else 0.0
    if label:
        print(f"  vs {label}: {wins}/{total} ({win_rate:.1%}) | "
              f"avg margin {avg_margin:+.2f}")
    return wins, total, avg_margin


# -------------------------
# ROLLING STATS
# -------------------------

class RollingStats:
    """Tracks rolling averages over a fixed window of generations."""
    def __init__(self, window=3):
        self.window    = window
        self.win_rates = collections.deque(maxlen=window)
        self.margins   = collections.deque(maxlen=window)

    def update(self, win_rate, margin):
        self.win_rates.append(win_rate)
        self.margins.append(margin)

    def avg_win_rate(self):
        return sum(self.win_rates) / len(self.win_rates) if self.win_rates else 0.0

    def avg_margin(self):
        return sum(self.margins) / len(self.margins) if self.margins else 0.0

    def full(self):
        return len(self.win_rates) == self.window


# -------------------------
# GAME STATS PRINTER
# -------------------------

def _print_game_stats(stats):
    turns    = stats['turns']
    times    = stats['times']
    avg_turns = sum(turns) / len(turns) if turns else 0
    avg_time  = sum(times) / len(times) if times else 0
    print(f"  [Games] white={stats['white_wins']} black={stats['black_wins']} "
          f"draws={stats['draws']} max_turns_hit={stats['max_turns_hit']}")
    print(f"  [Turns] avg={avg_turns:.1f} min={min(turns) if turns else 0} "
          f"max={max(turns) if turns else 0}")
    print(f"  [Speed] avg={avg_time:.1f}s/game | "
          f"positions_added={stats['positions']}")


# -------------------------
# SANITY CHECK
# -------------------------

def _sanity_check(model, encoder, criterion):
    """
    Verify model forward pass, output range, and gradient flow before training.
    Aborts with a clear message if nn.Tanh() is missing from the value head.
    """
    board = Board()
    enc   = encoder.encode(board, board.current_player)
    enc_d = {k: v.to(DEVICE) for k, v in enc.items()}

    model.train()
    val, aux = model.forward_with_aux([enc_d])
    assert val.shape == (1,), f"Value shape wrong: {val.shape}"
    assert aux.shape == (1,), f"Aux shape wrong: {aux.shape}"

    assert abs(val.item()) <= 1.0, (
        f"Value output {val.item():.4f} is outside [-1, 1]. "
        f"Did you add nn.Tanh() to the value head readout in network.py?")

    loss = criterion(val, torch.tensor([0.5], device=DEVICE))
    loss.backward()
    model.zero_grad()
    model.eval()

    print(f"  value={val.item():.4f} aux={aux.item():.4f}  (both should be in [-1, 1])")


# -------------------------
# MAIN
# -------------------------

def main():
    encoder         = BoardEncoder()
    heuristic_agent = HeuristicAgent()

    weights_path = SELFPLAY_WEIGHTS if os.path.exists(SELFPLAY_WEIGHTS) else DISTILL_WEIGHTS
    print(f"\nLoading weights from {weights_path}...")
    model = BoardGNN().to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    print(f"Loaded on {DEVICE} ({sum(p.numel() for p in model.parameters()):,} params)")

    # Distilled model kept frozen throughout as the collapse detection baseline
    distilled_model = load_model(DISTILL_WEIGHTS)
    distilled_model.eval()
    distilled_agent = GNNAgent(model=distilled_model)

    # Frozen pool starts with a copy of the distilled model
    frozen_pool = [copy.deepcopy(distilled_model)]
    frozen_pool[0].eval()

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY)
    criterion = nn.MSELoss()

    challenger_agent = GNNAgent(model=model)
    replay_buffer    = collections.deque(maxlen=BUFFER_SIZE)

    rolling_vs_heuristic = RollingStats(PROMOTION_ROLLING_GENS)
    rolling_vs_frozen    = RollingStats(PROMOTION_ROLLING_GENS)
    rolling_vs_distilled = RollingStats(PROMOTION_ROLLING_GENS)
    collapse_strikes     = 0
    best_frozen_margin   = -999.0

    generation = 0
    start_time = time.time()

    print("\nRunning sanity check...")
    _sanity_check(model, encoder, criterion)
    print("Sanity check passed.\n")

    try:
        while True:
            gen_start = time.time()
            model.eval()

            print(f"=== Generation {generation} ===")
            print(f"  [Data generation] {GAMES_PER_GEN} games...")

            game_stats = {'turns': [], 'draws': 0, 'max_turns_hit': 0,
                          'white_wins': 0, 'black_wins': 0,
                          'positions': 0, 'times': []}

            for g in range(GAMES_PER_GEN):
                t0              = time.time()
                seed            = random.randint(0, 2**31)
                use_heuristic   = (g % 2 == 0)
                training_player = random.choice(['white', 'black'])

                if use_heuristic:
                    record, winner, margin, terminal_value = play_game(
                        challenger_agent, encoder,
                        opponent_agent=None,
                        current_player_is_agent=training_player,
                        seed=seed,
                        heuristic_agent=heuristic_agent)
                else:
                    opp_model = random.choice(frozen_pool)
                    opp_agent = GNNAgent(model=opp_model)
                    record, winner, margin, terminal_value = play_game(
                        challenger_agent, encoder,
                        opponent_agent=opp_agent,
                        current_player_is_agent=training_player,
                        seed=seed)

                elapsed = time.time() - t0
                samples = compute_mc_targets(record, terminal_value)

                if samples:
                    tvals = [s[1] for s in samples]
                    print(f"    Game {g+1}: winner={winner} terminal={terminal_value:+.1f} "
                          f"turns={len(record)} "
                          f"MC mean={sum(tvals)/len(tvals):.3f} "
                          f"min={min(tvals):.3f} max={max(tvals):.3f}")

                replay_buffer.extend(samples)

                game_stats['turns'].append(len(record))
                game_stats['times'].append(elapsed)
                game_stats['positions'] += len(samples)
                if winner == 'white':   game_stats['white_wins'] += 1
                elif winner == 'black': game_stats['black_wins'] += 1
                else:                   game_stats['draws'] += 1
                if len(record) >= MAX_TURNS: game_stats['max_turns_hit'] += 1

            _print_game_stats(game_stats)

            if len(replay_buffer) < MIN_BUFFER:
                print(f"  [Training] Buffer too small "
                      f"({len(replay_buffer)}/{MIN_BUFFER}), skipping.")
                generation += 1
                continue

            # --- Training ---
            model.train()
            total_value_loss = 0.0
            total_aux_loss   = 0.0
            total_grad_norm  = 0.0

            REUSE_FACTOR = 4
            actual_steps = min(TRAINING_STEPS,
                               max(1, REUSE_FACTOR * len(replay_buffer) // BATCH_SIZE))

            for step in range(actual_steps):
                batch         = random.sample(replay_buffer, min(BATCH_SIZE, len(replay_buffer)))
                encoded_list  = [{k: v.to(DEVICE) for k, v in item[0].items()} for item in batch]
                value_targets = torch.tensor([item[1] for item in batch],
                                             dtype=torch.float32, device=DEVICE)
                aux_targets   = torch.tensor([item[2] for item in batch],
                                             dtype=torch.float32, device=DEVICE)

                optimizer.zero_grad()
                value_preds, aux_preds = model.forward_with_aux(encoded_list)
                value_loss = criterion(value_preds, value_targets)
                aux_loss   = criterion(aux_preds, aux_targets)
                loss       = value_loss + AUX_LOSS_WEIGHT * aux_loss
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5).item()
                optimizer.step()

                total_value_loss += value_loss.item()
                total_aux_loss   += aux_loss.item()
                total_grad_norm  += grad_norm

            avg_value_loss = total_value_loss / actual_steps
            avg_aux_loss   = total_aux_loss   / actual_steps
            avg_grad_norm  = total_grad_norm  / actual_steps

            with torch.no_grad():
                sample_encoded = [{k: v.to(DEVICE)
                                   for k, v in random.choice(replay_buffer)[0].items()}]
                sample_out = model(sample_encoded).abs().item()

            print(f"  [Training] steps={actual_steps} "
                  f"value_loss={avg_value_loss:.4f} aux_loss={avg_aux_loss:.4f} "
                  f"grad_norm={avg_grad_norm:.3f} mean_abs_output={sample_out:.3f} "
                  f"buffer={len(replay_buffer)} lr={optimizer.param_groups[0]['lr']:.2e}")

            # --- Evaluation ---
            model.eval()
            eval_seed = generation * 1000 + random.randint(0, 999)

            wins_h, total_h, margin_h = evaluate_vs_opponent(
                challenger_agent, None, EVAL_PAIRS, eval_seed,
                heuristic=True, heuristic_agent=heuristic_agent, label='heuristic')
            rolling_vs_heuristic.update(wins_h / total_h, margin_h)

            wins_f, total_f, margin_f = _eval_vs_pool(
                challenger_agent, frozen_pool, EVAL_PAIRS, eval_seed + 500,
                label='frozen pool')
            rolling_vs_frozen.update(wins_f / total_f, margin_f)

            wins_d, total_d, margin_d = evaluate_vs_opponent(
                challenger_agent, distilled_agent, EVAL_PAIRS, eval_seed + 1000,
                label='distilled')
            rolling_vs_distilled.update(wins_d / total_d, margin_d)

            print(f"  [Rolling {PROMOTION_ROLLING_GENS}-gen avg] "
                  f"vs_heuristic={rolling_vs_heuristic.avg_win_rate():.1%} "
                  f"margin={rolling_vs_heuristic.avg_margin():+.2f} | "
                  f"vs_frozen={rolling_vs_frozen.avg_win_rate():.1%} "
                  f"margin={rolling_vs_frozen.avg_margin():+.2f} | "
                  f"vs_distilled={rolling_vs_distilled.avg_win_rate():.1%} "
                  f"margin={rolling_vs_distilled.avg_margin():+.2f}")

            # --- Promotion ---
            if rolling_vs_frozen.full():
                avg_wr     = rolling_vs_frozen.avg_win_rate()
                avg_margin = rolling_vs_frozen.avg_margin()
                if avg_wr >= PROMOTION_WINRATE and avg_margin > best_frozen_margin:
                    print(f"  ⭐ PROMOTED! rolling win_rate={avg_wr:.1%} "
                          f"margin={avg_margin:+.2f} > best={best_frozen_margin:+.2f}")
                    best_frozen_margin = avg_margin
                    new_champion = BoardGNN().to(DEVICE)
                    new_champion.load_state_dict(copy.deepcopy(model.state_dict()))
                    new_champion.eval()
                    frozen_pool.append(new_champion)
                    if len(frozen_pool) > FROZEN_POOL_SIZE:
                        frozen_pool.pop(0)
                    save_model_and_buffer(model, SELFPLAY_WEIGHTS, replay_buffer)
                else:
                    print(f"  ✗ No promotion: win_rate={avg_wr:.1%} "
                          f"margin={avg_margin:+.2f} "
                          f"(need >={PROMOTION_WINRATE:.0%} and "
                          f"margin>{best_frozen_margin:+.2f})")

            # --- Collapse detection ---
            if rolling_vs_distilled.full():
                avg_margin_d = rolling_vs_distilled.avg_margin()
                if avg_margin_d < COLLAPSE_MARGIN_THRESHOLD:
                    collapse_strikes += 1
                    print(f"  ⚠️  Collapse warning {collapse_strikes}/{COLLAPSE_CONSECUTIVE}: "
                          f"avg margin vs distilled = {avg_margin_d:+.2f}")
                    if collapse_strikes >= COLLAPSE_CONSECUTIVE:
                        print(f"  🔴 COLLAPSE DETECTED — reloading weights")
                        if os.path.exists(SELFPLAY_WEIGHTS):
                            print(f"  ⬅️  Reverting to last champion: {SELFPLAY_WEIGHTS}")
                            model.load_state_dict(
                                torch.load(SELFPLAY_WEIGHTS, map_location=DEVICE))
                        else:
                            print(f"  ⬅️  No champion yet — reverting to distilled weights")
                            model.load_state_dict(
                                torch.load(DISTILL_WEIGHTS, map_location=DEVICE))
                        reset_model = BoardGNN().to(DEVICE)
                        reset_model.load_state_dict(copy.deepcopy(model.state_dict()))
                        reset_model.eval()
                        frozen_pool.clear()
                        frozen_pool.append(reset_model)
                        replay_buffer.clear()
                        rolling_vs_heuristic = RollingStats(PROMOTION_ROLLING_GENS)
                        rolling_vs_frozen    = RollingStats(PROMOTION_ROLLING_GENS)
                        rolling_vs_distilled = RollingStats(PROMOTION_ROLLING_GENS)
                        collapse_strikes     = 0
                        best_frozen_margin   = -999.0
                else:
                    collapse_strikes = 0

            # --- Checkpoint ---
            if generation % CHECKPOINT_INTERVAL == 0:
                ckpt_path = os.path.join(CHECKPOINT_DIR,
                                         f'gnn_s{SESSION}_g{generation}.pt')
                save_model_and_buffer(model, ckpt_path, replay_buffer)

            generation += 1
            scheduler.step()

            gen_time   = time.time() - gen_start
            total_time = time.time() - start_time
            print(f"  Gen {generation-1} done in {gen_time:.0f}s | "
                  f"Total {total_time/3600:.1f}h")

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving...")
        save_model_and_buffer(model, SELFPLAY_WEIGHTS, replay_buffer)
        print("Done.")


if __name__ == '__main__':
    main()
