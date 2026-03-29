"""
train_gnn.py — Self-play RL training for the GNN using TD(n) with shaped rewards.

Algorithm:
  - n-step TD: target for position t is r_t + γ*r_{t+1} + ... + γ^(n-1)*r_{t+n-1} + γ^n * V_target(s_{t+n})
  - V_target computed by a slowly-updated target network (hard copy every TARGET_UPDATE_INTERVAL steps)
  - Shaped rewards for saving pieces, captures, endgame entry
  - Auxiliary head predicts (my_saved - opp_saved) / 12, weight AUX_LOSS_WEIGHT

Data generation:
  - 50% vs frozen champion pool, 50% vs heuristic agent
  - 1-ply search (select_move_pair_fast) for speed
  - ε-greedy exploration

Evaluation (every generation):
  - vs heuristic agent: progress indicator
  - vs frozen champion pool: promotion gate
  - vs distilled baseline: collapse floor
  - Metrics: win rate + average margin, rolling 3-generation averages

Promotion:
  - win rate > PROMOTION_WINRATE AND avg margin > frozen champion margin
  - sustained over rolling 3-gen average
  - on promotion: add to frozen pool (max 3), drop oldest

Collapse detection:
  - rolling 3-gen avg margin vs distilled drops below COLLAPSE_MARGIN_THRESHOLD
  - for 2 consecutive evaluations
  - action: reload distilled weights, clear buffer, reset frozen pool

Usage:
    python3 train_gnn.py              # POC mode
    python3 train_gnn.py --full       # Full training (Colab)

Output:
    gnn_selfplay.pt                   — current best weights
    checkpoints/gnn_s{SESSION}_g{N}.pt
"""

import argparse
import collections
import copy
import math
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


CHECKPOINT_DIR  = os.environ.get('CHECKPOINT_DIR', 'checkpoints')
SESSION         = int(time.time())
BUFFER_FILE = os.path.join(CHECKPOINT_DIR, f'replay_buffer_s{SESSION}.pkl')

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -------------------------
# CONFIG
# -------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--full', action='store_true')
args = parser.parse_args()

if args.full:
    GAMES_PER_GEN        = 20
    EVAL_PAIRS           = 10      # paired games per opponent per eval
    BUFFER_SIZE          = 30_000
    MIN_BUFFER           = 1_000
    BATCH_SIZE           = 256
    TRAINING_STEPS       = 100
    LR                   = 3e-4
    CHECKPOINT_INTERVAL  = 5
    MAX_TURNS            = 300
else:
    GAMES_PER_GEN        = 6
    EVAL_PAIRS           = 4
    BUFFER_SIZE          = 3_000
    MIN_BUFFER           = 200
    BATCH_SIZE           = 64
    TRAINING_STEPS       = 20
    LR                   = 3e-4
    CHECKPOINT_INTERVAL  = 5
    MAX_TURNS            = 100

# Shared
N_STEPS                  = 5       # TD n-step
GAMMA                    = 0.99
EXPLORATION_RATE         = 0.10
TARGET_UPDATE_INTERVAL   = 50      # hard-copy target network every N training steps
LR_DECAY                 = 0.995   # per generation
MIN_LR                   = 1e-5

FROZEN_POOL_SIZE         = 3
PROMOTION_WINRATE        = 0.5
PROMOTION_ROLLING_GENS   = 3       # rolling average window for promotion
COLLAPSE_MARGIN_THRESHOLD = -1.5   # avg margin vs distilled below this = danger
COLLAPSE_CONSECUTIVE     = 3       # consecutive evals below threshold = collapse

DISTILL_WEIGHTS          = 'gnn_weights.pt'
SELFPLAY_WEIGHTS         = 'gnn_selfplay.pt'

# Shaped reward constants (in GNN output scale ~[-1, 1])
R_SAVE_BASE              = 0.05
R_SAVE_NUMBER_SCALE      = 0.005   # additional per piece number (max 0.03 for piece 6)
R_CAPTURE                = 0.02
R_ENDGAME                = 0.10

print(f"{'FULL' if args.full else 'POC'} mode | "
      f"games={GAMES_PER_GEN} eval_pairs={EVAL_PAIRS} "
      f"buffer={BUFFER_SIZE} min_buffer={MIN_BUFFER}")
print(f"Session {SESSION} | checkpoints -> {CHECKPOINT_DIR}/")


BUFFER_FILE = os.path.join(CHECKPOINT_DIR, f'buffer_s{SESSION}.pkl')

def save_model_and_buffer(model, path, buffer, checkpoint_dir=CHECKPOINT_DIR, session=SESSION):
    save_model(model, path)
    buffer_file = os.path.join(checkpoint_dir, f'buffer_s{session}.pkl')
    with open(buffer_file, 'wb') as f:
        pickle.dump(list(buffer), f)
    print(f"Saved replay buffer ({len(buffer)} positions) to {buffer_file}")

# -------------------------
# SHAPED REWARDS
# -------------------------

def compute_shaped_reward(board_before, board_after, current_player):
    """
    Compute shaped reward for one move transition, from current_player's perspective.
    Compares board state before and after the full move pair is applied.

    Rewards:
      +R_SAVE_BASE + R_SAVE_NUMBER_SCALE * piece.number  for each piece saved (ours)
      mirror negative                                     for each opponent piece saved
      +R_CAPTURE                                          for each opponent piece on home
      +/-R_ENDGAME                                        for entering/opponent entering endgame
    """
    reward = 0.0
    opponent = 'black' if current_player == 'white' else 'white'

    my_saved_before  = len(board_before.white_saved if current_player == 'white'
                           else board_before.black_saved)
    my_saved_after   = len(board_after.white_saved  if current_player == 'white'
                           else board_after.black_saved)
    opp_saved_before = len(board_before.black_saved if current_player == 'white'
                           else board_before.white_saved)
    opp_saved_after  = len(board_after.black_saved  if current_player == 'white'
                           else board_after.white_saved)

    # Pieces saved this turn (ours)
    newly_saved = my_saved_after - my_saved_before
    if newly_saved > 0:
        # We don't know which specific pieces were saved, so use average reward
        reward += newly_saved * R_SAVE_BASE
        # Approximate number bonus: average piece number in save rack
        my_save_rack = (board_after.white_saved if current_player == 'white'
                        else board_after.black_saved)
        # Use the most recently saved pieces (last `newly_saved` in rack)
        for piece in my_save_rack[-newly_saved:]:
            if piece.number <= 6:
                reward += R_SAVE_NUMBER_SCALE * piece.number

    # Opponent pieces saved (bad for us)
    opp_newly_saved = opp_saved_after - opp_saved_before
    if opp_newly_saved > 0:
        reward -= opp_newly_saved * R_SAVE_BASE
        opp_save_rack = (board_after.black_saved if current_player == 'white'
                         else board_after.white_saved)
        for piece in opp_save_rack[-opp_newly_saved:]:
            if piece.number <= 6:
                reward -= R_SAVE_NUMBER_SCALE * piece.number

    # Captures: opponent pieces on home tile
    opp_on_home_before = sum(1 for p in board_before.home_tile.pieces
                             if p.player == opponent)
    opp_on_home_after  = sum(1 for p in board_after.home_tile.pieces
                             if p.player == opponent)
    new_captures = opp_on_home_after - opp_on_home_before
    if new_captures > 0:
        reward += new_captures * R_CAPTURE

    # Endgame entry (one-time bonus)
    my_stage_before  = board_before.game_stages[current_player]
    my_stage_after   = board_after.game_stages[current_player]
    opp_stage_before = board_before.game_stages[opponent]
    opp_stage_after  = board_after.game_stages[opponent]

    if my_stage_before != 'endgame' and my_stage_after == 'endgame':
        reward += R_ENDGAME
    if opp_stage_before != 'endgame' and opp_stage_after == 'endgame':
        reward -= R_ENDGAME

    return reward


# -------------------------
# BOARD STATE SNAPSHOT
# -------------------------

def snapshot_board_state(board):
    """
    Lightweight snapshot of board state fields needed for reward computation.
    Returns a simple namespace — avoids deep-copying the full Board object.
    """
    class Snap:
        pass
    s = Snap()
    s.white_saved  = list(board.white_saved)
    s.black_saved  = list(board.black_saved)
    s.home_tile_pieces = list(board.home_tile.pieces)
    s.game_stages  = dict(board.game_stages)
    # Attach racks so compute_shaped_reward can use them
    s.white_saved_obj  = board.white_saved
    s.black_saved_obj  = board.black_saved
    return s


# -------------------------
# SELF-PLAY GAME
# -------------------------

def play_game(agent, encoder, opponent_agent, current_player_is_agent,
              seed, heuristic_agent=None):
    """
    Play one game. Returns a list of (encoded_position, shaped_reward, aux_target)
    tuples for each turn, plus the final (winner, margin).

    current_player_is_agent: which color the training agent plays.
      If None, agent plays both sides (pure self-play).
    opponent_agent: GNNAgent from frozen pool, or None if heuristic.
    heuristic_agent: HeuristicAgent instance, used when opponent_agent is None.

    The training agent always plays from its own perspective.
    We record positions + rewards only for the training agent's turns.
    """
    random.seed(seed)
    board  = Board()
    record = []   # (encoded, reward, aux_target) for training agent's turns
    turns  = 0
    consecutive_passes = 0

    # Track endgame entry to emit the one-time bonus only once
    endgame_entered = {'white': False, 'black': False}

    while turns < MAX_TURNS:
        winner, margin = board.check_game_over()
        if winner:
            return record, winner, margin

        current_player = board.current_player

        # Decide which agent moves
        is_training_agent_turn = (
            current_player_is_agent is None or
            current_player == current_player_is_agent
        )

        moves = board.get_valid_moves()
        if not moves:
            break

        # Snapshot before move (for reward computation)
        snap_before = snapshot_board_state(board)
        # Fix endgame_entered to match actual board state before this turn
        for p in ['white', 'black']:
            if board.game_stages[p] == 'endgame' and not endgame_entered[p]:
                endgame_entered[p] = True

        # Encode position (training agent's perspective if it's their turn,
        # otherwise we still encode for recording but won't store it)
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
            # Opponent's turn
            if opponent_agent is not None:
                chosen = opponent_agent.select_move_pair_fast(
                    moves, board, current_player)
            else:
                chosen = heuristic_agent.select_move_pair(
                    moves, board, current_player)

        if chosen == ((0, 0, 0), (0, 0, 0)):
            consecutive_passes += 1
            if consecutive_passes >= 6:
                ws = len(board.white_saved)
                bs = len(board.black_saved)
                if ws > bs: return record, 'white', ws - bs
                if bs > ws: return record, 'black', bs - ws
                return record, None, 0
        else:
            consecutive_passes = 0

        # Apply moves
        for move in chosen:
            if move != (0, 0, 0):
                board.apply_move(move, switch_turn=False)
        board.switch_turn()

        # Compute shaped reward for training agent's turn
        if is_training_agent_turn:
            snap_after = snapshot_board_state(board)
            # Patch snap objects so compute_shaped_reward can access rack lists
            snap_before.white_saved = snap_before.white_saved_obj
            snap_before.black_saved = snap_before.black_saved_obj
            snap_after.white_saved  = board.white_saved
            snap_after.black_saved  = board.black_saved
            snap_after.home_tile    = board.home_tile
            snap_before.home_tile   = type('T', (), {
                'pieces': snap_before.home_tile_pieces})()
            snap_after.game_stages  = board.game_stages

            # Suppress endgame bonus if already given
            if endgame_entered[current_player]:
                snap_before.game_stages[current_player] = 'endgame'
            if endgame_entered['white' if current_player == 'black' else 'black']:
                opp = 'white' if current_player == 'black' else 'white'
                snap_before.game_stages[opp] = 'endgame'

            reward = compute_shaped_reward(snap_before, snap_after, current_player)

            # Mark endgame as entered after reward computed
            for p in ['white', 'black']:
                if board.game_stages[p] == 'endgame':
                    endgame_entered[p] = True

            record.append((encoded_stored, reward, aux_target))

        turns += 1

    ws = len(board.white_saved)
    bs = len(board.black_saved)
    if ws > bs: return record, 'white', ws - bs
    if bs > ws: return record, 'black', bs - ws
    return record, None, 0


def _random_move_pair(moves, board):
    """Pick a random legal first move then a random legal second move."""
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
# N-STEP TD LABELING
# -------------------------

def compute_td_targets(record, target_model, encoder, gamma, n_steps):
    """
    Convert a game record into (encoded, value_target, aux_target) training samples
    using n-step TD.

    record: list of (encoded, shaped_reward, aux_target) from play_game()
    target_model: frozen copy of model used for bootstrapping V(s_{t+n})

    For each position t:
      target = sum_{k=0}^{n-1} gamma^k * r_{t+k}  +  gamma^n * V_target(s_{t+n})
    For terminal positions (within n steps of end): no bootstrap, just sum rewards.
    """
    T = len(record)
    if T == 0:
        return []

    # Gather all bootstrap positions in one batched forward pass
    # For position t, bootstrap from position t+n (if it exists)
    bootstrap_indices = []
    bootstrap_encoded = []
    for t in range(T):
        bootstrap_t = t + n_steps
        if bootstrap_t < T:
            bootstrap_indices.append((t, bootstrap_t))
            bootstrap_encoded.append(record[bootstrap_t][0])

    # Batch forward pass for bootstrap values
    bootstrap_values = {}
    if bootstrap_encoded:
        with torch.no_grad():
            encoded_dev = [{k: v.to(DEVICE) for k, v in e.items()}
                           for e in bootstrap_encoded]
            vals = target_model(encoded_dev)   # [N]
        for i, (t, bt) in enumerate(bootstrap_indices):
            bootstrap_values[t] = vals[i].item()

    # Build targets
    samples = []
    for t in range(T):
        encoded, _, aux_target = record[t]

        # Accumulate discounted rewards over n steps
        discounted_reward = 0.0
        for k in range(n_steps):
            if t + k < T:
                _, r, _ = record[t + k]
                discounted_reward += (gamma ** k) * r

        # Bootstrap
        if t in bootstrap_values:
            target = discounted_reward + (gamma ** n_steps) * bootstrap_values[t]
        else:
            target = discounted_reward

        # Clip to reasonable range
        target = max(-2.0, min(2.0, target))

        samples.append((encoded, target, aux_target))

    return samples


# -------------------------
# EVALUATION GAME
# -------------------------

def play_eval_game(agent_white, agent_black, seed, heuristic_white=False,
                   heuristic_black=False, heuristic_agent=None):
    """
    Play one evaluation game. Returns (winner, margin, turns).
    (Uses 2-ply search (select_move_pair) for GNN agents for stronger play.) -- changed this to 1-ply for speed
    """
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
    wins   = 0
    total  = 0
    margin_sum = 0.0

    for i in range(num_pairs):
        seed = seed_offset + i * 2

        # Early exit: if mathematically impossible to reach PROMOTION_WINRATE
        # with remaining games, stop early
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
            wins += 1
            margin_sum += margin
            result = 'Won by'
        elif winner == 'black':
            margin_sum -= margin
            result = 'Lost by'
        else:
            result = 'Draw'
        print(f"    Game {i*2+1}: {result} {margin}, turns={turns}")

        # Game 2: challenger = black
        winner, margin, turns = play_eval_game(
            opponent_agent, challenger, seed + 1,
            heuristic_white=heuristic, heuristic_agent=heuristic_agent)
        total += 1
        if winner == 'black':
            wins += 1
            margin_sum += margin
            result = 'Won by'
        elif winner == 'white':
            margin_sum -= margin
            result = 'Lost by'
        else:
            result = 'Draw'
        print(f"    Game {i*2+2}: {result} {margin}, turns={turns}")

    avg_margin = margin_sum / total if total > 0 else 0.0
    win_rate   = wins / total if total > 0 else 0.0
    if label:
        print(f"  vs {label}: {wins}/{total} ({win_rate:.1%}) | "
              f"avg margin {avg_margin:+.2f}")
    return wins, total, avg_margin
# -------------------------
# ROLLING STATS
# -------------------------

class RollingStats:
    """Tracks rolling averages over a fixed window."""
    def __init__(self, window=3):
        self.window = window
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
# MAIN
# -------------------------

def main():
    encoder         = BoardEncoder()
    heuristic_agent = HeuristicAgent()

    # Load starting weights
    weights_path = SELFPLAY_WEIGHTS if os.path.exists(SELFPLAY_WEIGHTS) else DISTILL_WEIGHTS
    print(f"\nLoading weights from {weights_path}...")
    model = BoardGNN().to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    print(f"Loaded on {DEVICE} ({sum(p.numel() for p in model.parameters()):,} params)")

    target_model = BoardGNN().to(DEVICE)
    target_model.load_state_dict(copy.deepcopy(model.state_dict()))
    target_model.eval()

    distilled_model = load_model(DISTILL_WEIGHTS)
    distilled_model.eval()
    distilled_agent = GNNAgent(model=distilled_model)

    frozen_pool = [copy.deepcopy(distilled_model)]
    frozen_pool[0].eval()

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY)
    criterion = nn.MSELoss()

    challenger_agent = GNNAgent(model=model)

    replay_buffer = collections.deque(maxlen=BUFFER_SIZE)

    rolling_vs_heuristic  = RollingStats(PROMOTION_ROLLING_GENS)
    rolling_vs_frozen     = RollingStats(PROMOTION_ROLLING_GENS)
    rolling_vs_distilled  = RollingStats(PROMOTION_ROLLING_GENS)
    collapse_strikes      = 0
    best_frozen_margin    = -999.0

    generation   = 0
    total_steps  = 0
    start_time   = time.time()

    print("\nRunning sanity check...")
    _sanity_check(model, target_model, encoder, criterion)
    print("Sanity check passed.\n")

    try:
        while True:
            gen_start = time.time()
            model.eval()

            print(f"=== Generation {generation} ===")
            print(f"  [Data generation] {GAMES_PER_GEN} games...")

            game_stats = {'turns': [], 'draws': 0, 'max_turns_hit': 0,
                          'white_wins': 0, 'black_wins': 0,
                          'positions': 0, 'times': [],
                          'final_saved_my': [], 'final_saved_opp': []}

            for g in range(GAMES_PER_GEN):
                t0 = time.time()
                seed = random.randint(0, 2**31)
                use_heuristic = (g % 2 == 0)
                training_player = random.choice(['white', 'black'])

                if use_heuristic:
                    record, winner, margin = play_game(
                        challenger_agent, encoder,
                        opponent_agent=None,
                        current_player_is_agent=training_player,
                        seed=seed,
                        heuristic_agent=heuristic_agent)
                else:
                    opp_model = random.choice(frozen_pool)
                    opp_agent = GNNAgent(model=opp_model)
                    record, winner, margin = play_game(
                        challenger_agent, encoder,
                        opponent_agent=opp_agent,
                        current_player_is_agent=training_player,
                        seed=seed,
                        heuristic_agent=None)

                elapsed = time.time() - t0

                samples = compute_td_targets(record, target_model, encoder, GAMMA, N_STEPS)
                replay_buffer.extend(samples)

                num_turns = len(record)
                game_stats['turns'].append(num_turns)
                game_stats['times'].append(elapsed)
                game_stats['positions'] += len(samples)
                if winner == 'white':   game_stats['white_wins'] += 1
                elif winner == 'black': game_stats['black_wins'] += 1
                else:                   game_stats['draws'] += 1
                if num_turns >= MAX_TURNS: game_stats['max_turns_hit'] += 1

            _print_game_stats(game_stats, GAMES_PER_GEN)

            if len(replay_buffer) < MIN_BUFFER:
                print(f"  [Training] Buffer too small ({len(replay_buffer)}/{MIN_BUFFER}), skipping.")
                generation += 1
                scheduler.step()
                continue

            model.train()
            total_value_loss = 0.0
            total_aux_loss   = 0.0
            total_grad_norm  = 0.0

            for step in range(TRAINING_STEPS):
                batch = random.sample(replay_buffer, min(BATCH_SIZE, len(replay_buffer)))
                encoded_list  = [{k: v.to(DEVICE) for k, v in item[0].items()} for item in batch]
                value_targets = torch.tensor([item[1] for item in batch], dtype=torch.float32, device=DEVICE)
                aux_targets   = torch.tensor([item[2] for item in batch], dtype=torch.float32, device=DEVICE)

                optimizer.zero_grad()
                value_preds, aux_preds = model.forward_with_aux(encoded_list)
                value_loss = criterion(value_preds, value_targets)
                aux_loss   = criterion(aux_preds, aux_targets)
                loss       = value_loss + AUX_LOSS_WEIGHT * aux_loss
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
                optimizer.step()

                total_value_loss += value_loss.item()
                total_aux_loss   += aux_loss.item()
                total_grad_norm  += grad_norm
                total_steps      += 1

                if total_steps % TARGET_UPDATE_INTERVAL == 0:
                    target_model.load_state_dict(copy.deepcopy(model.state_dict()))
                    target_model.eval()

            avg_value_loss = total_value_loss / TRAINING_STEPS
            avg_aux_loss   = total_aux_loss   / TRAINING_STEPS
            avg_grad_norm  = total_grad_norm  / TRAINING_STEPS

            with torch.no_grad():
                sample_encoded = [{k: v.to(DEVICE) for k, v in random.choice(replay_buffer)[0].items()}]
                sample_out = model(sample_encoded).abs().item()

            print(f"  [Training] value_loss={avg_value_loss:.4f} aux_loss={avg_aux_loss:.4f} "
                  f"grad_norm={avg_grad_norm:.3f} mean_abs_output={sample_out:.3f} "
                  f"buffer={len(replay_buffer)} lr={optimizer.param_groups[0]['lr']:.2e}")

            model.eval()

            eval_seed = generation * 1000 + random.randint(0, 999)
            wins_h, total_h, margin_h = evaluate_vs_opponent(
                challenger_agent, None, EVAL_PAIRS, eval_seed,
                heuristic=True, heuristic_agent=heuristic_agent, label='heuristic')
            rolling_vs_heuristic.update(wins_h / total_h, margin_h)

            wins_f, total_f, margin_f = _eval_vs_pool(
                challenger_agent, frozen_pool, EVAL_PAIRS, eval_seed + 500, label='frozen pool')
            rolling_vs_frozen.update(wins_f / total_f, margin_f)

            wins_d, total_d, margin_d = evaluate_vs_opponent(
                challenger_agent, distilled_agent, EVAL_PAIRS, eval_seed + 1000, label='distilled')
            rolling_vs_distilled.update(wins_d / total_d, margin_d)

            print(f"  [Rolling {PROMOTION_ROLLING_GENS}-gen avg] vs_heuristic={rolling_vs_heuristic.avg_win_rate():.1%} "
                  f"margin={rolling_vs_heuristic.avg_margin():+.2f} | vs_frozen={rolling_vs_frozen.avg_win_rate():.1%} "
                  f"margin={rolling_vs_frozen.avg_margin():+.2f} | vs_distilled={rolling_vs_distilled.avg_win_rate():.1%} "
                  f"margin={rolling_vs_distilled.avg_margin():+.2f}")

            promoted = False
            if rolling_vs_frozen.full():
                avg_wr = rolling_vs_frozen.avg_win_rate()
                avg_margin = rolling_vs_frozen.avg_margin()
                if avg_wr >= PROMOTION_WINRATE and avg_margin > best_frozen_margin:
                    print(f"  ⭐ PROMOTED! rolling win_rate={avg_wr:.1%} margin={avg_margin:+.2f} > best={best_frozen_margin:+.2f}")
                    best_frozen_margin = avg_margin
                    new_champion = BoardGNN().to(DEVICE)
                    new_champion.load_state_dict(copy.deepcopy(model.state_dict()))
                    new_champion.eval()
                    frozen_pool.append(new_champion)
                    if len(frozen_pool) > FROZEN_POOL_SIZE:
                        frozen_pool.pop(0)
                    save_model_and_buffer(model, SELFPLAY_WEIGHTS, replay_buffer)
                    promoted = True
                else:
                    print(f"  ✗ No promotion: win_rate={avg_wr:.1%} margin={avg_margin:+.2f} "
                          f"(need >{PROMOTION_WINRATE:.0%} and margin>{best_frozen_margin:+.2f})")

            if rolling_vs_distilled.full():
                avg_margin_d = rolling_vs_distilled.avg_margin()
                if avg_margin_d < COLLAPSE_MARGIN_THRESHOLD:
                    collapse_strikes += 1
                    print(f"  ⚠️  Collapse warning {collapse_strikes}/{COLLAPSE_CONSECUTIVE}: avg margin vs distilled = {avg_margin_d:+.2f}")
                    if collapse_strikes >= COLLAPSE_CONSECUTIVE:
                        print(f"  🔴 COLLAPSE DETECTED — reloading recent champion or distilled weights")
                        if os.path.exists(SELFPLAY_WEIGHTS):
                            print(f"  ⬅️ Reverting to last champion: {SELFPLAY_WEIGHTS}")
                            model.load_state_dict(torch.load(SELFPLAY_WEIGHTS, map_location=DEVICE))
                        else:
                            print(f"  ⬅️ No champion yet, reverting to distilled weights")
                            model.load_state_dict(torch.load(DISTILL_WEIGHTS, map_location=DEVICE))
                        target_model.load_state_dict(copy.deepcopy(model.state_dict()))
                        frozen_pool.clear()
                        frozen_pool.append(copy.deepcopy(model))
                        replay_buffer.clear()
                        rolling_vs_heuristic  = RollingStats(PROMOTION_ROLLING_GENS)
                        rolling_vs_frozen     = RollingStats(PROMOTION_ROLLING_GENS)
                        rolling_vs_distilled  = RollingStats(PROMOTION_ROLLING_GENS)
                        collapse_strikes      = 0
                        best_frozen_margin    = -999.0
                else:
                    collapse_strikes = 0  # reset on recovery

            if generation % CHECKPOINT_INTERVAL == 0:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f'gnn_s{SESSION}_g{generation}.pt')
                save_model_and_buffer(model, ckpt_path, replay_buffer)

            generation += 1
            scheduler.step()

            gen_time   = time.time() - gen_start
            total_time = time.time() - start_time
            print(f"  Gen {generation-1} done in {gen_time:.0f}s | Total {total_time/3600:.1f}h")

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model and buffer...")
        save_model_and_buffer(model, SELFPLAY_WEIGHTS, replay_buffer)
        print("Done.")

# -------------------------
# POOL EVALUATION
# -------------------------

def _eval_vs_pool(challenger, frozen_pool, num_pairs, seed_offset, label=''):
    wins   = 0
    total  = 0
    margin_sum = 0.0

    for i in range(num_pairs):
        seed = seed_offset + i * 2

        # Early exit
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
# GAME STATS PRINTER
# -------------------------

def _print_game_stats(stats, n_games):
    turns     = stats['turns']
    times     = stats['times']
    avg_turns = sum(turns) / len(turns) if turns else 0
    min_turns = min(turns) if turns else 0
    max_turns = max(turns) if turns else 0
    avg_time  = sum(times) / len(times) if times else 0

    print(f"  [Games] white={stats['white_wins']} black={stats['black_wins']} "
          f"draws={stats['draws']} max_turns_hit={stats['max_turns_hit']}")
    print(f"  [Turns] avg={avg_turns:.1f} min={min_turns} max={max_turns}")
    print(f"  [Speed] avg={avg_time:.1f}s/game | "
          f"positions_added={stats['positions']}")


# -------------------------
# SANITY CHECK
# -------------------------

def _sanity_check(model, target_model, encoder, criterion):
    board = Board()
    enc   = encoder.encode(board, board.current_player)
    enc_d = {k: v.to(DEVICE) for k, v in enc.items()}

    # Forward pass
    model.train()
    val, aux = model.forward_with_aux([enc_d])
    assert val.shape == (1,), f"Value shape wrong: {val.shape}"
    assert aux.shape == (1,), f"Aux shape wrong: {aux.shape}"

    # Backward pass
    loss = criterion(val, torch.tensor([0.5], device=DEVICE))
    loss.backward()
    model.zero_grad()
    model.eval()

    # Target network
    with torch.no_grad():
        t_val = target_model([enc_d])
    assert t_val.shape == (1,), f"Target model shape wrong: {t_val.shape}"

    print(f"  value={val.item():.4f} aux={aux.item():.4f} "
          f"target_val={t_val.item():.4f}")


if __name__ == '__main__':
    main()