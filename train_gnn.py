"""
train_gnn.py — Self-play training for the GNN.

Starts from the distilled GNN (gnn_weights.pt) and improves via self-play.
Uses paired games with shared seeds (same dice) to reduce luck variance.
Labels positions with discounted game outcomes.
Evaluates periodically and promotes if statistically better.

Usage:
    python3 train_gnn.py              # POC mode (small numbers, local)
    python3 train_gnn.py --full       # Full training (for Colab)

Output:
    gnn_selfplay.pt                   — current best self-play weights
    gnn_selfplay_s{SESSION}_gen{N}.pt — Drive checkpoint every CHECKPOINT_INTERVAL gens
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

from game import Board
from agent_gnn import GNNAgent
from network import BoardEncoder, BoardGNN, collate_batch, save_model, load_model, DEVICE

# Drive checkpoint directory — set by Colab notebook via env var
CHECKPOINT_DIR = os.environ.get('CHECKPOINT_DIR', '.')

# Unique session ID so checkpoints from different runs never collide
SESSION = int(time.time())


# -------------------------
# CONFIG  (POC vs full)
# -------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--full', action='store_true', help='Full training mode for Colab')
args = parser.parse_args()

if args.full:
    GAMES_PER_EVAL      = 60
    EVAL_PAIRS          = 50
    BUFFER_SIZE         = 50_000
    MIN_BUFFER          = 20_000
    BATCH_SIZE          = 512
    TRAINING_STEPS      = 300       
    LR                  = 5e-5      
    LR_DECAY            = 0.99
    CHECKPOINT_INTERVAL = 5
    MAX_TURNS           = 150
else:
    GAMES_PER_EVAL      = 10
    EVAL_PAIRS          = 5
    BUFFER_SIZE         = 2_000
    MIN_BUFFER          = 200
    BATCH_SIZE          = 128
    TRAINING_STEPS      = 50
    LR                  = 1e-5
    LR_DECAY            = 0.99
    CHECKPOINT_INTERVAL = 5
    MAX_TURNS           = 50

# Shared
GAMMA               = 0.99          # outcome discount per turn
OUTCOME_SCALE       = 100.0         # keeps self-play labels in ~[-0.6, +0.6]
SCORE_SCALE         = 1000.0        # must match agent_gnn.py and train_distill.py
PROMOTION_WINRATE   = 0.52
PROMOTION_PVALUE    = 0.05
COLLAPSE_PVALUE     = 0.50          # warn if challenger can't beat distilled baseline
EXPLORATION_RATE    = 0.08          # fraction of moves chosen randomly during self-play

DISTILL_WEIGHTS     = 'gnn_weights.pt'
SELFPLAY_WEIGHTS    = 'gnn_selfplay.pt'

print(f"{'FULL' if args.full else 'POC'} mode: "
      f"games_per_eval={GAMES_PER_EVAL}, eval_pairs={EVAL_PAIRS}, "
      f"buffer={BUFFER_SIZE}, min_buffer={MIN_BUFFER}")
print(f"Session ID: {SESSION}  (checkpoints will be named *_s{SESSION}_*)")


# -------------------------
# SELF-PLAY GAME
# -------------------------

def play_selfplay_game(agent, encoder, seed):
    """
    Play one game with agent as both players.
    Returns (record, winner, score) where:
      record  — list of (encoded_position, current_player, turn_number)
      winner  — 'white', 'black', or None
      score   — margin (unsaved opponent pieces) or 0
    board.current_player is 'white' or 'black'.
    """
    random.seed(seed)
    board  = Board()
    record = []
    turns  = 0
    consecutive_passes = 0

    while turns < MAX_TURNS:
        winner, score = board.check_game_over()
        if winner:
            return record, winner, score

        current_player = board.current_player
        encoded = encoder.encode(board, current_player)
        # Clone tensors so subsequent board mutations don't corrupt stored state
        encoded_stored = {k: v.clone() for k, v in encoded.items()}
        record.append((encoded_stored, current_player, turns))

        moves = board.get_valid_moves()
        if not moves:
            break

        # Exploration: occasionally pick a random move pair
        if random.random() < EXPLORATION_RATE:
            chosen = _random_move_pair(moves, board)
        else:
            chosen = agent.select_move_pair_fast(moves, board, current_player)   # use 1-ply search for speed

        if chosen == ((0, 0, 0), (0, 0, 0)):
            consecutive_passes += 1
            if consecutive_passes >= 6:
                # Break deadlock by saved piece count
                ws = len(board.white_saved)
                bs = len(board.black_saved)
                if ws > bs: return record, 'white', ws - bs
                if bs > ws: return record, 'black', bs - ws
                return record, None, 0
        else:
            consecutive_passes = 0

        for move in chosen:
            if move != (0, 0, 0):
                board.apply_move(move, switch_turn=False)
        board.switch_turn()
        turns += 1

    # Hit MAX_TURNS — tiebreak by saved pieces
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
# LABEL POSITIONS
# -------------------------

def label_positions(record, winner, outcome_score):
    """
    Assign discounted outcome labels to recorded positions.
    Label is from the current player's perspective at each position.
    Scaled to match distillation label range (~[-0.6, +0.6]).
    """
    labeled = []
    total_turns = len(record)

    for encoded, player, turn in record:
        turns_remaining = (total_turns - turn) / 2.0
        discount = GAMMA ** turns_remaining

        if winner is None:
            score = 0.0
        elif winner == player:
            score = float(outcome_score)
        else:
            score = -float(outcome_score)

        label = (score * OUTCOME_SCALE * discount) / SCORE_SCALE
        label = max(-1.0, min(1.0, label))
        labeled.append((encoded, label))

    return labeled


# -------------------------
# EVALUATION
# -------------------------

def binomial_p_value(successes, trials, target_p=0.55):
    """P(X >= successes) if true win rate is target_p. Used for promotion gate."""
    p_val = 0.0
    for k in range(successes, trials + 1):
        p_val += math.comb(trials, k) * (target_p ** k) * ((1 - target_p) ** (trials - k))
    return p_val


def evaluate_vs_agent(challenger_agent, opponent_agent, num_pairs, label=''):
    """
    Play num_pairs paired games (challenger as white + challenger as black).
    Uses early exit if promotion is mathematically impossible or statistically unlikely.
    board.current_player is 'white' or 'black'.
    Returns (wins, total, margins).
    """
    wins    = 0
    total   = 0
    margins = []

    for i in range(num_pairs):
        seed = random.randint(0, 1_000_000)

        # Game 1: challenger = white
        result1, score1, turns1 = play_eval_game(challenger_agent, opponent_agent, seed)
        wins  += 1 if result1 == 'white' else 0
        total += 1
        margins.append(score1 if result1 == 'white' else -score1)

        # Game 2: challenger = black
        result2, score2, turns2 = play_eval_game(opponent_agent, challenger_agent, seed)
        wins  += 1 if result2 == 'black' else 0
        total += 1
        margins.append(score2 if result2 == 'black' else -score2)

        # Hard math early exit
        remaining = (num_pairs * 2) - total
        if wins + remaining < num_pairs * 2 * PROMOTION_WINRATE:
            print(f"    [Early Exit] pair {i+1}/{num_pairs}  {wins}/{total}. "
                  f"Impossible to reach {PROMOTION_WINRATE:.0%}.")
            return wins, total, margins

        # Early success exit
        if total >= 20:
            p_val_success = binomial_p_value(wins, total, target_p=PROMOTION_WINRATE)
            if p_val_success <= PROMOTION_PVALUE and wins / total >= PROMOTION_WINRATE:
                print(f"    [Early Promote] pair {i+1}/{num_pairs}  {wins}/{total} (p={p_val_success:.3f}). Certain to promote.")
                return wins, total, margins

        # Statistical early exit
        if total >= 20:
            p_val = binomial_p_value(wins, total, target_p=PROMOTION_WINRATE)
            if p_val > 0.50:
                print(f"    [Stat Exit] pair {i+1}/{num_pairs}  {wins}/{total} "
                      f"(p={p_val:.3f}). Unlikely to promote.")
                return wins, total, margins

        print(f"    pair {i+1}/{num_pairs}  {wins}/{total} ({wins/total:.0%})  "
              f"turns={turns1},{turns2}  margin={score1 if result1=='white' else -score1:+d},{score2 if result2=='black' else -score2:+d}")

    return wins, total, margins


def play_eval_game(white_agent, black_agent, seed):
    """
    Play one evaluation game. No exploration, no recording.
    white_agent plays as 'white', black_agent plays as 'black'.
    Returns (winner, score, turns) where winner is 'white', 'black', or None.
    """
    random.seed(seed)
    board = Board()
    turns = 0
    consecutive_passes = 0

    while turns < MAX_TURNS:
        winner, score = board.check_game_over()
        if winner:
            return winner, score, turns

        current_player = board.current_player
        agent  = white_agent if current_player == 'white' else black_agent
        moves  = board.get_valid_moves()
        if not moves:
            break

        chosen = agent.select_move_pair_fast(moves, board, current_player)

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


# -------------------------
# MAIN TRAINING
# -------------------------

def load_distill_data(path):
    if os.path.exists(path):
        return torch.load(path)
    else:
        # Fallback: if file is missing, we can't stratify. 
        # You might want to raise an error here.
        print(f"⚠️ Warning: {path} not found. Training will lack an anchor!")
        return []
    
def main():
    encoder = BoardEncoder()

    # 1. Load starting weights — selfplay if exists, else distilled
    weights_path = SELFPLAY_WEIGHTS if os.path.exists(SELFPLAY_WEIGHTS) else DISTILL_WEIGHTS
    print(f"\nLoading weights from {weights_path}...")
    model = BoardGNN().to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.train()
    print(f"Model loaded on {DEVICE} ({sum(p.numel() for p in model.parameters()):,} params)")

    # 2. Load the Anchor Dataset (Gold Standard)
    # This prevents the model from forgetting how to play "rationally"
    distill_buffer = load_distill_data('distill_data.pt')
    if not distill_buffer:
        print("❌ CRITICAL: distill_data.pt is empty or missing. Stratification disabled.")
    else:
        print(f"Loaded {len(distill_buffer)} anchor positions for stratification.")

    # Frozen opponent — starts as same weights, updated only on promotion
    frozen_model = BoardGNN().to(DEVICE)
    frozen_model.load_state_dict(copy.deepcopy(model.state_dict()))
    frozen_model.eval()

    # Distilled baseline — never updated, used for collapse detection
    distilled_model = load_model(DISTILL_WEIGHTS)
    distilled_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY)
    criterion = nn.MSELoss()

    # Agents wrapping the live models
    challenger_agent = GNNAgent(model=model)
    frozen_agent     = GNNAgent(model=frozen_model)
    distilled_agent  = GNNAgent(model=distilled_model)

    # Replay buffer for Self-Play
    replay_buffer    = collections.deque(maxlen=BUFFER_SIZE)
    buffer_disk_path = os.path.join(CHECKPOINT_DIR, 'replay_buffer_latest.pt')

    if os.path.exists(buffer_disk_path):
        try:
            loaded = torch.load(buffer_disk_path)
            replay_buffer.extend(loaded)
            print(f"Resumed buffer with {len(replay_buffer)} positions.")
        except Exception as e:
            print(f"Buffer corrupted, starting fresh: {e}")
            os.remove(buffer_disk_path)

    print("\nRunning sanity check...")
    _sanity_check(model, encoder, criterion)
    print("Sanity check passed.\n")

    generation = 0
    start      = time.time()

    try:
        while True:
            gen_start     = time.time()
            gen_positions = 0
            print(f"Generation {generation}:")

            # ---- A. SELF-PLAY ----
            model.eval()
            for g in range(GAMES_PER_EVAL):
                seed = random.randint(0, 2**31)
                record, winner, score = play_selfplay_game(challenger_agent, encoder, seed)
                labeled = label_positions(record, winner, score)
                replay_buffer.extend(labeled)
                gen_positions += len(labeled)
                
                if (g + 1) % 5 == 0 or g == GAMES_PER_EVAL - 1:
                    print(f"  game {g+1}/{GAMES_PER_EVAL} buffer={len(replay_buffer)}")

            # ---- B. STRATIFIED TRAINING ----
            if len(replay_buffer) < MIN_BUFFER:
                print(f"  Buffer too small, skipping training.")
                generation += 1
                continue

            model.train()
            total_loss = 0.0
            
            # Ratio: 70% Self-Play, 30% Gold Standard
            SP_BATCH_SIZE = int(BATCH_SIZE * 0.7)
            DS_BATCH_SIZE = BATCH_SIZE - SP_BATCH_SIZE

            for _ in range(TRAINING_STEPS):
                # Sample from both sources
                batch_sp = random.sample(replay_buffer, SP_BATCH_SIZE)
                
                # If distill_buffer is missing, fallback to 100% self-play
                if distill_buffer:
                    batch_ds = random.sample(distill_buffer, DS_BATCH_SIZE)
                    batch = batch_sp + batch_ds
                else:
                    batch = random.sample(replay_buffer, BATCH_SIZE)
                
                random.shuffle(batch)
                
                encoded_list = [item[0] for item in batch]
                targets      = torch.tensor([item[1] for item in batch],
                                            dtype=torch.float32, device=DEVICE)
                
                optimizer.zero_grad()
                outputs = model(encoded_list)
                loss    = criterion(outputs.squeeze(), targets)
                loss.backward()
                
                # Gradient clipping to prevent collapse from volatile self-play
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()

            print(f"  Training loss: {total_loss / TRAINING_STEPS:.4f}")

            # ---- C. EVALUATION ----
            model.eval()
            print(f"  Evaluating vs frozen champion...")
            wins, total, margins = evaluate_vs_agent(challenger_agent, frozen_agent, EVAL_PAIRS)
            win_rate = wins / total if total else 0
            p_value  = binomial_p_value(wins, total, target_p=PROMOTION_WINRATE)

            # ---- D. PROMOTION ----
            promoted = (win_rate >= PROMOTION_WINRATE and p_value <= PROMOTION_PVALUE)

            if promoted:
                # Secondary check: Must still beat the distilled baseline
                wins_d, total_d, _ = evaluate_vs_agent(challenger_agent, distilled_agent, EVAL_PAIRS)
                if (wins_d / total_d) < 0.45: # Strict floor
                    print(f"  ⚠️  Promotion BLOCKED: Failed distilled baseline.")
                    promoted = False

            if promoted:
                print(f"  ⭐ PROMOTED! Winrate: {win_rate:.1%}")
                frozen_model.load_state_dict(copy.deepcopy(model.state_dict()))
                save_model(model, SELFPLAY_WEIGHTS)
            
            # Checkpoint and Scheduler
            if generation % CHECKPOINT_INTERVAL == 0:
                save_model(model, os.path.join(CHECKPOINT_DIR, f'gen_{generation}.pt'))
            
            scheduler.step()
            generation += 1

    except KeyboardInterrupt:
        print("\nStopping...")
        save_model(model, SELFPLAY_WEIGHTS)

def _sanity_check(model, encoder, criterion):
    """
    Quick forward + backward pass to catch bugs before the main loop.
    Verifies that training step works end-to-end with the correct data format.
    """
    board = Board()
    enc  = encoder.encode(board, board.current_player)
    fake_batch   = [(enc, 0.5)] * 4
    encoded_list = [item[0] for item in fake_batch]
    targets      = torch.tensor([item[1] for item in fake_batch],
                                dtype=torch.float32, device=DEVICE)
    model.train()
    out  = model(encoded_list)
    loss = criterion(out.squeeze(), targets)
    loss.backward()
    model.zero_grad()
    model.eval()
    print(f"  forward+backward OK  loss={loss.item():.4f}  output_shape={out.shape}")


if __name__ == '__main__':
    main()