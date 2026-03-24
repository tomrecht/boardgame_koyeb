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
    gnn_selfplay_genN.pt              — checkpoint every CHECKPOINT_INTERVAL gens
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
# Falls back to current directory if not set
CHECKPOINT_DIR = os.environ.get('CHECKPOINT_DIR', '.')


# -------------------------
# CONFIG  (POC vs full)
# -------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--full', action='store_true', help='Full training mode for Colab')
args = parser.parse_args()

if args.full:
    GAMES_PER_EVAL      = 60
    EVAL_PAIRS          = 30
    BUFFER_SIZE         = 100000
    MIN_BUFFER          = 4000
    BATCH_SIZE          = 512
    TRAINING_STEPS      = 200 # per gen
    LR                  = 1e-4
    LR_DECAY            = 0.99
    GAMMA               = 0.98 # outcome discount
    CHECKPOINT_INTERVAL = 1
    MAX_TURNS           = 150
    print(f"FULL mode: games_per_eval={GAMES_PER_EVAL}, eval_pairs={EVAL_PAIRS}, buffer={BUFFER_SIZE}, min_buffer={MIN_BUFFER}")
else:
    GAMES_PER_EVAL      = 10
    EVAL_PAIRS          = 5
    BUFFER_SIZE         = 2000
    MIN_BUFFER          = 200
    BATCH_SIZE          = 128
    TRAINING_STEPS      = 50
    LR                  = 5e-4
    LR_DECAY            = 0.95
    GAMMA               = 0.98
    CHECKPOINT_INTERVAL = 1
    MAX_TURNS           = 50
    print(f"POC mode: games_per_eval={GAMES_PER_EVAL}, eval_pairs={EVAL_PAIRS}, buffer={BUFFER_SIZE}, min_buffer={MIN_BUFFER}")

DISTILL_WEIGHTS     = 'gnn_weights.pt'
SELFPLAY_WEIGHTS    = 'gnn_selfplay.pt'

# We use a session ID to avoid overwriting files from previous runs
SESSION = random.randint(1000000000, 1999999999)
print(f"Session ID: {SESSION}  (checkpoints will be named *_s{SESSION}_*)")

# -------------------------
# UTILS
# -------------------------

def collect_selfplay_games(num_games, agent, encoder, start_seed=0):
    """Plays num_games and returns a list of (state, outcome) pairs."""
    data = []
    
    for i in range(num_games):
        seed = start_seed + i
        random.seed(seed)
        board = Board()
        history = [] # list of (encoded_state, current_player)
        
        turns = 0
        while turns < MAX_TURNS:
            winner, _ = board.check_game_over()
            if winner: break
            
            moves = board.get_valid_moves()
            if not moves: break
            
            # Record state
            # Note: We encode from the perspective of the current player
            encoded = encoder.encode(board, board.current_player)
            history.append((encoded, board.current_player))
            
            # Select move
            chosen = agent.select_move_pair_fast(moves, board, board.current_player)
            
            # Apply move
            for move in chosen:
                if move != (0, 0, 0):
                    board.apply_move(move, switch_turn=False)
            
            board.switch_turn()
            turns += 1
            
        # Game over — get winner
        winner, _ = board.check_game_over()
        
        # Label history
        for j, (encoded, player) in enumerate(history):
            if winner == 0:
                outcome = 0.0 # Draw
            elif winner == player:
                # Closer to end = more certain
                outcome = 1.0 * (GAMMA ** (len(history) - 1 - j))
            else:
                outcome = -1.0 * (GAMMA ** (len(history) - 1 - j))
            
            data.append((encoded, outcome))
            
    return data

def binomial_p_value(successes, trials, target_p=0.58):
    """Probability of getting 'successes' or more wins if true win rate is target_p."""
    p_val = 0
    for k in range(successes, trials + 1):
        p_val += math.comb(trials, k) * (target_p**k) * ((1 - target_p)**(trials - k))
    return p_val

def evaluate_vs_frozen(challenger_agent, frozen_weights_path, num_pairs=10):
    if not os.path.exists(frozen_weights_path):
        return num_pairs, num_pairs * 2 
    
    frozen_agent = GNNAgent(weights_path=frozen_weights_path)
    wins = 0
    total = 0
    
    for i in range(num_pairs):
        seed = random.randint(0, 1000000)
        wins += 1 if play_single_game(challenger_agent, frozen_agent, seed) == 1 else 0
        total += 1
        wins += 1 if play_single_game(frozen_agent, challenger_agent, seed) == 2 else 0
        total += 1
        
        # Hard Math Check
        remaining = (num_pairs * 2) - total
        if wins + remaining < (num_pairs * 2 * 0.58):
             print(f"    [Early Exit] pair {i+1}/{num_pairs}  {wins}/{total}. Impossible to promote.")
             return wins, total

        # Statistical P-Value Check (The "Failing Student" rule)
        if total >= 12:
            p_val = binomial_p_value(wins, total, target_p=0.58)
            if p_val > 0.30: # 70% chance this model is NOT a champion
                print(f"    [Stat Exit] pair {i+1}/{num_pairs}  {wins}/{total} (p={p_val:.3f}). Unlikely to promote.")
                return wins, total
             
        print(f"    pair {i+1}/{num_pairs}  {wins}/{total} ({wins/total:.0%})")
        
    return wins, total

def play_single_game(p1_agent, p2_agent, seed):
    random.seed(seed)
    board = Board()
    turns = 0
    while turns < MAX_TURNS:
        winner, _ = board.check_game_over()
        if winner: return winner
        
        moves = board.get_valid_moves()
        if not moves: break
        
        agent = p1_agent if board.current_player == 'white' else p2_agent
        chosen = agent.select_move_pair_fast(moves, board, board.current_player)
        
        for move in chosen:
            if move != (0, 0, 0):
                board.apply_move(move, switch_turn=False)
        board.switch_turn()
        turns += 1
    return 0 # Draw

# -------------------------
# MAIN TRAINING
# -------------------------

def main():
    # 1. Initialize
    encoder = BoardEncoder()
    
    # Load weights (Start from distilled if no selfplay exists)
    weights_path = SELFPLAY_WEIGHTS if os.path.exists(SELFPLAY_WEIGHTS) else DISTILL_WEIGHTS
    print(f"Loading weights from {weights_path}...")
    
    model = BoardGNN().to(DEVICE)
    try:
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        print(f"Model loaded from {weights_path} on {DEVICE}")
    except Exception as e:
        print(f"Error loading {weights_path}: {e}")
        print("Initializing new model...")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.ExponentialLR(optimizer, gamma=LR_DECAY)
    criterion = nn.MSELoss()
    
    # The Buffer
    replay_buffer = collections.deque(maxlen=BUFFER_SIZE)
    
    # --- BUFFER RESUME LOGIC ---
    buffer_disk_path = os.path.join(CHECKPOINT_DIR, f'replay_buffer_latest.pt')
    if os.path.exists(buffer_disk_path):
        try:
            print(f"Attempting to load buffer from {buffer_disk_path}...")
            loaded_data = torch.load(buffer_disk_path)
            replay_buffer.extend(loaded_data)
            print(f"Successfully resumed buffer with {len(replay_buffer)} positions.")
        except Exception as e:
            print(f"Could not load buffer: {e}")
    else:
        print("Buffer starts empty. Training begins after 4000 positions accumulated.")
    # ---------------------------

    generation = 0
    start = time.time()
    
    try:
        while True:
            gen_start = time.time()
            print(f"Generation {generation}:")
            
            # A. Self-play
            model.eval()
            current_agent = GNNAgent(model=model)
            
            for g in range(0, GAMES_PER_EVAL, 5):
                new_data = collect_selfplay_games(5, current_agent, encoder, start_seed=generation*1000 + g)
                replay_buffer.extend(new_data)
                
                # Report
                loss_val = 0.0 # to be filled after training
                print(f"  game {g+5}/{GAMES_PER_EVAL}  buffer={len(replay_buffer)}  loss={loss_val:.4f}  positions={len(new_data)}")

            # B. Training
            if len(replay_buffer) < MIN_BUFFER:
                print(f"  Buffer too small ({len(replay_buffer)}/{MIN_BUFFER}). Skipping training...")
                generation += 1
                continue
                
            model.train()
            total_loss = 0
            for _ in range(TRAINING_STEPS):
                batch = random.sample(replay_buffer, BATCH_SIZE)
                # batch is list of (encoded_dict, outcome)
                inputs, targets = collate_batch(batch)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets.to(DEVICE))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / TRAINING_STEPS
            print(f"  Finished training. Avg Loss: {avg_loss:.4f}")

            # C. Evaluation
            print(f"  Evaluating vs frozen champion ({EVAL_PAIRS} pairs)...")
            model.eval()
            challenger = GNNAgent(model=model)
            
            # Save a temp version for the evaluator to load
            temp_path = "gnn_temp_eval.pt"
            save_model(model, temp_path)
            
            wins, total = evaluate_vs_frozen(challenger, SELFPLAY_WEIGHTS, num_pairs=EVAL_PAIRS)
            win_rate = wins / total
            
            # Promotion logic (Requires > 58% win rate to replace the champion)
            if win_rate >= 0.58:
                print(f"  ⭐ PROMOTED! New champion win rate: {win_rate:.1%}")
                save_model(model, SELFPLAY_WEIGHTS)
                
                # Also save to Drive
                drive_best = os.path.join(CHECKPOINT_DIR, SELFPLAY_WEIGHTS)
                save_model(model, drive_best)

                if generation % CHECKPOINT_INTERVAL == 0:
                    ckpt_name  = f'gnn_selfplay_s{SESSION}_gen{generation}.pt'
                    drive_ckpt = os.path.join(CHECKPOINT_DIR, ckpt_name)
                    save_model(model, drive_ckpt)
                    print(f"  Checkpoint saved: {drive_ckpt}")
            else:
                print(f"  ✗ Not promoted. ({win_rate:.1%})")

            # Periodic save — session-stamped weights
            if generation % CHECKPOINT_INTERVAL == 0:
                periodic_path = os.path.join(CHECKPOINT_DIR,
                                             f'gnn_current_s{SESSION}_gen{generation}.pt')
                save_model(model, periodic_path)
                print(f"  Periodic weight save: {periodic_path}")

            # --- BUFFER SAVE LOGIC ---
            # Save the buffer to Drive at the end of every generation
            try:
                # Convert deque to list for saving
                torch.save(list(replay_buffer), buffer_disk_path)
                # Keep a backup with the session ID just in case
                backup_buffer = os.path.join(CHECKPOINT_DIR, f'buffer_s{SESSION}_gen{generation}.pt')
                # Optional: Uncomment if you want historical buffer snapshots (Warning: heavy storage use)
                # torch.save(list(replay_buffer), backup_buffer) 
                print(f"  Buffer saved to Drive ({len(replay_buffer)} positions).")
            except Exception as e:
                print(f"  ⚠️ Warning: Failed to save buffer: {e}")
            # ---------------------------

            model.train()
            scheduler.step() # Step LR Decay
            
            gen_time = time.time() - gen_start
            total_time = time.time() - start
            print(f"  Gen time: {gen_time:.0f}s  Total: {total_time/3600:.1f}h  LR: {optimizer.param_groups[0]['lr']:.2e}\n")

            generation += 1

    except KeyboardInterrupt:
        print("\nTraining stopped.")
        save_model(model, SELFPLAY_WEIGHTS)
        # Save final weights and buffer to Drive
        drive_final = os.path.join(CHECKPOINT_DIR, SELFPLAY_WEIGHTS)
        save_model(model, drive_final)
        torch.save(list(replay_buffer), buffer_disk_path)
        print("Final weights and buffer saved.")

if __name__ == "__main__":
    main()