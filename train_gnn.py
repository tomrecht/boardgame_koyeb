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
    MIN_BUFFER          = 4_000
    BATCH_SIZE          = 512
    TRAINING_STEPS      = 400       
    LR                  = 1e-4      
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
OUTCOME_SCALE       = 200.0         # keeps self-play labels in ~[-0.6, +0.6]
SCORE_SCALE         = 1000.0        # must match agent_gnn.py and train_distill.py
PROMOTION_WINRATE   = 0.55
PROMOTION_PVALUE    = 0.05
COLLAPSE_PVALUE     = 0.50          # warn if challenger can't beat distilled baseline
EXPLORATION_RATE    = 0.10          # fraction of moves chosen randomly during self-play

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
            chosen = agent.select_move_pair_fast(moves, board, current_player)

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
    Returns (wins, total).
    """
    wins  = 0
    total = 0

    for i in range(num_pairs):
        seed = random.randint(0, 1_000_000)

        # Game 1: challenger = white (player 1 — wins if check_game_over returns 'white')
        result1 = play_eval_game(challenger_agent, opponent_agent, seed)
        wins  += 1 if result1 == 'white' else 0
        total += 1

        # Game 2: challenger = black (player 2 — wins if check_game_over returns 'black')
        result2 = play_eval_game(opponent_agent, challenger_agent, seed)
        wins  += 1 if result2 == 'black' else 0
        total += 1

        # Hard math early exit — can't possibly reach promotion threshold
        remaining = (num_pairs * 2) - total
        if wins + remaining < num_pairs * 2 * PROMOTION_WINRATE:
            print(f"    [Early Exit] pair {i+1}/{num_pairs}  {wins}/{total}. "
                  f"Impossible to reach {PROMOTION_WINRATE:.0%}.")
            return wins, total
        
        # Early success exit — already statistically certain to promote
        if total >= 20:
            p_val_success = binomial_p_value(wins, total, target_p=PROMOTION_WINRATE)
            if p_val_success <= PROMOTION_PVALUE and wins / total >= PROMOTION_WINRATE:
                print(f"    [Early Promote] pair {i+1}/{num_pairs}  {wins}/{total} (p={p_val_success:.3f}). Certain to promote.")
                return wins, total

        # Statistical early exit — unlikely to promote
        if total >= 20:
            p_val = binomial_p_value(wins, total, target_p=PROMOTION_WINRATE)
            if p_val > 0.50:
                print(f"    [Stat Exit] pair {i+1}/{num_pairs}  {wins}/{total} "
                      f"(p={p_val:.3f}). Unlikely to promote.")
                return wins, total

        print(f"    pair {i+1}/{num_pairs}  {wins}/{total} ({wins/total:.0%})")

    return wins, total


def play_eval_game(white_agent, black_agent, seed):
    """
    Play one evaluation game. No exploration, no recording.
    white_agent plays as 'white', black_agent plays as 'black'.
    Returns 'white', 'black', or None.
    """
    random.seed(seed)
    board = Board()
    turns = 0
    consecutive_passes = 0

    while turns < MAX_TURNS:
        winner, score = board.check_game_over()
        if winner:
            return winner

        current_player = board.current_player
        # board.current_player is 'white' or 'black'
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
                if ws > bs: return 'white'
                if bs > ws: return 'black'
                return None
        else:
            consecutive_passes = 0

        for move in chosen:
            if move != (0, 0, 0):
                board.apply_move(move, switch_turn=False)
        board.switch_turn()
        turns += 1

    ws = len(board.white_saved)
    bs = len(board.black_saved)
    if ws > bs: return 'white'
    if bs > ws: return 'black'
    return None


# -------------------------
# MAIN TRAINING
# -------------------------

def main():
    encoder = BoardEncoder()

    # Load starting weights — selfplay if exists, else distilled
    weights_path = SELFPLAY_WEIGHTS if os.path.exists(SELFPLAY_WEIGHTS) else DISTILL_WEIGHTS
    print(f"\nLoading weights from {weights_path}...")
    model = BoardGNN().to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.train()
    print(f"Model loaded on {DEVICE}  ({sum(p.numel() for p in model.parameters()):,} params)")

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

    # Agents wrapping the live models (no weight file I/O needed)
    challenger_agent = GNNAgent(model=model)
    frozen_agent     = GNNAgent(model=frozen_model)
    distilled_agent  = GNNAgent(model=distilled_model)

    # Replay buffer
    replay_buffer    = collections.deque(maxlen=BUFFER_SIZE)
    buffer_disk_path = os.path.join(CHECKPOINT_DIR, 'replay_buffer_latest.pt')

    # Resume buffer if available (e.g. after Colab timeout)
    if os.path.exists(buffer_disk_path):
        try:
            loaded = torch.load(buffer_disk_path)
            replay_buffer.extend(loaded)
            print(f"Resumed buffer with {len(replay_buffer)} positions.")
        except Exception as e:
            print(f"Buffer corrupted, starting fresh: {e}")
            os.remove(buffer_disk_path)
    else:
        print(f"Buffer starts empty. Training begins after {MIN_BUFFER} positions.")

    # Sanity check — catches bugs before burning Colab time
    print("\nRunning sanity check...")
    _sanity_check(model, encoder, criterion)
    print("Sanity check passed.\n")

    generation = 0
    start      = time.time()

    print(f"Starting self-play training. Press Ctrl+C to stop.\n")

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
                    print(f"  game {g+1}/{GAMES_PER_EVAL}  "
                          f"buffer={len(replay_buffer)}  "
                          f"positions={gen_positions}")

            # ---- B. TRAINING ----
            if len(replay_buffer) < MIN_BUFFER:
                print(f"  Buffer too small ({len(replay_buffer)}/{MIN_BUFFER}), "
                      f"skipping training.")
                generation += 1
                continue

            model.train()
            total_loss = 0.0
            for _ in range(TRAINING_STEPS):
                batch        = random.sample(replay_buffer, BATCH_SIZE)
                encoded_list = [item[0] for item in batch]
                targets      = torch.tensor([item[1] for item in batch],
                                            dtype=torch.float32, device=DEVICE)
                optimizer.zero_grad()
                outputs = model(encoded_list)
                loss    = criterion(outputs.squeeze(), targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / TRAINING_STEPS
            print(f"  Training loss: {avg_loss:.4f}")

            # Label stats for the current generation's data
            recent = list(replay_buffer)[-gen_positions:]
            labels = [item[1] for item in recent]
            print(f"  label stats: mean={sum(labels)/len(labels):.3f}  "
                  f"std={torch.tensor(labels).std().item():.3f}  "
                  f"min={min(labels):.3f}  max={max(labels):.3f}")

            # ---- C. EVALUATION ----
            model.eval()

            print(f"  Evaluating vs frozen champion ({EVAL_PAIRS} pairs)...")
            wins, total = evaluate_vs_agent(challenger_agent, frozen_agent, EVAL_PAIRS)
            win_rate = wins / total if total else 0
            p_value  = binomial_p_value(wins, total, target_p=PROMOTION_WINRATE)
            print(f"  vs frozen:    {wins}/{total} ({win_rate:.1%})  p={p_value:.3f}")

            # ---- D. PROMOTION ----
            promoted = (win_rate >= PROMOTION_WINRATE and p_value <= PROMOTION_PVALUE)

            # Collapse detection — only worth checking if about to promote
            if promoted:
                print(f"  Evaluating vs distilled baseline ({EVAL_PAIRS} pairs)...")
                wins_d, total_d = evaluate_vs_agent(
                    challenger_agent, distilled_agent, EVAL_PAIRS)
                wr_d = wins_d / total_d if total_d else 0
                p_d  = binomial_p_value(wins_d, total_d, target_p=PROMOTION_WINRATE)
                print(f"  vs distilled: {wins_d}/{total_d} ({wr_d:.1%})  p={p_d:.3f}")
                if p_d > COLLAPSE_PVALUE:
                    print(f"  ⚠️  WARNING: not reliably beating distilled baseline — "
                          f"possible catastrophic forgetting!")
                    promoted = False

            if promoted:
                print(f"  ⭐ PROMOTED! ({win_rate:.1%}). Updating frozen opponent.")
                frozen_model.load_state_dict(copy.deepcopy(model.state_dict()))
                save_model(model, SELFPLAY_WEIGHTS)
                drive_best = os.path.join(CHECKPOINT_DIR, SELFPLAY_WEIGHTS)
                save_model(model, drive_best)
            else:
                print(f"  ✗ Not promoted. ({win_rate:.1%})")

            # Periodic checkpoint
            if generation % CHECKPOINT_INTERVAL == 0:
                ckpt_name = f'gnn_current_s{SESSION}_gen{generation}.pt'
                ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
                save_model(model, ckpt_path)
                print(f"  Periodic save: {ckpt_path}")

            # Save buffer to Drive so we can resume after timeout
            try:
                tmp_path = buffer_disk_path + '.tmp'
                torch.save(list(replay_buffer), tmp_path)
                os.replace(tmp_path, buffer_disk_path)
                print(f"  Buffer saved ({len(replay_buffer)} positions).")
            except Exception as e:
                print(f"  ⚠️  Warning: failed to save buffer: {e}")

            model.train()
            scheduler.step()

            gen_time   = time.time() - gen_start
            total_time = time.time() - start
            print(f"  Gen time: {gen_time:.0f}s  "
                  f"Total: {total_time/3600:.1f}h  "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}\n")

            generation += 1

    except KeyboardInterrupt:
        print("\nTraining stopped.")
        save_model(model, SELFPLAY_WEIGHTS)
        drive_final = os.path.join(CHECKPOINT_DIR, SELFPLAY_WEIGHTS)
        save_model(model, drive_final)
        try:
            tmp_path = buffer_disk_path + '.tmp'
            torch.save(list(replay_buffer), tmp_path)
            os.replace(tmp_path, buffer_disk_path)
        except Exception:
            pass
        print("Weights and buffer saved.")


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