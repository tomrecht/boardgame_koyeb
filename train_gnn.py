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
    GAMES_PER_EVAL      = 500       # self-play games between evaluations
    EVAL_PAIRS          = 40        # paired games per evaluation
    BUFFER_SIZE         = 100_000   # max positions in replay buffer
    MIN_BUFFER          = 5_000     # don't train until buffer has this many
    DISTILL_PREFILL     = 10_000    # positions to load from distill_data.pt
    CHECKPOINT_INTERVAL = 10        # save checkpoint every N generations
    EXPLORATION_RATE    = 0.10      # probability of picking from top-3 moves
else:
    GAMES_PER_EVAL      = 20
    EVAL_PAIRS          = 10
    BUFFER_SIZE         = 10_000
    MIN_BUFFER          = 200      # ~3-4 games worth before training starts
    DISTILL_PREFILL     = 1_000   # kept for reference but no longer used
    CHECKPOINT_INTERVAL = 5
    EXPLORATION_RATE    = 0.10

# Shared across POC and full
DISCOUNT            = 0.97          # per remaining player-move
OUTCOME_SCALE       = 1000.0        # scale outcome before /SCORE_SCALE
SCORE_SCALE         = 1000.0        # must match train_distill.py
LR                  = 1e-4          # lower than distillation — fine-tuning
GRAD_ACCUM_STEPS    = 32            # accumulate gradients over N samples
PROMOTION_WINRATE   = 0.55
PROMOTION_PVALUE    = 0.10
COLLAPSE_PVALUE     = 0.50          # if p_value this high vs distilled, warn
MAX_TURNS           = 150
MASTER_SEED         = 999
GNN_WEIGHTS         = 'gnn_weights.pt'        # distilled starting point
SELFPLAY_WEIGHTS    = 'gnn_selfplay.pt'       # best self-play weights
DISTILL_DATA        = 'distill_data.pt'


# -------------------------
# STATS HELPERS  (from train.py)
# -------------------------

def ttest_greater(data):
    n = len(data)
    if n < 2:
        return 1.0
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / (n - 1)
    if variance == 0:
        return 0.0 if mean > 0 else 1.0
    t_stat = mean / math.sqrt(variance / n)
    p_value = 0.5 * (1 - math.erf(t_stat / math.sqrt(2)))
    return p_value


# -------------------------
# REPLAY BUFFER
# -------------------------

class ReplayBuffer:
    """Fixed-size circular buffer of (encoded_position, label) pairs."""

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer   = collections.deque(maxlen=max_size)

    def add(self, encoded, label):
        self.buffer.append((encoded, label))

    def sample(self, n):
        n = min(n, len(self.buffer))
        return random.sample(list(self.buffer), n)

    def __len__(self):
        return len(self.buffer)


# -------------------------
# SELF-PLAY GAME
# -------------------------

def play_selfplay_game(agent, encoder, seed):
    """
    Play one game with agent as both players.
    Returns list of (encoded_position, current_player, turn_number) tuples.
    Game outcome returned separately as (winner, score).
    """
    random.seed(seed)
    board  = Board()
    record = []   # (encoded, player, turn)
    turns  = 0
    consecutive_passes = 0

    while turns < MAX_TURNS:
        winner, score = board.check_game_over()
        if winner:
            return record, winner, score

        current_player = board.current_player

        # Record position BEFORE move (pre-move)
        encoded = encoder.encode(board, current_player)
        encoded_stored = {k: v.clone() for k, v in encoded.items()}
        record.append((encoded_stored, current_player, turns))

        # Select move with optional exploration
        moves = board.get_valid_moves()
        if not moves:
            break

        if random.random() < EXPLORATION_RATE:
            chosen = _explore(agent, moves, board, current_player)
        else:
            chosen = agent.select_move_pair_fast(moves, board, current_player)

        if chosen == ((0, 0, 0), (0, 0, 0)):
            consecutive_passes += 1
            if consecutive_passes >= 6:
                # Tiebreak by saved pieces
                ws = len(board.white_saved)
                bs = len(board.black_saved)
                if ws > bs:   return record, 'white', ws - bs
                if bs > ws:   return record, 'black', bs - ws
                return record, None, 0
        else:
            consecutive_passes = 0

        for move in chosen:
            if move != (0, 0, 0):
                board.apply_move(move, switch_turn=False)
        board.switch_turn()
        turns += 1

    # Game hit max turns
    ws = len(board.white_saved)
    bs = len(board.black_saved)
    if ws > bs:   return record, 'white', ws - bs
    if bs > ws:   return record, 'black', bs - ws
    return record, None, 0


def _explore(agent, moves, board, player):
    """
    Exploration: with probability EXPLORATION_RATE pick a random legal move pair,
    otherwise use the agent's batched select_move_pair.
    Simpler than re-implementing move scoring and avoids non-batched evaluate() calls.
    """
    if random.random() < EXPLORATION_RATE:
        # Pick a random first move, then a random second move
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
    return agent.select_move_pair_fast(moves, board, player)


# -------------------------
# LABEL POSITIONS
# -------------------------

def label_positions(record, winner, outcome_score, total_turns):
    """
    Assign discounted outcome labels to recorded positions.
    Label is from the current player's perspective at each position.
    """
    labeled = []
    if winner is None:
        outcome_score = 0

    for encoded, player, turn in record:
        # Turns remaining for this player
        turns_remaining = (total_turns - turn) / 2.0
        discount = DISCOUNT ** turns_remaining

        # Score from this player's perspective
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
# TRAINING STEP
# -------------------------

def training_step(model, optimizer, criterion, buffer, n_samples=GRAD_ACCUM_STEPS):
    """
    Sample n_samples from buffer, evaluate as a batch, do one optimizer step.
    Returns mean loss.
    """
    samples = buffer.sample(n_samples)
    if not samples:
        return 0.0

    model.train()
    optimizer.zero_grad()

    encoded_list = [e for e, _ in samples]
    targets = torch.tensor([l for _, l in samples],
                           dtype=torch.float32, device=DEVICE)

    preds = model(encoded_list)              # [n_samples] — true batched forward
    loss  = criterion(preds, targets)
    loss.backward()
    optimizer.step()

    return loss.item()


# -------------------------
# EVALUATION  (paired games)
# -------------------------

def evaluate(challenger_agent, opponent_agent, encoder, seeds):
    """
    Play paired games between challenger and opponent.
    Returns (wins, games, margins, p_value).
    """
    wins    = 0
    total   = 0
    margins = []

    for seed in seeds:
        # Game 1: challenger=white
        random.seed(seed)
        board = Board()
        agents = {'white': challenger_agent, 'black': opponent_agent}
        w1, s1 = _play_eval_game(board, agents)
        s1 = s1 or 0
        total += 1
        if w1 == 'white':
            wins += 1; margins.append(s1)
        elif w1 == 'black':
            margins.append(-s1)
        else:
            margins.append(0)

        # Game 2: challenger=black
        random.seed(seed)
        board = Board()
        agents = {'white': opponent_agent, 'black': challenger_agent}
        w2, s2 = _play_eval_game(board, agents)
        s2 = s2 or 0
        total += 1
        if w2 == 'black':
            wins += 1; margins.append(s2)
        elif w2 == 'white':
            margins.append(-s2)
        else:
            margins.append(0)

    p_value = ttest_greater(margins) if len(margins) > 1 else 1.0
    return wins, total, margins, p_value


def _play_eval_game(board, agents):
    """Play one evaluation game, no exploration, no recording."""
    turns = 0
    consecutive_passes = 0

    while turns < MAX_TURNS:
        winner, score = board.check_game_over()
        if winner:
            return winner, score

        current_player = board.current_player
        current_agent  = agents[current_player]
        moves = board.get_valid_moves()
        if not moves:
            break

        chosen = current_agent.select_move_pair(moves, board, current_player)

        if chosen == ((0,0,0),(0,0,0)):
            consecutive_passes += 1
            if consecutive_passes >= 6:
                ws = len(board.white_saved)
                bs = len(board.black_saved)
                if ws > bs: return 'white', ws - bs
                if bs > ws: return 'black', bs - ws
                return None, 0
        else:
            consecutive_passes = 0

        for move in chosen:
            if move != (0,0,0):
                board.apply_move(move, switch_turn=False)
        board.switch_turn()
        turns += 1

    ws = len(board.white_saved)
    bs = len(board.black_saved)
    if ws > bs: return 'white', ws - bs
    if bs > ws: return 'black', bs - ws
    return None, 0


# -------------------------
# MAIN TRAINING LOOP
# -------------------------

def train():
    print(f"{'POC' if not args.full else 'FULL'} mode: "
          f"games_per_eval={GAMES_PER_EVAL}, eval_pairs={EVAL_PAIRS}, "
          f"buffer={BUFFER_SIZE}, min_buffer={MIN_BUFFER}")

    # Load starting network (distilled)
    print(f"\nLoading distilled weights from {GNN_WEIGHTS}...")
    encoder    = BoardEncoder()
    model      = load_model(GNN_WEIGHTS)
    model.train()

    # Frozen opponent starts as distilled network
    frozen_model = load_model(GNN_WEIGHTS)
    frozen_model.eval()

    # Keep distilled network as permanent baseline for collapse detection
    distilled_model = load_model(GNN_WEIGHTS)
    distilled_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # Replay buffer
    buffer = ReplayBuffer(BUFFER_SIZE)

    # Buffer starts empty — self-play samples only
    # Distilled weights provide the starting point; buffer doesn't need anchoring
    print(f"Buffer starts empty, min_buffer={MIN_BUFFER} before training begins")

    # Seed generator for self-play and evaluation
    rng = random.Random(MASTER_SEED)

    # Agents
    challenger_agent = GNNAgent.__new__(GNNAgent)
    challenger_agent.encoder = encoder
    challenger_agent.model   = model

    frozen_agent = GNNAgent.__new__(GNNAgent)
    frozen_agent.encoder = encoder
    frozen_agent.model   = frozen_model

    distilled_agent = GNNAgent.__new__(GNNAgent)
    distilled_agent.encoder = encoder
    distilled_agent.model   = distilled_model

    generation   = 0
    total_games  = 0
    train_losses = []
    start        = time.time()

    print(f"\nStarting self-play training. Press Ctrl+C to stop.\n")

    try:
        while True:
            gen_start    = time.time()
            gen_wins     = 0
            gen_games    = 0
            gen_positions = 0

            print(f"Generation {generation}:")

            # --- SELF-PLAY PHASE ---
            for game_idx in range(GAMES_PER_EVAL):
                seed = rng.randint(0, 2**31)
                record, winner, score = play_selfplay_game(
                    challenger_agent, encoder, seed)

                total_turns = len(record)
                labeled = label_positions(record, winner, score, total_turns)

                for encoded, label in labeled:
                    buffer.add(encoded, label)

                gen_positions += len(labeled)
                gen_games     += 1
                total_games   += 1

                # Training step (if buffer is big enough)
                if len(buffer) >= MIN_BUFFER:
                    loss = training_step(model, optimizer, criterion, buffer)
                    train_losses.append(loss)

                if (game_idx + 1) % 5 == 0 or game_idx == GAMES_PER_EVAL - 1:
                    mean_loss = sum(train_losses[-20:]) / max(1, len(train_losses[-20:]))
                    print(f"  game {game_idx+1}/{GAMES_PER_EVAL}  "
                          f"buffer={len(buffer)}  "
                          f"loss={mean_loss:.4f}  "
                          f"positions={gen_positions}")

            # --- EVALUATION PHASE ---
            print(f"\n  Evaluating vs frozen opponent ({EVAL_PAIRS} pairs)...")
            model.eval()
            eval_seeds = [rng.randint(0, 2**31) for _ in range(EVAL_PAIRS)]

            wins, games, margins, p_value = evaluate(
                challenger_agent, frozen_agent, encoder, eval_seeds)

            win_rate    = wins / games if games else 0
            mean_margin = sum(margins) / len(margins) if margins else 0
            model.train()

            print(f"  vs frozen:    {wins}/{games} ({win_rate:.1%})  "
                  f"margin={mean_margin:+.2f}  p={p_value:.3f}")

            # Check against distilled baseline (collapse detection)
            wins_d, games_d, margins_d, p_d = evaluate(
                challenger_agent, distilled_agent, encoder, eval_seeds)
            wr_d = wins_d / games_d if games_d else 0
            print(f"  vs distilled: {wins_d}/{games_d} ({wr_d:.1%})  "
                  f"p={p_d:.3f}")

            if p_d > COLLAPSE_PVALUE and generation > 2:
                print(f"  ⚠️  WARNING: not beating distilled baseline (p={p_d:.3f})")

            # --- PROMOTION ---
            promoted = (win_rate >= PROMOTION_WINRATE and
                        p_value  <= PROMOTION_PVALUE  and
                        mean_margin > 0)

            if promoted:
                print(f"  ✓ New champion! Updating frozen opponent.")
                frozen_model.load_state_dict(copy.deepcopy(model.state_dict()))
                save_model(model, SELFPLAY_WEIGHTS)

                # Save to Drive
                drive_best = os.path.join(CHECKPOINT_DIR, SELFPLAY_WEIGHTS)
                save_model(model, drive_best)

                if generation % CHECKPOINT_INTERVAL == 0:
                    ckpt_name  = SELFPLAY_WEIGHTS.replace('.pt', f'_gen{generation}.pt')
                    drive_ckpt = os.path.join(CHECKPOINT_DIR, ckpt_name)
                    save_model(model, drive_ckpt)
                    print(f"  Checkpoint saved: {drive_ckpt}")
            else:
                print(f"  ✗ Not promoted.")

            # Periodic save regardless of promotion — guard against session timeout
            if generation % CHECKPOINT_INTERVAL == 0:
                periodic_path = os.path.join(CHECKPOINT_DIR, f'gnn_current_gen{generation}.pt')
                save_model(model, periodic_path)
                print(f"  Periodic save: {periodic_path}")

            gen_time = time.time() - gen_start
            total_time = time.time() - start
            print(f"  Gen time: {gen_time:.0f}s  Total: {total_time/3600:.1f}h\n")

            generation += 1

    except KeyboardInterrupt:
        print("\nTraining stopped.")
        save_model(model, SELFPLAY_WEIGHTS)
        # Save final weights to Drive
        drive_final = os.path.join(CHECKPOINT_DIR, SELFPLAY_WEIGHTS)
        save_model(model, drive_final)
        print(f"Weights saved to {SELFPLAY_WEIGHTS} and {drive_final}")


if __name__ == '__main__':
    train()
