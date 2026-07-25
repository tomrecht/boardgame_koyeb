"""
train_td.py — TD(lambda) fine-tuning of the board-game GNN value head.

Replaces the Monte-Carlo target (final outcome) with the lambda-return,
bootstrapped from a FROZEN copy of the model that is refreshed each epoch
(target-network trick — essential once you bootstrap, or targets chase the
live weights and training oscillates).

Reuses the existing pipeline (BoardEncoder, collate_batch, ReduceLROnPlateau,
split-by-game) and the value-head convention (tanh output, raw * NUM_PIECES =
expected margin, mover's-frame sign).

Mode B (offline sanity check), runnable directly:
    python train_td.py \
        --data generated_positions.jsonl \
        --load best_iter5_m46.pt \
        --save gnn_td.pt \
        --lam 0.9 --gamma 1.0 --lr 2e-4 --epochs 30

The per-epoch routines (`compute_epoch_targets`, `run_td_epoch`) are imported
by td_selfplay_loop.py for Mode A.

Device note: network.DEVICE is the single source of truth for where tensors
live. We set it once here and move the model onto it; the encoder builds
tensors on network.DEVICE, so the target model, encoder output, batches, and
labels are all guaranteed to be on the same device (this was the cause of past
"expected all tensors on same device" errors).
"""
import argparse
import json
import copy
import random

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import network
from network import BoardEncoder, BoardGNN, collate_batch
from game import Board

from td_returns import compute_td_targets, NUM_PIECES


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_records(paths, max_ply=None, min_ply=None):
    records = []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get('final_score') is None:
                    continue
                ply = rec.get('ply_from_end', 0)
                if max_ply is not None and ply > max_ply:
                    continue
                if min_ply is not None and ply < min_ply:
                    continue
                records.append(rec)
    return records


def split_by_game(records, val_frac=0.15, seed=0):
    """Split by GAME so no game leaks across train/val. Critical for TD: states
    within a game are highly correlated and share a bootstrap chain."""
    game_ids = sorted({r['game_id'] for r in records})
    rng = random.Random(seed)
    rng.shuffle(game_ids)
    n_val = max(1, int(len(game_ids) * val_frac))
    val_ids = set(game_ids[:n_val])
    train = [r for r in records if r['game_id'] not in val_ids]
    val   = [r for r in records if r['game_id'] in val_ids]
    return train, val


def compute_weight(rec, uniform=True, floor=0.1):
    """Sample weight. TD(lambda) already reduces early-position variance via
    bootstrapping, so the old 1/ply_from_end down-weighting double-counts and
    starves the early/midgame positions we want TD to improve. Default uniform.
    """
    if uniform:
        return 1.0
    ply = rec.get('ply_from_end', 10)
    return max(floor, 1.0 / max(1, ply))


def print_data_stats(records, label=''):
    from collections import defaultdict
    by_stage = defaultdict(int)
    by_outcome = defaultdict(int)
    for r in records:
        by_stage[r.get('game_stage', 'unknown')] += 1
        fs = r.get('final_score', 0)
        by_outcome['win' if fs > 0 else ('loss' if fs < 0 else 'draw')] += 1
    games = len({r['game_id'] for r in records})
    print(f"  {label}: {len(records)} positions, {games} games")
    print(f"    Stage:   {dict(by_stage)}")
    print(f"    Outcome: {dict(by_outcome)}")


# ---------------------------------------------------------------------------
# Encoding a batch against a precomputed td_target
# ---------------------------------------------------------------------------

def encode_batch_with_targets(records, encoder, board, uniform_weights=True):
    """Encode records into a collated batch + td_target labels + weights.
    Records must already carry 'td_target' (added by compute_td_targets)."""
    encoded, labels, weights = [], [], []
    failed = 0
    for rec in records:
        try:
            board.update_state(rec['raw_state'])
            enc = encoder.encode(board, rec['player'])
            encoded.append(enc)
            labels.append(rec['td_target'])
            weights.append(compute_weight(rec, uniform=uniform_weights))
        except Exception:
            failed += 1
    if not encoded:
        return None, None, None, failed
    batch = collate_batch(encoded)
    labels  = torch.tensor(labels,  dtype=torch.float32, device=network.DEVICE)
    weights = torch.tensor(weights, dtype=torch.float32, device=network.DEVICE)
    return batch, labels, weights, failed


# ---------------------------------------------------------------------------
# Per-epoch target computation + training
# ---------------------------------------------------------------------------

def compute_epoch_targets(model, records, encoder, board, lam, gamma, verbose=False):
    """Snapshot a frozen target net from `model` and compute TD(lambda) targets
    for every record. Returns the new record list (each with 'td_target')."""
    target_model = copy.deepcopy(model)
    target_model.eval()
    for p in target_model.parameters():
        p.requires_grad_(False)
    return compute_td_targets(records, target_model, encoder, board,
                              lam=lam, gamma=gamma, verbose=verbose)


def run_td_epoch(model, records_with_targets, encoder, board, optimizer,
                 batch_size, training, uniform_weights=True):
    """One pass of weighted-MSE regression toward precomputed td_targets."""
    if training:
        model.train()
        order = list(range(len(records_with_targets)))
        random.shuffle(order)
        recs = [records_with_targets[i] for i in order]
    else:
        model.eval()
        recs = records_with_targets

    total_loss = 0.0
    total_weight = 0.0
    correct_sign = 0
    counted = 0

    for start in range(0, len(recs), batch_size):
        chunk = recs[start:start + batch_size]
        batch, labels, weights, _ = encode_batch_with_targets(
            chunk, encoder, board, uniform_weights=uniform_weights)
        if batch is None:
            continue

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            preds = model(batch)                       # [B], mover frame, on DEVICE
            diff = preds - labels
            loss = (weights * diff * diff).sum() / weights.sum()
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        total_loss += loss.item() * weights.sum().item()
        total_weight += weights.sum().item()
        # "sign accuracy" vs the target's sign — a coarse sanity signal
        with torch.no_grad():
            same = ((preds > 0) == (labels > 0))
            correct_sign += same.sum().item()
            counted += same.numel()

    avg_loss = total_loss / total_weight if total_weight else 0.0
    acc = correct_sign / counted if counted else 0.0
    return avg_loss, acc


# ---------------------------------------------------------------------------
# Main (Mode B / standalone)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TD(lambda) GNN fine-tuning")
    parser.add_argument('--data', nargs='+', required=True,
                        help="JSONL file(s) of recorded self-play positions")
    parser.add_argument('--load', type=str, default='best_iter5_m46.pt',
                        help="warm-start weights (strongly recommended)")
    parser.add_argument('--save', type=str, default='gnn_td.pt')
    parser.add_argument('--lam', type=float, default=0.9, help="TD lambda")
    parser.add_argument('--gamma', type=float, default=1.0, help="discount")
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--val-frac', type=float, default=0.15)
    parser.add_argument('--max-ply', type=int, default=None)
    parser.add_argument('--patience', type=int, default=8,
                        help="early-stop patience on val loss")
    parser.add_argument('--min-epochs', type=int, default=10)
    parser.add_argument('--ply-weights', action='store_true',
                        help="restore old 1/ply down-weighting (default: uniform)")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cpu', action='store_true', help="force CPU")
    args = parser.parse_args()

    if args.cpu:
        network.DEVICE = torch.device('cpu')
    print(f"Device: {network.DEVICE}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    uniform_weights = not args.ply_weights

    # Model (warm start)
    model = BoardGNN().to(network.DEVICE)
    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=network.DEVICE))
        print(f"Warm-started from {args.load}")
    else:
        print("WARNING: training TD from scratch (no --load); strongly discouraged")

    encoder = BoardEncoder()
    board = Board()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.5)

    # Data
    records = load_records(args.data, max_ply=args.max_ply)
    print_data_stats(records, 'All data')
    train_recs, val_recs = split_by_game(records, val_frac=args.val_frac, seed=args.seed)
    print(f"  Split: {len(train_recs)} train / {len(val_recs)} val positions")
    print(f"  lambda={args.lam}  gamma={args.gamma}  lr={args.lr}  "
          f"weights={'1/ply' if args.ply_weights else 'uniform'}")

    best_val = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        # Refresh the frozen target net & recompute lambda-returns each epoch.
        train_t = compute_epoch_targets(model, train_recs, encoder, board,
                                        args.lam, args.gamma, verbose=(epoch == 1))
        val_t = compute_epoch_targets(model, val_recs, encoder, board,
                                      args.lam, args.gamma, verbose=False)

        tr_loss, tr_acc = run_td_epoch(model, train_t, encoder, board, optimizer,
                                       args.batch_size, training=True,
                                       uniform_weights=uniform_weights)
        val_loss, val_acc = run_td_epoch(model, val_t, encoder, board, optimizer,
                                         args.batch_size, training=False,
                                         uniform_weights=uniform_weights)
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d} | train {tr_loss:.5f} (sgn {tr_acc:.2f}) | "
              f"val {val_loss:.5f} (sgn {val_acc:.2f}) | lr {lr_now:.2e}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience and epoch >= args.min_epochs:
                print(f"Early stop at epoch {epoch} (no val improvement in "
                      f"{args.patience}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), args.save)
    print(f"Saved best model (val {best_val:.5f}) to {args.save}")


if __name__ == '__main__':
    main()
