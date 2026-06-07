"""
train_gnn.py — Train the GNN on position data from generated or human games.

Supports:
  - Pretraining on generated heuristic vs heuristic data
  - Fine-tuning on human game data (lower LR, starts from pretrained weights)
  - Mixing both data sources

Usage:
    # Pretrain on generated games:
    python train_gnn.py --data training_data/generated_positions.jsonl

    # Fine-tune on human games starting from pretrained weights:
    python train_gnn.py --data training_data/positions_with_moves.jsonl \
                        --load gnn_weights.pt --lr 0.0001 --save gnn_finetuned.pt

    # Mix both:
    python train_gnn.py --data training_data/generated_positions.jsonl \
                                training_data/positions_with_moves.jsonl

    # Dry run to check data loading:
    python train_gnn.py --data training_data/generated_positions.jsonl --dry-run
"""

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from game import Board
from network import BoardEncoder, BoardGNN, collate_batch, DEVICE

NUM_PIECES = 12
CONTRASTIVE_WEIGHT = 0.5   # weight on counterfactual-pair difference loss; raise if pairs are few

# -------------------------
# DATA LOADING
# -------------------------

def load_records(paths, max_ply=None, min_ply=None):
    records = []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get('final_score') is None:
                    continue
                ply = rec.get('ply_from_end', 999)
                if max_ply is not None and ply > max_ply:
                    continue
                if min_ply is not None and ply < min_ply:
                    continue
                records.append(rec)
    return records


def split_by_game(records, val_frac=0.15):
    """Split into train/val by game_id to prevent leakage."""
    game_ids = list({r['game_id'] for r in records})
    random.shuffle(game_ids)
    n_val = max(1, int(len(game_ids) * val_frac))
    val_ids = set(game_ids[:n_val])
    train = [r for r in records if r['game_id'] not in val_ids]
    val   = [r for r in records if r['game_id'] in val_ids]
    return train, val


def compute_label(rec):
    """Normalized margin target in [-1, 1]."""
    return max(-1.0, min(1.0, rec['final_score'] / NUM_PIECES))


def compute_weight(rec):
    """Higher weight near game end; up-weight MC-averaged (lower-variance) labels."""
    ply = rec.get('ply_from_end', 10)
    w = 1.0 / max(1, ply)
    mc = rec.get('mc_count')
    if mc:                       # endgame records: inverse-variance-ish, capped
        w *= min(mc, 20) / 4.0
    return w


def print_data_stats(records, label=''):
    by_stage   = defaultdict(int)
    by_player  = defaultdict(int)
    by_outcome = defaultdict(int)
    for r in records:
        by_stage[r.get('game_stage', 'unknown')] += 1
        by_player[r.get('player', 'unknown')] += 1
        by_outcome['win' if r.get('final_score', 0) > 0 else 'loss'] += 1
    games = len({r['game_id'] for r in records})
    print(f"  {label}: {len(records)} positions, {games} games")
    print(f"    Stage:   {dict(by_stage)}")
    print(f"    Player:  {dict(by_player)}")
    print(f"    Outcome: {dict(by_outcome)}")


# -------------------------
# ENCODING + BATCH
# -------------------------

def encode_batch(records, encoder, board):
    """Encode a list of records into a collated batch + labels + weights."""
    encoded = []
    labels  = []
    weights = []
    failed  = 0
    for rec in records:
        try:
            board.update_state(rec['raw_state'])
            enc = encoder.encode(board, rec['player'])
            encoded.append(enc)
            labels.append(compute_label(rec))
            weights.append(compute_weight(rec))
        except Exception as e:
            failed += 1
    if failed:
        print(f"    Warning: {failed} records failed to encode")
    if not encoded:
        return None, None, None
    batch   = collate_batch(encoded)
    labels  = torch.tensor(labels,  dtype=torch.float32, device=DEVICE)
    weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    return batch, labels, weights

def build_pairs(records):
    """Group counterfactual records into (branch_a, branch_b) tuples by pair_id."""
    by_id = {}
    for r in records:
        by_id.setdefault(r['pair_id'], {})[r.get('branch')] = r
    return [(d['a'], d['b']) for d in by_id.values() if 'a' in d and 'b' in d]


def encode_pairs(pairs, encoder, board):
    """Encode both branches, keeping A/B aligned (drop a pair if either fails)."""
    ea, eb, la, lb = [], [], [], []
    for ra, rb in pairs:
        try:
            board.update_state(ra['raw_state']); a = encoder.encode(board, ra['player'])
            board.update_state(rb['raw_state']); b = encoder.encode(board, rb['player'])
        except Exception:
            continue
        ea.append(a); eb.append(b)
        la.append(compute_label(ra)); lb.append(compute_label(rb))
    if not ea:
        return None, None, None, None
    return (collate_batch(ea), collate_batch(eb),
            torch.tensor(la, dtype=torch.float32, device=DEVICE),
            torch.tensor(lb, dtype=torch.float32, device=DEVICE))


def run_contrastive_epoch(model, pairs, encoder, board, optimizer, batch_size,
                          training, weight=CONTRASTIVE_WEIGHT):
    """Train value DIFFERENCE on shared-dice pairs: (pred_a - pred_b) ~= (label_a - label_b)."""
    if not pairs:
        return 0.0
    if training:
        model.train(); random.shuffle(pairs)
    else:
        model.eval()
    total = 0.0
    n_pairs = 0
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        ba, bb, la, lb = encode_pairs(chunk, encoder, board)
        if ba is None:
            continue
        with torch.set_grad_enabled(training):
            diff_pred = model(ba) - model(bb)
            diff_targ = la - lb
            loss = weight * ((diff_pred - diff_targ) ** 2).mean()
        if training:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total += loss.item() * len(chunk)
        n_pairs += len(chunk)
    return total / n_pairs if n_pairs else 0.0

def run_joint_epoch(model, point_recs, pairs, encoder, board, optimizer,
                    batch_size, training, ctr_weight=CONTRASTIVE_WEIGHT, pair_every=8):
    """One epoch combining pointwise MSE and contrastive-pair loss in a single
    optimizer step per batch. The (small) contrastive set is cycled so every
    pointwise batch is paired with one, giving both objectives equal step-share."""
    if training:
        model.train()
        random.shuffle(point_recs)
        if pairs:
            random.shuffle(pairs)
    else:
        model.eval()

    total_loss = total_weight = 0.0
    total_ctr = 0.0
    ctr_batches = 0
    correct = 0
    pair_pos = 0
    pair_bs = max(1, batch_size // 2)

    for bidx, start in enumerate(range(0, len(point_recs), batch_size)):
        chunk = point_recs[start:start + batch_size]
        batch, labels, weights = encode_batch(chunk, encoder, board)

        ba = None
        ctr_loss = torch.zeros((), device=DEVICE)
        if pairs and (bidx % pair_every == 0):
            if pair_pos >= len(pairs):
                pair_pos = 0
                if training:
                    random.shuffle(pairs)
            pchunk = pairs[pair_pos:pair_pos + pair_bs]
            pair_pos += pair_bs
            ba, bb, la, lb = encode_pairs(pchunk, encoder, board)
            if ba is not None:
                with torch.set_grad_enabled(training):
                    ctr_loss = ((model(ba) - model(bb)) - (la - lb)).pow(2).mean()

        if batch is None and ba is None:
            continue

        with torch.set_grad_enabled(training):
            point_loss = torch.zeros((), device=DEVICE)
            if batch is not None:
                preds = model(batch)
                d = preds - labels
                point_loss = (weights * d * d).sum() / weights.sum()
                correct += ((preds > 0) == (labels > 0)).sum().item()
            loss = point_loss + ctr_weight * ctr_loss

        if training:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if batch is not None:
            total_loss += point_loss.item() * weights.sum().item()
            total_weight += weights.sum().item()
        if ba is not None:
            total_ctr += ctr_loss.item()
            ctr_batches += 1

    avg_loss = total_loss / total_weight if total_weight else 0.0
    avg_ctr = total_ctr / ctr_batches if ctr_batches else 0.0
    acc = correct / len(point_recs) if point_recs else 0.0
    return avg_loss, acc, avg_ctr
    
# -------------------------
# TRAINING
# -------------------------

def run_epoch(model, records, encoder, board, optimizer, batch_size, training):
    if training:
        model.train()
        random.shuffle(records)
    else:
        model.eval()

    total_loss   = 0.0
    total_weight = 0.0
    correct      = 0
    n_batches    = 0

    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        batch, labels, weights = encode_batch(chunk, encoder, board)
        if batch is None:
            continue

        with torch.set_grad_enabled(training):
            preds = model(batch)
            # Weighted MSE
            diff   = preds - labels
            losses = weights * diff * diff
            loss   = losses.sum() / weights.sum()

        if training:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss   += loss.item() * weights.sum().item()
        total_weight += weights.sum().item()
        correct      += ((preds > 0) == (labels > 0)).sum().item()
        n_batches    += 1

    avg_loss = total_loss / total_weight if total_weight else 0.0
    accuracy = correct / len(records) if records else 0.0
    return avg_loss, accuracy


# -------------------------
# MEAN ABS OUTPUT DIAGNOSTIC
# -------------------------

def mean_abs_output(model, records, encoder, board, n=200):
    """
    Sample up to n positions and report mean absolute model output.
    Collapse warning sign: this drops toward 0.
    """
    model.eval()
    sample = random.sample(records, min(n, len(records)))
    outputs = []
    for rec in sample:
        try:
            board.update_state(rec['raw_state'])
            enc = encoder.encode(board, rec['player'])
            with torch.no_grad():
                val = model(enc)
            outputs.append(abs(val.item()))
        except Exception:
            pass
    return sum(outputs) / len(outputs) if outputs else 0.0


# -------------------------
# MAIN
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',       nargs='+', required=True,
                        help='One or more .jsonl data files')
    parser.add_argument('--load',       type=str,  default=None,
                        help='Load pretrained weights from this path')
    parser.add_argument('--save',       type=str,  default='gnn_weights.pt')
    parser.add_argument('--epochs',     type=int,  default=60)
    parser.add_argument('--batch-size', type=int,  default=32)
    parser.add_argument('--lr',         type=float,default=0.001)
    parser.add_argument('--val-frac',   type=float,default=0.15)
    parser.add_argument('--max-ply',    type=int,  default=None)
    parser.add_argument('--patience',   type=int,  default=8,
                        help='Early stopping patience (epochs)')
    parser.add_argument('--min-epochs', type=int,  default=15)
    parser.add_argument('--dry-run',    action='store_true',
                        help='Load data and report stats without training')
    args = parser.parse_args()

    # Load data
    print(f"\nLoading data from: {args.data}")
    records = load_records(args.data, max_ply=args.max_ply)
    print(f"Total records: {len(records)}")
    print_data_stats(records, 'All')

    if not records:
        print("No records found. Exiting.")
        return

    train_recs, val_recs = split_by_game(records, args.val_frac)

    def _split(recs):
        pair_recs = [r for r in recs if 'pair_id' in r]
        pts   = [r for r in recs if 'pair_id' not in r] + pair_recs
        pairs = build_pairs(pair_recs)
        return pts, pairs
    train_pts, train_pairs = _split(train_recs)
    val_pts,   val_pairs   = _split(val_recs)
    print(f"  Contrastive pairs: train={len(train_pairs)} val={len(val_pairs)}")

    print()
    print_data_stats(train_recs, 'Train')
    print_data_stats(val_recs,   'Val')

    if args.dry_run:
        print("\nDry run complete.")
        return

    # Initialize
    board   = Board()
    encoder = BoardEncoder()
    model   = BoardGNN().to(DEVICE)

    if args.load:
        if os.path.exists(args.load):
            model.load_state_dict(torch.load(args.load, map_location=DEVICE))
            print(f"\nLoaded weights from {args.load}")
        else:
            print(f"\nWarning: --load path {args.load} not found. Starting from scratch.")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.5, verbose=True)

    print(f"\nTraining config:")
    print(f"  Device:     {DEVICE}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR:         {args.lr}")
    print(f"  Epochs:     {args.epochs} (patience={args.patience}, min={args.min_epochs})")
    print(f"  Save to:    {args.save}")

    best_val_loss     = float('inf')
    patience_counter  = 0
    start_time        = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_ctr = run_joint_epoch(
            model, train_pts, train_pairs, encoder, board, optimizer,
            args.batch_size, training=True)
        val_loss, val_acc, val_ctr = run_joint_epoch(
            model, val_pts, val_pairs, encoder, board, optimizer,
            args.batch_size, training=False)
        val_metric = val_loss + CONTRASTIVE_WEIGHT * val_ctr

        mae = mean_abs_output(model, val_pts or val_recs, encoder, board)
        lr  = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:3d}/{args.epochs}: "
              f"train={train_loss:.4f} ({train_acc:.1%})  "
              f"val={val_loss:.4f} ({val_acc:.1%})  "
              f"ctr={train_ctr:.4f}/{val_ctr:.4f}  "
              f"mean_abs={mae:.3f}  lr={lr:.2e}")

        # Collapse warning
        if mae < 0.05:
            print(f"  WARNING: mean_abs_output={mae:.4f} — possible collapse")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), args.save)
            print(f"  ✓ Saved (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience and epoch >= args.min_epochs:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed/60:.1f} minutes")
    print(f"  ✓ Saved (val_metric={best_val_loss:.4f})")
    print(f"Weights saved to: {args.save}")


if __name__ == '__main__':
    main()