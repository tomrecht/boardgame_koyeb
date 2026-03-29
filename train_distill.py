"""
train_distill.py — Train the GNN to mimic the existing weighted-sum agent.

Loads distill_data.pt, trains BoardGNN with mini-batch gradient descent,
saves best model to gnn_weights.pt.

Trains both value head (vs heuristic scores) and aux head
(predicting saved piece balance), weight AUX_LOSS_WEIGHT=0.2.

Usage:
    python3 train_distill.py              # train from scratch
    python3 train_distill.py --resume     # resume from gnn_weights.pt

Output:
    gnn_weights.pt
"""

import argparse
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim

from network import BoardGNN, save_model, DEVICE, AUX_LOSS_WEIGHT

print(f"Training on {DEVICE}")

# -------------------------
# CONFIG
# -------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--resume', action='store_true',
                    help='Resume training from gnn_weights.pt')
args = parser.parse_args()

DATA_FILE    = 'distill_data.pt'
OUTPUT_FILE  = 'gnn_weights.pt'
VAL_FRACTION = 0.1
MAX_SAMPLES  = None
EPOCHS       = 100
BATCH_SIZE   = 256
LR           = 1e-3
LR_DECAY     = 0.5
PATIENCE     = 8
MIN_LR       = 1e-5
SEED         = 42

SCORE_SCALE  = 1000.0


# -------------------------
# AUX TARGET FROM ENCODED
# -------------------------

def aux_target_from_encoded(encoded):
    """
    Derive (my_saved - opp_saved) / 12 directly from piece features.
    Piece features: [24, 8]
      col 0: player_id (0=current player, 1=opponent)
      cols 2-6: status onehot, col 6 = STATUS_SAVED (index 2 + 4)
    First 12 rows = current player pieces, last 12 = opponent pieces.
    """
    pf        = encoded['piece_feats']
    saved_col = 6
    my_saved  = pf[:12, saved_col].sum().item()
    opp_saved = pf[12:, saved_col].sum().item()
    return (my_saved - opp_saved) / 12.0


# -------------------------
# TRAINING
# -------------------------

def train():
    random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Loading data from {DATA_FILE}...")
    all_samples = torch.load(DATA_FILE, map_location='cpu')
    print(f"  {len(all_samples)} samples loaded")

    if MAX_SAMPLES is not None and len(all_samples) > MAX_SAMPLES:
        all_samples = all_samples[:MAX_SAMPLES]
        print(f"  Truncated to {MAX_SAMPLES} samples")

    # Train / val split
    n_val      = max(1, int(len(all_samples) * VAL_FRACTION))
    n_train    = len(all_samples) - n_val
    train_data = all_samples[:n_train]
    val_data   = all_samples[n_train:]
    print(f"  {n_train} train, {n_val} val")

    scores   = [s for _, s in train_data]
    scores_t = torch.tensor(scores)
    print(f"\nScore stats: mean={scores_t.mean():.1f}  std={scores_t.std():.1f}  "
          f"min={scores_t.min():.1f}  max={scores_t.max():.1f}")

    # Pre-compute aux targets
    print("Pre-computing aux targets...")
    train_aux = [aux_target_from_encoded(enc) for enc, _ in train_data]
    val_aux   = [aux_target_from_encoded(enc) for enc, _ in val_data]
    print(f"  aux target stats: "
          f"mean={sum(train_aux)/len(train_aux):.3f}  "
          f"min={min(train_aux):.3f}  max={max(train_aux):.3f}")

    # Build model — resume or fresh
    model = BoardGNN().to(DEVICE)
    if args.resume:
        import os
        if os.path.exists(OUTPUT_FILE):
            state = torch.load(OUTPUT_FILE, map_location=DEVICE)
            missing, unexpected = model.load_state_dict(state, strict=False)
            expected_missing = {'aux_head.0.weight', 'aux_head.0.bias',
                                'aux_head.2.weight', 'aux_head.2.bias'}
            unexpected_missing = set(missing) - expected_missing
            if unexpected_missing:
                print(f"WARNING: unexpected missing keys: {unexpected_missing}")
            else:
                print(f"Resumed from {OUTPUT_FILE} "
                      f"({'aux_head randomly initialized' if missing else 'all keys loaded'})")
        else:
            print(f"WARNING: --resume passed but {OUTPUT_FILE} not found. "
                  f"Starting from scratch.")
    else:
        print("Starting from random initialization.")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_DECAY,
        patience=PATIENCE, min_lr=MIN_LR)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")
    print(f"Training: epochs={EPOCHS} batch_size={BATCH_SIZE} "
          f"lr={LR} patience={PATIENCE}\n")

    # Run one val pass to get starting loss when resuming
    # so we don't accidentally overwrite a good checkpoint with an early bad one
    if args.resume:
        model.eval()
        val_value_loss = 0.0
        val_aux_loss   = 0.0
        val_batches    = 0
        val_indices    = list(range(len(val_data)))
        with torch.no_grad():
            for batch_start in range(0, len(val_data), BATCH_SIZE):
                batch_idx    = val_indices[batch_start: batch_start + BATCH_SIZE]
                encoded_list = []
                value_targets = []
                aux_targets   = []
                for i in batch_idx:
                    encoded, score = val_data[i]
                    encoded_list.append({k: v.to(DEVICE) for k, v in encoded.items()})
                    value_targets.append(score / SCORE_SCALE)
                    aux_targets.append(val_aux[i])
                value_t = torch.tensor(value_targets, dtype=torch.float32, device=DEVICE)
                aux_t   = torch.tensor(aux_targets,   dtype=torch.float32, device=DEVICE)
                value_preds, aux_preds = model.forward_with_aux(encoded_list)
                val_value_loss += criterion(value_preds, value_t).item()
                val_aux_loss   += criterion(aux_preds,   aux_t).item()
                val_batches    += 1
        best_val_loss = (val_value_loss + AUX_LOSS_WEIGHT * val_aux_loss) / val_batches
        print(f"Starting val loss (from resumed weights): {best_val_loss:.4f}")
    else:
        best_val_loss = float('inf')

    best_epoch = 0
    indices    = list(range(n_train))

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # --- TRAIN ---
        model.train()
        random.shuffle(indices)
        train_value_loss = 0.0
        train_aux_loss   = 0.0
        n_batches        = 0

        for batch_start in range(0, n_train, BATCH_SIZE):
            batch_idx     = indices[batch_start: batch_start + BATCH_SIZE]
            encoded_list  = []
            value_targets = []
            aux_targets   = []

            for i in batch_idx:
                encoded, score = train_data[i]
                encoded_list.append({k: v.to(DEVICE) for k, v in encoded.items()})
                value_targets.append(score / SCORE_SCALE)
                aux_targets.append(train_aux[i])

            value_t = torch.tensor(value_targets, dtype=torch.float32, device=DEVICE)
            aux_t   = torch.tensor(aux_targets,   dtype=torch.float32, device=DEVICE)

            optimizer.zero_grad()
            value_preds, aux_preds = model.forward_with_aux(encoded_list)

            value_loss = criterion(value_preds, value_t)
            aux_loss   = criterion(aux_preds,   aux_t)
            loss       = value_loss + AUX_LOSS_WEIGHT * aux_loss

            loss.backward()
            optimizer.step()

            train_value_loss += value_loss.item()
            train_aux_loss   += aux_loss.item()
            n_batches        += 1

        train_value_loss /= n_batches
        train_aux_loss   /= n_batches

        # --- VALIDATE ---
        model.eval()
        val_value_loss = 0.0
        val_aux_loss   = 0.0
        val_batches    = 0
        val_indices    = list(range(len(val_data)))

        with torch.no_grad():
            for batch_start in range(0, len(val_data), BATCH_SIZE):
                batch_idx     = val_indices[batch_start: batch_start + BATCH_SIZE]
                encoded_list  = []
                value_targets = []
                aux_targets   = []

                for i in batch_idx:
                    encoded, score = val_data[i]
                    encoded_list.append({k: v.to(DEVICE) for k, v in encoded.items()})
                    value_targets.append(score / SCORE_SCALE)
                    aux_targets.append(val_aux[i])

                value_t = torch.tensor(value_targets, dtype=torch.float32, device=DEVICE)
                aux_t   = torch.tensor(aux_targets,   dtype=torch.float32, device=DEVICE)

                value_preds, aux_preds = model.forward_with_aux(encoded_list)
                val_value_loss += criterion(value_preds, value_t).item()
                val_aux_loss   += criterion(aux_preds,   aux_t).item()
                val_batches    += 1

        val_value_loss /= val_batches
        val_aux_loss   /= val_batches
        val_combined    = val_value_loss + AUX_LOSS_WEIGHT * val_aux_loss

        scheduler.step(val_combined)

        if val_combined < best_val_loss:
            best_val_loss = val_combined
            best_epoch    = epoch
            save_model(model, OUTPUT_FILE)
            marker = " <- best"
        else:
            marker = ""

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"train_val={train_value_loss:.4f} train_aux={train_aux_loss:.4f}  "
              f"val_val={val_value_loss:.4f} val_aux={val_aux_loss:.4f}  "
              f"lr={lr:.2e}  {time.time()-t0:.1f}s{marker}",
              flush=True)

        if lr <= MIN_LR and epoch > best_epoch + PATIENCE * 2:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"\nBest combined val loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Saved to {OUTPUT_FILE}  (SCORE_SCALE={SCORE_SCALE})")


if __name__ == '__main__':
    train()