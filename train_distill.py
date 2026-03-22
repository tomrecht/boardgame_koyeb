"""
train_distill.py — Train the GNN to mimic the existing weighted-sum agent.

Loads distill_data.pt, trains BoardGNN with mini-batch gradient descent,
saves best model to gnn_weights.pt.

Usage:
    python3 train_distill.py

Output:
    gnn_weights.pt
"""

import random
import time
import torch
import torch.nn as nn
import torch.optim as optim

from network import BoardGNN, collate_batch, save_model, DEVICE

print(f"Training on {DEVICE}")

# -------------------------
# CONFIG
# -------------------------

DATA_FILE    = 'distill_data.pt'
OUTPUT_FILE  = 'gnn_weights.pt'
VAL_FRACTION = 0.1
MAX_SAMPLES  = None      # None = use all; set to 2000 for quick POC
EPOCHS       = 50
BATCH_SIZE   = 32
LR           = 1e-3
LR_DECAY     = 0.5
PATIENCE     = 5
MIN_LR       = 1e-5
SEED         = 42

# Scores from evaluate() have std ~916; dividing by 1000 keeps targets near [-1,1]
SCORE_SCALE  = 1000.0


# -------------------------
# TRAINING
# -------------------------

def train():
    random.seed(SEED)
    torch.manual_seed(SEED)

    # Load data — keep on CPU, move to device per batch
    print(f"Loading data from {DATA_FILE}...")
    all_samples = torch.load(DATA_FILE, map_location='cpu')
    print(f"  {len(all_samples)} samples loaded")

    if MAX_SAMPLES is not None and len(all_samples) > MAX_SAMPLES:
        all_samples = all_samples[:MAX_SAMPLES]
        print(f"  Truncated to {MAX_SAMPLES} samples")

    # Train / val split (data pre-shuffled by generate_distill_data.py)
    n_val      = max(1, int(len(all_samples) * VAL_FRACTION))
    n_train    = len(all_samples) - n_val
    train_data = all_samples[:n_train]
    val_data   = all_samples[n_train:]
    print(f"  {n_train} train, {n_val} val")

    scores = [s for _, s in train_data]
    scores_t = torch.tensor(scores)
    print(f"\nScore stats: mean={scores_t.mean():.1f}  std={scores_t.std():.1f}  "
          f"min={scores_t.min():.1f}  max={scores_t.max():.1f}")

    model     = BoardGNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_DECAY, patience=PATIENCE, min_lr=MIN_LR)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")
    print(f"Training: {EPOCHS} epochs, batch_size={BATCH_SIZE}\n")

    best_val_loss = float('inf')
    best_epoch    = 0
    indices       = list(range(n_train))

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # --- TRAIN ---
        model.train()
        random.shuffle(indices)
        train_loss_sum = 0.0
        n_batches = 0

        for batch_start in range(0, n_train, BATCH_SIZE):
            batch_idx = indices[batch_start: batch_start + BATCH_SIZE]
            encoded_list = []
            targets      = []
            for i in batch_idx:
                encoded, score = train_data[i]
                # Move each encoded dict to device
                encoded_list.append({k: v.to(DEVICE) for k, v in encoded.items()})
                targets.append(score / SCORE_SCALE)

            target_t = torch.tensor(targets, dtype=torch.float32, device=DEVICE)
            batch    = collate_batch(encoded_list)

            optimizer.zero_grad()
            preds = model(encoded_list)          # [batch_size]
            loss  = criterion(preds, target_t)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            n_batches      += 1

        train_loss = train_loss_sum / n_batches

        # --- VALIDATE ---
        model.eval()
        val_loss_sum = 0.0
        val_batches  = 0
        val_indices  = list(range(len(val_data)))

        with torch.no_grad():
            for batch_start in range(0, len(val_data), BATCH_SIZE):
                batch_idx    = val_indices[batch_start: batch_start + BATCH_SIZE]
                encoded_list = []
                targets      = []
                for i in batch_idx:
                    encoded, score = val_data[i]
                    encoded_list.append({k: v.to(DEVICE) for k, v in encoded.items()})
                    targets.append(score / SCORE_SCALE)
                target_t  = torch.tensor(targets, dtype=torch.float32, device=DEVICE)
                preds     = model(encoded_list)
                val_loss_sum += criterion(preds, target_t).item()
                val_batches  += 1

        val_loss = val_loss_sum / val_batches
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            save_model(model, OUTPUT_FILE)
            marker = " <- best"
        else:
            marker = ""

        lr  = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{EPOCHS}  train={train_loss:.4f}  "
              f"val={val_loss:.4f}  lr={lr:.2e}  {time.time()-t0:.1f}s{marker}",
              flush=True)

        if lr <= MIN_LR and epoch > best_epoch + PATIENCE * 2:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"\nBest val loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Saved to {OUTPUT_FILE}  (SCORE_SCALE={SCORE_SCALE})")


if __name__ == '__main__':
    train()
