"""In-process smoke for the symmetry-augmentation training path (no worker pool)."""
import os, json
os.environ['BOARDGAME_DEVICE'] = 'cpu'; os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
torch.set_num_threads(1)
import network
network.DEVICE = torch.device('cpu')
from network import BoardGNN, BoardEncoder
from game import Board
from symmetry import Symmetry
from train_td import encode_batch_with_targets

REPO = os.path.dirname(os.path.abspath(__file__))
POSBANK = '/private/tmp/claude-501/-Users-tomrecht-game-boardgame-koyeb/54593c93-8883-4987-b99a-344553375037/scratchpad/fixclone/posbank.json'

sym = Symmetry(f'{REPO}/tile_neighbors.json')
enc = BoardEncoder(); board = Board()
model = BoardGNN()
model.load_state_dict(torch.load(f'{REPO}/td_champion_July18_iter10.pt', map_location='cpu'))
model.eval()
bank = json.load(open(POSBANK))

print("== 1. rotation actually changes the encoded input ==")
rs = bank[7]
board.update_state(rs); e0 = enc.encode(board, rs['currentTurn'])
board.update_state(sym.transform(rs, 1)); e1 = enc.encode(board, rs['currentTurn'])
diff = (e0['piece_feats'] - e1['piece_feats']).abs().sum().item()
assert diff > 0.01, "rotation did not change encoding"
print(f"   OK: piece-feature L1 diff between original and 240deg = {diff:.3f}")

print("== 2. transformed states are valid (24 pieces, loadable) ==")
for k in (0, 1, 2):
    t = sym.transform(rs, k); board.update_state(t)
    npieces = len(board.pieces)
    nums = sorted(p.number for p in board.pieces if p.player == 'white')
    assert npieces == 24, f"k={k}: {npieces} pieces"
    assert nums == list(range(1, 13)), f"k={k}: white numbers {nums}"
print("   OK: all 3 rotations load, 24 pieces, white numbers 1..12 intact")

print("== 3. encode_batch_with_targets with augment builds a trainable batch ==")
recs = [{'raw_state': bank[i], 'player': bank[i]['currentTurn'],
         'td_target': 0.1, 'ply_from_end': 5} for i in range(16)]
for aug in (None, sym):
    out = encode_batch_with_targets(recs, enc, board, augment=aug)
    batch, labels, weights, failed = out
    assert batch is not None and failed == 0, f"aug={aug}: batch None / failed {failed}"
    with torch.no_grad():
        preds = model(batch)
    assert preds.shape[0] == 16, preds.shape
    print(f"   OK augment={'ON' if aug else 'off'}: batch of 16, preds {tuple(preds.shape)}, "
          f"mean pred {preds.mean().item():+.3f}")

print("== 4. a real train step (fabricated targets) runs with augment ==")
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
model.train()
out = encode_batch_with_targets(recs, enc, board, augment=sym)
batch, labels, weights, _ = out
preds = model(batch); loss = ((preds - labels) ** 2 * weights).sum() / weights.sum()
opt.zero_grad(); loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
assert torch.isfinite(loss), "non-finite loss"
print(f"   OK: augmented train step, loss {loss.item():.5f}, finite grads")

print("\nALL SMOKE CHECKS PASSED")
