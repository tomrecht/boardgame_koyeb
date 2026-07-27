"""Validate the D3 automorphism before training on it.

STRUCTURAL: sigma is a bijection, sigma^3 = identity, preserves tile type, maps
each goal tile number n to pi(n), and preserves adjacency (graph automorphism).
EMPIRICAL value-preservation gate: with iter10, V(pos) vs V(rotate(pos)) over
many posbank positions -- if the mean |gap| is small the rotation is a good
enough game symmetry to augment with; if not, do NOT train on it.
"""
import os, json, statistics
os.environ['BOARDGAME_DEVICE'] = 'cpu'; os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
torch.set_num_threads(1); torch.set_grad_enabled(False)
import network
network.DEVICE = torch.device('cpu')
from network import BoardGNN
from game import Board
from agent_gnn import GNNAgent
from symmetry import Symmetry, PI

REPO = os.path.dirname(os.path.abspath(__file__))
ITER10 = f'{REPO}/td_champion_July18_iter10.pt'
POSBANK = '/private/tmp/claude-501/-Users-tomrecht-game-boardgame-koyeb/54593c93-8883-4987-b99a-344553375037/scratchpad/fixclone/posbank.json'
NUM_PIECES = 12

sym = Symmetry(f'{REPO}/tile_neighbors.json')
sigma = sym.sigma
TN = json.load(open(f'{REPO}/tile_neighbors.json'))


def key(r, s): return f'ring{r}_sector{s}'
def parse(k):
    r, s = k.replace('ring', '').replace('sector', '').split('_'); return int(r), int(s)


print("=== STRUCTURAL ===")
# 1. bijection
assert len(set(sigma.values())) == len(sigma), "sigma not a bijection"
print(f"  sigma bijection over {len(sigma)} real tiles: OK")
# 2. sigma^3 = identity
bad = [t for t in sigma if sigma[sigma[sigma[t]]] != t]
assert not bad, f"sigma^3 != id for {bad[:5]}"
print("  sigma^3 = identity: OK")
# 3. type preservation + 4. goal number -> pi(n)
type_fail, num_fail = [], []
for k, v in TN.items():
    r, s = parse(k); img = sigma[(r, s)]
    ik = key(*img)
    if TN[ik]['type'] != v['type']:
        type_fail.append((k, ik))
    if v['type'] == 'save':
        if TN[ik].get('number') != PI[v['number']]:
            num_fail.append((k, v.get('number'), TN[ik].get('number'), PI[v['number']]))
assert not type_fail, f"type not preserved: {type_fail[:5]}"
assert not num_fail, f"goal number != pi(n): {num_fail[:5]}"
print(f"  tile type preserved (all {len(TN)}): OK")
print(f"  goal-tile number maps by pi (2->1->3, 4->6->5): OK")
# 5. adjacency preservation
adj_fail = 0
for k, v in TN.items():
    r, s = parse(k); img = sigma[(r, s)]
    nb = {tuple(sigma[(n['ring'], n['sector'])]) for n in v['neighbors']
          if (n['ring'], n['sector']) in sigma}
    nb_img = {(n['ring'], n['sector']) for n in TN[key(*img)]['neighbors']
              if (n['ring'], n['sector']) in sigma}
    if nb != nb_img:
        adj_fail += 1
print(f"  adjacency preserved: {len(TN)-adj_fail}/{len(TN)} tiles"
      + (" OK" if adj_fail == 0 else f"  ({adj_fail} FAIL)"))

print("\n=== EMPIRICAL value-preservation (iter10) ===")
m = BoardGNN(); m.load_state_dict(torch.load(ITER10, map_location='cpu'), strict=False); m.eval()
agent = GNNAgent(model=m)
board = Board()
bank = json.load(open(POSBANK))


def val(rs):
    board.update_state(rs)
    _f, info = agent.evaluate(board, rs['currentTurn'])
    return info['gnn_raw'] * NUM_PIECES


gaps = {1: [], 2: []}
identity_fail = 0
for i, rs in enumerate(bank):
    try:
        v0 = val(rs)
        # sanity: k=0 (identity) must reproduce exactly
        if abs(val(sym.transform(rs, 0)) - v0) > 1e-6:
            identity_fail += 1
        for k in (1, 2):
            gaps[k].append(val(sym.transform(rs, k)) - v0)
    except Exception as e:
        if i < 5:
            print("  transform/eval error:", e)
print(f"  positions: {len(gaps[1])}  (identity mismatches: {identity_fail})")
for k in (1, 2):
    g = gaps[k]; ab = [abs(x) for x in g]
    print(f"  k={k} (rotate {240*k}deg): signed mean {statistics.mean(g):+.4f} | "
          f"mean|gap| {statistics.mean(ab):.4f} | median|gap| {statistics.median(ab):.4f} | "
          f"p90|gap| {sorted(ab)[int(0.9*len(ab))]:.4f} | max {max(ab):.4f}")
print("\nGATE: mean|gap| well under ~0.1 margin units (cf. the +0.002 diff-in-diff\n"
      "finding) => rotation preserves value => SAFE to augment. Large gaps => the\n"
      "dice dual-role breaks it => do NOT train on this augmentation.")
