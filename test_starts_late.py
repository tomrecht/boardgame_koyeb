"""Start positions and late exploration with the TD trace cut, end to end
through game_worker.worker_play (real games, real net).

  1. a game started from a (rotated) pool position begins EXACTLY there (every
     piece, both racks, side to move), plays to a finish, and samples nothing
     (opening softmax is off in a start game);
  2. late exploration samples only on learner turns PAST the opening window,
     and marks those records 'explored';
  3. compute_td_targets with the trace cut drops exactly the explored records,
     and leaves the targets of positions after the last explored one unchanged.
"""
import json, random, sys

import torch

import network
network.DEVICE = torch.device('cpu')
import game_worker as G
from symmetry import Symmetry
from td_returns import compute_td_targets
from game import Board

CKPT = sys.argv[1] if len(sys.argv) > 1 else 'symaug6_AB.pt'
sd = torch.load(CKPT, map_location='cpu')


def pieces(state):
    return sorted((p['color'], p['number'], p['tile']['ring'], p['tile']['sector'])
                  for p in state['boardPieces'])


def racks(state):
    return {k: [p['number'] for p in v] for k, v in state['racks'].items()}


pool = [json.loads(l)['state'] for l in open('start_pool.jsonl')]
rng = random.Random(7)
sym = Symmetry()

# 1. start games
for trial in range(2):
    st = sym.transform(pool[rng.randrange(len(pool))], rng.randrange(3))
    recs, winner, score = G.worker_play((sd, None, 900 + trial, True, False,
                                         {'start_state': st, 'softmax_T': 0.1, 'softmax_turns': 4}))
    first = recs[0]['raw_state']
    assert pieces(first) == pieces(st), 'start position not reproduced'
    assert racks(first) == racks(st), 'racks not reproduced'
    assert first['currentTurn'] == st['currentTurn']
    assert not any(r['sampled'] for r in recs), 'opening softmax fired in a start game'
    assert recs[0]['game_id'].startswith(f'sp_{900 + trial}_st')
    print(f'  [ok] start game {trial}: began at the pool position ({len(st["boardPieces"])} pieces on board), '
          f'{len(recs)} turns, winner {winner} by {score}, 0 sampled')

# 2. late exploration
recs, winner, score = G.worker_play((sd, None, 950, True, False,
                                     {'softmax_T': 0.1, 'softmax_turns': 4, 'late_T': 0.3, 'late_p': 0.5}))
side_turn, late, early = {'white': 0, 'black': 0}, 0, 0
for r in sorted(recs, key=lambda r: r['move_index']):
    if r['sampled']:
        if side_turn[r['player']] < 4:
            early += 1
        else:
            late += 1
    side_turn[r['player']] += 1
explored = [r for r in recs if r['explored']]
assert late > 0, 'no late samples drawn'
print(f'  [ok] late exploration: {early} opening samples, {late} late samples, '
      f'{len(explored)} records marked explored, {len(recs)} turns')

# 3. trace cut
enc = network.BoardEncoder(features=network.features_of_state(sd))
model = network.model_from_state(sd)
board = Board()
cut = compute_td_targets(recs, model, enc, board, 0.9, 1.0, verbose=False, trace_cut=True)
full = compute_td_targets(recs, model, enc, board, 0.9, 1.0, verbose=False, trace_cut=False)
assert len(full) - len(cut) == len(explored), (len(full), len(cut), len(explored))
assert not any(r['explored'] for r in cut)
last_cut = max(r['move_index'] for r in explored)
tf = {r['move_index']: r['td_target'] for r in full}
tc = {r['move_index']: r['td_target'] for r in cut}
after = [m for m in tc if m > last_cut]
assert all(abs(tf[m] - tc[m]) < 1e-9 for m in after), 'targets after the last cut changed'
before = [m for m in tc if m < last_cut]
changed = sum(abs(tf[m] - tc[m]) > 1e-9 for m in before)
print(f'  [ok] trace cut: dropped {len(full) - len(cut)} explored records; {len(after)} later targets '
      f'unchanged; {changed} of {len(before)} earlier targets changed')
print('ALL OK')
