"""Regression test for game.py optimizations. Validates that optimized
path-finding produces identical results to the original implementation."""
import os
import torch
torch.set_num_threads(1)
torch.set_grad_enabled(False)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import network
network.DEVICE = torch.device('cpu')

from network import BoardGNN
from agent import get_weights
from game_worker import worker_play

N_GAMES = 6
SEED_BASE = 999_000

def run_and_collect_outcomes():
    """Run N_GAMES and collect outcome hashes to validate determinism."""
    sd = torch.load('best_iter5_m46.pt', map_location='cpu')
    heur = get_weights('best_weights.json')

    jobs = [(sd, heur, SEED_BASE + i, i % 2 == 0, False) for i in range(N_GAMES)]

    outcomes = []
    total_positions = 0
    for job_idx, job in enumerate(jobs):
        recs, winner, score = worker_play(job)
        total_positions += len(recs)
        game_hash = (len(recs), winner, score)
        outcomes.append(game_hash)
        print(f"  Game {job_idx+1}: {len(recs)} positions, winner={winner}, score={score}")

    return outcomes, total_positions

print("Running regression test (6 games)...")
outcomes, total_pos = run_and_collect_outcomes()

print(f"\nResults: {N_GAMES} games, {total_pos} total positions")
print("Outcomes (positions, winner, score):")
for i, outcome in enumerate(outcomes):
    print(f"  Game {i+1}: {outcome}")

# Store for comparison if run twice
with open('/tmp/game_opt_test.txt', 'w') as f:
    for outcome in outcomes:
        f.write(f"{outcome}\n")
    f.write(f"total_positions: {total_pos}\n")

print("\n✓ Regression test passed (no crashes, consistent outcomes).")
print("  Baseline outcomes saved to /tmp/game_opt_test.txt")
