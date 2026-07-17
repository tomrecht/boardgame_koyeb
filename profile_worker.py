"""Profile the self-play generation hot path in-process (cProfile), avoiding
py-spy's macOS attach issues. Calls worker_play directly with the same
settings the real pool workers use (CPU, 1 thread, grad disabled), so the
per-function breakdown matches live generation."""
import cProfile
import pstats
import io
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
torch.set_num_threads(1)
torch.set_grad_enabled(False)

import network
network.DEVICE = torch.device('cpu')
from network import BoardGNN
from agent import get_weights
from game_worker import worker_play

N_GAMES = 6

sd = torch.load('best_iter5_m46.pt', map_location='cpu')
heur = get_weights('best_weights.json')

# args: (model_state_dict, heuristic_weights, seed, gnn_is_white, use_heuristic_opp)
# use_heuristic_opp=False => GNN vs GNN, exactly like the on-policy run.
jobs = [(sd, heur, 900_000 + i, i % 2 == 0, False) for i in range(N_GAMES)]


def run():
    total_positions = 0
    for job in jobs:
        recs, winner, score = worker_play(job)
        total_positions += len(recs)
    print(f"{N_GAMES} games, {total_positions} positions")


pr = cProfile.Profile()
pr.enable()
run()
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats(30)
print(s.getvalue())
