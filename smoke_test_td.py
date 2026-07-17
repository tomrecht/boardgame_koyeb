"""Smoke test for Mode A (td_selfplay_loop.run_td_selfplay) on new hardware.

Reduced games_per_iter/iterations to confirm the pool starts, training
completes, and eval runs end-to-end on MPS, and to get a real
per-iteration wall-clock number before committing to a full run.
"""
import time

import torch

import network
from network import BoardGNN
from game_worker import init_pool, shutdown_pool
from td_selfplay_loop import run_td_selfplay
from agent import get_weights


def main():
    print(f"network.DEVICE = {network.DEVICE}")

    heur_weights = get_weights('best_weights.json')

    init_pool(n_workers=5)

    model = BoardGNN().to(network.DEVICE)
    model.load_state_dict(torch.load('best_iter5_m46.pt', map_location=network.DEVICE))

    champion_sd = {k: v.cpu() for k, v in model.state_dict().items()}

    t0 = time.time()
    model, champion_sd, history = run_td_selfplay(
        model,
        champion_sd=champion_sd,
        heuristic_weights=heur_weights,
        iterations=2,
        games_per_iter=50,
        epochs_per_iter=12,
        lam=0.9, gamma=1.0, lr=2e-4,
        eval_games=50,
        promote_winrate=0.55,
        save_prefix='smoke_td',
    )
    print(f"\nTotal smoke test time: {time.time()-t0:.0f}s")
    print(history)

    shutdown_pool()


if __name__ == '__main__':
    main()