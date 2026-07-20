"""Full Mode A run driver.

Auto-resumes: if td_iterN.pt checkpoints already exist in the cwd, picks up
from the highest N found instead of restarting at iteration 1. Safe to
re-launch after a crash/kill/power loss with no arguments and no manual
bookkeeping. Optimizer momentum is NOT preserved across a resume (Adam state
isn't checkpointed) -- a minor cold-restart cost, not a correctness issue.

Usage:
    caffeinate -i python -u run_full_td.py 2>&1 | tee -a td_run.log
"""
import glob
import os
import re
import time

import torch

import network
from network import BoardGNN
from game_worker import init_pool, shutdown_pool
from td_selfplay_loop import run_td_selfplay
from agent import get_weights

SAVE_PREFIX = 'td'
TOTAL_ITERATIONS = 50
WARM_START = 'best_iter5_m46.pt'


def find_latest_checkpoint():
    files = glob.glob(f'{SAVE_PREFIX}_iter*.pt')
    if not files:
        return None, 0
    iters = [int(re.search(rf'{SAVE_PREFIX}_iter(\d+)\.pt$', f).group(1)) for f in files]
    latest = max(iters)
    return f'{SAVE_PREFIX}_iter{latest}.pt', latest


def main():
    print(f"network.DEVICE = {network.DEVICE}")
    heur_weights = get_weights('best_weights.json')
    init_pool(n_workers=8)

    ckpt_path, last_iter = find_latest_checkpoint()
    champion_path = f'{SAVE_PREFIX}_champion.pt'
    live_path = f'{SAVE_PREFIX}_live.pt'

    model = BoardGNN().to(network.DEVICE)
    if ckpt_path:
        # td_iterN.pt is used for ITERATION NUMBERING only. The weights to
        # continue from are td_live.pt: since the revert-on-failure
        # mechanism, the last iteration checkpoint may hold the FAILED
        # weights of a reverted iteration, while the live model was rolled
        # back to the champion. td_live.pt always records what the next
        # iteration should actually train from.
        if os.path.exists(live_path):
            print(f"Resuming: iter numbering from {ckpt_path} (through iter "
                  f"{last_iter}); live weights from {live_path}")
            model.load_state_dict(torch.load(live_path, map_location=network.DEVICE))
        else:
            print(f"Resuming: loading {ckpt_path} (completed through iter "
                  f"{last_iter}); no {live_path} found (pre-revert-era run)")
            model.load_state_dict(torch.load(ckpt_path, map_location=network.DEVICE))
    else:
        print(f"Fresh start: warm-starting from {WARM_START}")
        model.load_state_dict(torch.load(WARM_START, map_location=network.DEVICE))

    if os.path.exists(champion_path):
        print(f"Loading existing champion: {champion_path}")
        champion_sd = torch.load(champion_path, map_location='cpu')
    else:
        # No promotion has happened yet (this run or a prior one), so the
        # champion is still the warm-start baseline (best_iter5_m46.pt).
        print(f"No champion checkpoint yet; using {WARM_START} as champion baseline")
        base = BoardGNN()
        base.load_state_dict(torch.load(WARM_START, map_location='cpu'))
        champion_sd = base.state_dict()

    remaining = TOTAL_ITERATIONS - last_iter
    if remaining <= 0:
        print(f"Already completed {TOTAL_ITERATIONS} iterations. Nothing to do.")
        return

    t0 = time.time()
    model, champion_sd, history = run_td_selfplay(
        model,
        champion_sd=champion_sd,
        heuristic_weights=heur_weights,
        iterations=remaining,
        start_iter=last_iter + 1,
        games_per_iter=300,
        epochs_per_iter=8,
        # lr halved 1e-4 -> 5e-5 after three consecutive failed updates from
        # the iter4 champion (24.4%, 36.5%, 37.7%) -- same failure signature
        # as the 2e-4-era smoke test (healthy model degraded by one update),
        # and the 600-game pooled attempt ruled out data volume as the cause.
        lam=0.9, gamma=1.0, lr=5e-5,
        eval_games=200, promote_winrate=0.55,
        replay_iters=3,
        save_prefix=SAVE_PREFIX,
        seed_base=10_000,
    )
    print(f"\nTotal run time this session: {time.time()-t0:.0f}s")
    print(history)

    shutdown_pool()


if __name__ == '__main__':
    main()
