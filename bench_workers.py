"""One-off: time self-play generation at different worker counts on this
machine to decide n_workers for the full run. Machine is an M3 Pro: 5
performance cores + 6 efficiency cores = 11 total. 5 and 8 already tested
(8 was much faster); sweeping 9-11 here to find the actual peak before
oversubscription (zero cores left for the main process/OS) turns it
around."""
import time
import torch

import network
from network import BoardGNN
import game_worker
from game_worker import generate_games_parallel, init_pool, shutdown_pool
from agent import get_weights


class _Holder:
    def __init__(self, m):
        self._sd = {k: v.detach().cpu() for k, v in m.state_dict().items()}
    def state_dict(self):
        return self._sd


if __name__ == '__main__':
    N_GAMES = 20
    heur = get_weights('best_weights.json')
    model = BoardGNN().to(network.DEVICE)
    model.load_state_dict(torch.load('best_iter5_m46.pt', map_location=network.DEVICE))

    for nw in (9, 10, 11):
        init_pool(n_workers=nw)
        t0 = time.time()
        generate_games_parallel(_Holder(model), None, n_games=N_GAMES,
                                seed_offset=500_000 + nw * 1000, heuristic_weights=heur,
                                label=f'bench nw={nw}')
        dt = time.time() - t0
        print(f">>> nw={nw}: {N_GAMES} games in {dt:.0f}s "
              f"({dt/N_GAMES:.2f}s/game, {N_GAMES/dt*60:.1f} games/min)")
        shutdown_pool()
