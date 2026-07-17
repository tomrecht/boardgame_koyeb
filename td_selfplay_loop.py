"""
td_selfplay_loop.py — Mode A: iterative TD(lambda) self-play training.

Designed to be driven from a RunPod notebook cell, mirroring how
generate_games_parallel / evaluate_parallel are already called:

    import network, torch
    from network import BoardGNN
    from game_worker import init_pool, shutdown_pool
    from td_selfplay_loop import run_td_selfplay

    init_pool(n_workers=10)                      # CPU worker pool, once
    model = BoardGNN().to(network.DEVICE)
    model.load_state_dict(torch.load('best_iter5_m46.pt', map_location=network.DEVICE))

    champion_sd = {k: v.cpu() for k, v in model.state_dict().items()}
    run_td_selfplay(
        model,
        champion_sd=champion_sd,                 # promotion baseline (iter5)
        heuristic_weights=heur_weights,          # for any heuristic eval/games
        iterations=20,
        games_per_iter=800,
        epochs_per_iter=12,
        lam=0.9, gamma=1.0, lr=2e-4,
        eval_games=200, promote_winrate=0.55,
        save_prefix='td',
    )

Each iteration:
  1. Generate fresh GNN self-play with the CURRENT model (on-policy) on the
     CPU pool. Games terminate under the real rules (win / no-save draw /
     backstop), all scored consistently by game_worker.
  2. Compute TD(lambda) targets in the MAIN process on network.DEVICE (GPU if
     available) using a frozen snapshot of the current model.
  3. Train the live model for a few epochs of weighted-MSE toward those targets
     (target net refreshed each epoch inside the trainer routines).
  4. Evaluate the new model vs the current CHAMPION on the CPU pool.
  5. Promote (update champion) iff win-rate >= promote_winrate. Always
     checkpoint per iteration.

Device discipline: the GPU is used only in the main process (target model +
encoding live on network.DEVICE). Workers are CPU-only (game_worker forces
CUDA_VISIBLE_DEVICES=""). state_dicts are moved to CPU before being shipped to
workers. This keeps all main-process tensors on one device and avoids the
cross-device errors seen previously.
"""
import copy
import time

import torch

import network
from network import BoardEncoder, BoardGNN
from game import Board

from game_worker import generate_games_parallel, evaluate_parallel, init_pool
from train_td import (compute_epoch_targets, run_td_epoch, split_by_game,
                      print_data_stats)


def _cpu_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def train_on_records(model, records, encoder, board, optimizer,
                     epochs, lam, gamma, batch_size=32,
                     uniform_weights=True, val_frac=0.1, seed=0, log=True):
    """Train `model` in place for `epochs` on `records` using TD(lambda)
    targets recomputed (frozen target net) each epoch. Returns best val loss."""
    train_recs, val_recs = split_by_game(records, val_frac=val_frac, seed=seed)
    best_val = float('inf')
    best_state = None
    for ep in range(1, epochs + 1):
        train_t = compute_epoch_targets(model, train_recs, encoder, board,
                                        lam, gamma, verbose=(ep == 1 and log))
        val_t = compute_epoch_targets(model, val_recs, encoder, board,
                                      lam, gamma, verbose=False)
        tr_loss, tr_acc = run_td_epoch(model, train_t, encoder, board, optimizer,
                                       batch_size, training=True,
                                       uniform_weights=uniform_weights)
        val_loss, val_acc = run_td_epoch(model, val_t, encoder, board, optimizer,
                                         batch_size, training=False,
                                         uniform_weights=uniform_weights)
        if log:
            print(f"    epoch {ep:2d} | train {tr_loss:.5f} (sgn {tr_acc:.2f}) "
                  f"| val {val_loss:.5f} (sgn {val_acc:.2f})")
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)        # keep the best epoch in-iter
    return best_val


def run_td_selfplay(model,
                    champion_sd,
                    heuristic_weights,
                    iterations=20,
                    games_per_iter=800,
                    epochs_per_iter=12,
                    lam=0.9,
                    gamma=1.0,
                    lr=2e-4,
                    batch_size=32,
                    uniform_weights=True,
                    eval_games=200,
                    promote_winrate=0.55,
                    use_heuristic_opp=False,
                    save_prefix='td',
                    seed_base=10_000):
    """Iterative TD(lambda) self-play. `model` is trained in place and is the
    current/live network; `champion_sd` is the promotion baseline (start it at
    iter5). Returns (model, champion_sd, history)."""
    assert next(model.parameters()).device == network.DEVICE, \
        f"model must be on {network.DEVICE} (got {next(model.parameters()).device})"

    # Pool must already be running (init_pool called by the caller). If not,
    # start a default one so the loop is still usable standalone.
    try:
        import game_worker
        if game_worker._POOL is None:
            init_pool(10)
    except Exception:
        init_pool(10)

    encoder = BoardEncoder()
    board = Board()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    print(f"TD(lambda) self-play: lambda={lam} gamma={gamma} lr={lr} "
          f"games/iter={games_per_iter} epochs/iter={epochs_per_iter} "
          f"device={network.DEVICE}")

    for it in range(1, iterations + 1):
        t0 = time.time()
        print(f"\n=== Iteration {it}/{iterations} ===")

        # 1. Generate fresh on-policy self-play with the CURRENT model.
        # Workers are CPU-only; ship a CPU state_dict. opp = self (None) unless
        # a heuristic opponent is explicitly requested.
        gen_sd_holder = _CpuSDModel(model)        # tiny adapter exposing .state_dict()
        records = generate_games_parallel(
            gen_sd_holder, None,
            n_games=games_per_iter,
            seed_offset=seed_base + it * games_per_iter,
            heuristic_weights=heuristic_weights,
            use_heuristic_opp=use_heuristic_opp,
            label=f'gen it{it}',
        )
        print_data_stats(records, f'iter {it} generated')

        if len({r['game_id'] for r in records}) < 4:
            print("  Too few games to train this iteration; skipping.")
            continue

        # 2 + 3. TD(lambda) targets (main process, on DEVICE) + train.
        best_val = train_on_records(
            model, records, encoder, board, optimizer,
            epochs=epochs_per_iter, lam=lam, gamma=gamma,
            batch_size=batch_size, uniform_weights=uniform_weights,
            seed=it, log=True)

        # 4. Evaluate the new model vs the current champion (CPU pool).
        champ_sd_cpu = {k: v.cpu() for k, v in champion_sd.items()}
        wr = evaluate_parallel(
            model, champ_sd_cpu,
            n_games=eval_games,
            seed_offset=seed_base + 777 + it,
            heuristic_weights=heuristic_weights,
            label=f'eval it{it} vs champ')

        # 5. Promote on gate; always checkpoint.
        iter_path = f'{save_prefix}_iter{it}.pt'
        torch.save(_cpu_state_dict(model), iter_path)
        promoted = wr >= promote_winrate
        if promoted:
            champion_sd = _cpu_state_dict(model)
            torch.save(champion_sd, f'{save_prefix}_champion.pt')
            print(f"  PROMOTED (win rate {wr:.1%} >= {promote_winrate:.0%}). "
                  f"New champion saved.")
        else:
            print(f"  Not promoted (win rate {wr:.1%} < {promote_winrate:.0%}). "
                  f"Champion unchanged.")

        dt = time.time() - t0
        history.append({'iter': it, 'val_loss': best_val, 'winrate': wr,
                        'promoted': promoted, 'seconds': dt,
                        'positions': len(records)})
        print(f"  iter {it} done in {dt:.0f}s | val {best_val:.5f} | "
              f"wr {wr:.1%} | promoted={promoted}")

    return model, champion_sd, history


class _CpuSDModel:
    """Adapter so generate_games_parallel (which calls model.state_dict() and
    moves it to CPU) gets a CPU snapshot without us touching the live GPU model.
    generate_games_parallel only ever calls .state_dict() on what we pass."""
    def __init__(self, model):
        self._sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    def state_dict(self):
        return self._sd
