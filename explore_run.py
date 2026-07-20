"""Exploration fork driver (EXPLORATION_SPEC.md section 1).

Forks from the iter14 champion and runs TD self-play with epsilon-greedy
UNIFORM exploration in GENERATION ONLY (eval stays greedy -> the promotion
gate keeps measuring the deployed policy). Separate checkpoint namespace
(save_prefix='explore') and a disjoint seed_base, so it never collides with
the main 'td' run's files or seeds. Auto-resumes from explore_iterN.pt like
the main driver.

Annealing schedule (per EXPLORATION_SPEC.md): eps=0.20 held for the first 5
iterations, linearly annealed to 0 by iter 12, then pure greedy (clean
late-iteration data for final value calibration).

Usage:
    caffeinate -i python -u explore_run.py 2>&1 | tee -a explore_run.log
Env overrides for the smoke test: EXPLORE_SMOKE=1 shrinks games/eval/iters.
"""
import glob, os, re, time
import torch
import network
from network import BoardGNN
from game_worker import init_pool, shutdown_pool
from td_selfplay_loop import run_td_selfplay
from agent import get_weights

SAVE_PREFIX = 'explore'
WARM_START = 'td_champion_July19_iter14.pt'   # fork point / gate: the plateau champion
FORK_ALT = 'td_champion_July18_iter10.pt'     # older champ, sharper on-goal calibration
TOTAL_ITERATIONS = 14
SEED_BASE = 5_000_000                          # disjoint from the 'td' run (10_000)
SMOKE = os.environ.get('EXPLORE_SMOKE') == '1'


def eps_schedule(it):
    """it is absolute iteration number (1-based)."""
    if it <= 5:
        return 0.15
    if it >= 12:
        return 0.0
    return 0.15 * (12 - it) / (12 - 5)         # linear 0.20@it5 -> 0@it12


def find_latest():
    files = glob.glob(f'{SAVE_PREFIX}_iter*.pt')
    if not files:
        return None, 0
    iters = [int(re.search(rf'{SAVE_PREFIX}_iter(\d+)\.pt$', f).group(1)) for f in files]
    return f'{SAVE_PREFIX}_iter{max(iters)}.pt', max(iters)


def main():
    print(f"network.DEVICE = {network.DEVICE}  SMOKE={SMOKE}")
    heur = get_weights('best_weights.json')
    init_pool(n_workers=8 if not SMOKE else 4)   # 8 = benchmarked M3 Pro optimum

    ckpt, last_iter = find_latest()
    champion_path = f'{SAVE_PREFIX}_champion.pt'
    live_path = f'{SAVE_PREFIX}_live.pt'

    model = BoardGNN().to(network.DEVICE)
    if ckpt and os.path.exists(live_path):
        print(f"Resuming: numbering from {ckpt} (iter {last_iter}); live from {live_path}")
        model.load_state_dict(torch.load(live_path, map_location=network.DEVICE))
    else:
        print(f"Fresh fork: warm-starting live model from {WARM_START}")
        model.load_state_dict(torch.load(WARM_START, map_location=network.DEVICE))

    # Champion baseline = the iter14 fork point (a promotion must BEAT iter14).
    if os.path.exists(champion_path):
        champion_sd = torch.load(champion_path, map_location='cpu')
        print(f"Loading existing explore champion: {champion_path}")
    else:
        base = BoardGNN(); base.load_state_dict(torch.load(WARM_START, map_location='cpu'))
        champion_sd = base.state_dict()
        print(f"Champion baseline = fork point {WARM_START}")

    total = 2 if SMOKE else TOTAL_ITERATIONS
    remaining = total - last_iter
    if remaining <= 0:
        print(f"Already completed {total} iterations."); return

    # Random-fork revert pool: on a non-promoted iteration the live model is
    # rolled back to a RANDOM choice of {iter14 (=gate), iter10}, re-injecting
    # iter10's policy + its sharper on-goal value calibration (measured: on/off-
    # goal value gap 0.82 @iter10 vs 0.57 @iter14) into generation. The gate
    # (champion_sd) stays iter14, so a promotion must still beat iter14.
    revert_fork_sds = {
        'iter14': {k: v.cpu() for k, v in champion_sd.items()},
        'iter10': torch.load(FORK_ALT, map_location='cpu'),
    }
    print(f"Random-fork reverts across: {sorted(revert_fork_sds)} "
          f"(gate stays iter14)")

    kw = dict(games_per_iter=300, epochs_per_iter=8, eval_games=200)
    if SMOKE:
        kw = dict(games_per_iter=6, epochs_per_iter=2, eval_games=6)

    t0 = time.time()
    model, champion_sd, history = run_td_selfplay(
        model, champion_sd=champion_sd, heuristic_weights=heur,
        iterations=remaining, start_iter=last_iter + 1,
        lam=0.9, gamma=1.0, lr=5e-5,
        promote_winrate=0.55, replay_iters=3,
        save_prefix=SAVE_PREFIX, seed_base=SEED_BASE,
        explore_eps_fn=eps_schedule,
        revert_fork_sds=revert_fork_sds,
        **kw,
    )
    print(f"\nExplore run time: {time.time()-t0:.0f}s")
    print(history)
    shutdown_pool()


if __name__ == '__main__':
    main()
