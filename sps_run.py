"""Single-piece-save fork driver (new-rule lineage).

The first training run under the adopted single-piece block-save rule
(td-lambda now carries it). Warm-starts + gates from the newest champion and
runs TD self-play with random-FORK reverts between two old-rule champions
{primary, fork-alt} plus epsilon-greedy UNIFORM exploration in GENERATION ONLY
(eval stays greedy -> the promotion gate keeps measuring the deployed policy).

WHY exploration matters here specifically: every warm-start/fork checkpoint was
trained under the OLD whole-block-save rule and DECLINES block-saves. Under the
new gift-ONE rule the single-piece save is roughly break-even (newly viable),
but greedy generation + fork reverts would never TRY it -> the model would keep
playing as if the rule never changed. epsilon-greedy surfaces the now-live
saves so TD can learn their real value.

NOTE: these champions were trained under the old rule, so this is effectively a
FRESH lineage (cross-rule comparisons are not apples-to-apples). Separate
checkpoint namespace (save_prefix='sps') and a disjoint seed_base so it never
collides with the 'td'/'explore'/'variant' runs' files or seeds. Auto-resumes
from sps_iterN.pt / sps_live.pt.

Config via env:
  FORK_PRIMARY  warm-start + gate checkpoint (default aux_iter14)
  FORK_ALT      the fork-alt reverts inject   (default July18_iter10)
  FORK_EXPLORE  '1' (default) = epsilon-greedy on; '0' = pure fork, greedy
  SPS_SMOKE=1   shrink games/eval/iters for a smoke test

Usage:
    caffeinate -i python -u sps_run.py 2>&1 | tee -a sps_run.log
"""
import glob, os, re, time
import torch

import network
from network import BoardGNN
from game_worker import init_pool, shutdown_pool
from td_selfplay_loop import run_td_selfplay
from agent import get_weights

SAVE_PREFIX = 'sps'
WARM_START = os.environ.get('FORK_PRIMARY', 'td_champion_July21_aux_iter14.pt')  # primary + gate
FORK_ALT   = os.environ.get('FORK_ALT', 'td_champion_July18_iter10.pt')          # fork-alt
EXPLORE    = os.environ.get('FORK_EXPLORE', '1') == '1'
TOTAL_ITERATIONS = 24
SEED_BASE = 6_000_000                          # disjoint from td(10k)/explore(5M)/variant(10k)
SMOKE = os.environ.get('SPS_SMOKE') == '1'

# short tags for the two fork points (used as the revert pool keys)
_PRIMARY_TAG = 'aux_iter14' if 'aux_iter14' in WARM_START else (
    'iter14' if 'iter14' in WARM_START else 'primary')
_ALT_TAG = 'iter10' if 'iter10' in FORK_ALT else 'alt'


def eps_schedule(it):
    """it is absolute iteration number (1-based). eps=0.15 for iters 1-5,
    linearly annealed to 0 by iter 12, then pure greedy."""
    if it <= 5:
        return 0.15
    if it >= 12:
        return 0.0
    return 0.15 * (12 - it) / (12 - 5)


def find_latest():
    files = glob.glob(f'{SAVE_PREFIX}_iter*.pt')
    if not files:
        return None, 0
    iters = [int(re.search(rf'{SAVE_PREFIX}_iter(\d+)\.pt$', f).group(1)) for f in files]
    return f'{SAVE_PREFIX}_iter{max(iters)}.pt', max(iters)


def main():
    print(f"network.DEVICE = {network.DEVICE}  SMOKE={SMOKE}  EXPLORE={EXPLORE}")
    print(f"primary/gate = {WARM_START} ({_PRIMARY_TAG}) | fork-alt = {FORK_ALT} ({_ALT_TAG})")
    heur = get_weights('best_weights.json')
    init_pool(n_workers=8 if not SMOKE else 4)

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

    # Champion baseline = the primary fork point (a promotion must BEAT it).
    if os.path.exists(champion_path):
        champion_sd = torch.load(champion_path, map_location='cpu')
        print(f"Loading existing sps champion: {champion_path}")
    else:
        base = BoardGNN(); base.load_state_dict(torch.load(WARM_START, map_location='cpu'))
        champion_sd = base.state_dict()
        print(f"Champion baseline = primary fork point {WARM_START}")

    total = 2 if SMOKE else TOTAL_ITERATIONS
    remaining = total - last_iter
    if remaining <= 0:
        print(f"Already completed {total} iterations."); return

    # Random-fork revert pool: on a non-promoted iteration the live model is
    # rolled back to a per-iteration-deterministic RANDOM choice of the two
    # fork points, re-injecting the alt's policy/calibration into generation.
    # The gate (champion_sd) stays the primary, so a promotion must beat it.
    revert_fork_sds = {
        _PRIMARY_TAG: {k: v.cpu() for k, v in champion_sd.items()},
        _ALT_TAG: torch.load(FORK_ALT, map_location='cpu'),
    }
    print(f"Random-fork reverts across: {sorted(revert_fork_sds)} (gate stays {_PRIMARY_TAG})")

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
        explore_eps_fn=(eps_schedule if EXPLORE else None),
        revert_fork_sds=revert_fork_sds,
        **kw,
    )
    print(f"\nSPS fork run time: {time.time()-t0:.0f}s")
    print(history)
    shutdown_pool()


if __name__ == '__main__':
    main()
