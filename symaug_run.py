"""Symmetry-augmented TD fine-tuning to build a GENERALIST goal-pair blocker.

D3 board-rotation augmentation (symmetry.py) rotates each TRAINING position to a
random one of the 3 symmetric variants, balancing the goal pairs 2&4 / 1&6 / 3&5
by construction and synthesizing the rare 1&6 / 3&5 experience the base model
barely generates -- transferring its blocking competence onto all three pairs
(owner's hypothesis: a generalist blocker is stronger, esp. vs a human).

Runs on `main` against the CURRENT engine, i.e. the new **single-piece
block-save** rule (peel one, gift one -- commit 0682ba8), NOT the old
whole-block save the earlier symmetry-aug branch used. Base checkpoints predate
the rule; they adapt via training under it (the gate is fair -- both sides play
the new rule).

Gate = the warm-start base (a promotion must beat it). Isolated: save_prefix=
'symaug', disjoint seed_base, auto-resume from symaug_live.pt.

Success metric is BLOCK GENERALIZATION (per-goal INF-denial across all three
pairs in self-play), not just win-rate -- measure that on symaug self-play once
it has run a few iterations.

Usage: caffeinate -i python -u symaug_run.py 2>&1 | tee -a symaug_run.log
Env: SYMAUG_WARM_START overrides the base; EXPLORE_SMOKE=1 shrinks the run.
"""
import glob, os, re, time
import torch
import network
from network import BoardGNN
from game_worker import init_pool, shutdown_pool
from td_selfplay_loop import run_td_selfplay
from agent import get_weights
from symmetry import Symmetry

SAVE_PREFIX = 'symaug'
# Default base = iter10 (the heavy-blocking model the aug is meant to generalise;
# see symmetry-aug notes). Override with SYMAUG_WARM_START, e.g. the current
# champion td_champion_July21_aux_iter14.pt.
WARM_START = os.environ.get('SYMAUG_WARM_START', 'td_champion_July18_iter10.pt')
TOTAL_ITERATIONS = 14
SEED_BASE = 9_000_000                     # disjoint from td(10k)/explore(5M)/aux(7M)
SMOKE = os.environ.get('EXPLORE_SMOKE') == '1'


def find_latest():
    files = glob.glob(f'{SAVE_PREFIX}_iter*.pt')
    if not files:
        return None, 0
    iters = [int(re.search(rf'{SAVE_PREFIX}_iter(\d+)\.pt$', f).group(1)) for f in files]
    return f'{SAVE_PREFIX}_iter{max(iters)}.pt', max(iters)


def main():
    print(f"network.DEVICE = {network.DEVICE}  SMOKE={SMOKE}  base={WARM_START}")
    heur = get_weights('best_weights.json')
    init_pool(n_workers=8 if not SMOKE else 4)
    sym = Symmetry()
    print("D3 symmetry augmentation ON (rotate each training position "
          "0/240/480 deg; targets on original).")

    ckpt, last_iter = find_latest()
    champion_path = f'{SAVE_PREFIX}_champion.pt'
    live_path = f'{SAVE_PREFIX}_live.pt'

    model = BoardGNN().to(network.DEVICE)
    if ckpt and os.path.exists(live_path):
        print(f"Resuming: numbering from {ckpt} (iter {last_iter}); live from {live_path}")
        model.load_state_dict(torch.load(live_path, map_location=network.DEVICE))
    else:
        print(f"Fresh: warm-starting from {WARM_START}")
        model.load_state_dict(torch.load(WARM_START, map_location=network.DEVICE))

    # Gate = the base (a promotion must beat the warm-start champion).
    if os.path.exists(champion_path):
        champion_sd = torch.load(champion_path, map_location='cpu')
        print(f"Loading existing symaug champion: {champion_path}")
    else:
        base = BoardGNN(); base.load_state_dict(torch.load(WARM_START, map_location='cpu'))
        champion_sd = base.state_dict()
        print(f"Champion baseline = base {WARM_START}")

    total = 2 if SMOKE else TOTAL_ITERATIONS
    remaining = total - last_iter
    if remaining <= 0:
        print(f"Already completed {total} iterations."); return

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
        augment=sym,
        **kw,
    )
    print(f"\nSymaug run time: {time.time()-t0:.0f}s")
    print(history)
    shutdown_pool()


if __name__ == '__main__':
    main()
