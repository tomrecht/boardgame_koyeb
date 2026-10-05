"""league_run.py -- TD(lambda) self-play with the three panel-league changes:

  1. FIXED-PANEL GATE (panel_gate.py) instead of the 200-game vs-parent gate:
     a vs-parent PRE-SCREEN (mean margin >= PRESCREEN_BAR) first, then the
     panel played in batches with an O'Brien-Fleming group-sequential test on
     the paired (CRN) panel-margin difference vs the champion: promote / reject
     early when a boundary is crossed, cap at PANEL_MAX_PAIRS per member; and
     no single member may beat the candidate by more than GATE_GUARD.
  2. LEAGUE OPPONENTS in generation: a fraction LEAGUE_FRAC of games is played
     against a frozen panel member (optionally the heuristic agent); only the
     learner's positions are trained on.
  3. SOFTMAX OPENING EXPLORATION (explore.py): in generation only, each side's
     first EXPLORE_TURNS turns sample from softmax(margin / T), T annealed
     linearly from EXPLORE_T to 0 over EXPLORE_ANNEAL iterations. Eval and
     gating stay greedy (asserted per game).

Symmetry augmentation stays ON (the deployed champion was trained with it), and
the agent's hand-coded play rules are OFF for generation and gating.

Usage:   caffeinate -i python -u league_run.py 2>&1 | tee -a league_run.log   (mac)
         nohup python -u league_run.py >> league_run.log 2>&1 &               (linux)
Resume:  rerun the same command; numbering continues from the newest
         <prefix>_iterN.pt and the live weights come from <prefix>_live.pt.
Smoke:   SMOKE=1 python -u league_run.py   (1 iteration, a handful of games)

Every knob is an environment variable (defaults in CONFIG below); the full
config is printed at start and written to <prefix>_config.json.
"""
import glob
import json
import os
import re
import time

import torch

import network
from network import BoardGNN
from game_worker import init_pool, shutdown_pool
from td_selfplay_loop import run_td_selfplay
from agent import get_weights
from symmetry import Symmetry
from panel_gate import PanelGate, PANEL_DEFAULT, PANEL_FIVE
from explore import temperature_for_iter

SMOKE = os.environ.get('SMOKE') == '1'


def _env(name, default, cast):
    v = os.environ.get(name)
    return default if v is None or v == '' else cast(v)


def _bool(v):
    return str(v).lower() in ('1', 'true', 'yes', 'on')


CONFIG = {
    'PREFIX':          _env('PREFIX', 'league', str),
    'WARM_START':      _env('WARM_START', 'symaug_iter6.pt', str),
    'ITERS':           _env('ITERS', 1 if SMOKE else 20, int),
    'GAMES_PER_ITER':  _env('GAMES_PER_ITER', 6 if SMOKE else 300, int),
    'EPOCHS':          _env('EPOCHS', 1 if SMOKE else 8, int),
    'LR':              _env('LR', 5e-5, float),
    'LAM':             _env('LAM', 0.9, float),
    'REPLAY_ITERS':    _env('REPLAY_ITERS', 3, int),
    'SEED_BASE':       _env('SEED_BASE', 11_000_000, int),
    'N_WORKERS':       _env('N_WORKERS', max(1, (os.cpu_count() or 2) - 1), int),
    'SYMAUG':          _env('SYMAUG', True, _bool),
    'HAND_RULES':      _env('HAND_RULES', False, _bool),
    # 1. panel gate
    'PANEL':           _env('PANEL', ','.join(PANEL_DEFAULT), str),   # or 'five'
    'PANEL_MAX_PAIRS': _env('PANEL_MAX_PAIRS', 2 if SMOKE else 100, int),   # cap per member
    'PANEL_BATCH_PAIRS': _env('PANEL_BATCH_PAIRS', 1 if SMOKE else 20, int),  # per look
    'SEQUENTIAL':      _env('SEQUENTIAL', True, _bool),   # False = one look at the cap
    'GATE_ALPHA':      _env('GATE_ALPHA', 0.05, float),   # false-promotion rate
    'GATE_GUARD':      _env('GATE_GUARD', -1.0, float),
    'PRESCREEN_PAIRS': _env('PRESCREEN_PAIRS', 1 if SMOKE else 100, int),  # 0 = off
    'PRESCREEN_BAR':   _env('PRESCREEN_BAR', 0.0, float),
    # 2. league
    'LEAGUE_FRAC':     _env('LEAGUE_FRAC', 0.5 if SMOKE else 0.3, float),
    'LEAGUE_HEURISTIC': _env('LEAGUE_HEURISTIC', False, _bool),
    # 3. exploration
    'EXPLORE_T':       _env('EXPLORE_T', 0.1, float),          # margin points (see RUNBOOK)
    'EXPLORE_TURNS':   _env('EXPLORE_TURNS', 4, int),          # per side
    'EXPLORE_ANNEAL':  _env('EXPLORE_ANNEAL', 10, int),        # iters to reach greedy
    # 4. start positions (features_v2 run): a fraction of games begin from a
    #    pool of real positions (build_start_pool.py), played greedily
    'START_POOL':      _env('START_POOL', 'start_pool.jsonl' if os.path.exists('start_pool.jsonl') else '', str),
    'START_FRAC':      _env('START_FRAC', 0.3, float),
    'START_ROTATE':    _env('START_ROTATE', True, _bool),
    # 5. late exploration: needs the TD trace CUT at sampled moves (td_returns)
    'LATE_T':          _env('LATE_T', 0.0, float),          # margin points; 0 = off
    'LATE_P':          _env('LATE_P', 0.15, float),         # per learner turn past the opening
}
if CONFIG['LATE_T'] > 0:
    # Never explore past the opening without cutting the trace: an off-policy
    # move would bias every earlier position's lambda-return.
    os.environ['TRACE_CUT'] = '1'


def panel_dict(spec):
    """'five' (PANEL_FIVE), 'iter4,iter10' (tags from PANEL_FIVE) or
    'tag=path,...'."""
    if spec.strip() == 'five':
        return dict(PANEL_FIVE)
    out = {}
    for item in filter(None, (x.strip() for x in spec.split(','))):
        if '=' in item:
            t, p = item.split('=', 1)
            out[t.strip()] = p.strip()
        else:
            out[item] = PANEL_FIVE[item]
    return out


def find_latest(prefix):
    files = glob.glob(f'{prefix}_iter*.pt')
    if not files:
        return None, 0
    its = [int(re.search(rf'{prefix}_iter(\d+)\.pt$', f).group(1)) for f in files]
    return f'{prefix}_iter{max(its)}.pt', max(its)


def main():
    C = CONFIG
    P = C['PREFIX']
    panel = panel_dict(C['PANEL'])
    print(f"network.DEVICE = {network.DEVICE}  SMOKE={SMOKE}")
    print('config:', json.dumps(C, indent=1))
    print('panel:', json.dumps(panel, indent=1))
    with open(f'{P}_config.json', 'w') as f:
        json.dump({'config': C, 'panel': panel}, f, indent=1)

    heur = get_weights('best_weights.json')
    init_pool(n_workers=C['N_WORKERS'])

    gate = PanelGate(panel=panel, max_pairs=C['PANEL_MAX_PAIRS'],
                     batch_pairs=C['PANEL_BATCH_PAIRS'], sequential=C['SEQUENTIAL'],
                     alpha=C['GATE_ALPHA'], guard=C['GATE_GUARD'],
                     prescreen_pairs=C['PRESCREEN_PAIRS'],
                     prescreen_bar=C['PRESCREEN_BAR'],
                     cache_path=f'{P}_panel_cache.json',
                     hand_rules=C['HAND_RULES'], prefix=P)

    ckpt, last_iter = find_latest(P)
    champion_path, live_path = f'{P}_champion.pt', f'{P}_live.pt'
    # The checkpoint decides the feature set (features_v2): a v1 warm start
    # trains a v1 net, a distilled A/AB student (distill_v2.py) an A/AB net.
    if ckpt and os.path.exists(live_path):
        print(f"Resuming: numbering from {ckpt} (iter {last_iter}); live from {live_path}")
        start_sd = torch.load(live_path, map_location='cpu')
    else:
        print(f"Fresh: warm-starting from {C['WARM_START']}")
        start_sd = torch.load(C['WARM_START'], map_location='cpu')
    model = network.model_from_state(start_sd).to(network.DEVICE)
    model.train()
    print(f"Feature set: {model.features}")
    if os.path.exists(champion_path):
        champion_sd = torch.load(champion_path, map_location='cpu')
        print(f"Champion: {champion_path}")
    else:
        champion_sd = torch.load(C['WARM_START'], map_location='cpu')
        print(f"Champion: warm start {C['WARM_START']}")

    remaining = C['ITERS'] - last_iter
    if remaining <= 0:
        print(f"Already completed {C['ITERS']} iterations.")
        return

    def gen_cfg_fn(it):
        T = temperature_for_iter(it, C['EXPLORE_T'], C['EXPLORE_ANNEAL'])
        cfg = {'hand_rules': C['HAND_RULES']}
        if T > 0 and C['EXPLORE_TURNS'] > 0:
            cfg.update(softmax_T=T, softmax_turns=C['EXPLORE_TURNS'])
        if C['LATE_T'] > 0 and C['LATE_P'] > 0:
            cfg.update(late_T=C['LATE_T'], late_p=C['LATE_P'],
                       softmax_turns=cfg.get('softmax_turns', C['EXPLORE_TURNS']))
        return cfg

    start_pool = []
    if C['START_POOL'] and C['START_FRAC'] > 0:
        with open(C['START_POOL']) as fh:
            start_pool = [json.loads(line)['state'] for line in fh if line.strip()]
        print(f"Start pool: {len(start_pool)} positions from {C['START_POOL']}, "
              f"{C['START_FRAC']:.0%} of games, rotate {C['START_ROTATE']}")

    def start_cfg_fn(it):
        if not start_pool:
            return None
        return {'frac': C['START_FRAC'], 'pool': start_pool, 'seed': it,
                'rotate': C['START_ROTATE']}

    league_opps = dict(panel)
    if C['LEAGUE_HEURISTIC']:
        league_opps['heuristic'] = 'heuristic'

    def league_cfg_fn(it):
        if C['LEAGUE_FRAC'] <= 0:
            return None
        return {'frac': C['LEAGUE_FRAC'], 'opponents': league_opps, 'seed': it}

    # Calibration benchmark (calib_bench.py): how well the net's judgement of
    # two moves tracks playouts, on owner's 559 disagreement positions. The
    # deployed champion: corr +0.09, slope 0.29 -- higher is better.
    calib_ok = os.path.exists('calib_bench.json')
    if calib_ok:
        import calib_bench
        c, sl, g, n = calib_bench.score_model(network.model_from_state(champion_sd))
        print(f"Calibration benchmark, champion: corr {c:+.3f} slope {sl:+.3f} (n {n})")

    def gate_fn(model, champ_sd, it):
        sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        rep = gate.evaluate(sd, champ_sd, label=f'it{it}')
        if calib_ok:
            c, sl, g, n = calib_bench.score_model(network.model_from_state(sd))
            rep['calib'] = {'corr': round(c, 4), 'slope': round(sl, 4), 'n': n}
            print(f"  calibration benchmark it{it}: corr {c:+.3f} slope {sl:+.3f}")
        t = gate.timing
        if t['games']:
            print(f"  gate games this run: {t['games']}, "
                  f"{t['secs'] / t['games']:.1f}s/game single-core, "
                  f"{t['turns'] / t['games']:.0f} turns/game")
        with open(f'{P}_gate_log.jsonl', 'a') as f:
            f.write(json.dumps({'iter': it, **rep}) + '\n')
        return rep

    t0 = time.time()
    model, champion_sd, history = run_td_selfplay(
        model, champion_sd=champion_sd, heuristic_weights=heur,
        iterations=remaining, start_iter=last_iter + 1,
        games_per_iter=C['GAMES_PER_ITER'], epochs_per_iter=C['EPOCHS'],
        lam=C['LAM'], gamma=1.0, lr=C['LR'], replay_iters=C['REPLAY_ITERS'],
        save_prefix=P, seed_base=C['SEED_BASE'],
        augment=Symmetry() if C['SYMAUG'] else None,
        gen_cfg_fn=gen_cfg_fn, league_cfg_fn=league_cfg_fn, gate_fn=gate_fn,
        start_cfg_fn=start_cfg_fn,
    )
    print(f"\nLeague run time: {time.time() - t0:.0f}s")
    for h in history:
        print({k: v for k, v in h.items() if k != 'gate'})
    shutdown_pool()


if __name__ == '__main__':
    main()
