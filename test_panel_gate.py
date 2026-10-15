"""test_panel_gate.py -- the cheaper panel gate.

Part A (simulation): synthetic paired margins d ~ N(edge, 1.8) (paired-margin
SD from ARCHIVE). The group-sequential O'Brien-Fleming test must hold its
false-promotion rate at the nominal alpha at edge 0, and we report power and
the expected panel units used at edges 0 / 0.3 / 0.5 against the fixed design.
A naive CI re-checked at every look is shown for contrast (it inflates).

Part B (branches): PanelGate driven end to end with a SYNTHETIC game runner
(no network), showing each branch fires: pre-screen reject, early promote,
early reject, cap hit, guard fail -- and that the champion's games are cached.

Run: python test_panel_gate.py [n_sims]
"""
import math
import random
import sys

import numpy as np
import torch

from panel_gate import (PanelGate, SequentialTest, _norm_ppf, CONFIRM_SEED_BASE,
                        CONFIRM_STRIDE, PANEL_SEED_BASE)

SD = 1.8


def simulate(seq, look_units, edge, n_sims, rng, naive=False):
    """Return (P(promote), mean units used, P(early stop)) for one design."""
    N = look_units[-1]
    d = rng.normal(edge, SD, size=(n_sims, N))
    cs = np.cumsum(d, axis=1)
    cs2 = np.cumsum(d * d, axis=1)
    decided = np.zeros(n_sims, dtype=bool)
    promote = np.zeros(n_sims, dtype=bool)
    used = np.full(n_sims, N)
    z_naive = _norm_ppf(1 - seq.alpha)
    for k, n in enumerate(look_units):
        m = cs[:, n - 1] / n
        var = (cs2[:, n - 1] - n * m * m) / (n - 1)
        t = m / np.sqrt(var / n)
        b = z_naive if naive else seq.boundary(k, n)
        last = k == len(look_units) - 1
        up = (t >= b) & ~decided
        lo = ((t <= -b) if not naive else (t <= -b)) & ~decided
        promote |= up
        stop = up | lo
        used[stop] = n
        decided |= stop
        if last:
            used[~decided] = n
    early = np.mean(used < N)
    return promote.mean(), used.mean(), early


def part_a(n_sims):
    print('Part A: group-sequential test, synthetic paired margins (SD 1.8)')
    rng = np.random.default_rng(7)
    designs = [('default: 3 members, batch 20, cap 100 pairs', 3, 20, 100),
               ('5 members, batch 20, cap 100 pairs', 5, 20, 100)]
    ok = True
    for name, M, b, cap in designs:
        pairs = list(range(b, cap, b)) + [cap]
        units = [p * M for p in pairs]
        seq = SequentialTest(units, alpha=0.05)
        N = units[-1]
        print(f'  {name}: looks at {units} units, OBF c={seq.c:.3f}, '
              f'first-look t bound {seq.boundary(0, units[0]):.2f}, final {seq.boundary(len(units) - 1, N):.2f}')
        print(f'    {"edge":>6}{"P(promote)":>12}{"E[units]":>10}{"E[games]":>10}'
              f'{"vs fixed":>10}{"early stop":>12}{"fixed-N power":>15}')
        for edge in (0.0, 0.3, 0.5):
            p, u, e = simulate(seq, units, edge, n_sims, rng)
            # fixed-N one-look test at the same alpha, for reference
            pf, _, _ = simulate(SequentialTest([N], 0.05), [N], edge, n_sims, rng)
            print(f'    {edge:6.1f}{p:12.3f}{u:10.0f}{2 * u:10.0f}{u / N:10.0%}{e:12.0%}{pf:15.3f}')
            if edge == 0.0:
                se = math.sqrt(0.05 * 0.95 / n_sims)
                if p > 0.05 + 3 * se:
                    print(f'    FAIL: false-promotion {p:.3f} above nominal 0.05 (+3se {3 * se:.3f})')
                    ok = False
        pn, un, _ = simulate(seq, units, 0.0, n_sims, rng, naive=True)
        print(f'    contrast -- naive 95% one-sided CI re-checked at every look, edge 0: '
              f'false-promotion {pn:.3f} (inflated)')
    return ok


# ---------------- Part B: branches with a synthetic game runner ----------------
def sd_of(x):
    return {'w': torch.tensor([float(x)])}


class FakeGate(PanelGate):
    """PanelGate with the game runner replaced. Every model is identified by
    its weight value; `strength[(value, tag)]` is its mean unit margin against
    that member ('parent' for the pre-screen). Per-(member, seed, colour) noise
    is SHARED across models -- common random numbers, as with real dice."""

    def __init__(self, strength, **kw):
        self.strength = strength
        self.weights_by_path = {}
        super().__init__(panel={'symaug6': 'A', 'iter10': 'B', 'iter14': 'C'},
                         check_paths=False, cache_path=None, **kw)

    def save_weights(self, sd, path):
        self.weights_by_path[path] = float(sd['w'][0])

    def run_tasks(self, tasks):
        out = []
        for t in tasks:
            cand_path, _h, tag, member_path, seed, cw = t[:6]
            v = self.weights_by_path[cand_path]
            rng = random.Random(f'{tag}|{seed}|{cw}')
            noise = rng.gauss(0, 2.5)
            mu = self.strength[(v, tag)]
            # colour-swapped pair: (mu + noise, mu - noise) -> unit mean mu +- small
            m = mu + (noise if cw else -noise) + random.Random(f'{v}|{tag}|{seed}|{cw}').gauss(0, 1.2)
            out.append({'tag': tag, 'seed': seed, 'cand_white': cw,
                        'margin': max(-12, min(12, round(m))), 'secs': 0, 'turns': 0})
        self.timing['games'] += len(out)
        return out


def S(champ, cand, parent):
    """strength table: champion value 0, candidate value 1."""
    st = {}
    for tag, x in champ.items():
        st[(0.0, tag)] = x
    for tag, x in cand.items():
        st[(1.0, tag)] = x
    st[(1.0, 'parent')] = parent
    return st


def part_b():
    print('\nPart B: gate branches with a synthetic game runner '
          '(3 members, batch 5 pairs, cap 20 pairs, pre-screen 10 pairs)')
    champ = {'symaug6': 0.0, 'iter10': 0.0, 'iter14': 0.0}
    cases = [
        ('pre-screen reject', S(champ, {t: 2.0 for t in champ}, -1.5), 'prescreen_reject'),
        ('early promote',     S(champ, {t: 2.0 for t in champ}, 1.0), 'promote'),
        ('early reject',      S(champ, {t: -2.0 for t in champ}, 2.0), 'reject'),
        ('cap hit',           S(champ, {t: 0.0 for t in champ}, 2.0), 'cap'),
        ('guard fail',        S(champ, {'symaug6': 4.0, 'iter10': 4.0, 'iter14': -2.5}, 1.0), 'guard_fail'),
    ]
    ok = True
    for name, strength, want in cases:
        g = FakeGate(strength, max_pairs=20, batch_pairs=5, prescreen_pairs=10, guard=-1.0)
        rep = g.evaluate(sd_of(1.0), sd_of(0.0), label=name)
        looks = len(rep['looks'])
        good = rep['outcome'] == want
        if want in ('promote', 'reject'):
            good = good and looks < len(g.look_pairs)          # stopped EARLY
        if want == 'cap':
            good = good and looks == len(g.look_pairs)
        print(f'  => {name}: outcome {rep["outcome"]} after {looks} look(s) of '
              f'{len(g.look_pairs)}, candidate panel games {rep["panel_games"]}, '
              f'pre-screen {rep["prescreen_games"]}: {"ok" if good else "FAIL"}\n')
        ok &= good
    # champion cache: a second candidate re-uses the champion's games
    g = FakeGate(S(champ, {t: 2.0 for t in champ}, 1.0), max_pairs=20, batch_pairs=5,
                 prescreen_pairs=0)
    r1 = g.evaluate(sd_of(1.0), sd_of(0.0), label='first')
    r2 = g.evaluate(sd_of(1.0), sd_of(0.0), label='again')
    good = r1['champ_games_new'] > 0 and r2['champ_games_new'] == 0 and r2['panel_games'] == 0
    print(f'  => cache: champion games first {r1["champ_games_new"]}, second {r2["champ_games_new"]}, '
          f'candidate replays {r2["panel_games"]}: {"ok" if good else "FAIL"}')
    ok &= good
    return ok and part_c()


class LuckyGate(FakeGate):
    """Candidate 1.0 is strong ONLY on the original panel seeds (a lucky draw);
    on any other seed block it is `elsewhere`."""

    def __init__(self, strength, elsewhere, **kw):
        self.elsewhere = elsewhere
        super().__init__(strength, **kw)

    def run_tasks(self, tasks):
        out = super().run_tasks(tasks)
        for t, r in zip(tasks, out):
            v = self.weights_by_path[t[0]]
            if v == 1.0 and t[2] != 'parent' and t[4] >= CONFIRM_SEED_BASE:
                r['margin'] += self.elsewhere - self.strength[(v, t[2])]
        return out


def part_c():
    print('\nPart C: fresh-seed confirmation of a panel PROMOTE')
    champ = {'symaug6': 0.0, 'iter10': 0.0, 'iter14': 0.0}
    strong = S(champ, {t: 2.0 for t in champ}, 1.0)
    ok = True
    # off: behaves as before
    g = FakeGate(strong, max_pairs=20, batch_pairs=5, prescreen_pairs=0)
    r = g.evaluate(sd_of(1.0), sd_of(0.0), label='confirm off')
    good = r['outcome'] == 'promote' and r['confirm'] is None
    print(f'  => confirm off: {r["outcome"]}: {"ok" if good else "FAIL"}'); ok &= good
    # real improver: passes, panel moves to the confirmation block
    g = FakeGate(strong, max_pairs=20, batch_pairs=5, prescreen_pairs=0, confirm_pairs=20)
    r = g.evaluate(sd_of(1.0), sd_of(0.0), label='real')
    good = (r['outcome'] == 'promote' and r['promote'] and r['confirm']['passed']
            and g.seeds[0] == CONFIRM_SEED_BASE and r['confirm']['games'] == 2 * 20 * 3 * 2)
    print(f'  => real improver: {r["outcome"]}, confirm diff {r["confirm"]["diff"]:+.2f}, '
          f'panel block {g.seeds[0]}: {"ok" if good else "FAIL"}'); ok &= good
    # the promoted model is now the champion: its games on the new block are cached
    st = dict(strong); st[(2.0, 'symaug6')] = st[(2.0, 'iter10')] = st[(2.0, 'iter14')] = 2.0
    g.strength = st
    r2 = g.evaluate(sd_of(2.0), sd_of(1.0), label='next')
    good = r2['champ_games_new'] == 0
    print(f'  => new champion baseline cached on the new block: champion games played '
          f'{r2["champ_games_new"]}: {"ok" if good else "FAIL"}'); ok &= good
    # lucky candidate: promotes on the panel seeds, fails on fresh ones, not kept
    g = LuckyGate(strong, -0.5, max_pairs=20, batch_pairs=5, prescreen_pairs=0, confirm_pairs=20)
    r = g.evaluate(sd_of(1.0), sd_of(0.0), label='lucky')
    good = (r['outcome'] == 'confirm_fail' and not r['promote'] and not r['keep']
            and g.seeds[0] == PANEL_SEED_BASE)
    print(f'  => lucky: {r["outcome"]}, confirm diff {r["confirm"]["diff"]:+.2f}, '
          f'panel block {g.seeds[0]}: {"ok" if good else "FAIL"}'); ok &= good
    # a second attempt uses a NEW block
    r = g.evaluate(sd_of(1.0), sd_of(0.0), label='lucky again')
    good = r['confirm']['seed_base'] == CONFIRM_SEED_BASE + CONFIRM_STRIDE
    print(f'  => second attempt block {r["confirm"]["seed_base"]}: {"ok" if good else "FAIL"}')
    return ok and good


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    a = part_a(n)
    b = part_b()
    print('\nALL OK' if a and b else '\nFAILURES')
    sys.exit(0 if a and b else 1)
