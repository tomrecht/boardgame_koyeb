"""Sequential pre-screen vs the fixed one, simulated (synthetic game runner from
test_panel_gate.FakeGate, fresh noise per trial). For candidates of known true
margin vs the champion: how often each version rejects, and how many games it
plays. The sequential version must not reject an EQUAL candidate more often
than the fixed one, and should save games on clear cases.

    python3 test_prescreen_seq.py [trials=600]
"""
import random, sys

import torch

from test_panel_gate import FakeGate, sd_of


class TrialGate(FakeGate):
    trial = 0

    def run_tasks(self, tasks):
        out = []
        for t in tasks:
            cand_path, _h, tag, member_path, seed, cw = t[:6]
            v = self.weights_by_path[cand_path]
            rng = random.Random(f'{self.trial}|{tag}|{seed}|{cw}')
            noise = rng.gauss(0, 2.5)
            mu = self.strength[(v, tag)]
            m = mu + (noise if cw else -noise) + random.Random(
                f'{self.trial}|{v}|{tag}|{seed}|{cw}').gauss(0, 1.2)
            out.append({'tag': tag, 'seed': seed, 'cand_white': cw,
                        'margin': max(-12, min(12, round(m))), 'secs': 0, 'turns': 0})
        self.timing['games'] += len(out)
        return out


def run(edge, sequential, trials):
    rej, games = 0, 0
    st = {(1.0, 'parent'): edge}
    g = TrialGate(st, prescreen_pairs=100, prescreen_bar=0.0, prescreen_seq=sequential,
                  prescreen_batch_pairs=20, max_pairs=20, batch_pairs=5)
    import builtins
    _print = builtins.print
    builtins.print = lambda *a, **k: None          # the gate is chatty
    try:
        for t in range(trials):
            g.trial = t
            passed, m, n = g.prescreen(sd_of(1.0), sd_of(0.0), 'sim')
            rej += (not passed)
            games += n
    finally:
        builtins.print = _print
    return rej / trials, games / trials


def main():
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    print(f'{trials} trials per cell; 100 seed pairs (200 games) at most; looks every 20 pairs')
    print(f'{"true margin":>12} | {"fixed: reject":>14} {"games":>6} | {"sequential: reject":>19} {"games":>6}')
    worst = 0.0
    for edge in (-0.5, -0.3, -0.15, 0.0, 0.15, 0.3, 0.5):
        rf, gf = run(edge, False, trials)
        rs, gs = run(edge, True, trials)
        print(f'{edge:>+12.2f} | {rf:>14.1%} {gf:>6.0f} | {rs:>19.1%} {gs:>6.0f}')
        if edge == 0.0:
            worst = rs - rf
    print(f'equal candidate: sequential rejects {worst:+.1%} vs fixed '
          f'({"OK" if worst <= 0.03 else "TOO HIGH"})')


if __name__ == '__main__':
    main()
