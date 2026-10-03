"""test_explore.py -- softmax opening exploration: sampling rules, annealing,
and the GREEDY-EVAL tripwire (the assertion fires if a sample is drawn during a
gate game). Fast: no network, no real game.  Run: python test_explore.py"""
import math
import random

import explore
from explore import softmax_pick, temperature_for_iter, DRAW_PAIR

SCALE = 1000.0          # agent_gnn.SCORE_SCALE
A, B, C = ('a',), ('b',), ('c',)


def ranked(*margins):
    """[(score, pair)] best-first from margins in game points."""
    pairs = [A, B, C][:len(margins)]
    return sorted(((m / 12 * SCALE, p) for m, p in zip(margins, pairs)),
                  key=lambda x: -x[0])


def test_distribution():
    rng = random.Random(0)
    T = 0.5
    r = ranked(0.0, -0.5, -3.0)      # B is 0.5 points worse, C 3 points worse
    n = 20000
    cnt = {A: 0, B: 0, C: 0}
    expl = 0
    for _ in range(n):
        p, e = softmax_pick(r, T, rng, SCALE)
        cnt[p] += 1
        expl += e > 0
    w = [1.0, math.exp(-1.0), math.exp(-6.0)]
    exp = [x / sum(w) for x in w]
    for p, e in zip((A, B, C), exp):
        assert abs(cnt[p] / n - e) < 0.015, (p, cnt[p] / n, e)
    assert expl == cnt[B] + cnt[C]
    print(f'  [ok] softmax frequencies {[round(cnt[p]/n, 3) for p in (A, B, C)]} '
          f'vs expected {[round(e, 3) for e in exp]}')


def test_rules():
    rng = random.Random(1)
    # guaranteed win always taken, never counted as a sample
    s0 = explore.samples()
    p, e = softmax_pick([(float('inf'), C), (0.0, A)], 5.0, rng, SCALE)
    assert p == C and e == 0 and explore.samples() == s0
    # draw pair never sampled, even if it scores best
    for _ in range(500):
        p, _e = softmax_pick([(0.0, DRAW_PAIR), (-1.0, A), (-2.0, B)], 50.0, rng, SCALE)
        assert p != DRAW_PAIR
    # draw kept only if it is the only option
    p, _e = softmax_pick([(0.0, DRAW_PAIR)], 1.0, rng, SCALE)
    assert p == DRAW_PAIR
    # T = 0 is greedy and draws no sample
    s0 = explore.samples()
    p, e = softmax_pick(ranked(0.0, -0.1), 0.0, rng, SCALE)
    assert p == A and e == 0 and explore.samples() == s0
    print('  [ok] win taken, draw never sampled, T=0 greedy')


def test_anneal():
    assert temperature_for_iter(1, 0.5, 10) == 0.5
    assert abs(temperature_for_iter(6, 0.5, 10) - 0.25) < 1e-12
    assert temperature_for_iter(11, 0.5, 10) == 0.0
    assert temperature_for_iter(30, 0.5, 10) == 0.0
    assert temperature_for_iter(30, 0.5, 0) == 0.5
    print('  [ok] linear anneal to greedy')


def test_gate_tripwire():
    """panel_game must raise if anything draws an exploration sample during
    a gate game. Replace arena's game loop with one that samples once."""
    import arena
    import panel_gate
    real_play, real_agent = arena._play, arena._agent

    def sneaky_play(seed, w, b, stats=None):
        softmax_pick(ranked(0.0, -0.1), 1.0, random.Random(0), SCALE)
        return 'white', 3

    def honest_play(seed, w, b, stats=None):
        return 'white', 3
    arena._agent = lambda *a, **k: object()
    try:
        arena._play = honest_play
        r = panel_gate.panel_game(('x', 'k', 'm', 'y', 1, True, False))
        assert r['margin'] == 3
        arena._play = sneaky_play
        try:
            panel_gate.panel_game(('x', 'k', 'm', 'y', 1, True, False))
            raise SystemExit('FAIL: gate game drew an exploration sample unnoticed')
        except AssertionError as e:
            assert 'GATING' in str(e)
    finally:
        arena._play, arena._agent = real_play, real_agent
    print('  [ok] gate game asserts greedy (tripwire fires on a sample, silent otherwise)')


if __name__ == '__main__':
    print('Running explore tests...')
    test_distribution()
    test_rules()
    test_anneal()
    test_gate_tripwire()
    print('All explore tests passed.')
