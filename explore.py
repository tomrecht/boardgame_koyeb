"""explore.py -- softmax OPENING exploration for self-play generation.

Replaces epsilon-greedy (uniform over ALL legal pairs, at every stage), which
degraded play at eps 0.20 and oscillated at 0.15 (ARCHIVE.md, "Exploration
retry1"). Here, only for the first `turns` turns of EACH side and only in
GENERATION, the move pair is SAMPLED from a softmax over the agent's own 2-ply
scores instead of taken by argmax:

    p(pair) ~ exp(margin(pair) / T),   margin = raw value * 12 (game points)

so a pair the net rates 1 point worse than the best is chosen e^(-1/T) as often.
T is in MARGIN POINTS, which makes it interpretable and independent of
SCORE_SCALE. T anneals to 0 (= greedy) over iterations in the run script.
Near-equal pairs are explored a lot, clearly worse ones almost never -- the
opposite of uniform epsilon, which spent most of its samples on moves the net
correctly rated terrible.

Rules (same as the old epsilon code):
  * a guaranteed win (score +inf) is always taken;
  * the draw call is never sampled (its value is exact; random game endings
    teach nothing) -- it is still taken if it is the only option;
  * sampling uses a DEDICATED RNG derived from the game seed, so the dice
    stream (the global `random`) is identical with exploration on or off.

EVALUATION AND GATING MUST STAY GREEDY. Every call that samples bumps the
process-global SAMPLES counter; the eval paths (game_worker.worker_eval,
panel_gate.panel_game) snapshot it around each game and assert it did not move.
"""
import math

SAMPLES = 0          # process-global count of softmax samples drawn (any outcome)
DRAW_PAIR = ((1, 1, 1), (0, 0, 0))


def samples():
    return SAMPLES


def temperature_for_iter(it, t0, anneal_iters):
    """Linear anneal: T = t0 at iteration 1, reaching 0 (greedy) at
    iteration anneal_iters + 1. anneal_iters <= 0 means constant t0."""
    if t0 <= 0:
        return 0.0
    if anneal_iters <= 0:
        return float(t0)
    frac = max(0.0, 1.0 - (it - 1) / float(anneal_iters))
    return float(t0) * frac


def softmax_pick(ranked, T, rng, score_scale, num_pieces=12):
    """Sample a pair from `ranked` = [(score, pair), ...] as returned by
    GNNAgent.select_move_pair(return_scores=True) (sorted best first, scores in
    raw*score_scale units). Returns (pair, gap): `gap` is how many margin
    points WORSE than the best the net rates the sampled pair (0.0 for a pair
    tied with the best -- in the opening most ties are transpositions, i.e. the
    same end-of-turn position, so a tie is not exploration)."""
    global SAMPLES
    if not ranked:
        return ((0, 0, 0), (0, 0, 0)), 0.0
    if ranked[0][0] == float('inf'):
        return ranked[0][1], 0.0
    cands = [(s, p) for s, p in ranked if p != DRAW_PAIR]
    if not cands:
        return ranked[0][1], 0.0
    if T <= 0:
        return ranked[0][1], 0.0
    if len(cands) == 1:
        return cands[0][1], 0.0
    SAMPLES += 1
    m = [s / score_scale * num_pieces for s, _ in cands]
    top = max(m)
    w = [math.exp((x - top) / T) for x in m]
    tot = sum(w)
    r = rng.random() * tot
    acc = 0.0
    idx = len(cands) - 1
    for i, wi in enumerate(w):
        acc += wi
        if r < acc:
            idx = i
            break
    return cands[idx][1], max(0.0, top - m[idx])
