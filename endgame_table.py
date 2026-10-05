"""Exact expected turns to bank every remaining piece, for one side whose
unsaved pieces ALL stand on goals they can bank from (the endgame proper).

State: which numbered pieces remain (each on its own goal; bitmask over 1-6)
and how many blanks stand on each goal (b1..b6). 64 x 924 = 59,136 states.

A turn: roll two dice (21 distinct rolls), use each die on
  * banking a numbered piece n: die == n (exact match, game.py get_saving_die);
  * banking a blank on goal g: die == g, or die > g when g is the side's
    highest occupied goal (the endgame higher-die rule);
  * hopping a blank between goals: route distance 4 with one die, or 4 / 7 / 11
    with both dice on the one blank (via a field tile);
  * or not at all.
in either order, minimising expected turns. The last-piece rule (a lone numbered
piece becomes a blank at the start of its owner's turn) is applied to every
state before it moves. Simplification: pieces never step off onto the field; the
lone-piece DP measured that option at ~0.04 turns at most (goals 4-6).

Hops can cycle, so values are found by value iteration, one piece-count level
at a time (banking only lowers the count, so lower levels are final first).

    python3 endgame_table.py            -> endgame_table.json, + checks
"""
import itertools, json, sys, time

ROLLS = [(a, b, (1 if a == b else 2) / 36.0) for a in range(1, 7) for b in range(a, 7)]
# goal-to-goal shortest routes (measured on the board graph, see CLAUDE.md:
# paired 4 apart -- 6&1, 5&3, 4&2 -- and 7 / 11 / 14 otherwise)
GOAL_DIST = {
    1: {2: 11, 3: 11, 4: 7, 5: 14, 6: 4}, 2: {1: 11, 3: 11, 4: 4, 5: 7, 6: 14},
    3: {1: 11, 2: 11, 4: 14, 5: 4, 6: 7}, 4: {1: 7, 2: 4, 3: 14, 5: 11, 6: 11},
    5: {1: 14, 2: 7, 3: 4, 4: 11, 6: 11}, 6: {1: 4, 2: 14, 3: 7, 4: 11, 5: 11},
}


def canon(s):
    """Last-piece rule: a lone numbered piece becomes a blank on its goal."""
    mask, b = s
    if sum(b) == 0 and mask and mask & (mask - 1) == 0:
        n = mask.bit_length()
        nb = list(b); nb[n - 1] = 1
        return (0, tuple(nb))
    return s


def count(s):
    return bin(s[0]).count('1') + sum(s[1])


def highest(s):
    mask, b = s
    for g in range(6, 0, -1):
        if (mask >> (g - 1)) & 1 or b[g - 1]:
            return g
    return 0


def one_die(s, d):
    """States reachable by using a single die of d (not counting 'unused')."""
    mask, b = s
    out = set()
    if (mask >> (d - 1)) & 1:                       # bank numbered d
        out.add((mask & ~(1 << (d - 1)), b))
    h = highest(s)
    for g in range(1, 7):
        if not b[g - 1]:
            continue
        if d == g or (d > g and g == h):            # bank a blank from g
            nb = list(b); nb[g - 1] -= 1
            out.add((mask, tuple(nb)))
        for g2, dist in GOAL_DIST[g].items():       # hop a blank g -> g2
            if dist == d:
                nb = list(b); nb[g - 1] -= 1; nb[g2 - 1] += 1
                out.add((mask, tuple(nb)))
    return out


def sum_hops(s, total):
    mask, b = s
    out = set()
    for g in range(1, 7):
        if b[g - 1]:
            for g2, dist in GOAL_DIST[g].items():
                if dist == total:
                    nb = list(b); nb[g - 1] -= 1; nb[g2 - 1] += 1
                    out.add((mask, tuple(nb)))
    return out


def all_states(max_blanks=6):
    for mask in range(64):
        for k in range(max_blanks + 1):
            for combo in itertools.combinations_with_replacement(range(6), k):
                b = [0] * 6
                for g in combo:
                    b[g] += 1
                yield (mask, tuple(b))


def build(max_blanks=6, tol=1e-10):
    states = sorted({canon(s) for s in all_states(max_blanks)}, key=count)
    one = {}

    def od(s, d):
        k = (s, d)
        if k not in one:
            one[k] = one_die(s, d)
        return one[k]

    # successor sets per (state, roll), computed once
    succ = {}
    for s in states:
        for i, (a, b, w) in enumerate(ROLLS):
            res = {s}
            for x, y in ((a, b), (b, a)):
                for s1 in od(s, x) | {s}:
                    res.add(s1)
                    res |= od(s1, y)
            res |= sum_hops(s, a + b)
            succ[(s, i)] = [canon(t) for t in res]
    V = {}
    by_count = {}
    for s in states:
        by_count.setdefault(count(s), []).append(s)
    for c in sorted(by_count):
        level = by_count[c]
        if c == 0:
            for s in level:
                V[s] = 0.0
            continue
        for s in level:
            V[s] = 10.0 * c                             # pessimistic start
        for it in range(500):
            delta = 0.0
            for s in level:
                # E[turns] = 1 + sum_roll w * min over successors. Staying put
                # (all successors in this level) is resolved by iteration.
                v = 1.0 + sum(w * min(V[t] for t in succ[(s, i)])
                              for i, (a, b, w) in enumerate(ROLLS))
                delta = max(delta, abs(v - V[s]))
                V[s] = v
            if delta < tol:
                break
        print(f'  level {c}: {len(level)} states, {it + 1} iterations', flush=True)
    return V


def key(s):
    mask, b = s
    return f'{mask}:' + ''.join(map(str, b))


def main():
    t0 = time.time()
    V = build()
    print(f'{len(V)} states in {time.time() - t0:.0f}s')
    # checks against the lone-piece DP (CLAUDE.md domain facts)
    lone = {1: 1.000, 2: 1.029, 3: 1.125, 4: 1.303, 5: 1.462, 6: 1.644}
    for g, want in lone.items():
        b = [0] * 6; b[g - 1] = 1
        print(f'  lone blank on goal {g}: {V[(0, tuple(b))]:.3f}  (lone-piece DP {want:.3f})')
    for n in (2, 5):
        b = [0] * 6; b[0] = 1
        print(f'  numbered {n} + blank on goal 1: {V[canon((1 << (n - 1), tuple(b)))]:.3f}')
    json.dump({key(s): round(v, 5) for s, v in V.items()}, open('endgame_table.json', 'w'))
    print('saved endgame_table.json')


if __name__ == '__main__':
    main()
