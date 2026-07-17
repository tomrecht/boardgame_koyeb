"""
td_returns.py — TD(lambda) target computation for the board-game GNN value head.

The model's value head outputs a scalar in (-1, 1) (tanh) from the perspective
of the player to move, where raw_output * NUM_PIECES = expected final margin in
points for that player. TD(lambda) replaces the Monte-Carlo target (the final
game outcome stamped on every position) with the lambda-return: a
geometrically-weighted blend of n-step bootstrapped estimates that lean on a
frozen target model's own predictions at later states.

This module is pure logic + a single batched forward pass through a frozen
target model. The trickiest part — keeping signs consistent across the
alternating-player trajectory — is isolated and unit-tested (run this file as a
script: `python td_returns.py`).

Sign convention
---------------
Every record carries `final_score` = the game's margin FROM THAT RECORD'S OWN
player's perspective (positive = that player won by that many points, negative
= lost, 0 = draw). This is identical in magnitude across all records of a game,
flipped only by player. The model output `model(batch)` is likewise in the
mover's frame.

To chain values across turns (which alternate white/black) we convert
everything into WHITE'S frame, run the standard backward lambda-return
recursion, then convert each per-record target back into that record's mover
frame so it matches what the head predicts.

A game's recorded states are s_0 .. s_{n-1} (one per turn, ordered by
move_index). There is no terminal record; the terminal value is recovered from
`final_score` (z). The bootstrap chain in white's frame is therefore

    [ V(s_0), V(s_1), ..., V(s_{n-1}), z_white ]

with reward 0 on every transition (the only signal is the terminal margin).
"""

from collections import defaultdict

NUM_PIECES = 12  # must match network.NUM_PIECES / the value-head margin scale


# ---------------------------------------------------------------------------
# Sign helpers (unit-tested below)
# ---------------------------------------------------------------------------

def to_white_frame(value_mover, player):
    """Convert a mover-frame value to white's frame."""
    return value_mover if player == 'white' else -value_mover


def from_white_frame(value_white, player):
    """Convert a white-frame value to the given player's (mover) frame."""
    return value_white if player == 'white' else -value_white


def terminal_z_white(final_score, player):
    """White-frame terminal value in [-1, 1] from a record's final_score
    (which is in that record's own player's frame)."""
    z_player = final_score / NUM_PIECES
    z_white = to_white_frame(z_player, player)
    # clamp to the tanh-reachable range so the regression target is attainable
    return max(-1.0, min(1.0, z_white))


# ---------------------------------------------------------------------------
# Trajectory grouping
# ---------------------------------------------------------------------------

def group_trajectories(records, min_len=2, verbose=True):
    """Group flat records into ordered per-game trajectories.

    Returns a list of (game_id, [records sorted by move_index], z_white).
    - Sorts by move_index (does NOT assume gap-free indices).
    - Drops games with < min_len records (can't form a TD transition); these
      are returned separately so the caller can fall back to MC if desired.
    - Derives z_white from the first record and asserts the rest agree, which
      also catches corrupt/mixed games.

    Returns (trajectories, dropped) where `dropped` is a list of the short
    games' record-lists.
    """
    by_game = defaultdict(list)
    for r in records:
        by_game[r['game_id']].append(r)

    trajectories = []
    dropped = []
    bad = 0
    for gid, recs in by_game.items():
        recs = sorted(recs, key=lambda r: r['move_index'])
        if len(recs) < min_len:
            dropped.append(recs)
            continue

        # derive the white-frame terminal value and verify consistency
        z_white = terminal_z_white(recs[0]['final_score'], recs[0]['player'])
        ok = True
        for r in recs:
            zw = terminal_z_white(r['final_score'], r['player'])
            if abs(zw - z_white) > 1e-6:
                ok = False
                break
        if not ok:
            bad += 1
            continue

        trajectories.append((gid, recs, z_white))

    if verbose:
        n_drop = sum(len(d) for d in dropped)
        print(f"  group_trajectories: {len(trajectories)} games, "
              f"{len(dropped)} short games dropped ({n_drop} positions), "
              f"{bad} inconsistent games dropped")
    return trajectories, dropped


# ---------------------------------------------------------------------------
# lambda-return recursion (white frame)
# ---------------------------------------------------------------------------

def lambda_returns_white_frame(values_white, z_white, lam, gamma):
    """Backward lambda-return recursion over one trajectory, in white's frame.

    values_white : [V(s_0), ..., V(s_{n-1})]  (frozen target model predictions)
    z_white      : terminal value (after s_{n-1})
    Reward is 0 on every transition; only the terminal carries signal.

    The (offline / forward-view) lambda-return obeys the recursion
        G_t = r_{t+1} + gamma * [ (1 - lam) * V(s_{t+1}) + lam * G_{t+1} ]
    with G at the terminal equal to z_white, and V(s_n) := z_white.

    Returns G[0..n-1] in white's frame.
    """
    n = len(values_white)
    G = [0.0] * n
    # bootstrap value of the state AFTER s_t; for the last real state it's z
    next_G = z_white          # G_{t+1}, starts as the terminal return
    next_V = z_white          # V(s_{t+1}), starts as the terminal value
    for t in range(n - 1, -1, -1):
        G[t] = gamma * ((1.0 - lam) * next_V + lam * next_G)  # r=0
        next_G = G[t]
        next_V = values_white[t]
    return G


# ---------------------------------------------------------------------------
# Full target computation (one batched forward pass per game)
# ---------------------------------------------------------------------------

def compute_td_targets_for_game(recs, z_white, target_model, encoder, board,
                                lam, gamma):
    """Compute per-record TD(lambda) targets (in each record's mover frame) for
    one game trajectory.

    Uses ONE batched forward pass through the frozen `target_model`. All tensor
    work happens on whatever device `target_model` / the encoder are on; we read
    scalars back to Python floats for the (cheap) recursion, then hand back a
    plain list of float targets aligned to `recs`.

    Returns (targets, ok). On any encoding/inference failure returns
    (mc_targets, False) so the caller can fall back to MC labels for that game.
    """
    import torch
    from network import collate_batch

    # MC fallback targets (mover frame) in case anything goes wrong
    mc_targets = [max(-1.0, min(1.0, r['final_score'] / NUM_PIECES)) for r in recs]

    try:
        encoded = []
        for r in recs:
            board.update_state(r['raw_state'])
            encoded.append(encoder.encode(board, r['player']))
        batch = collate_batch(encoded)
        with torch.no_grad():
            preds = target_model(batch)            # [n], mover frame, on DEVICE
        preds = preds.detach().cpu().tolist()
        if not isinstance(preds, list):            # n == 1 safety
            preds = [preds]
    except Exception as e:
        print(f"    TD target encode/infer failed for game "
              f"{recs[0].get('game_id')}: {e}")
        return mc_targets, False

    # mover frame -> white frame
    values_white = [to_white_frame(v, r['player']) for v, r in zip(preds, recs)]
    G_white = lambda_returns_white_frame(values_white, z_white, lam, gamma)
    # white frame -> each record's mover frame, clamp to tanh range
    targets = [max(-1.0, min(1.0, from_white_frame(g, r['player'])))
               for g, r in zip(G_white, recs)]
    return targets, True


def compute_td_targets(records, target_model, encoder, board, lam, gamma,
                       include_short_as_mc=True, verbose=True):
    """Compute TD(lambda) targets for a flat list of records.

    Returns a NEW list of records (shallow copies) each with an added
    'td_target' field, ready to feed to the trainer. Games too short for a TD
    transition fall back to their MC label (if include_short_as_mc).
    """
    trajectories, dropped = group_trajectories(records, verbose=verbose)

    out = []
    n_fallback = 0
    for gid, recs, z_white in trajectories:
        targets, ok = compute_td_targets_for_game(
            recs, z_white, target_model, encoder, board, lam, gamma)
        if not ok:
            n_fallback += 1
        for r, t in zip(recs, targets):
            r2 = dict(r)
            r2['td_target'] = t
            out.append(r2)

    if include_short_as_mc:
        for recs in dropped:
            for r in recs:
                r2 = dict(r)
                r2['td_target'] = max(-1.0, min(1.0, r['final_score'] / NUM_PIECES))
                out.append(r2)

    if verbose and n_fallback:
        print(f"  compute_td_targets: {n_fallback} games fell back to MC labels")
    return out


# ---------------------------------------------------------------------------
# Unit tests — run `python td_returns.py`
# ---------------------------------------------------------------------------

def _approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


def _test_signs():
    assert to_white_frame(0.3, 'white') == 0.3
    assert to_white_frame(0.3, 'black') == -0.3
    assert from_white_frame(0.3, 'white') == 0.3
    assert from_white_frame(0.3, 'black') == -0.3
    # round trip
    for p in ('white', 'black'):
        assert _approx(from_white_frame(to_white_frame(0.42, p), p), 0.42)
    # terminal z: white lost by 4 -> white-frame -4/12
    assert _approx(terminal_z_white(-4, 'white'), -4 / 12)
    # same game from black's record: black won by 4 -> final_score +4, white-frame still -4/12
    assert _approx(terminal_z_white(4, 'black'), -4 / 12)
    print("  [ok] sign helpers")


def _test_lambda_one_is_mc():
    """lam=1, gamma=1 must reproduce the Monte-Carlo target (terminal z) at
    every state, regardless of the bootstrap values."""
    vals = [0.1, -0.5, 0.9, 0.0]
    z = 0.6
    G = lambda_returns_white_frame(vals, z, lam=1.0, gamma=1.0)
    for g in G:
        assert _approx(g, z), f"lam=1 should give MC target {z}, got {g}"
    print("  [ok] lam=1 reproduces MC target")


def _test_lambda_zero_is_one_step():
    """lam=0, gamma=1: G_t = V(s_{t+1}); last state bootstraps from z."""
    vals = [0.1, -0.5, 0.9]
    z = 0.6
    G = lambda_returns_white_frame(vals, z, lam=0.0, gamma=1.0)
    assert _approx(G[2], z)            # last -> terminal
    assert _approx(G[1], vals[2])      # -> V(s_2)
    assert _approx(G[0], vals[1])      # -> V(s_1)
    print("  [ok] lam=0 reproduces one-step bootstrap")


def _test_lambda_intermediate():
    """Hand-computed check for lam=0.5, gamma=1 on a 2-state trajectory."""
    vals = [0.2, -0.4]
    z = 1.0
    lam = 0.5
    # t=1 (last): G1 = (1-lam)*z + lam*z = z = 1.0
    # t=0: next_V = V(s_1) = -0.4, next_G = G1 = 1.0
    #      G0 = (1-0.5)*(-0.4) + 0.5*(1.0) = -0.2 + 0.5 = 0.3
    G = lambda_returns_white_frame(vals, z, lam=lam, gamma=1.0)
    assert _approx(G[1], 1.0), G[1]
    assert _approx(G[0], 0.3), G[0]
    print("  [ok] intermediate lam hand-check")


def _test_gamma():
    """gamma<1 discounts the bootstrap. 1-state trajectory: G0 = gamma*z."""
    G = lambda_returns_white_frame([0.5], z_white=1.0, lam=1.0, gamma=0.9)
    assert _approx(G[0], 0.9), G[0]
    print("  [ok] gamma discount")


def _test_grouping():
    recs = [
        {'game_id': 'g1', 'move_index': 2, 'player': 'white', 'final_score': -4},
        {'game_id': 'g1', 'move_index': 0, 'player': 'white', 'final_score': -4},
        {'game_id': 'g1', 'move_index': 1, 'player': 'black', 'final_score': 4},
        {'game_id': 'g2', 'move_index': 0, 'player': 'white', 'final_score': 0},  # short
    ]
    trajs, dropped = group_trajectories(recs, verbose=False)
    assert len(trajs) == 1 and trajs[0][0] == 'g1'
    gid, sorted_recs, z = trajs[0]
    assert [r['move_index'] for r in sorted_recs] == [0, 1, 2]   # sorted
    assert _approx(z, -4 / 12)                                   # consistent z
    assert len(dropped) == 1 and dropped[0][0]['game_id'] == 'g2'
    print("  [ok] trajectory grouping/sorting/consistency")


if __name__ == '__main__':
    print("Running td_returns unit tests...")
    _test_signs()
    _test_lambda_one_is_mc()
    _test_lambda_zero_is_one_step()
    _test_lambda_intermediate()
    _test_gamma()
    _test_grouping()
    print("All td_returns tests passed.")
