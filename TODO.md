# Durable TODO

Parked, deliberately-not-now items. (Active work lives in CLAUDE.md / OVERNIGHT_NOTES.md.)

## Deployment / hosting

- [x] **ONNX export for cheap hosting.** DONE 2026-07-25. `encoder.py` (numpy,
  torch-free) + `gnn_backend.py` (TorchBackend/OnnxBackend) + `onnx_export.py`.
  Deployed image is Flask + numpy + onnxruntime, no torch, ~80-100 MB resident
  and ~0.1 s/move. Re-export with `python onnx_export.py <ckpt> model.onnx`; it
  self-verifies and app.py serves `model.onnx` by default. Currently serving
  `symaug_champ_July27_iter6.pt` (the promoted champion, not `almostchamp` iter11).
- [x] **Tutorial step-runner** — DONE 2026-07-25/26. Ten steps walking one
  continuous game to a win, hard-blocked to the scripted moves, scripted Black
  replies, positions verified against `game.py`. See CLAUDE.md.


## Planned experiment: how far can the prefilter be cut? (2026-08-15)

Motivation: in-browser move latency is median 1.25s / p90 2.97s at 4x CPU
throttle (a mid-range phone). Candidates scored per move is 40 median / 145 max,
and 40 is exactly `prefilter_top_k`. The encoder's per-candidate BFS is the
likely hot spot rather than the forward pass, so cutting candidates should cut
nearly all of the time. The open question is what it costs in strength.

NOT a blind tuning knob: these settings change which moves the value net ever
sees. F=12 was adopted only after 120 paired games showed no measurable loss.

**Method** (same shape as `match_prefilter.py`, which already exists for this):
- Paired/CRN games, same model both sides, ONLY the prefilter settings differing.
- Sweep `prefilter_top_k` 40 (current) / 24 / 16 / 10, and `first_move_prefilter`
  12 / 8 / 6 -- one at a time first, then the promising corner.
- Primary metric: paired margin with a CI, not bare win rate. Measured paired
  margin SD is ~1.74/pair, so ~100 paired games resolves ~0.5 and ~290 resolves
  ~0.3; the existing 120-game run gave +/-0.27.
- Record per-setting: median/p90 candidates scored AND in-browser latency, so
  the output is a strength-vs-latency curve rather than a single verdict.
- Secondary check: how often a save-containing pair survives the cull at each k.
  Saves are exempt from the top-K cull for a reason (the heuristic undervalues
  them), and a smaller k makes that exemption load-bearing -- worth confirming
  it still fires rather than assuming.
- Run locally on the Mac (BOARDGAME_DEVICE=cpu), not on the server.

**RUN, and the answer is: leave top_k alone (2026-08-15).** 200 positions,
F pinned at the served 12:

    top_k  cand med  cand max  s/move med  s/move p90  lost all  saves kept
       40        40       115       0.253       0.354         0        100%
       24        24        99       0.206       0.304         0        100%
       16        16        91       0.194       0.296         0        100%
       10        10        85       0.164       0.247         0        100%

- **Safe, as designed.** 31 of the 200 positions carry save pairs and every one
  survives at every top_k, with zero "lost all". Save PAIRS are exempt from the
  top-K cull, so this holds by construction rather than by luck.
- **But not worth taking on its own.** Median 0.253 -> 0.164 s/move is ~35%, and
  the number that matters for how a phone FEELS is the tail: cand max only falls
  115 -> 85 (26%), because `prefilter_min_k`, the save exemption and the pass
  pair all survive the cull whatever top_k is. Scaled to the browser (~4x) that
  is p90 ~1.4s -> ~1.0s. Owner's call: not touching it for now.
- **The floor is stage 1**, which scores ~150 first moves before any pair is
  expanded; top_k cannot reach that. It is also the stage where the
  save-enabling-first-move gap lives (see the next section), so that is where
  both the time and the correctness questions actually are.
- Strength was NOT measured -- the paired match was not run, since there is no
  point measuring the cost of a change we are not making.

Earlier n=60 runs reported ~2.3x speedups and non-monotonic orderings (top_k=10
slower than top_k=16); that was machine load, not signal. Timings need n in the
hundreds and a quiet machine.

**The harness: `match_topk.py`** (2026-08-15). Two modes, and the
order matters:

    python match_topk.py probe 60                  # minutes, no games
    N_WORKERS=8 python match_topk.py match 120 24,16,10 12

`probe` walks recorded positions and reports, per setting, the candidate count,
seconds per move, and how often a save was legal but NO save pair survived the
cull. Run it FIRST: saves are exempt from the top-K cull because the heuristic
undervalues them, so at small k that exemption becomes load-bearing, and a
setting that drops saves is disqualified whatever the match data says.
`match` is the paired-seed measurement against the served baseline (top_k=40,
F=12), colours swapped within each pair. Seed base 7_900_000, disjoint from
match_prefilter.py's 7_700_000.

Watch-out: the arena finding applies here too -- promotion-gate-style single
matchups mislead, and the top four champions were statistically inseparable at
9-16 games per matchup. Budget the sample before starting, not after.


## Stage 1 can cull a save-ENABLING first move — MEASURED, PARKED (2026-08-15)

Found while probing top_k. `select_move_pair`'s first-move prefilter exempts a
first move that IS a save:

    keep += [m for _, m in first_scored if m not in kept and m[1] == 'save']

but nothing protects a first move that ENABLES a save as the second half of the
pair -- step a piece onto its goal with one die, save it with the other. If that
first move scores poorly on its own it is culled, and the pair is never
expanded, so the save never reaches the value head at all.

Measured (match_topk.py probe, 60 positions, saves kept vs the F=12 baseline):
100% at every top_k from 40 to 10, but 94% at F=8 and 91% at F=6. So the cull
that loses save pairs is stage ONE, not the top-K cull -- top_k is safe by
construction because save PAIRS are exempt there.

Probably rare at the served F=12 because `goal_bonuses` is large, so a move
landing on a goal tends to score well alone and survive anyway. But that is the
heuristic happening to agree, not a guarantee -- and the exemption exists
precisely because the heuristic's view of saves is not trusted. Relevant to the
known pass-over-save rough edge: `debug_pass_over_save` already distinguishes
"the save was scored and rejected" (value error) from "no save pair reached the
net" (candidate drop), and this is a mechanism for the second.

**The fix is nearly free.** Stage 1 already APPLIES each first move to score it,
so while it is applied, ask whether a save has become available:

    any(board.get_saving_die(p) for p in board.pieces if p.player == player)

and exempt the move from the cull if so. No simulation, no second move
generation.

**MEASURED, and it does not bite. Not shipping it.**

Against true ground truth (`BASELINE=40,0`, i.e. no first-move prefilter):

    positions        save pairs kept   lost all   moves differ   saves lost
    60 walked        55%               6 of 11    13 of 60       0
    40 endgame       77%               0 of 31    0 of 40        0 (of 30)

The served F=12 really does discard save pairs -- over half of them in walked
positions, and in 6 of 11 it discards every one, so the agent frequently never
sees a save pair at all. But it never changes the MOVE: in the endgames, where
saves are dense and the baseline banks a piece in 30 of 40 positions, F=12
plays the identical pair every time. The culled pairs are ones the net would
not have chosen.

So the gap is real in principle and inert in practice, and the fix would change
play for no measured gain. Parked.

**Read the denominators, not the zeros.** This took three attempts to measure,
each defeated by an empty denominator: asking whether a save was legal as a
first MOVE qualified 1 position in 60; the next cut had 5; the walked run had 30
save-carrying positions but the baseline PLAYED a save in none of them, so "save
lost" could not fire. Only hand-built endgames put a real denominator (30) under
the question. A "0" from this harness means nothing until the denominator beside
it is checked.

**What this does NOT establish.** 0 of 30 is not "never" -- by the rule of three
the 95% upper bound is ~10% of endgame save opportunities. The endgames are
synthetic. And 13 of 60 walked positions DO play a different move under F=12;
none lost a save, but whether those differences cost strength is unmeasured --
that is what `match_topk.py match` would answer, and it never has been for F
itself (the original F=12 validation compared win rates, not pair coverage).
