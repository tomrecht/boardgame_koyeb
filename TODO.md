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

**Written, not yet run: `match_topk.py`** (2026-08-15). Two modes, and the
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


## Candidate fix: stage 1 can cull a save-ENABLING first move (2026-08-15)

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

**Do not ship it unmeasured.** It changes which candidates the net sees, i.e.
it changes play. Validate the same way F=12 itself was validated -- paired games
via match_topk.py match -- and re-run the probe with BASELINE=40,0 first to size
how often the case actually arises.
