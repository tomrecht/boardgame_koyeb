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

Watch-out: the arena finding applies here too -- promotion-gate-style single
matchups mislead, and the top four champions were statistically inseparable at
9-16 games per matchup. Budget the sample before starting, not after.
