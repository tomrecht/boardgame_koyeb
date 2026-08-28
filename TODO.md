# Durable TODO

Parked, deliberately-not-now items. (Active work lives in CLAUDE.md / OVERNIGHT_NOTES.md.)

## UNREPRODUCED: score row invisible during a match on Safari (2026-08-16)

Owner played a match on Safari and could not see the score row; minutes later it
was present on all browsers and **neither of us could reproduce it**. Recorded
only so a recurrence does not start from zero.

Ruled out by direct probe (`scoreText.visible`, text, position, and whether it
is inside `camera.worldView`): desktop, phone landscape, phone portrait, on the
welcome screen and in a running game, inside a 6-game match, and before/during/
after the tutorial — **present, visible and in view in every one**. `_tutHudVisible`
is the only thing that hides it, every tutorial exit routes through `_tutEnd`
which restores it before restarting the scene, and `matchScoreLine()` has no
branch that returns an empty string.

If it happens again, capture before touching anything: `s = gameInstance.scene
.getScene('MainGameScene')`, then `s.scoreText.visible`, `s.scoreText.text`,
`s.scoreText.x/y`, `s.cameras.main.worldView`, and `window._tutorialActive`.
That distinguishes hidden / empty / off-camera / never-built, which need
different fixes.

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


## Rendering: the static board is redrawn every frame — DONE (2026-08-16)

**Fixed: the board is baked into a RenderTexture.** 15,310 -> 2,584 draw
commands a frame on desktop (83%), 0 of 70 tiles drawing at rest. See CLAUDE.md
("THE BOARD IS NOW BAKED INTO A RENDERTEXTURE") for the design, the 2x
supersample decision, the texture budget and the pixel proof.

**Measured in real browsers (owner, 2026-08-16): Safari 17.0 -> 24.8 fps (+46%),
Firefox 42.3 -> 58.5 (+38%), Chrome 60 -> 60 (already at vsync).** Failure-mode
checks pass on all three. `?fpstest=1` runs that A/B on screen, for phones.

**CLOSED: owner confirms the Safari sluggishness is gone.** That is the test that
matters — the original report was a felt symptom, not a number, so a felt
recovery is what settles it.

Not chased, deliberately: the OFF baselines never reproduced the 7.8 / 24.0 fps
that motivated the work (the same OFF state measured 17.0 / 60), so that
condition stays unidentified. With the symptom resolved there is nothing to
diagnose — reopen only if it recurs. Likewise the ~2,300 commands of racks, dice
and HUD still drawn every frame: static, bakeable the same way, and NOT worth
doing on current evidence.

The original write-up follows, for the numbers it records.

Owner reported Safari feeling sluggish; measured in-page:

    browser          renderer        actualFps
    Safari (macOS)   WebGL (type 2)      7.8
    Chrome           WebGL (type 2)     24.0

So it is NOT a Canvas2D fallback, and it is NOT the old leak: sampled over 37 AI
moves in a driven game, objects 210 -> 211, draw commands 15427 -> 15362, fps
25.7 -> 26.5 -- flat on all three. The `Tile.highlight` command-list leak (55 ->
4 fps over 60 turns) is genuinely fixed.

**The steady state is the problem: ~15,300 draw commands across 108 Graphics
objects EVERY FRAME**, for a board that does not change between moves. Chrome
absorbs it at ~25 fps, Safari does not.

**Fix direction** (this is what was built): stop re-tessellating a static board.
Bake the tiles into a RenderTexture once and redraw only the tiles that actually
change (highlight, theme switch, relayout).

Watch-outs already known from this codebase: `Tile` caches its points, hit area
and goal-number text behind `_points`/`_built`, and `drawTile` is called on every
colour change; the themes are switched LIVE (`applyThemeLive` recolours tiles in
place), so a baked texture has to be invalidated on a theme change as well as on
a relayout/rotation. All three held up — the theme switch is the one that needed
an explicit re-bake hook.

Note the earlier "~44 fps at turn 150" figure in CLAUDE.md was after the leak fix
but is not reproduced now; ~25 fps in Chrome is the current steady state.

`?maxmp=N` caps the phone render buffer for A/B'ing fill rate, but it is
phone-only (`_sizeCanvasToScreen` returns early on desktop, which uses Scale.FIT
at 1800x1200), so it is irrelevant to the desktop case.


## Planned experiment: how far can the prefilter be cut? (2026-08-15)

Motivation: in-browser move latency is median 1.25s / p90 2.97s at 4x CPU
throttle (a mid-range phone). Candidates scored per move is 40 median / 145 max,
and 40 is exactly `prefilter_top_k`. ~~The encoder's per-candidate BFS is the
likely hot spot rather than the forward pass~~ — **measured 2026-08-15 and it is
NOT: the forward pass is 88% of a move, the encoder 2%** (see
`latency_breakdown.html` and PORTING.md §7). Cutting candidates does still cut
the forward pass, since it scores fewer of them, but the cheap-BFS reasoning
that motivated this was wrong. The open question is still what it costs in
strength.

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


## Stage 1 could cull a save-ENABLING first move — FIXED (2026-08-15)

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

**MEASURED: real, but it never changed a move.** Against true ground truth
(`BASELINE=40,0`, i.e. no first-move prefilter), BEFORE the fix:

    positions        save pairs kept   lost all   moves differ   saves lost
    60 walked        55%               6 of 11    13 of 60       0
    40 endgame       77%               0 of 31    0 of 40        0 (of 30)
    200 endgame      77%               0 of 164   3 of 200       0 (of 160)

The served F=12 really did discard save pairs -- over half of them in walked
positions, and in 6 of 11 it discarded every one, so the agent frequently never
saw a save pair at all. But it never changed the MOVE: in the endgames, where
saves are dense and the baseline banks a piece in 163 of 166 positions, F=12
played the identical pair every time. The culled pairs were ones the net would
not have chosen. 0 of 160 is not "never", though -- by the rule of three the 95%
upper bound is ~1.9% of endgame save opportunities.

**APPLIED ANYWAY (owner's call, 2026-08-15): "never cull a save pair" should
hold by construction, not by luck.** Stage 1 already APPLIES each first move in
order to score it, so while it is applied it now asks whether a save has become
available and exempts the move if so -- no simulation, no second-move
generation:

    prev_stage = board.game_stages[player]
    board.game_stages[player] = board.get_game_stage(player)
    enables = any(board.get_saving_die(p) for p in board.pieces if p.player == player)
    board.game_stages[player] = prev_stage

The stage is computed fresh and PUT BACK deliberately: `get_saving_die` reads
`game_stages`, and stage 1 applies a move without the `get_valid_moves` call
that refreshes them, so leaving it changed would drift the candidates scored
after it -- the exact bug that once scored identical candidates 20-80 points
apart.

AFTER, same harness:

    positions        save pairs kept   lost all    moves differ   s/move med
    200 endgame      77% -> 100%       0           3 -> 0         0.123 -> 0.115
    60 walked        55% -> 100%       6 of 11 -> 0  13 (unchanged) 0.257 -> 0.245

Every save pair now survives, in both samples. No slowdown: the exemption fires
where saves are dense, and stage 1 was already applying every first move
regardless. The 13 walked positions that still play a different pair under F=12
are the ones that never had anything to do with saves -- which is the point of
the "what this does NOT establish" note below.

**Both twins, one commit.** `agent_gnn.py` and `agent.js` are proven identical
at 110/110 and that equivalence is an asset, so the change was mirrored and
committed together, and the fixtures REGENERATED first (they encode Python's
choices -- testing new JS against an old fixture compares the change with
itself). `agent_test.js` re-passed 50/50 on candidate sets and chosen pairs.

Note this is served-only in effect: `first_move_prefilter`'s library default is
0, so training, self-play and the arena never ran stage 1 at all and are
unchanged. It is NOT a fix for the pass-over-save rough edge, which is not
observed in the current champion; `debug_pass_over_save` distinguishes "the save
was scored and rejected" (value error) from "no save pair reached the net"
(candidate drop), and this only ever addressed the second.

**Read the denominators, not the zeros.** This took three attempts to measure,
each defeated by an empty denominator: asking whether a save was legal as a
first MOVE qualified 1 position in 60; the next cut had 5; the walked run had 30
save-carrying positions but the baseline PLAYED a save in none of them, so "save
lost" could not fire. Only hand-built endgames put a real denominator (30) under
the question. A "0" from this harness means nothing until the denominator beside
it is checked.

**What this still does NOT establish.** The endgames are synthetic. And the
walked positions play a different move under F=12 for reasons that have nothing
to do with saves; whether THOSE differences cost strength is unmeasured -- that
is what `match_topk.py match` would answer, and it never has been for F itself
(the original F=12 validation compared win rates, not pair coverage). The fix
above closes the save-coverage gap only; it does not make served F=12 equal to
the F=0 configuration training and the arena use.

## PARKED DESIGN IDEA: randomize the goal numbering at start (owner, 2026-08-27)

Shuffle which NUMBER sits at which rim position, so a player cannot memorise the
map and has to re-derive it each game.

Worth recording alongside it, from the endgame maths done the same day: the
useful structure is a property of the GEOMETRY, not of the labels. Three pairs of
goal positions are always exactly 4 apart (currently 6&1, 5&3, 4&2), and the
tiles one ring inside a goal always reach three goals with a single die. Shuffle
the numbering and those facts survive unchanged -- only which NUMBERS they apply
to moves. So the variant would reward understanding the structure over
memorising one arrangement, which is a point in its favour.

Two consequences to check before building it:
  * the trained model is fed per-piece goal DISTANCES and a `save_number/6` tile
    feature, so the information it needs is present and it might generalise --
    but it has only ever seen one arrangement, so expect it to be weaker until
    retrained across arrangements (which would also be a free data-augmentation
    scheme, in the same family as the D3 symmetry work);
  * the validated D3 automorphism is tied to the current numbering, so the
    symmetry-augmentation code would need generalising too.
Only ~120 of the 6! = 720 numberings are distinct up to the board's own symmetry.
