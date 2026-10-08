# Board Game GNN — TD(λ) Training Project

**Companion file: `ARCHIVE.md`** — the settled history this file used to carry
(the whole TD(λ) training programme, deep search, the arena measurements, and
the July–August 2026 frontend and port history). Verbatim, nothing deleted.
Read it for the reasoning behind a settled decision or a measured number; this
file keeps what a fresh session needs to act.

Context for Claude Code picking up this project. Owner (Tom) prefers terse
responses. Tom is the domain expert on the boardgame, you are the ML expert; make suggestions on training accordingly, but you may also suggest potential improvements to game rules and mechanics when relevant.

**KEEP THIS FILE CURRENT, BUT IN PROPORTION (owner, 2026-08-04, amended
2026-09-20).** Update CLAUDE.md as part of the work, not as a favour asked for
afterwards. Anything a fresh session would have to rediscover belongs here: a
rule change, a deployed model, a config the app depends on, a measured number
that settles a question, a bug whose cause was non-obvious, a decision and why.
Land it in the same commit as the change where practical. Prune too: when a note
is superseded, replace it rather than stacking a contradiction on top. Numbers
beat adjectives — record what was measured, over what sample, and what it rules
out.
**But not every tiny change needs an entry** (owner, 2026-09-20): a cosmetic
tweak, a label, a sound does not. Settled history belongs in **`ARCHIVE.md`**,
not here — this file had reached 4,273 lines and the re-reading cost was real.

**VERIFY IN PROPORTION TO RISK (owner, 2026-09-20).** The "measured, not
assumed" standard below earned its place on rules and input bugs; it earns
nothing on a chime.
  * **Drive it in a browser** for anything touching move legality, reachability,
    input handling, turn state, the agent, or layout geometry.
  * **Ship on the code path** — read it, reason it through, commit — for sounds,
    copy, colours, and anything that cannot change what the game accepts as a
    move.
  * Owner's words: *"Not every tiny change needs to be verified and
    documented."* Three sessions were lost to theorising without data, so the
    standard is not gone — it is targeted.

**`testing` IS THE STAGING BRANCH (owner, 2026-09-25).** New features land on
`testing`, which KOYEB builds for owner to play; once he is happy they reach
`main`, which Cloudflare builds as quahuru.com, by a fast-forward merge. The old
cherry-pick-to-both rule is GONE — the branches share one history now. See
"`testing` IS A STAGING BRANCH" below.

**WORKSPACE CONSTRAINT (owner, 2026-07-21): Ownder works on two machines, iMac and MacBook. 
Work only inside this folder (`/Users/tomrecht/game/boardgame_koyeb` on MacBook, `/Users/tom/Game/BoardGame` on iMac).** Reads/writes outside it trigger permission prompts. Do NOT create git worktrees or files in sibling dirs; run new experiments from a branch checked out *in this folder* (or keep files here).
When you must inspect another branch's file, use `git show <branch>:<path>`;
when you must read an out-of-folder artifact (e.g. a running worktree's log),
`cp` it in first. (Some existing runs — e.g. the symmetry-aug worktree — predate
this rule; monitor them by copying their logs in.) The symmetry-aug work is now
on origin as `symmetry-aug` / `symmetry-aug-main`, so it no longer lives only on
the MacBook; its checkpoints `symaug_champ_July27_iter6.pt` and
`symaug_almostchamp_July27_iter11.pt` are in this folder (iter6, the promoted
champion, is the one deployed).

**OVERNIGHT / AUTONOMOUS WORK RULES (owner, 2026-07-23, after a failure).**
The agent only acts *while a turn is running* (issuing tool calls). The moment
it ends a turn with a text message, it goes IDLE until the next user message or
a scheduled/background event — there is NO background execution otherwise.
A "continuing meanwhile…" sign-off followed by ending the turn = doing nothing.
So, for any "work through the night / while I sleep" task:
  1. NEVER promise to "keep going" and then end the turn. If work remains, keep
     issuing tool calls until it is actually done or at a genuine checkpoint.
  2. To truly run unattended, set up REAL continuation: launch long jobs as
     tracked background processes (`run_in_background`) AND schedule a self-
     wakeup (ScheduleWakeup / CronCreate) so the agent re-enters and continues;
     verify the wakeup actually fires. Chain wakeups until the job is complete.
  3. During an unattended window, do the DELEGATED autonomous task first (it is
     the only thing that can't happen while the owner is asleep). Don't let
     interactive polish requests displace it; run them in parallel via
     background if needed.
  4. Checkpoint durably every step (commit, update CLAUDE.md/memory) so progress
     survives an interruption and is visible on waking.

## Project background

Custom hub-and-spoke dice board game. Full stack built from scratch: game
engine (`game.py`), Flask backend (`app.py`), Phaser.js frontend (`game.js`).
Current AI approach: GNN value network (`network.py`) driving 2-ply move
selection (`agent_gnn.py`), trained via self-play. Strongest checkpoint before TD(λ): `best_iter5_m46.pt`.

An earlier heuristic evolutionary track (`train.py`) reached 90%+ vs.
baseline but was recognized as fundamentally limited without a neural
approach and is no longer the active line of work. The heuristic agent is still used to filter top-K moves for the neural network, for speed.


## Domain facts worth knowing (distilled; full analysis in ARCHIVE.md)

Board geometry and endgame arithmetic that has been measured and
engine-verified. These keep coming up, so they live here rather than in the
archive; the derivations, the verification counts and the caveats are in
ARCHIVE.md under "Current benchmark".

- **Every goal is exactly 7 from the home tile.** So an entering piece reaches a
  goal only on a dice sum of exactly 7 (6 rolls in 36), never on one die, and at
  sum 7 *all six* goals are legal destinations at once.
- **Goals are paired 4 APART — 6&1, 5&3, 4&2** (the other gaps are 7, 11, 14).
  This is why a lone piece can step from a high goal to its cheap partner, and
  why two walls can seal a whole pair.
- **Landing on a goal makes the stage `endgame` within the same turn**, so "step
  onto a goal with one die, bank with the other" is legal in one turn.
- **Expected turns to bank a lone blank, by goal:** 1.000 / 1.029 / 1.125 /
  1.303 / 1.462 / 1.644 for goals 1-6. **A numbered piece takes 3.273 turns from
  its own goal, the same for every number** — the endgame higher-die rule is
  blank-only, so it needs an exact match (11/36 a turn). Numbered pieces are the
  endgame bottleneck.
- **Stack low and extras ride out free; stack level and they queue.** A blank on
  goal 1 under a numbered-2 on goal 2 costs nothing (3.273 either way); the same
  blank *on* goal 2 costs 0.059, and two of them 0.353.
- **A goal tile can never be blocked** (`is_blocked` requires `type == 'field'`)
  — only the routes to it. Sealing a goal from home takes 2 walls (4 pieces),
  and those same 2 walls seal its whole 4-apart pair.
- **Walls are worth ~3x more against a numbered piece** (best +0.398 turns, all
  of it on the outer end of the radial spoke to that piece's goal) than against
  a blank (+0.128). Against a blank, **28 of 63 field tiles are
  counterproductive** — a wall can shorten their route into die range.
- **Endgame offgoaling is NOT a defect** — it tracks the banking odds exactly.
  The agent never leaves goals 1-3 (where the value gradient is steep) and
  sometimes leaves 4-6 (where standing on goal 6 is worth 0.039 turns more than
  standing beside it).

## AI and model — current status

- **Deployed model: `model.onnx` = `blend4way_Oct6.pt` (shipped 2026-10-06)**,
  the plain AVERAGE of four nets' weights: symaug iter6 (the previous champion),
  iter10, iter14 and symaug iter11 (one lineage, so they share a basin).
  **Confirmed vs the previous champion on fresh seeds in the app's config:
  +0.158 pts/game (95% CI +0.056..+0.259), 52.3%, 2,000 games.** Sweep (300
  pairs each vs the old champion, same seeds): 4-way +0.153, 75/25 +0.102,
  50/50 +0.102, 3-way +0.043, 25/75 (iter10-heavy) -0.160. 4-way vs 50/50 head
  to head: -0.072 +- 0.143 (level). A play-time ENSEMBLE (champion + iter10
  values averaged) +0.150 +- 0.187 vs the old champion -- no better than a
  blend at twice the inference, not pursued. Calibration (calib_bench): +0.159.
  **Round 2 vs the NEW champion (300 pairs each): 6-way (+aux14, iter4) +0.020,
  champion-heavy 4-way (0.4/0.2/0.2/0.2) +0.010, 5-way (+aux14) -0.007 -- all
  +-0.19, i.e. the soup gain has saturated at the 4-way; don't repeat it.**
  Previous: `symaug_champ_July27_iter6.pt`. `REC_MODEL_TAG` = 'blend4way_Oct6'.
  **`sw.js` cache bumped to v9 and `model.onnx` now served `no-cache`** (it was
  cache-first with a week's max-age, so a new net could take a week to reach
  returning players). The Android package carries it from versionCode 8. Re-export with `BOARDGAME_DEVICE=cpu
  python3 onnx_export.py <ckpt> model.onnx` (self-verifies; on this iMac torch
  must be on CPU).
- **Inference runs on the device.** `local_agent.js` + the ported stack
  (`route.js`, `encoder.js`, `infer.js`, `engine.js`, `heuristic.js`,
  `agent.js`) answer every move; there is no application server in play. The
  port was proven by trace-diff, 110/110 chosen pairs. `PORTING.md` has the
  account.
- **Training is not the active line of work.** The TD(λ) programme, the arena
  measurements, the aux-head and symmetry-aug runs, deep search and the
  interpretability probe are all in **ARCHIVE.md**. The headline that still
  matters if training resumes: promotions are correctly measured but **~76% of
  the certified local gain does not generalise** (non-transitivity from
  parent-only gating), and the fix is a fixed-panel gate.
- `?dev=1` unlocks debug / eval / setup modes and un-silences `console.log`.
  **Any harness that reads console output must pass it.**

## Prefilter latency, measured 2026-10-04 (`latency_probe.mjs`, 104 moves, desktop Chrome)

Shipped (F=12, K=40): median 301 ms, p90 1.10 s, max 2.4 s. K=80: 1.8x median,
p90 +16%, move changes 2/104. No prefilter: 10x median, p90 11.9 s, max 56 s --
not viable. One-stage (F=0) is NOT slower than two-stage in JS (285 ms median),
unlike Python where two-stage cut the worst move ~3x. Prefilter cost by the net's
own valuation: misses its best on 10% of turns, ~0.23 pts/game (owner's 83 games);
the Aug 120-game match bounds a real loss below ~0.27. Leave it as is.

## Later idea: refit the prefilter heuristic for RECALL (owner, 2026-10-04)

The heuristic's weights were grid-searched to make it PLAY well; as the
prefilter its only job is to keep the net's favourite move among the 12 first
moves / 40 pairs it passes on. Fit for that directly: on the gaps runs' positions
(`weakness_gaps.jsonl` has the net's full ranking, ~4,000 positions and growing),
a logistic ranking fit pushing the net's top move above the alternatives, scored
by held-out keep-rate at the 12/40 cuts. Add the validated `features_v2`
quantities (threats, per-die roll counts, dice-count turns) as components. Today
it misses the net's best on ~10% of the computer's turns (~0.23 pts/game by the
net's own valuation; the Aug match bounds the real cost < ~0.27). Gain is bounded
by that, at zero latency cost; an afternoon's work, no retraining.

## Next training run: encoder changes (owner + Claude, 2026-10-04)

Any new input reshapes the input layer, so these need a **from-scratch** run, not
a fine-tune. Today's encoder (`encoder.py`; its header list is stale, the
per-function docstrings are right): tile = type, ring, sector, goal number,
neighbour count, my/opp piece count, a wall flag for the MOVER's blanks only;
piece = owner, number, numbered, status (unentered/home/board/saveable/saved),
rack slot, lone-on-field, fully walled off, own-goal distance band, distances to
all six goals (raw and banded, in GOAL order); global = dice, both stages,
numbered-saved per side, highest occupied goal per side, my saveable count.
6 message-passing rounds over tiles + one global node.

**Add:**
- **No-save counter** (drives the draw rule; the net is blind to it).
- **Opponent wall flag** per tile (only the mover's walls are flagged now).
- **Opponent saveable count.**
- **Per-side counts: unentered, on field, saved (total, not just numbered).**
- **Distance to OWN goal as a raw number, NUMBERED pieces only** (blanks have no
  own goal; their six distances already cover them). It exists only as one of six slots in
  goal order, so the net must gate the slot by the piece's number. Owner: distance
  is also a direct proxy for blockability (a piece on ring 1 has far more route to
  wall than one a tile from goal).
- **Per-piece threats: P(opponent's next roll can capture it), P(it can wall its
  route).** Two-dice threats are 7-12 tiles away, beyond 6 rounds of tile-to-tile
  passing. The EXACT value (`weakness_probe.threats`, every opponent reply over 21
  rolls) costs 0.4-0.9 s a position -- far too slow for an input, which is encoded
  for every search candidate. Use a distance-based approximation instead (capture:
  an opponent piece at route distance d hits on a die of d or a sum of d with an
  unwalled midpoint; wall: rolls that land two opponent pieces on a tile every
  shortest route of mine crosses), est. a few ms, and VALIDATE it against the exact
  version on logged positions before adopting. Needs a JS twin in encoder.js.
  Compute threats for BOTH sides' pieces (the opponent's exposure is my attack).
- **Expected turns to finish, per side** (from per-piece turns-to-bank: exact DP
  values on goals -- blank 1.00-1.64 by goal, numbered 3.27 -- approximated by
  distance elsewhere). Speaks to the bank-the-most bug and the endgame probe's
  "deep banks sooner" lean.
- **Race count per side** (backgammon's pip count): own-goal distance for
  numbered, nearest goal for blanks, +7 per piece on rack or home.
- Considered and dropped: a "contact broken" bit -- captures stay possible until
  every piece is on a goal, which the two stage features already say (owner).
- **If the features don't close the gap:** die-distance edges (tiles 1-6 steps
  apart, labelled), so two-dice threats sit within 1-2 hops and the net can learn
  threats itself. More general, costlier on the phone.

**Run plan (agreed 2026-10-04): two arms, same seeds and data.** Arm A = the
exact bookkeeping features together (no-save counter, opp wall flag, opp
saveable, per-side counts, own-goal distance, race count, turns to finish --
low-risk, no individual hypothesis, too small to ablate one by one). Arm B = A +
the threat features (approximate, cost phone time, and carry the one testable
claim: they close the opening blockability gap). If compute allows only one run,
run A+B and lean on the probes. In the SAME run, fix coverage too -- features do
not fix what self-play never visits: exploration (ARCHIVE.md) and rollout-labelled
positions from the ~1,200 turns where the net disagrees with owner. Gate against a
fixed panel that includes the deployed champion.

**BUILT (2026-10-04) on branch `train-features-v2`** (off `train-panel-league`);
its CLAUDE.md has the RUNBOOK. Feature sets v1/A/AB with the checkpoint carrying
its set; threats validated (capture exact, wall corr 0.98); per-die roll
opportunities exact vs the engine; turns to finish = dice count (corr 0.957 held
out) plus the exact endgame table (`endgame_table.json`) as its own input;
warm starts are the champion WIDENED with zero weights on new inputs (exact copy,
not distillation); start-position games from owner's 5,828 positions; late
exploration with a Watkins-style TD trace cut (off by default). Smoke run of the
full loop on AB: clean. Not built: rollout-labelled disagreement targets, the JS
twin of features_v2. Gate games ran 528 s single-core on the loaded iMac, so
launch real runs on a quiet machine or a cloud box.

**OPENING BLOCKABILITY IS PROBABLY NOT A REAL WEAKNESS (rollouts, 2026-10-04).**
In 99 opening positions from owner's games where the computer had a near-equal
alternative (within 0.25 pts by its own scoring) that left its numbered pieces
>0.25 less blockable, 24 paired playouts per move (net 1-ply policy both sides,
common dice) gave the safer move **-0.024 pts (95% CI -0.157..+0.109)**, better
in 46 / worse in 52, flat across exposure-cut sizes (0.25-0.5 / 0.5-1.0 / >1.0
all about -0.02). Owner: the net is strong at blocking, so the playouts' opponent
exploits exposure fine -- the extra exposure just doesn't cost points. Stopped at
99 of 171 (`rollout_probe.py block`). Arm B keeps the threat features anyway, as
the general case; capture-vs-goal (`rollout_probe.py cvg`, all 470 positions) is
the next lead.

**PREFILTER REFIT: FITTED COMPONENT SCALES (2026-10-05, on `testing` for owner
to play).** The heuristic's 23 components get per-component scales
(`prefilter_scales.json` -> Python `GNNAgent`'s prefilter copy only, JS via
`heuristic_weights.json` 'component_scale'; the plain heuristic player is
untouched; `PREFILTER_SCALES=0` restores the old ranking in Python). Fitted
(`prefilter_fit.py`) so the net's own best move survives the cut: keep-rate
88.5% -> 95.7% (5-fold CV by game, owner's 150 games) and 88.3% -> 96.4% on
UNSEEN computer-vs-computer positions. The offline replay of the filter matched
the real agent 45/45 (after using the TUNED weights: `Agent()` with no argument
loads INITIAL_WEIGHTS, `Agent(weights=None)` loads best_weights.json -- that
mix-up once looked like a stale-game-stage bug; it was not). JS/Python parity
50/50 on a regenerated fixture. **Arena, new vs old filter, 1,500 paired games:
+0.020 pts/game (95% CI -0.064..+0.103), 50.1% wins** -- no measurable strength
change, as expected: the old filter's cost was bounded small, and the net's own
0.23 estimate shrinks to ~0.07 by its measured 0.3 calibration slope. Shipped
for fidelity: the agent and the HINT now play the pure net's preferred move in
~96% of positions instead of ~88%. A 24th component (exact endgame-table
banking turns of the goal layout) added nothing in CV and was dropped. Later:
look for patterns in the remaining ~4% of culls.

**THE DIFFICULTY SLIDER IS MIS-SCALED FOR THE APP -- ITS FLOOR IS FAR TOO STRONG
(measured 2026-10-05, `difficulty_fine.py` on `train-features-v2`, the APP's
config: ONNX net, prefilter 12/40/5, hand rules on; each level vs d=1.0, 120
games = 60 colour-swapped pairs).**

        d     slider  win% (95% CI)   margin
        0.99   95%    47.5 (39-56)    -0.23
        0.97   85%    53.3 (44-62)    +0.04
        0.95   75%    46.7 (38-56)    -0.23
        0.92   60%    31.7 (24-40)    -1.05
        0.90   50%    35.8 (28-45)    -0.78
        0.85   25%    35.8 (28-45)    -1.22
        0.80   Easiest 25.8 (19-34)   -1.62
        0.70   (below) 7.5 (4-14)     -3.12
        0.60   (below) 0.8 (0-5)      -4.33

The September sweep below (0.80 = 0/94, margin -6.06) used `difficulty_arena.py`,
which samples among ALL candidate pairs; the app samples only among the ~40 the
prefilter keeps, so lowering d hurts far less in the app. The remap onto
0.8..1.0 therefore made "Easiest" nearly full strength, and the top quarter of
the slider (1.0-0.95) is indistinguishable from Max. Proposed (owner to decide):
**DONE (owner chose 0.65-1.0, 2026-10-05):** `getAIDifficulty()` is now
piecewise-linear through `DIFFICULTY_KNOTS` [pos, d] = [0, 0.65] [0.25, 0.74]
[0.5, 0.80] [0.75, 0.93] [1, 1.0] -- knots spaced by the measured win rate
(roughly 4% / 15% / 26% / 37% / 50% vs full strength), so equal slider steps are
roughly equal strength steps. Verified in the page: saved positions 0 / 0.25 /
0.5 / 0.75 / 1 give 0.65 / 0.74 / 0.80 / 0.93 / 1.0 and the labels Easiest /
Gentle / Medium / Strong / Max. The tutorial's "Go easy" (position 0) now means
0.65. Saved settings are REINTERPRETED (a player at Easiest gets much weaker),
deliberately. Rates are against full strength, not against a beginner.

**CALIBRATION OF OLDER NETS (`calib_bench.py`, 2026-10-05).** On the 559-position
benchmark (corr / slope of the net's move gap vs playouts): iter10 +0.186 / 0.42,
iter14 +0.160 / 0.42, iter4 +0.149 / 0.29, aux14 +0.137 / 0.32, symaug iter11
+0.097 / 0.30, **deployed champion symaug iter6 +0.090 / 0.30**, best_iter5
+0.085 / 0.18. The champion is among the LEAST precise about move differences
(SE of a corr ~0.04): the symmetry-aug run seems to have bought strength at some
cost in value precision. Keep iter10/iter14 on the panel; watch this benchmark.

**THE NET'S DISAGREEMENTS WITH OWNER ARE MOSTLY ITS OWN NOISE (overnight
2026-10-04/05, `rollout_probe.py disagree`, `overnight_summary.txt`).** 559 of
owner's positions from 150 games, his move vs the net's best, 20 paired playouts
each (net 1-ply both sides, common dice):

        net's gap band   n    net claims   playouts say      owner/net better
        >= 0.26         358     +0.42      +0.036 +- 0.069       175 / 171
        0.1 - 0.26      102     +0.16      +0.029 +- 0.145        46 / 54
        0.01 - 0.1       99     +0.06      -0.101 +- 0.132        58 / 37
        all             559     +0.31      +0.011 +- 0.057       279 / 262

Per position the net's gap predicts the playouts at corr +0.09, slope 0.29 -- it
is ~3x overconfident about how much moves differ. Only 47 of 559 are clear beyond
2 SE (21 owner, 26 net; chance alone gives ~28); median per-position SE 0.62. So
the "2.1 pts/game owner gives away" (net-judged) is an artefact of the net
scoring its own argmax over noisy values; owner's 57% / +0.38 is consistent.
**Consequences:** (1) the net's weakness is VALUE NOISE between near-equal moves,
not wrong ideas -- consistent with blockability, capture-vs-goal and endgame all
flat; (2) these playout labels are too noisy (+-0.6 vs real differences of
tenths) to train a pairwise loss on -- that plan is SHELVED, and the self-play
pairs batch with it; (3) keep the 559 as a CALIBRATION BENCHMARK for any new net
(slope / corr of its gaps vs the playouts); (4) the levers are better value
estimates: the v2 training run, and cheap noise reduction at play time such as
averaging over the 3 symmetric rotations (test it against this benchmark).
Capture-vs-goal (330 positions): goal beats capture +0.160 +- 0.094 overall, but
the net already knows it (corr +0.37; where it prefers capture, capture did
better, -0.231 +- 0.220) -- no over-capturing.

**Testing protocol for new features (owner):** validate any approximation against
the exact computation on logged positions; train with/without in otherwise
identical runs judged by a fixed-panel gate; then re-run `weakness_probe` /
`endgame_probe` on the new net to confirm the gap actually closed.

**Evidence so far** (owner's 83 recorded games, `weakness_probe.py`): the
computer leaves its numbered pieces more blockable than owner does in the OPENING,
per piece +0.070 (95% CI +0.029..+0.113; 0.290 vs 0.220); midgame and capture
exposure indistinguishable. Capture-vs-goal: computer captures 20/210 vs owner
19/260 (p=0.38), but 5/99 vs 1/124 when the capturable piece has barely left home
(p=0.013) and 9/39 vs 5/55 when its cheapest goal move is its 2 (lead only). The
net rates owner's moves 2.1 pts/game worse than its own while owner wins 48/83,
mean +0.25 -- so the positions where they disagree are the richest training
signal. Endgame horizon (`endgame_probe.py`, 90 self-play games): deep and
shallow agree MORE as the end nears (disagree 38% at 6 opponent pieces left, 15%
at 1); no broad endgame weakness beyond the bank-the-most case.

## Traps that keep biting

Method notes that have each cost a session or more. The full stories are in
ARCHIVE.md; these are the rules that came out of them.

- **A measured MECHANISM is not a defect until the SYMPTOM is observed.** "The
  UI freezes" was written up twice from a profile alone and disproved both times
  by owner simply looking. Symmetrically, a felt symptom is only fixed when it
  stops being felt.
- **Believe the owner's observation over the instrument.** A harness artefact
  once "measured" the AI forfeiting half its turns.
- **Read the denominator, not the zero.** A "0 of 0" reads exactly like a pass.
  Check that a fixture actually exercises the branch you think it covers.
- **Build test positions by playing into them, or check them by hand.** A
  hand-mutated board has stale derived state — `Player.getGamePhase()` is a
  cached field, `mustMovePieces` has two writers, `tile.addPiece` does NOT set
  `piece.currentTile` (only `piece.move` does). Any of these makes a correct
  result look like a bug.
- **Derived state beats a flag** whenever another code path already mutates what
  the flag summarises.
- **Test the gate, not the legality.** A test of a toggle must ask whether the
  branch was reached, not whether the outcome was legal.
- **A short-circuited reference is invisible to a test that never enters the
  branch.** Check the branch, not the function.
- **Grep the file after any scripted edit, and never let a test enable the thing
  it tests.** Both have produced confident false passes.
- **Never pipe a build into `tail` and trust the exit code** — it reports
  `tail`'s.
- **Browser-harness specifics:** `page.evaluate` runs in an ISOLATED world and
  cannot see game.js's globals — inject with `page.addScriptTag` and pass
  results back through a DOM attribute. `_currentGame` is a FUNCTION (reading it
  as an object silently yields `{}`). Pieces live on the GAME (`game.pieces`,
  24), tiles are `game.tiles` (94). Wait for `_gameFrozen === false`, not just
  for the welcome card to go. Set **both sides human before the game starts** —
  a tap on the computer's turn reads as a dead tap, and flipping a role mid-game
  races two agent requests. `camera.worldView` is all zeros until a frame
  renders. A healthy boot: 94 tiles, 24 pieces, `renderer.type === 2`, one
  active scene, 0 failed requests; the Canvas2D `willReadFrequently` warnings
  are the board bake's readbacks and are expected.
- **HARNESS: DRIVE SYSTEM CHROME, NOT A DOWNLOADED CHROMIUM (fixed 2026-09-24).**
  The browser-automation skill resolves `patchright` from the newest CodeGPT
  extension and then looks for that patchright's own chromium build. **That build
  cannot be installed here** -- patchright 1.63 dropped macOS 13 support and this
  iMac is Darwin 22.6 (Ventura); the 3.24.69 extension that had `chromium-1193`
  on disk is **gone**, so the workaround recorded here on 2026-09-20 is dead.
  **What works: point patchright at the system Google Chrome.** No download, no
  extension archaeology:

        const req = createRequire('/Users/tom/.vscode/extensions/danielsanmedium.dscodegpt-3.24.74/standalone/')
        const { chromium } = req('patchright')
        await chromium.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
          headless: true, args: ['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'] })

  The swiftshader args are what get **WebGL** in headless, so `renderer.type === 2`
  and the board actually bakes; without them Phaser falls back and the geometry
  measurements are not the ones the player sees.
  **PATCHRIGHT RELAYS NO PAGE CONSOLE AT ALL.** It suppresses `Runtime.enable` to
  stay undetectable, so `page.on('console')` sees browser-level warnings (the
  Canvas2D readback ones) and **not a single `console.log`, `warn` or `error` from
  the page** -- verified against a bare `data:` URL: zero messages of any type.
  So `?dev=1` is still right and still necessary for a human reading the console,
  but **a harness must assert through the DOM**, never by reading log output. A
  test that "found no errors" in the console found nothing at all.

## Project direction: a real Android/iOS app (owner, 2026-08-14)

Standing goal — the web client should eventually ship as an actual app, with
possible store presence. Not being built yet; **future changes should be made
with it in mind.** Assessment and the concrete implications:

- **Packaging is the cheap part.** A PWA (manifest + service worker) gets a
  home-screen icon, no browser chrome and offline loading in hours, no store
  needed. A Capacitor wrapper adds real store presence in about a day plus store
  admin (Play $25 once; Apple $99/yr and review) and changes no game code — the
  WebView is Chrome on Android, WKWebView on iOS, so rendering is what we have.
- **The real work is on-device AI**, which is what removes the server, the cold
  starts, the 504s and MOVE_BUDGET juggling entirely. `model.onnx` (1.66 MB)
  runs as-is under **onnxruntime-web** (WASM); what needs porting to JS is
  `encoder.py` (pure numpy, mechanical) and `agent_gnn.py`'s 2-ply search +
  heuristic prefilter, plus whatever of `game.py` the search touches (game.js
  already has move generation/validation for human play). Days to a couple of
  weeks; the risk is correctness, so verify with the established standard —
  seeded trace-diff (`PYTHONHASHSEED=0`) asserting identical move traces against
  the Python agent. A native rewrite (Kotlin/Swift/Flutter) is ruled out: it
  would discard a polished Phaser UI to solve a problem we do not have.
- **DONE (2026-08-14): Phaser is served locally** (`phaser.min.js`, 1.04 MB, the
  same 3.55.2 build the CDN served). Measured: zero CDN requests, game boots,
  play unaffected. This was the app prerequisite AND the fix for the opaque
  `Script error.` traces — Phaser exceptions now carry a real stack.
- **DONE (2026-08-14): web-app manifest, icons, fullscreen toggle.**
  `manifest.json` (display `fullscreen` — see the edge-swipe note below for
  why not `standalone`; orientation `any`, theme `#b5623b`) plus
  `icon-192/512/512-maskable.png` (generated by a PIL script — board motif),
  a `<title>`, apple-* meta and apple-touch-icon, so Add to Home Screen launches
  chromeless on both platforms. Settings gains a **Fullscreen** row, phone-only
  and only where `document.fullscreenEnabled` (absent on iPhone Safari — there
  the manifest route IS the equivalent). Fullscreen needs a user gesture, so a
  saved preference is applied on the first pointerdown after load, not at
  start-up, and a `fullscreenchange` listener keeps the checkbox honest when the
  OS exits it. Measured: manifest + all three icons 200, row present on phone /
  absent on desktop, toggle calls requestFullscreen and persists the setting.
  **The app is named QUAHURU** and every shipped surface already carries it:
  `manifest.json`'s `name` and `short_name`, `<title>`, and the
  apple-mobile-web-app-title meta. (Confirmed 2026-08-15 — the earlier note here
  called it an unchosen placeholder, which was stale.)
  **Not done: a service worker** for true offline. Deliberately deferred — with
  deploys this frequent, a cache-first worker would serve a stale game.js, the
  exact class of bug that costs an afternoon. Add it with the packaging work,
  network-first for game.js/index.html.
- **Consequences for work being done now:**
  1. ~~Phaser is CDN-loaded~~ — done, see above; keep it local.
  2. **Zoom/pan belongs inside Phaser** (camera), not on browser `touch-action` /
     `visualViewport`. Already decided after the selection bug; the app goal is a
     second, independent reason — a WebView arbitrates pinch-zoom differently
     and often not at all.
  3. **Portrait layout is a prerequisite**, not a nicety: phone apps are expected
     to work held upright. It is last in the queue but gates the app work.
  4. **Don't add server-only game behaviour** without remembering it needs a JS
     twin later. Settings already live in localStorage, which a WebView keeps.

## Current state

- **LEAVING A GAME BY ACCIDENT: AUTOSAVE + BACK HANDLING (owner, 2026-10-07).**
  Owner's phone: a tap near the edge read as a back swipe and left the site,
  losing the game -- in fullscreen too, because on Android Chrome a back gesture
  in fullscreen EXITS fullscreen, and the game re-entered it only on the first
  tap after load, so the next swipe navigated. Four layers:
  1. **Fullscreen re-enters on any tap** while the setting is on
     (`_armFullscreenOnFirstGesture` is no longer one-shot).
  2. **Web back guard** (`_armBackGuard`): while `_gameHasProgress()`, a
     pointerup pushes a dummy history entry, so back pops it (popstate, notice)
     and a second back leaves. **Plus `beforeunload`** ("Leave site?").
     **Owner, on device, fullscreen off: sometimes the notice, sometimes the
     prompt, sometimes it just leaves.** Chrome may skip page-added entries on
     its back button and may decline the prompt; neither is visible to the page.
     That is the browser's anti-hijacking stance, not a bug to chase.
  3. **AUTOSAVE** (`_savedGameWrite`, key `savedGame`, this browser's
     localStorage only): written at every TURN START from `switchTurn` (so a
     half-played turn resumes from its start with the SAME dice), cleared in
     `endGame` and when a fresh real game is built. Stores the position
     notation plus each blank's identity (`ids`, so recorded moves still name the
     right pieces), both isAI flags, starter, the no-save counter, matchTracker
     and the recorder's open entry (same id, `resumed` count; `merge_games.py`
     now keeps the duplicate that got furthest). The welcome card offers
     **Resume game / Resume match**. Measured: 10 self-play turns, reload,
     Resume -> notation AND every piece's identity identical, recorder 10 turns
     same id, play continues (10 -> 17 turns).
     **Resume REBUILDS the scene** (`_pendingResume`, picked up in create() after
     the HUD is built from the restored matchTracker; `checkInitialAIReady` and the
     save-clear both stand aside for it), so a match resumes with the match HUD.
     **Between games of a match** `endGame` writes `{between: true, match}`;
     Resume match starts the next game with `matchStarterForGame`. Measured:
     mid-match resume -> position identical, HUD [hidden New Game, New Match, How
     to Play] same as before; after game 1 ends 12-0, reload -> Resume match ->
     game 2, gamesPlayed 1, scores 12-0, black (the alternate starter) to move,
     and it plays.
  4. **Android app: `@capacitor/app` installed** (8.1.2; `cap sync` wired it into
     `capacitor.settings.gradle` / `capacitor.build.gradle`; `assembleDebug`
     builds). Back no longer reaches the WebView history: during a game the first
     back shows a notice, a second within 2.5 s MINIMISES (not exit); otherwise
     minimises at once. Without a listener the plugin would make back do nothing
     at all at the root, so the listener is required. **Ships with the next
     package (versionCode 8); not yet run on a device.**
- **ENDGAME MARK ON THE SAVED RACK (owner, 2026-10-07; on `testing`).** A
  player in the endgame gets a faint accent ground and accent edge on their SAVED
  rack, plus a one-off double pulse when they enter it. `_endgameMarkTick` runs in
  `MainGameScene.update()` and DERIVES the mark from `getGamePhase()` each frame;
  the pulse fires only on a change seen on the same game, and tutorial / `?pos`
  layouts call `_endgameMarkSync` (silent). Tutorial step "The endgame" says the
  rack lights up -- text trimmed so it is no taller than before (desktop 96px
  same, portrait 174 -> 152, landscape no longer scrolls). Measured: stepping the
  12 onto goal 3 via the real handlers -> phase endgame, mark on, 1 pulse, at
  desktop / portrait / portrait+insets / landscape.
  **Radial lines on the phone (owner, Pixel, 2026-10-08):** Phaser 3.55's
  `strokeRoundedRect` does moveTo(p) then an arc starting at p, and WebGL
  `batchLine` divides by the segment length unguarded -> NaN vertices at every
  corner; phone GPUs draw them as slivers to screen centre (swiftshader drops
  them, so headless never shows it). Always there in the pale grey rack edge;
  the accent mark made it visible. **Use `strokeRoundRect(g, ...)` (game.js), never
  `strokeRoundedRect`.** Measured: 60 zero-length segments per 400ms -> 0.

- **POSITION NOTATION, `?pos=`, AND `pos_image.mjs` (owner, 2026-10-02; dev only,
  on main since 2026-10-03).** A FEN-like text form, documented beside `positionToNotation`
  in game.js:
  `b 3'3 | W 2@6.6 x@G1 ... r:- s:1,3,5,x | B 1@3.2 ... r:- s:... | f:1@4.12~3.2`
  -- side, dice (`'` = used, `-` = roll fresh), per side `<n>@<loc>` with n 1-6
  or `x` (blanks are interchangeable) and loc `ring.sector` / `G1`-`G6` / `H`,
  `r:` rack front first, `s:` saved (owner wanted it though it never affects
  play), optional `f:` = first move `<n>@<start>~<now>`. 7 blanks + 5 numbers is
  accepted (the last-piece rule); the missing number becomes a blank 13.
  **Tools:** `?dev=1&pos=<notation>` opens a playable board (both sides human;
  `&posai=w|b|wb` hands sides to the computer; `&posshot=1` hides DOM chrome);
  Settings > Position (dev) > Copy / Load...; `node pos_image.mjs "<notation>"
  out.png [--phone] [--caption ...]` renders through the game itself (finds
  patchright under ~/.vscode/extensions and the system Chrome, so it runs on
  either machine). Result for harnesses on `body[data-pos]` ("ok" / "error: ...").
  Without `?dev=1`, `?pos` is ignored and the Settings row is absent.
  **Measured:** round trip (load -> export -> load -> export) identical, with the
  recorder fingerprint identical, for case 1 and for a position from 90s of real
  self-play; a mid-turn export (`f:` + a spent die) reloads with the moved piece's
  reachable set identical to the live game's (1.3 2.4 3.5 4.4 both); a legal move
  plays after loading; `posai=b` makes the computer move (it saved blank 9 and put
  its 1 on goal 1 -- case 1 with the new save rule); the 7-blank form loads and
  re-exports identically; malformed input is refused with a reason.
  **Not built: screenshot -> notation.** Claude reads a screenshot directly when
  given one; an automatic reader would be fragile across devices and themes.

- **NEVER PASS A DIE THAT COULD SAVE A PIECE (owner, 2026-10-02; on main and in
  versionCode 7 since 2026-10-03).** After the probe below found 3 / 1002 clear give-aways (images:
  `case_a1-3.png` in the repo root), the search drops any candidate pair that ends
  with a die unused while a save -- blank OR numbered -- is legal for it, whenever
  any candidate survives. `wastesSave` in agent.js, `_wastes_save` in agent_gnn.py,
  evaluated where the search already holds each candidate's resulting position;
  both restore the mover's stage, which `getValidMoves` rewrites and later
  candidates read. Measured with the REAL model on the three logged positions:
  rule off reproduces each logged pass exactly; rule on, cases 1 and 2 now save the
  blank, case 3 saves the 6 and spends the 5 on a field move (allowed -- no die
  wasted). JS with an adversarial stub preferring no saves: no chosen pair leaves
  a die idle with a save on. `agent_test.js` 50/50 (the rule never fires in that
  fixture).
  **IF A STRONGER MODEL IS EVER TRAINED, TRY IT WITH THIS RULE OFF** (owner): it
  may have learned this itself, and conceivably a piece left unsaved is sometimes
  worth keeping on the board as a spare capturer of more important pieces.
  **KNOWN WEAKNESS OF THE CURRENT CHAMPION, deliberately NOT ruled out (owner):**
  class (b) below -- spending a die on a blank save when it could put a numbered
  piece on its goal, 45 / 1002 qualifying positions. Harder to fix by rule (13 of
  the 45 were two saves, arguably fine); a training target if training resumes.

- **HOW OFTEN THE NET MAKES TWO SUSPECTED MIDGAME ERRORS (owner's question,
  measured 2026-10-02).** Deployed champion, 2-ply, full strength, Python reference
  agent. Positions from games played forward by the net's 1-ply policy; at each
  midgame turn start all 21 rolls screened by move generation, one qualifying roll
  per class put to the 2-ply search. 1,002 positions per class (the rolls are drawn
  uniformly among QUALIFYING rolls, so these are rates per qualifying position, not
  per game):
    (a) a blank save is legal, yet a die is left unused while that save was still
        available for it -> **3 / 1002 (0.3%)**. Rare but real; consistent with
        owner seeing it twice.
    (b) one die can save a blank or put a numbered piece on its own goal, and it
        goes on the blank save with no numbered piece reaching its goal ->
        **45 / 1002 (4.5%)**: 24 save + another move, 13 TWO saves (arguably
        defensible), 8 save + pass. The net chose numbered-to-goal in 871 of 1002.
  A flag is the class definition firing, NOT a proven error -- the deeper search
  over all 21 replies is what could arbitrate. Script was a scratchpad one-off
  (`probe_classes.py`), not committed.

- **CERTAIN LOSS -> NO SHUFFLING (owner, 2026-10-07).** Inside the bank-the-most
  rule below, among the max-save pairs keep only QUIET ones: every half a pass,
  an own save, an entry onto home (the rack obligation), or a move onto a goal
  the piece can bank from (a numbered piece onto its own goal from anywhere; a
  blank onto any goal but NOT from another goal -- owner: shuffling between
  goals is pointless); if none, the max-save set stands. Max saves come first,
  so a goal-to-goal step that ENABLES a save is still played (measured: roll
  4-4, blank G2 -> G4 then banked). Second position (`B 3@G3 6@4.12 x@G2 x@G5
  x@3.6`): quiet 10/21 before, 20/21 after (the 21st is that 4-4), JS = Python.
  **A lone NUMBERED 1 on goal 1 also triggers it (owner, 2026-10-07)** -- the
  last-piece rule blanks it. Nothing to add: both engines apply that rule to
  BOTH sides in the endgame when the search's position is built, so the piece
  already reads as number 13. Measured (`W 1@G1`, same black): 20/21 quiet, JS =
  Python, vs 10/21 with the rule off. `quietPair` (agent.js) / `_quiet_pair` (agent_gnn.py).
  **Quiet pairs are also exempt from both prefilter culls** while the loss is
  certain -- without that the cull had already dropped them (even plain pass),
  and the rule fired on only 17 of 21 rolls. Measured with the real net on
  `b - | W x@G1 ... | B 3@G3 6@4.12 x@5.2 x@3.6 x@1.4 ...`, all 21 rolls: quiet
  pair chosen 11/21 before, **21/21 after, identical in JS and Python**.
  `agent_test.js` 50/50.

- **CERTAIN LOSS NEXT TURN -> BANK THE MOST (owner, 2026-10-01; on main and in
  versionCode 7 since 2026-10-03).** Owner saw the computer, unable to win, bring a piece
  onto a goal instead of saving when his last two pieces were blanks on goal 1 --
  any roll banks both, goals cannot be blocked or their pieces captured, so the
  game is lost and only the margin is in play. **Not a heavy-loss effect** (owner:
  it happens at ordinary margins). Rule, exactly as owner specified and no wider:
  opponent's only unsaved pieces are 2 blanks on goal 1, and no pair wins this
  turn (the search returns a winning pair before scoring, so reaching the scoring
  step implies it) -> keep only the candidate pairs with the most OWN saves; the
  net, ties, difficulty, the draw check and hint rankings choose among those.
  `opponentWinsNextTurnRegardless` / `ownSaves` in agent.js, twins in
  agent_gnn.py. Every max-save pair is a candidate (save pairs are exempt from the
  heuristic cull; first moves that enable a save are kept). Measured in node with
  an adversarial stub net preferring no-save pairs: condition -> chosen pair saves
  1 of a possible 1, all 7 surviving candidates save 1, and 30 picks at
  difficulty 0.8 never go below; control (one white blank on goal 2 instead) -> a
  no-save pair, 47 candidates, unchanged. `agent_test.js` 50/50.
  **WIDENED TO ONE OR TWO BLANKS (2026-10-03).** Owner's recorded game had white's
  LAST piece, one blank on goal 1, and the rule (exactly 2) did not fire: black
  (1-5) played 4 -> 6.4 -> 7.2 instead of saving its blank 11 off goal 5, and lost
  12-10 rather than 12-11. Replayed from the log with the real model: old rule
  reproduces the logged pair exactly, new rule saves the 11. `agent_test.js` 50/50.

- **THE RECORDER KEEPS WHAT THE COMPUTER MEANT, AND THE BOARD USES THE AGENT'S
  SAVE DIE (owner, 2026-10-01).** Owner twice saw the computer, at Max, pass its
  second half with a legal save of a blank on 6 / a 2-to-goal available. Two
  explanations: the net chose it, or the BOARD refused the half -- a refused tile
  move in `applyMovePair` ends the turn and reads exactly like a pass, and `m` in
  the log records only what was played, so the two were indistinguishable. Each
  computer turn now also carries **`a`** (every pair the agent returned, agent
  format WITH the die: `"7>5.4:3"`, `"7>s:6"`, `"o7>b"`, `"-"` pass; a second
  entry is the extra-move re-ask) and **`x`** (any half the board refused). Both
  absent when empty; `replay_games.py` ignores them.
  **A real divergence found on the way:** `Piece.save()` picks its own die and,
  for a blank, avoids one a numbered piece "needs" -- ignoring the die the agent
  named, which the engine had marked used and chose the other half against. The
  AI path now passes it (`save(dieRoll)`), honoured when it is a legal die; human
  saves pass nothing and keep the smart pick. Tile moves have no such gap: a
  single-die move's die is fixed by the distance.
  Measured: 2 self-play games in the browser, recorder on -- 105/105 turns carry
  `a`, 35 of them with a save, 0 refused halves, 0 turns where the agent's pair
  has more real moves than the board played, and `replay_games.py` 2/2 clean.
  Self-play is not where owner saw it, so this shows the plumbing works, not that
  the bug is gone. Still to diagnose: owner's two cases, once a recorded game
  shows one (look for `x`, or `a` containing a move that `m` lacks).

- **FROZEN SCREEN AFTER NEW MATCH FROM THE END CARD, ON A PHONE (owner,
  2026-10-01).** `EndGameScene.create` added a `this.scale.on('resize')` listener
  and never removed it; the ScaleManager is game-wide, so after the main scene
  took over, the next resize (Chrome's URL bar) called it on a stopped scene with
  no camera and it threw. **When Phaser's own size check runs that emit inside the
  game step, the throw kills the rAF loop permanently** -- measured: loop frame
  stuck at 134 forever. Now removed on `shutdown` and guarded by
  `sys.isActive()`. Measured after: frame keeps advancing, 0 errors over six
  resizes (was one per resize). **Rule: any `scale.on` / `window` listener added
  in a scene's `create` needs a matching `off` on `shutdown`.**

- **A MOVE REFUSED BECAUSE ANOTHER PIECE MUST MOVE NOW SAYS WHY (owner,
  2026-10-01).** It used to answer only with the amber pulse on the obliged piece.
  `_refuseForObligation(game, piece)` replaces `_flashMustMove` at all five
  refusal sites (three in `Piece.handleClick`, two in `movePiece`) and names the
  rule: a captured piece first / keep a die for the front rack piece / rack order.
  **First try while first-game rule tips are on, otherwise the second try at the
  same piece with the board unchanged** (`_whySig`, the route explanation's rule).
  Tagged `'move'`, so dismissible. Never in the tutorial; human turns only.
  **The commonest case was invisible to those five sites:** with the entry owed the
  game WITHHOLDS the sum from every other piece, so trying to spend both dice on a
  board piece is not a refusal at all -- the tile is simply unlit and the tap goes
  to `_noticeWhyUnreachable`, which deliberately stays silent when the dice CAN make
  the distance. That branch now hands an outstanding obligation to the same helper.
  Measured, tips on / off: captured piece waiting, tap a board piece -> text on try
  1 / only on try 2; entry owed, tap a sum destination for a board piece -> same;
  third rack piece -> same. Control: a legal one-die move in that position moves,
  spends one die, no notice; the tutorial still plays through.

- **ADVICE NOTICES CAN BE WAVED AWAY, AND GO WHEN YOU ACT (owner, 2026-09-30).**
  A notice tagged `'tip'` (rule tips, the hint nudge) or `'move'` (why a move was
  refused, the hint's suggestion) now takes the pointer and shows an ×: a tap
  anywhere on it, the ×, or a swipe dismisses it (`_dismissNotice`, sliding off
  in the swipe's direction). It also goes when a HUMAN shows they are ready --
  taps one of their own pieces (`handleClick`), moves (`_clearMoveNotice` at the
  commit points), or ends the turn (`switchTurn`) -- via `_dismissAdvice`, which
  does nothing on the computer's turn so a tip is never swept away unread. An
  untagged STATUS notice ("White passed", "Getting the computer ready") keeps
  `pointer-events:none`, no ×, and its full dwell. Dismissing a tip lets the next
  queued one follow 800ms later. Measured, each path: ×, swipe, own-piece tap, end
  turn all dismiss; the computer's move does not; a status notice survives an
  own-piece tap. **The hint lamp is hidden while the end card (`EndGameScene`) is
  up** -- `refreshHintButton` checks it and `_hintTick` re-derives it every 250ms,
  so it returns with the next game (measured: flex -> none on the card -> flex).

- **TUTORIAL POLISH (owner, 2026-09-30).** (1) **Desktop text 14.5 -> 16px, title
  17 -> 18.5px**, phones unchanged -- and the desktop card widened 640 -> 760 so
  the TALLEST step (which fixes the board's size for all steps) got SHORTER, 227 ->
  216px: measured at 1280x900 / 1440x900 / 1920x1080 / 1366x768 / 1024x768, the
  board grew 11px at each and no step's text scrolls. (2) **Step 7's move is
  animated along its real route** (`routeAnim: true` on the step ->
  `Game._routeBetween`, a BFS under `_bfsDistances`' own rules, ->
  `Piece.animateRoute`, 220ms a tile). Measured: 9 steps, 3,3 up spoke 4 into
  goal 2, round 7,21 / 7,20 / 7,19 into goal 4, passing within 6 world px of goal
  2's centre. Every other move keeps the plain 160ms slide.
  (3) **A step completes only once no piece is still animating** (`_tutPoll`), and
  **step 7's Black reply plays under card 8** (`blackAfterCard` ->
  `_tutNextThenBlack`), since card 8 describes that wall: owner saw Black set off
  while the 4 was still travelling. Measured order: route ends -> "✓ Nice!" -> card
  8 with Black's 10/11 still on 5,21 / 5,22 -> "Black plays…" -> both on 6,4 ->
  buttons back; input held (`_tutPieceOK` refuses while `_tut.busy`).

- **THE TUTORIAL SCRIPT IS ORDERED AND ONLY ITS PIECE IS SELECTABLE (owner,
  2026-09-30).** Each step with moves has a `seq`: the exact sequence -- which
  piece, and a `tile` / `save` / `block` / `end` (end turn). The CURRENT item is
  the first whose `done(g)` is false, DERIVED from the board every time (so undo
  walks it back -- measured). `_tutPieceOK` gates `Piece.handleClick` and
  `handleDoubleClick`: only the item's piece, or, with it selected, whatever
  stands on its destination (tapping an enemy piece there is how a capture is
  made). A refused tap flashes the right piece (`_flashPieces`, the must-move
  amber, now shared); ending the turn in step 10 before the save flashes the piece
  and does not end it. `_tutMoveOK` / `_tutSaveOK` / `_tutBlockSaveOK` answer from
  the current item, so moves happen ONLY in order, and the second rack piece can no
  longer be brought out first (it is simply not the piece named). The destination
  TILE IS FILLED in the colour its move will light up once the piece is selected
  -- the die that reaches it, or the sum yellow, read from the mover's own
  `getReachableTilesByDice` (owner: why a new colour?); pale violet
  `TUT_TARGET_FILL` only if no die can be named. Via `tile._tutTarget` /
  `_tutTargetColor` in `drawTile`, beneath any pieces on it from the moment its item becomes current --
  step 2's "highlighted tile" is highlighted before anything is touched. **Not a
  ring: owner found a ring read as "this is the piece to move".** Only a save or
  block-save -- where the thing to act on IS a piece -- rings the piece. Redrawn by
  the 300ms poll when the item changes, gone during "✓ Nice!"; a selected piece's
  own destination colours still take precedence. Hover lights only the scripted
  piece too (`onHover` asks `_tutPieceOK`; owner saw the second rack piece light).
  The old per-step `move`/`save`/`blockSave` predicates are gone from every step;
  the fallbacks in the three hooks remain for a step with no `seq`.
  Measured, desktop AND `?phone=1`, the whole tutorial played through the real
  handlers trying the wrong thing first at every step: every wrong tap refused and
  flashed (second rack piece in steps 2/3/4/5, the piece already moved, the 10 before
  the 6, the white 2 during the block-save, 11 before 12 moves, end turn before the
  save), exactly one lit destination per move, both captures (one made by tapping
  the enemy piece), and it reaches "You win!".

- **THE ENGINES DISAGREED ABOUT THE RACK ENTRY (owner's question, 2026-09-30).**
  The rule: at least one rack piece enters each turn, unless a captured piece is
  waiting, in which case THAT comes out and the other die is free -- the entry may
  be the first move or the second. game.js enforces exactly that. `game.py` and
  its port `engine.js` got there differently: at turn start they offer ONLY the
  entry (so the agent always enters first), and `must_move_unentered` then returned
  False for ANY first move. Equivalent for the agent's own turns, **wrong for a
  state where a board piece moved first** -- which only the HINT produces: the
  engine offered a free second move game.js refuses, and the hint said "No hint for
  a half-finished turn". Measured on main: white 7 moved first, the engine offered
  `pass, w7, w8(rack)` where game.js allows only w8.
  **Fix, in both engines:** the obligation is met by the first move only if its
  ORIGIN was the rack (engine -1 / game.py None) or the home tile (a captured
  re-entry). Measured after: the same position offers only `w8(rack)` and the hint
  marks that entry, legal; entry-first still leaves the second die free (`pass,
  w7, w8, w9(rack)`).
  **The origin now matters, so `getGameState` sends it** (`_turnFirstMove`, from
  `_turnStartTile`; the hint already did). The old marker -- a posted
  `reachableBySum` -- names the piece's CURRENT tile, so an entry made first would
  have looked like a board move and been demanded twice. That reaches the
  computer's own mid-turn re-asks (a single-move reply; the extra-move request when
  a die is left over), not just hints.
  **Nothing about the agent's own play changed, measured:** `agent_test.js` 50/50
  candidate sets and chosen pairs against the OLD fixture, the fixture regenerated
  from the new game.py is BYTE-IDENTICAL, and a full self-play game in the browser
  completed (12-9).

- **LEARNING CURVE, ROUND 2 (owner, 2026-09-30; on main since the same day).** A new player met
  "send the front one out" before being told what the game is FOR -- "save all
  twelve" first appeared in the closing panel. Three changes, owner's picks from a
  list of eight (the others -- a persistent goal line in the card header, board
  labels, splitting heavy steps, a first-launch tutorial offer, a "learning"
  opponent -- were not taken up).
  **1. A NO-MOVE INTRO STEP, "What you're playing for"** (`intro: true`, now step 1
  of 12). It plays one white piece's whole life ON A LOOP with the real pieces:
  rings on both saved racks -> slide rack -> home -> slide out to 4,6, CAPTURING
  Black's lone 5 there (sent home with the capture burst) -> goal 5 -> saved rack
  -> reset, ~13.2s a loop. **No arrows** (owner: "too low-tech"; the slow 900ms
  slides carry it) -- they were in the first cut. **The card sits ON THE BOARD and
  shows ONE SENTENCE AT A TIME** (owner: following the moves and a paragraph at once
  was too much): `step.beats`, each replacing the last with a fade as its leg plays
  (paced for slow readers, measured 0 / 4.2 / 7.7 / 12.2 / 15.7s, loop 19.9s; no
  saved-rack rings any more), card 520 wide / 19px text, text height held at the
  tallest beat so the card never resizes. `_tutPlaceIntro` grid-searches the
  viewport (12px) for the spot nearest "just above the home tile" that covers none
  of the demo path, the home tile or the four rack panels. Measured: desktop above
  centre, 0 covered; portrait phone (with and without insets) the band below the
  racks, 0 covered; **landscape phone has no clear spot** -- a compact card
  (`_tutIntroCompact`, H <= 560: title folded into the header, 15px, 440 wide)
  covers 9-15 sample points, all rack-panel corners, not the path. The board keeps
  the size the other steps reserve, so nothing jumps at Start (canvas 962x641
  before and after), which on desktop leaves the card's usual band empty during
  step 1. The card's usual placement returns on Start and comes back on Back.
  **Start appears only after the text has been through once** (owner):
  hidden-not-absent (`#tutStart`, visibility) so the row does not shift, revealed by
  the demo at 18.5s (`_tutRevealStart`); `_tut.introSeen` keeps it shown on a later
  Back, reset by `startTutorial`. Only the button advances step 1 -- no key does.
  Trap hit: `style.padding = ''` DROPS the card's own padding from its cssText; set
  the value. Button reads **Start ->**, and the
  intro LOCKS INPUT (`_inputLocked` returns true for an intro step; every other
  tutorial step stays exempt) because the demo is moving the pieces a tap would
  pick up. `_tutDemo.run` is a generation counter: every timer checks it, so
  `_tutDemoStop` (called from `_tutRender` and `_tutEnd`) is just a bump.
  Phaser advances tweens by capped frame deltas, so on a slow frame rate a slide is
  still under way when the next leg begins (measured headless); a new slide stops
  any unfinished one on that piece first.
  **BACK BUTTONS (owner, 2026-09-30)** on every step after the first, including
  the closing panel. Every step lays out its own position, so Back is just
  `_tutRender` of the earlier step -- with two catches. (a) `_tut.gen` is bumped on
  every render and checked by the "✓ Nice!" pause and Black's scripted reply, so a
  reply belonging to the step being left cannot fire into the one arrived at.
  (b) **The last-piece step BLANKS white's 2** (number -> 13, label destroyed), so
  every earlier position would be laid out a piece short; the step records
  `_tutNumber` and `_tutApply` restores it via `Piece._makeNumberText` (split out
  of `drawPiece` for this). On a landscape phone Back is a bare "←": the card is
  ~240px and Exit + "← Back" + Skip measured 225px against a 204px row and wrapped.
  Measured: forward to the end then Back x11 -- 24 pieces at every step, the 2
  numbered with its label from "Some dice do nothing" down, no Back on step 1;
  Back during "✓ Nice!" and 3.5s later still on the earlier step.
  Measured (intro): legs in order (rack/home/4,6 with black 5 home/goal5/saved/reset),
  clause lit per leg, a tap during the intro selects nothing, **Start pressed
  MID-DEMO** (piece out, black captured) lays the next step out clean and its move
  still works (lit 3,10 + 5,6, piece lands on 5,6, one die spent).
  **2. EVERY STEP'S FIRST SENTENCE SAYS WHY** ("Nothing reaches a goal from the
  rack, so...", "A capture costs your opponent a whole trip", "Walls are how you
  slow your opponent down", "A wall doesn't stop you, it makes you pay", ...).
  Length was paid for by cutting what the intro now shows (the home-then-spoke
  route in step 2) and tightening the longest steps. **Measured against main's
  game.js at 5 viewports** (desktop, portrait, portrait + `?safeinset=48,0,56,0`,
  landscape, landscape + `?safeinset=0,48,24,48`): desktop canvas height
  IDENTICAL at every step (tallest card unchanged at 227px, nothing scrolls); on
  phones every existing step's `#tutText` overflow is <= main's, and the three
  that already scrolled got shorter (Buy the door open 86 -> 43px, The endgame
  64 -> 21px, Take what's exposed 43 -> 21px in landscape).
  **3. RULE TIPS FOR THE FIRST REAL GAME** (`_ruleTipTick`, 300ms poll): one
  notice the first time each rule comes up -- the opening obligation (first turn
  of the game), the first capture, the first wall, the first rack emptied (saving
  starts), the first endgame, the first last piece losing its number, and **the
  first block-save** (owner's addition). **Each fires for WHICHEVER SIDE does it
  first (owner, 2026-09-30)**, not only the human -- the computer's first capture
  teaches capturing as well as the player's own. The wording is picked by who did
  it: "you" when the one human in a game against the computer did, otherwise "The
  computer" / the colour. Seven ids (`opening capture wall saving endgame
  last-piece block-save`), each fired once ever (`ruleTipsSeen`); tips queue
  rather than overwrite each other and wait out the hint nudge.
  **SWITCHING THE SETTING ON RESETS THE SEEN LIST (owner, 2026-09-30: had it on,
  saw nothing).** Once-ever plus the auto-off after the first game meant a player
  who turned it back on got silence -- every id was already in `ruleTipsSeen`.
  Reproduced (all ids seen, setting on: 0 tips in 4 turns of a real human game),
  fixed (`_resetRuleTips` on the checkbox going ON, and in `_tutFinish`): same
  profile after off/on, 3 tips in 4 turns. Switched off, the scan also drops the
  game's baseline, so switching on mid-game does not replay what happened while
  it was off. A fresh profile's first real game, measured with human moves: 3 tips
  in 8 turns (opening, wall, the computer's capture).
  **Block-save wording says "thins", not "breaks" (owner):** saving one piece off
  a wall of THREE leaves a wall, so every block-save text (tip, hint, How to Play)
  now says a wall of TWO becomes a single piece.
  **Double-click-to-goal now explains a waiting captured piece** ("A captured piece
  has to come back out first ...") -- it was another silent decline. Measured with
  its control: same position, captured piece on home -> that notice, no move;
  without it -> the piece goes to goal 5 on both dice. `sumToGoal` is OFF by
  default, so a test of this path must switch it on or it never gets past line 1.
  **FIRST GAME ONLY (owner):** the setting turns ITSELF OFF when a game the tips
  were watching FINISHES; an abandoned game does not count. On for a first-ever
  visit (seeded with hints) and after the tutorial (`_tutFinish`); Settings row
  "Explain rules as they come up (first game)".
  **Detected by polling the position, not by hooks** -- except captures and
  block-saves, which the board cannot tell apart from an entry or a turn switch,
  so `capturePiece` and BOTH block-save sites (the human gesture and
  `applyMovePair`'s branch -- the same pair the recorder needed) append to
  `game._captureLog` / `game._blockSaveLog`. A baseline is taken the first time a
  game is seen and nothing fires for what was already true then.
  Measured: a fresh profile seeds `ruleTips=1` and the opening tip appears on the
  human's first turn (after the computer opened); with `ruleTips=0` nothing in 25s
  in the same state. A full self-play game with the scan fed a proxy that reads
  white as human fired **9 of 10** -- opening, capture, captured, wall, enemy wall,
  saving, block-saved, block-save, endgame; last-piece did not arise (white lost
  10-12) -- and on game over `ruleTips` went to 0 and the Settings box unticked.
  (That run predates the either-side rework.) **Re-run after it**, same setup, 6
  tips: opening, capture (white's, "Capture!" form), wall, saving, block-save,
  and **"The computer is in the endgame"** -- Black got there first, so the tip
  fired for the COMPUTER's side in the other-side wording, which is the new
  behaviour. Last-piece again did not arise. `ruleTips` -> 0 at game over.
  **`WORDING.md` IS A READ-ONLY SNAPSHOT OF EVERY PLAYER-FACING STRING** (tutorial,
  tips, hints, route explanations, notices, confirms, welcome, end card, How to
  Play, Settings labels), made for owner to read and edit. **It is NOT shipped and
  the game does not read it** -- owner asked for exactly that, rather than moving
  the strings into a loaded file. When he edits it, carry the edits into game.js by
  hand; when wording changes in game.js, update it too, or it drifts silently.
  **Harness note:** the CodeGPT extension is now **3.24.75** -- update the
  `createRequire` path. And headless Chrome's canvas lags the DOM by seconds, so a
  screenshot of an animation must FREEZE it (clear the timers, wait ~2.5s) first.

- **THE TUTORIAL DID NOT START FROM THE END-GAME CARD (owner, 2026-09-29).** From
  Settings over the end card it half-started: the gear vanished, the end card stayed
  up, and a pale band appeared at the bottom.
  **Cause: `endGame` starts a SEPARATE `EndGameScene`, while `_setupScene()` returns
  `scenes[0]` -- the MainGameScene OBJECT -- whatever is actually running.** So
  `startTutorial` found a game, set `_tut.active`, hid the gear and refitted the
  canvas, but nothing was drawn: the main scene was stopped and the end card was
  still on top. `startTutorial` now defers via `_tutPendingStart`, and the main
  scene's `create()` resumes it a frame later.
  **AND `SceneManager.start()` IS NOT `ScenePlugin.start()`** -- it does not stop
  anything else, so the first fix left EndGameScene running and its card still over
  the board: the symptom, half-cured. The active non-main scenes are stopped
  explicitly first.
  Measured: end card up (`endActive` true, `mainActive` false) -> Settings >
  Interactive tutorial -> `mainActive` TRUE, `endActive` FALSE, tutorial active,
  the Step 1 card on screen, 94 tiles / 24 pieces.
  **THE PALE BAND IS NOT PART OF THIS BUG.** Measured against a control -- the
  tutorial started the normal way from the welcome card, same viewport -- and it is
  **identical, 251px**: the desktop tutorial reserves width for the side card, the
  board's 3:2 aspect then gives a 641px-tall canvas in a 900px viewport, and the
  band is index.html's `rgb(213,219,228)` showing through. `_paintPageGround`
  deliberately paints the page the theme's ground on PHONES ONLY, because desktop
  letterbox bands have always been that colour. Left alone rather than changed,
  since that is a recorded decision, not an oversight.
  **Harness note:** Phaser queues `scene.start` to a frame boundary and headless
  Chrome paints unreliably, so a scene-transition test must WAIT on
  `scene.isActive(key)`, never sleep -- an earlier run clicked while MainGameScene
  was still active and exercised the wrong path entirely. And close one Phaser page
  before opening the next: two live ones under software WebGL starve each other
  until the second page-load times out.

- **THE STACK PICKER'S CHIPS ARE NOW DOUBLE-TAPPABLE (owner, 2026-09-28: "the
  expanded line of pieces that appear when you tap the plus sign ... should be
  double tappable").** Two separate gaps, neither of them about timing:
  **1. Own chips had NO double handler at all** -- only select and drag -- and the
  plain-tap path called `hideStackPicker()` immediately, so **the chip was gone
  before a second tap could land on it.** That is the whole reason it did nothing.
  The first tap still selects with no added latency, and the picker now LINGERS for
  `STACK_PICKER_DBL_MS` (300, matching `Piece.handleClick`'s own window) instead of
  closing at once; a second tap on the same chip is the double-tap. Delaying the
  SELECT instead would have put 300ms in front of every pick.
  **`Piece.handleClick` calls `hideStackPicker()` itself** -- right for a tap on the
  board, wrong for one that came from the picker -- so the chip path restores
  `display = 'block'` after selecting and lets the timer close it. Without that the
  linger silently did nothing.
  **2. Opponent chips used `chip.ondblclick`, a MOUSE event**, so the block-save
  gesture was desktop-only; and the own-chip pointerdown's `preventDefault()`
  suppresses the synthesised dblclick sequence anyway. Counted from pointer events
  now, so both platforms behave identically.
  Measured on desktop AND with `?phone=1`, identical both ways: double-tapping an
  own chip on a goal banks it (saved 4 -> 5, stack 3 -> 2, picker closed); a single
  tap still selects that piece, with the picker open at 80ms and closed by 530ms
  and nothing banked; double-tapping an enemy chip on a block peels it (black saved
  3 -> 4, block 2 -> 1, both dice spent).
  **AND THE SAME SLIP ON THE BOARD, FIXED FOR BLANKS (owner's suggestion,
  2026-09-28).** Double-tapping pieces **on the board** in a stack of 3+ often did
  nothing: `lastClickTime` is a PER-PIECE field, and a fingertip is wider than an
  18px piece -- measured, a second tap only 6px from the first lands on a
  NEIGHBOURING piece, so neither ever saw two clicks. Intermittent, because whether
  the slip crosses into another piece depends on the stack's geometry.
  **Owner's fix, which is better than moving the window to the tile: BLANKS ON ONE
  TILE ARE INTERCHANGEABLE, so use that.** Two taps on two blanks of the same tile
  and player are two taps on one target -- the game's own rule, the same one that
  makes `get_valid_moves` dedupe them and the replay fingerprint anonymise them --
  and acting on either gives an identical position. `Piece.handleClick` now pairs a
  tap with a recent one on a sibling blank.
  **Deliberately NOT widened to the whole tile:** a NUMBERED piece is
  interchangeable with nothing (each has its own goal), so numbered-then-blank must
  stay two separate taps. Measured both ways at `?phone=1`: an all-blank stack with
  a 6px slip gives `tap10, tap12 -> DBL12` and BANKS the piece, where it used to do
  nothing; a mixed stack gives `tap6, tap11` and correctly does nothing.
  **Mixed stacks therefore still have the original problem, by design** -- owner:
  *"that's what zoom is for so not a big deal"* -- and the picker is the reliable
  route regardless.
  `DBL_TAP_MS` (300) is now ONE constant for every double-tap in the game, the
  chips and the board pieces alike.
  **HARNESS LIMIT, and it produced a convincing wrong answer first:** patchright's
  touch emulation imposes a floor of about **360ms between taps** (measured: 40ms
  and 120ms requested both gave ~360ms), which is ABOVE the game's 300ms
  double-tap window -- so **no double-tap can ever be tested by touch in this
  harness**, and an early "reproduced: phone does nothing" was the instrument, not
  the game. Use **`?phone=1` with MOUSE input** to exercise phone logic without the
  touch latency. Also: the picker element has NO id, and `_stackPicker` is a
  main-world variable, so a probe must reach it through `addScriptTag`, not
  `page.evaluate`.

- **GAME RECORDER FOR OWNER'S OWN GAMES (owner, 2026-09-27). Local-only, opt-in,
  nothing transmitted.** Purposes, in owner's words: (a) his real stats against
  the model, (b) learning from positions where his move and the model's differ,
  (c) data for future training.
  **THE PRIVACY ARGUMENT IS STRUCTURAL, NOT A PROMISE.** There is no endpoint:
  games go to this browser's localStorage and, on desktop, to a file the player
  chose. So there is no other data subject, `privacy.html`'s "collects no data"
  stays literally true, and other players need no notice because nothing of theirs
  is touched. Owner asked specifically for the version with no privacy surface and
  no need to tell anyone; this is that version, and the property is provable from
  the code rather than intended.
  **Unlocked once by `?rec=rec-quahuru-7f3a`**, which writes `recEnabled`;
  dormant in every other browser. **THE TOKEN DOES NOT NEED TO BE SECRET and it
  does not matter that the repo is public:** because nothing is transmitted, the
  worst a reader can do is enable recording of their OWN games in their OWN
  browser. It only has to be non-obvious enough not to be tripped by accident.
  Rejected: automatic upload (needs a server, storage, a policy change, and puts
  an endpoint in a public bundle), always-on local recording for everyone (writes
  to other people's devices unasked), and a separate private build (owner would
  stop playing the build his testers play, and it would mean a fourth origin).
  **WHAT IS RECORDED: the INPUTS, not the positions.** The engine is
  deterministic, so a game is `{starter, both rack orders, per turn [d0,d1] +
  the moves applied}` and everything else is recomputable -- which is also what
  lets the disagreement analysis be re-run against ANY future net rather than
  against whatever the model was on the day. Per game also: uuid, device id,
  timestamp, `REC_MODEL_TAG`, **effective difficulty AND slider position** (two
  different quantities since the remap), both `isAI` flags, match context, the
  turn indices where a HINT was consulted, result and a completed flag.
  **Hints are recorded because otherwise (a) is contaminated** -- a game where the
  pill was consulted is not a measure of unaided play, and `merge_games.py` reports
  those separately rather than folding them in. **Undo is deliberately NOT
  recorded** (owner: you cannot undo after finishing your turn, so intra-turn undo
  is composing the turn, not a takeback).
  **`REC_MODEL_TAG` must be updated whenever `model.onnx` is re-exported** --
  nothing else in the shipped bundle carries a version, and a stats table that
  silently mixes two champions is worse than no table.
  **FOUR MUTATION SITES, and the fourth was nearly missed:** `movePiece`'s commit
  point, `Piece.save`, `handleDoubleClick`'s block-save (the human gesture) and
  **`applyMovePair`'s own block-save branch** -- the computer does not go through
  the human gesture, so its block-saves would have gone unrecorded. Found by
  checking which move types a replayed game actually exercised, not by reading.
  (`Board.saveOpponentPieces` is DEAD CODE and deliberately not hooked.)
  **`replay_games.py` IS THE PROOF, and it does two jobs.** It replays a recorded
  game through `game.py` and compares the per-turn fingerprint, which (1) proves
  the recording is complete -- an unhooked path shows up as a divergence naming
  the game and turn -- and (2) **conformance-tests game.js against game.py over
  real play**, which had never been done: the port was verified 110/110 on chosen
  agent PAIRS, never on move application across a whole arbitrary game.
  A recorded move is resolved against `get_valid_moves()` rather than rebuilt as a
  tuple, which derives the die the log omits AND asserts legality.
  **THE DIE IS NOT RECORDED** because `movePiece` and `save` both pick it
  themselves from the position and the destination. Measured: every recorded move
  resolved to exactly ONE legal game.py move, so the format is unambiguous.
  **SIZE, measured, and it was 3x my estimate.** The first format was **5.9 KB a
  game** (2.0 MB/month at a dozen games a day, which fills localStorage in ~2.5
  months). Moves are now short strings -- `"7>5.4"`, `"7>s"`, `"o7>b"` for a
  block-save -- with the colour omitted because the turn records its mover, and
  the hash in base36: **2.9 KB a game, ~1.0 MB/month**. The Settings row shows
  count and size and turns amber past 2 MB. Export-and-clear is the intended
  rhythm; IndexedDB is the upgrade if that ever chafes.
  **VERIFIED: 3 of 3 games, 143 turns, every fingerprint matching.** Note that an
  earlier "1 of 1 clean" was LUCK -- the next run was 0 of 3. One game is not a
  sample for this.
  **FOUR BUGS, ALL MINE, EACH OF WHICH WOULD HAVE MADE THE LOG LOOK UNTRUSTWORTHY
  MONTHS LATER:**
  1. **`Math.imul` is load-bearing.** FNV-1a multiplies by `0x01000193`, and a
     plain `h * prime` in JS exceeds 2^53 so floating point drops the low bits.
     The result still looks like a fine 32-bit hash; it simply is not FNV-1a and
     matches no other implementation. Confirmed rather than assumed: the lossy
     multiply reproduces the recorded `1onk8rc` exactly, `Math.imul` gives
     `12k6tp5`, and Python's exact integers give `12k6tp5`. **Without the Python
     checker the fingerprints would have been self-consistent forever** -- JS
     always comparing against JS -- and this would have surfaced only on the first
     attempt to replay a year of games, looking like broken recording.
  2. **Blanks are anonymous in the fingerprint (`w*`).** The GAME treats
     same-colour blanks as interchangeable and `get_valid_moves` dedups them on a
     shared tile, so game.py legally moved blank 10 where game.js recorded blank 9
     -- an identical position that a fingerprint naming the pieces called
     different. Numbered pieces keep their number: each has its own goal, so they
     are not interchangeable.
  3. **The dice came OUT of the fingerprint.** Including each die's `used` flag
     would have flagged DIE SELECTION as a position divergence: both engines pick
     "whichever unused die reaches the target" but need not agree when either
     would do, so `"3,4u"` vs `"3u,4"` for the same board. The values are recorded
     per turn anyway. What it hashes is the position: every piece's tile, the two
     saved counts, whose turn it was.
  4. **NEVER SUBSTITUTE AN INTERCHANGEABLE BLANK -- replay the EXACT piece.** The
     obvious response to bug 2 was to let the checker accept an equivalent blank
     when game.py does not offer the named one. That is a trap: substituting once
     makes the piece-number mapping drift PERMANENTLY, so every later move naming
     that blank is "illegal" and it reads like an engine divergence. Measured: 0 of
     3 games replayed under substitution, 3 of 3 once identity is preserved. The
     checker now asks the PIECE for its own reachable set (`get_reachable_tiles_by_dice`,
     which is pre-dedup) and derives the roll from that, calling `get_valid_moves()`
     only for its side effect of refreshing `game_stages`.
  **NO ENGINE DIVERGENCE FOUND.** Once the fingerprint was defined correctly and
  identity preserved, game.js and game.py agreed on the position at every one of
  143 turns -- the first time the two have been compared on move application over
  whole games rather than on chosen agent pairs.
  **TWO GAPS THAT ONLY A REAL HUMAN GAME COULD EXPOSE (owner's first recorded game,
  2026-09-28), both now fixed:**
  1. **A SUM MOVE WAS UNREPLAYABLE.** `game.py` has no single "sum move" -- the
     agent always plays two half-moves -- so a human's one-gesture move on the dice
     sum was logged as one entry no single die can reach, AND the intermediate tile,
     which decides an en-route capture, was not recorded at all. Now recorded as its
     two halves, using the intermediate `checkEnRouteCapture` actually took (it
     returns it). Self-play cannot produce one: the computer never makes a
     single-gesture sum move.
  2. **AN UNDO LEFT A PHANTOM MOVE.** Owner's game had a turn with THREE half-moves
     against two dice -- made, undone, remade elsewhere. Undo TRACKING was dropped
     on purpose (you cannot undo after ending a turn, so it does not contaminate
     the stats), but the recorder still has to REMOVE an undone move from the turn.
     `_recUndo` is keyed on the undo stack's DEPTH, not a count, because one undo
     can revert a move that produced two records. The computer never undoes.
  **`replay_games.py` tolerates the OLD one-entry sum format** so already-collected
  games are not wasted, and it mirrors the game's en-route rule when doing so:
  exactly one capturable intermediate is TAKEN (the game auto-captures), none means
  any route ends in the same position, two or more means the destination was
  withheld and cannot have been one gesture. **A first version SKIPPED capturing
  intermediates and so silently replayed a DIFFERENT position instead of failing** --
  the worst thing a checker can do.
  **Owner's first game replays turns 1-18 and stops at 19 on the undo phantom,**
  which is unrecoverable from the old format (nothing says which entry was
  reverted). It is still fully usable for the STATS -- result, margin, difficulty
  and model tag are all sound -- and only the disagreement analysis needs the
  replay. Regression after both fixes: 3/3 self-play games still clean.
  **THE LAST-PIECE RULE IS APPLIED AT DIFFERENT MOMENTS (found 2026-10-03, 17 of
  83 of owner's games):** game.py blanks a player's last piece at EVERY turn switch,
  game.js only at the start of that player's OWN turn, so during the opponent's
  turn the label differs. Equivalent for play; the checker now mirrors game.js
  (`shown_numbers`). After it: 82/83 replay, the one left being the undo phantom.
  **BLOCK-SAVES ARE NOT PRODUCED BY SELF-PLAY** (0 in 143 turns, against 220 tile
  moves, 60 saves and 1 pass), so that path was verified on its own: the tutorial's
  "Buy the door open" position, `handleDoubleClick` on the blocked black piece,
  recorded as **`"o11>b"`** with black's saved rack 3 -> 4 and both dice spent. The
  computer's block-save is the one-line twin at the `applyMovePair` site and is
  verified by reading only.
  **NOT VERIFIED AND CANNOT BE HEADLESSLY: the desktop auto-append.**
  `showSaveFilePicker` needs a real user gesture and shows OS UI, so the File
  System Access path (handle cached in IndexedDB, permission re-confirmed on the
  first pointerdown of a session, `createWritable({keepExistingData:true})` +
  seek-to-end to APPEND rather than truncate) is verified by reading only. **Check
  it on the real browser once**: Settings > Game log > "Log to file...".
  **Two devices mean two files** (localStorage is per browser) and that is fine:
  every game carries a uuid, so `merge_games.py` dedupes exactly and re-exporting
  without clearing cannot double-count. It reports the record split by effective
  difficulty with hint-assisted games counted separately.
  **HOW LONG BEFORE (a) MEANS ANYTHING:** if owner's true rate is 55%, the 95% CI
  still spans 50% at 200 games and separates only at **~400 games, about a month
  at a dozen a day**. Watch the average margin sooner -- it is lower variance than
  win/loss (measured; see the arena notes).
  **AND THE HONEST LIMIT ON (b):** outcomes cannot arbitrate a single position.
  ~28 decisions a game share one win/loss bit, and winning correlates with playing
  well overall rather than with that move. What can arbitrate: the existing
  DEEPER SEARCH (expectiminimax over all 21 rolls) run on each disagreement, and
  the model's own value gap between his move and its preferred one. The interesting
  set is where the model says he gave up half a piece and the deeper search says he
  was right -- those are positions where the MODEL is wrong, which is the training
  signal. Outcomes remain useful only as a population-level check.

- **THE DIFFICULTY SLIDER IS REMAPPED ONTO 0.8..1.0, BECAUSE THE BOTTOM OF ITS
  RANGE WAS NOT A DIFFICULTY SETTING AT ALL (measured 2026-09-25,
  `difficulty_arena.py`, 566 games).** The deployed champion played BOTH sides,
  one at difficulty d and the other always at 1.0, over paired colour-swapped
  seeds, plain 2-ply shallow. `d=1.0` was in the sweep as the control.

        d   slider  games  win%   (95% CI)   avg margin  shutouts  turns
       1.00   Max      96  50.0  (40-60)       +0.00        0%      56
       0.80   80%      94   0.0  ( 0-3.9)      -6.06        0%      61
       0.60   60%      94   0.0  ( 0-3.9)     -10.06       15%      56
       0.40   40%      94   0.0  ( 0-3.9)     -11.27       53%      53
       0.20   20%      94   0.0  ( 0-3.9)     -11.68       72%      51
       0.00  Easy      94   0.0  ( 0-3.9)     -11.89       90%      49

  **The control is clean, so the rest is trustworthy:** d=1.0 against itself gives
  exactly 50.0% (48/96), the colour-swapped halves are 26/48 and 22/48 at margin
  +0.21 / -0.21, and white takes 52 of 96. The harness is symmetric.
  **Even 80% -- the mildest weakening the slider offered -- never won a game** (0
  of 94; its BEST result in 94 was losing by 2), and at 40% and below it is SHUT
  OUT (beaten by the maximum margin of 12) in 53%, 72% and 90% of games. Every
  setting from 0.0 to 0.8 has the same 0% win rate, so **four fifths of the
  control's travel was undifferentiated and the "50%" on its label meant nothing.**
  Cause: at d=0.8 the ramps give temp 0.92 and top_p 0.40, and because the softmax
  in `_pick_move_index` is z-scored by the CANDIDATE SPREAD, a temperature near one
  std flattens it relative to the scores -- so the agent deviates from its best
  move on a large fraction of turns, which over ~56 turns is dozens of blunders.
  The ramps (`0.4 + (1-d)*2.6`, `0.25 + (1-d)*0.75`, mirrored in `agent.js`'s
  `pickMoveIndex`) are scaled wrong for this game's length.
  **THE FIX, and the two quantities it introduces.** `DIFFICULTY_FLOOR = 0.8`, and
  `getAIDifficulty()` now returns `0.8 + 0.2 * position`. Do not confuse them:
    * `getDifficultySetting()` -- where the SLIDER sits, 0..1, and what is
      persisted in `localStorage.aiDifficulty`;
    * `getAIDifficulty()` -- the EFFECTIVE number handed to the agent.
  **A saved setting is REINTERPRETED, not migrated.** `aiDifficulty` used to hold
  the effective value, so a player who had saved 0.5 is now read as position 0.5 ->
  effective 0.9 and their opponent gets stronger. Deliberate: every saved value
  below 0.8 was a setting that could not play the game, so there is no old position
  worth preserving. The default is unchanged -- no stored value means position 1.0
  means full strength.
  **THE LABEL IS ORDINAL WORDS AND DELIBERATELY CARRIES NO PERCENTAGE**
  (`difficultyLabel`): Max / Strong / Medium / Gentle / Easiest. A figure on that
  label reads as a win rate, and **nobody has measured inside 0.8..1.0** -- only
  the endpoints. The ORDERING is safe to claim, because both ramps move
  monotonically with the value; a number is not.
  **STILL TO DO: the fine sweep of the band** (0.99 / 0.97 / 0.95 / 0.92 / 0.90 /
  0.85), which is where the transition from 50% to 0% happens. Deferred at owner's
  request -- it ties up his machine. Those games are cheaper than the ones already
  run (178-267s against 300-345s at the bottom of the range), so ~2.5 h per pass of
  ten games per level. Put numbers on the label once it exists.
  **Cost of the run already done:** ~216s per game of throughput on 3 workers of
  this 4-core iMac, so ~3.6 h per pass of ten games per level across six levels.
  `difficulty_arena.jsonl` holds every game and the script skips what is already
  recorded, so a longer run resumes -- but **that file is GITIGNORED** (`*.jsonl`,
  the same convention as `arena.jsonl`), so the raw games are local to this iMac
  and the table above is the record.
  **A CLAIM MADE FROM WALL CLOCK AND DISPROVED BY THE TURNS COLUMN:** the first six
  games suggested weakened agents "play 2.3-2.5x longer games", inferred from
  seconds. Turns say otherwise -- 56 / 61 / 56 / 53 / 51 / 49, flat and if anything
  DECREASING. The inflation is **seconds per turn** (3.2 -> 6.6): the losing side
  banks nothing, so it keeps ~12 pieces on the board all game, its move generation
  is maximal every turn and the 2-ply search costs twice as much. Seconds are not a
  proxy for game length when the branching factor is what changed.
  **CAVEAT ON WHAT THIS MEASURES:** strength relative to the net at full strength,
  NOT relative to a human. Owner BEATS d=1.0 ~50-60%, so full strength is roughly
  an expert's equal -- which is why it crushes beginners, the complaint that
  started this. Where d=0.8 sits for a beginner is still unmeasured and needs a
  human, not an arena.

- **LEARNABILITY WORK, ON `testing` FOR TRIAL (owner, 2026-09-24).** Two testers
  reported the game still hard to learn after the tutorial, one asking for a hint
  mode. Four changes, all on `testing` and NOT yet on main, at owner's request.
  **1. A HINT PILL ("💡 Hint", bottom right beside the "?" legend).** Tapping it asks the SAME
  on-device agent that plays the computer's side what it would do in your
  position, and rings the piece and the destination tile. It is presentation, not
  new search -- `local_agent.js` has answered `selectMoves()` since the port, so a
  hint costs one inference batch, the same as one of the computer's turns.
  **Always at full strength**, whatever the difficulty slider says: a top-p
  sampled hint would sometimes recommend a move the agent itself ranks worse.
  **ONE MOVE AT A TIME, and that is not laziness** -- the agent picks a PAIR whose
  second half is chosen against the board as it stands AFTER the first, so that
  destination is frequently not a destination yet and marking both would ring a
  tile the player cannot legally tap. Default ON (`hintsEnabled`), with a Settings
  row; a lamp you never tap costs nothing, so there is no first-game-only state
  machine.
  **Two correctness traps, both real:**
  `getGameState` posts `reachableBySum` on a piece, and `engineState` reads that
  as the marker for "this piece already moved this turn" (the engine's
  `first_move`) -- but the frontend sets it on the SELECTED piece too. Harmless
  while only the computer's own turn asked (nothing is selected then), WRONG for a
  hint asked mid-turn on a human's. `_hintGameState` strips it and passes
  `gs.firstMove` explicitly, carrying the ORIGIN tile, which the marker cannot:
  it reports the piece's CURRENT tile. (`engineState` gained 13 lines for this and
  the old marker path is untouched, so the computer's own turns are unchanged.)
  And **the engine used to treat the rack-entry obligation as met by ANY first
  move** -- FIXED 2026-09-30, see "THE ENGINES DISAGREED ABOUT THE RACK ENTRY" at
  the top of Current state. `_renderHint` still asks the LIVE game whether the half
  it is about to mark is playable (`_hintMoveIsLegalNow`) and **refuses rather than
  falling back to marking it anyway**: a hint pointing at an illegal move is worse
  than none. That refusal is now a backstop, not an expected path.
  The marker is cleared by a 250ms poll against a signature (turn + dice + every
  piece's tile) rather than by hooks in movePiece / undo / switchTurn / the picker
  -- one place to be right instead of six to remember.
  **2. "why can't I move there?" NOW ANSWERS ITSELF.** The shortest-route rule is
  the one rule that contradicts what a player can see: a piece always travels its
  shortest route, so a longer path they have traced by hand is not a move and the
  tile just refuses the tap. `_noticeWhyUnreachable` (called from the ordinary
  refusal in `_noticeIfRouteWithheld`) **names the number** -- "That tile is 7
  steps away by the shortest route, so it takes a 7" -- which is what turns an
  arbitrary-feeling refusal into a rule. It also covers a wall on the tile, no
  route at all, and the mid-turn no-doubling-back half of the same rule.
  **Deliberately narrow:** it stays SILENT when the dice CAN make the distance,
  because then some other rule refused it and this function has nothing true to
  say -- a confident wrong explanation teaches a rule that does not exist.
  **AND IT ONLY EXPLAINS WHEN YOU ASK TWICE (owner, 2026-09-25: "too trigger
  happy, they often fire when I've just mistapped by one tile").** The first
  suppression was a bare 1200ms timer with no memory of WHICH tile, so every
  isolated refused tap got a full explanation -- and a mistap by one tile is
  GUARANTEED to hit an unmakeable distance: if the dice make 3, 5 and 8, the
  neighbours of the intended tile sit at 2, 4 and 6. The message was correct every
  time and answering a question nobody asked; exploratory tapping did the same.
  Now a first refusal on a tile is SILENT and a second on the SAME tile speaks
  (`game._whyLast` = {tile, at}, plus `_whyShownAt` for the post-message cooldown).
  A slip is corrected and never repeated; a misunderstanding taps the same tile
  again because the player still thinks it should work. **Silence is not absence of
  feedback** -- a refused move re-asserts the lit destinations in `Tile.onClick`'s
  else branch, which is the answer without prose.
  **"ASKING TWICE" MEANS THE SAME QUESTION ABOUT THE SAME BOARD (owner's
  correction, and it is the right rule).** The first cut used an 8-second window as
  a PROXY for "the board has moved on"; the condition is now tested directly and
  there is **no maximum gap at all** -- if the board is identical the player has not
  moved, so they are still asking the same thing however long they took. Anything
  that changes breaks it and the tap counts as a fresh first one.
  `_whySig(game, piece)` is `_hintSig(game)` (turn + both dice, value AND used +
  every piece's tile) plus the SELECTED PIECE, because the distance in the message
  is measured from it -- the same tile asked about by a different piece is a
  different question with a different answer.
  `WHY_REPEAT_MIN_MS` 300 stays: a physical double-tap has an identical board state
  and is one gesture, not asking twice.
  **The WALL and NO-ROUTE messages still speak on the FIRST tap** -- they are about
  board state the player may genuinely not have seen rather than a rule they know,
  and they are rare, since a wall has to be on the exact tile tapped. Only the two
  ARITHMETIC messages (distance, no-doubling-back) go through `sayOnRepeat`.
  Measured: 1st tap silent; 2nd on the same tile with the board unchanged speaks;
  **the same pair 9 SECONDS apart still speaks** (no max window); a first tap then a
  DIFFERENT tile stays silent; **spending a die between the two taps makes the
  second a first tap** (sig changed, silent); **the same tile asked by a different
  selected piece is a first tap** (silent); a same-tick double-tap stays silent; a
  wall 4 away with a die of 4 speaks on the first tap; and on a clean board a legal
  sum-7 move still lands and spends both dice.
  **Fixture trap, hit a third time:** `_clearSelection(game)` did NOT return the
  tentatively entered piece to its rack in the harness, so selecting the next rack
  piece put a SECOND own piece on home, which blanks `reachableBySum` and made the
  legal-move control fail for reasons unrelated to the change. Give that control its
  own fresh page and assert `ownOnHomeBefore === 0` and `sumOffered === true` so the
  fixture proves itself.
  **Rejected alternatives** (asked for and considered): suppressing when the
  distance is within 1 of a makeable value silences a large slice of genuine cases
  and guesses at intent from arithmetic; suppressing when the tile is adjacent to a
  legal destination silences almost everything, since with two dice plus the sum
  most tiles are; snapping a near-miss onto the adjacent legal destination would
  move a piece the player did not tap. A dosage cap (N per game) was offered and
  owner chose second-tap alone for now.
  **3. THE TUTORIAL IS REACHABLE FROM HOW TO PLAY**, leading the panel above the
  first section. It was previously only on the welcome card (gone the moment you
  start playing) and in Settings (where nobody looks for a tutorial). It
  **confirms first when a game is in progress** -- the tutorial scripts positions
  into the live game and ends by restarting the scene to the welcome card, so
  launching it over a game throws that game away. "In progress" is asked of the
  board (anything entered or banked), because the frontend `Game` has no move
  history -- that lives on the ported engine.
  **4. THE CLOSING PANEL ASKS THE DIFFICULTY QUESTION AS TWO BUTTONS**, "Go easy"
  (sets `TUT_EASY_DIFFICULTY` = 0.5) and "Full strength", replacing the sentence
  that pointed at the slider. The sentence was the wrong instrument for the reason
  the 2026-09-13 entry gives: the phone card is capped and `#tutText` scrolls, so
  a sentence at the END of the text is the first thing below the fold. The buttons
  are pinned to the bottom of the card and cannot scroll away. Closing text is now
  190 characters, the SHORTEST of the eleven steps (step 7, at 450, is the tallest
  and still sets the pinned height), so nothing about the other steps moved.
  **The constant is `TUT_EASY_POSITION = 0.0`, a SLIDER POSITION** (= effective
  0.8 after the remap). It was `TUT_EASY_DIFFICULTY = 0.5`, a guess, then briefly
  0.8, and both were wrong the same way -- written as effective difficulties, so
  after the remap a value of 0.8 would have meant effective 0.96, nearly full
  strength and the opposite of the button's promise. See the difficulty-slider
  entry above for the measurement.
  **Measured in a browser** (5 viewports: desktop, phone portrait, phone
  portrait + `?safeinset=48,0,56,0`, phone landscape, phone landscape +
  `?safeinset=0,48,24,48`), against main's game.js served side by side as a
  baseline:

        every goal is 7 from home, so a front rack piece tapped at a goal is
        a distance-7 move -- which makes the rule testable with no fixture:
          dice 2+3 (sum 5): sum not offered, no move, 0 dice spent,
                            "That tile is 7 steps away ... so it takes a 7"
          dice 3+4 (sum 7): sum offered, LANDED ON GOAL 3, 2 dice spent,
                            no notice            <- the control that matters
          wall (2 black) 4 away with a die of 4: wall message, not distance
        hint: 2 markers at depth 76, one on the piece and one on the tile
              centre, markedMoveIsLegalNow TRUE, cleared 2 -> 0 when the board
              moved, "Hints are for your own turn." on the computer's turn
        lamp inside the viewport and clear of the legend at all 5, and it
              honours the insets (portrait bottom 903 -> 847, landscape right
              865 -> 817)
        tutorial card box BYTE-IDENTICAL to main on every step at every
              viewport except the closing one in landscape, which SHRANK
              314 -> 292; closing-step text overflow is LOWER than main's
              (portrait 56 vs 78; with insets 85 vs 107)
        0 console errors, 0 failed requests, 94 tiles, 24 pieces, WebGL

  **AMENDED AFTER OWNER'S TRIAL (2026-09-25). Three changes, all verified:**
  **(a) ONE PIECE MOVED TWICE IS NOW ONE HINT, ON THE DICE SUM.** The agent
  returns it as two halves because that is how it searched, but the player makes
  it in a single gesture, so hinting the intermediate tile and only revealing the
  real destination on a second tap was teaching the hard way. `_hintSumMove`
  collapses it: both halves the same piece, both ordinary tile moves, and the
  SECOND half's destination (pair order is application order -- select_move_pair
  chooses the second against the board after the first) actually in
  `reachableBySum`.
  **THE EXCEPTION IS THE GAME'S OWN, NOT A RE-IMPLEMENTATION:** a sum destination
  whose routes offer a choice of captures is withheld in `ambiguousSum` precisely
  so the player spends the dice one at a time to pick the capture, and there the
  two-step hint is correct -- so it falls through. A save half, a block-save half,
  two different pieces and a one-move pair all decline to collapse too.
  Measured, and with the PAIRED CONTROL that proves the gate is doing the work
  rather than the position: the same stubbed reader collapses when the tile is in
  `reachableBySum` (roll 7) and does NOT when the only difference is that the tile
  sits in `ambiguousSum`. On a real opening position the front rack piece offers
  6 sum destinations (the six goals, all exactly 7 from home), and the render puts
  a ring on the piece and on the FINAL tile, none on the intermediate, with
  "...to goal 4 — both dice on the one piece."
  **(b) THE LAMP IS A LABELLED PILL IN THE ACCENT COLOUR** -- owner: *"the hint
  lamp is tiny."* It was a 30px translucent dot matching the legend "?", which is
  wrong by intent: the legend is a reference you consult once and is deliberately
  faint, while this is an action to reach for mid-game. Now 77x34 (2606px2 against
  the legend's 900), accent ground, white glyph plus the word "Hint", which also
  removes the guesswork a bare emoji leaves. Measured inside the viewport with a
  10px gap to the legend at desktop, phone portrait, portrait + `?safeinset=
  48,0,56,0`, landscape + `?safeinset=0,48,24,48`, and a 320px-wide phone.
  **Watch-out:** the "thinking" state must rewrite only the GLYPH span --
  `btn.textContent = '…'` flattens the pill's two spans and the label never
  returns.
  **(c) THE TUTORIAL HANDS OVER WITH HINTS ON AND SAYS SO.** Both closing buttons
  go through `_tutFinish(position)`, which sets the difficulty, WRITES
  `hintsEnabled = '1'` (rather than relying on the default, so a player who had
  turned hints off still gets them back for this one game, and the notice is never
  a lie) and arms `_hintNudgePending`.
  **The notice cannot fire where the button is pressed:** `_tutEnd` restarts the
  scene to the WELCOME CARD, so a notice shown then would sit under it and be gone
  before the first game began. It is delivered from `_hintTick`, the 250ms poll the
  marker already uses, once a real unfrozen game exists with no card over it --
  one waiting place instead of hooking the four paths a game can start from.
  Measured from a start with hints explicitly OFF: "Go easy" gives position 0 /
  effective 0.8 / hints on / nudge armed; **no notice under the welcome card**, and
  the notice appears with the pill visible once the game starts.
  **It is a NOTICE and not another line in the card** for the same reason the
  difficulty sentence became buttons: the card is capped with `#tutText` scrolling
  inside it, so a sentence about hints would be the next thing below the fold.

  **(d) THE MARKER DIES THE MOMENT A MOVE COMMITS (owner, 2026-09-25: it
  "persists a bit too long").** `_hintTick` would have caught it within 250ms, but
  that is long enough to watch the ring linger through the slide animation. There
  is now a synchronous `clearHint()` at the commit point of `movePiece` and of
  `Piece.save` -- **after every legality check, so a REFUSED move leaves the hint
  and the selection exactly as they were.** The 250ms poll STAYS as the backstop
  for everything that changes the board without coming through those two (undo, a
  turn switch, a scene restart).
  Measured in the same tick, giving the poll no chance to run: a committed move
  2 markers -> 0 (move happened, 2 dice spent); a committed save 2 -> 0 (piece off
  the board, saved count up); a refused move 2 -> 2, no move, no die spent.
  **Two fixture traps in verifying just this:** a save is ILLEGAL IN THE OPENING,
  so the first attempt tested nothing -- `save()` returned false and the marker
  "failed" to clear because no save had happened. Borrow the tutorial's own
  `Saving` step position (verified by construction) and apply it to a normal game
  with `_tutApply` / `_tutPhases` / `_tutRefresh`. And a probe that reuses the page
  after a refused-move case finds the rack piece still TENTATIVELY ENTERED on home,
  which gives the mover two pieces there, blanks `reachableBySum` and silently
  turns the "committed move" case into another refused one.

  **(e) HINTS ARE OFF BY DEFAULT, WITH TWO EXCEPTIONS (owner, 2026-09-25).** A
  prominent pill is clutter for a player who does not want it. `getHintsEnabled()`
  now defaults FALSE; what turns it on is finishing the tutorial (`_tutFinish`) or
  a FIRST-EVER VISIT, seeded once by `_seedFirstRunDefaults`.
  **HOW GOOD IS "first-ever visit"? Honest answer: good on desktop, imprecise on a
  phone.** Every localStorage key this app writes is written only when the player
  changes something, so "no key at all" means EITHER a first visit OR a returning
  player who has never touched a setting. `_ALL_SETTING_KEYS` lists all fourteen
  and must stay complete -- a key missing from it makes a returning player look
  new. On desktop `seenNudge` is written on the very first load, so a desktop
  returner is always identified correctly; **on a phone the nudge is skipped
  entirely**, so a phone player who has never changed a setting is misread as new
  and offered hints once. Accepted deliberately: the cost is one dismissible pill,
  and there is no unconditional visit marker to key off without inventing one that
  would only help from now on anyway.
  Measured: a browser with nothing stored seeds `hintsEnabled=1` and shows the
  pill; one with other keys but no `hintsEnabled` gets OFF and no pill; an explicit
  '1' and an explicit '0' are both respected; and the tutorial still turns them on
  from an explicit OFF.
  **(f) THE PILL SAT ON TOP OF "How to Play" ON A PORTRAIT PHONE (owner,
  2026-09-25).** The three HUD buttons are world furniture and in portrait they run
  the whole width of the band below the racks -- measured on a 412px phone, the row
  occupies y 845..880 and x 14..398, and How to Play is the RIGHTMOST, exactly
  under a pill pinned to the bottom-right corner. There is a **133px free band**
  between the rack bottom (712) and the row top (845), so `_placeHintButton` raises
  the pill into it at `calc(78px + var(--safe-b))`, right-aligned.
  **Portrait only, and only on a phone:** in landscape and on desktop the HUD row
  is at world x=150, on the far LEFT, so the corner is free and the pill stays
  beside the legend where it is easiest to reach. Driven from JS rather than a media
  query so it honours `?phone=` and `?portrait=`, and re-run from
  `_relayoutFurniture`, which is the handler rotation and resize already go through.
  Measured clear of the row at all four: portrait 803..837 (8px gap), portrait with
  insets 747..781 (25px), landscape and desktop unchanged in the corner.
  **Pre-existing and NOT fixed:** the legend "?" also overlaps How to Play's right
  edge by a few px in portrait. Owner did not report it and it is not new.
  **(g) PRIVACY POLICY AND LICENCES LINKED UNDER SETTINGS (tester, 2026-09-25).**
  Both were already reachable from How to Play > Credits; this is a second door,
  because Settings is where people look for a policy and both stores expect it to
  be easy to find. `target="_blank"`, since leaving the page would drop the game.
  **Trap, and the test caught it:** `createSettingsPanel` has its OWN local `mk()`
  which sets **textContent, not innerHTML** (unlike the one in
  `createLegendButton`), so the first cut rendered the anchor tags as literal text
  and the panel contained zero links. A probe that counted anchors found 0 and said
  so; one that had checked for the word "Privacy" would have passed.

  **(h) THE ROUTE EXPLANATION VANISHES ON A MOVE TOO (owner, 2026-09-25).** "That
  tile is 7 steps away" answers a move that did NOT happen, so once one does it is
  describing a board that has gone. `flashNotice` gained an optional third
  argument, a **tag**, and `_clearMoveNotice()` dismisses a notice only when it is
  tagged `'move'` -- called from the same two commit points as `clearHint`.
  **The tag is what makes it safe:** an UNTAGGED notice keeps its full dwell, which
  matters for "Getting the computer ready — retrying", "White passed" and the
  graphics warnings, none of which a move makes stale. Tagged: the four
  `_noticeWhyUnreachable` messages, `_noticeIfRouteWithheld`'s capture-choice
  notice, and the three hint messages.
  Measured with the control: a refused tap shows the notice with `tag='move'`, a
  committed move takes it from opacity 1 to 0 in the same tick (moved onto the
  goal, 2 dice spent), and an untagged notice survives the identical move.
  **Fixture trap:** a refused move leaves the piece SELECTED, so re-clicking it to
  set up the legal move DESELECTS it and returns the tentative entry to the rack --
  the first version of the test moved nothing and the clearing looked broken. Keep
  the selection and recompute `piece.reachableTiles` after changing the dice.
  **(i) RING 3 FORWARDS THE TAP TO THE TILE TOO (owner, 2026-09-25: "they're also
  pretty small").** `TILE_ROOM_IN_PIECE_WIDTHS` 2.2 -> **2.8**. Re-measured on a
  phone at `tilePieceRadius(1)`, arc / piece-diameter per geometry:

        field ring1  1.26 (arc  63)      field ring3  2.51 (arc 126)  <- was "room"
        field ring7  1.68 (arc  84)      field ring4  3.14 (arc 157)
        field ring2  1.88 (arc  94)      goal  ring7  3.60 (arc 259, bigger pieces)
        field ring5  1.88 (arc  94)      field ring5  3.77 (arc 188)
                                         field ring6  4.40 (arc 220)

  2.8 is the only sensible value: it excludes ring 3 at 2.51 and keeps ring 4 at
  3.14, with 0.29 of margin below and 0.34 above. **Still expressed in PIECE WIDTHS
  and not per ring** -- ring 5 is not uniform (6 tiles at 1.88, 6 at 3.77) so no
  ring rule can express it, and the goal pieces are larger (72px against 50).
  **New census, measured by asking `_tileHasRoomBeside` about all 70 tiles** (not
  by re-deriving the arithmetic): FORWARDS = ring1 x9, ring2 x9, **ring3 x12**,
  ring5 x6, field ring7 x9 = **45**; PASSES THE SELECTION = ring4 x6, ring5 x6,
  ring6 x6, goal x6, home = **25**. Goals keep the pass behaviour, as intended.

  **Three fixture traps hit while measuring, each of which produced a convincing
  false pass** -- all three are the "read the denominator" rule again:
  selecting a rack piece **TENTATIVELY ENTERS it onto the home tile**, and a
  stranded one there makes two own pieces on home, which blanks `reachableBySum`
  -- so the "it should move" control silently could not move, and its `notice:
  null` looked like the code correctly staying quiet. Every case now gets a fresh
  page. `piece.move(tile)` **does not remove the piece from its rack**, so a
  hand-placed piece left the board empty and the "game in progress" confirm test
  passed against a fresh game; play into the position with the real handlers
  instead. And a gate test must pull the lever the gate actually reads:
  `currentPlayerIsHuman()` reads the **Player's `isAI`**, never the module-level
  `WHITE_IS_AI`, so setting the global left the predicate true and the hint went
  quietly ahead.

- **THE LAST GAME OF A MATCH SOUNDS FOR THE MATCH, NOT THE GAME (owner,
  2026-09-20).** A match is decided on TOTAL SCORE, so you can lose the final
  game and still take the match -- and the lose chime there read as having lost
  the whole thing. `endGame` now picks the chime from `matchTracker.winner` when
  `recordMatchGame` reports the match over, and from the game's own winner
  otherwise; the call had to MOVE below `recordMatchGame`, which is what sets
  that field. Earlier games in a match still sound for their own result.
  Untouched next to it: a `'tie'` game plays the LOSE chime for a human, because
  the condition special-cases `'draw'` and not `'tie'`. Pre-existing, unrelated
  to matches, not fixed.

- **THE TUTORIAL'S CLOSING PANEL NOW POINTS AT THE DIFFICULTY SLIDER (first
  tester, 2026-09-13).** `getAIDifficulty()` defaults to **1.0 = argmax, full
  strength** -- and OWNER beats it ~50-60% (corrected 2026-09-25; this file had
  it backwards, as the champion beating him), having learned to play it -- so it
  is roughly an expert's equal and will crush a beginner. A player who
  had just finished the tutorial had no idea the setting existed. The last step
  now ends: *"The computer plays at full strength by default. For a gentler
  first game, turn Difficulty down under the ⚙ settings."*
  **It had to be SHORT, and the first two drafts were not.** The phone card is
  capped to the band under the rack (`_tutFitBoard`), and `#tutText` scrolls
  inside it -- so a longer closing panel pushes the difficulty sentence, which
  is at the END of the text, below the fold. Measured: at 288 characters it
  scrolled on a phone WITH system-bar insets (the tester's own case, where the
  card is 256px rather than 270), at 226 it does not. **Do not lengthen this
  step without re-checking `scrollHeight > clientHeight` at
  `?safeinset=48,0,56,0`.** Verified no scroll and the Finish button on screen
  at phone portrait (with and without insets), phone landscape and desktop, and
  step 1's card box is byte-identical -- the bubble is pinned to the TALLEST
  step, so a longer final step would otherwise have shrunk the board on desktop
  for every step.
  **Not done, and the open question:** the default itself. Options discussed
  with owner -- lower it globally (but a silently weaker opponent is its own
  problem), ask once at the end of the tutorial with two buttons, or relabel
  the slider's ends ("Max"/"Easy"/"70%" says nothing about what it does).
  **ANSWERED, AND THE SLIDER HAS BEEN REMAPPED (2026-09-25) -- see the entry at
  the top of Current state.**

- **ANDROID 15/16 EDGE-TO-EDGE PUT THE STATUS BAR ON THE RACKS AND THE
  NAVIGATION BAR ON THE TUTORIAL'S BUTTONS (first tester, Pixel 10,
  2026-09-13).** The packaged app targets **SDK 36**; Android 15 enforces
  edge-to-edge for anything targeting 35+, and Android 16 removed
  `windowOptOutEdgeToEdgeEnforcement` entirely -- so the WebView renders BEHIND
  the system bars and there is no opt-out at that target. Owner's own Android
  never showed it (older OS), so this arrived with the first outside tester.
  **Capacitor 8.5 already hands the page the numbers, two different ways**
  (`SystemBars.java`, `insetsHandling: "css"` by default): on WebView **140+**
  with `viewport-fit=cover` it passes the insets through, so `env(safe-area-
  inset-*)` is correct AND it injects `--safe-area-inset-*` as custom
  properties; on anything older it pads the WebView instead. Either way the
  PAGE has to read them, and this one read only the bottom (for the iPhone home
  indicator) and nothing read the top at all.
  **index.html defines `--safe-t/r/b/l` as the `max()` of the two sources** and
  everything else reads only those four.
  **THE CANVAS IS NOW INSET BY THE SAFE AREA** (`_sizeCanvasToScreen` sets
  `--vx/--vy` alongside `--vw/--vh`). One change covers every piece of
  furniture the game draws -- board, racks, dice, arrows, score, HUD row -- on
  all four edges, because they are all inside the camera's frame; and it costs
  nothing elsewhere, since every world-to-CSS conversion in game.js already
  goes through the canvas's own bounding rect and **Phaser maps pointers
  through it too**. Measured: a tap aimed at the front rack piece selects that
  piece with a 48/56 inset in portrait, with 48/24/48 in landscape, and with no
  inset -- the offset canvas does not move the input.
  `_safeBottomWorld()` is **deleted**; the portrait bottom stack no longer
  subtracts anything, because the frame it sits in is already inset.
  **The DOM chrome is a separate job** -- it is positioned against the
  VIEWPORT, not the canvas, so it does not ride that inset: the gear and its
  panel, the legend button and popup, the flash notice, the first-run toast and
  the tutorial card all add the relevant `--safe-*` in `calc()`, and the five
  full-screen centred overlays (welcome, match setup, confirm, How to Play,
  coin flip) pad their centring box by all four.
  Measured on an emulated 412x915 phone, `?safeinset=48,0,56,0`:

        canvas 0..915 -> 48..859 (= 915 - 56)   tutorial rack top 15 -> 61
        tutorial card bottom 899 -> 843          gear top 10 -> 58

  and landscape `?safeinset=0,48,24,48`: canvas x 48..867, card right edge 851,
  gear right edge 855 -- all inside. **With no inset the layout is unchanged**
  (canvas, racks, HUD row, tutorial card and the desktop build all identical to
  the previous commit, measured against a stashed baseline).
  **`?safeinset` now takes `T,R,B,L` as well as a single bottom value, and it
  WRITES THE CSS VARIABLES rather than short-circuiting the JS read.** The
  first cut only overrode the JS side, so the gear -- which is CSS `calc()` --
  measured at its uninset position while the test passed: an override that only
  half the code can see tests only half the fix.
  **The insets can arrive late and without a resize**, because Capacitor
  injects them from a window-insets listener after the page is up, so
  `setupCameraControls` watches the reading itself every 250ms for the first
  six seconds and re-lays out when it moves; after that a resize covers it.
  When the BUFFER is unchanged but the canvas has MOVED, `scale.updateBounds()`
  has to be called by hand or Phaser keeps mapping pointers through the old
  rect.
  **The page is painted the theme's ground on phones** (`_paintPageGround`), so
  the strips behind the bars read as a continuation of the board rather than as
  a grey frame. Desktop keeps index.html's colour -- its letterbox bands have
  always been that colour.
  **Open, not established: whether Settings > Fullscreen actually hides the
  bars in the PACKAGED app.** It is on by default on phones, and if it worked
  there the tester would have seen no bars at all -- Capacitor's
  `onShowCustomView` may satisfy `requestFullscreen` without going immersive.
  Worth checking on a device; the layout fix stands either way, and in
  fullscreen the insets simply go to zero and the board gets the whole screen
  back.
  **Two harness traps, both of which produced a convincing wrong answer:**
  `camera.worldView` is all zeros until a frame RENDERS and headless Chrome
  does not always paint, so a world-to-CSS test must derive the view from
  `scrollX`/`zoom` (`worldView.x = scrollX + (camW - camW/zoom)/2`) instead;
  and a tap test that happens to run on the COMPUTER's turn reads as a dead tap
  -- `_inputLocked` is doing its job. Set both sides human before the game
  starts.

- **NARROW TILES KEEP THE OLD FORWARD-TAP-TO-TILE BEHAVIOUR (owner, 2026-09-11).**
  The face/halo split (tap the piece = take the selection, tap around it = move)
  needs somewhere to aim BESIDE the piece, and on a narrow tile the piece IS the
  tile. Owner reported it on rings 1-2. Measured, largest disc fitting inside the
  tile and outside the drawn piece, at rest zoom on a phone:

        field ring1  arc  63 ->  7.5 CSS px    field ring3  arc 126 -> 18.0
        field ring7  arc  84 -> 11.3           field ring4  arc 157 -> 18.1
        field ring5  arc  94 -> 13.8           field ring6  arc 220 -> 18.1
        field ring2  arc  94 -> 13.9           goal  ring7  arc 259 -> 27.1

  **The discriminator is the tile's ARC, not its ring** -- every field tile has
  the same 60 of radial extent. The OUTER field ring 7 is tighter than ring 2, so
  a ring-number rule would have fixed half the problem.
  **AND RING 5 IS NOT UNIFORM (owner's correction, and it settles the design):**
  its 12 tiles are 6 at arc 94 and 6 at arc 188, so no per-ring rule can express
  it at all. `_tileHasRoomBeside(tile, piece)` therefore asks the TILE:
  `arc >= pieceDiameter * TILE_ROOM_IN_PIECE_WIDTHS` (2.2, which falls in the gap
  between 1.88 at arc 94 and 2.52 at arc 126). Expressed in piece-widths so it
  holds for the larger goal pieces too.
  **SUPERSEDED 2026-09-25: the threshold is now 2.8 and RING 3 FORWARDS TOO** --
  45 tiles forward, 25 pass the selection. See item (i) of the learnability entry
  at the top of Current state for the re-measured ratios and the new census. The
  census below is the state at 2.2 and is kept only to show what changed:
  no room = ring1 x9 (arc 63), ring2 x9 (94), field ring7 x9 (84), ring5 x6 (94)
  = **33 tiles**; room = ring3 x12, ring4 x6, ring5 x6 (188), ring6 x6,
  **goal x6 (all wide, 259)**, home = 37.
  Behaviour measured on both sides: a face tap on a lone piece on **field ring1
  (arc 63) MOVES** (reverted, as asked), on **field ring3 (arc 126) PASSES THE
  SELECTION**.
  **Measurement trap, hit twice in this session's harness:** `_tileHasRoomBeside`
  reads the PIECE's current radius, which shrinks as a tile fills. Evaluating it
  after the tap (when the tile holds two pieces) or against the wrong piece
  reports "room" on a tile that has none -- both misreads made a narrow tile look
  wide and a correct result look like a bug. In production it is only ever called
  with `_alone` true, so the radius is `tilePieceRadius(1)`; any test must use
  that too.
  **Second harness trap, same session: `tile.addPiece(piece)` does NOT set
  `piece.currentTile`** -- it only pushes onto `tile.pieces` and re-lays out. Only
  `piece.move(tile)` sets both. A position built with `addPiece` therefore has the
  tile and the piece disagreeing about where it is, and every reachability test
  run against it silently reports nonsense (here: a destination that "was not
  reachable", making a correct result look like a bug). Build test positions with
  `piece.move(tile, false)` after removing the piece from wherever it was.

- **TAPPING EMPTY SPACE DESELECTS (owner, 2026-09-11, both platforms).** A tap
  that lands on nothing -- the nogo surround, the background outside the board --
  now clears the selection. On a phone that was previously only possible by
  tapping the piece again; Esc is desktop-only.
  **"Nothing" is Phaser's own answer, not a geometry test.** The handler sits on
  the scene's `pointerup` and returns unless `currentlyOver` is EMPTY, so tiles,
  pieces, ghosts, the racks and every HUD button are excluded for free. That
  exclusion is load-bearing: a tap on your SAVED RACK means "bank the selected
  piece", and turning it into a deselect first would silently kill that gesture.
  nogo tiles never call `setInteractive` (`drawTile` returns before
  `buildTileChrome` for them), so the surround genuinely is empty.
  Guarded like `onTap`: not a ghost mouse event, not part of a pinch, and not a
  drag or camera pan -- releasing a one-finger pan over empty space would
  otherwise deselect every time the board was moved.
  Measured: empty tap -> deselected; a tap with any object under it -> still
  selected; a 9999px release -> still selected; mid-pinch -> still selected; the
  saved-rack tap still banks (0 -> 1 saved).
  **Consequence worth knowing: it also returns a TENTATIVELY ENTERED piece to its
  rack**, because `_clearSelection` does -- so a stray background tap undoes an
  entry. Same as Esc, and fully reversible: measured rack 12 -> tentative entry
  onto home -> empty tap -> back in **slot 0**, rack 12 again, **no die spent**.
  Wired in `setupDragging`, not `setupCameraControls`, because the latter is
  phone-only and this is wanted on both.

- **THE RACKS SHUFFLE BEFORE THE COIN FLIP, NOT AFTER (owner, 2026-09-11).** The
  shuffle lived in `createPieces`, which runs during the scene restart that
  FOLLOWS the flip -- so the order was: coin lands -> board rebuilds -> racks
  visibly reshuffle. Reversed.
  **It could not just be moved, because the shuffle has to happen to the game
  that is ON SCREEN.** `_shuffleRacksThen(cb)` shuffles the HELD game's two
  unentered racks in place (`shiftPiecesUp` re-lays out from the new order),
  waits `SHUFFLE_BEAT_MS` (500) so it reads as its own step rather than flashing
  under the overlay in the same frame, then runs the coin flip -- and hands the
  resulting order to the fresh game as `rackOrder`, which `init` stores and
  `createPieces` ADOPTS instead of drawing a new one. Without that carry-through
  the racks would shuffle twice and the order the player just watched settle
  would be thrown away.
  Gated on `_gameFrozen`, the same test `createPieces` already used to identify
  the held game: anything else (a finished game, the end-of-match card) has no
  meaningful rack to shuffle and falls straight through to the old behaviour.
  Measured on both entry points -- welcome "Single game" and a match's first game
  (via match setup's Start): rack goes 1..12 -> shuffled at frame 0, coin appears
  at 300/400ms, **shuffle before coin both times**, and the order after the game
  starts is byte-identical to the one shown before the flip (no reshuffle). Later
  New Games still draw a FRESH order -- four consecutive games gave four distinct
  orders, none of them sorted -- since they pass no `rackOrder` and fall back to
  the `!_gameFrozen` shuffle.
  `SHUFFLE_BEAT_MS` is declared immediately above the function it serves, not
  among unrelated constants: `RACK_TAP_WINDOW_MS` was once deleted along with the
  log block it happened to sit under.

- **TAPPING YOUR OWN PIECE NOW PASSES THE SELECTION TO IT, WHEN IT STANDS ALONE
  ON A REACHABLE TILE (owner, 2026-09-11, both platforms).** With a piece
  selected, a tap on one of your own pieces used to ALWAYS mean "move onto the
  tile it stands on" whenever that tile was a destination — so a piece within a
  die of the selected one **could not be selected at all**; you had to deselect
  first. Owner hit this repeatedly ("sometimes I've intended the former and
  gotten the latter").
  **The obvious split — piece = reselect, tile = move — is not implementable as
  hit areas, and measuring is what showed it.** A piece's tap target grows to
  half the distance to its nearest neighbour (up to 2.4r), so a piece ALONE on a
  tile swallows the tile whole: measured on a phone, drawn radius 25 against an
  **85** target, leaving a largest free tappable disc of **0 CSS px on five of
  the eight tile geometries** (ring4 0.9, ring6 15.2, goal 8.3) against a ~22px
  fingertip. Implemented literally, moving onto your own piece's tile would have
  become impossible by tap.
  **So the split is by WHERE inside the target the tap fell**, not by which
  object got it (`_tapOnPieceFace`): the visible disc is the piece, the ring
  around it is the tile. No hit area changes, so nothing else regresses, and the
  move keeps the generous target while reselect gets the piece you can see.
  **Restricted to a LONE occupant.** A crowded tile is the case the forwarding
  exists for — there the faces are most of the tile and the slivers between them
  are unhittable — so with 2+ pieces every tap still moves.
  Measured, identical on phone and desktop: lone piece, tap its face ->
  **selection passes**, no move; same piece, tap just outside its drawn edge ->
  **moves**; 3 occupants, tap a face -> **moves**; an opponent's piece alone on a
  destination, tap its face -> **moves and captures** (`selectable` is false for
  them, so they never take the selection); own piece off the reachable set ->
  selection passes, unchanged.
  **Desktop gets it for free and is not a special case**: `_applyHitArea` returns
  early off-phone, so a desktop piece's hit area IS its drawn disc — clicking the
  piece is always a face tap, and the tile is easy to hit with a mouse.

- **THE GAME PLAYED ON BEHIND THE NEW GAME / NEW MATCH CARD (owner, 2026-09-08,
  all platforms).** Pressing either mid-game asks for confirmation over the
  running board (and How to Play covers it outright), and the computer kept
  moving and the turn kept switching while the card sat there -- so closing it
  handed the player back a position they had not been watching. Now it freezes.
  **`_gamePausedByCard()`** (beside `_preGameCardUp`) is true while `#confirmDlg`,
  `#matchSetup` or `#welcomeScreen` is in the DOM. **Derived from the DOM, not
  stored**, for the same reason the settings gear's z-index is: these cards are
  opened and removed from several places and one missed call would strand the
  game paused for the session. The existing body MutationObserver -- already
  there for the dice -- resumes on the down edge, so every dismissal route
  (Cancel, Esc, the backdrop) is covered without each having to remember.
  **Two gates were needed, and the first alone was not enough.** Everything the
  computer does funnels through `getAgentMoves`, so gating it there covers the
  trigger, the second half of a pair and the retry (a reply already in flight is
  DROPPED, not held -- `_resumeHeldAgentTurn` re-asks from the live board, which
  cannot go stale). But `applyMovePair` plays the pair out through **chained
  1-second setTimeouts**, so up to two moves and a turn switch were already
  scheduled: measured, the board visibly kept moving behind the match-setup
  card. Each deferred step now goes through `later()`, which WAITS (re-polling
  at 250ms) while a card is up instead of firing under it.
  **Confirming is not a resumption.** `showConfirm`'s Yes handler and match
  setup's Start clear `_agentTurnHeld` BEFORE removing the card: the observer's
  resume runs as a microtask, ahead of a queued `scene.restart` building the new
  game, so it would otherwise re-ask the computer for a board about to be
  discarded (the reply would be binned by the `instanceId` guard, but only after
  spending an inference and hiding the new game's thinking icon). The `Game`
  constructor clears it too, as a backstop. A confirmed New Game also ends the
  `later()` chain through the existing `stillCurrent()` check.
  **Measured** (both sides computer, so the board never stops on its own; the
  fingerprint is turn + dice + every tile's pieces + saved counts): unchanged
  over 6s with the card up and moving again within 9s of Cancel, **6/6 trials
  with the card opened 300-3100ms into the pair's chain**; before the `later()`
  gate that same test was 0/1 for match setup. Confirm starts a fresh game that
  plays (new instanceId, flag cleared, not paused). **Nothing else regressed:**
  the risky-end-turn confirm shares `#confirmDlg`, and confirming it still hands
  over -- the computer played and gave the turn back, no deadlock -- and New
  Match Start still starts a match.
  **`#howToPlay` is in the set too (owner, 2026-09-08)**, added right after the
  first cut: it covers the board outright, so the same argument applies. Nothing
  extra was needed -- it opens and closes from one place and starts no game, so
  it takes no confirm-path clearing. Measured 3/3 frozen and resumed with it
  opened 500-2500ms into the pair's chain. It is deliberately NOT in
  `_preGameCardUp`, which blanks the dice: that rule is about a board nobody is
  playing, and How to Play sits over a live one -- verified the dice still draw
  (208/169 commands) while it is up.

- **THE BOARD ACCEPTED INPUT DURING THE COMPUTER'S TURN (owner, 2026-09-02).**
  Tapping or double-tapping pieces while the computer was thinking moved them
  and corrupted the position. `Piece.onClick` gates on `player === game.turn`,
  and on the computer's turn **the computer's own pieces satisfy that** -- so a
  tap selected one, lit its destinations, and a tile tap moved it and spent its
  die. A double-tap could bank one outright. **Reproduced before fixing** (AI
  stubbed so its turn hangs, live unfrozen game): black to move, tap black 7 ->
  selected, **18 destinations offered, piece moved to field 6,2, die 6
  consumed**. The turn/thinking pill, the hover highlight, the keyboard
  shortcuts and the end-turn arrow all already refused; the piece and tile
  handlers never did, and the line that would have (`if (this.player !==
  this.game.turn) return;` in `handleClick`) had been commented out.
  Fixed with one predicate, **`_inputLocked(game)`** (beside `_currentGame`),
  applied at every entry point that mutates state: `Piece.handleClick`,
  `Piece.handleDoubleClick` (reachable straight from the stack picker's
  opponent chips, so it needs its own guard), `Tile.onClick`, `Rack.onSaveTap`,
  `Rack.onEntryPanelTap`, the **undo arrow** (the twin of the guard the end-turn
  arrow already had) and `dragstart`.
  **Two deliberate exemptions.** Setup mode -- free placement is outside the
  turn rules by design. And the **tutorial**, which scripts both sides and never
  touches the `isAI` flags: without the exemption a stored "White = computer"
  would have frozen it. Verified with exactly that setting -- `whiteIsAI true`,
  `currentPlayerIsHuman false`, `_inputLocked false`, piece selected, 2
  destinations.
  Measured after the fix: computer's piece on its turn **not selected, 0
  destinations, not moved, no die spent**; human's piece on the computer's turn
  likewise inert; double-tap, saved-rack tap and entry-panel tap all no-ops;
  undo arrow leaves the dice untouched. **Normal play intact** -- own piece on
  own turn still selects, 15 destinations, moves, spends a die -- and setup mode
  still selects a piece during an "AI" turn.
  **The AI is unaffected because it never uses these paths**: `applyMovePair`
  goes through `game.movePiece` / `piece.save` directly. Confirmed by a 70s
  both-sides-computer game: 0 -> 15 pieces on the board, 1 saved.
  **Harness note for any future frontend test:** the browser-automation skill's
  `page.evaluate` runs in an **isolated world** and cannot see game.js's globals
  (`typeof window._currentGame` is `"undefined"` there). Inject with
  `page.addScriptTag` -- that lands in the main world -- and pass results back
  through a DOM attribute, which both worlds share. Also wait for
  `_gameFrozen === false`, not just for the welcome card to go: the coin flip
  and scene restart run after it is removed, and the held game is still frozen.
  **And get the accessors right before believing a probe (2026-09-13):
  `_currentGame` is a FUNCTION, not a variable** -- reading it as an object
  yields `{}` with no error, which looks exactly like "the game never built".
  Pieces live on the GAME (`game.pieces`, 24 of them), NOT on `Player`, which
  holds only `name`/`isAI`/`gamePhase`; tiles are `game.tiles` (94), an array.
  A healthy boot reads: **94 tiles, 24 pieces (12 per colour),
  `renderer.type === 2` (WebGL), exactly one active scene, 0 failed requests.**
  The Canvas2D `willReadFrequently` warnings are the RenderTexture bake's
  readbacks -- expected, not a fault.

- **SESSION UPDATE (2026-08-27) — LIVE ON CLOUDFLARE, AND THE STORE PREP HAS
  STARTED.** The static hosting step of the roadmap is DONE and verified.
  - **LIVE AT `https://quahuru.com`** (owner registered it through Cloudflare
    Registrar, so the domain was already in the account on Cloudflare
    nameservers — attaching it was Worker › Settings › Domains & Routes › Add
    custom domain, apex and www, with DNS and certificate automatic). The
    `quahuru.rechttom.workers.dev` URL still answers the same Worker.
    **NOTHING IN THE REPO NEEDED CHANGING** — no shipped file bakes in an
    origin, and the manifest's `start_url`/`scope` are relative (`./`), so the
    app works on any hostname. Verified on the custom domain exactly as below:
    headers, Brotli, service worker controlling at the new scope, 0 API calls,
    offline reload boots the board. **The Play privacy-policy URL is
    `https://quahuru.com/privacy`.**
    **This is the THIRD origin (Koyeb → workers.dev → quahuru.com) and must be
    the last before testers are recruited**: each move means a fresh service
    worker and cache, and orphans any home-screen install pointing at the old
    one. **Koyeb still tracks main and still deploys on push, but is now
    vestigial** and can be retired once the domain has been stable a few days.
  - **Deployed at `https://quahuru.rechttom.workers.dev`.** Note it is a
    **Worker with static assets, NOT Cloudflare Pages** — the dashboard now
    steers new projects to the Workers flow, whose form has a *deploy command*
    (`npx wrangler deploy`) and NO "build output directory" field. The output
    directory comes from `wrangler.jsonc` in the repo instead:
    `{ name, compatibility_date, assets: { directory: "./dist" } }`, with no
    Worker script at all (assets-only Workers are supported). Build command is
    `python3 build_web.py --out dist`; `build_web.py` imports only the stdlib, so
    the build needs no pip install.
  - **VERIFIED AGAINST THE LIVE SITE, not assumed:** `_headers` **is** honoured
    by Workers static assets (game.js and sw.js come back `no-cache`, ort/ and
    phaser long-lived) — this was the open question when choosing Workers over
    Pages. The wasm serves as `application/wasm` with **`content-encoding: br`**,
    so a cold visit pulls ~2.8 MB rather than 11 MB. Driven under CDP: service
    worker registered at scope `/` and **controlling**, on-device runtime
    `ready`, **0 API calls** during self-play, and an **offline reload boots the
    board** (94 tiles, 24 pieces). `/index.html` 307-redirects to `/`, which is
    normal canonicalisation.
  - **Koyeb is still up deliberately**, so reverting is just handing out the old
    URL. Anyone who home-screened the Koyeb URL is still pointed at it — a
    different origin with its own service worker — and must re-add from the new
    URL. **Settle the final URL BEFORE recruiting the 12 Play testers**, or they
    all have to re-add mid-clock.
  - **`privacy.html` shipped** (in `build_web.py`'s list, `no-cache`, and
    deliberately NOT in `sw.js`'s precache — it is not needed to play offline,
    and the build script only enforces that the worker's list is a SUBSET of the
    bundle). Required by both stores even though the app collects nothing.
    **It carries a `CONTACT_EMAIL` placeholder** — publishing an address is the
    owner's call, so it was not filled in.
  - **Capacitor scaffold in place**: `package.json`, `capacitor.config.json`
    (appId **`com.tomrecht.quahuru`**, which can NEVER change once published;
    `webDir: dist`), and `android/` from `npx cap add android` (Capacitor 8.5.0).
    Only **280 KB / 53 files** of it is tracked — `android/app/src/main/assets/
    public` is the 15 MB copy of `dist` and is gitignored, since `cap sync`
    regenerates it. `npm run sync` rebuilds `dist` and syncs in one step.
    **Gotcha:** `npm install` failed with EACCES on `~/.npm`; worked with
    `--cache ./.npm-cache` (also gitignored). The real fix is
    `sudo chown -R 501:20 ~/.npm`.
  - **THE ANDROID WRAPPER NO LONGER SHIPS CAPACITOR'S BRANDING (2026-08-27).**
    `npx cap add android` generates the Capacitor logo as BOTH the launcher icon
    and the splash screen; uploading that would have put someone else's mark on
    the store. `make_android_assets.py` rewrites all 26 of them from the SHIPPED
    web icons, so launcher, listing and PWA cannot drift: `ic_launcher` /
    `ic_launcher_round` from `icon-512.png` (the board is concentric precisely so
    a round mask cannot clip it, so round reuses the same art), the adaptive
    `ic_launcher_foreground` from `icon-512-maskable.png` (already drawn with the
    padding the 72-of-108dp safe zone needs), `ic_launcher_background` set to the
    parchment #ECE3D3 so bleed at the mask edge is invisible rather than white,
    and every `splash.png` as the board centred on that ground at 42% of the
    short edge. Re-run it after changing the web icons, then `npx cap sync`.
    **The launcher icon is baked at install time — seeing a change on a device
    needs a remove-and-re-add, not a reinstall.**
    **THE FOREGROUND MUST BE SCALED DOWN — THE WEB MASKABLE ICON IS FAR TOO BIG
    FOR ANDROID (owner, on the device, 2026-08-27).** An adaptive icon is 108dp
    but only the central **72dp — 66.7%** — survives the launcher's mask, much
    tighter than the web maskable spec. Measured: `icon-512-maskable.png` draws to
    **85.2%** of its canvas (and `icon-512.png` to 95.2%), so pasting it unscaled
    put the board's rim outside the visible circle on a real phone. The script now
    MEASURES the art's extent — as a radius about the centre, since the board is
    circular and a bounding box would overstate the corners — and scales it to
    **64%**, pasted onto the ground rather than resized to fill, so the mask can
    only ever cut parchment. Verified on the rebuilt asset: art spans **64.7%**,
    and simulating a 66.7% circular mask cuts **0** non-parchment pixels.
  - **RELEASE SIGNING is wired to `android/keystore.properties`**, which is
    GITIGNORED along with `*.jks` / `*.keystore`; `keystore.properties.example`
    shows the four keys. `app/build.gradle` applies the signing config ONLY when
    that file exists, so a fresh clone still builds debug and `bundleRelease`
    fails safely rather than producing an unsigned upload. Create the key with
    `keytool -genkey -v -keystore ~/quahuru-upload.jks -keyalg RSA -keysize 2048
    -validity 10000 -alias quahuru`. **Use Play App Signing and back the .jks and
    its passwords up** — the upload key is how Google knows an update is from
    the same developer.
  - **A SIGNED `.aab` HAS BEEN BUILT: 8.3 MB, `jarsigner` says "jar verified"**,
    at `android/app/build/outputs/bundle/release/app-release.aab`. Confirmed to
    contain the whole app — game.js, model.onnx, the 11 MB ort wasm and phaser
    under `base/assets/public/`. Project is targetSdk/compileSdk **36**, minSdk
    24, applicationId **`com.quahuru.game`**, label "Quahuru". **Every later
    upload needs a HIGHER versionCode** — Play rejects a repeat. Hit
    immediately: the first upload predated the adaptive-icon fix, so Play kept
    serving the clipped icon through an uninstall/reinstall, and replacing it
    needed versionCode 2. **A reinstall from Play does NOT pick up a local
    rebuild** — obvious in hindsight, easy to misread as the icon fix having
    failed.
    **CURRENT PACKAGE: versionCode 8, versionName 1.0.7, 8.7 MB, built
    2026-10-06** -- the FIRST PRODUCTION RELEASE (Play granted production access
    2026-10-06). Carries the 4-way blend net (`blend4way_Oct6`), the 0.65-1.0
    difficulty knots, the fitted prefilter scales and `sw.js` `quahuru-v9`.
    Verified on the bundle: `jar verified`, versionCode 8 / 1.0.7 /
    com.quahuru.game, and game.js, agent.js, engine.js, local_agent.js,
    index.html, sw.js, model.onnx hash-identical to the working tree.
    Previous: versionCode 7, versionName 1.0.6, 8.7 MB, built 2026-10-03 -- everything from 2026-09-26 to 10-02: the two computer rules
    (never pass a die that could save; bank the most when the opponent's last two
    blanks sit on goal 1), the AI save using the agent's die, the end-card resize
    fix, the refusal explanations, dismissible advice notices, the tutorial rework
    and rule tips, the stack-picker double-tap and the rack-entry engine fix. The
    recorder and the position tools ship but are DORMANT -- both need a URL
    parameter (`?rec=` / `?dev=1`) the app never receives. Only `game.js`,
    `agent.js` and `engine.js` changed among shipped files, all network-first, so
    `sw.js` keeps `quahuru-v8`. Verified on the BUNDLE: `jar verified`, versionCode
    7 / 1.0.6 / com.quahuru.game from its manifest, and game.js, agent.js,
    engine.js, local_agent.js, index.html, sw.js, model.onnx hash-identical to the
    working tree.
    Previous: versionCode 6, versionName 1.0.5, 8.7 MB, built
    2026-09-25 -- the learnability batch: the hint pill and its sum-move
    collapse, the shortest-route explanation, the tutorial's entry point in How to
    Play, the closing difficulty buttons, the difficulty-slider remap onto
    0.8..1.0, the withheld-sum highlight and the match-winner chime. See the
    entries at the top of Current state.
    **`game.js` and `local_agent.js` are the ONLY shipped files that changed**
    since versionCode 5; index.html, sw.js, the manifest, the ported agent and the
    ort runtime are byte-identical, so **`sw.js` keeps `quahuru-v8`** -- CHECKED,
    not assumed: `ALWAYS_FRESH` is `!VENDORED && /(\/|\.html|\.js|\.json)$/`, so
    every non-vendored `.js` is network-first and both changed files are covered.
    Verified on the BUNDLE: `jar verified`; versionCode `6` read as the
    length-delimited value after the `versionCode` attribute name in
    `base/manifest/AndroidManifest.xml`, versionName `1.0.5` and package
    `com.quahuru.game` as UTF-8 strings; and `base/assets/public/game.js`,
    `local_agent.js` and `index.html` all hash-identical to the working tree.
    **Two PATH gotchas on this iMac:** `npx` and `node` are not on the default
    PATH for a non-interactive shell -- prefix with `PATH="/usr/local/bin:$PATH"`
    -- and `jarsigner` comes from `./.jdk/jdk-21.0.12.1+1/Contents/Home/bin/`, not
    the system.
    Previous: versionCode 5, versionName 1.0.4, 8.3 MB, built
    2026-09-13** -- the safe-area fix for Android 15/16 edge-to-edge (see the
    entry at the top of Current state); it is the one to upload for the closed
    test, since versionCode 4 puts the status bar on the racks on any Android
    15+ device. Verified: `jar verified`, versionCode 5 / versionName 1.0.4 /
    com.quahuru.game read out of the bundle's protobuf manifest (an .aab's
    manifest is PROTOBUF, not binary XML -- `aapt2 dump xmltree` cannot read
    the bundle at all, and a UTF-16 string scan finds nothing; grep it as
    UTF-8), and `base/assets/public/game.js` and `index.html` hash-identical to
    the working tree.
    versionCode 4 / 1.0.3 (also 2026-09-13) carried
    the September input and pre-game work — the ghost-tap fix, the input lock
    during the computer's turn, the freeze behind New Game / New Match / How to
    Play, the tap-claims-the-gesture fix, tap-to-pass-the-selection, empty-space
    deselect, the shuffle-before-the-coin-flip order and the narrow-tile
    exemption. **`game.js` was the ONLY shipped file that changed** (542 diff
    lines over ten commits); index.html, sw.js, the manifest, the ported agent
    and the ort runtime are byte-identical, so `sw.js` keeps `quahuru-v8` — it
    is network-first for index.html and game.js, and no cache-first asset moved.
    **DO NOT REBUILD THE PACKAGE FOR EVERY CHANGE (owner, 2026-09-13).** Push
    commits as normal -- the web build at quahuru.com carries each one -- and
    package only when a BATCH is worth a new version; the versionCode bump
    belongs to that moment. Every upload costs a Play review cycle and a
    permanent versionCode, and the closed test's 14-day clock counts opted-in
    testers, not versions, so batching costs nothing.
    **Rebuilding the package (the whole recipe):**
    `python3 build_web.py --out dist` → bump `versionCode`/`versionName` in
    `android/app/build.gradle` → `npx --cache ./.npm-cache cap sync android` →
    `cd android && ./gradlew bundleRelease > /tmp/rel.log 2>&1; echo $?`.
    `npm run sync` does the first and third in one step.
    **Verify before uploading, and verify the BUNDLE, not the source:**
    `jarsigner -verify` (expect "jar verified"; the PKIX warning is normal for a
    self-signed upload key), the versionCode/versionName/package read out of
    `base/manifest/AndroidManifest.xml` in the .aab, and a hash of
    `base/assets/public/game.js` against the working tree's.
    **The package name was changed from `com.tomrecht.quahuru` to
    `com.quahuru.game` (owner, before any upload)** — it matches the domain he
    now owns, and it is PERMANENT once anything is uploaded to any track, so this
    was the last moment. It lives in seven places: `capacitor.config.json`,
    `android/app/build.gradle` (namespace AND applicationId), `strings.xml`
    (`package_name` and `custom_url_scheme`), the copied
    `android/app/src/main/assets/capacitor.config.json`, and the Java package
    line — plus the source DIRECTORY has to move to `java/com/quahuru/game/`.
    Verified by reading the string pool out of the built bundle's compiled
    manifest, not by grepping the source.
  - **THE BUILD NEEDS A JDK 21 EXACTLY — BOTH JDKs ON THE MACHINE ARE WRONG.**
    A narrow window, and it cost two failed builds:
    * system **JDK 17** is too OLD — Capacitor 8 wants 21+, giving
      **`invalid source release: 21`**;
    * Android Studio's bundled **JBR is Java 25**, too NEW — Gradle 8.14 cannot
      run on it and dies with **`Unsupported class file major version 69`**.
    **The JBR appeared to work twice and did not.** Those builds only succeeded
    because the compiled BUILD SCRIPTS were still cached; the moment
    `app/build.gradle` was edited (the package rename) and had to be recompiled,
    it failed. A cached artefact can hide a broken toolchain — re-verify after
    touching a build script, not just after touching code.
    So `./.jdk` holds a self-contained **Temurin 21** (gitignored, no sudo, no
    system install), pointed at by `org.gradle.java.home` in
    `android/gradle.properties`. **That path is ABSOLUTE and iMac-specific** —
    on the MacBook either re-download into `./.jdk` or, for a path that is the
    same on both machines, `brew install --cask temurin@21` and point at
    `/Library/Java/JavaVirtualMachines/temurin-21.jdk/Contents/Home`.
    `android/local.properties` (gitignored) carries `sdk.dir`; writing it by hand
    means the project never has to be opened in Android Studio at all — the SDK
    it installs is the only thing needed from it.
  - **METHOD NOTE: `./gradlew ... | tail` REPORTS EXIT 0 ON A FAILED BUILD.**
    The pipeline's status is `tail`'s, so the first failure was recorded as a
    success and only the log text gave it away. Redirect to a file and check
    `$?`, never pipe a build into `tail` and trust the code.
  - **`feature-graphic.png` (1024x500)** from `make_feature_graphic.py`, the
    Play listing's required banner. It composites the SHIPPED `icon-512.png`
    rather than re-rendering from `make_icons.py` — that cannot drift, and it
    asserts the icon's ground still equals the banner's (#ece3d3) so the square
    icon has no visible seam. The lockup is centred and auto-shrinks to a 900px
    safe width (Play crops this asset differently per placement); measured
    margins 77/72/80/80.
  - **Licence position, checked this session:** there is **no LICENSE file and
    that is correct** — no licence means all rights reserved, so nothing is
    foreclosed. Do NOT add MIT/Apache to this repo; it would let anyone ship
    Quahuru. The repo is PUBLIC, which grants no use rights but does make the
    code and model readable. **Open compliance item: `phaser.min.js` ships with
    NO MIT notice** (onnxruntime's is intact) — MIT requires the notice in
    copies, and both stores expect an attributions screen. Fix with a NOTICE
    file plus an Acknowledgements line in How to Play. Board-game specific:
    rules and mechanics are not copyrightable, only their expression — the
    NAME is the protectable asset, via trademark.
    **A COPYRIGHT LINE IS IN THE APP (owner, 2026-08-27)**, in How to Play >
    Credits: "Quahuru — the game, its rules, artwork and neural network — is
    © 2026 Tom Recht. All rights reserved." Copyright subsists without a notice,
    so this is not a legal necessity, but both stores expect one and it is what
    tells a reader whose game this is. `privacy.html` and `licenses.html` already
    carried the same line in their footers.
    **DONE (2026-08-27): `licenses.html`** carries the full MIT text for both,
    with the exact upstream copyright lines fetched rather than remembered —
    Phaser "Copyright (c) 2020 Richard Davey, Photon Storm Ltd." (from
    `unpkg.com/phaser@3.55.2/LICENSE.md`) and ONNX Runtime "Copyright (c)
    Microsoft Corporation" (from the runtime we actually ship). Reachable from
    **How to Play › Credits**, which also links the privacy policy — verified on
    both platforms, and the phone text still contains no "click".

- **BRANCHES PRUNED 14 -> 5 (owner, 2026-08-27).** Kept: **`main`** (Cloudflare /
  quahuru.com), **`testing`** (Koyeb), **`symmetry-aug-main`** (the deployed
  champion's lineage, and the ONLY home of `symmetry.py` — main does not have the
  symmetry work at all), and **`gnn` / `gnn2`** (the pre-TD training history, 67
  and 101 unique commits, kept pending owner's call). Deleted: seven branches
  with **zero** commits not already reachable from main (`app-packaging`,
  `deeper-search`, `fast-prefilter`, `frontend-overhaul`, `good_gnn`,
  `rule-single-piece-save`, `td-lambda`), plus `rule-numbered-win` (owner dropped
  the idea) and `symmetry-aug` — after carrying `symaug_smoke.py` onto
  `symmetry-aug-main`, the one file that existed nowhere else. `symmetry.py` was
  byte-identical on both.
  **Method note that nearly caused a wrong conclusion:** `git rev-list -n 1 main
  -- <file>` applies history simplification and reports NOTHING for files that
  only ever appear on a merged side-branch. Use `--full-history`. And the commit
  it names is the one that DELETED the file, so extract from its parent —
  `git show 782c43c0^:train.py`, not `782c43c0:train.py`.

- **REFERENCE: the interactive blocking tool** built 2026-08-27 lives at
  `https://claude.ai/code/artifact/b16ccf81-f0a9-4192-bf58-46a103eaaf64`
  ("Where to Build a Wall"): click a tile to pick the piece to slow, and every
  other tile shows the extra turns a wall there costs it, for a blank or any
  numbered piece. Data is the source x wall matrix from the DP; the generator
  scripts were scratchpad one-offs and are NOT committed, so re-deriving it means
  re-running the value iteration described above.

- **`testing` IS A STAGING BRANCH: FEATURES GO THERE FIRST, THEN TO `main`
  (owner, 2026-09-25).** This REPLACES the old cherry-pick-to-both rule, which was
  the right answer only while `testing` carried instrumentation on top of main.
  It does not any more -- the tap log is retired and the two branches share one
  history, with main an ancestor of testing -- so the flow is now one-directional
  and there is nothing to apply twice.

        new feature  ->  commit on `testing`  ->  push  ->  KOYEB builds it
                     ->  owner plays it there
                     ->  `git checkout main && git merge --ff-only testing`
                     ->  push  ->  CLOUDFLARE builds quahuru.com

  * **`main` is live.** Cloudflare builds it on push and it is what testers
    install and play, so it must stay clean.
  * **`testing` is tracked by KOYEB**, which serves `app.py` from the repo root,
    so a branch works there with no build step.
  * **SMALL FIXES GO STRAIGHT TO `main` (owner, 2026-09-25).** Staging is for a
    FEATURE worth playing before testers see it; a label, a default, a position
    fix or a link does not earn a round trip through Koyeb. Commit on main, push,
    then fast-forward `testing`.
  * **A HOTFIX for the live site may likewise go straight to `main`** -- then
    `git checkout testing && git merge --ff-only main` so testing does not fall
    behind. Never cherry-pick: the branches share history, so a cherry-pick
    would recreate the parallel-history problem this replaced.
  * **Keep them fast-forwardable.** After a feature lands on main both should be
    the same commit; `git diff main testing` empty is the normal state between
    pieces of work, not a sign something is missing.
  * **KOYEB TESTS THE GAME, NOT THE HOSTING.** The two targets are not the same
    environment: quahuru.com is a Cloudflare Worker serving static assets with
    `_headers`, Brotli and a service worker at scope `/`, while Koyeb is a Flask
    process serving files from the repo root. Anything that depends on caching,
    headers, the service worker or offline behaviour has to be checked on
    quahuru.com after the merge -- staging on Koyeb will not show it.
  * **A DIAGNOSTIC STILL DOES NOT BELONG ON EITHER.** `testing` is now a staging
    branch that owner PLAYS, so instrumentation left there is instrumentation in
    a build he is playing. **An investigation gets its own throwaway branch,
    deleted when the bug is found** -- that is the lesson the retired tap log
    left, and a `?dev=1`-style gate is not a substitute, because asking owner to
    set a query parameter before a bug he cannot predict has failed twice.
  * **Koyeb is also still the revert path** for the live site (hand out the old
    URL), which is the other reason the branch is kept. Anyone who home-screened
    the Koyeb URL is still pointed at it -- a different origin with its own
    service worker.
  **Watch-out from the tap log's time there, still worth knowing:** a settings
  control gated on state that only exists later never appears, because
  `createSettingsPanel` runs ONCE at start-up -- the same race as the first-run
  nudge.
