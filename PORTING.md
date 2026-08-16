# On-device inference — porting plan (branch `app-packaging`)

Goal: run move selection in the browser so the app needs no server. That removes
the Koyeb instance, the cold starts, the 504s and the MOVE_BUDGET juggling, and
it is the prerequisite for a store build that works offline.

`model.onnx` (1.66 MB) already runs as-is under **onnxruntime-web** (WASM). What
has to be ported to JS is the *input* side and the search:

| piece | source | size | notes |
|---|---|---|---|
| static tile graph + base tile features | `encoder.py` | — | **DONE** — exported, not ported (see below) |
| dynamic tile features | `encoder.py:update_tile_features_dynamic` | ~70 lines | pure numpy |
| piece features | `encoder.py:encode_piece_features` | ~110 lines | needs the BFS distances |
| piece↔tile edges | `encoder.py:encode_piece_tile_edges` | ~25 lines | trivial |
| global features | `encoder.py:encode_global_features` | ~70 lines | |
| route BFS | `game.py:shortest_route_to_goal`, `all_goal_distances`, `_get_blocked_key` | ~120 lines | the measured hot spot (~35% of worker time) |
| 2-ply search + heuristic prefilter | `agent_gnn.py` | ~400 lines of the 1452 | **DONE** — `agent.js` |
| game engine | `game.py` | ~600 lines | **DONE** — `engine.js` (the search needs apply/undo) |
| heuristic evaluator | `agent.py` | ~250 lines | **DONE** — `heuristic.js` (the prefilter needs it) |

## Decision: export the static half, don't port it

`build_tile_index` collects edges into a `set`, so the column order of
`tile_edge_index` is Python's set-iteration order. A JS reimplementation would
produce a different order and, while sum-aggregation is order-independent in
exact arithmetic, float addition is **not associative** — the scores would
differ in the last bits, which is enough to flip an argmax in a near-tie (and
near-ties are exactly where this agent's known rough edges live).

So `export_encoder_static.py` writes **`encoder_static.json`** (10 KB: 70 tiles,
186 directed edges, the 70×12 base tile features, tile types/numbers and the
feature dims) and JS loads it. Re-run the exporter whenever
`tile_neighbors.json` or `encode_tile_features` changes; it self-verifies by
reloading and comparing arrays exactly.

## Decision: a plain state snapshot, not a ported Board

The encoder touches a small, enumerable surface of the board:

```
board.pieces  .white_unentered .black_unentered  .white_saved .black_saved
     .tiles   .dice  .current_player  .game_stages
     .get_game_stage()  .shortest_route_to_goal()  .all_goal_distances()  ._get_blocked_key()
```

Rather than porting `game.py`'s `Board` (game.js already has its own, different,
board model for human play), the JS encoder should take a **serialisable
snapshot** that both sides can produce:

- Python emits it in the test fixture, so expected outputs can be compared.
- `game.js` builds it from its own `Board` at runtime.

This keeps the two board models apart and makes every step independently
testable. The BFS routines are the one part that must be genuinely ported, since
they are called *from inside* the encoder — note `game.js` already has
`getReachableTiles`, but that answers "where can this piece go with N steps",
which is not the same question as `shortest_route_to_goal`; do not assume it can
be reused without checking.

## Order of work

1. ~~Export the static half + self-check.~~ **done**
2. ~~Fixture.~~ **done** — `python dump_encoder_fixture.py [n] [out.json]`
   writes N seeded positions as `{snapshot, expected}`. 40 positions is 538 KB,
   so it is gitignored and regenerated on demand (seeded, so deterministic).
   `snapshot` is the shape the JS encoder takes: pieces with their tile/rack
   location, rack ORDER (entry order matters), dice, current player, stages.
   `expected` carries tile_feats 70x12, piece_feats 24x24, global 11, both edge
   arrays, and **`piece_order`** — `all_pieces` decides the row order of
   piece_feats and the edge indices, so the port has to reproduce it.
3. ~~Port the BFS trio.~~ **done** — `route.js`: `blockedTiles`,
   `shortestRouteToGoal`, `allGoalDistances`, plus `piecesFromSnapshot` to turn
   a fixture snapshot into the piece shape they want. The tile graph is built
   from `encoder_static.json`, whose exclusion of nogo tiles is equivalent here
   (nogo is never traversable and never a goal). Two details that matter: the
   goal test runs BEFORE the traversability test, so a goal is found even though
   goals are not walked through; and unreachable is `Infinity`, stored as null
   in the fixture since JSON has no infinity.
   **Verified against encoder.py over 40 positions: 960/960 shortest routes and
   5760/5760 goal distances, 0 blocked-set mismatches.**
4. ~~Port the feature encoders.~~ **done** — `encoder.js`. Verified against
   encoder.py over 40 positions, exactly: tile_feats 33600/33600 cells,
   piece_feats 23040/23040, global_feats 440/440, both edge arrays 40/40, and
   piece_order 40/40.
   Three things that would have been silent bugs:
   - Tile counts are per CURRENT PLAYER, so `encode()` takes the player
     explicitly rather than reading the snapshot.
   - `distance_category` runs the OPPOSITE way from `_dist_bin`: unreachable is
     0 and "close" is 1, where the bin has unreachable at 1.
   - Game stage is computed FRESH, never read from the snapshot's
     `game_stages` — that dict is mutated as a side effect of get_valid_moves
     and goes stale during candidate enumeration (the bug that once scored
     identical candidates 20-80 points apart).

5. ~~Wire onnxruntime-web.~~ **done** — `infer.js` + `infer_test.html`.
   **Max |JS - Python| = 4.47e-08 over 40 positions**, the same order as
   onnx_export.py's own graph-vs-torch check (8.8e-08), so the JS path adds no
   meaningful error. Batching verified (5 positions in one call), which matters
   because the search scores thousands of candidates a move.
   Four traps, all of which presented as something other than what they were:
   - `ort.env.wasm.wasmPaths` must be ROOT-ABSOLUTE. The loader dynamically
     imports the `.mjs` beside the wasm, and a relative path is not a valid
     module specifier -- it fails as "no available backend found", which reads
     like a missing build.
   - Classic scripts share ONE global lexical scope, so a top-level `const` of
     the same name in two files is a redeclaration that kills the second file
     with a SyntaxError and leaves its exports undefined. Hit twice
     (`buildGraph`, then `_api`); both export footers are now IIFE-wrapped and
     the route import is namespaced as `_R`.
   - The service worker was serving stale copies of route/encoder/infer.js,
     because ALWAYS_FRESH listed only index.html and game.js. A fix literally
     could not reach the browser.
   - The threaded wasm runs single-threaded via `numThreads = 1`, so no
     COOP/COEP headers are needed.
6. Port the prefilter + 2-ply search; then the real proof: **a trace-diff** —
   identical move traces over 40 turns against the Python agent.

   **6.0 (done): the Python agent now agrees with itself.** The trace-diff
   standard was not reachable as written. `select_move_pair` enumerates over
   `set`s of move tuples, so its order varies with the hash seed, and every tie
   was settled by "whichever came first" — the argmax, the prefilter's stable
   sorts, and the fewest-relocations tie-break. Measured over 30 *pinned*
   positions, **3 chose a different pair under a different PYTHONHASHSEED**, all
   transpositions (same piece, same destination, dice the other way round).
   Ties now break on a canonical move key (`_move_sort_key`): identical choices
   across three hash seeds, and where the choice differs from the old code the
   score is identical to 0.0, so only exact ties moved.
   *Measurement trap this hid behind:* the first probe showed 24/30 differing,
   because `get_valid_moves` walks a `set` of **Tile** objects whose hashes are
   ids — so the random walk building the positions diverged too, and it was
   measuring position drift, not agent drift. Pin the walk (`sorted(..., key=repr)`)
   and record a position fingerprint alongside the choice.

   **6.1 (done): the engine.** `engine.js` + `dump_engine_fixture.py`.
   Verified over 50 positions (10 opening / 30 midgame / 10 endgame): positions
   rebuilt identically 50/50, move sets 50/50, position after a move 3054/3054,
   undo restores the start 3054/3054.
   Four things that were only visible because the fixture checked them:
   - **Within-tile occupancy order is not semantic, and the engine does not
     preserve it** — undo APPENDS to origin_tile rather than restoring the
     index, which reordered 125 of 739 apply/undo pairs. Every reader is
     `pieces[0].player` on a field tile (only ever one player's, since landing
     on a lone enemy captures it) or `pieces.pop()` on a single-enemy tile, so
     the contract is the multiset.
   - **`board.pieces` ORDER is observable** through the dedupe of
     interchangeable blanks on one tile (it keeps whichever comes first), so
     the same destinations get attributed to a different blank. The served path
     always rebuilds via `update_state` → `assign_piece_indices` (white first,
     then by number); a Board built and walked in a script keeps
     `initialize_pieces`' shuffle. Canonicalise, or the two disagree.
   - **`get_valid_moves` mutates `game_stages`** for the current player, and
     later code reads it. The port reproduces the side effect deliberately.
   - `encoder_static.json`'s tile order is identical to `tile_neighbors.json`'s
     file order and so to `game.py`'s indices (asserted 70/70), so a tile index
     means the same thing on both sides and no second geometry file is needed.
     Neighbour ORDER is not load-bearing (the one place it is walked wraps the
     result in `set()`).
   A random walk never reaches the endgame — saving needs a piece parked on its
   own goal — so those positions are hand-built (`endgame_board`). Without them
   the sample was 38 opening / 2 midgame / 0 endgame.

   **6.1 as originally written** (kept for the reasoning): the search runs on
   `get_valid_moves` / `apply_move` / `undo_last_move` / `check_game_over`, so it
   needs an engine, not just the search. game.js's board model is NOT usable —
   it is entangled with Phaser display objects, has no search-shaped undo, and
   deliberately diverges (it offers the rack reordering the engine does not).
   Port `game.py`'s Board as plain data (`engine.js`), fed by the same snapshot
   shape `encoder.js` already takes. ~600 lines. Fixture first: dump
   `sorted(get_valid_moves())` plus the resulting position after each move, for
   N seeded positions, and assert against it.
   Watch-outs, all of which are silent if wrong: block-save (`destination == 0`)
   marks BOTH dice used and its undo must clear both unconditionally, and it
   restores the turn to the mover's OPPONENT; `undo` returns a rack piece to
   `origin_rack_index`, not the front; `game_stages` is mutated as a side effect
   of `get_valid_moves`; and `apply_last_piece_rule` RENUMBERS a piece to 13.

   **6.2 (done): the heuristic.** `heuristic.js` + `export_heuristic_weights.py`
   + `dump_heuristic_fixture.py`. Only `evaluate` is ported — that is all
   agent_gnn uses it for; agent.py's own `select_move_pair` is the retired
   evolutionary agent. **2646/2646 evaluations bit-exact** (worst |JS − Python|
   = 0), over 37 positions and the position after every legal move.
   Weights are EXPORTED for the same reason the static encoder half is: the
   served values are `get_weights()`'s merge of best_weights.json over
   INITIAL_WEIGHTS (the file has no `enemy_blot_penalties`,
   `high_goal_proximity_penalties` or `permanent_block_bonus`), then
   `_expand_weights`' `a*n**b` tables. A wrong exponent would be silent.
   Three traps, the first of which contradicts 6.1's own note:
   - **NEIGHBOUR ORDER IS LOAD-BEARING.** It makes no difference to a
     shortest-path length — which is why route.js passed 960/960 without it —
     but `count_enemy_blots_on_shortest_path` is a plain FIFO BFS where the
     first predecessor to reach a tile fixes its blot count. 946 evaluations
     differed until `tile_neighbors.json`'s own order was exported
     (`tile_neighbors` in encoder_static.json) and used in place of the order
     derived from `tile_edge_index`, which came out of a Python set.
   - **The SAVED RACK's order is observable**: `saved_bonuses` is summed over
     the rack in order, so sorting it in the fixture moved the score by 9e-13.
     Tiny, but enough to reorder a prefilter near-tie, and bit-exactness is
     worth keeping — it removes float noise as a suspect in 6.4 entirely.
   - `if p.tile` in Python is true for ANY Tile **including home**, whose index
     is 0. In JS `p.tile` alone is falsy there; the faithful test is
     `p.tile >= 0`.

   **6.3 `select_move_pair`** itself (~290 lines) plus `_select_filtered`,
   `_dedupe_save_pair` and `_pick_move_index`. `_fix_never_good` is disabled by
   default (`enable_never_good=False`) and can be skipped.

   **6.4 (done): the proof.** `dump_trace_fixture.py` + `trace_test.html`.
   Two seeded games played by the SERVED agent (real model.onnx, prefilter
   F=12), 110 turns, each position replayed through the JS agent with real
   browser inference: **110/110 identical chosen pairs**, and the candidate sets
   agree 110/110.
   The worry going in was that infer.js matches Python to 4.5e-08 rather than
   exactly, so a Python tie can be a 1e-8 gap in JS and the canonical tie-break
   would not fire. It is a real effect — **30 of the 110 turns are decided by a
   margin below 1e-5, and the smallest is exactly 0** — but it does not reach the
   answer, because those ties are almost all TRANSPOSITIONS, and the
   fewest-relocations tie-break groups candidates by their RESULTING POSITION
   rather than by score equality. It therefore fires identically on both sides
   whatever the last bits do. A tie between genuinely different positions could
   still diverge; none occurred here.
   Run it: serve the repo, `python dump_trace_fixture.py 2`, then open
   `/trace_test.html` (or drive it with the CDP runner). onnxruntime-node is not
   installed, so the browser is the only place the JS agent can score.

## Watch-outs carried over

- ~~Self-play is **not** run-to-run deterministic unless `PYTHONHASHSEED` is
  pinned.~~ Fixed for the AGENT by 6.0 above — `select_move_pair` is now a pure
  function of the position. `get_valid_moves`' own return ORDER is still
  unstable (it walks a `set` of Tile objects, whose hashes are ids), so anything
  that consumes that list positionally — a random walk, a "pick the first legal
  move" — must still sort it canonically.
- The served prefilter is `FIRST_MOVE_PREFILTER=12`; the library default is 0.
  Match whichever you are comparing against, or the traces will diverge for
  reasons that have nothing to do with the port.
- `agent_gnn` does top-k/argmax/difficulty-softmax in numpy already
  (`gnn_backend.py`), so those are small and mechanical.


## Step 7: wiring it into game.js — DONE (2026-08-15)

`local_agent.js` + a rewritten `getAgentMoves`. The computer now moves on the
device, with `/select_moves` as the fallback. Shipped as designed below: both
platforms, lazy, server as fallback.

**What was actually new.** The port was already proven; the only fresh logic is
`LocalAgent.engineState`, which turns game.js's board into the engine's. It is
written as a direct mirror of `game.py`'s `Board.update_state` and takes the SAME
payload `getGameState()` posts, which is what makes the differential check below
a real test rather than an approximation.

**Verified, in this order:**

* **The rebuild** — `state_test.html` + `scratchpad/state_expected.py`, over the
  181 states three real games actually posted, plus firstMove-carrying variants:
  **702/702 positions and 702/702 legal-move sets identical to `update_state`,
  523/523 with a firstMove set.**
* **End to end** — `?aicompare=1` plays the real app with both sides Computer and
  asks the server the same question every turn: **162/162 identical pairs over
  three complete games**, no page errors.
* **Fail-closed** — `?localai=0` never loads it; an injected failure mid-game
  retires the local path and the game plays on from the server.

**A REAL BUG THIS FOUND, in code that was already "verified".**
`Engine.fromState` resolved `first_move` with `e.find()` *before* `reindex()`,
which is what builds `lookup` — so `firstMove.piece` was **always null**. That is
silent: `firstMove.piece === piece` simply never matches, so the turn's first
mover is not held to the dice SUM and the legal-move list comes out wrong.
**Every one of step 6.1's 50 fixture cases had `first_move: null`** (a random
walk samples turn starts), so 3054/3054 passed over a branch it never entered.
Step 7 hit it immediately, because `update_state` sets `firstMove` from a posted
`reachableBySum`. Fixed; the 6.1 fixture is byte-identical before and after.
*Lesson for the rest of the port: check the fixture's DENOMINATOR for the branch
you think it covers, not just its pass rate.*

**A measurement trap that cost an hour, worth remembering.** The first compare
run showed 7 disagreements, all of them the canonically-smaller pair losing —
the exact signature of a missing tie-break. It was neither the port nor the
agent: `game.js` hardcodes `SERVER_URL` to `localhost:10000` for any local page,
and `LocalAgent.init()` re-applied that argument on every call, so a harness
testing against a second Flask instance had its comparison silently retargeted
at the first one — which was a **stale process running code from before the step
6.0 determinism fix**. Two things came out of it: `?aiserver=` now pins the
compare target explicitly, and `init()` no longer lets a later call move it.
Before blaming the port, check the server you are actually talking to
(`ps -o lstart=` on the listener) — and never leave a long-lived dev server
running across a change to the agent.

**Main-thread cost — MEASURED (`latency_breakdown.html`), and the conclusion is
"leave it alone for now".**

Where a move's time goes, 40 recorded positions, mean ms/turn:

| | heuristic | encode | **ort** | engine | median move |
|---|---|---|---|---|---|
| 1x | 49 (8.7%) | 11 (2.0%) | **498 (88.1%)** | 7 (1.2%) | 390ms |
| 4x throttle | 205 (9.4%) | 52 (2.4%) | **1884 (86.7%)** | 32 (1.5%) | 1443ms |

**This overturns the hypothesis recorded below and in TODO.md** that "the
encoder's per-candidate BFS is the likely hot spot, not the forward pass". It is
the forward pass: ~88%, stable across throttling. The encoder is 2%. Cutting
candidates therefore buys far less than assumed, which is a second, independent
reason not to touch `prefilter_top_k`.

The thread really is held: a 10ms `setInterval` fired **0 times out of 2243
expected** during moves at 1x (0 of 8673 at 4x). So the UI cannot paint for the
length of a move.

**But it is not visible, and owner confirms it in play.** The only thing
animating during the computer's turn is the thinking icon's 1000ms yoyo fade
(`showThinkingIcon`), and a ~0.4s stall in a slow fade reads as an icon sitting
still, not as a freeze. The board is static and there is nothing to interact
with. So on desktop the cost is real and inert.

Where it would show is a phone: median 1.4s and p90 4.9s at 4x is a visibly
stalled pulse and queued taps. **That has not been tested on a real device yet**,
and that test — not a worker — is the next step.

If it does show, the fix is graded, and the measurement says start small:
`ort.env.wasm.proxy = true` moves inference alone into a worker for ONE LINE and
removes ~88% of the hold. A full Web Worker around the whole agent buys the last
12% and is a much bigger change (every port file already falls back to `self`,
so it is buildable, but it is not obviously worth it).

Measurement traps this hit, both of which reported a serene zero:
- **A rAF ticker is useless headless** — no compositor, so no animation frames
  fire whether or not the thread is held.
- **A gap metric cannot see a full block.** Measuring the longest interval
  between timer firings needs two firings INSIDE the window; when the thread is
  held for the whole move there are none, and the metric reports 0 — identical
  to "never blocked". Count firings against expected instead.
- `PerformanceObserver` callbacks are delivered asynchronously, so clearing the
  buffer synchronously after each turn races them and loses every entry.

**Still on `/select_moves`:** nothing, for move selection. `/call_draw`,
`/start_game`, `/evaluate_board`, `/query_agent_move` and the `record_*` routes
are untouched and are the subject of the hosting audit (see CLAUDE.md).

### The design as it was settled (kept for the reasoning)

Measured first, because it decides the shape. `latency_test.html` runs the real
agent over 40 recorded positions:

| | init | median move | p90 | max | candidates scored |
|---|---|---|---|---|---|
| desktop (no throttle) | 0.68s | 0.29s | 0.71s | 0.96s | 40 median, 145 max |
| 4x CPU throttle (~mid-range phone) | 2.55s | **1.25s** | **2.97s** | 4.11s | same |

Playable on a phone but not free: the p90 is ~3s against the served agent's
measured worst of 1.28s plus network. The tuning lever if it needs one is
`first_move_prefilter` and `prefilter_top_k` (40 median candidates comes from
top_k=40). ~~the encoder's per-candidate BFS is the likely hot spot, not the
forward pass, so cutting candidates cuts nearly all of it~~ **WRONG — measured
above: the forward pass is 88% and the encoder 2%.** Cutting candidates still
cuts the forward pass (it is scoring fewer of them), but the BFS saving that
made it look cheap is not there. Do NOT lower these blind -- they change which
moves the net sees, and match_prefilter.py is the tool for showing that a change
costs no strength.

**The design, settled:**
- **Both platforms**, not phone-only. Leaving desktop on `/select_moves` keeps
  the Koyeb instance, the cold starts and MOVE_BUDGET alive and maintains two AI
  paths forever, which is the whole cost the port was meant to remove.
- **Lazy, with the server as fallback.** Start each session on `/select_moves`,
  load onnxruntime in the background, switch to local once ready. A first-time
  browser visitor would otherwise pay ~4.5 MB (2.9 MB gzipped runtime + 1.66 MB
  model) before the AI can move; an installed app has already paid it. If local
  init fails -- old browser, no WASM -- it simply never switches.
- Koyeb then becomes a fallback that can be switched off when confident, rather
  than a dependency.

**Two things easy to get wrong:**
- `game.js` builds the snapshot from its OWN board model, which is the one place
  the two engines meet -- and it is the model that deliberately diverges on rack
  reordering. The snapshot must describe what the ENGINE would see, not what the
  UI offers.
- `model.onnx` is precached cache-first by the service worker, so swapping the
  deployed model stops being a server-side file replace and becomes a deploy
  plus a CACHE bump. engine/heuristic/agent.js are already ALWAYS_FRESH; add
  them and latency/trace pages to the shell list when they go live.
