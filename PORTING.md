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
| 2-ply search + heuristic prefilter | `agent_gnn.py` | ~400 lines of the 1452 | `select_move_pair` + `first_move_prefilter` |

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

   **6.2 the heuristic** (`agent.py`, ~414 lines) — needed because the served
   config prefilters (`FIRST_MOVE_PREFILTER=12`), so the candidate set the GNN
   sees depends on it.

   **6.3 `select_move_pair`** itself (~290 lines) plus `_select_filtered`,
   `_dedupe_save_pair` and `_pick_move_index`. `_fix_never_good` is disabled by
   default (`enable_never_good=False`) and can be skipped.

   **6.4 the proof.** With 6.0 in place the criterion is exact: identical chosen
   pairs over a 40-turn seeded game. Compare against the SERVED config (prefilter
   on, F=12), not the library default of 0.

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
