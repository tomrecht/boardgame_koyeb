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
2. Fixture: `dump_encoder_fixture.py` writes N seeded positions as
   `{snapshot, expected:{tile_feats, piece_feats, global_feats, p2t, t2p}}`.
3. Port the BFS trio; verify against the fixture's distances alone first.
4. Port dynamic tile features → piece features → global features → edges, each
   asserted array-equal against the fixture.
5. Wire onnxruntime-web; assert the ONNX output matches Python's scores for the
   same positions (the existing `onnx_export.py` self-check is the model here:
   it got graph-vs-torch agreement to 8.8e-08).
6. Port the prefilter + 2-ply search; then the real proof:
   **a seeded trace-diff** — `PYTHONHASHSEED=0`, identical move traces over 40
   turns against the Python agent, the standard already used for the game-logic
   optimisation and the ONNX refactor.

## Watch-outs carried over

- Self-play is **not** run-to-run deterministic unless `PYTHONHASHSEED` is
  pinned: transposed-identical candidates score identically and the argmax
  tie-break follows `set` iteration order. Pin it for any comparison.
- The served prefilter is `FIRST_MOVE_PREFILTER=12`; the library default is 0.
  Match whichever you are comparing against, or the traces will diverge for
  reasons that have nothing to do with the port.
- `agent_gnn` does top-k/argmax/difficulty-softmax in numpy already
  (`gnn_backend.py`), so those are small and mechanical.
