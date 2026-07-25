# Durable TODO

Parked, deliberately-not-now items. (Active work lives in CLAUDE.md / OVERNIGHT_NOTES.md.)

## Deployment / hosting

- [ ] **ONNX export for cheap hosting.** Export the champion GNN to ONNX and serve
  with `onnxruntime` instead of full PyTorch. Motivation: torch CPU is ~200 MB on
  disk / ~250–400 MB resident (the blocker for small free tiers), while the model
  itself is ~1.6 MB; `onnxruntime` is ~40–60 MB, lower RAM, faster CPU inference,
  and the 2-ply search already batches forward passes so it maps onto a single
  `session.run`. Deliverable: one-time export + a thin `onnxruntime` inference path
  parallel to `agent_gnn`, keeping torch as the training/dev path. Gets serving
  comfortably inside a 512 MB instance (Koyeb bumped one size, Fly, or Cloud Run)
  and cuts cold starts. Also useful locally (faster moves, lower memory).
  See OVERNIGHT_NOTES.md §C for the full host comparison.

## Frontend / UX

- [ ] **Tutorial step-runner** — scripted mini-positions with an instruction
  bubble and input gating (see OVERNIGHT_NOTES.md §E for the design + suggested
  8 steps). *(In progress separately.)*
