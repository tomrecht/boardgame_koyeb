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
