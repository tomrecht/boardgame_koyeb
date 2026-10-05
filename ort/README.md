# onnxruntime-web 1.19.2 (vendored)

`npm pack onnxruntime-web@1.19.2`, three files copied from `package/dist`:

| file | size | why |
|---|---|---|
| `ort.wasm.min.js` | 0.3 MB | the loader; the `.wasm`-only build, not `ort.all` |
| `ort-wasm-simd-threaded.wasm` | 10.5 MB (2.8 MB gzipped) | the runtime |
| `ort-wasm-simd-threaded.mjs` | 0.2 MB | glue the loader fetches alongside the wasm |

Vendored rather than CDN-loaded for the same reasons as `phaser.min.js`: a CDN
breaks the service worker's offline guarantee, cannot ship in a store build, and
costs real stack traces.

**1.19.x ships only the threaded SIMD wasm** — there is no separate
single-threaded build to pick. Run it single-threaded with
`ort.env.wasm.numThreads = 1`, which avoids needing COOP/COEP headers for
`SharedArrayBuffer`. Do not switch to threads without adding those headers.

**Serve it compressed.** 10.5 MB raw versus 2.8 MB gzipped is the whole
difference between an annoying first load and an unremarkable one, and Flask
does not compress by default.

To upgrade: re-run `npm pack`, copy the same three files, and re-run the
score-parity check against `encoder_fixture.json` — the fixture stores Python's
score per position precisely so a runtime upgrade can be verified rather than
assumed.
