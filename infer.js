/* Value-network inference in the browser (step 5 of PORTING.md).
 *
 * Wraps onnxruntime-web around the ported encoder, mirroring gnn_backend.py's
 * OnnxBackend so the JS agent can score positions with no server.
 *
 * Threads are OFF deliberately: onnxruntime-web 1.19.x ships only the threaded
 * SIMD wasm, and using its threads needs SharedArrayBuffer, which needs COOP and
 * COEP headers on every response. numThreads = 1 runs the same binary
 * single-threaded and needs no headers at all.
 */
const _enc = (typeof require !== 'undefined') ? require('./encoder.js')
                                              : (typeof window !== 'undefined' ? window : self);

const Infer = (() => {
    let session = null;
    let staticData = null;

    async function init({ ort, modelUrl = 'model.onnx', staticUrl = 'encoder_static.json',
                          wasmPath = '/ort/' } = {}) {
        ort.env.wasm.numThreads = 1;          // see header: no COOP/COEP needed
        // Must be root-absolute (or a full URL): the loader dynamically IMPORTS
        // the .mjs beside the wasm, and a bare relative path is not a valid
        // module specifier -- it fails with "no available backend found", which
        // reads like a missing build rather than a bad path.
        ort.env.wasm.wasmPaths = wasmPath;
        staticData = await (await fetch(staticUrl)).json();
        session = await ort.InferenceSession.create(modelUrl, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all',
        });
        return { inputs: session.inputNames, outputs: session.outputNames };
    }

    /* Batch several encoded positions into one set of feeds.
     *
     * The edge arrays are the fiddly part and must match gnn_backend.py's
     * _collate exactly: positions are concatenated into one flat graph, so every
     * piece index is offset by b*P and every tile index by b*T. Getting that
     * wrong does not crash -- it silently wires piece features to the wrong
     * tiles, which is why the fixture check compares whole scores.
     */
    function collate(ort, encodedList) {
        const B = encodedList.length;
        const P = encodedList[0].piece_feats.length;
        const T = encodedList[0].tile_feats.length;
        const F = encodedList[0].piece_feats[0].length;
        const TF = encodedList[0].tile_feats[0].length;
        const G = encodedList[0].global_feats.length;

        const tile = new Float32Array(B * T * TF);
        const piece = new Float32Array(B * P * F);
        const glob = new Float32Array(B * G);
        const p2tS = [], p2tD = [], t2pS = [], t2pD = [];

        encodedList.forEach((e, b) => {
            for (let i = 0; i < T; i++)
                for (let j = 0; j < TF; j++) tile[(b * T + i) * TF + j] = e.tile_feats[i][j];
            for (let i = 0; i < P; i++)
                for (let j = 0; j < F; j++) piece[(b * P + i) * F + j] = e.piece_feats[i][j];
            for (let j = 0; j < G; j++) glob[b * G + j] = e.global_feats[j];
            const [ps, ts] = e.piece_to_tile;
            for (let k = 0; k < ps.length; k++) {
                p2tS.push(BigInt(ps[k] + b * P)); p2tD.push(BigInt(ts[k] + b * T));
                t2pS.push(BigInt(ts[k] + b * T)); t2pD.push(BigInt(ps[k] + b * P));
            }
        });

        const edge = (a, d) => new ort.Tensor('int64',
            BigInt64Array.from(a.concat(d)), [2, a.length]);
        return {
            tile_feats: new ort.Tensor('float32', tile, [B, T, TF]),
            piece_feats: new ort.Tensor('float32', piece, [B, P, F]),
            global_feats: new ort.Tensor('float32', glob, [B, G]),
            piece_to_tile: edge(p2tS, p2tD),
            tile_to_piece: edge(t2pS, t2pD),
        };
    }

    /* Score one snapshot, or a batch of them. Returns a number / array. */
    async function score(ort, snapshots, currentPlayer) {
        const list = Array.isArray(snapshots) ? snapshots : [snapshots];
        const encoded = list.map(s => _enc.encode(staticData, s, currentPlayer || s.current_player));
        const out = await session.run(collate(ort, encoded));
        const values = out[session.outputNames[0]].data;
        return Array.isArray(snapshots) ? Array.from(values) : values[0];
    }

    return { init, score, collate, ready: () => !!session };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = Infer;
else (typeof window !== 'undefined' ? window : self).Infer = Infer;
