/* The shipped value net in node (onnxruntime-node), wired exactly as
   local_agent.js wires it in the page: the layout's own net when one is shipped
   (model_<layout>.json sidecar), its endgame table, the patched static data.
   ORT_NODE = path to an onnxruntime-node install (default /home/user/node_ort). */
const fs = require('fs');
const path = require('path');
const { ROOT, staticFor } = require('./eg_common.js');
const ort = require(path.join(process.env.ORT_NODE || '/home/user/node_ort', 'node_modules/onnxruntime-node'));
const enc = require(path.join(ROOT, 'encoder.js'));
const Infer = require(path.join(ROOT, 'infer.js'));
const A = require(path.join(ROOT, 'agent.js'));

async function loadNet(layout) {
    const side = layout && fs.existsSync(path.join(ROOT, 'model_' + layout + '.json'))
        ? 'model_' + layout + '.json' : 'model.json';
    const meta = JSON.parse(fs.readFileSync(path.join(ROOT, side), 'utf8'));
    const sd = staticFor(layout);
    const table = JSON.parse(fs.readFileSync(path.join(ROOT,
        layout ? 'endgame_table_' + layout + '.json' : 'endgame_table.json'), 'utf8'));
    const opts = { features: meta.features, afterstate: !!meta.afterstate, table,
                   scales: meta.input_scales || null };
    const session = await ort.InferenceSession.create(path.join(ROOT, meta.onnx));
    async function score(snaps, player) {
        const encoded = snaps.map(s => enc.encode(sd, s, player || s.current_player, opts));
        const out = await session.run(Infer.collate(ort, encoded));
        return Array.from(out[session.outputNames[0]].data);
    }
    return { sd, meta, score, source: meta.source };
}

const W = JSON.parse(fs.readFileSync(path.join(ROOT, 'heuristic_weights.json'), 'utf8'));

/* The app's own choice for the side to move (local_agent.selectMoves). */
async function choose(net, e, opts = {}) {
    const player = e.currentPlayer;
    const moves = e.getValidMoves();
    if (!moves.length) return null;
    return A.selectMovePair(e, W, moves, player, Object.assign({
        difficulty: 1.0,
        snapshot: (eng) => eng.snapshot(),
        score: async (snaps) => net.score(snaps, player),
    }, opts));
}

module.exports = { loadNet, choose, W };
