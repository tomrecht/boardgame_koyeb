/* Re-decide eg_selfplay.js positions with the CURRENT agent.js (new = two-sided,
   old = opponent-only), so a changed formula needs no new games.
     node lab/eg_rescore.js sp.jsonl out.jsonl [shard=0] [nShards=1] */
const fs = require('fs');
const { engineFrom } = require('./eg_common.js');
const { loadNet, choose } = require('./eg_net.js');
const { esKey } = require('../endgame_solve.js');
const A = require('../agent.js');
const [inp, outp] = process.argv.slice(2, 4);
const shard = +(process.argv[4] || 0), nShards = +(process.argv[5] || 1);
function endKey(e, pair) {
    const n = e.moves.length;
    for (const m of pair) if (!A.isPass(m) && !A.isDraw(m)) e.applyMove(m, false);
    const k = esKey(e);
    while (e.moves.length > n) e.undoLastMove();
    return k;
}
(async () => {
    const rows = fs.readFileSync(inp, 'utf8').trim().split('\n').map(JSON.parse).filter((_, i) => i % nShards === shard);
    const nets = {};
    const out = fs.createWriteStream(outp);
    let diff = 0;
    for (const r of rows) {
        const L = r.layout || 'today';
        const net = nets[L] || (nets[L] = await loadNet(r.layout || null));
        const e = engineFrom(net.sd, r.state);
        const oldP = await choose(net, e, { lookaheadSelf: false });
        const newP = await choose(net, e, { lookaheadSelf: true });
        const same = !oldP || !newP || endKey(e, oldP) === endKey(e, newP);
        if (!same) diff++;
        out.write(JSON.stringify(Object.assign({}, r, { old: oldP, new: newP, same })) + '\n');
    }
    out.end();
    console.error('rows', rows.length, 'differ', diff);
})();
