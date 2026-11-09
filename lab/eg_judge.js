/* Judge old vs new lookahead choices from eg_selfplay.js with the exact solver.
     node lab/eg_judge.js sp.jsonl [maxDepth=2] [shard=0] [nShards=1]
   Only rows where the two versions chose different end positions. Deepens until
   one is proven better than the other or maxDepth. diff = new - old, as an
   interval [lo, hi] in points (lo > 0: new proven better). */
const fs = require('fs');
const { staticFor, engineFrom } = require('./eg_common.js');
const { EndgameSolver, esKey } = require('../endgame_solve.js');
const A = require('../agent.js');

const rows = fs.readFileSync(process.argv[2], 'utf8').trim().split('\n').map(JSON.parse)
    .filter(r => !r.same);
const maxDepth = +(process.argv[3] || 2), shard = +(process.argv[4] || 0), nShards = +(process.argv[5] || 1);
const sds = {}, fcs = {};
function endKey(e, pair) {
    const n = e.moves.length;
    for (const m of pair) if (!A.isPass(m) && !A.isDraw(m)) e.applyMove(m, false);
    const k = esKey(e);
    while (e.moves.length > n) e.undoLastMove();
    return k;
}
rows.filter((_, i) => i % nShards === shard).forEach(r => {
    const L = r.layout || 'today';
    const sd = sds[L] || (sds[L] = staticFor(r.layout));
    const e = engineFrom(sd, r.state);
    const S = new EndgameSolver(1, fcs[L] || (fcs[L] = new Map()));
    const kOld = endKey(e, r.old), kNew = endKey(e, r.new);
    const t0 = Date.now();
    S.maxNodes = +(process.env.MAX_NODES || 4000);
    let out, ranked = null;
    for (let d = 1; d <= maxDepth; d++) {
        S.depth = d; S.nodes = 0;
        try { ranked = S.rank(e, ranked, ranked ? (k) => k === kOld || k === kNew : null); }
        catch (x) { if (String(x.message) !== 'node budget') throw x; if (out) out.budget = d; break; }
        ranked.sort((a, b) => (b.lo - a.lo) || (b.hi - a.hi));
        const o = ranked.find(x => x.key === kOld), n = ranked.find(x => x.key === kNew), b = ranked[0];
        out = { game: r.game, k: r.k, turn: r.turn, layout: L, side: r.side, mine: r.mine, theirs: r.theirs,
                depth: d, old: o && [o.lo, o.hi], new: n && [n.lo, n.hi], best: [b.lo, b.hi] };
        if (!o || !n) { out.err = 'pair not among end positions'; break; }
        out.diff = [n.lo - o.hi, n.hi - o.lo];
        if (out.diff[0] > 1e-9 || out.diff[1] < -1e-9) break;      // resolved
    }
    out.ms = Date.now() - t0;
    process.stdout.write(JSON.stringify(out) + '\n');
});
