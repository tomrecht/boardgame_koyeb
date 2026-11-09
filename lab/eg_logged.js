/* Judge the moves played in owner's logged small endings with the exact solver.
     node lab/eg_logged.js dump.jsonl [depth=2] > out.jsonl
   One line per turn: the played position's [lo, hi], the best position's,
   whether the played move is PROVEN worse (hi_played < lo_best) and the loss
   range [lo_best - hi_played, hi_best - lo_played] (clipped at 0). */
const fs = require('fs');
const { staticFor, engineFrom, applyRecorded } = require('./eg_common.js');
const { EndgameSolver, esKey } = require('../endgame_solve.js');

// node lab/eg_logged.js dump.jsonl [maxDepth=2] [shard=0] [nShards=1]
// Solves at depth 1, then deeper only while the played move is neither proven
// worse nor proven best (one solver, so the deeper pass reuses the cache).
const all = fs.readFileSync(process.argv[2], 'utf8').trim().split('\n').map(JSON.parse);
const maxDepth = +(process.argv[3] || 2);
const shard = +(process.argv[4] || 0), nShards = +(process.argv[5] || 1);
const rows = all.filter((_, i) => i % nShards === shard);
const sds = {}, fcaches = {};
for (const r of rows) {
    const sd = sds[r.layout || 'today'] || (sds[r.layout || 'today'] = staticFor(r.layout));
    const e = engineFrom(sd, r.state);
    const t0 = Date.now();
    const fc = process.env.NOCACHE ? { get: () => undefined, set: () => {} }
        : (fcaches[r.layout || 'today'] || (fcaches[r.layout || 'today'] = new Map()));
    const S = new EndgameSolver(1, fc);
    S.maxNodes = +(process.env.MAX_NODES || 4000);
    let out = null, ranked = null, playedKey = null, err = null;
    try {
        const n = e.moves.length;
        applyRecorded(e, r.side, r.m);
        playedKey = esKey(e);
        while (e.moves.length > n) e.undoLastMove();
    } catch (x) { err = String(x.message || x); }
    for (let depth = 1; depth <= maxDepth; depth++) {
        S.depth = depth;
        S.nodes = 0;
        const prevPl = ranked && ranked.find(x => x.key === playedKey);
        try {
            ranked = S.rank(e, ranked, prevPl ? (k) => k === playedKey
                || ranked.find(x => x.key === k).hi >= prevPl.lo - 1e-9 : null);
        } catch (x) {
            if (String(x.message) !== 'node budget') throw x;
            if (out) out.budget = depth;
            break;
        }
        ranked.sort((a, b) => (b.lo - a.lo) || (b.hi - a.hi));
        const best = ranked[0];
        const pl = ranked.find(x => x.key === playedKey);
        out = { game: r.game, t: r.t, layout: r.layout || 'today', side: r.side, ai: r.ai,
                notation: r.notation, m: r.m, a: r.a, n_pos: ranked.length, depth,
                best: best && [best.lo, best.hi], err };
        if (!pl) { if (!err) out.err = 'played position not among end positions'; break; }
        out.played = [pl.lo, pl.hi];
        out.proven_worse = ranked.some(x => x.lo > pl.hi + 1e-9);
        out.loss = [Math.max(0, best.lo - pl.hi), Math.max(0, Math.max(...ranked.map(x => x.hi)) - pl.lo)];
        out.proven_best = ranked.every(x => x.key === playedKey || x.hi <= pl.lo + 1e-9);
        if (out.proven_worse || out.proven_best) break;
    }
    out.ms = Date.now() - t0;
    process.stdout.write(JSON.stringify(out) + '\n');
}
