/* agent.js vs agent_fixture.json, with a STUBBED value net.
 *
 * The cheap half of the port's verification (PORTING.md step 6.3). It checks the
 * SEARCH -- candidate sets and the chosen pair, including every tie-break -- and
 * runs in plain node in seconds, because the stub replaces inference with a hash
 * of the position. That makes it deterministic and browser-free, so it is the
 * one to run after touching select_move_pair.
 *
 * It does NOT exercise real model scores, so it cannot see a near-tie decided by
 * float noise; trace_test.html is the expensive half that does (real model, real
 * browser, 110/110).
 *
 * The fixture encodes PYTHON's choices, so regenerate it FIRST after any change
 * to agent_gnn.py or this compares new JS against old Python:
 *
 *     PYTHONHASHSEED=0 python dump_agent_fixture.py 40
 *     node agent_test.js
 */
const fs = require('fs');
const path = require('path');
// Repo-relative: this file lives in the repo root, and the owner works from two
// machines with different absolute paths.
const ROOT = __dirname;
const { Engine } = require(path.join(ROOT, 'engine.js'));
const A = require(path.join(ROOT, 'agent.js'));

const staticData = JSON.parse(fs.readFileSync(path.join(ROOT, 'encoder_static.json'), 'utf8'));
const W = JSON.parse(fs.readFileSync(path.join(ROOT, 'heuristic_weights.json'), 'utf8'));
const fx = JSON.parse(fs.readFileSync(path.join(ROOT, 'agent_fixture.json'), 'utf8'));

// The same stub: FNV-1a over the position key, into [-1, 1).
function stubScore(key) {
    let h = 0x811c9dc5;
    for (let i = 0; i < key.length; i++) {
        h ^= key.charCodeAt(i);
        h = Math.imul(h, 0x01000193) >>> 0;
    }
    return (h / 0x100000000) * 2 - 1;
}

const norm = (m) => JSON.stringify([m.piece, m.piece === null ? m.lone : null, m.dest, m.roll]);
const normPair = (p) => p.map(norm).join(' + ');

let ok = 0, n = 0, movesOK = 0;
const fails = [];

(async () => {
for (const c of fx.cases) {
    n++;
    const e = Engine.fromState(staticData, c.state);
    const moves = e.getValidMoves();
    // Sanity: the search must be choosing from the same candidate set.
    const gotMoves = moves.map(norm).sort().join('|');
    const wantMoves = c.moves.map(norm).sort().join('|');
    if (gotMoves === wantMoves) movesOK++;

    const pair = await A.selectMovePair(e, W, moves, c.player, {
        snapshot: (eng) => A.positionKey(eng),
        score: async (keys) => keys.map(stubScore),
    });
    const got = normPair(pair), want = normPair(c.chosen);
    if (got === want) ok++;
    else if (fails.length < 6) {
        // Rank both choices under the fixture's own scoring, to say whether the
        // port scored differently or merely broke a tie differently.
        const ranked = await A.selectMovePair(e, W, moves, c.player, {
            snapshot: (eng) => A.positionKey(eng),
            score: async (keys) => keys.map(stubScore),
            returnScores: true,
        });
        const find = (label) => {
            const hit = ranked.find(r => normPair(r.pair) === label);
            return hit ? hit.score : null;
        };
        fails.push({ seed: c.seed, kind: c.kind, nMoves: moves.length,
                     js: got, py: want,
                     jsScore: find(got), pyScore: find(want),
                     pyTop: c.top.slice(0, 2).map(t => ({ s: t.score, p: normPair(t.pair) })),
                     jsTop: ranked.slice(0, 2).map(t => ({ s: t.score, p: normPair(t.pair) })) });
    }
}

console.log(`positions            ${n}`);
console.log(`candidate sets match ${movesOK}/${n}`);
console.log(`chosen pair matches  ${ok}/${n}`);
if (fails.length) {
    console.log('\nfirst mismatches:');
    for (const f of fails) console.log(' ', JSON.stringify(f).slice(0, 800));
    process.exit(1);
}
console.log('\nALL MATCH');
})();
