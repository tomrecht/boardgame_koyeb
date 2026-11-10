/* engine.js vs engine_fixture.json (dump_engine_fixture.py): for every position,
 * the SET of legal moves, the position after each move, and that undo restores
 * the start. Run from the repo root:
 *     python3 dump_engine_fixture.py 60 && node ci/engine_test.js
 * Exit code 1 on any mismatch. */
const fs = require('fs');
const path = require('path');
const ROOT = path.join(__dirname, '..');
const { Engine, moveCmp } = require(path.join(ROOT, 'engine.js'));
const staticData = JSON.parse(fs.readFileSync(path.join(ROOT, 'encoder_static.json'), 'utf8'));
const fx = JSON.parse(fs.readFileSync(process.argv[2] || path.join(ROOT, 'engine_fixture.json'), 'utf8'));

// Key-sorted JSON, so object key order never decides equality.
const canon = (v) => Array.isArray(v) ? '[' + v.map(canon).join(',') + ']'
    : (v && typeof v === 'object') ? '{' + Object.keys(v).sort().map(k => JSON.stringify(k) + ':' + canon(v[k])).join(',') + '}'
    : JSON.stringify(v);
const mkey = (m) => canon({ piece: m.piece, lone: m.piece ? null : m.lone, dest: m.dest, roll: m.roll });

let positions = 0, moves = 0, bad = 0;
const fail = (c, msg) => { bad++; if (bad <= 10) console.log(`MISMATCH seed ${c.seed} (${c.kind}): ${msg}`); };
for (const c of fx.cases) {
    positions++;
    const e = Engine.fromState(staticData, c.state);
    const got = e.getValidMoves().slice().sort(moveCmp).map(mkey);
    const want = c.after.map(a => mkey(a.move));
    if (got.join('|') !== [...want].sort().join('|') && got.slice().sort().join('|') !== [...want].sort().join('|')) {
        fail(c, `move set differs: JS ${got.length}, Python ${want.length}`);
        continue;
    }
    const start = canon(e.fingerprint());
    const live = e.getValidMoves();
    for (const a of c.after) {
        moves++;
        const m = live.find(x => mkey(x) === mkey(a.move));
        const base = e.moves.length;
        e.applyMove(m, false);
        if (canon(e.fingerprint()) !== canon(a.result)) fail(c, `position after ${mkey(a.move)} differs`);
        while (e.moves.length > base) e.undoLastMove();
        if (m.piece === null && m.lone === 1) e.drawCalled = false;
        if (canon(e.fingerprint()) !== start) fail(c, `undo of ${mkey(a.move)} did not restore the start`);
    }
}
console.log(`engine: ${positions} positions, ${moves} moves, ${bad} mismatches`);
process.exit(bad ? 1 : 0);
