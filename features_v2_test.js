/* encoder.js + features_v2.js vs features_v2_fixture.json (encoder.py, branch
 * train-features-v2). Every case is encoded four ways -- v1, A, AB, and AB in
 * AFTERSTATE mode -- and compared value by value (float32 tolerance).
 *
 *     (on train-features-v2) python3 dump_v2_fixture.py 300 300 features_v2_fixture.json
 *     node features_v2_test.js [features_v2_fixture.json]
 */
const fs = require('fs');
const path = require('path');
const E = require(path.join(__dirname, 'encoder.js'));
const staticData = JSON.parse(fs.readFileSync(path.join(__dirname, 'encoder_static.json'), 'utf8'));
const table = JSON.parse(fs.readFileSync(path.join(__dirname, 'endgame_table.json'), 'utf8'));
const scales = JSON.parse(fs.readFileSync(path.join(__dirname, 'input_scales.json'), 'utf8'));
const fx = JSON.parse(fs.readFileSync(process.argv[2] || path.join(__dirname, 'features_v2_fixture.json'), 'utf8'));

const TOL = 2e-6;
function cmp(a, b, where) {
    if (a.length !== b.length) return `${where}: length ${a.length} vs ${b.length}`;
    for (let i = 0; i < a.length; i++) {
        if (Array.isArray(a[i])) {
            const r = cmp(a[i], b[i], `${where}[${i}]`);
            if (r) return r;
        } else if (Math.abs(Math.fround(a[i]) - b[i]) > TOL) {
            return `${where}[${i}]: js ${a[i]} vs py ${b[i]}`;
        }
    }
    return null;
}

const modes = {
    v1: { features: 'v1' },
    A: { features: 'A', table },
    AB: { features: 'AB', table },
    AB_after: { features: 'AB', afterstate: true, table },
    A2: { features: 'A2', table, scales: scales.A2 },
    AB2: { features: 'AB2', table, scales: scales.AB2 },
    AB2_after: { features: 'AB2', afterstate: true, table, scales: scales.AB2 },
};
const stats = {};
const firstFail = {};
let tested = 0;
// a missing scale set must throw, never encode unscaled
let throwOk = false;
try {
    E.encode(staticData, fx.cases[0].snapshot, fx.cases[0].player, { features: 'AB2', table });
    console.log('FAIL: AB2 without scales did not throw');
} catch (e) { throwOk = /input scales/.test(e.message); console.log(`AB2 without scales throws: ${throwOk ? 'ok' : 'FAIL ' + e.message}`); }
for (const c of fx.cases) {
    for (const [mode, opts] of Object.entries(modes)) {
        const exp = c.expected[mode];
        if (!exp) continue;
        const got = E.encode(staticData, c.snapshot, c.player, opts);
        const s = stats[mode] || (stats[mode] = { n: 0, ok: 0, byKind: {} });
        s.n++;
        const k = s.byKind[c.kind] || (s.byKind[c.kind] = [0, 0]);
        k[1]++;
        let err = null;
        if (JSON.stringify(got.piece_order) !== JSON.stringify(exp.piece_order)) err = 'piece_order';
        err = err || cmp(got.tile_feats, exp.tile_feats, 'tile')
                  || cmp(got.piece_feats, exp.piece_feats, 'piece')
                  || cmp(got.global_feats, exp.global_feats, 'global');
        if (err) {
            if (!firstFail[mode]) firstFail[mode] = `${c.kind} case: ${err}`;
        } else {
            s.ok++;
            k[0]++;
        }
        tested++;
    }
}
let all = throwOk;
for (const [mode, s] of Object.entries(stats)) {
    const kinds = Object.entries(s.byKind).map(([k, [ok, n]]) => `${k} ${ok}/${n}`).join(', ');
    console.log(`${mode.padEnd(9)} ${s.ok}/${s.n}  (${kinds})`);
    if (s.ok !== s.n) { all = false; console.log(`   first mismatch: ${firstFail[mode]}`); }
}
console.log(all ? 'ALL MATCH' : 'MISMATCHES');
process.exit(all ? 0 : 1);
