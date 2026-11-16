/* agent.js's close-call search vs close_call_fixture.json (Python close_call.py),
 * with the stubbed net of agent_test.js. Checks the trigger, the deep values of
 * the searched candidates and the chosen pair, at each reply cap.
 *
 *     PYTHONHASHSEED=0 python3 close_call_fixture.py && node close_call_test.js
 */
const fs = require('fs');
const path = require('path');
const ROOT = __dirname;
const { Engine } = require(path.join(ROOT, 'engine.js'));
const A = require(path.join(ROOT, 'agent.js'));
const staticData = JSON.parse(fs.readFileSync(path.join(ROOT, 'encoder_static.json'), 'utf8'));
const W = JSON.parse(fs.readFileSync(path.join(ROOT, 'heuristic_weights.json'), 'utf8'));
const fx = JSON.parse(fs.readFileSync(path.join(ROOT, process.argv[2] || 'close_call_fixture.json'), 'utf8'));

function stubScore(key) {
    let h = 0x811c9dc5;
    for (let i = 0; i < key.length; i++) { h ^= key.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
    return (h / 0x100000000) * 2 - 1;
}
const stub = async (keys) => keys.map(stubScore);
const norm = (m) => JSON.stringify([m.piece, m.piece === null ? m.lone : null, m.dest, m.roll]);
const normPair = (p) => p.map(norm).join(' + ');

(async () => {
    let n = 0, ok = 0, trigOK = 0, deepOK = 0, deepN = 0, trig = 0;
    const fails = [];
    for (const c of fx.cases) {
        for (const r of c.runs) {
            n++;
            const e = Engine.fromState(staticData, c.state);
            const moves = e.getValidMoves();
            const st = {};
            const pair = await A.selectMovePair(e, W, moves, c.player, {
                snapshot: (eng) => A.positionKey(eng), score: stub, scoreFor: () => stub,
                lookahead: false, closeCall: true, closeCallT: c.T, closeCallK: c.K,
                closeCallBudgetMs: Infinity, closeCallMaxReplies: r.cap > 0 ? r.cap : Infinity,
                closeCallStats: st,
            });
            const jsTrig = st.triggered || 0;
            if (jsTrig === r.triggered) trigOK++;
            trig += r.triggered;
            if (r.deep) {
                deepN++;
                const d = st.lastDeep || [];
                if (d.length === r.deep.length && d.every((x, i) => Math.abs(x - r.deep[i]) < 1e-6)) deepOK++;
                else if (fails.length < 6) fails.push({ seed: c.seed, cap: r.cap, what: 'deep', js: d, py: r.deep });
            }
            if (normPair(pair) === normPair(r.chosen)) ok++;
            else if (fails.length < 6) fails.push({ seed: c.seed, cap: r.cap, what: 'pair', js: normPair(pair), py: normPair(r.chosen) });
        }
    }
    console.log(`runs ${n} (positions x caps), triggered in Python ${trig}`);
    console.log(`trigger matches     ${trigOK}/${n}`);
    console.log(`deep values match   ${deepOK}/${deepN}`);
    console.log(`chosen pair matches ${ok}/${n}`);
    if (fails.length) { for (const f of fails) console.log(' ', JSON.stringify(f).slice(0, 600)); process.exit(1); }
    console.log('ALL MATCH');
})();
