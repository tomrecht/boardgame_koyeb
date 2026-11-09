/* Thinking time of the app's agent with the two-sided lookahead on vs off, on
   eg_selfplay.js positions (mover <= 2 unsaved).  node lab/eg_latency.js sp.jsonl [n] */
const fs = require('fs');
const { engineFrom } = require('./eg_common.js');
const { loadNet, choose } = require('./eg_net.js');
(async () => {
    const rows = fs.readFileSync(process.argv[2], 'utf8').trim().split('\n').map(JSON.parse)
        .filter(r => !r.layout).slice(0, +(process.argv[3] || 60));
    const net = await loadNet(null);
    const t = { off: [], on: [] };
    for (const r of rows) {
        for (const [k, self] of [['off', false], ['on', true]]) {
            const e = engineFrom(net.sd, r.state);
            const t0 = process.hrtime.bigint();
            await choose(net, e, { lookaheadSelf: self });
            t[k].push(Number(process.hrtime.bigint() - t0) / 1e6);
        }
    }
    const q = (a, p) => { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))].toFixed(0); };
    for (const k of ['off', 'on']) console.log(k, 'n', t[k].length, 'median', q(t[k], 0.5), 'p90', q(t[k], 0.9), 'max', q(t[k], 0.999), 'ms');
    const extra = t.on.map((v, i) => v - t.off[i]);
    console.log('extra median', q(extra, 0.5), 'p90', q(extra, 0.9), 'max', q(extra, 0.999), 'ms');
})();
