/* match.js sanity on the shipped outcome model: equity is a probability, both
 * players' chances sum to 1, a match already won is 1, a bigger win is never
 * worse, and an even games-match with alternating starters is exactly fair. */
const fs = require('fs');
const path = require('path');
const M = require(path.join(__dirname, '..', 'match.js'));
const model = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'match_model_sum7.json'), 'utf8'));
let bad = 0;
const check = (ok, msg) => { if (!ok) { bad++; console.log('FAIL', msg); } };
const close = (a, b) => Math.abs(a - b) < 1e-9;
const g = M.gamesValue(model, 'extra');
check(close(g(6, 0, 0, 1) + g(6, 0, 0, 0), 1), 'games: chances sum to 1');
check(close(g(6, 0, 0, 1), 0.5), 'games: 6 alternating games are fair');
const r = M.raceValue(model, 21);
check(close(r(0, 0, 1) + r(0, 0, 0), 1), 'race: chances sum to 1');
check(r(0, 0, 1) > 0.5, 'race: starting first is an edge');
for (const mode of ['games', 'race']) {
    for (const [ws, bs, gp] of [[0, 0, 0], [5, 9, 3], [17, 19, 6], [20, 3, 4]]) {
        const match = { mode, target: mode === 'race' ? 21 : 6, tieRule: 'extra', gamesPlayed: Math.min(gp, 5),
                        whiteScore: ws, blackScore: bs, whiteWins: 1, blackWins: 2, firstStarter: 'white' };
        const ew = M.equityTable(match, 'white', model), eb = M.equityTable(match, 'black', model);
        let prev = -1;
        for (let m = -12; m <= 12; m++) {
            const e = ew.get(m);
            check(e >= -1e-12 && e <= 1 + 1e-12, `${mode} ${ws}-${bs}: equity in [0,1] at ${m}`);
            check(e >= prev - 1e-12, `${mode} ${ws}-${bs}: monotone in the margin at ${m}`);
            check(close(e + eb.get(-m), 1), `${mode} ${ws}-${bs}: white + black = 1 at ${m}`);
            prev = e;
        }
    }
}
const d = M.marginDist(model, 1.3, 14);
check(Math.abs(d.reduce((a, b) => a + b, 0) - 1) < 1e-5, 'margin distribution sums to 1 (table stored to 6 decimals)');
// A straight-line equity (no match pressure) must reproduce the plain choice:
// the candidate with the highest expected margin, whatever the phases.
{
    let seed = 7; const R = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    let diffs = 0;
    for (let t = 0; t < 2000; t++) {
        const n = 2 + Math.floor(R() * 30);
        const ranked = [...Array(n)].map(() => ({ score: (R() * 8 - 4) / 12 * 1000, pair: null }));
        const phases = ranked.map(() => Math.floor(R() * 25));
        const eq = new Map(); for (let m = -12; m <= 12; m++) eq.set(m, 0.5 + 0.01 * m);
        const i = M.chooseByEquity(ranked, eq, model, phases, 1000);
        const best = ranked.reduce((b, r, k) => r.score > ranked[b].score ? k : b, 0);
        if (i !== best) diffs++;
    }
    check(diffs === 0, `linear equity changed the plain choice in ${diffs}/2000`);
}
console.log(bad ? `match.js: ${bad} failures` : 'match.js: all checks pass');
process.exit(bad ? 1 : 0);
