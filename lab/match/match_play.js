/* Whole matches in node, the app's agent on both sides (shipped net for the
 * layout, full strength): side A plays MATCH-AWARE (match.js), side B plain.
 * Each match index is played twice with the same dice, A as white then as
 * black, so luck cancels in the pair.
 *     ORT_NODE=<dir with onnxruntime-node> node lab/match/match_play.js model.json out.jsonl
 *         <matches> <seed_base> <mode games|race> <target> [layout=sum7]
 * One line per match: {seed, aWhite, winner: 'A'|'B'|'draw', games, scoreA, scoreB, decisions, changed}.
 * `changed` = turns where A's match-aware pick differed from its plain pick. */
const fs = require('fs');
const path = require('path');
const { ROOT, staticFor } = require('../eg_common.js');
const { Engine } = require(path.join(ROOT, 'engine.js'));
const A = require(path.join(ROOT, 'agent.js'));
const M = require(path.join(ROOT, 'match.js'));
const { loadNet, W } = require('../eg_net.js');

const [modelPath, outp, nStr, baseStr, mode = 'games', targetStr = '6', layout = 'sum7'] = process.argv.slice(2);
const model = JSON.parse(fs.readFileSync(modelPath, 'utf8'));
const N = +nStr, BASE = +baseStr, TARGET = +targetStr;
const MAX_TURNS = 300;

function rng(seed) {            // mulberry32
    return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0;
        let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}
const unsaved = (e, p) => 12 - e.savedRack(p).length;

async function choosePair(net, e, matchState, aware) {
    const player = e.currentPlayer;
    const moves = e.getValidMoves();
    if (!moves.length) return { pair: null };
    const base = { difficulty: 1.0, snapshot: (eng) => eng.snapshot(),
                   score: async (snaps) => net.score(snaps, player) };
    const plain = await A.selectMovePair(e, W, moves, player, base);
    if (!aware) return { pair: plain };
    const ranked = await A.selectMovePair(e, W, e.getValidMoves(), player, Object.assign({}, base, { returnScores: true }));
    if (!Array.isArray(ranked) || !ranked.length || !ranked[0].pair) return { pair: plain };
    const phases = ranked.map(r => {
        const root = e.moves.length;
        for (const m of r.pair) if (!A.isPass(m) && !A.isDraw(m)) e.applyMove(m, false);
        const ph = unsaved(e, 'white') + unsaved(e, 'black');
        while (e.moves.length > root) e.undoLastMove();
        return ph;
    });
    const eq = M.equityTable(matchState, player, model);
    const i = M.chooseByEquity(ranked, eq, model, phases, A.SCORE_SCALE, (p) => A.isDraw(p[0]));
    const pair = A.dedupeSavePair(ranked[i].pair);
    const key = (p) => JSON.stringify(p);
    return { pair, changed: key(pair) !== key(plain) };
}

async function playGame(net, sd, R, starter, matchState, awareColor, stats) {
    const order = () => { const a = [...Array(12)].map((_, i) => i + 1);
        for (let i = 11; i > 0; i--) { const j = Math.floor(R() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
    const st = { racks: { white_unentered: order(), white_saved: [], black_unentered: order(), black_saved: [] },
                 occupancy: {}, dice: [[1, false], [1, false]], current_player: starter,
                 stages: { white: 'opening', black: 'opening' }, first_move: null };
    const e = Engine.fromState(sd, st);
    for (let t = 0; t < MAX_TURNS; t++) {
        const [w, s] = e.checkGameOver();
        if (w) return [w, s];
        e.dice = [{ value: 1 + Math.floor(R() * 6), used: false }, { value: 1 + Math.floor(R() * 6), used: false }];
        e.firstMove = null;
        const aware = e.currentPlayer === awareColor;
        const r = await choosePair(net, e, matchState, aware);
        if (aware) { stats.decisions++; if (r.changed) stats.changed++; }
        if (r.pair) for (const m of r.pair) if (!A.isPass(m) && !A.isDraw(m)) e.applyMove(m, false);
        e.switchTurn();
    }
    return ['draw', 0];
}

async function playMatch(net, sd, idx, aWhite) {
    const R = rng(BASE + idx);
    const m = { mode, target: TARGET, tieRule: 'extra', gamesPlayed: 0, whiteScore: 0, blackScore: 0,
                whiteWins: 0, blackWins: 0, draws: 0, firstStarter: R() < 0.5 ? 'white' : 'black', over: false };
    const awareColor = aWhite ? 'white' : 'black';
    const stats = { decisions: 0, changed: 0 };
    while (!m.over && m.gamesPlayed < 60) {
        const starter = (m.gamesPlayed % 2 === 0) ? m.firstStarter : (m.firstStarter === 'white' ? 'black' : 'white');
        const [w, s] = await playGame(net, sd, R, starter, m, awareColor, stats);
        m.gamesPlayed++;
        if (w === 'white') { m.whiteScore += s; m.whiteWins++; } else if (w === 'black') { m.blackScore += s; m.blackWins++; } else m.draws++;
        if (mode === 'race') {
            if (m.whiteScore >= TARGET) { m.over = true; m.winner = 'white'; }
            else if (m.blackScore >= TARGET) { m.over = true; m.winner = 'black'; }
        } else if (m.gamesPlayed >= m.target) {
            const d = m.whiteScore - m.blackScore, wd = m.whiteWins - m.blackWins;
            if (d || wd) { m.over = true; m.winner = (d > 0 || (!d && wd > 0)) ? 'white' : 'black'; }
            else m.target += 2;                     // tieRule 'extra'
        }
    }
    const win = !m.over ? 'draw' : (m.winner === awareColor ? 'A' : 'B');
    return { seed: BASE + idx, aWhite, winner: win, games: m.gamesPlayed,
             scoreA: aWhite ? m.whiteScore : m.blackScore, scoreB: aWhite ? m.blackScore : m.whiteScore, ...stats };
}

(async () => {
    const net = await loadNet(layout);
    const sd = staticFor(layout);
    const done = new Set();
    if (fs.existsSync(outp)) for (const l of fs.readFileSync(outp, 'utf8').split('\n')) if (l) { const r = JSON.parse(l); done.add(r.seed + ':' + r.aWhite); }
    for (let i = 0; i < N; i++) {
        for (const aWhite of [true, false]) {
            if (done.has((BASE + i) + ':' + aWhite)) continue;
            const t0 = Date.now();
            const r = await playMatch(net, sd, i, aWhite);
            r.secs = Math.round((Date.now() - t0) / 1000);
            fs.appendFileSync(outp, JSON.stringify(r) + '\n');
            console.log(JSON.stringify(r));
        }
    }
})();
