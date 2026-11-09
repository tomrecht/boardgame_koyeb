/* Self-play endings for judging the two-sided lookahead.
     node lab/eg_selfplay.js starts.jsonl out.jsonl [playouts=2] [shard=0] [nShards=1]
   starts.jsonl = endgame_dump.py output; the FIRST row of each game is used
   (both sides <= its LIMIT unsaved). Each start is played out `playouts` times
   with fresh dice by the app's agent (current lookahead). On every turn where a
   mover has <= 2 unsaved (the only turns the two-sided part acts on) it records the position and the
   pair each version would play: old = opponent-only lookahead, new = two-sided. */
const fs = require('fs');
const { engineFrom } = require('./eg_common.js');
const { loadNet, choose } = require('./eg_net.js');
const { esKey } = require('../endgame_solve.js');
const A = require('../agent.js');

const [inp, outp] = process.argv.slice(2, 4);
const playouts = +(process.argv[4] || 2), shard = +(process.argv[5] || 0), nShards = +(process.argv[6] || 1);

function rng(seed) {            // mulberry32
    return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0;
        let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}
function stateOf(e) {
    const f = e.fingerprint();
    return { racks: f.racks, occupancy: f.occupancy, dice: f.dice, current_player: f.current_player,
             stages: f.stages, first_move: null, draw_called: false,
             draw_callable: !!e.drawCallable, no_save_turns: e.noSaveTurns || 0 };
}
function endKey(e, pair) {
    const n = e.moves.length;
    for (const m of pair) if (!A.isPass(m) && !A.isDraw(m)) e.applyMove(m, false);
    const k = esKey(e);
    while (e.moves.length > n) e.undoLastMove();
    return k;
}
const unsaved = (e, s) => 12 - e.savedRack(s).length;

(async () => {
    const rows = fs.readFileSync(inp, 'utf8').trim().split('\n').map(JSON.parse);
    const firsts = [], seen = new Set();
    for (const r of rows) if (!seen.has(r.game)) { seen.add(r.game); firsts.push(r); }
    const mine = firsts.filter((_, i) => i % nShards === shard);
    const nets = {};
    const out = fs.createWriteStream(outp, { flags: 'a' });
    let nTrig = 0, nDiff = 0;
    for (const r of mine) {
        const layout = r.layout || null;
        const net = nets[layout || 'today'] || (nets[layout || 'today'] = await loadNet(layout));
        for (let k = 0; k < playouts; k++) {
            const rand = rng(Number.parseInt(r.game.slice(0, 8), 16) + 7919 * k);
            const e = engineFrom(net.sd, r.state);
            if (k > 0) e.dice.forEach(d => { d.value = 1 + Math.floor(rand() * 6); d.used = false; });
            for (let turn = 0; turn < 60; turn++) {
                if (e.checkGameOver()[0]) break;
                const me = e.currentPlayer, opp = me === 'white' ? 'black' : 'white';
                let pair;
                if (unsaved(e, me) <= 2) {          // the only turns where the two versions can differ
                    const st = stateOf(e);
                    const oldP = await choose(net, e, { lookaheadSelf: false });
                    const newP = await choose(net, e, { lookaheadSelf: true });
                    const same = !oldP || !newP || endKey(e, oldP) === endKey(e, newP);
                    nTrig++; if (!same) nDiff++;
                    out.write(JSON.stringify({ game: r.game, k, turn, layout, side: me,
                        mine: unsaved(e, me), theirs: unsaved(e, opp), same,
                        old: oldP, new: newP, state: st }) + '\n');
                    pair = oldP;
                } else pair = await choose(net, e);
                if (pair) {
                    if (pair.some(A.isDraw)) break;
                    for (const m of pair) if (!A.isPass(m)) e.applyMove(m, false);
                }
                if (e.checkGameOver()[0]) break;
                e.switchTurn();
                e.stages.white = e.getGameStage('white'); e.stages.black = e.getGameStage('black');
                e.dice.forEach(d => { d.value = 1 + Math.floor(rand() * 6); d.used = false; });
            }
        }
    }
    out.end();
    console.error('trigger turns', nTrig, 'differ', nDiff);
})();
