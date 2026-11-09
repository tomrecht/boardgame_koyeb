/* Shared by the endgame lab scripts: static data for a goal layout (patched as
   local_agent.js's applyGoalLayout does) and an engine built from a dumped state
   the way local_agent.js's engineFor does. */
const fs = require('fs');
const path = require('path');
const ROOT = path.join(__dirname, '..');
const { Engine } = require(path.join(ROOT, 'engine.js'));

const LAYOUTS = { sum7: [1, 6, 2, 5, 3, 4], B: [1, 4, 5, 2, 3, 6] };

function staticFor(layout) {
    const sd = JSON.parse(fs.readFileSync(path.join(ROOT, 'encoder_static.json'), 'utf8'));
    const L = layout && LAYOUTS[layout];
    if (L) {
        const SECTORS = [2, 4, 6, 8, 10, 12];
        sd.tile_types.forEach((ty, i) => {
            if (ty !== 'save') return;
            const n = L[SECTORS.indexOf(sd.tile_coords[i][1])];
            sd.tile_numbers[i] = n;
            sd.base_tile_feats[i][6] = n / 6;
        });
    }
    return sd;
}

function engineFrom(sd, st) {
    const e = Engine.fromState(sd, st);
    e.stages.white = e.getGameStage('white');
    e.stages.black = e.getGameStage('black');
    e.applyLastPieceRule();
    return e;
}

/* Apply a recorded turn ("7>5.4", "7>s", "o7>b") by matching legal moves.
   Blanks are matched by number first, then any same-side blank on the tile the
   named one would leave from is impossible to know, so by number only. */
function applyRecorded(e, side, ms) {
    for (const s of ms) {
        const opp = s.startsWith('o');
        const body = opp ? s.slice(1) : s;
        const [num, dest] = body.split('>');
        const player = opp ? (side === 'white' ? 'black' : 'white') : side;
        const moves = e.getValidMoves().filter(m => m.piece && m.piece[0] === player);
        let want;
        if (dest === 's') want = moves.filter(m => m.dest === 'save');
        else if (dest === 'b') want = moves.filter(m => m.dest === 0);
        else {
            const [r, c] = dest.split('.').map(Number);
            want = moves.filter(m => Array.isArray(m.dest) && m.dest[0] === r && m.dest[1] === c);
        }
        let hit = want.find(m => m.piece[1] === +num);
        if (!hit) {
            // the last-piece rule renumbers a lone piece to 13; a blank may be
            // deduplicated under another blank's number on the same tile
            const p = e.find(player, +num) || e.find(player, 13);
            hit = want.find(m => { const q = e.find(m.piece[0], m.piece[1]);
                                   return q && p && q.tile === p.tile && q.number > 6 && p.number > 6; })
                || (want.length === 1 ? want[0] : null);
        }
        if (!hit) throw new Error('cannot apply ' + s + ' (' + want.length + ' candidates)');
        e.applyMove(hit, false);
    }
}

module.exports = { ROOT, staticFor, engineFrom, applyRecorded, LAYOUTS };
