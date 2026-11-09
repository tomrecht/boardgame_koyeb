/* endgame_solve.js -- exact search for small endings (owner, 2026-11-09).
 *
 * Owner's game f4252fca t50 (sum7): black 3@G3 x@G1 vs white's last piece, a
 * blank on G4; roll 2-2 -> black stepped the 3 to G4. Staying finished next turn
 * on 11/36 rolls, the move on 0/36. The endgame lookahead could not see it: it
 * counts only the OPPONENT's chance to finish, and both moves had the same one.
 *
 * No net. From a position with dice rolled it searches every end-of-turn
 * position the mover can reach, every opponent roll and reply, and so on for
 * `depth` opponent turns, scoring finished games by their exact margin. At the
 * horizon it returns BOUNDS, not an estimate (see leaf()), so every value is an
 * interval [lo, hi] containing the true value under best play, in POINTS from
 * the mover's view. Move A is PROVEN better than B when lo(A) > hi(B).
 *
 * Layout-neutral: it plays by whatever goal numbers the static data carries
 * (patch it the way local_agent.js's applyGoalLayout does).
 *
 * The JS engine carries no no-save counter, so a draw becomes callable only if
 * it already was; endings that close to the draw rule are rare and flagged by
 * the caller.
 */
const _ESE = (typeof require !== 'undefined') ? require('./engine.js')
                                              : (typeof window !== 'undefined' ? window : self);

const ES_N = 12;
const ES_ROLLS = [];
for (let a = 1; a <= 6; a++) for (let b = a; b <= 6; b++) ES_ROLLS.push([a, b, (a === b ? 1 : 2) / 36]);

const esIsPass = (m) => m.piece === null && m.lone === 0;
const esIsDraw = (m) => m.piece === null && m.lone === 1;
const esIsMove = (m) => m.piece !== null;
const esOther = (s) => (s === 'white' ? 'black' : 'white');
const esUnsaved = (e, side) => ES_N - e.savedRack(side).length;

/* Where every piece is (blanks anonymous), rack order, draw state. */
function esKey(e) {
    const on = [];
    for (const p of e.pieces) if (p.tile >= 0) on.push((p.player[0]) + (p.number > 6 ? 'x' : p.number) + '@' + p.tile);
    on.sort();
    const r = (k) => e.racks[k].map(p => p.number).join('.');
    return on.join(',') + '|' + r('white_unentered') + '|' + r('black_unentered') + '|' +
        e.savedRack('white').length + ',' + e.savedRack('black').length + '|' + (e.drawCallable ? 1 : 0);
}

function esUndoTo(e, n) { while (e.moves.length > n) e.undoLastMove(); }
function esSetDice(e, a, b) {
    e.dice[0].value = a; e.dice[0].used = false;
    e.dice[1].value = b; e.dice[1].used = false;
}

/* switchTurn without rolling, reversibly (agent.js oppFinishProb does the same). */
function esEnter(e) {
    const saved = { player: e.currentPlayer, dice: e.dice.map(d => [d.value, d.used]),
                    firstMove: e.firstMove, stages: Object.assign({}, e.stages), n: e.moves.length };
    e.firstMove = null;
    e.currentPlayer = esOther(e.currentPlayer);
    const before = new Map(e.pieces.filter(p => p.number <= 6).map(p => [p, p.number]));
    e.applyLastPieceRule();
    saved.renumbered = [...before].filter(([p, n]) => p.number !== n);
    return saved;
}
function esLeave(e, saved) {
    esUndoTo(e, saved.n);
    for (const [p, n] of saved.renumbered) {
        if (e.lookup.get(p.player + ',' + p.number) === p) e.lookup.delete(p.player + ',' + p.number);
        p.number = n;
        e.lookup.set(p.player + ',' + n, p);
    }
    e.currentPlayer = saved.player;
    e.dice.forEach((d, i) => { d.value = saved.dice[i][0]; d.used = saved.dice[i][1]; });
    e.firstMove = saved.firstMove;
    Object.assign(e.stages, saved.stages);
}

/* Every distinct end-of-turn position for the side to move (dice as set):
   cb(pair) is called with the engine AT that position (turn not switched).
   A draw call is reported as pair ['draw'] with the board unchanged. */
function esEndPositions(e, cb) {
    const side = e.currentPlayer;
    const root = e.moves.length;
    const seen = new Set();
    const emit = (pair) => {
        const k = esKey(e);
        if (seen.has(k)) return;
        seen.add(k);
        cb(pair);
    };
    const m1s = e.getValidMoves();
    if (m1s.some(esIsDraw)) cb(['draw']);
    const real1 = m1s.filter(esIsMove);
    if (!real1.length || m1s.some(esIsPass)) emit([]);
    for (const m1 of real1) {
        e.applyMove(m1, false);
        const mid = e.moves.length;
        if (e.checkGameOver()[0] === side) emit([m1]);
        else {
            const m2s = e.getValidMoves();
            const real2 = m2s.filter(esIsMove);
            if (!real2.length || m2s.some(esIsPass)) emit([m1]);
            for (const m2 of real2) {
                e.applyMove(m2, false);
                emit([m1, m2]);
                esUndoTo(e, mid);
            }
        }
        esUndoTo(e, root);
    }
}

/* Can `side`, to move with the dice as set, bank everything this turn? */
function esCanFinish(e, side, savesOnly) {
    // Exact shortcut: a bank takes one die and needs the piece ON a goal, and a
    // turn has two dice. So 3+ pieces can never finish, and 2 only by banking
    // both from goals (saves alone).
    const left = esUnsaved(e, side);
    if (left > 2) return false;
    if (left === 2) {
        if (!esAllOnGoals(e, side)) return false;
        savesOnly = true;
    }
    const root = e.moves.length;
    const ok = (m) => esIsMove(m) && (!savesOnly || m.dest === 'save');
    for (const m1 of e.getValidMoves()) {
        if (!ok(m1)) continue;
        e.applyMove(m1, false);
        let won = e.checkGameOver()[0] === side;
        if (!won) {
            const mid = e.moves.length;
            for (const m2 of e.getValidMoves()) {
                if (!ok(m2)) continue;
                e.applyMove(m2, false);
                won = e.checkGameOver()[0] === side;
                esUndoTo(e, mid);
                if (won) break;
            }
        }
        esUndoTo(e, root);
        if (won) return true;
    }
    return false;
}

/* What one side's chance to bank out this turn depends on: its own unsaved
   pieces and rack, and the OTHER side's walls (the only thing that can block a
   goal-to-goal step). Captures do not matter (a landing on a goal never
   captures, and a capture elsewhere changes nothing about banking). */
function esFinishKey(e, side, savesOnly) {
    const mine = [], walls = [];
    for (let t = 0; t < e.occ.length; t++) {
        const list = e.occ[t];
        if (!list.length) continue;
        for (const p of list) if (p.player === side) mine.push((p.number > 6 ? 'x' : p.number) + '@' + t);
        if (list.length > 1 && list[0].player !== side && e.graph.types[t] === 'field') walls.push(t);
    }
    mine.sort();
    return side[0] + (savesOnly ? 's' : 'f') + mine.join(',') + '|' + walls.join(',') + '|' +
        e.unenteredRack(side).map(p => p.number).join('.') + '|' + e.savedRack(side).length;
}

/* P(side, whose turn has been entered, banks out on its roll). Dice left dirty.
   `cache` (a Map) may be shared across positions on one layout. */
function esFinishProb(e, side, savesOnly, cache) {
    let key = null;
    if (cache) {
        key = esFinishKey(e, side, savesOnly);
        const hit = cache.get(key);
        if (hit !== undefined) return hit;
    }
    const v = esFinishProbRaw(e, side, savesOnly);
    if (cache) cache.set(key, v);
    return v;
}
function esFinishProbRaw(e, side, savesOnly) {
    let p = 0;
    for (const [a, b, w] of ES_ROLLS) {
        esSetDice(e, a, b);
        if (esCanFinish(e, side, savesOnly)) p += w;
    }
    return p;
}

function esAllOnGoals(e, side) {
    if (e.unenteredRack(side).length) return false;
    return e.pieces.every(p => p.player !== side || p.rack === 'saved'
                          || (p.tile >= 0 && e.graph.types[p.tile] === 'save'));
}

class EndgameSolver {
    constructor(depth = 2, finishCache = null) {
        this.depth = depth; this.memo = new Map(); this.nodes = 0;
        this.fcache = finishCache || new Map();
    }

    /* Bounds at the horizon, `mover`'s turn just ended. The opponent banks out
       next turn with P_o (exact): margin -my. If they miss they keep >= 1 piece,
       and `mover` then wins with probability >= q, the share of rolls on which it
       banks out by SAVES ALONE from goals -- pieces on goals cannot be captured
       or blocked, and the higher-die rule looks only at the owner's own pieces,
       so q holds whatever the opponent does. Otherwise the margin lies in
       [-my, +op] (unsaved counts never rise). */
    leaf(e, mover) {
        const opp = esOther(mover);
        const my = esUnsaved(e, mover), op = esUnsaved(e, opp);
        const key = 'L' + mover[0] + esKey(e);
        const hit = this.memo.get(key);
        if (hit) return hit;
        let pO = 0, q = 0;
        const s1 = esEnter(e);
        try {
            if (op <= 2) pO = esFinishProb(e, opp, false, this.fcache);
            if (my <= 2 && esAllOnGoals(e, mover)) {
                const s2 = esEnter(e);
                try { q = esFinishProb(e, mover, true, this.fcache); } finally { esLeave(e, s2); }
            }
        } finally { esLeave(e, s1); }
        const v = [pO * -my + (1 - pO) * (q * 1 + (1 - q) * -my),
                   pO * -my + (1 - pO) * op];
        this.memo.set(key, v);
        return v;
    }

    /* [lo, hi] for `mover`, whose turn has just ended on this board. */
    after(e, mover, depth) {
        const [w, s] = e.checkGameOver();
        if (w) {
            const v = w === 'draw' ? 0 : (w === mover ? s : -s);
            return [v, v];
        }
        if (depth <= 0) return this.leaf(e, mover);
        if (this.maxNodes && this.nodes > this.maxNodes) throw new Error('node budget');
        const key = depth + mover[0] + esKey(e);
        const hit = this.memo.get(key);
        if (hit) return hit;
        this.nodes++;
        const opp = esOther(mover);
        let lo = 0, hi = 0;
        const saved = esEnter(e);
        try {
            for (const [a, b, wt] of ES_ROLLS) {
                esSetDice(e, a, b);
                const [olo, ohi] = this.turn(e, opp, depth - 1);
                lo += wt * -ohi;
                hi += wt * -olo;
            }
        } finally { esLeave(e, saved); }
        const v = [lo, hi];
        this.memo.set(key, v);
        return v;
    }

    /* Best [lo, hi] for `side`, to move with the dice as set. */
    turn(e, side, depth) {
        let lo = -Infinity, hi = -Infinity;
        esEndPositions(e, (pair) => {
            const v = pair[0] === 'draw' ? [0, 0] : this.after(e, side, depth);
            if (v[0] > lo) lo = v[0];
            if (v[1] > hi) hi = v[1];
        });
        return [lo, hi];
    }

    /* Every distinct end position for the side to move: [{lo, hi, pair, key}],
       best lower bound first. */
    rank(e, prev = null, only = null) {
        // prev: an earlier rank() (shallower) -- intervals are intersected with
        // it, since every depth's bounds are valid. only(key): search just these
        // positions at this depth; the rest keep prev's interval.
        const side = e.currentPlayer;
        const out = [];
        const before = new Map((prev || []).map(x => [x.key, x]));
        esEndPositions(e, (pair) => {
            if (pair[0] === 'draw') { out.push({ lo: 0, hi: 0, pair, key: 'draw' }); return; }
            const key = esKey(e);
            const p = before.get(key);
            if (only && p && !only(key)) { out.push(p); return; }
            if (this.maxNodes && this.nodes > this.maxNodes) throw new Error('node budget');
            let [lo, hi] = this.after(e, side, this.depth);
            if (p) { lo = Math.max(lo, p.lo); hi = Math.min(hi, p.hi); }
            out.push({ lo, hi, pair, key, depth: this.depth });
        });
        out.sort((a, b) => (b.lo - a.lo) || (b.hi - a.hi));
        return out;
    }
}

(function () {
    const api = { EndgameSolver, esKey, esEnter, esLeave, esFinishProb, esCanFinish, esAllOnGoals,
                  esEndPositions, esUnsaved, esSetDice, ES_ROLLS };
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    else Object.assign(typeof window !== 'undefined' ? window : self, api);
})();
