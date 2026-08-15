/* 2-ply move selection, ported from agent_gnn.py (step 6.3 of PORTING.md).
 *
 * Mirrors GNNAgent.select_move_pair as the SERVED app configures it:
 * use_prefilter=true, first_move_prefilter=12, prefilter_top_k=40,
 * prefilter_min_k=5, frac and score_alpha unset, enable_never_good=false.
 * `_fix_never_good` is therefore not ported -- it is off by default and is a
 * hand-coded patch over value-head errors that TD is meant to fix instead.
 *
 * Every tie is broken on the canonical move key rather than on enumeration
 * order (see agent_gnn._move_sort_key): sets of move tuples iterate in
 * hash-seed order in Python, which used to make the agent's answer depend on
 * the process. Reproducing THAT would be impossible; reproducing the canonical
 * order is exact.
 *
 * Scoring goes through infer.js, which matched Python to 4.5e-08 -- close, but
 * not bit-exact, so an exact score TIE in Python can be a 1e-8 gap here. That
 * is the one place this port cannot be made identical by construction; see
 * PORTING.md 6.4 for how the trace-diff treats it.
 */
const _AG = (typeof require !== 'undefined') ? require('./engine.js')
                                             : (typeof window !== 'undefined' ? window : self);
const _AH = (typeof require !== 'undefined') ? require('./heuristic.js')
                                             : (typeof window !== 'undefined' ? window : self);

const SCORE_SCALE = 1000.0;

const PASS = { piece: null, lone: 0, dest: 0, roll: 0 };
const DRAW = { piece: null, lone: 1, dest: 1, roll: 1 };
const isPass = (m) => m.piece === null && m.lone === 0;
const isDraw = (m) => m.piece === null && m.lone === 1;
const isSave = (m) => m.piece !== null && m.dest === 'save';

/* Where every piece sits: tile index, -2 saved, -1 racked. Two pairs with the
   same key leave the board in the same state. */
function pieceLocs(engine) {
    return engine.pieces.map(p => p.tile >= 0 ? p.tile : (p.rack === 'saved' ? -2 : -1)).join(',');
}

/* Everything the heuristic reads: placement plus the dice. */
function positionKey(engine) {
    return pieceLocs(engine) + '|' + (engine.dice[0].used ? 1 : 0) + (engine.dice[1].used ? 1 : 0);
}

/* How many of a pair's moves shuffle a piece around, as opposed to saving it
   or passing -- the legibility cost of a line. */
function pairRelocations(pair) {
    return pair.filter(m => m.piece !== null && m.dest !== 'save' && m.dest !== 0).length;
}

const cmpKey = (a, b) => {
    for (let i = 0; i < a.length; i++) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
};
const pairKey = (pair) => pair.map(_AG.moveKey);
const cmpPairKey = (a, b) => {
    const ka = pairKey(a), kb = pairKey(b);
    for (let i = 0; i < ka.length; i++) {
        const c = cmpKey(ka[i], kb[i]);
        if (c) return c;
    }
    return 0;
};

/* A piece can be saved at most once per turn: if both halves resolved to the
   same piece-id, drop the second to a pass. */
function dedupeSavePair(pair) {
    const pid = (m) => (isSave(m) ? m.piece[0] + ',' + m.piece[1] : null);
    const a = pid(pair[0]), b = pid(pair[1]);
    return (a !== null && a === b) ? [pair[0], PASS] : pair;
}

/* agent_gnn._select_filtered, with the served knobs. Ties break canonically. */
function selectFiltered(scored, opts) {
    const n = scored.length;
    if (!n) return [];
    const s = scored.slice().sort((x, y) => (y.score - x.score) || cmpPairKey(x.pair, y.pair));

    let cutCount = n;
    const alpha = opts.prefilterScoreAlpha;
    if (alpha !== null && alpha !== undefined && alpha < 1.0) {
        const best = s[0].score, worst = s[n - 1].score, spread = best - worst;
        cutCount = spread <= 1e-12 ? n : s.filter(x => x.score >= best - alpha * spread).length;
    }
    let ceil = opts.prefilterTopK || n;
    if (opts.prefilterFrac !== null && opts.prefilterFrac !== undefined) {
        ceil = Math.min(ceil, Math.max(1, Math.round(opts.prefilterFrac * n)));
    }
    const minK = opts.prefilterMinK || 1;
    let k = Math.max(cutCount, Math.min(minK, n));
    k = Math.min(k, ceil);
    k = Math.max(1, Math.min(k, n));
    return s.slice(0, k).map(x => x.pair);
}

/* Difficulty-controlled index. 1 (or unset) = argmax; below that, top-p over a
   z-scored softmax, so the agent plays weaker without picking absurd moves. */
function pickMoveIndex(scores, difficulty, rand) {
    const n = scores.length;
    const d = (difficulty === null || difficulty === undefined) ? 1.0 : difficulty;
    if (n <= 1 || d >= 0.999) {
        let best = 0;
        for (let i = 1; i < n; i++) if (scores[i] > scores[best]) best = i;
        return best;
    }
    const dd = Math.max(0, Math.min(1, d));
    const mean = scores.reduce((a, b) => a + b, 0) / n;
    let std = Math.sqrt(scores.reduce((a, b) => a + (b - mean) * (b - mean), 0) / n);
    if (!(std > 1e-6)) std = 1.0;
    const temp = 0.4 + (1 - dd) * 2.6;
    const topP = 0.25 + (1 - dd) * 0.75;
    const max = Math.max(...scores);
    const z = scores.map(v => (v - max) / (std * temp));
    const zmax = Math.max(...z);
    const e = z.map(v => Math.exp(v - zmax));
    const sum = e.reduce((a, b) => a + b, 0);
    const probs = e.map(v => v / sum);
    const order = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0] || a[1] - b[1]);
    let cum = 0, k = 0;
    for (const [p] of order) { if (cum < topP) k++; cum += p; }
    k = Math.max(1, Math.min(k, n));
    const keep = order.slice(0, k);
    const tot = keep.reduce((a, x) => a + x[0], 0);
    let r = (rand || Math.random)() * tot;
    for (const [p, i] of keep) { r -= p; if (r <= 0) return i; }
    return keep[keep.length - 1][1];
}

/* --- the search ------------------------------------------------------- */

/* Returns the chosen [move1, move2], or the full ranking when returnScores.
 * `score` is async (onnxruntime), so this is too.
 *
 *   score(engineList) -> Promise<number[]>   raw model outputs, one per position
 */
async function selectMovePair(engine, W, moves, player, opts = {}) {
    const o = Object.assign({
        prefilter: true, firstMovePrefilter: 12, prefilterTopK: 40, prefilterMinK: 5,
        prefilterFrac: null, prefilterScoreAlpha: null,
        difficulty: null, returnScores: false, rand: null,
    }, opts);
    const score = o.score;

    const drawLegal = moves.some(isDraw);
    const drawPair = [DRAW, PASS];

    const moveKeys = [];        // [move1, move2] pairs, GNN-scored
    const snapshots = [];       // the position for each, when not prefiltering
    const outcomeKeys = [];
    const scored = [];          // {score, pair} when prefiltering

    // Move orders transpose heavily -- about half a midgame turn's pairs reach
    // a position some other pair already reached -- and the heuristic is the
    // expensive part, so remember its verdict per position for this turn.
    const evalCache = new Map();
    const heur = () => {
        const key = positionKey(engine);
        let s = evalCache.get(key);
        if (s === undefined) {
            s = _AH.evaluate(engine, W, player).score;
            evalCache.set(key, s);
        }
        return s;
    };
    const record = (pair) => {
        if (o.prefilter) {
            scored.push({ score: heur(), pair });
        } else {
            moveKeys.push(pair);
            snapshots.push(o.snapshot(engine));
            outcomeKeys.push(pieceLocs(engine));
        }
    };

    if (moves.some(isPass)) record([PASS, PASS]);

    // A draw is never simulated: apply_move((1,1,1)) sets draw_called without
    // pushing an undo record, so probing it would silently end the game -- and
    // it moves no piece, so the position would encode identically to a pass.
    // Its true value is exactly 0 and is compared directly at the end.
    // Canonical enumeration order, matching Python's: the winning-move
    // short-circuits return the first win they meet, and a position can have
    // several (the same save with either die).
    const candidates = moves.filter(m => !isPass(m) && !isDraw(m)).sort(_AG.moveCmp);

    // --- Stage 1: rank first moves on their own -------------------------
    // Each score here is also that move's (move, pass) pair score -- a pass
    // changes nothing -- so stage 2 gets them back from evalCache for free.
    let movesIter = candidates;
    if (o.prefilter && o.firstMovePrefilter && candidates.length > o.firstMovePrefilter) {
        const firstScored = [];
        for (const move of candidates) {
            const base = engine.moves.length;
            engine.applyMove(move, false);
            if (engine.checkGameOver()[0] === player) {
                while (engine.moves.length > base) engine.undoLastMove();
                const win = [move, PASS];
                return o.returnScores ? [{ score: Infinity, pair: win }] : win;
            }
            // Scored BEFORE the stage is touched, matching the Python twin's
            // order exactly (the block below restores the stage, so the two are
            // equivalent either way -- but the twins are kept textually parallel
            // on purpose).
            const s = heur();
            // Does a save become available after this move? See the twin in
            // agent_gnn.py: stage 1 ranks first moves ALONE, so "step onto the
            // goal with one die, save with the other" is judged on the step and
            // can be culled before the value head sees the save. getSavingDie
            // reads the stages, and stage 1 applies moves without the
            // getValidMoves call that refreshes them, so compute fresh and put
            // it back -- leaving it changed would drift later candidates.
            const prevStage = engine.stages[player];
            engine.stages[player] = engine.getGameStage(player);
            const enables = engine.pieces.some(p => p.player === player
                                                    && engine.getSavingDie(p).length);
            engine.stages[player] = prevStage;
            firstScored.push({ score: s, move, enables });
            while (engine.moves.length > base) engine.undoLastMove();
        }
        firstScored.sort((a, b) => (b.score - a.score) || cmpKey(_AG.moveKey(a.move), _AG.moveKey(b.move)));
        const keep = firstScored.slice(0, o.firstMovePrefilter).map(x => x.move);
        const kept = new Set(keep);
        // A first move that SAVES is never culled, for the same reason save
        // pairs are exempt from the top-K cull below -- nor is one that ENABLES
        // a save as the pair's second half.
        for (const x of firstScored) {
            if (!kept.has(x.move) && (isSave(x.move) || x.enables)) keep.push(x.move);
        }
        movesIter = keep;
    }

    for (const move of movesIter) {
        const base = engine.moves.length;
        engine.applyMove(move, false);

        // A winning move beats any learned value: the net never sees game_over
        // and a won board is maximally out of distribution.
        if (engine.checkGameOver()[0] === player) {
            while (engine.moves.length > base) engine.undoLastMove();
            const win = [move, PASS];
            return o.returnScores ? [{ score: Infinity, pair: win }] : win;
        }

        const stillCaptured = engine.occ[engine.home].some(p => p.player === engine.currentPlayer);
        if (!stillCaptured) record([move, PASS]);

        if (engine.dice.every(d => d.used)) {
            while (engine.moves.length > base) engine.undoLastMove();
            continue;
        }
        const nextMoves = engine.getValidMoves().filter(m => !isPass(m) && !isDraw(m)).sort(_AG.moveCmp);
        if (!nextMoves.length) {
            while (engine.moves.length > base) engine.undoLastMove();
            continue;
        }
        for (const next of nextMoves) {
            engine.applyMove(next, false);
            if (engine.checkGameOver()[0] === player) {
                while (engine.moves.length > base) engine.undoLastMove();
                const win = [move, next];
                return o.returnScores ? [{ score: Infinity, pair: win }] : win;
            }
            record([move, next]);
            engine.undoLastMove();
        }
        while (engine.moves.length > base) engine.undoLastMove();
    }

    // --- Encode only the kept candidates --------------------------------
    if (o.prefilter) {
        if (!scored.length) return drawLegal ? (o.returnScores ? [{ score: 0, pair: drawPair }] : drawPair)
                                             : [PASS, PASS];
        // Save pairs are exempt from the heuristic cull: the heuristic can
        // undervalue a save against a flashier non-save, which would drop it
        // before the net ever saw it.
        const saveScored = scored.filter(x => x.pair.some(isSave));
        const otherScored = scored.filter(x => !x.pair.some(isSave));
        const topPairs = saveScored.map(x => x.pair).concat(selectFiltered(otherScored, o));
        for (const pair of topPairs) {
            const base = engine.moves.length;
            for (const m of pair) if (!isPass(m)) engine.applyMove(m, false);
            moveKeys.push(pair);
            snapshots.push(o.snapshot(engine));
            outcomeKeys.push(pieceLocs(engine));
            while (engine.moves.length > base) engine.undoLastMove();
        }
    }

    if (!moveKeys.length) return drawLegal ? (o.returnScores ? [{ score: 0, pair: drawPair }] : drawPair)
                                           : [PASS, PASS];

    // One candidate and no draw to weigh it against: the argmax is already
    // decided, so skip the forward pass -- but run the same tail.
    if (moveKeys.length === 1 && !drawLegal && !o.returnScores) return dedupeSavePair(moveKeys[0]);

    const raw = await score(snapshots);
    const finalScores = Array.from(raw, v => v * SCORE_SCALE);

    if (o.returnScores) {
        const ranked = finalScores.map((s, i) => ({ score: s, pair: moveKeys[i] }));
        if (drawLegal) ranked.push({ score: 0.0, pair: drawPair });
        ranked.sort((a, b) => (b.score - a.score) || cmpPairKey(a.pair, b.pair));
        return ranked;
    }

    let bestIdx = 0;
    for (let i = 1; i < finalScores.length; i++) if (finalScores[i] > finalScores[bestIdx]) bestIdx = i;
    // A draw's true value is exactly 0 in these units. Compared against the true
    // best, not a difficulty-sampled pick, so a lower difficulty never makes the
    // agent resign a game it should not.
    if (drawLegal && 0.0 >= finalScores[bestIdx]) return drawPair;

    bestIdx = pickMoveIndex(finalScores, o.difficulty, o.rand);

    // Exact score ties settled canonically, not by enumeration order.
    const top = finalScores[bestIdx];
    const tied = [];
    for (let i = 0; i < finalScores.length; i++) if (finalScores[i] === top) tied.push(i);
    if (tied.length > 1) {
        tied.sort((a, b) => cmpPairKey(moveKeys[a], moveKeys[b]));
        bestIdx = tied[0];
    }

    // Among candidates leaving the board in the SAME state, take the one that
    // moves the fewest pieces: the position after the turn is identical and an
    // unused die is worthless once the turn ends, so this is free -- and it
    // stops the agent walking a blank round the rim to save it somewhere else.
    const key = outcomeKeys[bestIdx];
    const same = [];
    for (let i = 0; i < outcomeKeys.length; i++) if (outcomeKeys[i] === key) same.push(i);
    if (same.length > 1) {
        same.sort((a, b) => (pairRelocations(moveKeys[a]) - pairRelocations(moveKeys[b]))
                            || cmpPairKey(moveKeys[a], moveKeys[b]));
        bestIdx = same[0];
    }

    return dedupeSavePair(moveKeys[bestIdx]);
}

(function () {
    const api = { selectMovePair, selectFiltered, pickMoveIndex, dedupeSavePair,
                  pieceLocs, positionKey, pairRelocations, cmpPairKey,
                  PASS, DRAW, isPass, isDraw, isSave, SCORE_SCALE };
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    else Object.assign(typeof window !== 'undefined' ? window : self, api);
})();
