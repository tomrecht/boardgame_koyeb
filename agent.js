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

/* OPPONENT CERTAIN TO WIN NEXT TURN: BANK AS MANY AS YOU CAN (owner, 2026-10-01).
   Their last one or two pieces are blanks on goal 1, so ANY roll banks them (an
   exact 1 or, from the highest goal they hold, anything higher), and nothing we do can
   stop it: goals cannot be blocked and pieces on them cannot be captured. If we
   cannot win this turn -- and the search returns a winning pair before it ever
   gets to scoring -- the game is lost and only the margin is still in play, so
   the pair that banks the most of our own pieces is right whatever the net says.
   The net was seen to prefer bringing a piece onto a goal for a turn that never
   comes. The lone-blank case was added 2026-10-03 after the net did it again
   against one blank. Deliberately narrow otherwise; the twin is in agent_gnn.py. */
function opponentWinsNextTurnRegardless(engine, player) {
    const opp = player === 'white' ? 'black' : 'white';
    if (engine.unenteredRack(opp).length) return false;
    const left = engine.pieces.filter(p => p.player === opp && p.tile >= 0);
    if (left.length < 1 || left.length > 2) return false;
    return left.every(p => p.number > 6 && engine.graph.types[p.tile] === 'save'
                                        && engine.graph.numbers[p.tile] === 1);
}
function ownSaves(pair, player) {
    return pair.filter(m => isSave(m) && m.piece[0] === player).length;
}
/* ...AND NO SHUFFLING (owner, 2026-10-07). With the game lost, moving pieces
   about the field means nothing, so among the max-save pairs keep those where
   every half is a pass, an own save, or a tile move onto a goal the piece can
   bank from (its own goal if numbered, from anywhere; any goal if blank, but not
   from another goal -- owner: shuffling between goals) -- pointless too unless
   it enables a save, but it looks human. An entry onto home counts as quiet: it
   is the rack obligation, not a choice. If no pair qualifies (some forced move),
   the max-save set stands. Twin of _quiet_pair in agent_gnn.py. */
function quietPair(engine, pair, player) {
    // Call with the engine at the TURN START: a half's origin is the piece's
    // tile there, or the first half's destination if the same piece moved first.
    const tileOf = (d) => engine.graph.indexOf(d[0], d[1]);
    return pair.every((m, i) => {
        if (isPass(m)) return true;
        if (!m.piece || m.piece[0] !== player) return false;
        if (isSave(m)) return true;
        if (!Array.isArray(m.dest) || !(m.roll > 0)) return false;   // dest is [ring, pos]
        const t = tileOf(m.dest);
        if (t === engine.home) return true;
        if (engine.graph.types[t] !== 'save') return false;
        if (m.piece[1] <= 6) return engine.graph.numbers[t] === m.piece[1];
        // A blank: any goal, but not from another goal -- that is shuffling.
        const prev = i === 1 ? pair[0] : null;
        const from = (prev && prev.piece && prev.piece[0] === m.piece[0] && prev.piece[1] === m.piece[1]
                      && Array.isArray(prev.dest))
            ? tileOf(prev.dest)
            : (engine.pieces.find(p => p.player === m.piece[0] && p.number === m.piece[1]) || {}).tile;
        return !(from >= 0 && engine.graph.types[from] === 'save');
    });
}

/* NEVER LEAVE A DIE UNUSED WHEN A SAVE IS AVAILABLE FOR IT (owner, 2026-10-02).
   Measured: in 3 of 1,002 midgame positions with a blank save legal, the net
   played one move and passed a die that could still have saved a piece -- a save
   given away for nothing (goals cannot be blocked, pieces on them cannot be
   captured). So a candidate pair that ends with a die unused while a save (blank
   OR numbered) is legal for it is dropped, whenever any candidate survives.
   IF A STRONGER MODEL IS EVER TRAINED, TRY IT WITH THIS RULE OFF: it may have
   learned this itself, and conceivably a piece left unsaved is sometimes worth
   keeping on the board as a spare capturer. Twin in agent_gnn.py.
   Call with the engine IN the pair's resulting position. getValidMoves rewrites
   the mover's stage, which other candidates read, so it is put back. */
function wastesSave(engine, pair, player) {
    const played = pair.filter(m => !isPass(m) && !isDraw(m));
    if (played.length >= 2) return false;
    if (played.some(m => m.dest === 0 && m.roll === 0)) return false;   // block-save: both dice
    if (engine.dice.every(d => d.used)) return false;
    const stage = engine.stages[player];
    const any = engine.getValidMoves().some(m => isSave(m) && m.piece[0] === player);
    engine.stages[player] = stage;
    return any;
}

/* ENDGAME LOOKAHEAD (owner, 2026-10-08). The opponent has at most two unsaved
   pieces, so their next turn may end the game -- and on a roll that does, the
   result is EXACT: they win by our unsaved count, however they do it. The net
   valued a piece stepped onto a goal as nearly a saved piece even then. Twin of
   _opponent_near_finish / _select_with_lookahead in agent_gnn.py. */
const LOOKAHEAD_PIECES = 12;
function opponentNearFinish(engine, player) {
    const opp = player === 'white' ? 'black' : 'white';
    return LOOKAHEAD_PIECES - engine.savedRack(opp).length <= 2;
}

const ROLLS_21 = [];
for (let a = 1; a <= 6; a++) for (let b = a; b <= 6; b++) ROLLS_21.push([a, b, (a === b ? 1 : 2) / 36]);

/* Can `opp`, to move, bank everything on the dice as set? Board left as found. */
function canFinish(engine, opp, base) {
    for (const m of engine.getValidMoves()) {
        if (isPass(m) || isDraw(m)) continue;
        engine.applyMove(m, false);
        let won = engine.checkGameOver()[0] === opp;
        if (!won && !engine.dice.every(d => d.used)) {
            const mid = engine.moves.length;
            for (const m2 of engine.getValidMoves()) {
                if (isPass(m2) || isDraw(m2)) continue;
                engine.applyMove(m2, false);
                won = engine.checkGameOver()[0] === opp;
                while (engine.moves.length > mid) engine.undoLastMove();
                if (won) break;
            }
        }
        while (engine.moves.length > base) engine.undoLastMove();
        if (won) return true;
    }
    return false;
}

/* P(`opp`, moving next from here -- our pair applied, turn not yet switched --
   can bank out on the roll they get). Enters their turn the way switchTurn
   does, without rolling, and undoes it exactly, including the last-piece
   rule's renumbering (agent_gnn._enter_opponent_turn_deterministic). */
function oppFinishProb(engine, opp) {
    const saved = {
        player: engine.currentPlayer,
        dice: engine.dice.map(d => [d.value, d.used]),
        firstMove: engine.firstMove,
        stages: Object.assign({}, engine.stages),
    };
    engine.firstMove = null;
    engine.currentPlayer = opp;
    const before = new Map(engine.pieces.filter(p => p.number <= 6).map(p => [p, p.number]));
    engine.applyLastPieceRule();
    const renumbered = [...before].filter(([p, n]) => p.number !== n);
    const base = engine.moves.length;
    let prob = 0;
    try {
        for (const [d1, d2, w] of ROLLS_21) {
            engine.dice[0].value = d1; engine.dice[0].used = false;
            engine.dice[1].value = d2; engine.dice[1].used = false;
            if (canFinish(engine, opp, base)) prob += w;
        }
    } finally {
        while (engine.moves.length > base) engine.undoLastMove();
        for (const [p, n] of renumbered) {
            if (engine.lookup.get(p.player + ',' + p.number) === p) engine.lookup.delete(p.player + ',' + p.number);
            p.number = n;
            engine.lookup.set(p.player + ',' + n, p);
        }
        engine.currentPlayer = saved.player;
        engine.dice.forEach((d, i) => { d.value = saved.dice[i][0]; d.used = saved.dice[i][1]; });
        engine.firstMove = saved.firstMove;
        Object.assign(engine.stages, saved.stages);
    }
    return prob;
}

/* Every candidate the shallow search keeps (same prefilter, same rules),
   rescored as P(finish) * exact final margin + (1 - P(finish)) * net value.
   Move generation only; no extra net evaluations. */
async function selectWithLookahead(engine, W, moves, player, o) {
    const stages0 = Object.assign({}, engine.stages);
    let ranked;
    try {
        ranked = await selectMovePair(engine, W, moves, player,
                                      Object.assign({}, o, { returnScores: true, _inLookahead: true }));
    } finally {
        Object.assign(engine.stages, stages0);
    }
    if (!ranked.length || ranked[0].pair === undefined) return ranked;     // a bare pair: nothing to score
    if (ranked[0].score === Infinity) return o.returnScores ? ranked : ranked[0].pair;
    const drawPair = [DRAW, PASS];
    const drawLegal = ranked.some(r => isDraw(r.pair[0]));
    const scored = ranked.filter(r => !isDraw(r.pair[0]));
    if (!scored.length) return o.returnScores ? ranked : (drawLegal ? drawPair : [PASS, PASS]);

    const opp = player === 'white' ? 'black' : 'white';
    const root = engine.moves.length;
    const vals = [], cands = [], outcomes = [], shallow = [], memo = new Map();
    try {
        for (const r of scored) {
            for (const m of r.pair) if (!isPass(m)) engine.applyMove(m, false);
            const key = pieceLocs(engine);
            if (!memo.has(key)) {
                memo.set(key, [oppFinishProb(engine, opp),
                               -(LOOKAHEAD_PIECES - engine.savedRack(player).length) / LOOKAHEAD_PIECES * SCORE_SCALE]);
            }
            while (engine.moves.length > root) engine.undoLastMove();
            const [p, vFin] = memo.get(key);
            vals.push(p * vFin + (1 - p) * r.score);
            shallow.push(r.score);
            cands.push(r.pair);
            outcomes.push(key);
        }
    } finally {
        Object.assign(engine.stages, stages0);
    }

    if (o.returnScores) {
        const out = vals.map((v, i) => ({ score: v, pair: cands[i] }));
        if (drawLegal) out.push({ score: 0.0, pair: drawPair });
        out.sort((a, b) => (b.score - a.score) || cmpPairKey(a.pair, b.pair));
        return out;
    }
    if (cands.length === 1 && !drawLegal) return dedupeSavePair(cands[0]);
    if (drawLegal && 0.0 >= Math.max(...vals)) return drawPair;
    let best = pickMoveIndex(vals, o.difficulty, o.rand);
    const top = vals[best];
    const tied = [];
    for (let i = 0; i < vals.length; i++) if (vals[i] === top) tied.push(i);
    if (tied.length > 1) {
        // Certain to be over (P(finish) = 1) ties every pair with the same saves;
        // the net's own score then decides, as without the lookahead.
        tied.sort((a, b) => (shallow[b] - shallow[a]) || cmpPairKey(cands[a], cands[b]));
        best = tied[0];
    }
    const same = [];
    for (let i = 0; i < outcomes.length; i++) if (outcomes[i] === outcomes[best]) same.push(i);
    if (same.length > 1) {
        same.sort((a, b) => (pairRelocations(cands[a]) - pairRelocations(cands[b]))
                            || cmpPairKey(cands[a], cands[b]));
        best = same[0];
    }
    return dedupeSavePair(cands[best]);
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
        lookahead: true, _inLookahead: false,
    }, opts);
    const score = o.score;

    if (o.lookahead && !o._inLookahead && opponentNearFinish(engine, player))
        return selectWithLookahead(engine, W, moves, player, o);

    const drawLegal = moves.some(isDraw);
    const lossCertain = opponentWinsNextTurnRegardless(engine, player);
    const drawPair = [DRAW, PASS];

    const moveKeys = [];        // [move1, move2] pairs, GNN-scored
    const snapshots = [];       // the position for each, when not prefiltering
    const outcomeKeys = [];
    const scored = [];          // {score, pair} when prefiltering
    const wastes = [];          // per moveKeys entry: leaves a die unused with a save on

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
            wastes.push(wastesSave(engine, pair, player));
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
            if (!kept.has(x.move) && (isSave(x.move) || x.enables
                                      || (lossCertain && quietPair(engine, [x.move], player)))) keep.push(x.move);
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
        // With the game lost, a quiet pair (see quietPair) is exempt too, or
        // the filter below would have nothing quiet left to choose from.
        const exempt = (x) => x.pair.some(isSave) || (lossCertain && quietPair(engine, x.pair, player));
        const saveScored = scored.filter(exempt);
        const otherScored = scored.filter(x => !exempt(x));
        const topPairs = saveScored.map(x => x.pair).concat(selectFiltered(otherScored, o));
        for (const pair of topPairs) {
            const base = engine.moves.length;
            for (const m of pair) if (!isPass(m)) engine.applyMove(m, false);
            moveKeys.push(pair);
            snapshots.push(o.snapshot(engine));
            outcomeKeys.push(pieceLocs(engine));
            wastes.push(wastesSave(engine, pair, player));
            while (engine.moves.length > base) engine.undoLastMove();
        }
    }

    if (!moveKeys.length) return drawLegal ? (o.returnScores ? [{ score: 0, pair: drawPair }] : drawPair)
                                           : [PASS, PASS];

    // One candidate and no draw to weigh it against: the argmax is already
    // decided, so skip the forward pass -- but run the same tail.
    if (moveKeys.length === 1 && !drawLegal && !o.returnScores) return dedupeSavePair(moveKeys[0]);

    const raw = await score(snapshots);
    let finalScores = Array.from(raw, v => v * SCORE_SCALE);
    const keepOnly = (keep) => {
        if (keep.length === moveKeys.length) return;
        const pick = (arr) => keep.map(i => arr[i]);
        moveKeys.splice(0, moveKeys.length, ...pick(moveKeys));
        outcomeKeys.splice(0, outcomeKeys.length, ...pick(outcomeKeys));
        wastes.splice(0, wastes.length, ...pick(wastes));
        finalScores = pick(finalScores);
    };

    // Never pass a die that could still save a piece (see wastesSave).
    {
        const keep = wastes.map((w, i) => w ? -1 : i).filter(i => i >= 0);
        if (keep.length) keepOnly(keep);
    }

    // Lost whatever happens: keep only the pairs that bank the most (see
    // opponentWinsNextTurnRegardless). Everything below -- argmax, ties,
    // difficulty, the draw check, a hint's ranking -- then chooses among those.
    if (lossCertain) {
        const n = moveKeys.map(pr => ownSaves(pr, player));
        const most = Math.max(...n);
        keepOnly(n.map((c, i) => c === most ? i : -1).filter(i => i >= 0));
        const quiet = moveKeys.map((pr, i) => quietPair(engine, pr, player) ? i : -1).filter(i => i >= 0);
        if (quiet.length) keepOnly(quiet);
    }

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
    const api = { selectMovePair, selectFiltered, oppFinishProb, opponentNearFinish, pickMoveIndex, dedupeSavePair,
                  pieceLocs, positionKey, pairRelocations, cmpPairKey,
                  PASS, DRAW, isPass, isDraw, isSave, SCORE_SCALE };
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    else Object.assign(typeof window !== 'undefined' ? window : self, api);
})();
