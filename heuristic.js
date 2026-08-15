/* The heuristic evaluator, ported from agent.py (step 6.2 of PORTING.md).
 *
 * Only `evaluate` is needed: agent_gnn uses the heuristic solely to rank
 * candidates before the GNN sees them (`self.heuristic.evaluate(board, player)`,
 * score only). agent.py's own select_move_pair is the retired evolutionary
 * agent and is NOT ported.
 *
 * It matters that this is faithful even though it is "only" a prefilter: with
 * FIRST_MOVE_PREFILTER=12 the served agent keeps just twelve first moves, so a
 * scoring difference here changes which candidates the value net ever sees.
 *
 * Weights come from heuristic_weights.json, EXPORTED rather than ported -- see
 * export_heuristic_weights.py for why.
 *
 * Two BFS routines live here rather than in route.js because nothing else uses
 * them: targetGoalNumber and countEnemyBlotsOnShortestPath.
 */
const _RH = (typeof require !== 'undefined') ? require('./route.js')
                                             : (typeof window !== 'undefined' ? window : self);

const GAME_OVER_SCORE = 10000;

/* A weights table entry, keyed by piece number. Python does
   `weights[cat].get(n, dflt)`, and the expanded tables only hold 0..6 -- so a
   BLANK piece (number > 6, and 13 after the last-piece rule) falls through to
   the default. That default is 0 everywhere except enemy_blot_penalties, where
   it is 1. Getting this wrong is silent. */
const wget = (table, number, dflt) => {
    const v = table[String(number)];
    return v === undefined ? dflt : v;
};

/* The goal a piece is heading for, ignoring blocking entirely.
   NOTE this walks through nogo, home and blocked tiles alike -- get_target_goal_
   number has no traversability test at all, unlike every other BFS in the
   engine. That makes it a pure function of (number, start tile), which is why
   game.py can cache it on a Board that never invalidates the cache. */
function targetGoalNumber(engine, piece) {
    if (engine.canBeSaved(piece)) return null;
    const g = engine.graph;
    const start = piece.tile >= 0 ? piece.tile : engine.home;
    const visited = new Uint8Array(g.n);
    visited[start] = 1;
    let frontier = [start];
    while (frontier.length) {
        const next = [];
        for (const tile of frontier) {
            for (const nb of g.neighbors[tile]) {
                if (visited[nb]) continue;
                if (g.types[nb] === 'save' && (piece.number > 6 || piece.number === g.numbers[nb])) {
                    return g.numbers[nb];
                }
                visited[nb] = 1;
                next.push(nb);
            }
        }
        frontier = next;
    }
    return null;
}

/* Enemy blots on the way to the nearest usable goal. Infinity when no route
   exists, which the caller treats as "no penalty". */
function countEnemyBlotsOnShortestPath(engine, piece) {
    if (engine.canBeSaved(piece)) return 0;
    const g = engine.graph;
    const start = piece.tile >= 0 ? piece.tile : engine.home;
    const visited = new Uint8Array(g.n);
    visited[start] = 1;
    let frontier = [[start, 0]];
    while (frontier.length) {
        const next = [];
        for (const [tile, blots] of frontier) {
            for (const nb of g.neighbors[tile]) {
                if (visited[nb]) continue;
                visited[nb] = 1;
                let n = blots;
                const occ = engine.occ[nb];
                if (g.types[nb] === 'field' && occ.length === 1 && occ[0].player !== piece.player) n += 1;
                if (g.types[nb] === 'save' && (piece.number > 6 || piece.number === g.numbers[nb])) return n;
                if (g.types[nb] !== 'nogo' && g.types[nb] !== 'home' && !engine.isBlocked(nb, piece.player)) {
                    next.push([nb, n]);
                }
            }
        }
        frontier = next;
    }
    return Infinity;
}

function shortestRoutes(engine) {
    // One blocked set per player, as game.py caches per position.
    const blocked = {
        white: _RH.blockedTiles(engine.graph, engine.pieces, 'white'),
        black: _RH.blockedTiles(engine.graph, engine.pieces, 'black'),
    };
    const d = new Map();
    for (const p of engine.pieces) {
        d.set(p, _RH.shortestRouteToGoal(engine.graph,
            { player: p.player, number: p.number, tile: p.tile, saved: p.rack === 'saved' },
            blocked[p.player]));
    }
    return d;
}

function evaluatePlayer(engine, W, player, distances) {
    const g = engine.graph;
    const opponent = engine.other(player);
    const saveRack = engine.savedRack(player);
    const unentered = engine.unenteredRack(player);
    const oppUnentered = engine.unenteredRack(opponent);

    const playerPieces = engine.pieces.filter(p => p.player === player);
    const opponentPieces = engine.pieces.filter(p => p.player === opponent);
    // `if p.tile` in Python is true for ANY Tile object -- including the home
    // tile, whose index here is 0. `p.tile >= 0` is the faithful test; `p.tile`
    // alone would silently drop every piece standing on home.
    const boardPieces = playerPieces.filter(p => p.tile >= 0);
    const oppBoardPieces = opponentPieces.filter(p => p.tile >= 0);

    const savedBonus = saveRack.reduce((s, p) => s + wget(W.saved_bonuses, p.number, 0), 0);

    const goalPieces = playerPieces.filter(p => engine.canBeSaved(p));
    const goalBonus = goalPieces.filter(p => p.number <= 6)
        .reduce((s, p) => s + wget(W.goal_bonuses, p.number, 0), 0);

    const occupiedGoals = goalPieces.filter(p => p.tile >= 0 && p.number > 6).map(p => p.tile);
    const highGoalPenalty = occupiedGoals.reduce(
        (s, t) => s + wget(W.goal_bonuses, g.numbers[t], 0) * W.high_goal_penalty, 0);

    const near = boardPieces.filter(p => distances.get(p) >= 1 && distances.get(p) <= 6);
    const nearer = boardPieces.filter(p => p.number > 6 && distances.get(p) >= 1 && distances.get(p) <= 4);
    const nearGoalBonus = near.filter(p => p.number <= 6)
        .reduce((s, p) => s + wget(W.near_goal_bonuses, p.number, 0), 0);

    let highGoalProximityPenalty = 0;
    if (engine.stages[player] !== 'endgame') {
        const nearerSet = new Set(nearer);
        for (const piece of near.filter(p => p.number > 6 && !nearerSet.has(p))) {
            const target = targetGoalNumber(engine, piece);
            if (target) {
                const idx = target - 1;
                highGoalProximityPenalty +=
                    W.high_goal_proximity_penalties.a * Math.pow(idx, W.high_goal_proximity_penalties.b);
            }
        }
    }

    const numberedOffGoal = playerPieces.filter(
        p => p.number <= 6 && !engine.canBeSaved(p) && p.rack !== 'saved');
    const offGoalPenalty = -numberedOffGoal.reduce((s, p) => s + wget(W.goal_bonuses, p.number, 0), 0);
    const numberedFar = numberedOffGoal.filter(p => distances.get(p) > 6 && p.tile >= 0
        && (g.types[p.tile] === 'field' || g.types[p.tile] === 'save'));
    const farFromGoalPenalty = -numberedFar.reduce((s, p) => s + wget(W.goal_bonuses, p.number, 0), 0);

    const notNear = playerPieces.filter(p => distances.get(p) > 6);
    let totalDistance = Math.min(notNear.reduce((s, p) => s + distances.get(p), 0), 100);
    totalDistance += notNear.filter(p => p.number <= 6)
        .reduce((s, p) => s + wget(W.goal_bonuses, p.number, 0), 0) / 10;

    // > 1000 is how agent.py spells "unreachable": the distance is Infinity.
    const blockedPieces = playerPieces.filter(p => distances.get(p) > 1000);
    const blockedPieceBonus = blockedPieces.filter(p => p.number <= 6)
        .reduce((s, p) => s + wget(W.blocked_piece_penalties, p.number, 0), 0);

    const loose = boardPieces.filter(p => g.types[p.tile] === 'field' && engine.occ[p.tile].length === 1);
    let loosePiecePenalty = loose.length * W.loose_piece;
    if (engine.stages[opponent] === 'endgame') loosePiecePenalty *= -1;

    let loosePieceBonus = loose.filter(p => p.number <= 6)
        .reduce((s, p) => s + wget(W.loose_piece_penalties, p.number, 0), 0);
    const oppOnBoard = oppBoardPieces.filter(
        p => g.types[p.tile] === 'field' || g.types[p.tile] === 'home').length
        + Math.min(1, oppUnentered.length);
    loosePieceBonus *= (oppOnBoard / 12);

    let enemyBlotPenalty = 0;
    if (engine.stages.white !== 'endgame' && engine.stages.black !== 'endgame') {
        for (const piece of boardPieces) {
            if (engine.canBeSaved(piece)) continue;
            const blots = countEnemyBlotsOnShortestPath(engine, piece);
            if (blots !== Infinity && blots > 0) {
                enemyBlotPenalty += blots * wget(W.enemy_blot_penalties, piece.number, 1);
            }
        }
    }

    const captured = opponentPieces.filter(p => p.tile >= 0 && g.types[p.tile] === 'home');
    const capturedBonus = captured.filter(p => p.number <= 6)
        .reduce((s, p) => s + wget(W.captured_bonuses, p.number, 0), 0);

    const stageBonus = W.game_stage_bonuses[engine.stages[player]] || 0;

    // Die values that would do something useful this turn.
    const useful = new Set();
    for (const piece of playerPieces) {
        if (engine.canBeSaved(piece)) for (const r of engine.getSavingDie(piece)) useful.add(r);
    }
    for (const piece of boardPieces) {
        if (engine.canBeSaved(piece) || piece.rack === 'saved') continue;
        const d = distances.get(piece);
        if (d >= 1 && d <= 6) useful.add(d);
    }
    const diceSpreadBonus = useful.size * (W.dice_spread === undefined ? 3 : W.dice_spread);

    let permanentBlockBonus = 0;
    if (engine.stages[opponent] !== 'endgame') {
        for (let t = 0; t < g.n; t++) {
            if (g.types[t] !== 'field') continue;
            const friendly = engine.occ[t].filter(p => p.player === player);
            if (friendly.length >= 2 && friendly.every(p => p.number > 6)) {
                permanentBlockBonus += (W.permanent_block_bonus === undefined ? 13 : W.permanent_block_bonus);
            }
        }
    }

    const components = {
        saved_pieces: saveRack.length * W.saved_piece,
        saved_bonus: savedBonus,
        goal_pieces: goalPieces.length * W.goal_piece,
        goal_bonus: goalBonus,
        captured_pieces: captured.length * W.captured_opponent_piece,
        captured_bonus: capturedBonus,
        pieces_near_goal: near.length * W.near_goal_piece,
        pieces_nearer_goal: nearer.length * W.nearer_goal_piece,
        near_goal_bonus: nearGoalBonus,
        blocked_pieces: blockedPieces.length * W.blocked_piece,
        blocked_piece_bonus: blockedPieceBonus,
        loose_pieces: loosePiecePenalty,
        loose_piece_bonus: loosePieceBonus,
        total_distance: totalDistance * W.distance_penalty,
        unentered_pieces: unentered.length * W.unentered_piece,
        off_goal_penalty: offGoalPenalty,
        far_from_goal_penalty: farFromGoalPenalty,
        high_goal_penalty: highGoalPenalty,
        high_goal_proximity_penalty: highGoalProximityPenalty,
        enemy_blot_penalty: enemyBlotPenalty,
        game_stage_bonus: stageBonus,
        dice_spread_bonus: diceSpreadBonus,
        permanent_block_bonus: permanentBlockBonus,
    };
    // Python sums dict.values() in insertion order; the key order above is
    // agent.py's, so the float rounding matches term for term.
    let total = 0;
    for (const k of Object.keys(components)) total += components[k];
    return { total, components };
}

/* The whole evaluation: mine minus theirs, or a terminal score if the game is
   already over. */
function evaluate(engine, W, player) {
    const [winner, score] = engine.checkGameOver();
    if (winner) {
        return { score: (winner === player ? 1 : -1) * score * GAME_OVER_SCORE, components: {} };
    }
    const distances = shortestRoutes(engine);
    const mine = evaluatePlayer(engine, W, player, distances);
    const theirs = evaluatePlayer(engine, W, engine.other(player), distances);
    return { score: mine.total - theirs.total,
             components: { player: mine.components, opponent: theirs.components } };
}

(function () {
    const api = { evaluate, evaluatePlayer, targetGoalNumber,
                  countEnemyBlotsOnShortestPath, shortestRoutes, GAME_OVER_SCORE };
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    else Object.assign(typeof window !== 'undefined' ? window : self, api);
})();
