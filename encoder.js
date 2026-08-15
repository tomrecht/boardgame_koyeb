/* Board encoder, ported from encoder.py (step 4 of PORTING.md).
 *
 * Consumes the same snapshot shape the fixture records, and the static half
 * exported by export_encoder_static.py. Every function here is asserted
 * array-equal against encoder.py over the fixture positions.
 *
 * Piece counts are per CURRENT PLAYER, so a position encodes differently
 * depending on whose turn it is -- the caller passes currentPlayer explicitly
 * rather than reading it off the snapshot, matching encode(board, player).
 */
const { buildGraph, blockedTiles, shortestRouteToGoal, allGoalDistances,
        isSaveableOn, piecesFromSnapshot } = require('./route.js');

const TILE_FEAT_DIM = 12;
const PIECE_FEAT_DIM = 24;
const NUM_PIECES = 12;
const MAX_DIST = 14.0;
// status one-hot, in the order encoder.py assigns
const ST_UNENTERED = 0, ST_ON_HOME = 1, ST_ON_BOARD = 2, ST_CAN_BE_SAVED = 3, ST_SAVED = 4;
const NUM_STATUSES = 5;

/* [8] has_permanent_block, [10] my_piece_count, [11] opp_piece_count.
 * The base array is zero in those slots, so only occupied tiles need touching
 * -- most tiles are empty most of the time. */
function tileFeatures(graph, base, pieces, currentPlayer) {
    const n = graph.n;
    const arr = base.map(row => row.slice());        // copy; the base is reused

    const onTile = new Map();                        // tile -> pieces
    for (const p of pieces) {
        if (p.tile < 0) continue;
        arr[p.tile][p.player === currentPlayer ? 10 : 11] += 1.0;
        let list = onTile.get(p.tile);
        if (!list) onTile.set(p.tile, (list = []));
        list.push(p);
    }

    for (const [tile, list] of onTile) {
        if (graph.types[tile] !== 'field') continue;
        const mine = list.filter(p => p.player === currentPlayer);
        // A permanent block needs TWO of my UNNUMBERED pieces: numbered ones
        // leave for their own goals, so a stack containing them is temporary.
        if (mine.length >= 2 && mine.filter(p => p.number > 6).length >= 2) arr[tile][8] = 1.0;
    }
    void n;
    return arr;
}

function pieceStatus(graph, p) {
    if (p.saved) return ST_SAVED;
    if (p.unentered) return ST_UNENTERED;
    if (p.tile >= 0) {
        if (graph.types[p.tile] === 'home') return ST_ON_HOME;
        if (isSaveableOn(graph, p.tile, p.number)) return ST_CAN_BE_SAVED;
        return ST_ON_BOARD;
    }
    return ST_ON_BOARD;
}

/* 0 already there | 0.25 one die | 0.5 two dice | 0.75 three+ | 1 unreachable */
function distBin(d) {
    if (d === 0) return 0.0;
    if (d === Infinity) return 1.0;
    if (d <= 6) return 0.25;
    if (d <= 12) return 0.5;
    return 0.75;
}

/* NOTE the inversion: distance_category runs the OTHER way from distBin --
   unreachable is 0 and "close" is 1. Reading them as the same scale is an easy
   and silent mistake. */
function distanceCategory(d) {
    if (d === Infinity) return 0.0;
    if (d <= 6) return 1.0;
    if (d <= 12) return 0.5;
    return 0.25;
}

/* Rows are ordered CURRENT player's pieces by number, then the opponent's by
   number. The edge arrays index into this order, so it is load-bearing. */
function pieceFeatures(graph, snapshot, pieces, currentPlayer) {
    const opponent = currentPlayer === 'white' ? 'black' : 'white';
    const byNumber = (a, b) => a.number - b.number;
    let cur = pieces.filter(p => p.player === currentPlayer).sort(byNumber);
    let opp = pieces.filter(p => p.player === opponent).sort(byNumber);
    if (cur.length > NUM_PIECES) cur = cur.slice(0, NUM_PIECES);
    if (opp.length > NUM_PIECES) opp = opp.slice(0, NUM_PIECES);
    const allPieces = cur.concat(opp);

    // rack slot, from the recorded rack ORDER (both players)
    const rackPos = new Map();
    for (const side of ['white', 'black']) {
        (snapshot.racks[side + '_unentered'] || []).forEach((num, i) => rackPos.set(side + num, i));
    }

    const occupancy = new Map();
    for (const p of pieces) {
        if (p.tile < 0) continue;
        occupancy.set(p.tile, (occupancy.get(p.tile) || 0) + 1);
    }

    const blockedFor = new Map();
    const blockedOf = (player) => {
        let b = blockedFor.get(player);
        if (!b) blockedFor.set(player, (b = blockedTiles(graph, pieces, player)));
        return b;
    };

    const rows = allPieces.map((p) => {
        const blocked = blockedOf(p.player);
        const status = pieceStatus(graph, p);
        const onehot = new Array(NUM_STATUSES).fill(0.0);
        onehot[status] = 1.0;
        const rp = rackPos.get(p.player + p.number) || 0;
        const isBlot = (p.tile >= 0 && graph.types[p.tile] === 'field'
                        && occupancy.get(p.tile) === 1) ? 1.0 : 0.0;
        const dist = shortestRouteToGoal(graph, p, blocked);
        const goals = allGoalDistances(graph, p, blocked);

        const raw = [], binned = [];
        for (let g = 1; g <= 6; g++) {
            const d = goals[g] === undefined ? Infinity : goals[g];
            raw.push(d === Infinity ? 1.0 : Math.min(d, MAX_DIST) / MAX_DIST);
            binned.push(distBin(d));
        }
        return [
            p.player === currentPlayer ? 0.0 : 1.0,
            p.number / 12.0,
            p.number <= 6 ? 1.0 : 0.0,
            ...onehot,
            status === ST_UNENTERED ? rp / 11.0 : 0.0,
            isBlot,
            dist === Infinity ? 1.0 : 0.0,
            distanceCategory(dist),
            ...raw,
            ...binned,
        ];
    });
    return { rows, allPieces };
}

/* opening while anything is still racked; endgame once every piece of the
   player's is saveable; midgame otherwise. Computed FRESH, never read off the
   snapshot's game_stages -- that dict is mutated as a side effect of
   get_valid_moves and goes stale during candidate enumeration (encoder.py has
   the full account). */
function gameStage(graph, pieces, snapshot, player) {
    if ((snapshot.racks[player + '_unentered'] || []).length > 0) return 0.0;   // opening
    const mine = pieces.filter(p => p.player === player);
    const saveable = (p) => p.saved || isSaveableOn(graph, p.tile, p.number);
    return mine.every(saveable) ? 1.0 : 0.5;                                    // endgame : midgame
}

/* Highest goal number occupied by a saveable piece of this player; -1 if none. */
function highestOccupiedGoal(graph, pieces, player) {
    let best = -1;
    for (const p of pieces) {
        if (p.player !== player || p.tile < 0) continue;
        if (graph.types[p.tile] !== 'save') continue;
        if (!isSaveableOn(graph, p.tile, p.number)) continue;
        best = Math.max(best, graph.numbers[p.tile]);
    }
    return best;
}

function globalFeatures(graph, snapshot, pieces, currentPlayer) {
    const opponent = currentPlayer === 'white' ? 'black' : 'white';
    const [d1, d2] = snapshot.dice;
    const savedNumbered = (player) => pieces.filter(
        p => p.player === player && p.number <= 6 && p.saved).length;
    const saveable = pieces.filter(
        p => p.player === currentPlayer && (p.saved || isSaveableOn(graph, p.tile, p.number))).length;
    return [
        d1.value / 6.0,
        d2.value / 6.0,
        d1.used ? 1.0 : 0.0,
        d2.used ? 1.0 : 0.0,
        gameStage(graph, pieces, snapshot, currentPlayer),
        gameStage(graph, pieces, snapshot, opponent),
        savedNumbered(currentPlayer) / 6.0,
        savedNumbered(opponent) / 6.0,
        Math.max(highestOccupiedGoal(graph, pieces, currentPlayer), 0) / 6.0,
        Math.max(highestOccupiedGoal(graph, pieces, opponent), 0) / 6.0,
        saveable / 12.0,
    ];
}

/* piece_to_tile [2, N] row0 = piece row index, row1 = tile index; and the
   reverse. Indexes into the piece ORDER, which is why that order is fixed. */
function pieceTileEdges(allPieces) {
    const psrc = [], tdst = [];
    allPieces.forEach((p, i) => { if (p.tile >= 0) { psrc.push(i); tdst.push(p.tile); } });
    return { piece_to_tile: [psrc, tdst], tile_to_piece: [tdst, psrc] };
}

/* The encoder's whole input for one position. */
function encode(staticData, snapshot, currentPlayer) {
    const graph = buildGraph(staticData);
    const pieces = piecesFromSnapshot(graph, snapshot);
    const player = currentPlayer || snapshot.current_player;
    const { rows, allPieces } = pieceFeatures(graph, snapshot, pieces, player);
    const edges = pieceTileEdges(allPieces);
    return {
        tile_feats: tileFeatures(graph, staticData.base_tile_feats, pieces, player),
        piece_feats: rows,
        piece_order: allPieces.map(p => [p.player, p.number]),
        global_feats: globalFeatures(graph, snapshot, pieces, player),
        piece_to_tile: edges.piece_to_tile,
        tile_to_piece: edges.tile_to_piece,
        tile_edge_index: staticData.tile_edge_index,
    };
}

module.exports = { buildGraph, blockedTiles, shortestRouteToGoal, allGoalDistances,
                   isSaveableOn, piecesFromSnapshot, tileFeatures, pieceFeatures,
                   pieceStatus, distBin, distanceCategory, globalFeatures,
                   gameStage, highestOccupiedGoal, pieceTileEdges, encode,
                   TILE_FEAT_DIM, PIECE_FEAT_DIM };
