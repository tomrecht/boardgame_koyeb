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

/* The encoder's whole input for one position. */
function encode(staticData, snapshot, currentPlayer) {
    const graph = buildGraph(staticData);
    const pieces = piecesFromSnapshot(graph, snapshot);
    const player = currentPlayer || snapshot.current_player;
    return {
        tile_feats: tileFeatures(graph, staticData.base_tile_feats, pieces, player),
        tile_edge_index: staticData.tile_edge_index,
    };
}

module.exports = { buildGraph, blockedTiles, shortestRouteToGoal, allGoalDistances,
                   isSaveableOn, piecesFromSnapshot, tileFeatures, encode, TILE_FEAT_DIM };
