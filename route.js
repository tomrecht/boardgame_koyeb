/* Route distances, ported from game.py (step 3 of PORTING.md).
 *
 * These three are the only part of the encoder that has to be genuinely
 * ported rather than exported: they are called from INSIDE the encoder, and
 * they were measured as ~35% of a self-play worker's time, so they are also
 * the part most worth getting right.
 *
 *   _get_blocked_key       -> blockedTiles(state, player)
 *   shortest_route_to_goal -> shortestRouteToGoal(...)
 *   all_goal_distances     -> allGoalDistances(...)
 *
 * The tile graph comes from encoder_static.json, so the ordering hazards that
 * made the static half an export rather than a port do not arise here. Note
 * the graph excludes nogo tiles, which is equivalent for these searches: nogo
 * is never traversable and never a goal, so dropping it cannot change a path.
 *
 * Unreachable is Infinity, matching Python's float('inf'). The fixture stores
 * it as null, since JSON has no infinity.
 */

function buildGraph(staticData) {
    const n = staticData.num_tiles;
    const neighbors = Array.from({ length: n }, () => []);
    const [src, dst] = staticData.tile_edge_index;
    for (let e = 0; e < src.length; e++) neighbors[src[e]].push(dst[e]);
    const byCoord = new Map();
    staticData.tile_coords.forEach(([ring, sector], i) => byCoord.set(ring + ',' + sector, i));
    return {
        n,
        neighbors,
        types: staticData.tile_types,
        numbers: staticData.tile_numbers,
        byCoord,
        home: staticData.tile_types.indexOf('home'),
        indexOf(ring, pos) {
            const i = byCoord.get(ring + ',' + pos);
            return i === undefined ? -1 : i;
        },
    };
}

/* Tiles that block `player`: a field tile holding MORE THAN ONE enemy piece.
   A single enemy piece is a blot, not a block. */
function blockedTiles(graph, pieces, player) {
    const counts = new Map();          // tile -> {owner, n}
    for (const p of pieces) {
        if (p.tile < 0) continue;
        const c = counts.get(p.tile);
        if (c) c.n++;
        else counts.set(p.tile, { owner: p.player, n: 1 });
    }
    const blocked = new Set();
    for (const [tile, c] of counts) {
        if (graph.types[tile] === 'field' && c.n > 1 && c.owner !== player) blocked.add(tile);
    }
    return blocked;
}

const isSaveableOn = (graph, tile, number) =>
    tile >= 0 && graph.types[tile] === 'save' && (number > 6 || number === graph.numbers[tile]);

/* Distance in tiles to the nearest goal this piece could be saved from.
   0 if it is already sitting on one (or already saved). */
function shortestRouteToGoal(graph, piece, blocked) {
    if (piece.saved || isSaveableOn(graph, piece.tile, piece.number)) return 0;
    const start = piece.tile >= 0 ? piece.tile : graph.home;
    const visited = new Uint8Array(graph.n);
    visited[start] = 1;
    let frontier = [start], distance = 0;
    while (frontier.length) {
        const next = [];
        for (const tile of frontier) {
            for (const nb of graph.neighbors[tile]) {
                if (visited[nb]) continue;
                visited[nb] = 1;
                // The goal test comes BEFORE the traversability test, so a goal
                // is found even though goals are not walked through.
                if (isSaveableOn(graph, nb, piece.number)) return distance + 1;
                const t = graph.types[nb];
                if (t !== 'nogo' && t !== 'home' && !blocked.has(nb)) next.push(nb);
            }
        }
        frontier = next;
        distance++;
    }
    return Infinity;
}

/* {1..6: distance}. One BFS collects every goal at once for an unnumbered
   piece; a numbered one can only ever use its own goal. */
function allGoalDistances(graph, piece, blocked) {
    const out = {};
    for (let g = 1; g <= 6; g++) out[g] = Infinity;

    if (piece.saved) {                                  // done: 0 to everything
        for (let g = 1; g <= 6; g++) out[g] = 0;
        return out;
    }
    if (isSaveableOn(graph, piece.tile, piece.number)) { // on a goal it can use
        if (piece.tile >= 0 && graph.types[piece.tile] === 'save') out[graph.numbers[piece.tile]] = 0;
        return out;
    }
    if (piece.number <= 6) {
        out[piece.number] = shortestRouteToGoal(graph, piece, blocked);
        return out;
    }

    const start = piece.tile >= 0 ? piece.tile : graph.home;
    const visited = new Uint8Array(graph.n);
    visited[start] = 1;
    let frontier = [start], distance = 0, found = 0;
    while (frontier.length && found < 6) {
        const next = [];
        for (const tile of frontier) {
            for (const nb of graph.neighbors[tile]) {
                if (visited[nb]) continue;
                visited[nb] = 1;
                if (graph.types[nb] === 'save' && out[graph.numbers[nb]] === Infinity) {
                    out[graph.numbers[nb]] = distance + 1;
                    found++;
                }
                const t = graph.types[nb];
                if (t !== 'nogo' && t !== 'home' && !blocked.has(nb)) next.push(nb);
            }
        }
        frontier = next;
        distance++;
    }
    return out;
}

/* A fixture snapshot -> the piece shape these functions want. */
function piecesFromSnapshot(graph, snapshot) {
    return snapshot.pieces.map(p => ({
        player: p.player,
        number: p.number,
        saved: p.kind === 'saved',
        unentered: p.kind === 'unentered',
        tile: p.kind === 'tile' ? graph.indexOf(p.ring, p.pos) : -1,
    }));
}

// Usable from Node (tests) and from the browser (the app), without a
// bundler: CommonJS when there is a module object, globals otherwise.
const _api = { buildGraph, blockedTiles, shortestRouteToGoal, allGoalDistances,
                   isSaveableOn, piecesFromSnapshot };
if (typeof module !== 'undefined' && module.exports) module.exports = _api;
else Object.assign(typeof window !== 'undefined' ? window : self, _api);

