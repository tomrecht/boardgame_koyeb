/* The game engine, ported from game.py (step 6.1 of PORTING.md).
 *
 * The 2-ply search is written against get_valid_moves / apply_move /
 * undo_last_move, so it needs a real engine, not just the encoder. game.js's
 * board model cannot serve: it is built out of Phaser display objects, it has
 * no search-shaped undo, and it deliberately offers a rack reordering the
 * engine does not (see CLAUDE.md). This is a plain-data twin of game.py's
 * Board instead, sharing nothing with game.js.
 *
 * Geometry comes from encoder_static.json, whose tile order is IDENTICAL to
 * tile_neighbors.json's file order and therefore to game.py's tile indices
 * (asserted, 70/70) -- so a tile index means the same thing on both sides and
 * no second geometry file is needed. Neighbour ORDER does not matter: the one
 * place game.py walks neighbours to build a move list wraps the result in
 * set(), so only the adjacency itself is load-bearing.
 *
 * Verified against dump_engine_fixture.py: the move SET per position, the
 * position after each move, and that undo restores the start exactly.
 */
const _RT = (typeof require !== 'undefined') ? require('./route.js')
                                             : (typeof window !== 'undefined' ? window : self);

const NUM_PIECES = 12;
const NO_SAVE_TURNS_FOR_DRAW = 10;

class Engine {
    constructor(staticData) {
        this.graph = _RT.buildGraph(staticData);
        // Neighbour ORDER is load-bearing after all, so prefer the exported
        // lists (tile_neighbors.json's own order) over the ones route.js
        // derives from tile_edge_index, whose column order came out of a Python
        // set. It makes no difference to a shortest-path LENGTH, which is why
        // the route port passed without it -- but the heuristic's blot count is
        // a FIFO BFS where the first predecessor to reach a tile fixes its
        // count, and 946 evaluations differed on exactly that component.
        if (staticData.tile_neighbors) this.graph.neighbors = staticData.tile_neighbors;
        this.coords = staticData.tile_coords;      // index -> [ring, pos]
        this.home = this.graph.home;
        this.pieces = [];
        this.racks = { white_unentered: [], black_unentered: [], white_saved: [], black_saved: [] };
        this.occ = Array.from({ length: this.graph.n }, () => []);
        this.dice = [{ value: 1, used: false }, { value: 1, used: false }];
        this.currentPlayer = 'white';
        this.firstMove = null;
        this.stages = { white: 'opening', black: 'opening' };
        this.drawCallable = false;
        this.drawCalled = false;
        this.moves = [];
        this.lookup = new Map();
    }

    /* Rebuild from the fixture's `state` (see dump_engine_fixture.py). */
    static fromState(staticData, st) {
        const e = new Engine(staticData);
        const rackOf = (player, kind) => e.racks[player + '_' + kind];
        // Racks first, in their recorded ORDER -- entry order decides which
        // piece may come out, and undo restores a piece to its own slot.
        for (const player of ['white', 'black']) {
            for (const kind of ['unentered', 'saved']) {
                for (const number of st.racks[player + '_' + kind]) {
                    const p = { player, number, tile: -1, rack: kind };
                    e.pieces.push(p);
                    rackOf(player, kind).push(p);
                }
            }
        }
        // Then the pieces standing on tiles, in the recorded occupancy order.
        for (const [idx, list] of Object.entries(st.occupancy)) {
            for (const [player, number] of list) {
                const p = { player, number, tile: +idx, rack: null };
                e.pieces.push(p);
                e.occ[+idx].push(p);
            }
        }
        e.dice = st.dice.map(([value, used]) => ({ value, used }));
        e.currentPlayer = st.current_player;
        e.stages = Object.assign({}, st.stages);
        e.drawCalled = !!st.draw_called;
        e.drawCallable = !!st.draw_callable;
        if (st.first_move) {
            const [player, number, originTile] = st.first_move;
            e.firstMove = { piece: e.find(player, number), originTile };
        }
        e.reindex();
        return e;
    }

    reindex() {
        // Board.assign_piece_indices sorts white first, then by number.
        this.pieces.sort((a, b) => (a.player === 'white' ? 0 : 1) - (b.player === 'white' ? 0 : 1)
                                   || a.number - b.number);
        this.pieces.forEach((p, i) => { p.index = i + 1; });
        this.lookup = new Map();
        for (const p of this.pieces) this.lookup.set(p.player + ',' + p.number, p);
    }

    find(player, number) { return this.lookup.get(player + ',' + number) || null; }
    other(player) { return player === 'white' ? 'black' : 'white'; }
    unenteredRack(player) { return this.racks[player + '_unentered']; }
    savedRack(player) { return this.racks[player + '_saved']; }
    isSaved(p) { return p.rack === 'saved'; }

    /* --- queries ------------------------------------------------------- */

    canBeSaved(p) {
        if (p.rack === 'saved') return true;
        if (p.tile < 0) return false;
        return this.graph.types[p.tile] === 'save'
            && (p.number > 6 || p.number === this.graph.numbers[p.tile]);
    }

    isBlocked(tile, player) {
        const list = this.occ[tile];
        return this.graph.types[tile] === 'field' && list.length > 1
            && list[0].player !== (player || this.currentPlayer);
    }

    getGameStage(player) {
        if (this.unenteredRack(player).length > 0) return 'opening';
        const mine = this.pieces.filter(p => p.player === player);
        return mine.every(p => this.canBeSaved(p)) ? 'endgame' : 'midgame';
    }

    checkGameOver() {
        if (this.drawCalled) return ['draw', 0];
        const w = this.savedRack('white').length, b = this.savedRack('black').length;
        if (w === NUM_PIECES) return ['white', NUM_PIECES - b];
        if (b === NUM_PIECES) return ['black', NUM_PIECES - w];
        return [null, null];
    }

    /* The ENGINE offers the front rack piece only. The frontend lets a human
       take the second one first, but that is a pure reordering -- see game.py's
       get_enterable_pieces. */
    getEnterablePieces() {
        return this.unenteredRack(this.currentPlayer).slice(0, 1);
    }

    mustMoveUnentered() {
        if (this.unenteredRack(this.currentPlayer).length === 0) return false;
        if (this.occ[this.home].some(p => p.player === this.currentPlayer)) return false;
        if (this.firstMove) return false;
        return true;
    }

    /* Which die numbers can lift this piece off its goal. */
    getSavingDie(piece) {
        if (this.stages[piece.player] === 'opening') return [];
        const t = piece.tile;
        if (t < 0 || this.graph.types[t] !== 'save') return [];
        if (!(piece.number > 6 || piece.number === this.graph.numbers[t])) return [];
        const goalNum = this.graph.numbers[t];
        let dice;
        if (this.stages[piece.player] === 'endgame' && piece.number > 6) {
            // A blank goes out on a HIGHER die too, but only from the highest
            // goal this player occupies -- otherwise the higher goal's own
            // piece would be stranded.
            let highest = -1;
            for (let i = 0; i < this.graph.n; i++) {
                if (this.graph.types[i] !== 'save' || !this.occ[i].length) continue;
                if (this.occ[i].some(p => p.player === piece.player))
                    highest = Math.max(highest, this.graph.numbers[i]);
            }
            dice = this.dice.filter(d => !d.used &&
                (d.value === goalNum || (d.value > goalNum && goalNum === highest)));
        } else {
            dice = this.dice.filter(d => !d.used && d.value === goalNum);
        }
        return dice.map(d => d.value);
    }

    /* Tiles exactly `steps` away, walking only through traversable tiles.
       NOTE the blocked test uses the CURRENT PLAYER, not the piece's owner --
       game.py calls is_blocked() with no argument here. */
    getReachableTiles(start, steps) {
        if (start < 0) return [];
        const visited = new Uint8Array(this.graph.n);
        visited[start] = 1;
        let frontier = [start], out = [];
        for (let d = 0; d < steps; d++) {
            const next = [];
            for (const tile of frontier) {
                for (const nb of this.graph.neighbors[tile]) {
                    if (visited[nb]) continue;
                    const t = this.graph.types[nb];
                    if (t === 'nogo' || t === 'home' || this.isBlocked(nb)) continue;
                    visited[nb] = 1;
                    next.push(nb);
                    if (d + 1 === steps) out.push(nb);
                }
            }
            frontier = next;
        }
        return steps === 0 ? [start] : out;
    }

    /* {dieValue: [tileIdx | 'save']}. A Map, because doubles collapse the two
       die keys into one -- exactly as the Python dict does. */
    getReachableTilesByDice(piece) {
        const res = new Map([[this.dice[0].value, []], [this.dice[1].value, []]]);
        const start = piece.rack === 'unentered' ? this.home : piece.tile;
        const sumOf = () => {
            const origin = (this.firstMove && this.firstMove.originTile >= 0)
                ? this.firstMove.originTile : this.home;
            return new Set(this.getReachableTiles(origin, this.dice[0].value + this.dice[1].value));
        };
        for (const die of this.dice) {
            if (die.used) continue;
            let tiles = this.getReachableTiles(start, die.value);
            if (this.firstMove && this.firstMove.piece === piece) {
                // The turn's first piece may continue, but only to somewhere the
                // dice SUM would also have reached from where it started.
                const sum = sumOf();
                tiles = tiles.filter(t => sum.has(t));
            }
            res.set(die.value, tiles);
        }
        if (piece.tile >= 0 && this.graph.types[piece.tile] === 'save'
                && this.stages[piece.player] !== 'opening') {
            for (const roll of this.getSavingDie(piece)) {
                if (res.has(roll) && !res.get(roll).includes('save')) res.get(roll).push('save');
            }
        }
        return res;
    }

    getValidMoves() {
        // Side effect, faithfully reproduced: game.py recomputes the current
        // player's stage here, and other code reads game_stages afterwards.
        this.stages[this.currentPlayer] = this.getGameStage(this.currentPlayer);
        if (this.dice[0].used && this.dice[1].used) return [];

        const captured = this.occ[this.home].filter(p => p.player === this.currentPlayer);
        let movers;
        if (captured.length) {
            movers = captured;
        } else if (this.mustMoveUnentered()) {
            movers = this.getEnterablePieces();
        } else {
            movers = this.pieces.filter(p => p.player === this.currentPlayer && p.tile >= 0
                && (this.graph.types[p.tile] === 'field' || this.graph.types[p.tile] === 'save'));
            movers = movers.concat(this.getEnterablePieces());
            // Blank pieces on the same tile are interchangeable: keep one.
            const seen = new Set(), deduped = [];
            for (const p of movers) {
                if (p.number > 6 && p.tile >= 0) {
                    if (seen.has(p.tile)) continue;
                    seen.add(p.tile);
                }
                deduped.push(p);
            }
            movers = deduped;
        }

        const out = [];
        for (const piece of movers) {
            for (const [roll, dests] of this.getReachableTilesByDice(piece)) {
                for (const dest of dests) {
                    out.push({ piece: [piece.player, piece.number],
                               dest: dest === 'save' ? 'save' : this.coordOf(dest), roll });
                }
            }
        }

        if (!captured.length && !this.mustMoveUnentered()) {
            out.push({ piece: null, lone: 0, dest: 0, roll: 0 });        // pass
            if (!this.dice[0].used && !this.dice[1].used
                    && this.stages[this.currentPlayer] !== 'opening') {
                // Peel ONE enemy piece off a block into their saved rack, at the
                // cost of both dice. One move per piece on the block.
                for (const p of this.pieces) {
                    if (p.player === this.currentPlayer || p.tile < 0) continue;
                    if (this.graph.types[p.tile] !== 'field' || this.occ[p.tile].length <= 1) continue;
                    out.push({ piece: [p.player, p.number], dest: 0, roll: 0 });
                }
            }
        }
        if (!this.dice[0].used && !this.dice[1].used && this.drawCallable) {
            out.push({ piece: null, lone: 1, dest: 1, roll: 1 });        // call a draw
        }
        return out;
    }

    coordOf(tile) { return this.coords[tile]; }

    /* --- mutation ------------------------------------------------------ */

    applyMove(move, switchTurn) {
        const isPass = move.piece === null && move.lone === 0;
        const isDraw = move.piece === null && move.lone === 1;
        if (isPass) {
            this.firstMove = null;
            if (switchTurn) this.currentPlayer = this.other(this.currentPlayer);
            return;
        }
        if (isDraw) { this.drawCalled = true; return; }

        const firstMoveBefore = this.firstMove;
        const piece = this.find(move.piece[0], move.piece[1]);
        if (!piece) return;
        let originTile = -1, originRack = null, originRackIndex = null, capturedPiece = null;

        if (move.dest === 0 && move.roll === 0) {           // block-save
            const rack = this.savedRack(piece.player);
            originTile = piece.tile;
            this.removeFromTile(piece);
            piece.rack = 'saved';
            rack.push(piece);
            this.moves.push({ piece, originTile, originRack: null, originRackIndex: null,
                              dest: 0, capturedPiece: null, roll: 0, firstMoveBefore });
            this.dice[0].used = true;
            this.dice[1].used = true;
            this.stages[this.currentPlayer] = this.getGameStage(this.currentPlayer);
            return;
        } else if (move.dest === 'save') {
            const rack = this.savedRack(piece.player);
            if (piece.tile >= 0) { originTile = piece.tile; this.removeFromTile(piece); }
            piece.rack = 'saved';
            rack.push(piece);
            this.stages[this.currentPlayer] = this.getGameStage(this.currentPlayer);
        } else {
            const newTile = this.graph.indexOf(move.dest[0], move.dest[1]);
            if (piece.rack) {
                originRack = piece.player + '_' + piece.rack;
                originRackIndex = this.racks[originRack].indexOf(piece);
                this.racks[originRack].splice(originRackIndex, 1);
                piece.rack = null;
            }
            if (piece.tile >= 0) { originTile = piece.tile; this.removeFromTile(piece); }
            if (!this.firstMove) this.firstMove = { piece, originTile };
            const occ = this.occ[newTile];
            if (this.graph.types[newTile] === 'field' && occ.length && occ[0].player !== piece.player) {
                capturedPiece = occ.pop();                  // the LAST piece, as in game.py
                capturedPiece.tile = this.home;
                this.occ[this.home].push(capturedPiece);
            }
            occ.push(piece);
            piece.tile = newTile;
            if (originRack !== null) this.stages[this.currentPlayer] = this.getGameStage(this.currentPlayer);
        }

        if (move.roll === this.dice[0].value && !this.dice[0].used) this.dice[0].used = true;
        else if (move.roll === this.dice[1].value && !this.dice[1].used) this.dice[1].used = true;
        this.moves.push({ piece, originTile, originRack, originRackIndex,
                          dest: move.dest, capturedPiece, roll: move.roll, firstMoveBefore });
        if (switchTurn && this.dice.every(d => d.used)) this.switchTurn();
    }

    undoLastMove() {
        if (!this.moves.length) return;
        const m = this.moves.pop();
        const piece = m.piece;
        if (m.dest === 'save' || m.dest === 0) {
            const rack = this.savedRack(piece.player);
            rack.splice(rack.indexOf(piece), 1);
            piece.rack = null;
            piece.tile = m.originTile;
            this.occ[m.originTile].push(piece);
            if (m.dest === 0) {
                // A block-save marks BOTH dice used but carries roll 0, which can
                // never match a die below -- so both must be cleared explicitly.
                // It is only ever legal with both dice unused, so this is exact.
                this.dice[0].used = false;
                this.dice[1].used = false;
            }
        } else {
            const newTile = this.graph.indexOf(m.dest[0], m.dest[1]);
            const list = this.occ[newTile];
            list.splice(list.indexOf(piece), 1);
            piece.tile = -1;
            if (m.originTile >= 0) {
                this.occ[m.originTile].push(piece);
                piece.tile = m.originTile;
            } else if (m.originRack !== null) {
                // Back to its OWN slot: unshifting to the front silently
                // reorders the rack now that a non-front piece can leave.
                const rack = this.racks[m.originRack];
                const i = m.originRackIndex === null ? 0 : Math.min(m.originRackIndex, rack.length);
                rack.splice(i, 0, piece);
                piece.rack = m.originRack.split('_')[1];
            }
            if (m.capturedPiece) {
                const h = this.occ[this.home];
                h.splice(h.indexOf(m.capturedPiece), 1);
                m.capturedPiece.tile = newTile;
                list.push(m.capturedPiece);
            }
        }
        if (m.roll === this.dice[0].value && this.dice[0].used) this.dice[0].used = false;
        else if (m.roll === this.dice[1].value && this.dice[1].used) this.dice[1].used = false;
        this.firstMove = m.firstMoveBefore;
        // A block-save record names the OPPONENT's piece, so the turn goes back
        // to the other side, not to piece.player.
        this.currentPlayer = m.dest === 0 ? this.other(piece.player) : piece.player;
        if (m.dest === 'save' || m.originRack !== null) {
            this.stages[this.currentPlayer] = this.getGameStage(this.currentPlayer);
        }
    }

    removeFromTile(piece) {
        const list = this.occ[piece.tile];
        list.splice(list.indexOf(piece), 1);
        piece.tile = -1;
    }

    switchTurn() {
        this.firstMove = null;
        this.currentPlayer = this.other(this.currentPlayer);
        this.applyLastPieceRule();
    }

    /* Alone in the endgame, a numbered piece loses its number (becoming
       saveable by any die of that number or more). It is RENUMBERED to 13,
       which changes its lookup key. */
    applyLastPieceRule() {
        for (const player of ['white', 'black']) {
            if (this.stages[player] !== 'endgame') continue;
            const unsaved = this.pieces.filter(p => p.player === player && p.rack !== 'saved');
            if (unsaved.length === 1 && unsaved[0].number <= 6) {
                const p = unsaved[0];
                this.lookup.delete(p.player + ',' + p.number);
                p.number = NUM_PIECES + 1;
                this.lookup.set(p.player + ',' + p.number, p);
            }
        }
    }

    /* --- comparison ----------------------------------------------------- */

    /* The same shape dump_engine_fixture.py's fingerprint() writes. */
    fingerprint() {
        const where = (p) => p.tile >= 0 ? ['tile', p.tile]
                           : p.rack === 'saved' ? ['saved', -1]
                           : p.rack ? ['rack', -1] : ['none', -1];
        const pieces = this.pieces.map(p => [p.player, p.number, ...where(p)])
            .sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0) || a[1] - b[1]);
        const occupancy = {};
        for (let i = 0; i < this.graph.n; i++) {
            if (!this.occ[i].length) continue;
            occupancy[String(i)] = this.occ[i].map(p => [p.player, p.number])
                .sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0) || a[1] - b[1]);
        }
        const numbersIn = (k) => this.racks[k].map(p => p.number);
        return {
            pieces,
            occupancy,
            racks: {
                white_unentered: numbersIn('white_unentered'),
                black_unentered: numbersIn('black_unentered'),
                white_saved: numbersIn('white_saved'),      // save ORDER, see the dumper
                black_saved: numbersIn('black_saved'),
            },
            dice: this.dice.map(d => [d.value, !!d.used]),
            current_player: this.currentPlayer,
            stages: Object.assign({}, this.stages),
            first_move: this.firstMove
                ? [this.firstMove.piece.player, this.firstMove.piece.number, this.firstMove.originTile]
                : null,
            draw_called: !!this.drawCalled,
            game_over: this.checkGameOver(),
        };
    }
}

/* Total order on moves. Python's get_valid_moves order is not reproducible
   (it walks a set of Tile objects), so only the SET is ever compared -- and
   agent_gnn breaks its ties on this same ordering. */
function moveKey(m) {
    if (m.piece === null) return [0, '', m.lone, 0, m.roll];
    const [player, number] = m.piece;
    if (m.dest === 'save') return [1, player, number, 0, m.roll];
    if (m.dest === 0) return [2, player, number, 0, m.roll];
    return [3, player, number, m.dest[0] * 1000 + m.dest[1], m.roll];
}
const moveCmp = (a, b) => {
    const ka = moveKey(a), kb = moveKey(b);
    for (let i = 0; i < ka.length; i++) {
        if (ka[i] < kb[i]) return -1;
        if (ka[i] > kb[i]) return 1;
    }
    return 0;
};

// Usable from Node (tests) and the browser, without a bundler. IIFE-wrapped:
// classic scripts share one global lexical scope, so a bare top-level const of
// a name another file also uses is a silent SyntaxError.
(function () {
    const api = { Engine, moveKey, moveCmp, NUM_PIECES, NO_SAVE_TURNS_FOR_DRAW };
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    else Object.assign(typeof window !== 'undefined' ? window : self, api);
})();
