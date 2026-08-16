/* On-device move selection (step 7 of PORTING.md).
 *
 * The port itself is finished and proven -- route/encoder/infer/engine/
 * heuristic/agent.js reproduce the served Python agent's chosen pair 110/110
 * over two real games (PORTING.md 6.4). This file is the wiring: it loads that
 * stack lazily, and answers the one question game.js actually asks, which is
 * "what does the computer play here?".
 *
 * The design, from PORTING.md:
 *   - BOTH platforms, not phone-only. Leaving desktop on /select_moves keeps
 *     the Koyeb instance and MOVE_BUDGET alive and maintains two AI paths for
 *     ever, which is the whole cost the port was meant to remove.
 *
 * **THE SERVER IS NO LONGER A FALLBACK (2026-08-16).** This used to start each
 * session on /select_moves and switch over once loaded. It does not any more:
 * the computer is answered HERE, always, and game.js waits for this to finish
 * loading rather than asking anything else. That is what makes the app a folder
 * of static files. Two consequences the old design did not have to handle:
 *   - The load is kicked off as soon as a game starts with a computer player,
 *     not on the first move request, so the ~4.5 MB (2.9 MB gzipped runtime +
 *     1.66 MB model) is usually in flight while the human takes their turn.
 *   - A failed LOAD is retried (see loadFailed). With a server behind it, one
 *     dropped fetch cost a moment; with nothing behind it, burning the session
 *     on a transient network error would leave the player with no opponent at
 *     all. A bad ANSWER still disables permanently -- see disable() -- because
 *     a port bug will not fix itself on the next try.
 *
 * The one genuinely new piece of logic is `engineState`: game.js's board model
 * is not the engine's, and the snapshot has to describe what the ENGINE would
 * see. It is written as a direct mirror of game.py's `Board.update_state`, so
 * the local agent takes the SAME payload the server does -- which is what makes
 * `?aicompare=1` below a real differential test rather than an approximation.
 *
 * `?localai=0` turns this off. It no longer falls back to a working server --
 * with the routes gone from the deployment there is nothing behind it, so the
 * flag now means "no computer opponent" unless a dev server is running. It is
 * kept for exactly that case, alongside `?aicompare=1` and `?aiserver=`, which
 * are how the port was verified and remain the tools for re-checking it.
 *
 * KNOWN COST: inference runs on the main thread, so the UI is frozen for the
 * duration of a move (measured median 0.29s desktop, 1.25s at 4x throttle).
 * Only the computer's turn is affected, when there is nothing to interact with.
 * Moving it into a Worker is the follow-up -- every port file already falls back
 * to `self`, so they are worker-ready by construction.
 */
const LocalAgent = (function () {
    'use strict';

    // Order is not load-bearing (each file aliases the global object rather than
    // importing at parse time) but it is the dependency order, so keep it.
    const FILES = ['ort/ort.wasm.min.js', 'route.js', 'encoder.js', 'infer.js',
                   'engine.js', 'heuristic.js', 'agent.js'];

    // A classic script that fails to PARSE still fires `load`, so the script
    // element tells us nothing -- a redeclaration against one of game.js's ~250
    // top-level names would look like a clean load and leave the exports
    // undefined. Checking for the exports themselves is the only honest test.
    // (This is not hypothetical: engine.js's NO_SAVE_TURNS_FOR_DRAW collided
    // with game.js's and had to be renamed.)
    const REQUIRED = ['ort', 'buildGraph', 'encode', 'Infer', 'Engine', 'evaluate',
                      'selectMovePair', 'moveCmp'];

    const NO_SAVE_TURNS_FOR_DRAW_FALLBACK = 10;
    const TOTAL_PIECES = 12;          // margin units for the eval readout

    let status = 'idle';            // idle | loading | ready | failed | off
    let staticData = null;
    let weights = null;
    let graph = null;
    let initPromise = null;
    let lastError = null;
    let serverUrl = '';
    let moveCount = 0;

    function param(name) {
        try { return new URLSearchParams(window.location.search).get(name); }
        catch (e) { return null; }
    }

    /* Off entirely for this session: either asked for, or already burnt. */
    function enabled() {
        if (param('localai') === '0') return false;
        return status !== 'failed' && status !== 'off';
    }
    function ready() { return status === 'ready'; }
    function state() { return { status, error: lastError && String(lastError), moves: moveCount,
                                loadFailures: initFailures }; }

    /* Permanent for the session. Used for a BAD ANSWER -- a port bug is not
       going to fix itself, and re-asking would just be wrong twice. */
    function disable(err) {
        lastError = err || lastError;
        status = 'failed';
        console.warn('[local-ai] disabled for this session:', lastError);
    }

    /* A failed LOAD is different: there is no server to fall back to any more,
       so one dropped fetch of the 11MB wasm must not cost the whole session's
       computer opponent. Reset to idle and let the next call try again, up to a
       few times, then treat it as burnt. */
    const MAX_LOAD_FAILURES = 3;
    let initFailures = 0;
    function loadFailed(err) {
        lastError = err || lastError;
        initFailures += 1;
        initPromise = null;
        if (initFailures >= MAX_LOAD_FAILURES) {
            status = 'failed';
            console.warn('[local-ai] giving up after ' + initFailures + ' load attempts:', lastError);
        } else {
            status = 'idle';
            console.warn('[local-ai] load attempt ' + initFailures + ' failed, will retry:', lastError);
        }
    }

    function loadScript(src) {
        return new Promise((resolve, reject) => {
            const el = document.createElement('script');
            el.src = src;
            el.async = false;                     // keep execution order
            el.onload = () => resolve();
            el.onerror = () => reject(new Error('could not load ' + src));
            document.head.appendChild(el);
        });
    }

    async function getJSON(url) {
        const r = await fetch(url);
        if (!r.ok) throw new Error(url + ' responded ' + r.status);
        return r.json();
    }

    /* onnxruntime dynamically IMPORTS the .mjs beside the wasm, so this must be
       root-absolute (or a full URL) -- a bare relative path is not a valid module
       specifier and fails as "no available backend found", which reads like a
       missing build rather than a bad path. Derived from the document base so a
       deploy under a subpath still works. */
    function ortPath() {
        try { return new URL('ort/', document.baseURI).pathname; }
        catch (e) { return '/ort/'; }
    }

    /* Idempotent. Resolves true when the local agent may be used, false when it
       may not -- it never rejects, because no caller should have to handle the
       background loader failing. */
    function init(opts) {
        // `?aiserver=` wins over whatever game.js passes, and is why it exists:
        // game.js hardcodes SERVER_URL to localhost:10000 for any local page, so
        // a harness testing against a second Flask instance had its comparison
        // silently retargeted at the first one on every turn (init is called per
        // move, and this line used to re-apply the argument each time). That
        // made a stale server's answers look like a port bug.
        serverUrl = param('aiserver') || (opts && opts.serverUrl) || serverUrl;
        if (initPromise) return initPromise;
        if (!enabled()) return Promise.resolve(false);
        status = 'loading';
        const t0 = (window.performance || Date).now();
        initPromise = (async () => {
            for (const f of FILES) await loadScript(f);
            const missing = REQUIRED.filter(n => typeof window[n] === 'undefined');
            if (missing.length) throw new Error('port files loaded but did not export: ' + missing.join(', '));

            const [sd, w] = await Promise.all([
                getJSON('encoder_static.json'),
                getJSON('heuristic_weights.json'),
            ]);
            staticData = sd;
            weights = w;
            graph = window.buildGraph(staticData);

            await window.Infer.init({
                ort: window.ort,
                modelUrl: 'model.onnx',
                staticUrl: 'encoder_static.json',
                wasmPath: ortPath(),
            });
            status = 'ready';
            console.log('[local-ai] ready in ' + Math.round((window.performance || Date).now() - t0) + 'ms'
                        + ' -- the computer now moves on this device');
            return true;
        })().catch(err => { loadFailed(err); return false; });
        return initPromise;
    }

    /* --- game.js's board -> the engine's ---------------------------------- */

    /* A direct mirror of game.py's Board.update_state, taking the very same
       payload getGameState() posts to /select_moves. Every departure from it
       would be a silent divergence, so the differences that ARE deliberate are
       none: the stages are recomputed and the last-piece rule applied in the
       same order update_state does, immediately after fromState's reindex()
       (which is assign_piece_indices).

       Two things carried over verbatim because they are observable:
       - `reachableBySum`'s PRESENCE on a board piece is what marks the turn's
         first mover; the last such piece wins, exactly as the Python loop's
         repeated assignment does. Its origin_tile is the tile the piece stands
         on NOW, not where it came from -- the rebuild has no history, and the
         served agent has always behaved this way.
       - occupancy keeps the payload's order within a tile. Not semantic (every
         reader is `pieces[0].player` on a field tile, or a pop() from a tile
         holding one enemy) but free to preserve. */
    function engineState(gs) {
        const occupancy = {};
        let firstMove = null;
        for (const bp of (gs.boardPieces || [])) {
            const idx = graph.indexOf(bp.tile.ring, bp.tile.sector);
            if (idx < 0) throw new Error('no tile at ring ' + bp.tile.ring + ' sector ' + bp.tile.sector);
            (occupancy[idx] || (occupancy[idx] = [])).push([bp.color, bp.number]);
            if ('reachableBySum' in bp) firstMove = [bp.color, bp.number, idx];
        }
        const numbers = (list) => (list || []).map(p => p.number);
        return {
            racks: {
                white_unentered: numbers(gs.racks.whiteUnentered),
                white_saved: numbers(gs.racks.whiteSaved),
                black_unentered: numbers(gs.racks.blackUnentered),
                black_saved: numbers(gs.racks.blackSaved),
            },
            occupancy,
            dice: gs.dice.map(d => [d.value, !!d.used]),
            current_player: gs.currentTurn,
            // Placeholders: update_state computes both stages fresh below, and
            // the encoder is documented never to trust a carried-in stage.
            stages: { white: 'opening', black: 'opening' },
            first_move: firstMove,
            draw_called: false,
            // The frontend owns the no-save counter; update_state adopts what it
            // reports, falling back to the threshold only when it is absent.
            draw_callable: gs.drawCallable === undefined
                ? (gs.noSaveTurns || 0) >= NO_SAVE_TURNS_FOR_DRAW_FALLBACK
                : !!gs.drawCallable,
        };
    }

    function engineFor(gs) {
        const e = window.Engine.fromState(staticData, engineState(gs));
        e.stages.white = e.getGameStage('white');
        e.stages.black = e.getGameStage('black');
        e.applyLastPieceRule();
        return e;
    }

    /* The agent's move objects -> the 3-element arrays app.py's jsonify produces
       from Python move tuples, which is what applyMovePair() reads. */
    function toWire(m) {
        if (m.piece === null) return [m.lone, m.dest, m.roll];   // pass (0,0,0) / draw (1,1,1)
        return [[m.piece[0], m.piece[1]], m.dest, m.roll];
    }

    /* Returns the move pair in wire format, or null when there is nothing to
       play (the server answers that with no `move` key). Rejects on any
       internal failure so the caller can fall back to the server. */
    async function selectMoves(gs) {
        if (status !== 'ready') throw new Error('local agent not ready');
        const engine = engineFor(gs);
        const player = engine.currentPlayer;
        const moves = engine.getValidMoves();
        if (!moves.length) return null;
        const pair = await window.selectMovePair(engine, weights, moves, player, {
            difficulty: gs.difficulty === undefined ? 1.0 : gs.difficulty,
            snapshot: (eng) => eng.snapshot(),
            // One batched call per turn, as the search intends.
            score: async (snaps) => window.Infer.score(window.ort, snaps, player),
        });
        if (!Array.isArray(pair) || pair.length !== 2) throw new Error('agent returned ' + JSON.stringify(pair));
        moveCount++;
        return pair.map(toWire);
    }

    /* --- the developer modes, answered on the device ---------------------- */

    /* What /evaluate_board returned, computed here instead, so the eval readout
       (E) survives a static host. Mirrors app.py's route field for field:
       agent.evaluate's raw value, best_play_value's margin, and the heuristic
       total with its per-side components. */
    async function evaluate(gs) {
        if (status !== 'ready') throw new Error('local agent not ready');
        const engine = engineFor(gs);
        const player = engine.currentPlayer;
        const h = window.evaluate(engine, weights, player);
        const [winner, margin] = engine.checkGameOver();
        if (winner) {
            // agent.evaluate short-circuits a finished game rather than asking
            // the net, which never sees game_over and would be out of
            // distribution. gnn_raw is left null, as it is server-side.
            const signed = (winner === player ? 1 : -1) * margin;
            return { message: 'Success', gnn_player: player, gnn_raw: null,
                     gnn_best_margin: signed, heur_score: h.score,
                     player: h.components.player, opponent: h.components.opponent };
        }
        const raw = (await window.Infer.score(window.ort, [engine.snapshot()], player))[0];
        return {
            message: 'Success',
            gnn_player: player,
            gnn_raw: raw,
            gnn_score: raw * window.SCORE_SCALE,
            eval: raw * window.SCORE_SCALE,
            total_score: raw * window.SCORE_SCALE,
            gnn_best_margin: await bestPlayMargin(engine, player),
            heur_score: h.score,
            player: h.components.player,
            opponent: h.components.opponent,
        };
    }

    /* agent_gnn.best_play_value: the margin after `player` plays its best pair,
       in the same units as the readout's "current" figure (raw * NUM_PIECES). */
    async function bestPlayMargin(engine, player) {
        const moves = engine.getValidMoves();
        if (!moves.length) return null;
        const ranked = await window.selectMovePair(engine, weights, moves, player, {
            returnScores: true,
            snapshot: (e) => e.snapshot(),
            score: async (snaps) => window.Infer.score(window.ort, snaps, player),
        });
        if (!ranked.length) return null;
        const best = ranked[0];
        if (!isFinite(best.score)) {
            // A guaranteed winning pair: report the ACTUAL final margin of the
            // won position, not a flat NUM_PIECES.
            const base = engine.moves.length;
            try {
                for (const m of best.pair) if (!window.isPass(m)) engine.applyMove(m, false);
                const [w, s] = engine.checkGameOver();
                if (w) return (w === player ? 1 : -1) * s;
            } finally {
                while (engine.moves.length > base) engine.undoLastMove();
            }
            return TOTAL_PIECES;
        }
        return (best.score / window.SCORE_SCALE) * TOTAL_PIECES;
    }

    /* What /debug_piece_blots returned. Also answers the debug hover tooltip,
       which called /debug_piece_info -- a route that never existed in app.py, so
       that tooltip has been failing silently all along and works again now. */
    function pieceDebug(gs, pieceInfo) {
        if (!graph) throw new Error('local agent not ready');
        const engine = engineFor(gs);
        const p = engine.find(pieceInfo.player, Number(pieceInfo.number));
        if (!p) return { error: 'Piece not found' };
        const blocked = window.blockedTiles(engine.graph, engine.pieces, p.player);
        const distance = window.shortestRouteToGoal(engine.graph,
            { tile: p.tile, number: p.number, saved: p.rack === 'saved' }, blocked);
        return {
            distance: isFinite(distance) ? distance : null,
            blot_count: window.countEnemyBlotsOnShortestPath(engine, p),
            can_be_saved: engine.canBeSaved(p),
        };
    }

    /* --- differential check (?aicompare=1) -------------------------------- */

    /* Asks the server the same question and records any disagreement. This is
       the proof that engineState() above is faithful: the trace-diff already
       proved the agent itself, from engine states dumped by Python, but nothing
       had yet exercised game.js's own board as the source. Off by default --
       it doubles the work and needs the server.

       Results accumulate on window.__aiCompare for a harness to read. */
    const compare = { on: false, n: 0, agreed: 0, failed: 0, diffs: [] };

    // Compared as emitted, order included: the port reproduces the Python
    // agent's tie-break exactly (PORTING.md 6.0/6.4), so even a transposition
    // is a real disagreement worth seeing rather than noise to normalise away.
    const normalise = (pair) => JSON.stringify(pair);

    async function compareWithServer(gs, localPair) {
        compare.n++;
        try {
            const r = await fetch(serverUrl + '/select_moves', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify(gs),
            });
            if (!r.ok) throw new Error('server responded ' + r.status);
            const data = await r.json();
            const server = data.move || null;
            const a = normalise(localPair), b = normalise(server);
            if (a === b) compare.agreed++;
            else compare.diffs.push({ turn: compare.n, local: localPair, server: server,
                                      state: gs });
        } catch (e) {
            compare.failed++;
            console.warn('[local-ai] compare failed', e);
        }
        window.__aiCompare = { n: compare.n, agreed: compare.agreed,
                              failed: compare.failed, diffs: compare.diffs.slice(0, 8) };
        return localPair;
    }

    function comparing() {
        if (!compare.on) compare.on = param('aicompare') === '1';
        return compare.on;
    }

    /* Geometry only, no inference runtime -- what state_test.html needs to
       exercise engineState/engineFor without downloading 4.5 MB to score
       positions it never scores. Not used in play. */
    function _seedStatic(sd) {
        staticData = sd;
        graph = window.buildGraph(sd);
    }

    return { init, enabled, ready, state, disable, selectMoves,
             evaluate, pieceDebug,          // the developer modes (E / D)
             comparing, compareWithServer,
             // exposed for tests/debugging, not used by game.js
             engineState, engineFor, toWire, _seedStatic, _compare: compare };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = LocalAgent;
