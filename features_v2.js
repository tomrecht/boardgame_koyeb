/* features_v2.js -- the JS twin of features_v2.py (branch train-features-v2).
 *
 * The extra encoder inputs of the A and AB feature sets, appended by encoder.js
 * after the v1 columns when the model says so (model.json, see infer.js):
 *   A   tile +1   opponent permanent wall (2+ opponent BLANKS on a field tile)
 *       piece +1  own-goal distance, numbered pieces only
 *       global +14 no-save counter, opponent saveable, per side unentered /
 *                  field / saved, race pips, turns to finish (dice count), and
 *                  the exact endgame-table value per side
 *   AB  A + piece +4  P(captured next roll), P(route walled next roll), enemy
 *                     blots on the route (>= 1, >= 2)
 *         + global +72 per side x die 1-6: saves numbered / blank, onto-goal
 *                     numbered / blank, captures of numbered / blank enemies
 *
 * Every function mirrors its Python namesake line for line, and the whole set is
 * asserted equal against encoder.py on features_v2_fixture.json (owner's
 * positions, random end-of-turn boards and played-into last-piece positions) by
 * features_v2_test.js. The tile graph is route.js's (encoder_static.json), whose
 * neighbour lists are game.py's tile.neighbors minus nogo -- every BFS here skips
 * nogo anyway, so dropping it cannot change a distance.
 */
const _RV = (typeof require !== 'undefined')
    ? require('./route.js')
    : (typeof window !== 'undefined' ? window : self);

const FV2 = (() => {
    const MAX_DIST = 14.0;
    const NO_SAVE_TURNS_FOR_DRAW = 10;
    const BLANK_BANK = { 1: 1.000, 2: 1.029, 3: 1.125, 4: 1.303, 5: 1.462, 6: 1.644 };
    const NUMBERED_BANK = 3.273;
    const BLANK_BANK_MEAN = (1.000 + 1.029 + 1.125 + 1.303 + 1.462 + 1.644) / 6;
    const PIPS_PER_TURN = 7.0;
    const HOME_TO_GOAL = 7;
    const ROLLS = [];
    for (let a = 1; a <= 6; a++) for (let b = a; b <= 6; b++) ROLLS.push([a, b, (a === b ? 1 : 2) / 36.0]);
    const EXTRA = { v1: [0, 0, 0], A: [1, 1, 14], AB: [1, 5, 86] };

    const other = (p) => (p === 'white' ? 'black' : 'white');

    // Distances depend only on (start, walls) and the static graph, and walls
    // rarely change between a turn's candidate positions -- so, as in Python's
    // _DIST_CACHE / _CHOKE_CACHE, keep them ACROSS positions. Values are pure
    // functions of their keys; the cap only bounds memory.
    const CACHE_MAX = 50000;
    const DIST = new Map(), CHOKE = new Map();
    const capped = (m) => { if (m.size > CACHE_MAX) m.clear(); return m; };

    /* Everything the features read, built once per position. */
    function context(graph, snapshot, pieces) {
        const occ = Array.from({ length: graph.n }, () => []);
        for (const p of pieces) if (p.tile >= 0) occ[p.tile].push(p);
        const goals = {};
        for (let t = 0; t < graph.n; t++) if (graph.types[t] === 'save') goals[graph.numbers[t]] = t;
        const unentered = { white: snapshot.racks.white_unentered || [],
                            black: snapshot.racks.black_unentered || [] };
        return { graph, snapshot, pieces, occ, goals, unentered, distCache: capped(DIST),
                 noSave: snapshot.no_save_turns || 0 };
    }

    const isSaveable = (ctx, p) => p.saved || _RV.isSaveableOn(ctx.graph, p.tile, p.number);

    /* game.py get_game_stage, fresh. */
    function stage(ctx, player) {
        if (ctx.unentered[player].length > 0) return 'opening';
        const mine = ctx.pieces.filter(p => p.player === player);
        return mine.every(p => isSaveable(ctx, p)) ? 'endgame' : 'midgame';
    }

    /* _walled_for: field tiles with 2+ pieces of the OTHER side. */
    function walledFor(ctx, mover) {
        const s = new Set();
        for (let t = 0; t < ctx.graph.n; t++) {
            const o = ctx.occ[t];
            if (ctx.graph.types[t] === 'field' && o.length > 1 && o[0].player !== mover) s.add(t);
        }
        return s;
    }
    const setKey = (s) => Array.from(s).sort((a, b) => a - b).join(',');

    /* _dist_from_raw: BFS from target through tiles that are not nogo, home or
       blocked (the target itself is never tested). Map tile -> distance. */
    function distFrom(ctx, target, blocked, bkey) {
        const key = target + '|' + (bkey !== undefined ? bkey : setKey(blocked));
        let d = ctx.distCache.get(key);
        if (d) return d;
        d = new Map([[target, 0]]);
        const q = [target];
        for (let i = 0; i < q.length; i++) {
            const t = q[i];
            for (const n of ctx.graph.neighbors[t]) {
                if (d.has(n)) continue;
                const ty = ctx.graph.types[n];
                if (ty === 'nogo' || ty === 'home' || blocked.has(n)) continue;
                d.set(n, d.get(t) + 1);
                q.push(n);
            }
        }
        ctx.distCache.set(key, d);
        return d;
    }

    /* _route_len_raw: BFS start -> goal (goal test on pop), or null. */
    function routeLen(ctx, start, goal, blocked) {
        const seen = new Map([[start, 0]]);
        const q = [start];
        for (let i = 0; i < q.length; i++) {
            const t = q[i];
            if (t === goal) return seen.get(t);
            for (const n of ctx.graph.neighbors[t]) {
                if (seen.has(n)) continue;
                const ty = ctx.graph.types[n];
                if (ty === 'nogo' || ty === 'home' || blocked.has(n)) continue;
                seen.set(n, seen.get(t) + 1);
                q.push(n);
            }
        }
        return null;
    }

    /* _dist_from_home: home's neighbours at 1, then outward. */
    function distFromHome(ctx, blocked, bkey) {
        const key = 'home|' + (bkey !== undefined ? bkey : setKey(blocked));
        let d = ctx.distCache.get(key);
        if (d) return d;
        d = new Map();
        const q = [];
        for (const n of ctx.graph.neighbors[ctx.graph.home]) {
            const ty = ctx.graph.types[n];
            if (ty === 'nogo' || ty === 'home' || blocked.has(n) || d.has(n)) continue;
            d.set(n, 1);
            q.push(n);
        }
        for (let i = 0; i < q.length; i++) {
            const t = q[i];
            for (const n of ctx.graph.neighbors[t]) {
                if (d.has(n)) continue;
                const ty = ctx.graph.types[n];
                if (ty === 'nogo' || ty === 'home' || blocked.has(n)) continue;
                d.set(n, d.get(t) + 1);
                q.push(n);
            }
        }
        ctx.distCache.set(key, d);
        return d;
    }

    // ------------------------------------------------------------------ A

    function oppWallFlags(ctx, currentPlayer) {
        const out = new Array(ctx.graph.n).fill(0.0);
        const opp = other(currentPlayer);
        for (let t = 0; t < ctx.graph.n; t++) {
            const o = ctx.occ[t];
            if (ctx.graph.types[t] !== 'field' || o.length < 2) continue;
            if (o.filter(p => p.player === opp && p.number > 6).length >= 2) out[t] = 1.0;
        }
        return out;
    }

    function shortestRoute(ctx, p) {
        return _RV.shortestRouteToGoal(ctx.graph, p, _RV.blockedTiles(ctx.graph, ctx.pieces, p.player));
    }

    function ownGoalDist(ctx, p) {
        if (p.number > 6 || p.tile < 0 || ctx.graph.types[p.tile] === 'home') return 0.0;
        const d = shortestRoute(ctx, p);
        if (d === Infinity) return 1.0;
        return Math.min(d, MAX_DIST) / MAX_DIST;
    }

    function pieceProgress(ctx, p, endgameRule) {
        if (p.saved) return [0.0, 0.0];
        let bank;
        if (p.number <= 6 || !endgameRule) bank = NUMBERED_BANK;
        else bank = (p.tile >= 0 && ctx.graph.types[p.tile] === 'save')
            ? BLANK_BANK[ctx.graph.numbers[p.tile]] : BLANK_BANK_MEAN;
        if (p.tile < 0 || ctx.graph.types[p.tile] === 'home') return [HOME_TO_GOAL, bank];
        if (isSaveable(ctx, p)) return [0.0, bank];
        let d;
        if (p.number <= 6) d = shortestRoute(ctx, p);
        else {
            const ds = _RV.allGoalDistances(ctx.graph, p, _RV.blockedTiles(ctx.graph, ctx.pieces, p.player));
            d = Math.min(...Object.values(ds));
        }
        if (d === Infinity) d = MAX_DIST;
        return [Math.min(d, MAX_DIST), bank];
    }

    function sideSummary(ctx, player) {
        const unentered = ctx.unentered[player].length;
        const saved = ctx.pieces.filter(p => p.player === player && p.saved).length;
        const field = ctx.pieces.filter(p => p.player === player && p.tile >= 0
                                         && ctx.graph.types[p.tile] === 'field').length;
        let pips = 0.0, n = 0;
        const eg = stage(ctx, player) === 'endgame';
        for (const p of ctx.pieces) {
            if (p.player !== player || p.saved) continue;
            const [d] = pieceProgress(ctx, p, eg);
            pips += d;
            n++;
        }
        return [unentered, field, saved, pips, (pips / 3.5 + n) / 2.0];
    }

    /* endgame_turns: exact table value when every unsaved piece stands on a goal
       it can bank from; null otherwise. */
    function endgameTurns(ctx, player, table) {
        if (ctx.unentered[player].length) return null;
        let mask = 0;
        const b = [0, 0, 0, 0, 0, 0];
        for (const p of ctx.pieces) {
            if (p.player !== player || p.saved) continue;
            if (p.tile < 0 || ctx.graph.types[p.tile] !== 'save' || !isSaveable(ctx, p)) return null;
            if (p.number <= 6) mask |= 1 << (p.number - 1);
            else b[ctx.graph.numbers[p.tile] - 1] += 1;
        }
        if (mask === 0 && !b.some(x => x)) return 0.0;
        if (b.every(x => x === 0) && (mask & (mask - 1)) === 0) {   // last-piece rule
            b[31 - Math.clz32(mask)] = 1;
            mask = 0;
        }
        const v = table ? table[mask + ':' + b.join('')] : undefined;
        return v === undefined ? null : v;
    }

    function tableInput(ctx, player, table) {
        const v = endgameTurns(ctx, player, table);
        return v === null ? 0.0 : Math.min(v, 12.0) / 12.0;
    }

    function globalExtras(ctx, currentPlayer, table) {
        const opp = other(currentPlayer);
        const me = sideSummary(ctx, currentPlayer);
        const them = sideSummary(ctx, opp);
        const oppSaveable = ctx.pieces.filter(p => p.player === opp && isSaveable(ctx, p)).length;
        return [
            Math.min(ctx.noSave, NO_SAVE_TURNS_FOR_DRAW) / NO_SAVE_TURNS_FOR_DRAW,
            oppSaveable / 12.0,
            me[0] / 12.0, me[1] / 12.0, me[2] / 12.0,
            them[0] / 12.0, them[1] / 12.0, them[2] / 12.0,
            me[3] / 84.0, them[3] / 84.0,
            Math.min(me[4], 30.0) / 30.0, Math.min(them[4], 30.0) / 30.0,
            tableInput(ctx, currentPlayer, table), tableInput(ctx, opp, table),
        ];
    }

    // ------------------------------------------------------------------ AB

    /* _enemy_movers: {cnt: Map(dist -> n), dh, nHome, oblig}. */
    function enemyMovers(ctx, enemy, target, blocked, bkey) {
        const dist = distFrom(ctx, target, blocked, bkey);
        const cnt = new Map();
        for (const p of ctx.pieces) {
            if (p.player !== enemy || p.tile < 0 || ctx.graph.types[p.tile] === 'home'
                || p.tile === target) continue;
            const d = dist.get(p.tile);
            if (d !== undefined) cnt.set(d, (cnt.get(d) || 0) + 1);
        }
        const captured = ctx.occ[ctx.graph.home].filter(p => p.player === enemy).length;
        const rack = ctx.unentered[enemy].length;
        const hn = ctx.graph.neighbors[ctx.graph.home].filter(n => dist.has(n)).map(n => dist.get(n));
        const dh = hn.length ? 1 + Math.min(...hn) : null;
        let oblig, nHome;
        if (captured >= 2) { oblig = 'two'; nHome = captured; }
        else if (captured === 1) { oblig = 'one'; nHome = 1; }
        else if (rack) { oblig = 'one'; nHome = Math.min(rack, 2); }
        else { oblig = 'none'; nHome = 0; }
        return { cnt, dh, nHome, oblig };
    }

    function lands(m, a, b, need) {
        const { cnt, dh, nHome, oblig } = m;
        if (need === 1) {
            if (oblig === 'none') return cnt.has(a) || cnt.has(b) || cnt.has(a + b);
            if (oblig === 'one') return (dh !== null && (dh === a || dh === b || dh === a + b))
                                        || cnt.has(a) || cnt.has(b);
            return dh !== null && (dh === a || dh === b);
        }
        if (oblig === 'none') return a === b ? (cnt.get(a) || 0) >= 2 : (cnt.has(a) && cnt.has(b));
        const homeBoth = dh !== null && nHome >= 2 && dh === a && a === b;
        if (oblig === 'one') return homeBoth || (dh !== null && ((dh === a && cnt.has(b)) || (dh === b && cnt.has(a))));
        return homeBoth;
    }

    /* _chokes: field tiles on EVERY shortest route of p to goal, not occupied by
       p's own side. */
    function chokes(ctx, p, goal, ownBlock, ownKey, ownOcc) {
        const base = routeLen(ctx, p.tile, goal, ownBlock);
        if (base === null) return [];
        const dFrom = distFrom(ctx, p.tile, ownBlock, ownKey);
        const dTo = distFrom(ctx, goal, ownBlock, ownKey);
        const out = [];
        for (let t = 0; t < ctx.graph.n; t++) {
            if (ctx.graph.types[t] !== 'field' || t === p.tile || !dFrom.has(t) || !dTo.has(t)) continue;
            if (dFrom.get(t) + dTo.get(t) !== base || ownOcc.has(t)) continue;
            const blk = new Set(ownBlock); blk.add(t);
            const after = routeLen(ctx, p.tile, goal, blk);
            if (after === null || after > base) out.push(t);
        }
        return out;
    }

    function routeBlots(ctx, start, goal, blocked, bkey, enemy) {
        const dFrom = distFrom(ctx, start, blocked, bkey);
        const dTo = distFrom(ctx, goal, blocked, bkey);
        if (!dTo.has(start)) return [null, null];
        const base = dTo.get(start);
        let n = 0;
        for (let t = 0; t < ctx.graph.n; t++) {
            if (ctx.graph.types[t] !== 'field' || t === start || !dFrom.has(t) || !dTo.has(t)) continue;
            if (dFrom.get(t) + dTo.get(t) !== base) continue;
            const o = ctx.occ[t];
            if (o.length === 1 && o[0].player === enemy) n++;
        }
        return [n, base];
    }

    function blotsOnRoute(ctx, p, blocked, bkey) {
        if (p.tile < 0 || ctx.graph.types[p.tile] !== 'field') return 0;
        const enemy = other(p.player);
        if (p.number <= 6) {
            const [n] = routeBlots(ctx, p.tile, ctx.goals[p.number], blocked, bkey, enemy);
            return n || 0;
        }
        let best = null;
        for (let g = 1; g <= 6; g++) {
            const [n, base] = routeBlots(ctx, p.tile, ctx.goals[g], blocked, bkey, enemy);
            if (base !== null && (best === null || base < best[0] || (base === best[0] && n < best[1])))
                best = [base, n];
        }
        return best ? best[1] : 0;
    }

    /* threat_features: Map(piece -> [cap, wall, blots>=1, blots>=2]). */
    function threatFeatures(ctx) {
        const walled = { white: walledFor(ctx, 'white'), black: walledFor(ctx, 'black') };
        const wkey = { white: setKey(walled.white), black: setKey(walled.black) };
        const occ = {};
        for (const pl of ['white', 'black']) {
            occ[pl] = new Set();
            for (let t = 0; t < ctx.graph.n; t++)
                if (ctx.graph.types[t] === 'field' && ctx.occ[t].some(q => q.player === pl)) occ[pl].add(t);
        }
        const out = new Map();
        for (const p of ctx.pieces) {
            const t = p.tile;
            if (t < 0 || ctx.graph.types[t] !== 'field') continue;
            const enemy = other(p.player);
            let cap = 0.0;
            if (ctx.occ[t].length === 1) {
                const mv = enemyMovers(ctx, enemy, t, walled[enemy], wkey[enemy]);
                for (const [a, b, w] of ROLLS) if (lands(mv, a, b, 1)) cap += w;
            }
            let wall = 0.0;
            if (p.number <= 6) {
                const goal = ctx.goals[p.number];
                const ckey = t + '|' + goal + '|' + wkey[p.player] + '|' + setKey(occ[p.player]);
                let ch = capped(CHOKE).get(ckey);
                if (!ch) CHOKE.set(ckey, (ch = chokes(ctx, p, goal, walled[p.player], wkey[p.player], occ[p.player])));
                if (ch.length) {
                    const per = ch.map(wt => [
                        ctx.occ[wt].some(q => q.player === enemy) ? 1 : 2,
                        enemyMovers(ctx, enemy, wt, walled[enemy], wkey[enemy])]);
                    for (const [a, b, w] of ROLLS)
                        if (per.some(([need, m]) => lands(m, a, b, need))) wall += w;
                }
            }
            const nb = blotsOnRoute(ctx, p, walled[p.player], wkey[p.player]);
            out.set(p, [cap, wall, nb >= 1 ? 1.0 : 0.0, nb >= 2 ? 1.0 : 0.0]);
        }
        return out;
    }

    function savesWith(ctx, die, p, stg, highest) {
        const t = p.tile;
        if (stg === 'opening' || t < 0 || ctx.graph.types[t] !== 'save') return false;
        const gn = ctx.graph.numbers[t];
        if (p.number <= 6) return p.number === gn && die === gn;
        if (die === gn) return true;
        return stg === 'endgame' && die > gn && gn === highest;
    }

    function rollOpportunities(ctx, player) {
        const enemy = other(player);
        const blocked = walledFor(ctx, player);
        const bkey = setKey(blocked);
        const stg = stage(ctx, player);
        let highest = -1;
        for (let t = 0; t < ctx.graph.n; t++)
            if (ctx.graph.types[t] === 'save' && ctx.occ[t].some(q => q.player === player))
                highest = Math.max(highest, ctx.graph.numbers[t]);
        const movers = [];
        for (const p of ctx.pieces) {
            if (p.player !== player || p.tile < 0) continue;
            if (ctx.graph.types[p.tile] === 'home') movers.push([p, distFromHome(ctx, blocked, bkey)]);
            else movers.push([p, distFrom(ctx, p.tile, blocked, bkey)]);
        }
        const rack = ctx.unentered[player];
        if (rack.length && !ctx.occ[ctx.graph.home].some(q => q.player === player)) {
            const front = ctx.pieces.find(q => q.player === player && q.unentered && q.number === rack[0]);
            movers.push([front, distFromHome(ctx, blocked, bkey)]);
        }
        const loneEnemy = [];
        for (let t = 0; t < ctx.graph.n; t++) {
            const o = ctx.occ[t];
            if (ctx.graph.types[t] === 'field' && o.length === 1 && o[0].player === enemy) loneEnemy.push([t, o[0]]);
        }
        const goalTiles = [];
        for (let t = 0; t < ctx.graph.n; t++) if (ctx.graph.types[t] === 'save') goalTiles.push(t);
        const out = [];
        for (let d = 1; d <= 6; d++) {
            let svN = 0, svB = 0, glN = 0, glB = 0;
            const capsN = new Set(), capsB = new Set();
            for (const [p, dist] of movers) {
                if (p.tile >= 0 && savesWith(ctx, d, p, stg, highest)) {
                    if (p.number <= 6) svN++; else svB++;
                }
                const alreadyHome = p.tile >= 0 && ctx.graph.types[p.tile] === 'save' && isSaveable(ctx, p);
                let onto = false;
                if (!alreadyHome) {
                    for (const g of goalTiles) {
                        if (dist.get(g) === d && (p.number > 6 || ctx.graph.numbers[g] === p.number)) {
                            onto = true;
                            break;
                        }
                    }
                }
                if (onto) { if (p.number <= 6) glN++; else glB++; }
                for (const [t, e] of loneEnemy) if (dist.get(t) === d) (e.number <= 6 ? capsN : capsB).add(e);
            }
            out.push(svN, svB, glN, glB, capsN.size, capsB.size);
        }
        return out.map(c => Math.min(c, 6) / 6.0);
    }

    /* The extra columns for one position: {tile: [n], piece: [[k] per row of
       allPieces], global: [m]}. */
    function extras(graph, snapshot, pieces, allPieces, currentPlayer, features, table) {
        const ctx = context(graph, snapshot, pieces);
        const tile = oppWallFlags(ctx, currentPlayer);
        let th = null;
        if (features === 'AB') th = threatFeatures(ctx);
        const piece = allPieces.map(p => {
            const row = [ownGoalDist(ctx, p)];
            if (th) row.push(...(th.get(p) || [0.0, 0.0, 0.0, 0.0]));
            return row;
        });
        let glob = globalExtras(ctx, currentPlayer, table);
        if (features === 'AB')
            glob = glob.concat(rollOpportunities(ctx, currentPlayer), rollOpportunities(ctx, other(currentPlayer)));
        return { tile, piece, global: glob };
    }

    return { extras, stage, EXTRA, ROLLS };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = FV2;
else (typeof window !== 'undefined' ? window : self).FV2 = FV2;
