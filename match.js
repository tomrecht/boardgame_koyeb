/* Match awareness: choose the move that maximises the chance of winning the
 * MATCH, not the expected margin of this game.
 *
 * A match (game.js matchTracker) is either `games` (N games, total score; a tie
 * goes to more games won, then two more games or a draw by tieRule) or `race`
 * (first to `target` points). Only a game's winner scores, by the margin.
 *
 *   equityTable(match, me)  -> function(m) = P(I win the match | this game ends
 *                              with margin m from my side; 0 = drawn game)
 *   chooseByEquity(ranked, eq, model, phase)
 *                           -> index into the agent's ranking (returnScores)
 *
 * Two assumptions, both stated where used: future games are played by two
 * EQUAL players (the single-game outcome distribution is self-play's, starter's
 * view); and the current game's final margin, given a candidate's expected
 * margin V, is distributed as self-play's margins were at that V and phase
 * (`model.cond`, from lab/match/fit_model.py).
 */
(function () {
    const MAXM = 12;

    // P(final margin from the STARTER's view), m = -12..12 (0 = draw), for a
    // future game. model.prior: {m: p} keyed by margin.
    function priorFor(model, iStart) {
        const p = new Map();
        for (let m = -MAXM; m <= MAXM; m++) {
            const q = model.prior[String(m)] || 0;
            if (q) p.set(iStart ? m : -m, (p.get(iStart ? m : -m) || 0) + q);
        }
        return p;
    }

    /* Race to T: E[a][b][s] = P(I win | my score a, theirs b, s = I start next). */
    function raceValue(model, T) {
        const P = [priorFor(model, 0), priorFor(model, 1)];
        const pd = [P[0].get(0) || 0, P[1].get(0) || 0];
        const E = [];
        for (let a = 0; a < T; a++) { E.push([]); for (let b = 0; b < T; b++) E[a].push([0, 0]); }
        const val = (a, b, s) => a >= T ? 1 : b >= T ? 0 : E[a][b][s];
        for (let tot = 2 * T - 2; tot >= 0; tot--) {
            for (let a = Math.min(T - 1, tot); a >= 0; a--) {
                const b = tot - a;
                if (b < 0 || b >= T) continue;
                const R = [0, 0];
                for (const s of [0, 1]) {
                    for (const [m, q] of P[s]) {
                        if (m > 0) R[s] += q * val(a + m, b, 1 - s);
                        else if (m < 0) R[s] += q * val(a, b - m, 1 - s);
                    }
                }
                // A drawn game scores nothing and hands the start over:
                // E0 = R0 + pd0 E1, E1 = R1 + pd1 E0.
                const den = 1 - pd[0] * pd[1];
                E[a][b][0] = (R[0] + pd[0] * R[1]) / den;
                E[a][b][1] = (R[1] + pd[1] * R[0]) / den;
            }
        }
        return val;
    }

    /* N games by total score: V(g, d, w, s) with g games left, d my lead in
       points, w my lead in games won, s = I start the next game. The 'extra'
       tie rule replays two more games from (2, 0, 0, s): a fixed point, found by
       iterating. */
    function gamesValue(model, tieRule) {
        const P = [priorFor(model, 0), priorFor(model, 1)];
        let tie = [0.5, 0.5];
        let memo;
        const V = (g, d, w, s) => {
            if (g === 0) {
                if (d !== 0) return d > 0 ? 1 : 0;
                if (w !== 0) return w > 0 ? 1 : 0;
                return tieRule === 'draw' ? 0.5 : tie[s];
            }
            const k = g + ',' + d + ',' + w + ',' + s;
            let v = memo.get(k);
            if (v !== undefined) return v;
            v = 0;
            for (const [m, q] of P[s]) v += q * V(g - 1, d + m, w + Math.sign(m), 1 - s);
            memo.set(k, v);
            return v;
        };
        for (let it = 0; it < 60; it++) {
            memo = new Map();
            const nt = [V(2, 0, 0, 0), V(2, 0, 0, 1)];
            const done = Math.abs(nt[0] - tie[0]) + Math.abs(nt[1] - tie[1]) < 1e-12;
            tie = nt;
            if (tieRule === 'draw' || done) break;
        }
        memo = new Map();
        return V;
    }

    /* match: game.js's matchTracker (mode, target, tieRule, gamesPlayed, whiteScore,
       blackScore, whiteWins, blackWins, firstStarter). me: 'white' | 'black'.
       Returns m -> P(I win the match) for THIS game ending with my margin m. */
    function equityTable(match, me, model) {
        const opp = me === 'white' ? 'black' : 'white';
        const k = match.gamesPlayed;                        // this game's index
        const starterOf = (i) => (i % 2 === 0) ? match.firstStarter
                                               : (match.firstStarter === 'white' ? 'black' : 'white');
        const sNext = starterOf(k + 1) === me ? 1 : 0;
        const myS = match[me + 'Score'], opS = match[opp + 'Score'];
        const out = new Map();
        if (match.mode === 'race') {
            const T = match.target;
            const val = raceValue(model, T);
            for (let m = -MAXM; m <= MAXM; m++) {
                const a = myS + Math.max(m, 0), b = opS + Math.max(-m, 0);
                out.set(m, a >= T ? 1 : b >= T ? 0 : val(a, b, sNext));
            }
        } else {
            const V = gamesValue(model, match.tieRule || 'extra');
            const left = match.target - k - 1;
            const myW = match[me + 'Wins'], opW = match[opp + 'Wins'];
            for (let m = -MAXM; m <= MAXM; m++)
                out.set(m, V(left, myS - opS + m, myW - opW + Math.sign(m), sNext));
        }
        return out;
    }

    /* The distribution of this game's final margin (my view) given a candidate's
       expected margin v (points) and the phase (total unsaved pieces after the
       move). model.cond = {phases: [lo...], vgrid: [...], dist: [phase][v][25]}:
       the SHAPE comes from self-play near that v and phase (interpolated in both,
       so no band edge can jump), and it is then exponentially tilted so its mean
       is exactly v. So where match equity is a straight line in the margin, every
       candidate's equity is a + b*v and the choice is the plain one; only real
       curvature (match pressure) can change a move. */
    function shapeAt(model, v, phase) {
        const c = model.cond, g = c.vgrid;
        const x = Math.max(g[0], Math.min(g[g.length - 1], v));
        let j = 0;
        while (j < g.length - 2 && x > g[j + 1]) j++;
        const t = (x - g[j]) / (g[j + 1] - g[j]);
        const atBand = (b) => c.dist[b][j].map((p, i) => (1 - t) * p + t * c.dist[b][j + 1][i]);
        // Phase: band b covers [phases[b], phases[b+1]); interpolate between band middles.
        const ph = c.phases, mid = ph.map((lo, b) => b + 1 < ph.length ? (lo + ph[b + 1]) / 2 : lo + 3);
        if (phase <= mid[0]) return atBand(0);
        if (phase >= mid[mid.length - 1]) return atBand(mid.length - 1);
        let b = 0;
        while (b < mid.length - 2 && phase > mid[b + 1]) b++;
        const u = (phase - mid[b]) / (mid[b + 1] - mid[b]);
        const A = atBand(b), B = atBand(b + 1);
        return A.map((p, i) => (1 - u) * p + u * B[i]);
    }
    function tilt(p, v) {
        // Mean of p * exp(l*m) is increasing in l: bisect for the l giving mean v.
        const target = Math.max(-MAXM + 0.05, Math.min(MAXM - 0.05, v));
        const meanAt = (l) => { let z = 0, s = 0;
            for (let i = 0; i < p.length; i++) { const w = p[i] * Math.exp(l * (i - MAXM)); z += w; s += w * (i - MAXM); }
            return s / z; };
        let lo = -30, hi = 30;
        for (let it = 0; it < 60; it++) { const mid = (lo + hi) / 2; if (meanAt(mid) < target) lo = mid; else hi = mid; }
        const l = (lo + hi) / 2;
        const w = p.map((q, i) => q * Math.exp(l * (i - MAXM)));
        const z = w.reduce((a, b) => a + b, 0);
        return w.map(q => q / z);
    }
    function marginDist(model, v, phase) {
        return tilt(shapeAt(model, v, phase), v);       // index i -> margin i - 12
    }

    /* ranked: the agent's returnScores list ({score, pair}), scores in SCORE_SCALE
       raw units (score / scale * 12 = expected margin in points). phases[i]: total
       unsaved after candidate i. Returns the index of the best candidate by
       match equity; ties go to the higher score (the plain choice). */
    function chooseByEquity(ranked, eq, model, phases, scale, isDrawPair) {
        let best = -1, bestE = -1, bestS = -Infinity;
        ranked.forEach((r, i) => {
            let e;
            if (r.score === Infinity) e = Infinity;
            else if (isDrawPair && isDrawPair(r.pair)) e = eq.get(0);   // calling a draw: exactly 0
            else {
                const d = marginDist(model, r.score / scale * MAXM, phases[i]);
                e = 0;
                for (let k = 0; k < d.length; k++) e += d[k] * eq.get(k - MAXM);
            }
            if (e > bestE + 1e-12 || (Math.abs(e - bestE) <= 1e-12 && r.score > bestS)) {
                best = i; bestE = e; bestS = r.score;
            }
        });
        return best;
    }

    const api = { equityTable, marginDist, chooseByEquity, raceValue, gamesValue, MATCH_MAXM: MAXM };
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    else Object.assign(typeof window !== 'undefined' ? window : self, { Match: api });
})();
