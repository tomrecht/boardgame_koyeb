"""panel_gate.py -- FIXED-PANEL promotion gate (replaces the 200-game vs-parent gate).

Why (ARCHIVE.md, "non-transitivity"): gating against the parent alone measured
~+35 Elo per promotion correctly, but only ~24% of it generalised -- a candidate
learns to beat the specific parent it was trained against. A fixed panel gates
on generality instead, and makes runs comparable across experiments.

Three stages, each configurable:

1. PARENT PRE-SCREEN (prescreen_pairs, default 100 pairs = 200 games; 0 = off).
   Candidate vs the current champion on fixed colour-swapped seeds. PASS iff
   the mean margin >= prescreen_bar (default 0.0, i.e. "not behind its
   parent"). Not the old 55% bar: that rejects many true improvers (a true
   +0.3-margin improver passes a >=0 bar ~95% of the time at 100 pairs,
   SD 1.8, but a 55%-win bar well under half the time), while a clear loser
   (-0.5) passes a >=0 bar only ~0.3% of the time. Its job is to save panel
   games on obvious failures, not to judge improvement -- the panel does that.

2. PANEL, PLAYED SEQUENTIALLY with GROUP-SEQUENTIAL EARLY STOPPING.
   * Every candidate plays every member on the SAME fixed dice seeds
     (seed_base + k), each seed twice with colours swapped -- common random
     numbers: a candidate and the champion face IDENTICAL dice against each
     member, so their difference is measured with dice luck cancelled.
   * Unit = one (member, seed) pair: the mean of its two colour-swapped margins
     (candidate's perspective). Statistic: the PAIRED difference
         d = unit(candidate) - unit(champion)   over the same units,
     i.e. the candidate's panel score minus the champion's own.
   * Games are played in BATCHES of batch_pairs seeds x every member
     (interleaved, so each look is balanced across members), up to
     max_pairs per member. After each batch k (information fraction t_k =
     n_k/N) the t-statistic of mean(d) is compared with an O'Brien-Fleming
     boundary b_k = c / sqrt(t_k):
         z >= +b_k  -> PROMOTE (subject to the guard)
         z <= -b_k  -> REJECT (futility; binding)
     c is calibrated by Monte Carlo for the actual t_k so that
     P(cross +b before -b | true edge 0) = alpha (default 0.05, one-sided) --
     a valid group-sequential test, NOT a naive CI re-checked at each look.
     The z boundary is converted to the t scale (df = n_k - 1) so the plug-in
     SD does not inflate early looks. If no boundary is crossed by the cap the
     outcome is CAP: not promoted; kept iff mean(d) > 0. test_panel_gate.py
     checks the false-promotion rate by simulation.
   * Champion's panel games are cached per (member, seed, colour) (json keyed
     by a hash of the weights + panel + rules flag). Play is deterministic
     given the seed, so the cache is exact; a promoted candidate's games become
     the new champion's entries for free.

3. GUARD: against no single member may the candidate's mean margin (over the
   units played) fall below `guard` (default -1.0 points). A promote-boundary
   crossing with the guard failing is recorded as GUARD_FAIL (not promoted).

Greedy only: every game asserts no exploration sample was drawn, and agents
are built with the hand-coded play rules OFF (configurable).

Panel default is THREE members (symaug6, iter10, iter14); PANEL_FIVE adds iter4
and aux14. Dropped from the default: iter4 is an ancestor on the iter10 line
and aux14 is iter14 fine-tuned with aux heads -- each near-duplicates the
lineage of a member kept (successive champions correlate 0.95-0.98 in value),
while the four are statistically inseparable in the arena. Kept: symaug6
(deployed, owner's strongest opponent), iter10 (human-strongest, heavy 2&4
blocker) and iter14 (rarely blocks; the documented non-transitive pair with
iter10) -- the most stylistic spread for the games.

Reuses arena.py's game loop (`_play`) and its per-process agent cache.
"""
import hashlib
import json
import math
import os
import random
import time

REPO = os.path.dirname(os.path.abspath(__file__))

PANEL_FIVE = {
    'symaug6': f'{REPO}/symaug_iter6.pt',               # deployed champion
    'iter4':   f'{REPO}/td_champion_July17_iter4.pt',
    'iter10':  f'{REPO}/td_champion_July18_iter10.pt',
    'iter14':  f'{REPO}/td_champion_July19_iter14.pt',
    'aux14':   f'{REPO}/td_champion_July21_aux_iter14.pt',  # aux heads already stripped ("washed")
}
PANEL_DEFAULT = {t: PANEL_FIVE[t] for t in ('symaug6', 'iter10', 'iter14')}
PANEL_SEED_BASE = 77_000_000      # disjoint from every generation/eval seed block
PRESCREEN_SEED_BASE = 78_000_000


def sd_hash(sd):
    """Stable content hash of a state_dict (sorted keys, raw tensor bytes)."""
    h = hashlib.sha1()
    for k in sorted(sd):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()[:16]


def file_hash(path):
    import torch
    return sd_hash(torch.load(path, map_location='cpu'))


# -------------------- sequential boundary --------------------
def _norm_ppf(p):
    from statistics import NormalDist
    return NormalDist().inv_cdf(p)


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _t_ppf(p, df):
    try:
        from scipy.stats import t as _t
        return float(_t.ppf(p, df))
    except Exception:                       # Cornish-Fisher fallback
        z = _norm_ppf(p)
        return z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)


def obf_constant(fracs, alpha, n_sims=200_000, seed=12345):
    """O'Brien-Fleming constant c for looks at information fractions `fracs`
    (increasing, last = 1), with BINDING symmetric futility: boundaries +-c/sqrt(t_k).
    Calibrated by Monte Carlo on the Brownian score process so that
    P(hit +b_k before -b_k | drift 0) = alpha. Deterministic (fixed seed)."""
    import numpy as np
    fr = np.asarray(fracs, dtype=float)
    rng = np.random.default_rng(seed)
    inc = np.diff(np.concatenate([[0.0], fr]))
    S = np.cumsum(rng.standard_normal((n_sims, len(fr))) * np.sqrt(inc), axis=1)
    Z = S / np.sqrt(fr)                                 # z-statistic at each look

    def rate(c):
        b = c / np.sqrt(fr)
        up = Z >= b
        lo = Z <= -b
        first_up = np.where(up.any(1), up.argmax(1), len(fr))
        first_lo = np.where(lo.any(1), lo.argmax(1), len(fr))
        return float(np.mean(first_up < first_lo))

    lo_c, hi_c = 0.5, 6.0
    for _ in range(50):
        mid = 0.5 * (lo_c + hi_c)
        if rate(mid) > alpha:
            lo_c = mid
        else:
            hi_c = mid
    return hi_c


class SequentialTest:
    """Group-sequential one-sided test on paired differences d."""

    def __init__(self, n_per_look, alpha=0.05):
        self.n_per_look = list(n_per_look)           # cumulative unit counts
        N = self.n_per_look[-1]
        self.fracs = [n / N for n in self.n_per_look]
        self.alpha = alpha
        self.c = obf_constant(self.fracs, alpha)

    def boundary(self, k, n):
        """t-scale boundary at look k (0-based) with n units."""
        bz = self.c / math.sqrt(self.fracs[k])
        if n < 2:
            return float('inf')
        # same one-sided tail probability, on the t(n-1) scale
        return _t_ppf(_norm_cdf(bz), n - 1)

    @staticmethod
    def tstat(d):
        n = len(d)
        m = sum(d) / n
        if n < 2:
            return m, float('inf'), 0.0
        var = sum((x - m) ** 2 for x in d) / (n - 1)
        se = math.sqrt(var / n)
        if se == 0:
            return m, se, (float('inf') if m > 0 else float('-inf') if m < 0 else 0.0)
        return m, se, m / se

    def decide(self, k, d):
        """-> 'promote' | 'reject' | 'continue' | 'cap' at look k."""
        m, se, t = self.tstat(d)
        b = self.boundary(k, len(d))
        if t >= b:
            return 'promote', m, se, t, b
        if t <= -b:
            return 'reject', m, se, t, b
        if k == len(self.n_per_look) - 1:
            return 'cap', m, se, t, b
        return 'continue', m, se, t, b


# -------------------- worker side --------------------
def panel_game(task):
    """One gate game, run in a pool worker. Candidate weights are read from
    `cand_path`; `cand_key` (its content hash) keys the per-process cache so a
    new candidate saved to the same path is reloaded. The opponent may also be
    keyed (the pre-screen's parent file)."""
    (cand_path, cand_key, tag, member_path, seed, cand_white, hand_rules) = task[:7]
    member_key = task[7] if len(task) > 7 else None
    import explore
    from arena import _agent, _play
    before = explore.samples()
    cand = _agent(cand_path, cand_key, hand_rules)
    mem = _agent(member_path, member_key, hand_rules)
    t0 = time.time()
    st = {}
    winner, score = _play(seed, cand if cand_white else mem,
                          mem if cand_white else cand, stats=st)
    secs = time.time() - t0
    # Gating is strictly greedy.
    assert explore.samples() == before, 'exploration fired during GATING'
    if winner is None:
        margin = 0
    else:
        won = (winner == 'white') == cand_white
        margin = score if won else -score
    return {'tag': tag, 'seed': seed, 'cand_white': cand_white,
            'margin': margin, 'winner': winner, 'turns': st.get('turns', 0),
            'secs': secs}


# -------------------- main-process side --------------------
class PanelGate:
    def __init__(self, panel=None, max_pairs=100, batch_pairs=20, alpha=0.05,
                 guard=-1.0, prescreen_pairs=100, prescreen_bar=0.0,
                 sequential=True, seed_base=PANEL_SEED_BASE,
                 cache_path='panel_cache.json', hand_rules=False, workdir='.',
                 prefix='panel', check_paths=True):
        assert max_pairs >= 1 and batch_pairs >= 1
        self.panel = dict(panel or PANEL_DEFAULT)
        if check_paths:
            for tag, p in self.panel.items():
                assert os.path.exists(p), f'panel member {tag}: {p} not found'
        self.max_pairs = max_pairs
        self.batch_pairs = min(batch_pairs, max_pairs) if sequential else max_pairs
        self.seeds = [seed_base + k for k in range(max_pairs)]
        self.prescreen_seeds = [PRESCREEN_SEED_BASE + k for k in range(prescreen_pairs)]
        self.prescreen_bar = prescreen_bar
        self.alpha = alpha
        self.guard = guard
        self.cache_path = cache_path
        self.hand_rules = bool(hand_rules)
        self.cand_path = os.path.join(workdir, f'{prefix}_gate_candidate.pt')
        self.parent_path = os.path.join(workdir, f'{prefix}_gate_parent.pt')
        M = len(self.panel)
        looks = list(range(self.batch_pairs, max_pairs, self.batch_pairs)) + [max_pairs]
        self.look_pairs = looks
        self.seq = SequentialTest([p * M for p in looks], alpha)
        if check_paths:
            member_hashes = {t: file_hash(p) for t, p in sorted(self.panel.items())}
        else:
            member_hashes = dict(self.panel)
        self.cfg_key = hashlib.sha1(json.dumps(
            {'members': member_hashes, 'seed_base': seed_base,
             'hand_rules': self.hand_rules,
             # cached champion games are only valid for the same gate player
             'gate_prefilter': os.environ.get('GATE_PREFILTER', '')},
            sort_keys=True).encode()).hexdigest()[:12]
        self.cache = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path) as f:
                self.cache = json.load(f)
        self.timing = {'games': 0, 'secs': 0.0, 'turns': 0}
        print(f'  panel gate: {M} members {sorted(self.panel)}, looks at '
              f'{looks} pairs/member (cap {max_pairs}), alpha {alpha}, OBF c={self.seq.c:.3f}, '
              f'pre-screen {len(self.prescreen_seeds)} pairs bar {prescreen_bar:+.2f}')

    # ---- game running (overridable in tests) ----
    def run_tasks(self, tasks):
        """Play tasks on the training pool; returns result dicts."""
        import game_worker
        if game_worker._POOL is None:
            game_worker.init_pool(max(1, (os.cpu_count() or 2) - 1))
        out = list(game_worker._POOL.imap_unordered(panel_game, tasks, chunksize=1))
        for r in out:
            self.timing['games'] += 1
            self.timing['secs'] += r['secs']
            self.timing['turns'] += r['turns']
        return out

    def save_weights(self, sd, path):
        import torch
        torch.save({k: v.detach().cpu() for k, v in sd.items()}, path)

    # ---- cache ----
    def _save_cache(self):
        if not self.cache_path:
            return
        tmp = self.cache_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(self.cache, f)
        os.replace(tmp, self.cache_path)

    def ensure(self, sd, h, n_pairs, label):
        """Make sure `sd` (hash h) has panel games on the first n_pairs seeds of
        every member; plays only what is missing. Returns its game dict and the
        number of games newly played."""
        ck = f'{self.cfg_key}:{h}'
        g = self.cache.setdefault(ck, {})
        tasks = []
        for seed in self.seeds[:n_pairs]:
            for tag, path in sorted(self.panel.items()):
                for cw in (True, False):
                    if f'{tag}|{seed}|{int(cw)}' not in g:
                        tasks.append((self.cand_path, h, tag, path, seed, cw, self.hand_rules))
        if tasks:
            self.save_weights(sd, self.cand_path)
            t0 = time.time()
            for r in self.run_tasks(tasks):
                g[f"{r['tag']}|{r['seed']}|{int(r['cand_white'])}"] = r['margin']
            self._save_cache()
            print(f'  panel[{label}]: +{len(tasks)} games ({time.time() - t0:.0f}s), '
                  f'{n_pairs} pairs/member')
        return g, len(tasks)

    def _units(self, g, tag, n_pairs):
        return [0.5 * (g[f'{tag}|{s}|1'] + g[f'{tag}|{s}|0']) for s in self.seeds[:n_pairs]]

    # ---- stage 1 ----
    def prescreen(self, cand_sd, champ_sd, label):
        """Candidate vs current champion. Returns (passed, mean, games)."""
        if not self.prescreen_seeds:
            return True, None, 0
        self.save_weights(cand_sd, self.cand_path)
        self.save_weights(champ_sd, self.parent_path)
        ch, ph = sd_hash(cand_sd), sd_hash(champ_sd)
        tasks = [(self.cand_path, ch, 'parent', self.parent_path, s, cw,
                  self.hand_rules, ph)
                 for s in self.prescreen_seeds for cw in (True, False)]
        t0 = time.time()
        res = self.run_tasks(tasks)
        m = sum(r['margin'] for r in res) / len(res)
        wins = sum(r['margin'] > 0 for r in res)
        passed = m >= self.prescreen_bar
        print(f'  PRE-SCREEN [{label}] vs parent: {len(res)} games ({time.time() - t0:.0f}s), '
              f'mean margin {m:+.3f}, wins {wins}/{len(res)} -> '
              f'{"pass" if passed else "REJECT"} (bar {self.prescreen_bar:+.2f})')
        return passed, m, len(res)

    # ---- full gate ----
    def evaluate(self, cand_sd, champ_sd, label='candidate'):
        """Gate `cand_sd` relative to `champ_sd`. Returns a report dict;
        report['promote'] is the decision, report['outcome'] one of
        prescreen_reject / promote / guard_fail / reject / cap."""
        ch, hh = sd_hash(cand_sd), sd_hash(champ_sd)
        passed, pm, pgames = self.prescreen(cand_sd, champ_sd, label)
        rep = {'prescreen_mean': pm, 'prescreen_games': pgames,
               'panel_games': 0, 'champ_games_new': 0}
        if not passed:
            rep.update(outcome='prescreen_reject', promote=False, keep=False,
                       cand_score=pm, diff=pm, diff_lo=None, diff_hi=None,
                       per_member={}, looks=[])
            self._print(rep, label)
            return rep
        tags = sorted(self.panel)
        looks = []
        for k, n_pairs in enumerate(self.look_pairs):
            gh, nh = self.ensure(champ_sd, hh, n_pairs, 'champion')
            gc, nc = self.ensure(cand_sd, ch, n_pairs, label)
            rep['panel_games'] += nc
            rep['champ_games_new'] += nh
            d, cu_all, hu_all = [], [], []
            per = {}
            for t in tags:
                cu, hu = self._units(gc, t, n_pairs), self._units(gh, t, n_pairs)
                dd = [a - b for a, b in zip(cu, hu)]
                wins = sum(1 for s in self.seeds[:n_pairs] for w in (0, 1)
                           if gc[f'{t}|{s}|{w}'] > 0)
                per[t] = {'cand': sum(cu) / n_pairs, 'champ': sum(hu) / n_pairs,
                          'diff': sum(dd) / n_pairs, 'cand_win': wins / (2 * n_pairs)}
                d += dd; cu_all += cu; hu_all += hu
            dec, m, se, tst, b = self.seq.decide(k, d)
            looks.append({'pairs': n_pairs, 'units': len(d), 'diff': m, 'se': se,
                          't': tst, 'bound': b, 'decision': dec})
            print(f'    look {k + 1}/{len(self.look_pairs)}: {n_pairs} pairs/member, '
                  f'diff {m:+.3f} se {se:.3f} t {tst:+.2f} vs +-{b:.2f} -> {dec}')
            if dec != 'continue':
                break
        worst = min(tags, key=lambda t: per[t]['cand'])
        guard_ok = per[worst]['cand'] >= self.guard
        outcome = dec
        if dec == 'promote' and not guard_ok:
            outcome = 'guard_fail'
        cs = sum(cu_all) / len(cu_all)
        rep.update(outcome=outcome, promote=(outcome == 'promote'),
                   keep=(outcome == 'promote') or (outcome in ('cap', 'guard_fail') and m > 0),
                   cand_score=cs, champ_score=sum(hu_all) / len(hu_all),
                   diff=m, diff_se=se, diff_lo=m - _norm_ppf(1 - self.alpha) * se,
                   diff_hi=m + _norm_ppf(1 - self.alpha) * se,
                   per_member=per, worst_member=worst, guard_ok=guard_ok,
                   looks=looks, n_units=len(d))
        self._print(rep, label)
        return rep

    def _print(self, rep, label):
        print(f'  PANEL GATE [{label}] outcome: {rep["outcome"].upper()} | games: '
              f'pre-screen {rep["prescreen_games"]} + panel {rep["panel_games"]} '
              f'(+{rep["champ_games_new"]} champion, cached)')
        if not rep['per_member']:
            return
        print(f'    {"member":8}{"cand":>8}{"champ":>8}{"diff":>8}{"cand win%":>11}')
        for t, p in rep['per_member'].items():
            print(f'    {t:8}{p["cand"]:+8.2f}{p["champ"]:+8.2f}{p["diff"]:+8.2f}'
                  f'{100 * p["cand_win"]:10.0f}%')
        print(f'    panel score cand {rep["cand_score"]:+.3f} champ {rep["champ_score"]:+.3f}; '
              f'paired diff {rep["diff"]:+.3f} +- {rep["diff_se"]:.3f} '
              f'(nominal one-sided {1 - self.alpha:.0%} CI [{rep["diff_lo"]:+.3f}, '
              f'{rep["diff_hi"]:+.3f}], {rep["n_units"]} units; decision uses the '
              f'sequential boundary)')
        print(f'    guard: worst member {rep["worst_member"]} '
              f'{rep["per_member"][rep["worst_member"]]["cand"]:+.2f} vs floor '
              f'{self.guard:+.2f} -> {"ok" if rep["guard_ok"] else "FAIL"}')
