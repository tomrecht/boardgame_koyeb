"""revert_policy.py -- when a league run resets its live model to the champion.

fv2B (2026-10-14) used the strict rule: reset unless the panel capped with a
positive paired diff. A one-iteration fine-tune of the blend was neutral-to-worse
vs its parent on average, so nearly every candidate was reset and the run was
~145 independent first steps -- nothing accumulated (val loss flat). For a run
whose first iterations are EXPECTED to trail the champion (fv2C: warm start
trained in the turn-start frame, now learning afterstates) that rule is fatal.

'lenient':
  * BURN-IN (iteration <= burnin): never reset, except on a COLLAPSE --
    pre-screen margin vs the champion below `collapse_floor` (the TD-bootstrap
    collapse td_selfplay_loop's revert was written for: 24% -> 0% wins).
  * After it, reset on a CLEAR loss -- panel REJECT (futility boundary) or a
    pre-screen that stopped EARLY to reject -- or after `streak` iterations in
    a row behind the champion (panel diff < 0, pre-screen mean < 0, or a failed
    fresh-seed confirmation). Anything else stays live.
  The champion is never at risk: only the gate (with its fresh-seed
  confirmation) replaces it.
'strict': the gate's own 'keep' stands (fv2B behaviour).

The streak persists in state_path so a resume continues it.
"""
import json
import os


class RevertPolicy:
    def __init__(self, mode='lenient', burnin=10, streak=3, collapse_floor=-2.0,
                 full_prescreen_games=200, state_path=None):
        assert mode in ('lenient', 'strict'), mode
        self.mode, self.burnin, self.streak = mode, burnin, streak
        self.collapse_floor = collapse_floor
        self.full_prescreen_games = full_prescreen_games
        self.state_path = state_path
        self.behind = 0
        if state_path and os.path.exists(state_path):
            self.behind = json.load(open(state_path)).get('behind', 0)

    @staticmethod
    def _behind(rep):
        if rep.get('confirm') and not rep['confirm']['passed']:
            return True
        if rep['outcome'] == 'prescreen_reject':
            return (rep.get('prescreen_mean') or 0) < 0
        return rep.get('diff', 0) < 0

    def apply(self, rep, it):
        """Set rep['keep'] (and rep['revert_reason']) for a non-promoted
        iteration. Returns rep."""
        if self.mode == 'strict' or rep.get('promote'):
            if rep.get('promote'):
                self.behind = 0
                self._save()
            return rep
        gate_keep = bool(rep.get('keep'))
        behind = self._behind(rep)
        self.behind = self.behind + 1 if behind else 0
        pm = rep.get('prescreen_mean')
        collapse = pm is not None and pm < self.collapse_floor
        early_ps = (rep['outcome'] == 'prescreen_reject'
                    and rep.get('prescreen_games', 0) < self.full_prescreen_games)
        if collapse:
            keep, why = False, f'collapse: pre-screen {pm:+.2f} < {self.collapse_floor:+.2f}'
        elif it <= self.burnin:
            keep, why = True, f'burn-in ({it}/{self.burnin})'
        elif rep['outcome'] == 'reject':
            keep, why = False, 'panel REJECT'
        elif early_ps:
            keep, why = False, 'pre-screen stopped early to reject'
        elif self.behind >= self.streak:
            keep, why = False, f'{self.behind} iterations in a row behind the champion'
        else:
            keep, why = True, (f'not clearly behind (streak {self.behind}/{self.streak})'
                               if behind else 'level or ahead')
        if not keep:
            self.behind = 0          # back on the champion: the streak restarts
        rep['keep'] = keep
        rep['revert_reason'] = why
        rep['gate_keep'] = gate_keep
        print(f"  REVERT POLICY [{self.mode}] it{it}: {'KEEP live model' if keep else 'RESET to champion'}"
              f" -- {why}")
        self._save()
        return rep

    def _save(self):
        if self.state_path:
            with open(self.state_path, 'w') as f:
                json.dump({'behind': self.behind}, f)
