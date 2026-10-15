"""new_inputs.py -- help a widened net learn to USE its new input columns.

fv2B (2026-10-14): the AB inputs began at zero weight (widen_v2.py) and after
145 iterations carried ~1-3% of an old input's influence. Three levers, all
confined to the NEW columns (those after v1's) of the three input-facing
matrices, so what the net already knows is untouched:

  * BoostedAdam: Adam moves each weight ~lr per step whatever its gradient, so
    a multiplier on the gradient does nothing. Instead the STEP on new columns
    is multiplied: mult(it) = 1 + (MULT - 1) * max(0, 1 - (it - 1) / ANNEAL),
    i.e. MULT at iteration 1 falling linearly to 1x by iteration ANNEAL + 1.
    The boost only has to get them off zero; past that they should learn at the
    old columns' pace, or they just add noise.
  * new_masks() also lets league_run's weight averaging skip the new columns
    (averaging halves every move a zero-started weight makes).
  * usage(): weight x input spread per column, old vs new -- logged per
    iteration so a run shows whether the new inputs are being picked up,
    separately from strength. (Input spread is measured once, on start-pool
    positions, in the run's own encoding.)
"""
import numpy as np
import torch

import encoder as E
import network

KINDS = (('tile', 'tile_embed.weight', 0), ('piece', 'piece_embed.weight', 0),
         ('global', 'readout.0.weight', None))


def _offset(key):
    return 3 * network.HIDDEN_DIM if key == 'readout.0.weight' else 0


def new_masks(features):
    """{param name: (column offset of the inputs, first new column)} -- empty for v1."""
    if features == 'v1':
        return {}
    v1 = E.feature_dims('v1')
    return {key: (_offset(key), _offset(key) + v1[i]) for i, (_, key, _o) in enumerate(KINDS)}


def mult_for(it, mult, anneal):
    if mult <= 1 or anneal <= 0:
        return 1.0
    return 1.0 + (mult - 1.0) * max(0.0, 1.0 - (it - 1) / anneal)


class BoostedAdam(torch.optim.Adam):
    """Adam whose step on the new input columns is multiplied by self.mult."""

    def __init__(self, model, lr, mult=10.0, anneal=20, it=1):
        super().__init__(model.parameters(), lr=lr)
        cols = new_masks(getattr(model, 'features', 'v1'))
        params = dict(model.named_parameters())
        self._boost = [(params[k], start) for k, (_off, start) in cols.items()]
        self.base_mult, self.anneal = mult, anneal
        self.set_iter(it)

    def set_iter(self, it):
        self.mult = mult_for(it, self.base_mult, self.anneal)

    @torch.no_grad()
    def step(self, closure=None):
        if self.mult == 1.0 or not self._boost:
            return super().step(closure)
        before = [p[:, s:].clone() for p, s in self._boost]
        out = super().step(closure)
        for (p, s), b in zip(self._boost, before):
            p[:, s:] += (self.mult - 1.0) * (p[:, s:] - b)
        return out


def blend_skip_new(avg_sd, trained_sd, beta, features):
    """league_run's running average, except the new columns take the trained
    weights outright (averaging would halve every step they make)."""
    cols = new_masks(features)
    out = {}
    for k, b in trained_sd.items():
        a = avg_sd.get(k)
        if a is None or k == 'afterstate_flag' or not b.dtype.is_floating_point:
            out[k] = b
            continue
        v = beta * a + (1 - beta) * b
        if k in cols:
            s = cols[k][1]
            v = v.clone()
            v[:, s:] = b[:, s:]
        out[k] = v
    return out


def input_spread(encoder, states, n=400, seed=0):
    """Per-column std of every input (tile, piece, global) over pool positions."""
    import random
    from game import Board
    rng = random.Random(seed)
    T, P, G = [], [], []
    for st in rng.sample(states, min(n, len(states))):
        b = Board(); b.update_state(st); b.firstMove = None
        b.get_valid_moves()
        f = E.BoardEncoder.encode(encoder, b, b.current_player)
        T.append(np.asarray(f['tile_feats'])); P.append(np.asarray(f['piece_feats']))
        G.append(np.asarray(f['global_feats']).reshape(-1))
    return {'tile': np.concatenate(T).std(0), 'piece': np.concatenate(P).std(0),
            'global': np.stack(G).std(0)}


def usage(sd, spread, features):
    """Influence of each input column = ||weight column|| x input std. Returns
    {kind: {'old_median', 'new_median', 'new_max', 'ratio'}} plus 'all_ratio'
    (median new / median old over all three kinds)."""
    if features == 'v1':
        return {}
    v1 = E.feature_dims('v1')
    out, olds, news = {}, [], []
    for i, (kind, key, _o) in enumerate(KINDS):
        off = _offset(key)
        W = sd[key].detach().cpu()[:, off:]
        u = W.norm(dim=0).numpy() * spread[kind]
        old, new = u[:v1[i]], u[v1[i]:]
        olds.append(old); news.append(new)
        out[kind] = {'old_median': round(float(np.median(old)), 4),
                     'new_median': round(float(np.median(new)), 4),
                     'new_max': round(float(new.max()), 4),
                     'ratio': round(float(np.median(new) / max(np.median(old), 1e-9)), 4)}
    out['all_ratio'] = round(float(np.median(np.concatenate(news)) /
                                   max(np.median(np.concatenate(olds)), 1e-9)), 4)
    return out
