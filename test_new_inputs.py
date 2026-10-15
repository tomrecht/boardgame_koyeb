"""Tests for new_inputs.py and the A2/AB2 input scaling (python3 test_new_inputs.py)."""
import json, random, sys
import numpy as np
import torch

import network
network.DEVICE = torch.device('cpu')
import encoder as E
import new_inputs as N
from network import BoardGNN


def main():
    ok = True

    def check(name, good, detail=''):
        nonlocal ok
        ok &= bool(good)
        print(f'  {name}: {"ok" if good else "FAIL"} {detail}')

    print('multiplier schedule (10x, anneal 20)')
    got = [round(N.mult_for(i, 10, 20), 2) for i in (1, 11, 21, 30)]
    check('it 1/11/21/30', got == [10.0, 5.5, 1.0, 1.0], got)

    print('BoostedAdam: new columns step 10x, everything else exactly as Adam')
    torch.manual_seed(0)
    a = BoardGNN(features='AB2'); b = BoardGNN(features='AB2'); b.load_state_dict(a.state_dict())
    oa = torch.optim.Adam(a.parameters(), lr=1e-3); ob = N.BoostedAdam(b, lr=1e-3, mult=10, anneal=20, it=1)
    for m in (a, b):
        for p in m.parameters():
            p.grad = torch.ones_like(p) * 0.3
    sd0 = {k: v.clone() for k, v in a.state_dict().items()}
    oa.step(); ob.step()
    cols = N.new_masks('AB2')
    worst_old, ratios = 0.0, []
    for k, v0 in sd0.items():
        da, db = a.state_dict()[k] - v0, b.state_dict()[k] - v0
        if k in cols:
            s = cols[k][1]
            worst_old = max(worst_old, float((da[:, :s] - db[:, :s]).abs().max()))
            ratios.append(float((db[:, s:] / da[:, s:]).mean()))
        else:
            worst_old = max(worst_old, float((da - db).abs().max()))
    check('old params identical', worst_old < 1e-9, f'max diff {worst_old:.1e}')
    check('new columns 10x', all(abs(r - 10) < 1e-3 for r in ratios), [round(r, 3) for r in ratios])
    ob.set_iter(25)
    check('mult after anneal', ob.mult == 1.0, ob.mult)

    print('averaging skips the new columns')
    avg = {k: torch.zeros_like(v) for k, v in a.state_dict().items()}
    tr = {k: torch.ones_like(v) if v.dtype.is_floating_point else v for k, v in a.state_dict().items()}
    out = N.blend_skip_new(avg, tr, 0.5, 'AB2')
    k = 'piece_embed.weight'; s = cols[k][1]
    check('old col averaged, new col trained',
          float(out[k][:, :s].mean()) == 0.5 and float(out[k][:, s:].mean()) == 1.0)

    print('input scaling: AB2 new columns multiplied by input_scales.json, v1 columns untouched')
    from game import Board
    states = [json.loads(l)['state'] for l in open('start_pool.jsonl')][:50]
    e0, e1 = E.BoardEncoder(features='AB2', scaled=False), E.BoardEncoder(features='AB2')
    sc = json.load(open('input_scales.json'))['AB2']
    v1 = E.feature_dims('v1')
    worst = 0.0
    for st in states:
        bd = Board(); bd.update_state(st); bd.firstMove = None; bd.get_valid_moves()
        f0, f1 = e0.encode(bd, bd.current_player), e1.encode(bd, bd.current_player)
        for i, kind, key in ((0, 'tile', 'tile_feats'), (1, 'piece', 'piece_feats'), (2, 'global', 'global_feats')):
            x0, x1 = np.asarray(f0[key]), np.asarray(f1[key])
            if x0.ndim == 1:
                x0, x1 = x0[None], x1[None]
            worst = max(worst, float(np.abs(x0[:, :v1[i]] - x1[:, :v1[i]]).max()),
                        float(np.abs(x0[:, v1[i]:] * np.asarray(sc[kind]) - x1[:, v1[i]:]).max()))
    check('50 pool positions', worst < 1e-6, f'max diff {worst:.1e}')
    eab = E.BoardEncoder(features='AB')
    check('AB is never scaled', eab._scales is None)

    print('usage: zero-initialised new columns read 0, the fv2B champion small')
    spread = N.input_spread(network.BoardEncoder(features='AB'), states, n=50)
    u0 = N.usage(torch.load('blend_AB.pt', map_location='cpu'), spread, 'AB')
    check('blend_AB ratio 0', u0['all_ratio'] == 0.0, u0['all_ratio'])
    import os
    if os.path.exists('runs/fv2B_champion.pt'):
        u1 = N.usage(torch.load('runs/fv2B_champion.pt', map_location='cpu'), spread, 'AB')
        check('fv2B champion ratio in (0, 0.1)', 0 < u1['all_ratio'] < 0.1, u1['all_ratio'])

    print('\nALL OK' if ok else '\nFAILURES')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
