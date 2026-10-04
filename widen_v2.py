"""Widen a v1 checkpoint into an A / AB net that computes EXACTLY the same value.

features_v2's extra inputs are appended columns, so the three input-facing
weight matrices just grow: the copied columns keep the champion's weights and
the new ones start at zero, so the new inputs contribute nothing until training
moves them (their gradients are not zero -- d(loss)/dW = upstream * input). Every
other parameter is copied unchanged. The result is a warm start for
league_run.py (WARM_START=<out>) that begins exactly at the champion's strength.

    python3 widen_v2.py symaug_iter6.pt AB symaug6_AB.pt

Verifies on positions from owner's logged games that the widened net, fed the
AB encoding, scores every position identically to the champion fed v1.
"""
import json, sys

import numpy as np
import torch

import network
from network import BoardGNN, BoardEncoder

GROWN = ('tile_embed.weight', 'piece_embed.weight', 'readout.0.weight')


def widen(v1_sd, features):
    student = BoardGNN(features=features)
    sd = student.state_dict()
    for k, v in v1_sd.items():
        if k not in sd:
            raise KeyError(f'{k} not in a {features} BoardGNN')
        if k in GROWN:
            w = torch.zeros_like(sd[k])
            w[:, :v.shape[1]] = v           # appended columns stay zero
            sd[k] = w
        else:
            if sd[k].shape != v.shape:
                raise ValueError(f'{k}: {tuple(v.shape)} vs {tuple(sd[k].shape)}')
            sd[k] = v.clone()
    student.load_state_dict(sd)
    student.eval()
    return student


def verify(teacher, student, n_games=3):
    import weakness_probe as W
    e1 = BoardEncoder(features='v1')
    e2 = BoardEncoder(features=student.features)
    recs = [json.loads(l) for l in open('quahuru-games-apvo2h65.jsonl')][5:5 + n_games]
    worst, n = 0.0, 0
    with torch.no_grad():
        for rec in recs:
            for t, b, played, b2 in W.turn_states(rec):
                for bb in (b, b2):
                    a = float(teacher(e1.encode(bb, bb.current_player)))
                    c = float(student(e2.encode(bb, bb.current_player)))
                    worst = max(worst, abs(a - c))
                    n += 1
    return n, worst


def main():
    src, features, out = sys.argv[1], sys.argv[2], sys.argv[3]
    network.DEVICE = torch.device('cpu')
    v1_sd = torch.load(src, map_location='cpu')
    if network.features_of_state(v1_sd) != 'v1':
        raise SystemExit(f'{src} is not a v1 checkpoint')
    teacher = network.model_from_state(v1_sd)
    student = widen(v1_sd, features)
    n, worst = verify(teacher, student)
    print(f'{features} student vs v1 teacher on {n} positions: max |diff| {worst:.2e}')
    if worst > 1e-5:
        raise SystemExit('widened net does not reproduce the teacher; not saved')
    torch.save(student.state_dict(), out)
    print(f'saved {out} ({features})')


if __name__ == '__main__':
    main()
