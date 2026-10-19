"""Widen a checkpoint into a larger feature set (v1 -> A / AB / A2 / AB2, A -> A2 /
AB / AB2, AB -> AB2) whose net computes EXACTLY the same value.

The three input-facing weight matrices grow: each source column is copied to
where its input sits in the target set (columns are mapped by BLOCK, because
A2/AB2's two globals sit BETWEEN A's and AB's roll opportunities, so AB -> AB2
is not an append), and the new columns start at zero, so the new inputs contribute nothing until training
moves them (their gradients are not zero -- d(loss)/dW = upstream * input). Every
other parameter is copied unchanged. The result is a warm start for
league_run.py (WARM_START=<out>) that begins exactly at the champion's strength.

    python3 widen_v2.py symaug_iter6.pt AB symaug6_AB.pt
    python3 widen_v2.py runs/fv2B_champion.pt AB2 fv2B_AB2.pt

Verifies on positions from owner's logged games that the widened net, fed the
target encoding, scores every position identically to the source fed its own.
The afterstate flag is carried over; league_run.py sets it on a fresh start.
"""
import json, sys

import numpy as np
import torch

import network
from network import BoardGNN, BoardEncoder

import encoder
import features_v2 as fv

# Input blocks per set, in column order (features_v2.py / encoder._extras).
BLOCKS = {
    'tile':   {'v1': ['base'], 'A': ['base', 'wall'], 'AB': ['base', 'wall'],
               'A2': ['base', 'wall'], 'AB2': ['base', 'wall']},
    'piece':  {'v1': ['base'], 'A': ['base', 'own'], 'AB': ['base', 'own', 'threat'],
               'A2': ['base', 'own'], 'AB2': ['base', 'own', 'threat']},
    'global': {'v1': ['base'], 'A': ['base', 'A'], 'AB': ['base', 'A', 'roll'],
               'A2': ['base', 'A', 'ns'], 'AB2': ['base', 'A', 'ns', 'roll']},
}
SIZE = {
    'tile':   {'base': encoder.TILE_FEAT_DIM, 'wall': 1},
    'piece':  {'base': encoder.PIECE_FEAT_DIM, 'own': 1, 'threat': 4},
    'global': {'base': encoder.GLOBAL_FEAT_DIM, 'A': 14, 'ns': 2, 'roll': 72},
}
# weight matrix -> (which inputs, column offset of the first input)
GROWN = {'tile_embed.weight': ('tile', 0), 'piece_embed.weight': ('piece', 0),
         'readout.0.weight': ('global', 3 * network.HIDDEN_DIM)}


def column_map(kind, src, dst):
    """[(src_col, dst_col, width)] for every block of src, located in dst."""
    if not set(BLOCKS[kind][src]) <= set(BLOCKS[kind][dst]):
        raise SystemExit(f'cannot widen {src} -> {dst}: {kind} inputs are not a superset')
    def starts(f):
        out, at = {}, 0
        for b in BLOCKS[kind][f]:
            out[b] = at
            at += SIZE[kind][b]
        assert at == encoder.feature_dims(f)[('tile', 'piece', 'global').index(kind)]
        return out
    s0, d0 = starts(src), starts(dst)
    return [(s0[b], d0[b], SIZE[kind][b]) for b in BLOCKS[kind][src]]


def col_scales(kind, f):
    """Per-column input factor of set f (1 for v1 columns; A2/AB2's new columns
    are multiplied by input_scales.json in the encoder)."""
    n = encoder.feature_dims(f)[('tile', 'piece', 'global').index(kind)]
    base = SIZE[kind]['base']
    out = np.ones(n, dtype=np.float32)
    if f in ('A2', 'AB2'):
        sc = json.load(open('input_scales.json'))[f][kind]
        out[base:] = sc
    return out


def widen(src_sd, features):
    src = network.features_of_state(src_sd)
    student = BoardGNN(features=features)
    sd = student.state_dict()
    for k, v in src_sd.items():
        if k not in sd:
            raise KeyError(f'{k} not in a {features} BoardGNN')
        if k in GROWN:
            kind, off = GROWN[k]
            w = torch.zeros_like(sd[k])
            w[:, :off] = v[:, :off]          # readout's hidden inputs
            ss, ds = col_scales(kind, src), col_scales(kind, features)
            for a, b, n in column_map(kind, src, features):
                # an input scaled up by s needs its weight divided by s
                r = torch.from_numpy(ss[a:a + n] / ds[b:b + n])
                w[:, off + b:off + b + n] = v[:, off + a:off + a + n] * r
            sd[k] = w                        # new columns stay zero
        else:
            if sd[k].shape != v.shape:
                raise ValueError(f'{k}: {tuple(v.shape)} vs {tuple(sd[k].shape)}')
            sd[k] = v.clone()
    student.load_state_dict(sd)
    student.eval()
    return student


def _verify_boards(n_games=3, n_pool=300):
    """Boards to check the widened net on: owner's game log replayed (both turn
    start and after) when it is here (it is gitignored, so iMac only), else the
    tracked start pool's positions."""
    import os
    log = 'quahuru-games-apvo2h65.jsonl'
    if os.path.exists(log):
        import weakness_probe as W
        recs = [json.loads(l) for l in open(log)][5:5 + n_games]
        for rec in recs:
            for t, b, played, b2 in W.turn_states(rec):
                yield b
                yield b2
        return
    from game import Board
    with open('start_pool.jsonl') as f:
        for i, l in enumerate(f):
            if i >= n_pool:
                break
            b = Board()
            b.update_state(json.loads(l)['state'])
            yield b


def verify(teacher, student):
    e1 = BoardEncoder(features=teacher.features)
    e2 = BoardEncoder(features=student.features)
    worst, n = 0.0, 0
    with torch.no_grad():
        for bb in _verify_boards():
            a = float(teacher(e1.encode(bb, bb.current_player)))
            c = float(student(e2.encode(bb, bb.current_player)))
            worst = max(worst, abs(a - c))
            n += 1
    return n, worst


def main():
    src, features, out = sys.argv[1], sys.argv[2], sys.argv[3]
    network.DEVICE = torch.device('cpu')
    src_sd = torch.load(src, map_location='cpu')
    teacher = network.model_from_state(src_sd)
    teacher.eval()
    student = widen(src_sd, features)
    n, worst = verify(teacher, student)
    print(f'{features} student vs {teacher.features} teacher on {n} positions: '
          f'max |diff| {worst:.2e} (afterstate flag {student.afterstate})')
    if worst > 1e-5:
        raise SystemExit('widened net does not reproduce the teacher; not saved')
    torch.save(student.state_dict(), out)
    print(f'saved {out} ({features})')


if __name__ == '__main__':
    main()
