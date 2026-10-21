"""make_soup.py -- weight average ("soup") of checkpoints from one basin.

The 4-way blend (blend4way_Oct6, +0.16 over its best ingredient) was this: nets
of one lineage averaged weight by weight. Ingredients must share a feature set
and frame; float tensors are averaged with the given weights, buffers such as
afterstate_flag must agree and are copied.

    python3 make_soup.py out.pt w1:a.pt w2:b.pt ...     (weights normalised)
"""
import sys

import torch

import network


def main():
    out, parts = sys.argv[1], [a.split(':', 1) for a in sys.argv[2:]]
    ws = [float(w) for w, _ in parts]
    total = sum(ws)
    sds = [torch.load(p, map_location='cpu') for _, p in parts]
    feats = {network.features_of_state(sd) for sd in sds}
    if len(feats) != 1:
        raise SystemExit(f'ingredients mix feature sets: {feats}')
    soup = {}
    for k, v in sds[0].items():
        if any(set(sd) != set(sds[0]) for sd in sds):
            raise SystemExit('ingredients have different keys')
        if v.dtype.is_floating_point and k != 'afterstate_flag':
            soup[k] = sum(w / total * sd[k].float() for w, sd in zip(ws, sds)).to(v.dtype)
        else:
            if any(not torch.equal(sd[k], v) for sd in sds):
                raise SystemExit(f'{k} differs between ingredients')
            soup[k] = v.clone()
    network.model_from_state(soup)          # loads strictly, or raises
    torch.save(soup, out)
    print(f'{out}: ' + ' + '.join(f'{w / total:.3f} x {p}' for w, (_, p) in zip(ws, parts))
          + f' ({feats.pop()}, afterstate {float(soup.get("afterstate_flag", 0))})')


if __name__ == '__main__':
    main()
