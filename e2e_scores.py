"""e2e_scores.py -- Python reference scores for model_e2e.mjs (js-features-v2).

Scores every case of a features_v2 fixture (dump_v2_fixture.py) with an exported
model through OnnxBackend, which reads the sidecar beside the .onnx. Afterstate
nets score the position as training and the search see it: rebuilt with
update_state, which applies the last-piece rule (as GNNAgent._encode_after).
model_e2e.mjs compares these with Infer.score in the real page.

    python3 e2e_scores.py <fixture.json> <model.onnx> <out.json>
"""
import json, sys

import numpy as np

from game import Board
from gnn_backend import OnnxBackend
from dump_v2_fixture import _raw_from_snapshot


def main(fx_path, onnx_path, out):
    be = OnnxBackend(onnx_path)
    tag = 'v1' if be.encoder.features == 'v1' else 'AB_after'
    b = Board()
    rows = []
    for c in json.load(open(fx_path))['cases']:
        b.update_state(_raw_from_snapshot(c['snapshot']))
        v = float(np.asarray(be(be.encoder.encode(b, c['player']))).ravel()[0])
        rows.append({'model': tag, 'kind': c['kind'], 'player': c['player'],
                     'snapshot': c['snapshot'], 'py': v})
    json.dump(rows, open(out, 'w'))
    print(f'wrote {out}: {len(rows)} scores ({be.encoder.features}, afterstate '
          f'{be.encoder.afterstate}), tag {tag}')


if __name__ == '__main__':
    main(*sys.argv[1:4])
