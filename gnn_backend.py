"""
gnn_backend.py — one interface over the two ways to run the value net.

    backend = make_backend('td_champion_July21_aux_iter14.pt')   # torch
    backend = make_backend('model.onnx')                         # onnxruntime
    scores  = backend(encoded_or_list)                           # -> np.ndarray

Training, self-play and the analysis scripts keep using torch directly. The
deployed web app uses the ONNX backend, which needs only numpy + onnxruntime —
torch is by far the largest thing in the image, and nothing at play time needs
it. Both backends own the encoder that suits them (torch tensors vs numpy
arrays), so callers just ask the backend for `.encoder`.

Scores always come back as a numpy array (a 0-d array for a single position),
whichever backend produced them.
"""

import os

import numpy as np


class TorchBackend:
    """The training-time path: BoardGNN under torch.no_grad()."""

    name = 'torch'

    def __init__(self, weights_path=None, model=None):
        import torch
        from network import BoardEncoder, load_model
        self._torch = torch
        self.encoder = BoardEncoder()
        if model is not None:
            self.model = model
        else:
            self.model = load_model(weights_path)
        self.model.eval()

    def __call__(self, encoded):
        with self._torch.no_grad():
            out = self.model(encoded)
        return out.detach().cpu().numpy()

    def describe(self):
        return f'torch on {next(self.model.parameters()).device}'


class OnnxBackend:
    """The deployment path: onnxruntime over the numpy encoder."""

    name = 'onnx'

    def __init__(self, onnx_path):
        import onnxruntime as ort
        from encoder import BoardEncoder
        self.encoder = BoardEncoder()
        opts = ort.SessionOptions()
        # One thread: the app scores many small batches inside a search loop, so
        # per-call thread pool overhead costs more than the parallelism gains —
        # and the deploy target is a fractional-vCPU instance anyway.
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        # No arena allocator: it grows to fit the largest batch it has ever seen
        # and never gives that back, which on a long game walks the worker up to
        # ~170 MB (batch sizes vary a lot move to move). Measured cost of turning
        # it off is small next to the deadline we care about, and the deploy
        # target is a 256 MB instance.
        opts.enable_cpu_mem_arena = bool(int(os.environ.get('ORT_MEM_ARENA', '0')))
        self.session = ort.InferenceSession(onnx_path, opts,
                                            providers=['CPUExecutionProvider'])
        self.path = onnx_path

    def _collate(self, encoded_list):
        B = len(encoded_list)
        P = encoded_list[0]['piece_feats'].shape[0]
        T = encoded_list[0]['tile_feats'].shape[0]
        p2t_s, p2t_d, t2p_s, t2p_d = [], [], [], []
        for b, e in enumerate(encoded_list):
            p2t, t2p = e['piece_to_tile'], e['tile_to_piece']
            if p2t.shape[1]:
                p2t_s.append(p2t[0] + b * P); p2t_d.append(p2t[1] + b * T)
                t2p_s.append(t2p[0] + b * T); t2p_d.append(t2p[1] + b * P)
        if p2t_s:
            p2t_b = np.stack([np.concatenate(p2t_s), np.concatenate(p2t_d)])
            t2p_b = np.stack([np.concatenate(t2p_s), np.concatenate(t2p_d)])
        else:
            p2t_b = np.zeros((2, 0), dtype=np.int64)
            t2p_b = np.zeros((2, 0), dtype=np.int64)
        return {
            'tile_feats': np.stack([e['tile_feats'] for e in encoded_list]),
            'piece_feats': np.stack([e['piece_feats'] for e in encoded_list]),
            'global_feats': np.stack([e['global_feats'] for e in encoded_list]),
            'piece_to_tile': p2t_b,
            'tile_to_piece': t2p_b,
        }

    def __call__(self, encoded):
        single = isinstance(encoded, dict)
        feeds = self._collate([encoded] if single else list(encoded))
        out = self.session.run(None, feeds)[0]
        return out[0] if single else out

    def describe(self):
        return f'onnxruntime ({self.path})'


def make_backend(path, model=None):
    """ONNX by file extension, torch otherwise (or when handed a live model)."""
    if model is None and str(path).endswith('.onnx'):
        return OnnxBackend(path)
    return TorchBackend(path, model=model)
