"""
onnx_export.py — export a champion checkpoint to ONNX for deployment.

    python onnx_export.py td_champion_July21_aux_iter14.pt model.onnx

Why a re-expressed forward pass: the training forward takes B/T/P as Python
ints and allocates with them (torch.zeros(B * P, H), .expand(B, T, H), ...).
Under tracing those become baked-in constants, so the exported graph would only
accept the one batch size it was traced with. `_ExportGNN` below computes every
shape from the input tensors instead, which keeps the batch axis (and both edge
counts) dynamic. It shares the trained module's own layers — no weights are
copied or re-implemented — and `verify()` checks it against the real forward
pass on live positions before anything is written.

The empty-edge branches in the training forward are not reproduced, and don't
need to be: mean-aggregating zero messages yields the same zeros that branch
returns (count.clamp(min=1) makes it 0/1), which is what happens on an opening
position where no piece is on a tile yet. verify() covers that case.
"""

import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from network import BoardEncoder, load_model, collate_batch


class _ExportGNN(nn.Module):
    """The trained model's forward pass, written with shape-derived sizes."""

    def __init__(self, model, tile_edge_index):
        super().__init__()
        self.m = model
        self.register_buffer('tei', tile_edge_index.cpu())

    @staticmethod
    def _mean_agg(messages, dst, like):
        """Mean-pool `messages` into rows of a zeros tensor shaped like `like`."""
        agg = torch.zeros_like(like)
        count = torch.zeros_like(like[:, :1])
        idx = dst.unsqueeze(1).expand_as(messages)
        agg = agg.scatter_add(0, idx, messages)
        count = count.scatter_add(0, dst.unsqueeze(1),
                                  torch.ones_like(dst.unsqueeze(1), dtype=count.dtype))
        return agg / count.clamp(min=1)

    def _layer(self, mp, tile_h, piece_h, global_h, p2t, t2p):
        tile_flat = tile_h.reshape(-1, tile_h.size(2))
        piece_flat = piece_h.reshape(-1, piece_h.size(2))

        # 1. tile -> piece
        msgs = mp.tile_to_piece_msg(tile_flat.index_select(0, t2p[0]))
        agg = self._mean_agg(msgs, t2p[1], piece_flat)
        piece_flat = piece_flat + F.relu(mp.piece_update(
            torch.cat([piece_flat, agg], dim=1)))

        # 2. piece -> tile
        msgs = mp.piece_to_tile_msg(piece_flat.index_select(0, p2t[0]))
        agg = self._mean_agg(msgs, p2t[1], tile_flat)
        tile_flat = tile_flat + F.relu(mp.tile_update_pieces(
            torch.cat([tile_flat, agg], dim=1)))

        # 3. tile -> tile
        tile_h = tile_flat.reshape(tile_h.size(0), tile_h.size(1), tile_h.size(2))
        src, dst = self.tei[0], self.tei[1]
        msgs = mp.tile_to_tile_msg(tile_h.index_select(1, src))
        agg = torch.zeros_like(tile_h)
        count = torch.zeros_like(tile_h[:, :, :1])
        agg = agg.scatter_add(1, dst.view(1, -1, 1).expand_as(msgs), msgs)
        count = count.scatter_add(1, dst.view(1, -1, 1).expand_as(msgs[:, :, :1]),
                                  torch.ones_like(msgs[:, :, :1]))
        agg = agg / count.clamp(min=1)
        tile_h = tile_h + F.relu(mp.tile_update_tiles(torch.cat([tile_h, agg], dim=2)))

        # 4. tile -> global
        global_agg = mp.tile_to_global_msg(tile_h).mean(dim=1, keepdim=True)
        global_h = global_h + F.relu(mp.global_update(
            torch.cat([global_h, global_agg], dim=2)))

        # 5. global -> tile
        global_msg = mp.global_to_tile_msg(global_h).expand_as(tile_h)
        tile_h = tile_h + F.relu(mp.tile_update_global(
            torch.cat([tile_h, global_msg], dim=2)))

        piece_h = piece_flat.reshape(piece_h.size(0), piece_h.size(1), piece_h.size(2))
        return tile_h, piece_h, global_h

    def forward(self, tile_feats, piece_feats, global_feats, p2t, t2p):
        m = self.m
        tile_h = F.relu(m.tile_embed(tile_feats))
        piece_h = F.relu(m.piece_embed(piece_feats))
        piece_h_init = piece_h
        # [1, 1, H] broadcast up to [B, 1, H] without ever naming B
        global_h = m.global_embed.weight.unsqueeze(0) + torch.zeros_like(tile_feats[:, :1, :1])

        for mp in m.mp_layers:
            tile_h, piece_h, global_h = self._layer(mp, tile_h, piece_h, global_h, p2t, t2p)

        combined = torch.cat([
            tile_h.mean(dim=1),
            (piece_h.mean(dim=1) + piece_h_init.mean(dim=1)) / 2,
            global_h.squeeze(1),
            global_feats,
        ], dim=1)
        return m.readout(combined).squeeze(1)


def _batches(encoder, sizes=(1, 2, 17, 64)):
    """Real positions from a random playout, collated into batches of each size."""
    import random
    from game import Board

    random.seed(3)
    encs, board = [], Board()
    while len(encs) < max(sizes) + 5:
        encs.append(encoder.encode(board, board.current_player))   # first is the empty opening
        moves = board.get_valid_moves()
        if not moves:
            board = Board()
            continue
        board.apply_move(random.choice(moves))
    return [collate_batch(encs[:n]) for n in sizes]


def verify(model, export_model, encoder, atol=1e-5):
    """Export wrapper must match the trained forward pass on live positions."""
    worst = 0.0
    for batch in _batches(encoder):
        with torch.no_grad():
            ref = model(batch)
            got = export_model(batch['tile_feats'], batch['piece_feats'],
                               batch['global_feats'], batch['piece_to_tile'],
                               batch['tile_to_piece'])
        worst = max(worst, float((ref - got).abs().max()))
    assert worst <= atol, f'export wrapper diverges from the model: {worst:.3e}'
    return worst


def verify_onnx(path, model, encoder, atol=1e-4):
    """The written .onnx must match torch, at several batch sizes."""
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    worst = 0.0
    for batch in _batches(encoder):
        with torch.no_grad():
            ref = model(batch).cpu().numpy()
        got = sess.run(None, {
            'tile_feats': batch['tile_feats'].cpu().numpy(),
            'piece_feats': batch['piece_feats'].cpu().numpy(),
            'global_feats': batch['global_feats'].cpu().numpy(),
            'piece_to_tile': batch['piece_to_tile'].cpu().numpy(),
            'tile_to_piece': batch['tile_to_piece'].cpu().numpy(),
        })[0]
        worst = max(worst, float(np.abs(ref - got).max()))
    assert worst <= atol, f'onnx output diverges from torch: {worst:.3e}'
    return worst


def export(ckpt, out_path):
    encoder = BoardEncoder()
    model = load_model(ckpt).cpu().eval()
    wrapper = _ExportGNN(model, encoder.tile_edge_index).eval()

    gap = verify(model, wrapper, encoder)
    print(f'export wrapper matches the model (max diff {gap:.2e})')

    sample = _batches(encoder, sizes=(8,))[0]
    args = (sample['tile_feats'], sample['piece_feats'], sample['global_feats'],
            sample['piece_to_tile'], sample['tile_to_piece'])
    torch.onnx.export(
        wrapper, args, out_path,
        input_names=['tile_feats', 'piece_feats', 'global_feats',
                     'piece_to_tile', 'tile_to_piece'],
        output_names=['value'],
        dynamic_axes={'tile_feats': {0: 'batch'}, 'piece_feats': {0: 'batch'},
                      'global_feats': {0: 'batch'}, 'value': {0: 'batch'},
                      'piece_to_tile': {1: 'p2t_edges'}, 'tile_to_piece': {1: 't2p_edges'}},
        opset_version=17,
        do_constant_folding=True,
    )
    gap = verify_onnx(out_path, model, encoder)
    import os
    print(f'wrote {out_path} ({os.path.getsize(out_path) / 1e6:.2f} MB), '
          f'matches torch across batch sizes (max diff {gap:.2e})')


if __name__ == '__main__':
    ckpt = sys.argv[1] if len(sys.argv) > 1 else 'td_champion_July21_aux_iter14.pt'
    out = sys.argv[2] if len(sys.argv) > 2 else 'model.onnx'
    export(ckpt, out)
