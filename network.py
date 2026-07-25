"""
network.py — torch value network over the board encoding.

The encoder itself now lives in encoder.py (pure numpy) so the deployed web
app can run inference through onnxruntime without torch. Everything the
encoder used to export is re-exported here, so `from network import ...`
keeps working unchanged; the BoardEncoder below is the encoder's output
wrapped back into torch tensors, exactly as before the split.

Architecture:
  - Tile nodes (70): board topology, static structure, dynamic features
  - Piece nodes (24): one per piece, always present
  - Global node (1): virtual node connected to all tiles
  - 6 message passing layers with residual connections
  - Hidden dim: 64
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import encoder as _enc
from encoder import (                      # re-exported: importers see no change
    NUM_PIECES, TOTAL_PIECES, NUM_GOALS, HIDDEN_DIM, NUM_MP_LAYERS,
    TILE_FEAT_DIM, PIECE_FEAT_DIM, GLOBAL_FEAT_DIM, AUX_LOSS_WEIGHT,
    build_tile_index, encode_tile_features, update_tile_features_dynamic,
    encode_piece_features, encode_piece_tile_edges, encode_global_features,
)

# Device: BOARDGAME_DEVICE env var overrides autodetection. Needed because
# MPS support varies by machine/torch build -- e.g. the iMac's Metal stack
# hard-aborts (MPSNDArrayScatter assertion) on the batched scatter_add that
# the M3 Pro handles fine. Launch with BOARDGAME_DEVICE=cpu on machines
# where MPS misbehaves; for the Flask app CPU is the right choice anyway
# (single-position inference is latency-bound, MPS buys nothing there).
_env_device = os.environ.get('BOARDGAME_DEVICE')
if _env_device:
    DEVICE = torch.device(_env_device)
else:
    DEVICE = torch.device('cuda' if torch.cuda.is_available()
                           else 'mps' if torch.backends.mps.is_available()
                           else 'cpu')


def _to_dev(t):
    return t if DEVICE.type == 'cpu' else t.to(DEVICE)


class BoardEncoder(_enc.BoardEncoder):
    """The numpy encoder, handing back torch tensors (zero-copy on CPU)."""

    def __init__(self, tile_neighbors_path='tile_neighbors.json'):
        super().__init__(tile_neighbors_path)
        self.tile_edge_index = _to_dev(torch.from_numpy(self.tile_edge_index))

    def encode(self, board, current_player):
        e = super().encode(board, current_player)
        return {
            'tile_feats':      _to_dev(torch.from_numpy(e['tile_feats'])),
            'piece_feats':     _to_dev(torch.from_numpy(e['piece_feats'])),
            'tile_edge_index': self.tile_edge_index,
            'piece_to_tile':   _to_dev(torch.from_numpy(e['piece_to_tile'])),
            'tile_to_piece':   _to_dev(torch.from_numpy(e['tile_to_piece'])),
            'global_feats':    _to_dev(torch.from_numpy(e['global_feats'])),
        }


# -------------------------
# BATCH CONSTRUCTION
# -------------------------

def collate_batch(encoded_list):
    """Stack a list of encoded dicts into batch tensors."""
    if not encoded_list:
        return {}

    B = len(encoded_list)
    T = encoded_list[0]['tile_feats'].size(0)
    P = encoded_list[0]['piece_feats'].size(0)

    tile_feats_b   = torch.stack([e['tile_feats']   for e in encoded_list])
    piece_feats_b  = torch.stack([e['piece_feats']  for e in encoded_list])
    global_feats_b = torch.stack([e['global_feats'] for e in encoded_list])
    tile_edge_index = encoded_list[0]['tile_edge_index']

    p2t_srcs, p2t_dsts = [], []
    t2p_srcs, t2p_dsts = [], []
    for b, e in enumerate(encoded_list):
        p2t = e['piece_to_tile']
        t2p = e['tile_to_piece']
        if p2t.size(1) > 0:
            p2t_srcs.append(p2t[0] + b * P)
            p2t_dsts.append(p2t[1] + b * T)
        if t2p.size(1) > 0:
            t2p_srcs.append(t2p[0] + b * T)
            t2p_dsts.append(t2p[1] + b * P)

    if p2t_srcs:
        p2t_b = torch.stack([torch.cat(p2t_srcs), torch.cat(p2t_dsts)])
        t2p_b = torch.stack([torch.cat(t2p_srcs), torch.cat(t2p_dsts)])
    else:
        p2t_b = torch.zeros(2, 0, dtype=torch.long, device=DEVICE)
        t2p_b = torch.zeros(2, 0, dtype=torch.long, device=DEVICE)

    return {
        'tile_feats':      tile_feats_b,
        'piece_feats':     piece_feats_b,
        'global_feats':    global_feats_b,
        'tile_edge_index': tile_edge_index,
        'piece_to_tile':   p2t_b,
        'tile_to_piece':   t2p_b,
        'B': B, 'T': T, 'P': P,
    }


# -------------------------
# MEAN AGGREGATION
# -------------------------

def _mean_agg(messages, dst, num_dst, dim, device):
    """Mean-pool messages into destination nodes."""
    agg   = torch.zeros(num_dst, dim, device=device)
    count = torch.zeros(num_dst, 1,   device=device)
    agg.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)
    count.scatter_add_(0, dst.unsqueeze(1),
                       torch.ones(dst.size(0), 1, device=device))
    return agg / count.clamp(min=1)


# -------------------------
# MESSAGE PASSING LAYER  (with residual connections)
# -------------------------

class MessagePassingLayer(nn.Module):
    """
    One round of batched heterogeneous message passing.
    Residual connections on all node updates for better gradient flow.
    """
    def __init__(self, dim):
        super().__init__()
        self.tile_to_piece_msg  = nn.Linear(dim, dim)
        self.piece_update       = nn.Linear(dim * 2, dim)
        self.piece_to_tile_msg  = nn.Linear(dim, dim)
        self.tile_update_pieces = nn.Linear(dim * 2, dim)
        self.tile_to_tile_msg   = nn.Linear(dim, dim)
        self.tile_update_tiles  = nn.Linear(dim * 2, dim)
        self.tile_to_global_msg = nn.Linear(dim, dim)
        self.global_update      = nn.Linear(dim * 2, dim)
        self.global_to_tile_msg = nn.Linear(dim, dim)
        self.tile_update_global = nn.Linear(dim * 2, dim)

    def forward(self, tile_h, piece_h, global_h,
                tile_edge_index, p2t, t2p, B, T, P):
        H   = tile_h.size(2)
        dev = tile_h.device

        tile_flat  = tile_h.reshape(B * T, H)
        piece_flat = piece_h.reshape(B * P, H)

        # 1. tile -> piece  (residual)
        if t2p.size(1) > 0:
            msgs = self.tile_to_piece_msg(tile_flat[t2p[0]])
            agg  = _mean_agg(msgs, t2p[1], B * P, H, dev)
        else:
            agg = torch.zeros(B * P, H, device=dev)
        piece_flat = piece_flat + F.relu(self.piece_update(
            torch.cat([piece_flat, agg], dim=1)))

        # 2. piece -> tile  (residual)
        if p2t.size(1) > 0:
            msgs = self.piece_to_tile_msg(piece_flat[p2t[0]])
            agg  = _mean_agg(msgs, p2t[1], B * T, H, dev)
        else:
            agg = torch.zeros(B * T, H, device=dev)
        tile_flat = tile_flat + F.relu(self.tile_update_pieces(
            torch.cat([tile_flat, agg], dim=1)))

        # 3. tile -> tile  (residual)
        tile_h = tile_flat.reshape(B, T, H)
        src, dst = tile_edge_index[0], tile_edge_index[1]
        src_feats = tile_h[:, src, :]
        msgs = self.tile_to_tile_msg(src_feats)
        agg   = torch.zeros(B, T, H, device=dev)
        count = torch.zeros(B, T, 1, device=dev)
        dst_exp = dst.view(1, -1, 1).expand(B, -1, H)
        agg.scatter_add_(1, dst_exp, msgs)
        count.scatter_add_(1, dst.view(1, -1, 1).expand(B, -1, 1),
                           torch.ones(B, src.size(0), 1, device=dev))
        agg = agg / count.clamp(min=1)
        tile_h = tile_h + F.relu(self.tile_update_tiles(
            torch.cat([tile_h, agg], dim=2)))

        # 4. tile -> global  (residual)
        global_agg = self.tile_to_global_msg(tile_h).mean(dim=1, keepdim=True)
        global_h = global_h + F.relu(self.global_update(
            torch.cat([global_h, global_agg], dim=2)))

        # 5. global -> tile  (residual)
        global_msg = self.global_to_tile_msg(global_h).expand(B, T, H)
        tile_h = tile_h + F.relu(self.tile_update_global(
            torch.cat([tile_h, global_msg], dim=2)))

        piece_h = piece_flat.reshape(B, P, H)
        return tile_h, piece_h, global_h


# -------------------------
# FULL NETWORK
# -------------------------

class BoardGNN(nn.Module):
    """
    Heterogeneous GNN with global node and residual connections.
    """
    def __init__(self,
                 hidden_dim=HIDDEN_DIM,
                 num_mp_layers=NUM_MP_LAYERS,
                 tile_feat_dim=TILE_FEAT_DIM,
                 piece_feat_dim=PIECE_FEAT_DIM,
                 global_feat_dim=GLOBAL_FEAT_DIM):
        super().__init__()
        H = hidden_dim
        self.hidden_dim = H

        self.tile_embed   = nn.Linear(tile_feat_dim,   H)
        self.piece_embed  = nn.Linear(piece_feat_dim,  H)
        self.global_embed = nn.Embedding(1, H)
        nn.init.zeros_(self.global_embed.weight)

        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(H) for _ in range(num_mp_layers)
        ])

        trunk_dim = H * 3 + global_feat_dim

        self.readout = nn.Sequential(
            nn.Linear(trunk_dim, H),
            nn.ReLU(),
            nn.Linear(H, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

    def _forward_batch(self, batch):
        B   = batch['B']
        T   = batch['T']
        P   = batch['P']
        dev = batch['tile_feats'].device

        tile_h   = F.relu(self.tile_embed(batch['tile_feats']))
        piece_h  = F.relu(self.piece_embed(batch['piece_feats']))
        piece_h_init = piece_h
        global_h = self.global_embed(
            torch.zeros(B, 1, dtype=torch.long, device=dev))

        for mp in self.mp_layers:
            tile_h, piece_h, global_h = mp(
                tile_h, piece_h, global_h,
                batch['tile_edge_index'],
                batch['piece_to_tile'],
                batch['tile_to_piece'],
                B, T, P)

        tile_pooled   = tile_h.mean(dim=1)
        piece_pooled  = (piece_h.mean(dim=1) + piece_h_init.mean(dim=1)) / 2
        global_pooled = global_h.squeeze(1)
        combined = torch.cat(
            [tile_pooled, piece_pooled, global_pooled, batch['global_feats']],
            dim=1)

        return self.readout(combined).squeeze(1)

    def forward(self, encoded):
        """
        Accept:
          - a single encoded dict (from encoder.encode)
          - a list of encoded dicts
          - a pre-collated batch dict (from collate_batch, has 'B' key)
        """
        if isinstance(encoded, dict) and 'B' in encoded:
            # Already collated
            return self._forward_batch(encoded)
        if isinstance(encoded, dict):
            batch = collate_batch([encoded])
            return self._forward_batch(batch).squeeze(0)
        batch = collate_batch(encoded)
        return self._forward_batch(batch)


# -------------------------
# SAVE / LOAD
# -------------------------

def save_model(model, path='gnn_weights.pt'):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(path='gnn_weights.pt', **kwargs):
    model = BoardGNN(**kwargs)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded from {path} on {DEVICE}")
    return model