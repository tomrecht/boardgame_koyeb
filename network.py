"""
network.py — GNN position encoder and value network with true GPU batching.

Architecture:
  - Tile nodes (70): board topology, static structure, dynamic features
  - Piece nodes (24): one per piece, always present
  - Global node (1): virtual node connected to all tiles
  - 6 message passing layers
  - Hidden dim: 64

Encoding: always from current player's perspective.
  Current player pieces -> player_id=0, opponent -> player_id=1.

Batching:
  BoardGNN.forward() accepts a list of encoded dicts and processes them
  as a true GPU batch — all N positions in one set of matrix multiplications.

  Tile-tile edges are shared (topology never changes) -> broadcast across batch.
  Piece-tile edges vary per position -> block-diagonal construction.
  Global node: one per batch item.

Auxiliary head:
  Predicts (my_saved - opponent_saved) / 12, computed directly from board state.
  Provides grounded supervision signal every step, prevents embedding drift.
  Weight: 0.2 relative to main value loss.
  Only used during training; forward() returns value only for move selection.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------
# CONSTANTS
# -------------------------

NUM_PIECES      = 12
TOTAL_PIECES    = 24
HIDDEN_DIM      = 64
NUM_MP_LAYERS   = 6
TILE_FEAT_DIM   = 8
PIECE_FEAT_DIM  = 8
GLOBAL_FEAT_DIM = 4   # [die1_val, die2_val, die1_used, die2_used]
AUX_LOSS_WEIGHT = 0.2


# -------------------------
# TILE GRAPH  (built once)
# -------------------------

def build_tile_index(tile_neighbors_path='tile_neighbors.json'):
    """
    Returns:
      tile_index:      dict (ring, sector) -> int
      tile_info:       dict (ring, sector) -> json entry
      tile_edge_index: LongTensor [2, E]  (both directions, on DEVICE)
    Nogo tiles excluded.
    """
    with open(tile_neighbors_path) as f:
        raw = json.load(f)

    tile_keys, tile_info = [], {}
    for key, val in raw.items():
        if val['type'] == 'nogo':
            continue
        r, s = _parse_key(key)
        tile_keys.append((r, s))
        tile_info[(r, s)] = val

    tile_index = {coords: idx for idx, coords in enumerate(tile_keys)}

    edges = set()
    for (r, s), val in tile_info.items():
        i = tile_index[(r, s)]
        for nb in val['neighbors']:
            nb_coords = (nb['ring'], nb['sector'])
            if nb_coords in tile_index:
                j = tile_index[nb_coords]
                edges.add((i, j))
                edges.add((j, i))

    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    tile_edge_index = torch.tensor([src, dst], dtype=torch.long, device=DEVICE)
    return tile_index, tile_info, tile_edge_index


def _parse_key(key):
    parts = key.replace('ring', '').replace('sector', '').split('_')
    return int(parts[0]), int(parts[1])


# -------------------------
# TILE FEATURES  (static)
# -------------------------

def encode_tile_features(tile_index, tile_info):
    """
    [num_tiles, TILE_FEAT_DIM] — never changes between positions.
    [0] is_home  [1] is_field  [2] is_save
    [3] traversable_as_waypoint (0 for home only)
    [4] ring/7   [5] sector/45  [6] save_number/6  [7] reserved
    """
    n = len(tile_index)
    f = torch.zeros(n, TILE_FEAT_DIM)
    for (ring, sector), idx in tile_index.items():
        info = tile_info[(ring, sector)]
        t = info['type']
        f[idx, 0] = float(t == 'home')
        f[idx, 1] = float(t == 'field')
        f[idx, 2] = float(t == 'save')
        f[idx, 3] = 0.0 if t == 'home' else 1.0
        f[idx, 4] = ring / 7.0
        f[idx, 5] = sector / 45.0
        f[idx, 6] = info.get('number', 0) / 6.0
    return f.to(DEVICE)


# -------------------------
# PIECE FEATURES  (per position)
# -------------------------

STATUS_UNENTERED    = 0
STATUS_ON_HOME      = 1
STATUS_ON_BOARD     = 2
STATUS_CAN_BE_SAVED = 3
STATUS_SAVED        = 4
NUM_STATUSES        = 5


def _piece_status(piece, board):
    if piece.rack in (board.white_saved, board.black_saved):
        return STATUS_SAVED
    if piece.rack in (board.white_unentered, board.black_unentered):
        return STATUS_UNENTERED
    if piece.tile is not None:
        if piece.tile.type == 'home':
            return STATUS_ON_HOME
        if piece.can_be_saved():
            return STATUS_CAN_BE_SAVED
        return STATUS_ON_BOARD
    return STATUS_ON_BOARD


def encode_piece_features(board, tile_index, current_player):
    opponent = 'black' if current_player == 'white' else 'white'
    cur_pieces = sorted([p for p in board.pieces if p.player == current_player],
                        key=lambda p: p.number)
    opp_pieces = sorted([p for p in board.pieces if p.player == opponent],
                        key=lambda p: p.number)
    all_pieces = cur_pieces + opp_pieces

    cur_un = board.white_unentered if current_player == 'white' else board.black_unentered
    opp_un = board.black_unentered if current_player == 'white' else board.white_unentered
    rack_pos = {p: i for i, p in enumerate(cur_un)}
    rack_pos.update({p: i for i, p in enumerate(opp_un)})

    statuses = [_piece_status(p, board) for p in all_pieces]

    # Build rows as Python lists, then construct tensor in one call
    rows = []
    for piece, status in zip(all_pieces, statuses):
        onehot = [0.0] * 5
        onehot[status] = 1.0
        rows.append([
            0.0 if piece.player == current_player else 1.0,
            piece.number / 12.0,
            *onehot,
            rack_pos.get(piece, 0) / 11.0 if status == STATUS_UNENTERED else 0.0,
        ])

    f = torch.tensor(rows, dtype=torch.float32, device=DEVICE)
    return f, all_pieces


# -------------------------
# PIECE->TILE EDGES  (per position)
# -------------------------

def encode_piece_tile_edges(all_pieces, tile_index):
    """
    piece_to_tile: [2, N_onboard]  row0=piece_idx, row1=tile_idx
    tile_to_piece: reverse
    """
    psrc, tdst = [], []
    for pidx, piece in enumerate(all_pieces):
        if piece.tile is not None:
            coords = (piece.tile.ring, piece.tile.pos)
            if coords in tile_index:
                psrc.append(pidx)
                tdst.append(tile_index[coords])
    if psrc:
        p2t = torch.tensor([psrc, tdst], dtype=torch.long, device=DEVICE)
        t2p = torch.tensor([tdst, psrc], dtype=torch.long, device=DEVICE)
    else:
        p2t = torch.zeros(2, 0, dtype=torch.long, device=DEVICE)
        t2p = torch.zeros(2, 0, dtype=torch.long, device=DEVICE)
    return p2t, t2p


# -------------------------
# GLOBAL FEATURES  (dice)
# -------------------------

def encode_global_features(board):
    d1, d2 = board.dice[0], board.dice[1]
    return torch.tensor([d1.number/6., d2.number/6.,
                         float(d1.used), float(d2.used)], device=DEVICE)


# -------------------------
# AUXILIARY TARGET
# -------------------------

def compute_aux_target(board, current_player):
    """
    Auxiliary supervision: (my_saved - opponent_saved) / 12.
    Range [-1, 1]. Computed directly from board state — always correct.
    """
    my_saved  = len(board.white_saved if current_player == 'white' else board.black_saved)
    opp_saved = len(board.black_saved if current_player == 'white' else board.white_saved)
    return (my_saved - opp_saved) / float(NUM_PIECES)


# -------------------------
# BOARD ENCODER
# -------------------------

class BoardEncoder:
    """
    Converts a Board into a dict of tensors.
    Instantiate once; call encode() per position.
    encode() optionally returns aux_target for training.
    """
    def __init__(self, tile_neighbors_path='tile_neighbors.json'):
        self.tile_index, self.tile_info, self.tile_edge_index = \
            build_tile_index(tile_neighbors_path)
        self.num_tiles = len(self.tile_index)
        self._tile_feats = encode_tile_features(self.tile_index, self.tile_info)

    def encode(self, board, current_player, include_aux_target=False):
        piece_feats, all_pieces = encode_piece_features(
            board, self.tile_index, current_player)
        p2t, t2p = encode_piece_tile_edges(all_pieces, self.tile_index)
        encoded = {
            'tile_feats':       self._tile_feats,
            'piece_feats':      piece_feats,
            'tile_edge_index':  self.tile_edge_index,
            'piece_to_tile':    p2t,
            'tile_to_piece':    t2p,
            'global_feats':     encode_global_features(board),
        }
        if include_aux_target:
            return encoded, compute_aux_target(board, current_player)
        return encoded


# -------------------------
# BATCH CONSTRUCTION
# -------------------------

def collate_batch(encoded_list):
    """
    Stack a list of encoded dicts into batch tensors.

    Tile features: [B, T, TF]  — stack along new batch dim
    Piece features:[B, P, PF]
    Global features:[B, GF]

    Tile-tile edges: shared — [2, E], used as-is for all batch items
      (message passing broadcasts over batch dim)

    Piece-tile edges: block-diagonal — offset each item's indices so
      batch item b's piece indices are in [b*P, (b+1)*P) and
      tile indices are in [b*T, (b+1)*T).
      Combined: [2, sum(N_i)]
    """
    B = len(encoded_list)
    T = encoded_list[0]['tile_feats'].size(0)
    P = encoded_list[0]['piece_feats'].size(0)

    tile_feats_b   = torch.stack([e['tile_feats']   for e in encoded_list])  # [B,T,TF]
    piece_feats_b  = torch.stack([e['piece_feats']  for e in encoded_list])  # [B,P,PF]
    global_feats_b = torch.stack([e['global_feats'] for e in encoded_list])  # [B,GF]

    # Tile-tile edges are shared — same for all batch items
    tile_edge_index = encoded_list[0]['tile_edge_index']  # [2, E]

    # Piece-tile edges: build block-diagonal
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
        'tile_feats':       tile_feats_b,
        'piece_feats':      piece_feats_b,
        'global_feats':     global_feats_b,
        'tile_edge_index':  tile_edge_index,
        'piece_to_tile':    p2t_b,
        'tile_to_piece':    t2p_b,
        'B': B, 'T': T, 'P': P,
    }


# -------------------------
# MEAN AGGREGATION
# -------------------------

def _mean_agg(messages, dst, num_dst, dim, device):
    """Mean-pool messages into destination nodes. [E,H] -> [num_dst, H]"""
    agg   = torch.zeros(num_dst, dim, device=device)
    count = torch.zeros(num_dst, 1,   device=device)
    agg.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)
    count.scatter_add_(0, dst.unsqueeze(1),
                       torch.ones(dst.size(0), 1, device=device))
    return agg / count.clamp(min=1)


# -------------------------
# MESSAGE PASSING LAYER  (batched)
# -------------------------

class MessagePassingLayer(nn.Module):
    """
    One round of batched heterogeneous message passing.
    All inputs are batched: tile_h [B,T,H], piece_h [B,P,H], global_h [B,1,H].
    Edge indices use block-diagonal offsets for piece-tile, broadcast for tile-tile.

    Steps:
      1. tile  -> piece
      2. piece -> tile
      3. tile  -> tile   (shared edges, batched scatter)
      4. tile  -> global (mean pool per batch item)
      5. global -> tile  (broadcast per batch item)
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
        """
        tile_h:   [B, T, H]
        piece_h:  [B, P, H]
        global_h: [B, 1, H]
        tile_edge_index: [2, E]  shared (not block-diagonal)
        p2t, t2p: [2, sum_N]  block-diagonal
        """
        H   = tile_h.size(2)
        dev = tile_h.device

        # Flatten batch for scatter operations
        tile_flat  = tile_h.view(B * T, H)
        piece_flat = piece_h.view(B * P, H)

        # --- 1. tile -> piece  (block-diagonal) ---
        if t2p.size(1) > 0:
            msgs = self.tile_to_piece_msg(tile_flat[t2p[0]])
            agg  = _mean_agg(msgs, t2p[1], B * P, H, dev)
        else:
            agg = torch.zeros(B * P, H, device=dev)
        piece_flat = F.relu(self.piece_update(
            torch.cat([piece_flat, agg], dim=1)))

        # --- 2. piece -> tile  (block-diagonal) ---
        if p2t.size(1) > 0:
            msgs = self.piece_to_tile_msg(piece_flat[p2t[0]])
            agg  = _mean_agg(msgs, p2t[1], B * T, H, dev)
        else:
            agg = torch.zeros(B * T, H, device=dev)
        tile_flat = F.relu(self.tile_update_pieces(
            torch.cat([tile_flat, agg], dim=1)))

        # --- 3. tile -> tile  (shared edges, apply to each batch item) ---
        tile_h = tile_flat.view(B, T, H)
        src, dst = tile_edge_index[0], tile_edge_index[1]
        src_feats = tile_h[:, src, :]
        msgs = self.tile_to_tile_msg(src_feats)
        agg = torch.zeros(B, T, H, device=dev)
        count = torch.zeros(B, T, 1, device=dev)
        dst_exp = dst.view(1, -1, 1).expand(B, -1, H)
        agg.scatter_add_(1, dst_exp, msgs)
        count.scatter_add_(1, dst.view(1, -1, 1).expand(B, -1, 1),
                           torch.ones(B, src.size(0), 1, device=dev))
        agg = agg / count.clamp(min=1)
        tile_h = F.relu(self.tile_update_tiles(
            torch.cat([tile_h, agg], dim=2)))

        # --- 4. tile -> global  (mean pool per batch item) ---
        global_agg = self.tile_to_global_msg(tile_h).mean(dim=1, keepdim=True)
        global_h   = F.relu(self.global_update(
            torch.cat([global_h, global_agg], dim=2)))

        # --- 5. global -> tile  (broadcast per batch item) ---
        global_msg = self.global_to_tile_msg(global_h).expand(B, T, H)
        tile_h = F.relu(self.tile_update_global(
            torch.cat([tile_h, global_msg], dim=2)))

        piece_h = piece_flat.view(B, P, H)
        return tile_h, piece_h, global_h


# -------------------------
# FULL NETWORK
# -------------------------

class BoardGNN(nn.Module):
    """
    Heterogeneous GNN with global node, true GPU batching, and auxiliary head.

    Value head:     predicts position value in [-1, 1]
    Auxiliary head: predicts (my_saved - opp_saved) / 12  in [-1, 1]
                    only used during training, weight AUX_LOSS_WEIGHT=0.2

    forward() returns value only — drop-in compatible with agent_gnn.py.
    forward_with_aux() returns (value, aux) for training.
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

        self.tile_embed   = nn.Linear(tile_feat_dim,  H)
        self.piece_embed  = nn.Linear(piece_feat_dim, H)
        self.global_embed = nn.Embedding(1, H)
        nn.init.zeros_(self.global_embed.weight)

        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(H) for _ in range(num_mp_layers)
        ])

        # Shared trunk input size: mean(tiles) + mean(pieces) + global + dice
        trunk_dim = H * 3 + global_feat_dim

        # Value head (unchanged from original)
        self.readout = nn.Sequential(
            nn.Linear(trunk_dim, H),
            nn.ReLU(),
            nn.Linear(H, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Auxiliary head — lightweight, branches off same combined vector
        self.aux_head = nn.Sequential(
            nn.Linear(trunk_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),   # output in [-1, 1] matching aux target range
        )

    def _forward_batch(self, batch):
        """
        Core batched forward pass.
        Returns: (value [B], aux [B])
        """
        B = batch['B']
        T = batch['T']
        P = batch['P']
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

        value = self.readout(combined).squeeze(1)   # [B]
        aux   = self.aux_head(combined).squeeze(1)  # [B]
        return value, aux

    def forward(self, encoded):
        """
        Value only — drop-in compatible with agent_gnn.py.
        encoded: single dict or list of dicts.
        Returns scalar (single) or [N] tensor (list).
        """
        if isinstance(encoded, dict):
            batch = collate_batch([encoded])
            value, _ = self._forward_batch(batch)
            return value.squeeze(0)
        batch = collate_batch(encoded)
        value, _ = self._forward_batch(batch)
        return value

    def forward_with_aux(self, encoded_list):
        """
        Returns (value [B], aux [B]) — used during training only.
        encoded_list: list of encoded dicts.
        """
        batch = collate_batch(encoded_list)
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