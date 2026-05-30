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

Features:
  Tile: neighbor_count, has_permanent_block
  Piece: is_numbered, is_blot, is_completely_blocked, distance_category
  Global: my_game_stage, opponent_game_stage, numbered_saved_count
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
TILE_FEAT_DIM   = 10
PIECE_FEAT_DIM  = 12
GLOBAL_FEAT_DIM = 7      # +1 for numbered_saved_count
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
# TILE FEATURES  (static + dynamic)
# -------------------------

def encode_tile_features(tile_index, tile_info):
    """
    [num_tiles, TILE_FEAT_DIM]
    [0] is_home  [1] is_field  [2] is_save
    [3] traversable  [4] ring/7  [5] sector/45
    [6] save_number/6  [7] neighbor_count/6
    [8] has_permanent_block  [9] reserved
    """
    n = len(tile_index)
    f = torch.zeros(n, TILE_FEAT_DIM)
    
    for (ring, sector), idx in tile_index.items():
        info = tile_info[(ring, sector)]
        t = info['type']
        
        # Basic type features
        f[idx, 0] = float(t == 'home')
        f[idx, 1] = float(t == 'field')
        f[idx, 2] = float(t == 'save')
        
        # Traversable (not home)
        f[idx, 3] = 0.0 if t == 'home' else 1.0
        
        # Position
        f[idx, 4] = ring / 7.0
        f[idx, 5] = sector / 45.0
        
        # Goal number (if save tile)
        f[idx, 6] = info.get('number', 0) / 6.0
        
        # neighbor_count
        f[idx, 7] = len(info.get('neighbors', [])) / 6.0
        
        # has_permanent_block (will be updated dynamically per position)
        f[idx, 8] = 0.0
        
        # Reserved
        f[idx, 9] = 0.0
    
    return f.to(DEVICE)


def update_tile_features_with_blocks(tile_feats, board, current_player):
    """
    Update the has_permanent_block feature for each tile based on current board state.
    """
    global _tile_index
    tile_feats = tile_feats.clone()
    
    for (ring, sector), idx in _tile_index.items():
        tile = board.get_tile(ring, sector)
        if tile and tile.type == 'field':
            friendly_pieces = [p for p in tile.pieces if p.player == current_player]
            friendly_count = len(friendly_pieces)
            
            if friendly_count >= 2:
                unnumbered_count = sum(1 for p in friendly_pieces if p.number > 6)
                has_permanent_block = 1.0 if unnumbered_count >= 2 else 0.0
                tile_feats[idx, 8] = has_permanent_block
    
    return tile_feats


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
    """
    [TOTAL_PIECES, PIECE_FEAT_DIM]
    [0] player_id  [1] piece_number/12  [2] is_numbered
    [3-7] status onehot (5 bits)  [8] rack_position/11
    [9] is_blot  [10] is_completely_blocked  [11] distance_category
    """
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

    rows = []
    for piece, status in zip(all_pieces, statuses):
        onehot = [0.0] * NUM_STATUSES
        onehot[status] = 1.0
        
        # is_numbered
        is_numbered = 1.0 if piece.number <= 6 else 0.0
        
        # is_blot (alone on field tile)
        is_blot = 0.0
        if piece.tile and piece.tile.type == 'field' and len(piece.tile.pieces) == 1:
            is_blot = 1.0
        
        # is_completely_blocked
        is_completely_blocked = 0.0
        dist = board.shortest_route_to_goal(piece)
        if dist == float('inf'):
            is_completely_blocked = 1.0
        
        # distance_category
        if dist == float('inf'):
            distance_category = 0.0      # blocked
        elif dist <= 6:
            distance_category = 1.0      # single die
        elif dist <= 12:
            distance_category = 0.5      # two dice
        else:
            distance_category = 0.25     # three+ dice
        
        rows.append([
            0.0 if piece.player == current_player else 1.0,  # player_id
            piece.number / 12.0,                             # piece_number
            is_numbered,                                     # is_numbered
            *onehot,                                         # status (5 bits)
            rack_pos.get(piece, 0) / 11.0 if status == STATUS_UNENTERED else 0.0,  # rack_position
            is_blot,                                         # is_blot
            is_completely_blocked,                           # is_completely_blocked
            distance_category,                               # distance_category
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
# GLOBAL FEATURES  (dice + game stages + numbered_saved_count)
# -------------------------

def encode_global_features(board):
    """
    [GLOBAL_FEAT_DIM]
    [0] die1/6  [1] die2/6  [2] die1_used  [3] die2_used
    [4] my_game_stage  [5] opponent_game_stage
    [6] numbered_saved_count / 6
    """
    d1, d2 = board.dice[0], board.dice[1]
    current_player = board.current_player
    opponent = 'black' if current_player == 'white' else 'white'
    
    stage_map = {'opening': 0.0, 'midgame': 0.5, 'endgame': 1.0}
    my_stage = stage_map.get(board.game_stages[current_player], 0.5)
    opp_stage = stage_map.get(board.game_stages[opponent], 0.5)
    
    # Count how many numbered pieces are already saved
    saved_rack = board.white_saved if current_player == 'white' else board.black_saved
    numbered_saved = 0
    for piece in board.pieces:
        if piece.player == current_player and piece.number <= 6:
            if piece.rack == saved_rack:
                numbered_saved += 1
    
    numbered_saved_norm = numbered_saved / 6.0
    
    return torch.tensor([
        d1.number / 6.0,
        d2.number / 6.0,
        float(d1.used),
        float(d2.used),
        my_stage,
        opp_stage,
        numbered_saved_norm,
    ], device=DEVICE)


# -------------------------
# BOARD ENCODER
# -------------------------

_tile_index = None
_tile_info = None

class BoardEncoder:
    """
    Converts a Board into a dict of tensors.
    Instantiate once; call encode() per position.
    """
    def __init__(self, tile_neighbors_path='tile_neighbors.json'):
        global _tile_index, _tile_info
        self.tile_index, self.tile_info, self.tile_edge_index = \
            build_tile_index(tile_neighbors_path)
        _tile_index = self.tile_index
        _tile_info = self.tile_info
        self.num_tiles = len(self.tile_index)
        self._base_tile_feats = encode_tile_features(self.tile_index, self.tile_info)

    def encode(self, board, current_player):
        # Update tile features with dynamic block information
        tile_feats = update_tile_features_with_blocks(
            self._base_tile_feats, board, current_player)
        
        piece_feats, all_pieces = encode_piece_features(
            board, self.tile_index, current_player)
        p2t, t2p = encode_piece_tile_edges(all_pieces, self.tile_index)
        
        encoded = {
            'tile_feats':       tile_feats,
            'piece_feats':      piece_feats,
            'tile_edge_index':  self.tile_edge_index,
            'piece_to_tile':    p2t,
            'tile_to_piece':    t2p,
            'global_feats':     encode_global_features(board),
        }
        return encoded


# -------------------------
# BATCH CONSTRUCTION
# -------------------------

def collate_batch(encoded_list):
    """
    Stack a list of encoded dicts into batch tensors.
    """
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
    """Mean-pool messages into destination nodes."""
    agg = torch.zeros(num_dst, dim, device=device)
    count = torch.zeros(num_dst, 1, device=device)
    agg.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)
    count.scatter_add_(0, dst.unsqueeze(1),
                       torch.ones(dst.size(0), 1, device=device))
    return agg / count.clamp(min=1)


# -------------------------
# MESSAGE PASSING LAYER
# -------------------------

class MessagePassingLayer(nn.Module):
    """
    One round of batched heterogeneous message passing.
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
        H = tile_h.size(2)
        dev = tile_h.device

        # Flatten batch
        tile_flat = tile_h.reshape(B * T, H)
        piece_flat = piece_h.reshape(B * P, H)

        # 1. tile -> piece
        if t2p.size(1) > 0:
            msgs = self.tile_to_piece_msg(tile_flat[t2p[0]])
            agg = _mean_agg(msgs, t2p[1], B * P, H, dev)
        else:
            agg = torch.zeros(B * P, H, device=dev)
        piece_flat = F.relu(self.piece_update(
            torch.cat([piece_flat, agg], dim=1)))

        # 2. piece -> tile
        if p2t.size(1) > 0:
            msgs = self.piece_to_tile_msg(piece_flat[p2t[0]])
            agg = _mean_agg(msgs, p2t[1], B * T, H, dev)
        else:
            agg = torch.zeros(B * T, H, device=dev)
        tile_flat = F.relu(self.tile_update_pieces(
            torch.cat([tile_flat, agg], dim=1)))

        # 3. tile -> tile
        tile_h = tile_flat.reshape(B, T, H)
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

        # 4. tile -> global
        global_agg = self.tile_to_global_msg(tile_h).mean(dim=1, keepdim=True)
        global_h = F.relu(self.global_update(
            torch.cat([global_h, global_agg], dim=2)))

        # 5. global -> tile
        global_msg = self.global_to_tile_msg(global_h).expand(B, T, H)
        tile_h = F.relu(self.tile_update_global(
            torch.cat([tile_h, global_msg], dim=2)))

        piece_h = piece_flat.reshape(B, P, H)
        return tile_h, piece_h, global_h


# -------------------------
# FULL NETWORK
# -------------------------

class BoardGNN(nn.Module):
    """
    Heterogeneous GNN with global node.
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

        self.tile_embed   = nn.Linear(tile_feat_dim, H)
        self.piece_embed  = nn.Linear(piece_feat_dim, H)
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
        B = batch['B']
        T = batch['T']
        P = batch['P']
        dev = batch['tile_feats'].device

        tile_h = F.relu(self.tile_embed(batch['tile_feats']))
        piece_h = F.relu(self.piece_embed(batch['piece_feats']))
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

        value = self.readout(combined).squeeze(1)
        return value

    def forward(self, encoded):
        """Value only."""
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