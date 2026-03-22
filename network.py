"""
network.py — GNN position encoder and value network.

Architecture:
  - BoardEncoder: converts a Board object into tensors
  - NackGNN: heterogeneous GNN, outputs expected score scalar
    (positive = good for current player, negative = bad)

Node types:
  - Tile nodes (70): fixed graph, features change each position
  - Piece nodes (24): one per piece, always present

Encoding is always from the current player's perspective:
  - Current player's pieces → player_id = 0
  - Opponent's pieces → player_id = 1

Global features: [die1_val, die2_val, die1_used, die2_used]  (4 values)
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# CONSTANTS
# -------------------------

NUM_PIECES      = 12   # per player
TOTAL_PIECES    = 24
HIDDEN_DIM      = 64
NUM_MP_LAYERS   = 3    # message passing rounds

# Feature dimensions
TILE_FEAT_DIM   = 8    # see encode_tile_features()
PIECE_FEAT_DIM  = 8    # see encode_piece_features()
GLOBAL_FEAT_DIM = 4    # [die1_val, die2_val, die1_used, die2_used]


# -------------------------
# FIXED GRAPH STRUCTURE
# -------------------------

def build_tile_index(tile_neighbors_path='tile_neighbors.json'):
    """
    Reads tile_neighbors.json and returns:
      tile_index: dict mapping (ring, sector) → integer node index
      tile_data:  list of tile dicts in node index order
      edge_index: LongTensor [2, num_edges] for tile-tile edges (both directions)
    
    Nogo tiles are excluded entirely.
    Called once at startup; result is reused for every forward pass.
    """
    with open(tile_neighbors_path) as f:
        raw = json.load(f)

    # Build ordered list of (ring, sector) keys, excluding nogo
    tile_keys = []
    tile_info = {}
    for key, val in raw.items():
        if val['type'] == 'nogo':
            continue
        ring, sector = _parse_key(key)
        tile_keys.append((ring, sector))
        tile_info[(ring, sector)] = val

    # Assign stable integer indices
    tile_index = {coords: idx for idx, coords in enumerate(tile_keys)}

    # Build edge list (undirected → add both directions)
    src, dst = [], []
    for (ring, sector), val in tile_info.items():
        i = tile_index[(ring, sector)]
        for nb in val['neighbors']:
            nb_coords = (nb['ring'], nb['sector'])
            if nb_coords in tile_index:   # skip nogo neighbors
                j = tile_index[nb_coords]
                src.append(i); dst.append(j)
                # undirected: also add reverse
                src.append(j); dst.append(i)

    # Deduplicate edges (undirected edges are added twice above, once per direction)
    edge_pairs = list(set(zip(src, dst)))
    src = [e[0] for e in edge_pairs]
    dst = [e[1] for e in edge_pairs]

    edge_index = torch.tensor([src, dst], dtype=torch.long)

    return tile_index, tile_info, edge_index


def _parse_key(key):
    """'ring3_sector7' → (3, 7)"""
    parts = key.replace('ring', '').replace('sector', '').split('_')
    return int(parts[0]), int(parts[1])


# -------------------------
# TILE FEATURE ENCODING
# -------------------------

def encode_tile_features(tile_index, tile_info):
    """
    Build a fixed [num_tiles, TILE_FEAT_DIM] tensor of tile features.
    These never change between positions (topology is static).
    
    Features per tile (8 total):
      [0]   is_home        (bool)
      [1]   is_field       (bool)
      [2]   is_save        (bool)
      [3]   is_traversable_as_waypoint  (False only for home tile)
      [4]   ring           (normalised 0→1, max ring = 7)
      [5]   sector         (normalised 0→1, max sector = 45)
      [6]   save_number    (normalised 1→6, 0 if not a save tile)
      [7]   (reserved / padding for future use)
    """
    num_tiles = len(tile_index)
    feats = torch.zeros(num_tiles, TILE_FEAT_DIM)

    for (ring, sector), idx in tile_index.items():
        info = tile_info[(ring, sector)]
        t = info['type']

        feats[idx, 0] = 1.0 if t == 'home'  else 0.0
        feats[idx, 1] = 1.0 if t == 'field' else 0.0
        feats[idx, 2] = 1.0 if t == 'save'  else 0.0
        feats[idx, 3] = 0.0 if t == 'home'  else 1.0   # not traversable as waypoint
        feats[idx, 4] = ring   / 7.0
        feats[idx, 5] = sector / 45.0
        feats[idx, 6] = info.get('number', 0) / 6.0    # 0 if no number
        feats[idx, 7] = 0.0   # reserved

    return feats


# -------------------------
# PIECE FEATURE ENCODING  (position-dependent)
# -------------------------

# Status indices (one-hot, 5 values)
STATUS_UNENTERED    = 0
STATUS_ON_HOME      = 1
STATUS_ON_BOARD     = 2
STATUS_CAN_BE_SAVED = 3
STATUS_SAVED        = 4
NUM_STATUSES        = 5

def _piece_status(piece, board):
    """
    Returns the STATUS_* constant for a piece.
    Assumes piece belongs to a player (not None).
    """
    # In save rack
    if piece.rack in (board.white_saved, board.black_saved):
        return STATUS_SAVED

    # In unentered rack
    if piece.rack in (board.white_unentered, board.black_unentered):
        return STATUS_UNENTERED

    # On a tile
    if piece.tile is not None:
        if piece.tile.type == 'home':
            return STATUS_ON_HOME
        if piece.can_be_saved():
            return STATUS_CAN_BE_SAVED
        return STATUS_ON_BOARD

    # Fallback (shouldn't happen in a valid game state)
    return STATUS_ON_BOARD


def encode_piece_features(board, tile_index, current_player):
    """
    Build a [TOTAL_PIECES, PIECE_FEAT_DIM] tensor of piece features,
    always from the current player's perspective (current player = 0).

    Features per piece (8 total):
      [0]   player_id      (0 = current player, 1 = opponent)
      [1]   piece_number   (normalised 1→12)
      [2..6] status one-hot (unentered, on_home, on_board, can_be_saved, saved)
      [7]   rack_position  (normalised 0→11, only for unentered pieces, else 0)

    Pieces are ordered: current player's pieces first (0..11),
    then opponent's pieces (12..23), each sorted by piece number.
    This gives a stable, consistent ordering every call.
    """
    opponent = 'black' if current_player == 'white' else 'white'

    # Gather all pieces, split by player, sort by number for stable ordering
    current_pieces  = sorted([p for p in board.pieces if p.player == current_player],
                              key=lambda p: p.number)
    opponent_pieces = sorted([p for p in board.pieces if p.player == opponent],
                              key=lambda p: p.number)

    all_pieces = current_pieces + opponent_pieces   # length = TOTAL_PIECES

    # Unentered rack positions for rack_position feature
    cur_unentered  = board.white_unentered if current_player == 'white' else board.black_unentered
    opp_unentered  = board.black_unentered if current_player == 'white' else board.white_unentered
    cur_rack_pos   = {p: i for i, p in enumerate(cur_unentered)}
    opp_rack_pos   = {p: i for i, p in enumerate(opp_unentered)}
    rack_pos_map   = {**cur_rack_pos, **opp_rack_pos}

    feats = torch.zeros(TOTAL_PIECES, PIECE_FEAT_DIM)

    for piece_idx, piece in enumerate(all_pieces):
        player_id  = 0 if piece.player == current_player else 1
        status     = _piece_status(piece, board)
        rack_pos   = rack_pos_map.get(piece, 0)

        feats[piece_idx, 0] = float(player_id)
        feats[piece_idx, 1] = piece.number / 12.0

        # One-hot status [2..6]
        feats[piece_idx, 2 + status] = 1.0

        # Rack position (only meaningful for unentered pieces)
        feats[piece_idx, 7] = rack_pos / 11.0 if status == STATUS_UNENTERED else 0.0

    return feats, all_pieces


# -------------------------
# PIECE → TILE EDGES  (position-dependent)
# -------------------------

def encode_piece_tile_edges(all_pieces, tile_index, board):
    """
    Returns edges connecting pieces to their current tile.
    Only on-board pieces (on_home, on_board, can_be_saved) get an edge.
    
    Returns:
      piece_to_tile: LongTensor [2, num_on_board_pieces]
                     row 0 = piece node indices
                     row 1 = tile node indices
      tile_to_piece: LongTensor [2, num_on_board_pieces]  (reverse)
    """
    piece_src, tile_dst = [], []

    for piece_idx, piece in enumerate(all_pieces):
        if piece.tile is not None:
            coords = (piece.tile.ring, piece.tile.pos)
            if coords in tile_index:
                tile_idx = tile_index[coords]
                piece_src.append(piece_idx)
                tile_dst.append(tile_idx)

    if piece_src:
        piece_to_tile = torch.tensor([piece_src, tile_dst], dtype=torch.long)
        tile_to_piece = torch.tensor([tile_dst, piece_src], dtype=torch.long)
    else:
        piece_to_tile = torch.zeros(2, 0, dtype=torch.long)
        tile_to_piece = torch.zeros(2, 0, dtype=torch.long)

    return piece_to_tile, tile_to_piece


# -------------------------
# GLOBAL FEATURES  (dice)
# -------------------------

def encode_global_features(board):
    """
    Returns a [GLOBAL_FEAT_DIM] tensor:
      [die1_val, die2_val, die1_used, die2_used]
    Values normalised: die values /6, used as 0/1 float.
    """
    d1, d2 = board.dice[0], board.dice[1]
    return torch.tensor([
        d1.number / 6.0,
        d2.number / 6.0,
        float(d1.used),
        float(d2.used),
    ])


# -------------------------
# BOARD ENCODER  (top-level)
# -------------------------

class BoardEncoder:
    """
    Converts a Board into tensors for the GNN.
    
    Usage:
        encoder = BoardEncoder()   # once at startup
        tensors = encoder.encode(board, current_player)
    
    Returns a dict with keys:
        tile_feats      [num_tiles, TILE_FEAT_DIM]        — static tile features
        piece_feats     [TOTAL_PIECES, PIECE_FEAT_DIM]    — dynamic piece features
        tile_edge_index [2, num_tile_edges]               — tile-tile adjacency
        piece_to_tile   [2, num_on_board]                 — piece→tile edges
        tile_to_piece   [2, num_on_board]                 — tile→piece edges
        global_feats    [GLOBAL_FEAT_DIM]                 — dice state
    """

    def __init__(self, tile_neighbors_path='tile_neighbors.json'):
        # Build fixed graph structure once
        self.tile_index, self.tile_info, self.tile_edge_index = \
            build_tile_index(tile_neighbors_path)
        self.num_tiles = len(self.tile_index)

        # Static tile features (never change)
        self._static_tile_feats = encode_tile_features(self.tile_index, self.tile_info)

    def encode(self, board, current_player):
        """Encode a board position into tensors."""
        piece_feats, all_pieces = encode_piece_features(
            board, self.tile_index, current_player)

        piece_to_tile, tile_to_piece = encode_piece_tile_edges(
            all_pieces, self.tile_index, board)

        global_feats = encode_global_features(board)

        return {
            'tile_feats':       self._static_tile_feats,
            'piece_feats':      piece_feats,
            'tile_edge_index':  self.tile_edge_index,
            'piece_to_tile':    piece_to_tile,
            'tile_to_piece':    tile_to_piece,
            'global_feats':     global_feats,
        }


# -------------------------
# GNN  (manual message passing)
# -------------------------

class MessagePassingLayer(nn.Module):
    """
    One round of heterogeneous message passing:
      1. tile → piece: each piece aggregates messages from its tile
      2. piece → tile: each tile aggregates messages from its pieces
      3. tile → tile:  spatial propagation through board topology

    All aggregations use mean pooling.
    Each direction has its own linear transform.
    """

    def __init__(self, dim):
        super().__init__()
        # tile → piece
        self.tile_to_piece_msg  = nn.Linear(dim, dim)
        self.piece_update       = nn.Linear(dim * 2, dim)

        # piece → tile
        self.piece_to_tile_msg  = nn.Linear(dim, dim)
        self.tile_update_pieces = nn.Linear(dim * 2, dim)

        # tile → tile
        self.tile_to_tile_msg   = nn.Linear(dim, dim)
        self.tile_update_tiles  = nn.Linear(dim * 2, dim)

    def forward(self, tile_h, piece_h, tile_edge_index, piece_to_tile, tile_to_piece):
        """
        tile_h:         [num_tiles,  dim]
        piece_h:        [num_pieces, dim]
        tile_edge_index:[2, num_tile_edges]
        piece_to_tile:  [2, num_on_board]   row0=piece, row1=tile
        tile_to_piece:  [2, num_on_board]   row0=tile,  row1=piece
        """
        num_tiles  = tile_h.size(0)
        num_pieces = piece_h.size(0)
        dim        = tile_h.size(1)

        # --- 1. tile → piece ---
        if tile_to_piece.size(1) > 0:
            tile_src   = tile_to_piece[0]   # tile indices
            piece_dst  = tile_to_piece[1]   # piece indices
            msgs       = self.tile_to_piece_msg(tile_h[tile_src])   # [E, dim]
            agg        = _mean_aggregate(msgs, piece_dst, num_pieces, dim)
        else:
            agg = torch.zeros(num_pieces, dim)

        piece_h = F.relu(self.piece_update(torch.cat([piece_h, agg], dim=1)))

        # --- 2. piece → tile ---
        if piece_to_tile.size(1) > 0:
            piece_src  = piece_to_tile[0]   # piece indices
            tile_dst   = piece_to_tile[1]   # tile indices
            msgs       = self.piece_to_tile_msg(piece_h[piece_src])
            agg        = _mean_aggregate(msgs, tile_dst, num_tiles, dim)
        else:
            agg = torch.zeros(num_tiles, dim)

        tile_h = F.relu(self.tile_update_pieces(torch.cat([tile_h, agg], dim=1)))

        # --- 3. tile → tile ---
        if tile_edge_index.size(1) > 0:
            t_src = tile_edge_index[0]
            t_dst = tile_edge_index[1]
            msgs  = self.tile_to_tile_msg(tile_h[t_src])
            agg   = _mean_aggregate(msgs, t_dst, num_tiles, dim)
        else:
            agg = torch.zeros(num_tiles, dim)

        tile_h = F.relu(self.tile_update_tiles(torch.cat([tile_h, agg], dim=1)))

        return tile_h, piece_h


def _mean_aggregate(messages, dst_indices, num_dst, dim):
    """
    Mean-pool messages into destination nodes.
    messages:    [num_edges, dim]
    dst_indices: [num_edges]  — which destination node each message goes to
    Returns:     [num_dst, dim]
    """
    agg   = torch.zeros(num_dst, dim)
    count = torch.zeros(num_dst, 1)
    agg.scatter_add_(0, dst_indices.unsqueeze(1).expand_as(messages), messages)
    count.scatter_add_(0, dst_indices.unsqueeze(1),
                       torch.ones(dst_indices.size(0), 1))
    count = count.clamp(min=1)   # avoid divide by zero
    return agg / count


# -------------------------
# FULL NETWORK
# -------------------------

class BoardGNN(nn.Module):
    """
    Heterogeneous GNN for board evaluation.
    
    Input:  encoded board tensors from BoardEncoder.encode()
    Output: scalar expected score from current player's perspective.
            Positive = good for current player, negative = bad.
    
    Architecture:
      - Linear embeddings for tile and piece features → HIDDEN_DIM
      - NUM_MP_LAYERS rounds of heterogeneous message passing
      - Mean pool tile and piece embeddings
      - Concatenate with global (dice) features
      - Two dense layers → scalar output
    """

    def __init__(self,
                 hidden_dim=HIDDEN_DIM,
                 num_mp_layers=NUM_MP_LAYERS,
                 tile_feat_dim=TILE_FEAT_DIM,
                 piece_feat_dim=PIECE_FEAT_DIM,
                 global_feat_dim=GLOBAL_FEAT_DIM):
        super().__init__()

        # Input projections
        self.tile_embed  = nn.Linear(tile_feat_dim,  hidden_dim)
        self.piece_embed = nn.Linear(piece_feat_dim, hidden_dim)

        # Message passing layers
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim)
            for _ in range(num_mp_layers)
        ])

        # Readout MLP
        # Input: mean(tile_h) + mean(piece_h) + global_feats
        readout_input_dim = hidden_dim + hidden_dim + global_feat_dim
        self.readout = nn.Sequential(
            nn.Linear(readout_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, encoded):
        """
        encoded: dict from BoardEncoder.encode()
        Returns: scalar tensor (expected score)
        """
        tile_feats      = encoded['tile_feats']
        piece_feats     = encoded['piece_feats']
        tile_edge_index = encoded['tile_edge_index']
        piece_to_tile   = encoded['piece_to_tile']
        tile_to_piece   = encoded['tile_to_piece']
        global_feats    = encoded['global_feats']

        # Initial embeddings
        tile_h  = F.relu(self.tile_embed(tile_feats))    # [num_tiles,  hidden_dim]
        piece_h = F.relu(self.piece_embed(piece_feats))  # [num_pieces, hidden_dim]

        # Message passing
        for mp_layer in self.mp_layers:
            tile_h, piece_h = mp_layer(
                tile_h, piece_h,
                tile_edge_index,
                piece_to_tile,
                tile_to_piece,
            )

        # Readout: mean pool both node types, concatenate with globals
        tile_pooled  = tile_h.mean(dim=0)    # [hidden_dim]
        piece_pooled = piece_h.mean(dim=0)   # [hidden_dim]

        combined = torch.cat([tile_pooled, piece_pooled, global_feats])  # [hidden_dim*2 + global_feat_dim]
        score = self.readout(combined)       # [1]

        return score.squeeze()              # scalar


# -------------------------
# CONVENIENCE: save / load
# -------------------------

def save_model(model, path='gnn_weights.pt'):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(path='gnn_weights.pt', **kwargs):
    model = BoardGNN(**kwargs)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    print(f"Model loaded from {path}")
    return model
