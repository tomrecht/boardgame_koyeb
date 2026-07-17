"""
network.py — GNN position encoder and value network with true GPU batching.

Architecture:
  - Tile nodes (70): board topology, static structure, dynamic features
  - Piece nodes (24): one per piece, always present
  - Global node (1): virtual node connected to all tiles
  - 6 message passing layers with residual connections
  - Hidden dim: 64

Encoding: always from current player's perspective.
  Current player pieces -> player_id=0, opponent -> player_id=1.

Features:
  Tile:   neighbor_count, has_permanent_block, my_piece_count, opp_piece_count
  Piece:  is_numbered, is_blot, is_completely_blocked, distance_category,
          goal_distances_raw (6 values, normalized), goal_distances_binned (6 values)
  Global: my_game_stage, opponent_game_stage, my_numbered_saved_count,
          opp_numbered_saved_count, my_highest_occupied_goal,
          opp_highest_occupied_goal, my_saveable_count
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

# -------------------------
# CONSTANTS
# -------------------------

NUM_PIECES      = 12
TOTAL_PIECES    = 24
NUM_GOALS       = 6
HIDDEN_DIM      = 64
NUM_MP_LAYERS   = 6
TILE_FEAT_DIM   = 12   # was 10: +my_piece_count, +opp_piece_count
PIECE_FEAT_DIM  = 24   # was 18: +6 goal distance bins (raw already there)
GLOBAL_FEAT_DIM = 11   # was 7: +opp_numbered_saved, +my_highest_goal, +opp_highest_goal, +my_saveable_count
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
    [0]  is_home
    [1]  is_field
    [2]  is_save
    [3]  traversable (not home)
    [4]  ring/7
    [5]  sector/45
    [6]  save_number/6
    [7]  neighbor_count/6
    [8]  has_permanent_block       (updated dynamically)
    [9]  reserved
    [10] my_piece_count            (updated dynamically)
    [11] opp_piece_count           (updated dynamically)
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
        f[idx, 7] = len(info.get('neighbors', [])) / 6.0
        # [8], [9], [10], [11] updated dynamically per position

    return f.to(DEVICE)


def update_tile_features_dynamic(base_np, board, current_player, tile_index):
    """
    Update dynamic tile features:
      [8]  has_permanent_block (2+ unnumbered friendly pieces)
      [9]  reserved
      [10] my_piece_count (current player)
      [11] opp_piece_count (opponent)

    Only iterates tiles that actually have pieces — the base feature array
    has zeros in slots 8/10/11, so empty tiles need no update. Most tiles are
    empty most of the time, so this is far cheaper than iterating all 70.

    Works on a numpy copy of the base features: per-element scalar updates in
    numpy avoid a torch dispatch per increment (the previous per-index tensor
    ops dominated this function's cost), then a single tensor conversion at
    the end. torch.from_numpy shares the fresh copy's memory, which is safe
    because the array is never touched after return.
    """
    arr = base_np.copy()

    # Group pieces by their tile (only occupied tiles)
    for piece in board.pieces:
        tile = piece.tile
        if tile is None:
            continue
        coords = (tile.ring, tile.pos)
        idx = tile_index.get(coords)
        if idx is None:
            continue
        # Increment the appropriate piece count
        if piece.player == current_player:
            arr[idx, 10] += 1.0
        else:
            arr[idx, 11] += 1.0

    # has_permanent_block: recompute only for occupied field tiles
    occupied_field_tiles = set()
    for piece in board.pieces:
        tile = piece.tile
        if tile is not None and tile.type == 'field':
            occupied_field_tiles.add(tile)

    for tile in occupied_field_tiles:
        coords = (tile.ring, tile.pos)
        idx = tile_index.get(coords)
        if idx is None:
            continue
        my_pieces = [p for p in tile.pieces if p.player == current_player]
        if len(my_pieces) >= 2:
            unnumbered = sum(1 for p in my_pieces if p.number > 6)
            if unnumbered >= 2:
                arr[idx, 8] = 1.0

    t = torch.from_numpy(arr)
    return t if DEVICE.type == 'cpu' else t.to(DEVICE)


# -------------------------
# PIECE FEATURES  (per position)
# -------------------------

STATUS_UNENTERED    = 0
STATUS_ON_HOME      = 1
STATUS_ON_BOARD     = 2
STATUS_CAN_BE_SAVED = 3
STATUS_SAVED        = 4
NUM_STATUSES        = 5

INF_DIST = float('inf')
# Normalize goal distances: cap at 14 (max meaningful distance on board)
MAX_DIST = 14.0


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


def _dist_bin(d):
    """
    Bin a goal distance into a normalized category value.
    0.0 = already there/saved
    0.25 = single die (d <= 6)
    0.5  = both dice (6 < d <= 12)
    0.75 = 3+ moves (d > 12)
    1.0  = blocked/unreachable
    """
    if d == 0:
        return 0.0
    if d == INF_DIST:
        return 1.0
    if d <= 6:
        return 0.25
    if d <= 12:
        return 0.5
    return 0.75


def _encode_goal_distances(piece, board):
    """
    Return (raw_list, binned_list), each of length 6 (one per goal 1-6).
    raw:    normalized distance in [0,1], capped at MAX_DIST; 1.0 if blocked
    binned: 0.0/0.25/0.5/0.75/1.0 per _dist_bin
    """
    goal_dists = board.all_goal_distances(piece)
    raw, binned = [], []
    for goal_num in range(1, NUM_GOALS + 1):
        d = goal_dists.get(goal_num, INF_DIST)
        raw.append(1.0 if d == INF_DIST else min(d, MAX_DIST) / MAX_DIST)
        binned.append(_dist_bin(d))
    return raw, binned


def encode_piece_features(board, tile_index, current_player, row_cache=None):
    """
    [TOTAL_PIECES, PIECE_FEAT_DIM]
    [0]     player_id (0=current, 1=opponent)
    [1]     piece_number/12
    [2]     is_numbered
    [3-7]   status onehot (5 bits)
    [8]     rack_position/11
    [9]     is_blot
    [10]    is_completely_blocked
    [11]    distance_category
    [12-17] goal_distances_raw    (6 values, normalized to [0,1])
    [18-23] goal_distances_binned (6 values, 0/0.25/0.5/0.75/1.0)
    """
    opponent = 'black' if current_player == 'white' else 'white'
    cur_pieces = sorted([p for p in board.pieces if p.player == current_player],
                        key=lambda p: p.number)
    opp_pieces = sorted([p for p in board.pieces if p.player == opponent],
                        key=lambda p: p.number)

    # Defensive: enforce at most NUM_PIECES per player. A known bug can
    # occasionally produce 13 pieces for a player (last-piece renumber to 13
    # combined with a serialization edge case). Truncate so the encoded
    # tensor is always [TOTAL_PIECES, PIECE_FEAT_DIM].
    if len(cur_pieces) > NUM_PIECES:
        cur_pieces = cur_pieces[:NUM_PIECES]
    if len(opp_pieces) > NUM_PIECES:
        opp_pieces = opp_pieces[:NUM_PIECES]

    all_pieces = cur_pieces + opp_pieces

    cur_un = board.white_unentered if current_player == 'white' else board.black_unentered
    opp_un = board.black_unentered if current_player == 'white' else board.white_unentered
    rack_pos = {p: i for i, p in enumerate(cur_un)}
    rack_pos.update({p: i for i, p in enumerate(opp_un)})

    # A piece's feature row is a pure function of:
    #   (its player, current-player bit, number, placement, rack slot,
    #    blot bit, the player's blocked-tile frozenset)
    # -- placement + number determine status/can_be_saved; the blocked set +
    # placement determine every distance-derived field. Rows are therefore
    # cacheable under that key, and in the 2-ply candidate loop ~22 of 24
    # pieces are unchanged between candidates, so hits dominate.
    blocked_keys = {}
    rows = []
    for piece in all_pieces:
        tile = piece.tile
        if tile is not None:
            place = tile.index
            is_blot = 1.0 if (tile.type == 'field' and len(tile.pieces) == 1) else 0.0
        else:
            place = 'u' if piece.rack in (board.white_unentered, board.black_unentered) else 's'
            is_blot = 0.0
        rp = rack_pos.get(piece, 0)
        bk = blocked_keys.get(piece.player)
        if bk is None:
            bk = board._get_blocked_key(piece.player)
            blocked_keys[piece.player] = bk

        key = (piece.player, piece.player == current_player, piece.number,
               place, rp, is_blot, bk)
        row = row_cache.get(key) if row_cache is not None else None
        if row is None:
            status = _piece_status(piece, board)
            onehot = [0.0] * NUM_STATUSES
            onehot[status] = 1.0

            is_numbered = 1.0 if piece.number <= 6 else 0.0

            dist = board.shortest_route_to_goal(piece)
            is_completely_blocked = 1.0 if dist == INF_DIST else 0.0

            if dist == INF_DIST:
                distance_category = 0.0
            elif dist <= 6:
                distance_category = 1.0
            elif dist <= 12:
                distance_category = 0.5
            else:
                distance_category = 0.25

            goal_dist_raw, goal_dist_binned = _encode_goal_distances(piece, board)

            row = (
                0.0 if piece.player == current_player else 1.0,
                piece.number / 12.0,
                is_numbered,
                *onehot,
                rp / 11.0 if status == STATUS_UNENTERED else 0.0,
                is_blot,
                is_completely_blocked,
                distance_category,
                *goal_dist_raw,
                *goal_dist_binned,
            )
            if row_cache is not None:
                row_cache[key] = row
        rows.append(row)

    arr = np.asarray(rows, dtype=np.float32)
    t = torch.from_numpy(arr)
    return (t if DEVICE.type == 'cpu' else t.to(DEVICE)), all_pieces


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
        p2t_np = np.array([psrc, tdst], dtype=np.int64)
        t2p_np = np.array([tdst, psrc], dtype=np.int64)
        p2t = torch.from_numpy(p2t_np)
        t2p = torch.from_numpy(t2p_np)
        if DEVICE.type != 'cpu':
            p2t = p2t.to(DEVICE)
            t2p = t2p.to(DEVICE)
    else:
        p2t = torch.zeros(2, 0, dtype=torch.long, device=DEVICE)
        t2p = torch.zeros(2, 0, dtype=torch.long, device=DEVICE)
    return p2t, t2p


# -------------------------
# GLOBAL FEATURES
# -------------------------

def _highest_occupied_goal(board, player):
    """Highest goal number occupied by a saveable piece of this player. -1 if none."""
    return max(
        (t.number for t in board.tiles
         if t.type == 'save'
         and any(p.player == player and p.can_be_saved() for p in t.pieces)),
        default=-1
    )


def encode_global_features(board):
    """
    [GLOBAL_FEAT_DIM]
    [0]  die1/6
    [1]  die2/6
    [2]  die1_used
    [3]  die2_used
    [4]  my_game_stage (0/0.5/1)
    [5]  opponent_game_stage
    [6]  my_numbered_saved_count / 6
    [7]  opp_numbered_saved_count / 6
    [8]  my_highest_occupied_goal / 6  (-1 -> 0)
    [9]  opp_highest_occupied_goal / 6 (-1 -> 0)
    [10] my_saveable_count / 12
    """
    current_player = board.current_player
    opponent = 'black' if current_player == 'white' else 'white'

    d1, d2 = board.dice[0], board.dice[1]

    stage_map = {'opening': 0.0, 'midgame': 0.5, 'endgame': 1.0}
    my_stage  = stage_map.get(board.game_stages[current_player], 0.5)
    opp_stage = stage_map.get(board.game_stages[opponent], 0.5)

    my_saved  = board.white_saved  if current_player == 'white' else board.black_saved
    opp_saved = board.black_saved  if current_player == 'white' else board.white_saved

    my_numbered_saved  = sum(1 for p in board.pieces
                             if p.player == current_player
                             and p.number <= 6 and p.rack is my_saved)
    opp_numbered_saved = sum(1 for p in board.pieces
                             if p.player == opponent
                             and p.number <= 6 and p.rack is opp_saved)

    my_highest  = _highest_occupied_goal(board, current_player)
    opp_highest = _highest_occupied_goal(board, opponent)

    my_saveable = sum(1 for p in board.pieces
                      if p.player == current_player and p.can_be_saved())

    return torch.tensor([
        d1.number / 6.0,
        d2.number / 6.0,
        float(d1.used),
        float(d2.used),
        my_stage,
        opp_stage,
        my_numbered_saved  / 6.0,
        opp_numbered_saved / 6.0,
        max(my_highest,  0) / 6.0,
        max(opp_highest, 0) / 6.0,
        my_saveable / 12.0,
    ], dtype=torch.float32, device=DEVICE)


# -------------------------
# BOARD ENCODER
# -------------------------

_tile_index = None
_tile_info  = None


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
        _tile_info  = self.tile_info
        self.num_tiles = len(self.tile_index)
        self._base_tile_feats = encode_tile_features(self.tile_index, self.tile_info)
        # numpy copy of the base tile features (identical float32 bits) for
        # the fast dynamic-update path; row cache for per-piece feature rows.
        self._base_tile_feats_np = np.ascontiguousarray(
            self._base_tile_feats.cpu().numpy())
        self._row_cache = {}

    def encode(self, board, current_player):
        tile_feats = update_tile_features_dynamic(
            self._base_tile_feats_np, board, current_player, self.tile_index)

        # Bound the row cache (keys embed blocked-set frozensets, so entries
        # never go stale -- this is purely a memory cap; one turn's candidate
        # loop needs only ~1-2K entries).
        if len(self._row_cache) > 100_000:
            self._row_cache.clear()
        piece_feats, all_pieces = encode_piece_features(
            board, self.tile_index, current_player, self._row_cache)

        p2t, t2p = encode_piece_tile_edges(all_pieces, self.tile_index)

        return {
            'tile_feats':      tile_feats,
            'piece_feats':     piece_feats,
            'tile_edge_index': self.tile_edge_index,
            'piece_to_tile':   p2t,
            'tile_to_piece':   t2p,
            'global_feats':    encode_global_features(board),
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