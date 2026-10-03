"""
encoder.py — Board -> feature arrays. NumPy only, no torch.

Split out of network.py so the deployed web app can run inference through
onnxruntime without pulling in torch (which dominates the image size). The
training path is unchanged: network.py re-exports everything here, and its
BoardEncoder hands back torch tensors as before.

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

ROW_CACHE_MAX = int(os.environ.get('ROW_CACHE_MAX', '100000'))

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
      tile_edge_index: int64 array [2, E]  (both directions)
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
    tile_edge_index = np.array([src, dst], dtype=np.int64)
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
    f = np.zeros((n, TILE_FEAT_DIM), dtype=np.float32)

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

    return f


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
    numpy avoid a per-increment dispatch (the previous per-index tensor
    ops dominated this function's cost), then a single tensor conversion at
    the end; callers get that fresh copy, which is safe
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

    return arr


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
    return arr, all_pieces


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
        p2t = p2t_np
        t2p = t2p_np
    else:
        p2t = np.zeros((2, 0), dtype=np.int64)
        t2p = np.zeros((2, 0), dtype=np.int64)
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
    # Compute stages FRESH -- do not read board.game_stages. That dict is
    # mutated as a side effect of get_valid_moves (current player's slot
    # only, at entry, never restored), so during candidate enumeration it
    # holds whatever the previous probe left behind: candidate #N gets
    # encoded with candidate #N-1's stage, the opponent's slot can stay
    # stale indefinitely, and successive calls from the same position can
    # score identical candidates 20-80 scaled points apart (measured).
    # Training encodes always saw fresh stages (update_state recomputes
    # both slots), so fresh computation here also closes a long-standing
    # train/selection semantics gap.
    my_stage  = stage_map.get(board.get_game_stage(current_player), 0.5)
    opp_stage = stage_map.get(board.get_game_stage(opponent), 0.5)

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

    return np.array([
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
    ], dtype=np.float32)


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
        self._base_tile_feats_np = np.ascontiguousarray(self._base_tile_feats)
        self._row_cache = {}

    def encode(self, board, current_player):
        tile_feats = update_tile_features_dynamic(
            self._base_tile_feats_np, board, current_player, self.tile_index)

        # Bound the row cache (keys embed blocked-set frozensets, so entries
        # never go stale -- this is purely a memory cap; one turn's candidate
        # loop needs only ~1-2K entries). ROW_CACHE_MAX trades a little speed for
        # resident memory: the deployed app runs a much smaller cache than
        # training does, because it shares a small instance with the net.
        if len(self._row_cache) > ROW_CACHE_MAX:
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

