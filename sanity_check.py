"""
sanity_check.py — comprehensive tests for network.py encoder and GNN.

Run from the same directory as game.py, network.py, tile_neighbors.json:
    python3 sanity_check.py

Tests are grouped into sections. Each test prints PASS or FAIL with details.
A summary at the end shows total pass/fail counts.
"""

import sys
import traceback
import random
import torch
from game import Board, Piece, NUM_PIECES
from network import (
    BoardEncoder, BoardGNN,
    build_tile_index, encode_tile_features, encode_piece_features,
    encode_piece_tile_edges, encode_global_features,
    STATUS_UNENTERED, STATUS_ON_HOME, STATUS_ON_BOARD,
    STATUS_CAN_BE_SAVED, STATUS_SAVED,
    NUM_STATUSES, TILE_FEAT_DIM, PIECE_FEAT_DIM, GLOBAL_FEAT_DIM,
    TOTAL_PIECES, HIDDEN_DIM,
    _piece_status,
)

# -------------------------
# TEST HARNESS
# -------------------------

PASS = 0
FAIL = 0

def check(name, condition, detail=''):
    global PASS, FAIL
    if condition:
        print(f"  PASS  {name}")
        PASS += 1
    else:
        print(f"  FAIL  {name}")
        if detail:
            print(f"        {detail}")
        FAIL += 1

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def run(name, fn):
    """Run a test function, catching exceptions as failures."""
    global PASS, FAIL
    try:
        fn()
    except Exception as e:
        print(f"  FAIL  {name} — EXCEPTION: {e}")
        traceback.print_exc()
        FAIL += 1


# -------------------------
# HELPERS
# -------------------------

def fresh_board():
    """Return a new Board in starting state."""
    return Board()

def play_n_moves(board, n, seed=42):
    """Play n random moves to get a mid-game board state."""
    random.seed(seed)
    moves_made = 0
    consecutive_passes = 0
    while moves_made < n:
        winner, _ = board.check_game_over()
        if winner:
            break
        moves = board.get_valid_moves()
        if not moves:
            break
        move = random.choice(list(moves))
        if move == (0, 0, 0):
            consecutive_passes += 1
            if consecutive_passes > 10:
                break
        else:
            consecutive_passes = 0
        board.apply_move(move)
        moves_made += 1
    return board

def decode_piece_status(feat_row):
    """
    Given a piece feature row tensor, return the status index (0-4)
    from the one-hot encoding at positions [2..6].
    Returns -1 if no status bit is set (bug).
    """
    status_bits = feat_row[2:2+NUM_STATUSES]
    nonzero = status_bits.nonzero(as_tuple=True)[0]
    if len(nonzero) == 1:
        return nonzero[0].item()
    return -1


# -------------------------
# SECTION 1: TILE GRAPH STRUCTURE
# -------------------------

section("1. Tile graph structure")

encoder = BoardEncoder()

def test_tile_count():
    check("Total tiles == 70", encoder.num_tiles == 70,
          f"got {encoder.num_tiles}")

def test_no_nogo_tiles():
    import json
    with open('tile_neighbors.json') as f:
        raw = json.load(f)
    nogo_count = sum(1 for v in raw.values() if v['type'] == 'nogo')
    check("No nogo tiles in tile_index",
          nogo_count == 0,
          f"found {nogo_count} nogo tiles in json (expected 0, they should never exist)")

def test_home_tile_in_index():
    check("Home tile (0,0) in tile_index",
          (0, 0) in encoder.tile_index)

def test_save_tiles_in_index():
    save_tiles = [(r, s) for (r, s), info in encoder.tile_info.items()
                  if info['type'] == 'save']
    check("6 save tiles present", len(save_tiles) == 6,
          f"found {len(save_tiles)}")

def test_save_tile_numbers():
    save_numbers = sorted([encoder.tile_info[(r, s)].get('number')
                           for (r, s), info in encoder.tile_info.items()
                           if info['type'] == 'save'])
    check("Save tile numbers are 1-6", save_numbers == [1,2,3,4,5,6],
          f"got {save_numbers}")

def test_edge_index_shape():
    ei = encoder.tile_edge_index
    check("Edge index has 2 rows", ei.shape[0] == 2,
          f"shape={ei.shape}")

def test_edge_index_valid_range():
    ei = encoder.tile_edge_index
    check("All edge indices in valid range",
          ei.min().item() >= 0 and ei.max().item() < encoder.num_tiles,
          f"min={ei.min()}, max={ei.max()}, num_tiles={encoder.num_tiles}")

def test_edges_symmetric():
    """Every edge (i,j) should have a reverse edge (j,i)."""
    ei = encoder.tile_edge_index
    edge_set = set(zip(ei[0].tolist(), ei[1].tolist()))
    all_symmetric = all((j, i) in edge_set for (i, j) in edge_set)
    check("All tile edges are symmetric (undirected)", all_symmetric)

def test_home_tile_has_neighbors():
    home_idx = encoder.tile_index[(0, 0)]
    ei = encoder.tile_edge_index
    home_edges = (ei[0] == home_idx).sum().item()
    check("Home tile has neighbors", home_edges > 0,
          f"home tile has {home_edges} outgoing edges")

def test_home_not_traversable():
    home_idx = encoder.tile_index[(0, 0)]
    feats = encoder._tile_feats
    check("Home tile is_traversable_as_waypoint == 0",
          feats[home_idx, 3].item() == 0.0)

def test_field_tiles_traversable():
    feats = encoder._tile_feats
    for (r, s), info in encoder.tile_info.items():
        if info['type'] == 'field':
            idx = encoder.tile_index[(r, s)]
            if feats[idx, 3].item() != 1.0:
                check("All field tiles traversable", False,
                      f"tile ({r},{s}) has traversable=0")
                return
    check("All field tiles traversable", True)

run("tile_count", test_tile_count)
run("no_nogo_tiles", test_no_nogo_tiles)
run("home_tile_in_index", test_home_tile_in_index)
run("save_tiles_in_index", test_save_tiles_in_index)
run("save_tile_numbers", test_save_tile_numbers)
run("edge_index_shape", test_edge_index_shape)
run("edge_index_valid_range", test_edge_index_valid_range)
run("edges_symmetric", test_edges_symmetric)
run("home_tile_has_neighbors", test_home_tile_has_neighbors)
run("home_not_traversable", test_home_not_traversable)
run("field_tiles_traversable", test_field_tiles_traversable)


# -------------------------
# SECTION 2: TILE FEATURE ENCODING
# -------------------------

section("2. Tile feature encoding")

def test_tile_feat_shape():
    feats = encoder._tile_feats
    check("Tile feats shape [num_tiles, TILE_FEAT_DIM]",
          feats.shape == (encoder.num_tiles, TILE_FEAT_DIM),
          f"got {feats.shape}")

def test_tile_type_onehot():
    """Each tile should have exactly one of is_home/is_field/is_save set."""
    feats = encoder._tile_feats
    type_bits = feats[:, 0:3]
    row_sums = type_bits.sum(dim=1)
    check("Each tile has exactly one type bit set",
          (row_sums == 1.0).all().item(),
          f"row sums: {row_sums.tolist()}")

def test_home_tile_features():
    idx = encoder.tile_index[(0, 0)]
    feats = encoder._tile_feats[idx]
    check("Home tile: is_home=1", feats[0].item() == 1.0)
    check("Home tile: is_field=0", feats[1].item() == 0.0)
    check("Home tile: is_save=0", feats[2].item() == 0.0)
    check("Home tile: not_traversable=0", feats[3].item() == 0.0)
    check("Home tile: ring=0", feats[4].item() == 0.0)
    check("Home tile: save_number=0", feats[6].item() == 0.0)

def test_save_tile_features():
    for (r, s), info in encoder.tile_info.items():
        if info['type'] == 'save':
            idx = encoder.tile_index[(r, s)]
            feats = encoder._tile_feats[idx]
            num = info['number']
            check(f"Save tile ({r},{s}) num={num}: is_save=1", feats[2].item() == 1.0)
            check(f"Save tile ({r},{s}) num={num}: save_number correct",
                  abs(feats[6].item() - num/6.0) < 1e-5,
                  f"expected {num/6.0}, got {feats[6].item()}")
            check(f"Save tile ({r},{s}) num={num}: traversable=1", feats[3].item() == 1.0)

def test_tile_feats_normalised():
    feats = encoder._tile_feats
    check("All tile features in [0,1]",
          feats.min().item() >= 0.0 and feats.max().item() <= 1.0,
          f"min={feats.min()}, max={feats.max()}")

run("tile_feat_shape", test_tile_feat_shape)
run("tile_type_onehot", test_tile_type_onehot)
run("home_tile_features", test_home_tile_features)
run("save_tile_features", test_save_tile_features)
run("tile_feats_normalised", test_tile_feats_normalised)


# -------------------------
# SECTION 3: PIECE STATUS ENCODING
# -------------------------

section("3. Piece status encoding (fresh board)")

board = fresh_board()
piece_feats, all_pieces = encode_piece_features(board, encoder.tile_index, 'white')

def test_piece_feat_shape():
    check("Piece feats shape [24, PIECE_FEAT_DIM]",
          piece_feats.shape == (TOTAL_PIECES, PIECE_FEAT_DIM),
          f"got {piece_feats.shape}")

def test_piece_ordering():
    """First 12 pieces should be current player's, next 12 opponent's."""
    current_ids = [p.player for p in all_pieces[:12]]
    opponent_ids = [p.player for p in all_pieces[12:]]
    check("First 12 pieces are current player's",
          all(p == 'white' for p in current_ids),
          f"got {current_ids}")
    check("Last 12 pieces are opponent's",
          all(p == 'black' for p in opponent_ids),
          f"got {opponent_ids}")

def test_player_id_feature():
    """Current player pieces → player_id=0, opponent → player_id=1."""
    cur_ids = piece_feats[:12, 0].tolist()
    opp_ids = piece_feats[12:, 0].tolist()
    check("Current player pieces have player_id=0",
          all(x == 0.0 for x in cur_ids), f"got {cur_ids}")
    check("Opponent pieces have player_id=1",
          all(x == 1.0 for x in opp_ids), f"got {opp_ids}")

def test_piece_numbers_normalised():
    nums = piece_feats[:, 1].tolist()
    check("All piece numbers in (0,1]",
          all(0.0 < n <= 1.0 for n in nums), f"got {nums}")

def test_fresh_board_all_unentered():
    """On a fresh board all pieces should be STATUS_UNENTERED."""
    for i in range(TOTAL_PIECES):
        status = decode_piece_status(piece_feats[i])
        if status != STATUS_UNENTERED:
            check("Fresh board: all pieces unentered", False,
                  f"piece {i} ({all_pieces[i]}) has status {status}")
            return
    check("Fresh board: all pieces unentered", True)

def test_status_onehot():
    """Each piece should have exactly one status bit set."""
    status_bits = piece_feats[:, 2:2+NUM_STATUSES]
    row_sums = status_bits.sum(dim=1)
    check("Each piece has exactly one status bit",
          (row_sums == 1.0).all().item(),
          f"row sums: {row_sums.tolist()}")

def test_rack_position_fresh():
    """On fresh board, rack positions should be set for all pieces."""
    rack_pos = piece_feats[:, 7].tolist()
    # All pieces are unentered, so rack_pos should be non-trivially set
    check("Fresh board: rack positions set (not all zero)",
          any(x > 0.0 for x in rack_pos),
          f"rack positions: {rack_pos}")

def test_piece_feats_normalised():
    check("All piece features in [0,1]",
          piece_feats.min().item() >= 0.0 and piece_feats.max().item() <= 1.0,
          f"min={piece_feats.min()}, max={piece_feats.max()}")

run("piece_feat_shape", test_piece_feat_shape)
run("piece_ordering", test_piece_ordering)
run("player_id_feature", test_player_id_feature)
run("piece_numbers_normalised", test_piece_numbers_normalised)
run("fresh_board_all_unentered", test_fresh_board_all_unentered)
run("status_onehot", test_status_onehot)
run("rack_position_fresh", test_rack_position_fresh)
run("piece_feats_normalised", test_piece_feats_normalised)


# -------------------------
# SECTION 4: MID-GAME PIECE STATUS
# -------------------------

section("4. Piece status encoding (mid-game board)")

mid_board = play_n_moves(fresh_board(), 40, seed=1)
mid_feats, mid_pieces = encode_piece_features(mid_board, encoder.tile_index, mid_board.current_player)

def test_mid_some_on_board():
    statuses = [decode_piece_status(mid_feats[i]) for i in range(TOTAL_PIECES)]
    on_board_count = statuses.count(STATUS_ON_BOARD)
    on_home_count  = statuses.count(STATUS_ON_HOME)
    total_on_board = on_board_count + on_home_count
    check("Mid-game: some pieces on board",
          total_on_board > 0, f"on_board={on_board_count}, on_home={on_home_count}")

def test_mid_status_onehot():
    status_bits = mid_feats[:, 2:2+NUM_STATUSES]
    row_sums = status_bits.sum(dim=1)
    check("Mid-game: each piece has exactly one status bit",
          (row_sums == 1.0).all().item(),
          f"row sums: {row_sums.tolist()}")

def test_mid_on_home_correct():
    """Pieces on home tile should have STATUS_ON_HOME."""
    home_tile = mid_board.home_tile
    for piece in mid_board.pieces:
        if piece.tile == home_tile:
            idx = mid_pieces.index(piece)
            status = decode_piece_status(mid_feats[idx])
            if status != STATUS_ON_HOME:
                check("Pieces on home tile have STATUS_ON_HOME", False,
                      f"piece {piece} has status {status}")
                return
    check("Pieces on home tile have STATUS_ON_HOME", True)

def test_mid_can_be_saved_correct():
    """Pieces that can_be_saved() should have STATUS_CAN_BE_SAVED."""
    for piece in mid_board.pieces:
        if piece.tile and piece.tile.type == 'save' and piece.can_be_saved():
            if piece not in mid_pieces:
                continue
            idx = mid_pieces.index(piece)
            status = decode_piece_status(mid_feats[idx])
            if status != STATUS_CAN_BE_SAVED:
                check("can_be_saved pieces have STATUS_CAN_BE_SAVED", False,
                      f"piece {piece} on {piece.tile} has status {status}")
                return
    check("can_be_saved pieces have STATUS_CAN_BE_SAVED", True)

def test_mid_saved_correct():
    """Pieces in save rack should have STATUS_SAVED."""
    for rack in [mid_board.white_saved, mid_board.black_saved]:
        for piece in rack:
            if piece not in mid_pieces:
                continue
            idx = mid_pieces.index(piece)
            status = decode_piece_status(mid_feats[idx])
            if status != STATUS_SAVED:
                check("Saved pieces have STATUS_SAVED", False,
                      f"piece {piece} has status {status}")
                return
    check("Saved pieces have STATUS_SAVED", True)

def test_mid_wrong_save_tile_is_on_board():
    """
    A numbered piece on the wrong save tile should be STATUS_ON_BOARD,
    not STATUS_CAN_BE_SAVED.
    """
    found_case = False
    for piece in mid_board.pieces:
        if piece.tile and piece.tile.type == 'save':
            if not piece.can_be_saved():
                found_case = True
                if piece not in mid_pieces:
                    continue
                idx = mid_pieces.index(piece)
                status = decode_piece_status(mid_feats[idx])
                if status != STATUS_ON_BOARD:
                    check("Numbered piece on wrong save tile → STATUS_ON_BOARD", False,
                          f"piece {piece} on {piece.tile} has status {status}")
                    return
    if found_case:
        check("Numbered piece on wrong save tile → STATUS_ON_BOARD", True)
    else:
        print("  SKIP  wrong_save_tile_is_on_board (no such piece in this mid-game state)")

def test_mid_unentered_rack_positions():
    """Unentered pieces should have non-zero rack positions if not first."""
    cur_player = mid_board.current_player
    cur_unentered = (mid_board.white_unentered if cur_player == 'white'
                     else mid_board.black_unentered)
    for rack_pos, piece in enumerate(cur_unentered):
        if piece not in mid_pieces:
            continue
        idx = mid_pieces.index(piece)
        encoded_rack_pos = mid_feats[idx, 7].item()
        expected = rack_pos / 11.0
        if abs(encoded_rack_pos - expected) > 1e-5:
            check("Unentered rack positions correct", False,
                  f"piece {piece} rack_pos={rack_pos}, encoded={encoded_rack_pos}, expected={expected}")
            return
    check("Unentered rack positions correct", True)

run("mid_some_on_board", test_mid_some_on_board)
run("mid_status_onehot", test_mid_status_onehot)
run("mid_on_home_correct", test_mid_on_home_correct)
run("mid_can_be_saved_correct", test_mid_can_be_saved_correct)
run("mid_saved_correct", test_mid_saved_correct)
run("mid_wrong_save_tile_is_on_board", test_mid_wrong_save_tile_is_on_board)
run("mid_unentered_rack_positions", test_mid_unentered_rack_positions)


# -------------------------
# SECTION 5: CURRENT PLAYER FLIP
# -------------------------

section("5. Current player perspective flip")

flip_board = play_n_moves(fresh_board(), 30, seed=7)

def test_flip_player_ids_swap():
    """
    Encoding from white's perspective vs black's perspective should
    flip which pieces have player_id=0 vs player_id=1.
    """
    feats_w, pieces_w = encode_piece_features(flip_board, encoder.tile_index, 'white')
    feats_b, pieces_b = encode_piece_features(flip_board, encoder.tile_index, 'black')

    # White's encoding: first 12 are white pieces (player_id=0)
    white_ids_from_white = feats_w[:12, 0].tolist()
    # Black's encoding: first 12 are black pieces (player_id=0)
    black_ids_from_black = feats_b[:12, 0].tolist()

    check("From white's perspective: white pieces have player_id=0",
          all(x == 0.0 for x in white_ids_from_white))
    check("From black's perspective: black pieces have player_id=0",
          all(x == 0.0 for x in black_ids_from_black))

def test_flip_piece_counts_match():
    """Both perspectives should encode the same 24 pieces."""
    feats_w, pieces_w = encode_piece_features(flip_board, encoder.tile_index, 'white')
    feats_b, pieces_b = encode_piece_features(flip_board, encoder.tile_index, 'black')
    check("Both perspectives encode 24 pieces",
          len(pieces_w) == TOTAL_PIECES and len(pieces_b) == TOTAL_PIECES)

def test_flip_statuses_consistent():
    """
    A piece's status should be the same regardless of which player's
    perspective we encode from.
    """
    feats_w, pieces_w = encode_piece_features(flip_board, encoder.tile_index, 'white')
    feats_b, pieces_b = encode_piece_features(flip_board, encoder.tile_index, 'black')

    piece_status_w = {p: decode_piece_status(feats_w[i]) for i, p in enumerate(pieces_w)}
    piece_status_b = {p: decode_piece_status(feats_b[i]) for i, p in enumerate(pieces_b)}

    for piece in flip_board.pieces:
        sw = piece_status_w.get(piece)
        sb = piece_status_b.get(piece)
        if sw != sb:
            check("Piece statuses consistent across perspectives", False,
                  f"piece {piece}: white_perspective={sw}, black_perspective={sb}")
            return
    check("Piece statuses consistent across perspectives", True)

def test_flip_numbers_consistent():
    """Piece numbers should be the same regardless of perspective."""
    feats_w, pieces_w = encode_piece_features(flip_board, encoder.tile_index, 'white')
    feats_b, pieces_b = encode_piece_features(flip_board, encoder.tile_index, 'black')

    nums_w = {p: feats_w[i, 1].item() for i, p in enumerate(pieces_w)}
    nums_b = {p: feats_b[i, 1].item() for i, p in enumerate(pieces_b)}

    for piece in flip_board.pieces:
        if abs(nums_w.get(piece, -1) - nums_b.get(piece, -1)) > 1e-5:
            check("Piece numbers consistent across perspectives", False,
                  f"piece {piece}: white={nums_w.get(piece)}, black={nums_b.get(piece)}")
            return
    check("Piece numbers consistent across perspectives", True)

run("flip_player_ids_swap", test_flip_player_ids_swap)
run("flip_piece_counts_match", test_flip_piece_counts_match)
run("flip_statuses_consistent", test_flip_statuses_consistent)
run("flip_numbers_consistent", test_flip_numbers_consistent)


# -------------------------
# SECTION 6: PIECE-TILE EDGES
# -------------------------

section("6. Piece-tile edges")

edge_board = play_n_moves(fresh_board(), 20, seed=3)
edge_feats, edge_pieces = encode_piece_features(edge_board, encoder.tile_index, edge_board.current_player)
p2t, t2p = encode_piece_tile_edges(edge_pieces, encoder.tile_index)

def test_edge_shapes():
    check("piece_to_tile has 2 rows", p2t.shape[0] == 2,
          f"shape={p2t.shape}")
    check("tile_to_piece has 2 rows", t2p.shape[0] == 2,
          f"shape={t2p.shape}")
    check("piece_to_tile and tile_to_piece same width",
          p2t.shape[1] == t2p.shape[1],
          f"p2t={p2t.shape}, t2p={t2p.shape}")

def test_edge_are_reverse():
    """piece_to_tile[0] == tile_to_piece[1] and vice versa."""
    if p2t.shape[1] == 0:
        print("  SKIP  edges_are_reverse (no on-board pieces)")
        return
    check("piece_to_tile row0 == tile_to_piece row1",
          (p2t[0] == t2p[1]).all().item())
    check("piece_to_tile row1 == tile_to_piece row0",
          (p2t[1] == t2p[0]).all().item())

def test_edge_piece_indices_valid():
    if p2t.shape[1] == 0:
        print("  SKIP  edge_piece_indices_valid (no on-board pieces)")
        return
    check("Piece indices in edge_index are valid",
          p2t[0].min().item() >= 0 and p2t[0].max().item() < TOTAL_PIECES,
          f"min={p2t[0].min()}, max={p2t[0].max()}")

def test_edge_tile_indices_valid():
    if p2t.shape[1] == 0:
        print("  SKIP  edge_tile_indices_valid (no on-board pieces)")
        return
    check("Tile indices in edge_index are valid",
          p2t[1].min().item() >= 0 and p2t[1].max().item() < encoder.num_tiles,
          f"min={p2t[1].min()}, max={p2t[1].max()}")

def test_edge_only_on_board_pieces():
    """Only pieces with a tile (not unentered/saved) should have edges."""
    on_board_pieces = [p for p in edge_pieces if p.tile is not None]
    check("Number of edges matches on-board pieces",
          p2t.shape[1] == len(on_board_pieces),
          f"edges={p2t.shape[1]}, on_board={len(on_board_pieces)}")

def test_edge_correct_tile():
    """Each piece→tile edge should point to the piece's actual tile."""
    for edge_idx in range(p2t.shape[1]):
        piece_idx = p2t[0, edge_idx].item()
        tile_idx  = p2t[1, edge_idx].item()
        piece = edge_pieces[piece_idx]
        coords = (piece.tile.ring, piece.tile.pos)
        expected_tile_idx = encoder.tile_index.get(coords, -1)
        if tile_idx != expected_tile_idx:
            check("Each edge points to piece's actual tile", False,
                  f"piece {piece} on {coords}: edge→{tile_idx}, expected {expected_tile_idx}")
            return
    check("Each edge points to piece's actual tile", True)

def test_unentered_pieces_no_edges():
    """Unentered pieces should have no tile edges."""
    unentered_indices = {i for i, p in enumerate(edge_pieces)
                         if p.rack in (edge_board.white_unentered, edge_board.black_unentered)}
    edge_piece_indices = set(p2t[0].tolist()) if p2t.shape[1] > 0 else set()
    overlap = unentered_indices & edge_piece_indices
    check("Unentered pieces have no tile edges",
          len(overlap) == 0,
          f"unentered pieces with edges: {overlap}")

def test_saved_pieces_no_edges():
    """Saved pieces should have no tile edges."""
    saved_indices = {i for i, p in enumerate(edge_pieces)
                     if p.rack in (edge_board.white_saved, edge_board.black_saved)}
    edge_piece_indices = set(p2t[0].tolist()) if p2t.shape[1] > 0 else set()
    overlap = saved_indices & edge_piece_indices
    check("Saved pieces have no tile edges",
          len(overlap) == 0,
          f"saved pieces with edges: {overlap}")

run("edge_shapes", test_edge_shapes)
run("edge_are_reverse", test_edge_are_reverse)
run("edge_piece_indices_valid", test_edge_piece_indices_valid)
run("edge_tile_indices_valid", test_edge_tile_indices_valid)
run("edge_only_on_board_pieces", test_edge_only_on_board_pieces)
run("edge_correct_tile", test_edge_correct_tile)
run("unentered_pieces_no_edges", test_unentered_pieces_no_edges)
run("saved_pieces_no_edges", test_saved_pieces_no_edges)


# -------------------------
# SECTION 7: GLOBAL FEATURES (DICE)
# -------------------------

section("7. Global features (dice)")

def test_global_shape():
    board = fresh_board()
    g = encode_global_features(board)
    check("Global feats shape [4]",
          g.shape == (GLOBAL_FEAT_DIM,), f"got {g.shape}")

def test_global_values_normalised():
    board = fresh_board()
    g = encode_global_features(board)
    check("Global feats in [0,1]",
          g.min().item() >= 0.0 and g.max().item() <= 1.0,
          f"min={g.min()}, max={g.max()}")

def test_global_die_values():
    board = fresh_board()
    board.dice[0].number = 3
    board.dice[1].number = 6
    board.dice[0].used = False
    board.dice[1].used = True
    g = encode_global_features(board)
    check("Die 1 value encoded correctly", abs(g[0].item() - 3/6.0) < 1e-5,
          f"expected {3/6.0}, got {g[0].item()}")
    check("Die 2 value encoded correctly", abs(g[1].item() - 6/6.0) < 1e-5,
          f"expected {6/6.0}, got {g[1].item()}")
    check("Die 1 used=0 encoded correctly", g[2].item() == 0.0,
          f"got {g[2].item()}")
    check("Die 2 used=1 encoded correctly", g[3].item() == 1.0,
          f"got {g[3].item()}")

run("global_shape", test_global_shape)
run("global_values_normalised", test_global_values_normalised)
run("global_die_values", test_global_die_values)


# -------------------------
# SECTION 8: FULL ENCODE (round-trip)
# -------------------------

section("8. Full encode — round-trip reconstruction")

def test_full_encode_shape():
    board = play_n_moves(fresh_board(), 25, seed=99)
    encoded = encoder.encode(board, board.current_player)
    check("tile_feats shape correct",
          encoded['tile_feats'].shape == (encoder.num_tiles, TILE_FEAT_DIM))
    check("piece_feats shape correct",
          encoded['piece_feats'].shape == (TOTAL_PIECES, PIECE_FEAT_DIM))
    check("global_feats shape correct",
          encoded['global_feats'].shape == (GLOBAL_FEAT_DIM,))
    check("tile_edge_index shape correct",
          encoded['tile_edge_index'].shape[0] == 2)
    check("piece_to_tile shape correct",
          encoded['piece_to_tile'].shape[0] == 2)
    check("tile_to_piece shape correct",
          encoded['tile_to_piece'].shape[0] == 2)

def test_roundtrip_piece_count():
    """
    Reconstruct piece positions from encoding and verify they match the board.
    For each on-board piece, the edge index should correctly identify its tile.
    """
    board = play_n_moves(fresh_board(), 35, seed=55)
    encoded = encoder.encode(board, board.current_player)
    _, all_pieces = encode_piece_features(board, encoder.tile_index, board.current_player)

    p2t_enc = encoded['piece_to_tile']
    on_board_actual = [p for p in all_pieces if p.tile is not None]
    check("Round-trip: on-board piece count matches edge count",
          p2t_enc.shape[1] == len(on_board_actual),
          f"edges={p2t_enc.shape[1]}, actual={len(on_board_actual)}")

def test_roundtrip_tile_identity():
    """
    For each edge in piece_to_tile, verify the encoded tile index
    corresponds to the piece's actual tile coordinates.
    """
    board = play_n_moves(fresh_board(), 35, seed=55)
    encoded = encoder.encode(board, board.current_player)
    _, all_pieces = encode_piece_features(board, encoder.tile_index, board.current_player)

    # Build reverse index: tile_index → (ring, sector)
    reverse_tile_index = {v: k for k, v in encoder.tile_index.items()}

    p2t_enc = encoded['piece_to_tile']
    mismatches = []
    for edge_i in range(p2t_enc.shape[1]):
        piece_idx = p2t_enc[0, edge_i].item()
        tile_idx  = p2t_enc[1, edge_i].item()
        piece = all_pieces[piece_idx]
        actual_coords = (piece.tile.ring, piece.tile.pos)
        encoded_coords = reverse_tile_index.get(tile_idx)
        if actual_coords != encoded_coords:
            mismatches.append(
                f"piece {piece}: actual={actual_coords}, encoded={encoded_coords}")

    check("Round-trip: all tile coordinates match",
          len(mismatches) == 0,
          "\n        ".join(mismatches[:3]))

def test_roundtrip_status_counts():
    """
    Count pieces in each status from the encoding and verify they match
    the board's actual rack/tile counts.
    """
    board = play_n_moves(fresh_board(), 40, seed=77)
    _, all_pieces = encode_piece_features(board, encoder.tile_index, board.current_player)
    feats, _ = encode_piece_features(board, encoder.tile_index, board.current_player)

    statuses = [decode_piece_status(feats[i]) for i in range(TOTAL_PIECES)]

    actual_unentered = len(board.white_unentered) + len(board.black_unentered)
    actual_saved     = len(board.white_saved) + len(board.black_saved)
    actual_on_home   = sum(1 for p in board.pieces
                           if p.tile and p.tile.type == 'home')
    actual_can_save  = sum(1 for p in board.pieces
                           if p.tile and p.can_be_saved()
                           and p.rack not in (board.white_saved, board.black_saved))
    actual_on_board  = sum(1 for p in board.pieces
                           if p.tile and p.tile.type not in ['home']
                           and not p.can_be_saved()
                           and p.rack not in (board.white_saved, board.black_saved))

    enc_unentered = statuses.count(STATUS_UNENTERED)
    enc_saved     = statuses.count(STATUS_SAVED)
    enc_on_home   = statuses.count(STATUS_ON_HOME)
    enc_can_save  = statuses.count(STATUS_CAN_BE_SAVED)
    enc_on_board  = statuses.count(STATUS_ON_BOARD)

    check("Round-trip: unentered count matches",
          enc_unentered == actual_unentered,
          f"encoded={enc_unentered}, actual={actual_unentered}")
    check("Round-trip: saved count matches",
          enc_saved == actual_saved,
          f"encoded={enc_saved}, actual={actual_saved}")
    check("Round-trip: on_home count matches",
          enc_on_home == actual_on_home,
          f"encoded={enc_on_home}, actual={actual_on_home}")
    check("Round-trip: can_be_saved count matches",
          enc_can_save == actual_can_save,
          f"encoded={enc_can_save}, actual={actual_can_save}")
    check("Round-trip: on_board count matches",
          enc_on_board == actual_on_board,
          f"encoded={enc_on_board}, actual={actual_on_board}")

def test_multiple_pieces_same_tile():
    """
    If multiple pieces are on the same tile (e.g. home tile with captured pieces),
    all of them should have edges.
    """
    board = play_n_moves(fresh_board(), 50, seed=13)
    _, all_pieces = encode_piece_features(board, encoder.tile_index, board.current_player)
    p2t_enc, _ = encode_piece_tile_edges(all_pieces, encoder.tile_index)

    # Find tiles with multiple pieces
    from collections import Counter
    tile_counts = Counter(p.tile for p in board.pieces if p.tile is not None)
    multi_tiles = {t: c for t, c in tile_counts.items() if c > 1}

    if not multi_tiles:
        print("  SKIP  multiple_pieces_same_tile (no stacked pieces in this game state)")
        return

    # For each multi-piece tile, count edges pointing to it
    for tile, count in multi_tiles.items():
        coords = (tile.ring, tile.pos)
        if coords not in encoder.tile_index:
            continue
        tile_idx = encoder.tile_index[coords]
        edge_count = (p2t_enc[1] == tile_idx).sum().item()
        if edge_count != count:
            check("Multiple pieces on same tile all get edges", False,
                  f"tile {coords} has {count} pieces but {edge_count} edges")
            return
    check("Multiple pieces on same tile all get edges", True)

run("full_encode_shape", test_full_encode_shape)
run("roundtrip_piece_count", test_roundtrip_piece_count)
run("roundtrip_tile_identity", test_roundtrip_tile_identity)
run("roundtrip_status_counts", test_roundtrip_status_counts)
run("multiple_pieces_same_tile", test_multiple_pieces_same_tile)


# -------------------------
# SECTION 9: GNN FORWARD PASS
# -------------------------

section("9. GNN forward pass")

model = BoardGNN()

def test_forward_scalar():
    board = play_n_moves(fresh_board(), 20, seed=5)
    encoded = encoder.encode(board, board.current_player)
    score = model(encoded)
    check("Forward pass returns scalar tensor",
          score.shape == torch.Size([]),
          f"got shape {score.shape}")

def test_forward_is_float():
    board = play_n_moves(fresh_board(), 20, seed=5)
    encoded = encoder.encode(board, board.current_player)
    score = model(encoded)
    check("Forward pass returns float",
          score.dtype in (torch.float32, torch.float64),
          f"got dtype {score.dtype}")

def test_forward_fresh_board():
    board = fresh_board()
    encoded = encoder.encode(board, 'white')
    score = model(encoded)
    check("Forward pass works on fresh board", True)
    print(f"        (fresh board score = {score.item():.4f})")

def test_forward_different_players():
    """
    The network should be sensitive to piece features.
    Verify by perturbing piece features and checking output changes.
    """
    board = play_n_moves(fresh_board(), 30, seed=8)
    encoded = encoder.encode(board, board.current_player)
    model.eval()
    with torch.no_grad():
        score_original = model(encoded).item()
        # Perturb piece features significantly
        perturbed = {k: v.clone() for k, v in encoded.items()}
        perturbed['piece_feats'] = torch.ones_like(perturbed['piece_feats'])
        score_perturbed = model(perturbed).item()
    check("Network is sensitive to piece features",
          abs(score_original - score_perturbed) > 1e-4,
          f"original={score_original:.6f}, perturbed={score_perturbed:.6f}")

def test_gradients_flow():
    board = play_n_moves(fresh_board(), 25, seed=2)
    encoded = encoder.encode(board, board.current_player)
    model.zero_grad()
    score = model(encoded)
    score.backward()
    # Check at least some gradients are non-zero
    has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0
                   for p in model.parameters())
    check("Gradients flow through network", has_grad)

def test_deterministic():
    """Same input should give same output (no dropout etc)."""
    board = play_n_moves(fresh_board(), 20, seed=6)
    encoded = encoder.encode(board, board.current_player)
    model.eval()
    with torch.no_grad():
        score1 = model(encoded).item()
        score2 = model(encoded).item()
    check("Forward pass is deterministic",
          abs(score1 - score2) < 1e-6,
          f"score1={score1}, score2={score2}")

def test_forward_multiple_positions():
    """Forward pass should work on various game states without error."""
    seeds = [10, 20, 30, 40, 50]
    moves = [5, 15, 30, 50, 80]
    errors = []
    for seed, n in zip(seeds, moves):
        try:
            board = play_n_moves(fresh_board(), n, seed=seed)
            encoded = encoder.encode(board, board.current_player)
            with torch.no_grad():
                score = model(encoded)
            assert score.shape == torch.Size([])
        except Exception as e:
            errors.append(f"seed={seed}, moves={n}: {e}")
    check("Forward pass works on 5 different game states",
          len(errors) == 0,
          "\n        ".join(errors))

def test_parameter_count():
    total = sum(p.numel() for p in model.parameters())
    print(f"        Total parameters: {total:,}")
    check("Parameter count is reasonable (100 - 1,000,000)",
          100 < total < 1_000_000,
          f"got {total}")

def test_forward_batch():
    """Batched forward should return [N] tensor matching individual scores."""
    boards = [play_n_moves(fresh_board(), n, seed=s)
              for n, s in [(10,1),(20,2),(30,3)]]
    encoded_list = [encoder.encode(b, b.current_player) for b in boards]
    model.eval()
    with torch.no_grad():
        # Batched
        scores_batch = model(encoded_list)
        # Individual
        scores_single = torch.stack([model(e) for e in encoded_list])
    check("Batched forward returns [N] tensor",
          scores_batch.shape == torch.Size([3]),
          f"got shape {scores_batch.shape}")
    check("Batched scores match individual scores",
          torch.allclose(scores_batch, scores_single, atol=1e-5),
          f"batch={scores_batch.tolist()}, single={scores_single.tolist()}")

run("forward_scalar", test_forward_scalar)
run("forward_is_float", test_forward_is_float)
run("forward_fresh_board", test_forward_fresh_board)
run("forward_different_players", test_forward_different_players)
run("gradients_flow", test_gradients_flow)
run("deterministic", test_deterministic)
run("forward_multiple_positions", test_forward_multiple_positions)
run("parameter_count", test_parameter_count)
run("forward_batch", test_forward_batch)


# -------------------------
# SUMMARY
# -------------------------

total = PASS + FAIL
print(f"\n{'='*60}")
print(f"  RESULTS: {PASS}/{total} passed, {FAIL} failed")
print(f"{'='*60}")

if FAIL > 0:
    sys.exit(1)
