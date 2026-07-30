# python app.py, then localhost:10000

import os
import sys
# The app serves one position per request -- latency-bound inference where
# GPU/MPS buys nothing, and some machines' MPS stacks hard-abort on the
# GNN's batched scatter_add (MPSNDArrayScatter rank assertion on the iMac).
# Default the app to CPU BEFORE network.py is imported (it reads this env
# var at import time); an explicitly exported BOARDGAME_DEVICE still wins.
os.environ.setdefault('BOARDGAME_DEVICE', 'cpu')

from flask import Flask, request, jsonify, session
from flask_cors import CORS
import json
import uuid
import time
import logging
import threading
from game import Board, NO_SAVE_TURNS_FOR_DRAW
from agent import Agent, get_weights
from agent_gnn import GNNAgent

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='', static_url_path='')
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
CORS(app, supports_credentials=True)

# -------------------------
# DATA COLLECTION SETUP
# -------------------------

# Human-play recording is off by default: the collected data has served its
# purpose, the agent trains from self-play, and on a deployed instance the disk
# is ephemeral anyway. RECORD_TRAINING=1 turns it back on for a local run.
RECORD_TRAINING = os.environ.get('RECORD_TRAINING', '') == '1'

DATA_DIR = 'training_data'
POSITIONS_FILE = os.path.join(DATA_DIR, 'positions_with_moves.jsonl')
CONTRASTIVE_FILE = os.path.join(DATA_DIR, 'contrastive_pairs.jsonl')
if RECORD_TRAINING:
    os.makedirs(DATA_DIR, exist_ok=True)

active_games = {}  # game_id -> {'positions': [], 'start_time': ...}
MAX_MOVES_IN_MEMORY = 200
SCHEMA_VERSION = 1

# -------------------------
# AGENT INITIALIZATION
# -------------------------

current_weights = get_weights(weights_file='best_weights.json')

# Opponent checkpoint is chosen WITHOUT editing this (tracked) file, so
# swapping opponents never leaves uncommitted changes that block a branch
# switch. Precedence: OPPONENT_MODEL env var > gitignored opponent_model.txt
# > the default below.
def _opponent_model_path():
    """Checkpoint (.pt, needs torch) or exported net (.onnx, needs only
    onnxruntime) to play as, in order of precedence:

        python app.py --model symaug_iter6.pt      # or -m, or --model=...
        python app.py symaug_iter6.pt              # bare filename
        OPPONENT_MODEL=... python app.py           # env (used in deployment)
        opponent_model.txt                         # first non-comment line
        model.onnx                                 # if present
        td_champion_July18_iter10.pt               # last resort

    Deployment defaults to the .onnx, which is what keeps torch out of the
    image."""
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg.startswith('--model='):
            return arg.split('=', 1)[1]
        if arg in ('--model', '-m') and i + 1 < len(argv):
            return argv[i + 1]
    for arg in argv:
        # bare filename: extension-checked so gunicorn's own flags can't match
        if arg.endswith('.pt') or arg.endswith('.onnx'):
            return arg
    env = os.environ.get('OPPONENT_MODEL')
    if env:
        return env.strip()
    try:
        with open('opponent_model.txt') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    return line
    except FileNotFoundError:
        pass
    if os.path.exists('model.onnx'):
        return 'model.onnx'
    return 'td_champion_July18_iter10.pt'

_model_path = _opponent_model_path()
if not os.path.exists(_model_path):
    logger.warning(f'Model file not found: {_model_path} (cwd {os.getcwd()})')
logger.info(f'Loading opponent model: {_model_path}')
#agent = Agent(weights=current_weights, log_to_file=True)
agent = GNNAgent(weights_path=_model_path, use_prefilter=True, prefilter_top_k=40, heuristic_weights=current_weights)
agent.debug_pass_over_save = True 
# Heuristic agent kept alongside the GNN so /evaluate_board can report both
# evals for the same position (the GNN is the player; the heuristic is shown
# for comparison / prefilter context).
heur_agent = Agent(weights=current_weights)

# Reuse a single board to keep caches across moves (optional, can be per‑request)
# One Board per thread, not one per process: update_state rebuilds it from the
# posted state, so it is only ever a scratch buffer -- but two requests sharing
# one buffer would interleave and corrupt each other. (Constructing a Board
# re-reads tile_neighbors.json, so we keep one per thread rather than one per
# request.) Separate gunicorn workers are separate processes and so are
# independent regardless.
_thread_local = threading.local()


def get_board():
    b = getattr(_thread_local, 'board', None)
    if b is None:
        b = Board()
        _thread_local.board = b
    return b

# -------------------------
# DATA COLLECTION HELPERS
# -------------------------

def save_position_to_disk(position_data):
    if not RECORD_TRAINING:      # belt and braces: the endpoints already refuse
        return
    with open(POSITIONS_FILE, 'a') as f:
        f.write(json.dumps(position_data) + '\n')

def save_contrastive_pair(record):
    if not RECORD_TRAINING:
        return
    with open(CONTRASTIVE_FILE, 'a') as f:
        f.write(json.dumps(record) + '\n')

# -------------------------
# CANONICALIZATION
# -------------------------

def _normalize_move(m):
    """Convert a move from JSON list format to a hashable tuple."""
    if m is None or m == [0, 0, 0]:
        return (0, 0, 0)
    piece, dest, roll = m
    piece = tuple(piece) if isinstance(piece, list) else piece
    dest = tuple(dest) if isinstance(dest, list) else dest
    return (piece, dest, roll)

def apply_pair_to_board(board, m1, m2):
    """Apply a move pair and return a frozenset snapshot of resulting piece positions."""
    initial_move_count = len(board.moves)

    def rack_kind(p):
        # racks are plain lists; identify by object identity
        if p.rack is None:
            return None
        if p.rack is board.white_saved or p.rack is board.black_saved:
            return 'saved'
        if p.rack is board.white_unentered or p.rack is board.black_unentered:
            return 'unentered'
        return 'other'

    try:
        if m1 != (0, 0, 0):
            board.apply_move(m1, switch_turn=False)
        if m2 != (0, 0, 0):
            board.apply_move(m2, switch_turn=False)
        result = frozenset(
            (p.player, p.number,
             p.tile.ring if p.tile else None,
             p.tile.pos if p.tile else None,
             rack_kind(p))
            for p in board.pieces
        )
    except Exception as e:
        logger.warning(f"apply_pair_to_board failed: {e} m1={m1} m2={m2}")
        result = None
    finally:
        while len(board.moves) > initial_move_count:
            board.undo_last_move()
    return result

def pairs_effectively_equal(human_pair, agent_pair, board):
    """True iff both move pairs produce identical resulting board states."""
    if human_pair == agent_pair:
        return True
    h_result = apply_pair_to_board(board, human_pair[0], human_pair[1])
    a_result = apply_pair_to_board(board, agent_pair[0], agent_pair[1])
    return h_result is not None and h_result == a_result

def flush_game_to_disk(game_id, winner, margin):
    if game_id not in active_games:
        logger.warning(f"Game {game_id} not found in active games")
        return False
    game_data = active_games[game_id]
    positions = game_data['positions']
    if not positions:
        logger.warning(f"Game {game_id} has no positions recorded")
        return False
    total_moves = len(positions)
    for i, pos in enumerate(positions):
        ply_from_end = total_moves - i
        if winner is None:
            final_score = 0
        else:
            final_score = margin if pos['player'] == winner else -margin
        pos['final_score'] = final_score
        pos['ply_from_end'] = ply_from_end
        pos['timestamp'] = int(time.time())
        pos['schema_version'] = SCHEMA_VERSION
        save_position_to_disk(pos)
    logger.info(f"Saved {len(positions)} positions from game {game_id} (winner: {winner}, margin: {margin})")
    del active_games[game_id]
    return True

def cleanup_stale_games(max_age_seconds=3600):
    current_time = time.time()
    stale_ids = []
    for game_id, game_data in active_games.items():
        last_move = game_data.get('last_move_time', game_data['start_time'])
        if current_time - last_move > max_age_seconds:
            stale_ids.append(game_id)
    for game_id in stale_ids:
        logger.warning(f"Removing stale game {game_id}")
        del active_games[game_id]

# -------------------------
# API ENDPOINTS
# -------------------------

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/select_moves', methods=['POST'])
def select_moves():
    try:
        state = request.json
        logger.debug("select_moves: received state")
        shared_board = get_board()
        shared_board.update_state(state)

        # The no-save draw counter is owned by the frontend and arrives inside
        # the state (noSaveTurns / drawCallable); update_state mirrors it onto
        # the board, so get_valid_moves will offer (1,1,1) when appropriate.
        game_id = session.get('game_id')
        no_save = {
            'no_save_turns': shared_board.no_save_turns,
            'draw_callable': shared_board.draw_callable,
        }

        # Difficulty (1 = full strength / argmax; lower = weaker via top-p
        # sampling). Passed per call -- setting it on the shared agent would let
        # one player's setting leak into another's move.
        try:
            difficulty = float(state.get('difficulty', 1.0))
        except (TypeError, ValueError):
            difficulty = 1.0

        moves = shared_board.get_valid_moves()
        logger.debug(f"select_moves: got {len(moves)} valid moves")
        if moves:
            chosen_moves = agent.select_move_pair(moves, shared_board, shared_board.current_player,
                                                  difficulty=difficulty)
            logger.debug(f"select_moves: selected {chosen_moves}")

            # Record black's position server-side
            if game_id and game_id in active_games:
                game_data = active_games[game_id]
                game_stage = shared_board.game_stages.get('black', 'unknown')
                move_index = len(game_data['positions'])
                position_data = {
                    'schema_version': SCHEMA_VERSION,
                    'game_id': game_id,
                    'player': 'black',
                    'source': 'heuristic',
                    'game_stage': game_stage,
                    'move_index': move_index,
                    'move_pair': chosen_moves,
                    'raw_state': state,
                }
                game_data['positions'].append(position_data)
                game_data['last_move_time'] = time.time()
                logger.debug(f"Recorded black position for game {game_id}, move {move_index}")

            return jsonify({"message": "Success", "move": chosen_moves, "no_save": no_save}), 200
        else:
            return jsonify({"message": "No valid moves available", "no_save": no_save}), 200
    except Exception as e:
        logger.exception("Error in select_moves")
        return jsonify({"message": "Internal server error", "error": str(e)}), 500

@app.route('/start_game', methods=['POST'])
def start_game():
    try:
        game_id = str(uuid.uuid4())
        session['game_id'] = game_id
        active_games[game_id] = {
            'positions': [],
            'start_time': time.time(),
            'last_move_time': time.time()
        }
        logger.info(f"Started new game: {game_id}")
        return jsonify({"game_id": game_id, "message": "Game started"}), 200
    except Exception as e:
        logger.error(f"Error in start_game: {e}")
        return jsonify({"message": "An error occurred"}), 500


@app.route('/update_impasse', methods=['POST'])
def update_impasse():
    """The no-save draw counter is owned by the frontend; this endpoint simply
    reflects what the posted state reports (kept for backward compatibility
    with the frontend's turn-end ping)."""
    try:
        data = request.json
        state = data.get('state') or {}
        no_save = {
            'no_save_turns': state.get('noSaveTurns', 0),
            'draw_callable': bool(state.get('drawCallable', False)),
            'both_midgame': bool(state.get('bothMidgame', False)),
        }
        return jsonify(no_save), 200
    except Exception as e:
        logger.exception("Error in update_impasse")
        return jsonify({"error": str(e)}), 500
    

@app.route('/record_position', methods=['POST'])
def record_position():
    if not RECORD_TRAINING:
        return jsonify({"message": "recording disabled"}), 200
    try:
        data = request.json
        raw_state = data.get('state')
        player = data.get('player')
        source = data.get('source', 'unknown')
        move_index = data.get('move_index', 0)
        game_stage = data.get('game_stage', 'unknown')
        move_pair = data.get('move_pair')   # store the chosen move pair

        if not raw_state:
            return jsonify({"message": "state required"}), 400
        if not player:
            return jsonify({"message": "player required"}), 400

        game_id = session.get('game_id')
        if not game_id:
            return jsonify({"message": "No active game. Call /start_game first"}), 400

        if game_id not in active_games:
            active_games[game_id] = {
                'positions': [],
                'start_time': time.time(),
                'last_move_time': time.time()
            }

        position_data = {
            'schema_version': SCHEMA_VERSION,
            'game_id': game_id,
            'player': player,
            'source': source,
            'game_stage': game_stage,
            'move_index': move_index,
            'move_pair': move_pair,
            'raw_state': raw_state,
        }
        active_games[game_id]['positions'].append(position_data)
        active_games[game_id]['last_move_time'] = time.time()

        if len(active_games[game_id]['positions']) > MAX_MOVES_IN_MEMORY:
            logger.warning(f"Game {game_id} exceeded {MAX_MOVES_IN_MEMORY} moves, auto-flushing with unknown outcome")
            flush_game_to_disk(game_id, None, 0)

        logger.debug(f"Recorded position for game {game_id}, move {move_index}")
        return jsonify({"message": "Position recorded"}), 200
    except Exception as e:
        logger.exception("Error in record_position")
        return jsonify({"message": "An error occurred"}), 500

@app.route('/record_game_result', methods=['POST'])
def record_game_result():
    try:
        data = request.json
        winner = data.get('winner')
        margin = data.get('score', 0)
        game_id = session.get('game_id')
        if not game_id:
            return jsonify({"message": "No active game"}), 400
        if game_id not in active_games:
            logger.warning(f"Game {game_id} not found in active games")
            return jsonify({"message": "Game not found"}), 404
        success = flush_game_to_disk(game_id, winner, margin)
        session.pop('game_id', None)
        if success:
            return jsonify({"message": "Game result recorded"}), 200
        else:
            return jsonify({"message": "Failed to record game result"}), 500
    except Exception as e:
        logger.error(f"Error in record_game_result: {e}")
        return jsonify({"message": "An error occurred"}), 500

@app.route('/call_draw', methods=['POST'])
def call_draw():
    try:
        game_id = session.get('game_id')
        if not game_id:
            return jsonify({"message": "No active game"}), 400
        if game_id not in active_games:
            logger.warning(f"Game {game_id} not found in active games")
            return jsonify({"message": "Game not found"}), 404
        logger.info(f"Draw called for game {game_id}")
        success = flush_game_to_disk(game_id, None, 0)   # winner=None => draw, margin 0
        session.pop('game_id', None)
        if success:
            return jsonify({"message": "Draw recorded"}), 200
        return jsonify({"message": "Failed to record draw"}), 500
    except Exception as e:
        logger.error(f"Error in call_draw: {e}")
        return jsonify({"message": "An error occurred"}), 500

@app.route('/abort_game', methods=['POST'])
def abort_game():
    try:
        game_id = session.get('game_id')
        if not game_id:
            return jsonify({"message": "No active game"}), 400
        if game_id in active_games:
            logger.info(f"Aborting game {game_id} without saving")
            del active_games[game_id]
        session.pop('game_id', None)
        return jsonify({"message": "Game aborted"}), 200
    except Exception as e:
        logger.error(f"Error in abort_game: {e}")
        return jsonify({"message": "An error occurred"}), 500

@app.route('/debug_piece_blots', methods=['POST'])
def debug_piece_blots():
    try:
        data = request.json
        game_state = data['gameState']
        piece_info = data['piece']
        local_board = Board()
        local_board.update_state(game_state)
        piece = None
        for p in local_board.pieces:
            if p.player == piece_info['player'] and p.number == piece_info['number']:
                piece = p
                break
        if not piece:
            return jsonify({"error": "Piece not found"}), 404
        distance = local_board.shortest_route_to_goal(piece)
        if hasattr(local_board, 'count_enemy_blots_on_shortest_path'):
            blot_count = local_board.count_enemy_blots_on_shortest_path(piece)
        else:
            blot_count = "Method not implemented"
        return jsonify({
            "distance": distance,
            "blot_count": blot_count,
            "can_be_saved": piece.can_be_saved()
        }), 200
    except Exception as e:
        logger.exception("Error in debug_piece_blots")
        return jsonify({"error": str(e)}), 500

@app.route('/evaluate_board', methods=['POST'])
def evaluate_board():
    try:
        state = request.json
        local_board = Board()
        local_board.update_state(state)
        player = local_board.current_player
        gnn_total, gnn_components = agent.evaluate(local_board, player)
        heur_total, heur_components = heur_agent.evaluate(local_board, player)
        gnn_best = agent.best_play_value(local_board, player)
        return jsonify({
            "message": "Success",
            "eval": gnn_total,
            "total_score": gnn_total,
            "gnn_raw": gnn_components.get("gnn_raw"),
            "gnn_score": gnn_total,
            "gnn_player": player,
            "gnn_best_margin": gnn_best,
            "heur_score": heur_total,
            "player": heur_components.get("player"),
            "opponent": heur_components.get("opponent"),
        }), 200
    except Exception as e:
        logger.error(f"Error in evaluate_board: {e}")
        return jsonify({"message": "An error occurred"}), 500

@app.route('/training_data_stats', methods=['GET'])
def training_data_stats():
    try:
        if not os.path.exists(POSITIONS_FILE):
            return jsonify({"total_positions": 0}), 200
        with open(POSITIONS_FILE, 'r') as f:
            line_count = sum(1 for _ in f)
        return jsonify({
            "total_positions": line_count,
            "active_games": len(active_games),
            "data_directory": DATA_DIR
        }), 200
    except Exception as e:
        logger.error(f"Error in training_data_stats: {e}")
        return jsonify({"message": "An error occurred"}), 500

@app.route('/query_agent_move', methods=['POST'])
def query_agent_move():
    """
    Given a pre-move board state and the human's chosen move pair,
    return what the agent would have chosen, whether it differs
    (by resulting board state comparison), and the agent's eval score.
    Used to build contrastive training pairs.
    """
    try:
        data = request.json
        state = data.get('state')
        human_pair_raw = data.get('human_pair')

        if not state or not human_pair_raw:
            return jsonify({"message": "state and human_pair required"}), 400

        local_board = Board()
        local_board.update_state(state)
        moves = local_board.get_valid_moves()

        if not moves:
            return jsonify({"differs": False, "agent_pair": None, "agent_score": None}), 200

        agent_pair = agent.select_move_pair(moves, local_board, local_board.current_player)
        agent_score, _ = agent.evaluate(local_board, local_board.current_player)

        # Normalize move encodings
        local_board.update_state(state)
        human_m1 = _normalize_move(human_pair_raw[0] if len(human_pair_raw) > 0 else None)
        human_m2 = _normalize_move(human_pair_raw[1] if len(human_pair_raw) > 1 else [0, 0, 0])
        agent_m1 = _normalize_move(agent_pair[0] if len(agent_pair) > 0 else None)
        agent_m2 = _normalize_move(agent_pair[1] if len(agent_pair) > 1 else [0, 0, 0])

        differs = not pairs_effectively_equal(
            (human_m1, human_m2), (agent_m1, agent_m2), local_board
        )

        return jsonify({
            "differs": differs,
            "agent_pair": agent_pair,
            "agent_score": agent_score,
        }), 200
    except Exception as e:
        logger.exception("Error in query_agent_move")
        return jsonify({"message": str(e)}), 500


@app.route('/record_contrastive_pair', methods=['POST'])
def record_contrastive_pair():
    """
    Record a disagreement between human and agent move choices.
    final_score is added post-game via a labeling pass (join by game_id + move_index).
    """
    if not RECORD_TRAINING:
        return jsonify({"message": "recording disabled"}), 200
    try:
        data = request.json
        game_id = session.get('game_id')
        if not game_id:
            return jsonify({"message": "No active game"}), 400

        record = {
            'schema_version': SCHEMA_VERSION,
            'game_id': game_id,
            'player': data.get('player'),
            'game_stage': data.get('game_stage', 'unknown'),
            'move_index': data.get('move_index', 0),
            'human_pair': data.get('human_pair'),
            'agent_pair': data.get('agent_pair'),
            'agent_score': data.get('agent_score'),
            'raw_state': data.get('state'),
            'timestamp': int(time.time()),
        }
        save_contrastive_pair(record)
        logger.debug(f"Recorded contrastive pair for game {game_id}, move {record['move_index']}")
        return jsonify({"message": "Contrastive pair recorded"}), 200
    except Exception as e:
        logger.exception("Error in record_contrastive_pair")
        return jsonify({"message": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, use_reloader=False)