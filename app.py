# python -m http.server 8000 & python app.py

from flask import Flask, request, jsonify, session
from flask_cors import CORS
import os
import json
import uuid
import time
import logging
from game import Board
from agent import Agent, get_weights

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='', static_url_path='')
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
CORS(app, supports_credentials=True)

# -------------------------
# DATA COLLECTION SETUP
# -------------------------

DATA_DIR = 'training_data'
POSITIONS_FILE = os.path.join(DATA_DIR, 'positions_with_moves.jsonl')   # clearer name
os.makedirs(DATA_DIR, exist_ok=True)

active_games = {}  # game_id -> {'positions': [], 'start_time': ...}
MAX_MOVES_IN_MEMORY = 200
SCHEMA_VERSION = 1

# -------------------------
# AGENT INITIALIZATION
# -------------------------

current_weights = get_weights()
agent = Agent(weights=current_weights, log_to_file=True)

# Reuse a single board to keep caches across moves (optional, can be per‑request)
shared_board = Board()

# -------------------------
# DATA COLLECTION HELPERS
# -------------------------

def save_position_to_disk(position_data):
    with open(POSITIONS_FILE, 'a') as f:
        f.write(json.dumps(position_data) + '\n')

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
        shared_board.update_state(state)
        moves = shared_board.get_valid_moves()
        logger.debug(f"select_moves: got {len(moves)} valid moves")
        if moves:
            chosen_moves = agent.select_move_pair(moves, shared_board, shared_board.current_player)
            logger.debug(f"select_moves: selected {chosen_moves}")

            # Record black's position server-side
            game_id = session.get('game_id')
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

            return jsonify({"message": "Success", "move": chosen_moves}), 200
        else:
            return jsonify({"message": "No valid moves available"}), 200
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

@app.route('/record_position', methods=['POST'])
def record_position():
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
        total_score, components = agent.evaluate(local_board, local_board.current_player)
        return jsonify({
            "message": "Success",
            "eval": total_score,
            "total_score": total_score,
            "player": components.get("player"),
            "opponent": components.get("opponent"),
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, use_reloader=False)