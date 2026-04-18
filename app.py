# python -m http.server 8000
# python app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from game import Board
from agent import Agent, get_weights
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[logging.StreamHandler()])

logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='', static_url_path='')
CORS(app)

current_weights = get_weights()
agent = Agent(weights=current_weights, log_to_file=True)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/select_moves', methods=['POST'])
def select_moves():
    try:
        state = request.json
        print(f"DEBUG select_moves: Received state with keys: {state.keys()}")
        
        local_board = Board()
        
        print("DEBUG select_moves: About to call update_state")
        local_board.update_state(state)
        print("DEBUG select_moves: update_state completed")
        
        print("DEBUG select_moves: About to get valid moves")
        moves = local_board.get_valid_moves()
        print(f"DEBUG select_moves: Got {len(moves)} valid moves")
        
        if moves:
            print("DEBUG select_moves: About to call agent.select_move_pair")
            chosen_moves = agent.select_move_pair(moves, local_board, local_board.current_player)
            print(f"DEBUG select_moves: Selected moves: {chosen_moves}")
            return jsonify({"message": "Success", "move": chosen_moves}), 200
        else:
            return jsonify({"message": "No valid moves available"}), 200
    except Exception as e:
        logger.error(f"Error in select_moves: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"message": "An error occurred"}), 500

@app.route('/debug_piece_blots', methods=['POST'])
def debug_piece_blots():
    try:
        data = request.json
        game_state = data['gameState']
        piece_info = data['piece']
        
        local_board = Board()
        local_board.update_state(game_state)
        
        # Find the piece
        piece = None
        for p in local_board.pieces:
            if p.player == piece_info['player'] and p.number == piece_info['number']:
                piece = p
                break
        
        if not piece:
            return jsonify({"error": "Piece not found"}), 404
        
        # Get distance and blot count
        distance = local_board.shortest_route_to_goal(piece)
        
        # Make sure the method exists
        if hasattr(local_board, 'count_enemy_blots_on_shortest_path'):
            blot_count = local_board.count_enemy_blots_on_shortest_path(piece)
        else:
            blot_count = "Method not implemented yet"
        
        return jsonify({
            "distance": distance,
            "blot_count": blot_count,
            "can_be_saved": piece.can_be_saved()
        }), 200
        
    except Exception as e:
        print(f"Error in debug_piece_blots: {e}")  # Use print instead of logger if logger not configured
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, use_reloader=False)