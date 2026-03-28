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
agent = Agent(weights=current_weights, log_to_file=False)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/select_moves', methods=['POST'])
def select_moves():
    try:
        state = request.json

        local_board = Board()
        local_board.update_state(state)
        
        moves = local_board.get_valid_moves()
        
        if moves:
            chosen_moves = agent.select_move_pair(moves, local_board, local_board.current_player)
            return jsonify({"message": "Success", "move": chosen_moves}), 200
        else:
            return jsonify({"message": "No valid moves available"}), 200
    except Exception as e:
        logger.error(f"Error in select_moves: {e}")
        return jsonify({"message": "An error occurred"}), 500

@app.route('/evaluate_board', methods=['POST'])
def evaluate_board():
    try:
        state = request.json
        local_board = Board()
        local_board.update_state(state)
        
        _, eval_data = agent.evaluate(local_board, local_board.current_player)
        
        if eval_data:
            return jsonify({"message": "Success", "eval": eval_data}), 200
        else:
            return jsonify({"message": "Evaluation failed"}), 200
    except Exception as e:
        logger.error(f"Error in evaluate_board: {e}")
        return jsonify({"message": "An error occurred"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, use_reloader=False)