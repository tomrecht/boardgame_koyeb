# python -m http.server 8000
# python app.py
# http://localhost:8000

from flask import Flask, request, jsonify, send_from_status
from flask_cors import CORS
import os
import copy
from game import Board
from agent import Agent
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[logging.StreamHandler()])

logger = logging.getLogger(__name__)


app = Flask(__name__, static_folder='', static_url_path='')
CORS(app)
#CORS(app, resources={r"/*": {"origins": "https://tom-recht.github.io"}})

# Initialize board
board = Board()
agent = Agent(log_to_file=True)


@app.route('/')
def index():
    # This serves your index.html file at the root URL
    return app.send_static_file('index.html')

@app.route('/select_moves', methods=['POST'])
def select_moves():
    try:
        state = request.json
        logger.debug(f"Received state: {state}")
        board.update_state(state)
        #moves = board.get_valid_moves(mask_offgoals=True)
        moves = board.get_valid_moves()
        logger.debug(f"Valid moves: {moves}")
        if moves:
            chosen_moves = agent.select_move_pair(moves, board, board.current_player)
            logger.debug(f"Chosen moves: {chosen_moves}")
            return jsonify({"message": "Game state updated successfully", "move": chosen_moves}), 200
        else:
            return jsonify({"message": "No valid moves available"}), 200
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"message": "An error occurred"}), 500

@app.route('/evaluate_board', methods=['POST'])
def evaluate_board():
    try:
        state = request.json
        logger.debug(f"Received state: {state}")
        new_board = copy.deepcopy(board)
        new_board.update_state(state)
        _, eval = agent.evaluate(new_board, new_board.current_player)
        if eval:
            logger.debug(eval)
            return jsonify({"message": "Game state updated successfully", "eval": eval}), 200
        else:
            return jsonify({"message": "No valid moves available"}), 200
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"message": "An error occurred"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, use_reloader=False)
