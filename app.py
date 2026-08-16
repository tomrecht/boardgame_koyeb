# python app.py, then localhost:10000

import os
import sys
# The app serves one position per request -- latency-bound inference where
# GPU/MPS buys nothing, and some machines' MPS stacks hard-abort on the
# GNN's batched scatter_add (MPSNDArrayScatter rank assertion on the iMac).
# Default the app to CPU BEFORE network.py is imported (it reads this env
# var at import time); an explicitly exported BOARDGAME_DEVICE still wins.
os.environ.setdefault('BOARDGAME_DEVICE', 'cpu')

from flask import Flask, request, jsonify, send_file, abort
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
CORS(app, supports_credentials=True)

# -------------------------
# MOVE BUDGET
# -------------------------

# Human-play recording was removed entirely (2026-08-15), server side and
# client side, as part of moving inference onto the device: with the AI local,
# nothing needs an application server at runtime and the recording chain was the
# last thing keeping one. Recover it from git history if a collection run is
# ever wanted again -- it needs BOTH halves, the routes here and the posting
# code in game.js.

# Hard ceiling on how long one move may take: past it the agent chooses from the
# candidates it has rather than hanging the worker until it is killed.
#
# Two things can time out above this, and the budget has to stay under BOTH:
#   * gunicorn's --timeout (WEB_TIMEOUT), which kills the worker mid-request;
#   * the platform's own request timeout, which returns a 504 to the browser
#     while the worker keeps grinding (Koyeb's is 60s unless raised).
# So the default is WEB_TIMEOUT minus a margin for scoring the collected
# candidates and sending the response, and MOVE_BUDGET can override it.
_WEB_TIMEOUT = float(os.environ.get('WEB_TIMEOUT', '60'))
MOVE_BUDGET = float(os.environ.get('MOVE_BUDGET') or max(5.0, _WEB_TIMEOUT - 20))
if MOVE_BUDGET > _WEB_TIMEOUT - 10:
    logger.warning(f'MOVE_BUDGET={MOVE_BUDGET:.0f}s leaves little room under the '
                   f'{_WEB_TIMEOUT:.0f}s worker timeout; a slow move may kill the worker')


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

logger.info(f'move budget {MOVE_BUDGET:.0f}s (worker timeout {_WEB_TIMEOUT:.0f}s)')
_model_path = _opponent_model_path()
if not os.path.exists(_model_path):
    logger.warning(f'Model file not found: {_model_path} (cwd {os.getcwd()})')
logger.info(f'Loading opponent model: {_model_path}')
#agent = Agent(weights=current_weights, log_to_file=True)
# Two-stage prefilter, for the served agent only: rank first moves, expand the
# best FIRST_MOVE_PREFILTER of them into pairs. Measured at F=12 over 120 paired
# games (match_prefilter.py): 52.1% for the fast agent, mean paired margin +0.05
# (95% CI -0.22..+0.33) -- no measurable strength change -- while the worst move
# drops from 3.65s to 1.28s, which is what a fractional-vCPU instance feels.
# Training, self-play and the arena keep the library default of 0.
FIRST_MOVE_PREFILTER = int(os.environ.get('FIRST_MOVE_PREFILTER', '12'))
agent = GNNAgent(weights_path=_model_path, use_prefilter=True, prefilter_top_k=40,
                 heuristic_weights=current_weights,
                 first_move_prefilter=FIRST_MOVE_PREFILTER)
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
# -------------------------
# API ENDPOINTS
# -------------------------

# Precompressed static assets. The inference runtime is 10.5 MB raw and 2.8 MB
# gzipped, and Flask does not compress anything by default -- so without this
# every first visit downloads the full 10.5 MB. Compressing on the fly instead
# would burn CPU on a small instance for every new client, so the file is
# gzipped once at image build time (see Dockerfile) and served as-is here.
#
# Falls straight through to the normal static handler when no .gz exists, which
# is what happens in local development.
@app.route('/ort/<path:filename>')
def serve_ort(filename):
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ort')
    raw = os.path.join(directory, filename)
    gz = raw + '.gz'
    accepts_gzip = 'gzip' in request.headers.get('Accept-Encoding', '')
    if accepts_gzip and os.path.exists(gz):
        resp = send_file(gz, mimetype=_ort_mimetype(filename), conditional=True)
        resp.headers['Content-Encoding'] = 'gzip'
        # Vary matters: a cache that saw the gzipped reply must not hand it to a
        # client that cannot decode it.
        resp.headers['Vary'] = 'Accept-Encoding'
        return resp
    if not os.path.exists(raw):
        abort(404)
    return send_file(raw, mimetype=_ort_mimetype(filename), conditional=True)


def _ort_mimetype(filename):
    # WebAssembly.instantiateStreaming REFUSES anything but application/wasm.
    if filename.endswith('.wasm'):
        return 'application/wasm'
    if filename.endswith('.mjs') or filename.endswith('.js'):
        return 'text/javascript'
    return 'application/octet-stream'


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
            t0 = time.monotonic()
            chosen_moves = agent.select_move_pair(moves, shared_board, shared_board.current_player,
                                                  difficulty=difficulty,
                                                  deadline=t0 + MOVE_BUDGET)
            dt = time.monotonic() - t0
            if dt > 5:
                logger.warning(f'slow move: {dt:.1f}s over {len(moves)} legal moves')
            else:
                logger.info(f'move chosen in {dt:.2f}s ({len(moves)} legal moves)')
            logger.debug(f"select_moves: selected {chosen_moves}")

            return jsonify({"message": "Success", "move": chosen_moves, "no_save": no_save}), 200
        else:
            return jsonify({"message": "No valid moves available", "no_save": no_save}), 200
    except Exception as e:
        logger.exception("Error in select_moves")
        return jsonify({"message": "Internal server error", "error": str(e)}), 500

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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, use_reloader=False)