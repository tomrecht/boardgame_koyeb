"""
agent_gnn.py — Drop-in replacement for Agent using the trained GNN.

Same interface as Agent:
    agent = GNNAgent()
    move_pair = agent.select_move_pair(moves, board, player)
    score, components = agent.evaluate(board, player)

To use in app.py, change:
    agent = Agent(weights=INITIAL_WEIGHTS)
to:
    from agent_gnn import GNNAgent
    agent = GNNAgent()
"""

import torch
from network import BoardEncoder, BoardGNN, load_model

GAME_OVER_SCORE = 10000
SCORE_SCALE     = 1000.0   # must match train_distill.py
GNN_WEIGHTS     = 'gnn_weights.pt'


class GNNAgent:
    """
    Drop-in replacement for Agent that uses the GNN for position evaluation.
    Keeps the same 2-ply lookahead logic from select_move_pair.
    BoardEncoder and BoardGNN are instantiated once and reused.
    """

    def __init__(self, weights_path=GNN_WEIGHTS):
        self.encoder = BoardEncoder()
        self.model   = load_model(weights_path)
        self.model.eval()   # inference mode — no dropout etc.

    def evaluate(self, board, player):
        """
        Evaluate board position from player's perspective.
        Returns (score, components) to match Agent.evaluate() interface.
        score is in the same units as the weighted-sum agent (multiplied
        back up by SCORE_SCALE so move selection comparisons are meaningful).
        components is a minimal dict for compatibility with /evaluate_board.
        """
        winner, score = board.check_game_over()
        if winner:
            factor = 1 if winner == player else -1
            return factor * score * GAME_OVER_SCORE, {'game_over': True}

        encoded = self.encoder.encode(board, player)

        with torch.no_grad():
            raw_score = self.model(encoded).item()

        # Denormalise to match the scale of the weighted-sum agent
        final_score = raw_score * SCORE_SCALE

        components = {
            'gnn_raw':   raw_score,
            'gnn_score': final_score,
            '_player':   player,
        }

        return final_score, components

    def select_move_pair(self, moves, board, player):
        """
        Identical 2-ply lookahead logic to Agent.select_move_pair,
        but uses self.evaluate() (GNN) instead of the weighted-sum evaluator.
        """
        move_scores = dict()

        if not isinstance(moves, (list, set)) or not all(isinstance(m, tuple) for m in moves):
            raise ValueError('Invalid moves format: expected a list or set of tuples.')

        # Evaluate pass if legal
        if (0, 0, 0) in moves:
            move_scores[((0, 0, 0), (0, 0, 0))] = self.evaluate(board, player)

        moves = set(moves)
        moves.discard((0, 0, 0))

        for move in moves:
            if not isinstance(move, tuple) or len(move) != 3:
                raise ValueError('Invalid move format.')

            initial_move_count = len(board.moves)

            board.apply_move(move, switch_turn=False)

            # Score pass as second move if legal
            remaining_captured = [p for p in board.home_tile.pieces
                                   if p.player == board.current_player]
            if not remaining_captured:
                move_scores[(move, (0, 0, 0))] = self.evaluate(board, player)

            if all(die.used for die in board.dice):
                while len(board.moves) > initial_move_count:
                    board.undo_last_move()
                continue

            next_moves = set(board.get_valid_moves())

            if not next_moves:
                while len(board.moves) > initial_move_count:
                    board.undo_last_move()
                continue

            next_moves.discard((0, 0, 0))

            for next_move in next_moves:
                if not isinstance(next_move, tuple) or len(next_move) != 3:
                    raise ValueError('Invalid next move format.')

                board.apply_move(next_move, switch_turn=False)
                move_scores[(move, next_move)] = self.evaluate(board, player)
                board.undo_last_move()

            while len(board.moves) > initial_move_count:
                board.undo_last_move()

        if not move_scores:
            return ((0, 0, 0), (0, 0, 0))

        best_move_pair = max(move_scores, key=lambda k: move_scores[k][0])
        return best_move_pair
