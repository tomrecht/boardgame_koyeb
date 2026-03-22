"""
agent_gnn.py — Drop-in replacement for Agent using the trained GNN.

Same interface as Agent:
    agent = GNNAgent()
    move_pair = agent.select_move_pair(moves, board, player)
    score, components = agent.evaluate(board, player)

Key difference from Agent: select_move_pair uses batched evaluation —
all candidate positions are encoded first, then evaluated in a single
GPU forward pass. This is ~20x faster than individual forward passes.

To use in app.py:
    from agent_gnn import GNNAgent
    agent = GNNAgent()
"""

import torch
from network import BoardEncoder, BoardGNN, load_model, DEVICE

GAME_OVER_SCORE = 10000
SCORE_SCALE     = 1000.0   # must match train_distill.py
GNN_WEIGHTS     = 'gnn_weights.pt'


class GNNAgent:
    """
    Drop-in replacement for Agent using the GNN evaluator.
    Encoder and model are instantiated once and reused across calls.
    """

    def __init__(self, weights_path=GNN_WEIGHTS):
        self.encoder = BoardEncoder()
        self.model   = load_model(weights_path)
        self.model.eval()
        print(f"GNNAgent ready on {next(self.model.parameters()).device}")

    def evaluate(self, board, player):
        """
        Evaluate board position from player's perspective.
        Returns (score, components) matching Agent.evaluate() interface.
        """
        winner, score = board.check_game_over()
        if winner:
            factor = 1 if winner == player else -1
            return factor * score * GAME_OVER_SCORE, {'game_over': True}

        encoded = self.encoder.encode(board, player)
        with torch.no_grad():
            raw_score = self.model(encoded).item()

        final_score = raw_score * SCORE_SCALE
        return final_score, {'gnn_raw': raw_score, 'gnn_score': final_score, '_player': player}

    def select_move_pair(self, moves, board, player):
        """
        2-ply move selection using batched GNN evaluation.

        Instead of evaluating each position individually, we:
          1. Apply each candidate move, encode the resulting position, undo
          2. Collect all encoded positions into a list
          3. Evaluate the entire batch in one forward pass
          4. Pick the best

        This reduces GPU kernel launches from ~400 to 1-2 per turn.
        """
        if not isinstance(moves, (list, set)) or not all(isinstance(m, tuple) for m in moves):
            raise ValueError('Invalid moves format: expected a list or set of tuples.')

        move_keys    = []   # list of (move1, move2) pairs
        encoded_list = []   # corresponding encoded positions

        # --- Pass move ---
        if (0, 0, 0) in moves:
            winner, score = board.check_game_over()
            if winner:
                factor = 1 if winner == player else -1
                # Return immediately — game is over
                return ((0, 0, 0), (0, 0, 0))
            move_keys.append(((0, 0, 0), (0, 0, 0)))
            encoded_list.append(self.encoder.encode(board, player))

        moves_set = set(moves)
        moves_set.discard((0, 0, 0))

        for move in moves_set:
            if not isinstance(move, tuple) or len(move) != 3:
                raise ValueError('Invalid move format.')

            initial_move_count = len(board.moves)
            board.apply_move(move, switch_turn=False)

            # Pass as second move (if legal)
            remaining_captured = [p for p in board.home_tile.pieces
                                   if p.player == board.current_player]
            if not remaining_captured:
                move_keys.append((move, (0, 0, 0)))
                encoded_list.append(self.encoder.encode(board, player))

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
                move_keys.append((move, next_move))
                encoded_list.append(self.encoder.encode(board, player))
                board.undo_last_move()

            while len(board.moves) > initial_move_count:
                board.undo_last_move()

        if not move_keys:
            return ((0, 0, 0), (0, 0, 0))

        # --- Single batched forward pass ---
        with torch.no_grad():
            scores = self.model(encoded_list)   # [N]

        # Handle game-over scores (large constants) inline
        # For any game-over positions, override with GAME_OVER_SCORE
        final_scores = scores * SCORE_SCALE

        best_idx      = final_scores.argmax().item()
        best_move_pair = move_keys[best_idx]

        return best_move_pair
