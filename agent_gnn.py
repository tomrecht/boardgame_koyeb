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

    def __init__(self, weights_path=GNN_WEIGHTS, model=None):
        self.encoder = BoardEncoder()
        if model is not None:
            self.model = model
            self.model.eval()
        else:
            self.model = load_model(weights_path)
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

        final_scores = scores * SCORE_SCALE
        best_idx     = final_scores.argmax().item()
        return move_keys[best_idx]

    def select_move_pair_fast(self, moves, board, player):
        """
        1-ply move selection for self-play data generation.
        Evaluates only after the first move, then greedily picks
        the best second move given the best first move.

        ~20x fewer forward passes than select_move_pair — suitable
        for generating training data quickly. NOT used for interactive play.
        """
        if not isinstance(moves, (list, set)) or not all(isinstance(m, tuple) for m in moves):
            raise ValueError('Invalid moves format.')

        moves_set = set(moves)

        # Handle pass-only case
        if moves_set == {(0, 0, 0)}:
            return ((0, 0, 0), (0, 0, 0))

        moves_set.discard((0, 0, 0))

        # Encode position after each first move
        first_move_keys    = []
        first_encoded_list = []

        for move in moves_set:
            if not isinstance(move, tuple) or len(move) != 3:
                continue
            initial = len(board.moves)
            board.apply_move(move, switch_turn=False)
            first_move_keys.append(move)
            first_encoded_list.append(self.encoder.encode(board, player))
            while len(board.moves) > initial:
                board.undo_last_move()

        if not first_move_keys:
            return ((0, 0, 0), (0, 0, 0))

        # Evaluate all first moves in one batch
        with torch.no_grad():
            first_scores = self.model(first_encoded_list) * SCORE_SCALE  # [N]

        best_first_idx = first_scores.argmax().item()
        best_first     = first_move_keys[best_first_idx]

        # Apply best first move and find best second move
        initial = len(board.moves)
        board.apply_move(best_first, switch_turn=False)

        if all(die.used for die in board.dice):
            while len(board.moves) > initial:
                board.undo_last_move()
            return (best_first, (0, 0, 0))

        next_moves = set(board.get_valid_moves()) - {(0, 0, 0)}

        if not next_moves:
            while len(board.moves) > initial:
                board.undo_last_move()
            return (best_first, (0, 0, 0))

        # Encode position after each second move
        second_move_keys    = []
        second_encoded_list = []

        for nm in next_moves:
            if not isinstance(nm, tuple) or len(nm) != 3:
                continue
            nm_initial = len(board.moves)
            board.apply_move(nm, switch_turn=False)
            second_move_keys.append(nm)
            second_encoded_list.append(self.encoder.encode(board, player))
            while len(board.moves) > nm_initial:
                board.undo_last_move()

        while len(board.moves) > initial:
            board.undo_last_move()

        if not second_move_keys:
            return (best_first, (0, 0, 0))

        # Evaluate all second moves in one batch
        with torch.no_grad():
            second_scores = self.model(second_encoded_list) * SCORE_SCALE  # [M]

        best_second = second_move_keys[second_scores.argmax().item()]
        return (best_first, best_second)

    def select_move_pair_beam(self, moves, board, player, K=2):

        valid_moves = [m for m in moves if m != (0, 0, 0)]
        if not valid_moves:
            return ((0, 0, 0), (0, 0, 0))

        with torch.no_grad():
            # ---- 1. Evaluate first moves ----
            first_states = []
            first_meta   = []

            initial_len = len(board.moves)

            for m1 in valid_moves:
                board.apply_move(m1, switch_turn=False)
                enc = self.encoder.encode(board, player)
                first_states.append(enc)
                first_meta.append(m1)
                while len(board.moves) > initial_len:
                    board.undo_last_move()

            values = self.model(first_states).squeeze()
            topk = torch.topk(values, min(K, len(values))).indices.tolist()

            best_pair = None
            best_value = -1e9

            # ---- 2. For each top-K first move ----
            for idx in topk:
                m1 = first_meta[idx]

                board.apply_move(m1, switch_turn=False)

                # If no second move
                if all(die.used for die in board.dice):
                    val = values[idx].item()
                    if val > best_value:
                        best_value = val
                        best_pair = (m1, (0, 0, 0))
                    while len(board.moves) > initial_len:
                        board.undo_last_move()
                    continue

                second_moves = list(set(board.get_valid_moves()) - {(0, 0, 0)})

                if not second_moves:
                    val = values[idx].item()
                    if val > best_value:
                        best_value = val
                        best_pair = (m1, (0, 0, 0))
                    while len(board.moves) > initial_len:
                        board.undo_last_move()
                    continue

                # ---- 3. Evaluate second moves ----
                second_states = []
                second_meta   = []

                mid_len = len(board.moves)

                for m2 in second_moves:
                    board.apply_move(m2, switch_turn=False)
                    enc = self.encoder.encode(board, player)
                    second_states.append(enc)
                    second_meta.append(m2)
                    while len(board.moves) > mid_len:
                        board.undo_last_move()

                vals2 = self.model(second_states).squeeze()

                best_idx = torch.argmax(vals2).item()
                val = vals2[best_idx].item()

                if val > best_value:
                    best_value = val
                    best_pair = (m1, second_meta[best_idx])

                while len(board.moves) > initial_len:
                    board.undo_last_move()

            return best_pair if best_pair else ((0,0,0),(0,0,0))