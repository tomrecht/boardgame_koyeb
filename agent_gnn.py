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
    
    _printed_ready = False  # class-level flag

    def __init__(self, weights_path=GNN_WEIGHTS, model=None,
                 use_prefilter=False, prefilter_top_k=40, heuristic_weights=None,
                 prefilter_min_k=5, prefilter_frac=None, prefilter_score_alpha=None):
        self.encoder = BoardEncoder()
        if model is not None:
            self.model = model
            self.model.eval()
        else:
            self.model = load_model(weights_path)

        # Optional move-pair pre-filter: rank candidates with the cheap heuristic
        # and only GNN-encode the top ones. Speeds up high-branching turns and
        # removes the GNN's heuristically-bad blind spots.
        #   prefilter_top_k       : absolute count cap (K).
        #   prefilter_frac        : dynamic cap as a fraction of all pairs (rank-based;
        #                           invariant to ANY monotonic heuristic change).
        #   prefilter_score_alpha : adaptive score cutoff in [0,1] on the within-
        #                           position normalized score (affine-invariant to
        #                           the heuristic scale; survives re-weighting).
        #   prefilter_min_k       : floor -- always keep at least this many.
        self.use_prefilter        = use_prefilter
        self.prefilter_top_k      = prefilter_top_k
        self.prefilter_min_k      = prefilter_min_k
        self.prefilter_frac       = prefilter_frac
        self.prefilter_score_alpha = prefilter_score_alpha
        # kept-set instrumentation (compute proxy for sweeps)
        self.dbg_kept_total = 0
        self.dbg_kept_calls = 0
        self.heuristic = None
        if use_prefilter:
            from agent import Agent, get_weights
            if isinstance(heuristic_weights, str):
                w = get_weights(heuristic_weights)
            else:
                w = heuristic_weights          # dict or None (None -> Agent loads defaults)
            self.heuristic = Agent(weights=w)

        if not GNNAgent._printed_ready:
            print(f"GNNAgent ready on {next(self.model.parameters()).device}")
            GNNAgent._printed_ready = True

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

        prefilter = self.use_prefilter and self.heuristic is not None

        move_keys    = []   # list of (move1, move2) pairs
        encoded_list = []   # corresponding encoded positions (filled when NOT prefiltering)
        scored       = []   # (heuristic_score, pair)        (filled when prefiltering)

        def record(pair):
            # board is currently IN the resulting position for `pair`
            if prefilter:
                s, _ = self.heuristic.evaluate(board, player)   # cheap; returns (score, components)
                scored.append((s, pair))
            else:
                move_keys.append(pair)
                encoded_list.append(self.encoder.encode(board, player))

        # --- Pass move ---
        if (0, 0, 0) in moves:
            record(((0, 0, 0), (0, 0, 0)))

        moves_set = set(moves)
        moves_set.discard((0, 0, 0))

        for move in moves_set:
            if not isinstance(move, tuple) or len(move) != 3:
                raise ValueError('Invalid move format.')

            initial_move_count = len(board.moves)
            board.apply_move(move, switch_turn=False)

            # Terminal short-circuit: a winning move beats any learned value.
            # The GNN never sees game_over (it just encodes the board), and a
            # fully-won board is maximally out-of-distribution, so its value is
            # unreliable -- take the guaranteed win directly.
            wgo, _ = board.check_game_over()
            if wgo == player:
                while len(board.moves) > initial_move_count:
                    board.undo_last_move()
                return (move, (0, 0, 0))

            # Pass as second move (if legal)
            remaining_captured = [p for p in board.home_tile.pieces
                                   if p.player == board.current_player]
            if not remaining_captured:
                record((move, (0, 0, 0)))

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
                wgo, _ = board.check_game_over()
                if wgo == player:
                    while len(board.moves) > initial_move_count:
                        board.undo_last_move()
                    return (move, next_move)
                record((move, next_move))
                board.undo_last_move()

            while len(board.moves) > initial_move_count:
                board.undo_last_move()

        # --- If prefiltering: pick the kept candidates, then encode ONLY those ---
        if prefilter:
            if not scored:
                return ((0, 0, 0), (0, 0, 0))
            top_pairs = self._select_filtered(scored)
            self.dbg_kept_total += len(top_pairs)
            self.dbg_kept_calls += 1
            for pair in top_pairs:
                base = len(board.moves)
                for m in pair:
                    if m != (0, 0, 0):
                        board.apply_move(m, switch_turn=False)
                move_keys.append(pair)
                encoded_list.append(self.encoder.encode(board, player))
                while len(board.moves) > base:
                    board.undo_last_move()

        if not move_keys:
            return ((0, 0, 0), (0, 0, 0))

        # --- Single batched forward pass over the (filtered) candidates ---
        with torch.no_grad():
            scores = self.model(encoded_list)   # [N]

        final_scores = scores * SCORE_SCALE
        best_idx     = final_scores.argmax().item()
        return self._fix_never_good(move_keys[best_idx], board, player)

    def _fix_never_good(self, pair, board, player):
        """
        Two never-good corrections to the GNN's chosen pair. Each produces a
        LEGAL sibling (same companion move, same dice):
          1) Saving an UNNUMBERED piece when a NUMBERED piece can be saved from
             the same goal with the same die -> save the numbered piece instead.
             (Numbered pieces can only ever be saved from their own goal, so
             clearing them first is strictly better.)
          2) PASSING a die when a piece could be saved instead -> save it.
        Conservative: only acts on clear, individually-legal swaps.
        """
        if not (isinstance(pair, tuple) and len(pair) == 2):
            return pair
        m1, m2 = pair

        def is_pass(m):  return m == (0, 0, 0)
        def is_save(m):  return isinstance(m, tuple) and len(m) == 3 and m[1] == 'save'
        def is_num(pid): return isinstance(pid, tuple) and len(pid) == 2 and pid[1] <= 6

        # ---- Rule 1: numbered save dominates unnumbered save from same goal ----
        def upgrade(save_move):
            if not is_save(save_move) or is_num(save_move[0]):
                return save_move
            pl, _num = save_move[0]
            roll = save_move[2]
            piece = board.piece_lookup.get((pl, _num))
            if not piece or not piece.tile:
                return save_move
            for p in piece.tile.pieces:                 # same goal tile
                if p.player == player and p.number <= 6 and p.number == roll:
                    return ((p.player, p.number), 'save', roll)
            return save_move

        u1, u2 = upgrade(m1), upgrade(m2)
        if (u1, u2) != (m1, m2):
            return (u1, u2)

        # ---- Rule 2: never pass when a save is available ----
        def best_save(valid):
            saves = [mv for mv in valid if is_save(mv)]
            if not saves:
                return None
            saves.sort(key=lambda mv: 0 if is_num(mv[0]) else 1)   # prefer numbered
            return saves[0]

        if is_pass(m1) and is_pass(m2):
            s = best_save(board.get_valid_moves())
            if s is not None:
                return (s, (0, 0, 0))
        elif is_pass(m2) and not is_pass(m1):
            base = len(board.moves)
            board.apply_move(m1, switch_turn=False)
            s = best_save(board.get_valid_moves())
            while len(board.moves) > base:
                board.undo_last_move()
            if s is not None:
                return (m1, s)
        return pair

    def _select_filtered(self, scored):
        """
        Given scored = list of (heuristic_score, pair), return the kept pairs.

        Combines three optional knobs (applied together):
          * prefilter_score_alpha (alpha in [0,1]): adaptive SCORE cutoff. Keep
            pairs whose within-position normalized score
            (score - worst) / (best - worst) is >= 1 - alpha. alpha None or >= 1
            disables it (keep all by score); alpha = 0 keeps only pairs tied with
            the best. Normalized per position, so it is AFFINE-INVARIANT to the
            heuristic's scale -- re-weighting the heuristic (s' = a*s + b, a > 0)
            leaves the kept set unchanged.
          * prefilter_frac (f in (0,1]): dynamic count cap = round(f * n_pairs).
            Purely rank/percentile based, so it is invariant to ANY monotonic
            transform of the heuristic score.
          * prefilter_top_k (K): absolute count cap.
          * prefilter_min_k (m): floor -- always keep at least m (capped by n) so
            the GNN always has options.

        Kept count = clamp(score_cutoff_count, min_k, min(top_k, frac_cap)).
        Pure top-K is recovered with alpha=None, frac=None.
        """
        n = len(scored)
        if n == 0:
            return []
        scored = sorted(scored, key=lambda x: x[0], reverse=True)

        # 1) adaptive count from the normalized score cutoff
        alpha = self.prefilter_score_alpha
        if alpha is None or alpha >= 1.0:
            cut_count = n
        else:
            best   = scored[0][0]
            worst  = scored[-1][0]
            spread = best - worst
            if spread <= 1e-12:
                cut_count = n                       # all equal: cutoff can't discriminate
            else:
                thresh = best - alpha * spread
                cut_count = sum(1 for s, _ in scored if s >= thresh)

        # 2) count ceilings: absolute top_k and/or fraction-of-pairs
        ceil_k = self.prefilter_top_k if self.prefilter_top_k else n
        if self.prefilter_frac is not None:
            ceil_k = min(ceil_k, max(1, round(self.prefilter_frac * n)))

        # 3) combine: floor by min_k, ceil by ceil_k
        min_k = self.prefilter_min_k or 1
        k = max(cut_count, min(min_k, n))
        k = min(k, ceil_k)
        k = max(1, min(k, n))
        return [pair for _, pair in scored[:k]]

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
            # Terminal short-circuit: winning first move.
            wgo, _ = board.check_game_over()
            if wgo == player:
                while len(board.moves) > initial:
                    board.undo_last_move()
                return (move, (0, 0, 0))
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
        # Terminal short-circuit: best first move is a win.
        wgo, _ = board.check_game_over()
        if wgo == player:
            while len(board.moves) > initial:
                board.undo_last_move()
            return (best_first, (0, 0, 0))

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
            # Terminal short-circuit: winning second move.
            wgo, _ = board.check_game_over()
            if wgo == player:
                while len(board.moves) > nm_initial:
                    board.undo_last_move()
                while len(board.moves) > initial:
                    board.undo_last_move()
                return (best_first, nm)
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
                # Terminal short-circuit: winning first move.
                wgo, _ = board.check_game_over()
                if wgo == player:
                    while len(board.moves) > initial_len:
                        board.undo_last_move()
                    return (m1, (0, 0, 0))
                enc = self.encoder.encode(board, player)
                first_states.append(enc)
                first_meta.append(m1)
                while len(board.moves) > initial_len:
                    board.undo_last_move()

            values = self.model(first_states).view(-1)
            topk = torch.topk(values, min(K, len(values))).indices.tolist()

            best_pair = None
            best_value = -1e9

            # ---- 2. For each top-K first move ----
            for idx in topk:
                m1 = first_meta[idx]

                board.apply_move(m1, switch_turn=False)
                # Terminal short-circuit: top-K first move is a win.
                wgo, _ = board.check_game_over()
                if wgo == player:
                    while len(board.moves) > initial_len:
                        board.undo_last_move()
                    return (m1, (0, 0, 0))

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
                    # Terminal short-circuit: winning second move.
                    wgo, _ = board.check_game_over()
                    if wgo == player:
                        while len(board.moves) > mid_len:
                            board.undo_last_move()
                        while len(board.moves) > initial_len:
                            board.undo_last_move()
                        return (m1, m2)
                    enc = self.encoder.encode(board, player)
                    second_states.append(enc)
                    second_meta.append(m2)
                    while len(board.moves) > mid_len:
                        board.undo_last_move()

                vals2 = self.model(second_states).view(-1)

                best_idx = torch.argmax(vals2).item()
                val = vals2[best_idx].item()

                if val > best_value:
                    best_value = val
                    best_pair = (m1, second_meta[best_idx])

                while len(board.moves) > initial_len:
                    board.undo_last_move()

            return best_pair if best_pair else ((0,0,0),(0,0,0))