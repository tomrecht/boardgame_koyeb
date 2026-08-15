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

import time

import numpy as np
from gnn_backend import make_backend

GAME_OVER_SCORE = 10000
SCORE_SCALE     = 1000.0   # must match train_distill.py
NUM_PIECES      = 12       # margin display unit (raw * NUM_PIECES = expected margin)
GNN_WEIGHTS     = 'gnn_weights.pt'


def move_keys_would_exist(scored, move_keys):
    """True once at least one candidate has been recorded, so an early stop can
    never leave the agent with nothing to choose from."""
    return bool(scored) or bool(move_keys)


def _piece_locs(board):
    """Where every piece sits: tile index, -2 saved, -1 still on the rack.
    board.pieces order is fixed for the whole search, so no sort is needed. Two
    states differing only by swapping identical blank pieces get different keys
    -- a missed cache hit, never a wrong one."""
    saved = (board.white_saved, board.black_saved)
    locs = []
    for p in board.pieces:
        if p.tile is not None:
            locs.append(p.tile.index)
        elif p.rack is saved[0] or p.rack is saved[1]:
            locs.append(-2)
        else:
            locs.append(-1)
    return tuple(locs)


def _position_key(board):
    """Everything the heuristic reads: piece placement plus the dice."""
    return (_piece_locs(board), board.dice[0].used, board.dice[1].used)


def _move_sort_key(move):
    """A total order on moves that does not depend on Python's hash seed.

    Enumeration runs over `set`s of move tuples, so the ITERATION ORDER varies
    per process. Wherever a tie is resolved by "whichever came first" -- the
    argmax, the stable sorts in the prefilter, the fewest-relocations tie-break
    -- that made the agent's answer a function of the hash seed. Measured over
    30 pinned positions: 3 gave a different pair under a different PYTHONHASHSEED,
    every one of them a transposition (same piece, same destination, dice in the
    other order), i.e. the same end-of-turn position at the same score.

    Ordering the ties by this key instead makes select_move_pair a pure function
    of the position. It only ever reorders EXACT ties, so it cannot change which
    line the agent judges best -- verified by scoring the old and new choices.
    """
    if not (isinstance(move, tuple) and len(move) == 3):
        return (3, '', 0, 0, 0)
    piece_id, destination, roll = move
    if not isinstance(piece_id, tuple):                 # pass (0,0,0) / draw (1,1,1)
        return (0, '', int(piece_id), 0, int(roll))
    player, number = piece_id
    if destination == 'save':
        return (1, player, int(number), 0, int(roll))
    if destination == 0:                                # block-save
        return (2, player, int(number), 0, int(roll))
    ring, pos = destination
    return (3, player, int(number), int(ring) * 1000 + int(pos), int(roll))


def _pair_sort_key(pair):
    return tuple(_move_sort_key(m) for m in pair)


def _pair_relocations(pair):
    """How many of a pair's moves shuffle a piece around the board, as opposed
    to saving it or passing -- the legibility cost of a line."""
    return sum(1 for m in pair
               if isinstance(m, tuple) and len(m) == 3 and m[1] not in ('save', 0))


def _top_indices(values, k):
    """Indices of the k largest values, best first (numpy stand-in for topk)."""
    v = np.asarray(values).reshape(-1)
    k = min(int(k), v.shape[0])
    idx = np.argpartition(-v, k - 1)[:k] if k < v.shape[0] else np.arange(v.shape[0])
    return [int(i) for i in idx[np.argsort(-v[idx], kind='stable')]]


class GNNAgent:
    """
    Drop-in replacement for Agent using the GNN evaluator.
    Encoder and model are instantiated once and reused across calls.
    """
    
    _printed_ready = False  # class-level flag

    def __init__(self, weights_path=GNN_WEIGHTS, model=None,
                 use_prefilter=False, prefilter_top_k=40, heuristic_weights=None,
                 prefilter_min_k=5, prefilter_frac=None, prefilter_score_alpha=None,
                 first_move_prefilter=0):
        # The backend owns both the net and the encoder that suits it: torch
        # for training/analysis, onnxruntime (numpy, no torch) for deployment.
        # Either way self.model(...) returns numpy scores.
        self.backend = make_backend(weights_path, model=model)
        self.model = self.backend
        self.encoder = self.backend.encoder

        # Two-stage prefilter (0 = off). The one-stage prefilter below scores
        # every candidate PAIR with the heuristic -- ~10k evaluations on a busy
        # midgame turn, which is ~94% of the time a move takes. With this set to
        # F, first moves are scored on their own (~150 evaluations), the best F
        # are kept, and only those get their second moves enumerated. The risk is
        # a pair whose first move looks poor alone but is strong in combination;
        # first moves that save a piece are kept regardless.
        self.first_move_prefilter = int(first_move_prefilter)

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
        # never-good correction logging. debug_never_good -> emit a logging.INFO
        # line each time a correction fires (visible in the Flask server terminal
        # during live play). last_never_good / never_good_counts are always kept
        # (readable from any process, e.g. app.py).
        self.debug_never_good = False
        self.last_never_good = None
        self.never_good_counts = {}
        # _fix_never_good is a hand-coded patch over the value head's errors
        # (e.g. undervaluing saves). DISABLED by default: TD(lambda) is meant to
        # fix these errors in the value function itself, and a buggy patch in the
        # loop corrupts data and masks whether training is working. The duplicate
        # -save dedupe (a hard game invariant) stays on regardless. Set
        # enable_never_good=True to re-enable the heuristic corrections.
        self.enable_never_good = False
        # Diagnostic: when True, log whenever the chosen pair saves nothing yet a
        # save was legally available, distinguishing a value-head error (save
        # scored but not chosen) from a candidate drop (save never reached the
        # GNN, i.e. prefilter/enumeration). See select_move_pair.
        self.debug_pass_over_save = False
        self.heuristic = None
        if use_prefilter:
            from agent import Agent, get_weights
            if isinstance(heuristic_weights, str):
                w = get_weights(heuristic_weights)
            else:
                w = heuristic_weights          # dict or None (None -> Agent loads defaults)
            self.heuristic = Agent(weights=w)

        if not GNNAgent._printed_ready:
            print(f"GNNAgent ready: {self.backend.describe()}")
            GNNAgent._printed_ready = True

    def best_play_value(self, board, player):
        """Expected margin (in pieces, from `player`'s perspective) of the
        position after `player` plays its best move pair from here -- i.e. the
        value the GNN assigns to its own chosen continuation. Returned in the
        same units as the 'current' margin shown in the UI (raw * NUM_PIECES).
        Returns None if there are no legal moves to evaluate."""
        winner, score = board.check_game_over()
        if winner:
            factor = 1 if winner == player else -1
            return factor * score

        moves = board.get_valid_moves()
        if not moves:
            return None
        ranked = self.select_move_pair(moves, board, player, return_scores=True)
        if not ranked:
            return None
        best_final, best_pair = ranked[0][0], ranked[0][1]
        if best_final == float('inf'):
            # Guaranteed winning pair: report the ACTUAL final margin of the
            # resulting won position (not a flat NUM_PIECES), from player's view.
            base = len(board.moves)
            try:
                for m in best_pair:
                    if m != (0, 0, 0):
                        board.apply_move(m, switch_turn=False)
                w, s = board.check_game_over()
            finally:
                while len(board.moves) > base:
                    board.undo_last_move()
            if w:
                return (1 if w == player else -1) * float(s)
            return float(NUM_PIECES)   # fallback (shouldn't happen)
        best_raw = best_final / SCORE_SCALE   # back to the model's raw output
        return best_raw * NUM_PIECES

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
        raw_score = float(self.model(encoded))
        final_score = raw_score * SCORE_SCALE
        return final_score, {'gnn_raw': raw_score, 'gnn_score': final_score, '_player': player}

    def _pick_move_index(self, final_scores, difficulty=None):
        """Difficulty-controlled index over candidate scores (a 1-D array).

        `difficulty` (else self.difficulty) in [0, 1]: 1 = argmax / full
        strength; lower =
        top-p sampling over a scale-invariant softmax (z-scored by the candidate
        spread), so the agent plays visibly weaker without picking terrible moves.
        """
        s = np.asarray(final_scores, dtype=np.float64).reshape(-1)
        d = difficulty if difficulty is not None else getattr(self, 'difficulty', 1.0)
        n = s.shape[0]
        if n <= 1 or d is None or float(d) >= 0.999:
            return int(np.argmax(s))
        d = max(0.0, min(1.0, float(d)))
        std = float(s.std())
        if not (std > 1e-6):
            std = 1.0
        temp  = 0.4 + (1.0 - d) * 2.6      # 0.4 (hard) .. 3.0 (easy)
        top_p = 0.25 + (1.0 - d) * 0.75    # 0.25 (hard) .. 1.0 (easy)
        z = (s - s.max()) / (std * temp)
        e = np.exp(z - z.max())
        probs = e / e.sum()
        order = np.argsort(-probs, kind='stable')
        sorted_probs = probs[order]
        cum = np.cumsum(sorted_probs)
        k = int((cum < top_p).sum()) + 1
        k = max(1, min(k, n))
        keep = sorted_probs[:k]
        keep = keep / keep.sum()
        choice = int(np.random.choice(k, p=keep))
        return int(order[choice])

    def select_move_pair(self, moves, board, player, return_scores=False, difficulty=None,
                         deadline=None):
        """
        2-ply move selection using batched GNN evaluation.

        Instead of evaluating each position individually, we:
          1. Apply each candidate move, encode the resulting position, undo
          2. Collect all encoded positions into a list
          3. Evaluate the entire batch in one forward pass
          4. Pick the best

        This reduces GPU kernel launches from ~400 to 1-2 per turn.

        Draw handling: calling a draw ((1,1,1)) is evaluated by DIRECT
        COMPARISON against its known terminal value (exactly 0), never by
        simulating it through apply_move/encode. Two reasons:
          1) board.apply_move((1,1,1)) sets board.draw_called=True WITHOUT
             pushing an entry onto board.moves, so the generic apply/undo
             search loop below can't undo it -- simulating it as a normal
             candidate would permanently and silently end the game the
             moment it's merely considered, even if not chosen.
          2) It moves no piece, so the resulting position would encode
             identically to a pass -- the network has no way to tell "end
             the game now at 0" from "do nothing and keep playing." Its
             true value is exact and known, so it doesn't need estimating.
        """
        if not isinstance(moves, (list, set)) or not all(isinstance(m, tuple) for m in moves):
            raise ValueError('Invalid moves format: expected a list or set of tuples.')

        draw_legal = (1, 1, 1) in moves
        draw_pair  = ((1, 1, 1), (0, 0, 0))

        prefilter = self.use_prefilter and self.heuristic is not None

        move_keys    = []   # list of (move1, move2) pairs
        encoded_list = []   # corresponding encoded positions (filled when NOT prefiltering)
        outcome_keys = []   # resulting piece placement per candidate, dice ignored
        scored       = []   # (heuristic_score, pair)        (filled when prefiltering)

        # Move orders transpose heavily -- about half of a midgame turn's
        # candidate pairs land on a position some other pair already reached --
        # and the heuristic prefilter is by far the most expensive part of a
        # move, so remember its verdict per position for this turn.
        eval_cache = {}

        def record(pair):
            # board is currently IN the resulting position for `pair`
            if prefilter:
                key = _position_key(board)
                s = eval_cache.get(key)
                if s is None:
                    s, _ = self.heuristic.evaluate(board, player)   # returns (score, components)
                    eval_cache[key] = s
                scored.append((s, pair))
            else:
                move_keys.append(pair)
                encoded_list.append(self.encoder.encode(board, player))
                outcome_keys.append(_piece_locs(board))

        # --- Pass move ---
        if (0, 0, 0) in moves:
            record(((0, 0, 0), (0, 0, 0)))

        moves_set = set(moves)
        moves_set.discard((0, 0, 0))
        moves_set.discard((1, 1, 1))   # draw handled separately, see docstring

        # --- Stage 1 (optional): rank first moves on their own ---------------
        # Every score here is also the score of that move's (move, pass) pair --
        # a pass changes nothing -- so stage 2 gets these back from eval_cache
        # for free.
        moves_iter = moves_set
        if prefilter and self.first_move_prefilter and len(moves_set) > self.first_move_prefilter:
            first_scored = []
            for move in moves_set:
                base = len(board.moves)
                board.apply_move(move, switch_turn=False)
                wgo, _ = board.check_game_over()
                if wgo == player:
                    while len(board.moves) > base:
                        board.undo_last_move()
                    win = (move, (0, 0, 0))
                    return [(float('inf'), win)] if return_scores else win
                key = _position_key(board)
                sc = eval_cache.get(key)
                if sc is None:
                    sc, _ = self.heuristic.evaluate(board, player)
                    eval_cache[key] = sc
                first_scored.append((sc, move))
                while len(board.moves) > base:
                    board.undo_last_move()
            # Ties broken canonically, not by set-iteration order: which first
            # moves survive the cut decides which pairs the GNN ever sees.
            first_scored.sort(key=lambda x: (-x[0], _move_sort_key(x[1])))
            keep = [m for _, m in first_scored[:self.first_move_prefilter]]
            kept = set(keep)
            # a first move that saves a piece is never culled here, for the same
            # reason save PAIRS are exempt from the top-K cull below
            keep += [m for _, m in first_scored
                     if m not in kept and isinstance(m, tuple) and m[1] == 'save']
            moves_iter = keep

        truncated = False
        for move in moves_iter:
            # Safety valve for pathological branching: keep whatever candidates
            # we have rather than let one request run away. Never trips in normal
            # play -- see MOVE_BUDGET in app.py.
            if deadline is not None and move_keys_would_exist(scored, move_keys) \
                    and time.monotonic() > deadline:
                truncated = True
                break
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
                win = (move, (0, 0, 0))
                return [(float('inf'), win)] if return_scores else win

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
            next_moves.discard((1, 1, 1))   # draw is only ever legal as a lone first action

            for next_move in next_moves:
                if not isinstance(next_move, tuple) or len(next_move) != 3:
                    raise ValueError('Invalid next move format.')
                board.apply_move(next_move, switch_turn=False)
                wgo, _ = board.check_game_over()
                if wgo == player:
                    while len(board.moves) > initial_move_count:
                        board.undo_last_move()
                    win = (move, next_move)
                    return [(float('inf'), win)] if return_scores else win
                record((move, next_move))
                board.undo_last_move()

            while len(board.moves) > initial_move_count:
                board.undo_last_move()

        # --- If prefiltering: pick the kept candidates, then encode ONLY those ---
        if prefilter:
            if not scored:
                if draw_legal:
                    return [(0.0, draw_pair)] if return_scores else draw_pair
                return ((0, 0, 0), (0, 0, 0))
            # Save pairs are exempt from the heuristic top-K cull: the heuristic
            # can undervalue saves relative to flashier non-save moves, which
            # would otherwise silently drop a legal save before the GNN ever
            # sees it. Always keep save-containing pairs; cull only the rest.
            save_scored  = [(s, p) for s, p in scored if self._pair_has_save(p)]
            other_scored = [(s, p) for s, p in scored if not self._pair_has_save(p)]
            top_pairs = [p for _, p in save_scored] + self._select_filtered(other_scored)
            self.dbg_kept_total += len(top_pairs)
            self.dbg_kept_calls += 1
            for pair in top_pairs:
                base = len(board.moves)
                for m in pair:
                    if m != (0, 0, 0):
                        board.apply_move(m, switch_turn=False)
                move_keys.append(pair)
                encoded_list.append(self.encoder.encode(board, player))
                outcome_keys.append(_piece_locs(board))
                while len(board.moves) > base:
                    board.undo_last_move()

        if truncated:
            import logging
            logging.getLogger('agent_gnn').warning(
                'move budget hit: scored %d of %d first moves',
                len(scored) or len(move_keys), len(moves_iter))

        if not move_keys:
            if draw_legal:
                return [(0.0, draw_pair)] if return_scores else draw_pair
            return ((0, 0, 0), (0, 0, 0))

        # Forced move: one candidate and no draw option to value it against --
        # the argmax is predetermined, skip the model forward. Must still run
        # the same postprocessing tail as the scored path. (Draw-legal and
        # return_scores paths still need the actual score.)
        if len(move_keys) == 1 and not draw_legal and not return_scores:
            chosen = move_keys[0]
            if self.enable_never_good:
                chosen = self._fix_never_good(chosen, board, player)
            return self._dedupe_save_pair(chosen)

        # --- Single batched forward pass over the (filtered) candidates ---
        scores = self.model(encoded_list)   # [N]
        final_scores = scores * SCORE_SCALE
        best_idx     = final_scores.argmax().item()

        if return_scores:
            ranked = list(zip(final_scores.tolist(), move_keys))
            if draw_legal:
                ranked.append((0.0, draw_pair))   # exact terminal value, not network-estimated
            ranked.sort(key=lambda x: (-x[0], _pair_sort_key(x[1])))
            return ranked

        # A draw's true value is exactly 0, in the same raw*SCORE_SCALE units
        # as final_scores -- compare directly, no encoding needed. (Use the true
        # best, not a difficulty-sampled pick, so lowering difficulty never makes
        # the agent resign a game it shouldn't.)
        if draw_legal and 0.0 >= final_scores[best_idx].item():
            return draw_pair

        # Difficulty: at full strength take the argmax; below that, sample among
        # the top candidates (top-p over a scale-invariant softmax) so the agent
        # plays visibly weaker without ever making obviously terrible moves.
        best_idx = self._pick_move_index(final_scores, difficulty)

        # Exact score ties (transpositions reaching the same position score
        # identically) used to be settled by whichever candidate the set
        # happened to enumerate first. Settle them canonically instead; this
        # runs after the difficulty pick, so it makes the tie deterministic
        # without overriding the sampled choice.
        top = float(final_scores[best_idx])
        tied = [i for i, s in enumerate(final_scores) if float(s) == top]
        if len(tied) > 1:
            best_idx = min(tied, key=lambda i: _pair_sort_key(move_keys[i]))

        # Among candidates that leave the board in exactly the same state, take
        # the one that moves the fewest pieces. This costs nothing -- the
        # position after the turn is identical and a die left unused is worthless
        # once the turn ends -- and it stops the agent walking a blank piece from
        # goal 4 round to goal 2 to save it there when it could have saved it
        # where it stood. (The scores differ only because "both dice used" is an
        # input feature, which has no consequence at the end of a turn.)
        if len(outcome_keys) == len(move_keys):
            key = outcome_keys[best_idx]
            same = [i for i, k in enumerate(outcome_keys) if k == key]
            if len(same) > 1:
                best_idx = min(same, key=lambda i: (_pair_relocations(move_keys[i]),
                                                    _pair_sort_key(move_keys[i])))

        chosen = move_keys[best_idx]

        # --- Diagnostic: localize pass-over-save -------------------------------
        # Fires when the chosen pair saves nothing but a save WAS available among
        # the GNN-scored candidates (move_keys). Tells us definitively whether
        # the save reached the value head and was simply scored lower (value
        # error) vs. never made it into the candidate set (prefilter/enumeration).
        if getattr(self, 'debug_pass_over_save', False):
            def _has_save(p):
                return any(isinstance(m, tuple) and len(m) == 3 and m[1] == 'save' for m in p)
            chosen_saves = _has_save(chosen)
            # was a save legally available at top level at all?
            save_available = any(isinstance(m, tuple) and len(m) == 3 and m[1] == 'save'
                                 for m in moves)
            if save_available and not chosen_saves:
                import logging
                log = logging.getLogger('agent_gnn')
                save_idxs = [i for i, p in enumerate(move_keys) if _has_save(p)]
                chosen_score = final_scores[best_idx].item()
                if save_idxs:
                    best_save_i = max(save_idxs, key=lambda i: final_scores[i].item())
                    log.info(
                        "[pass_over_save] VALUE-ERROR: save was scored but not chosen. "
                        f"chosen={chosen} score={chosen_score:.4f} | "
                        f"best_save={move_keys[best_save_i]} "
                        f"score={final_scores[best_save_i].item():.4f} | "
                        f"{len(save_idxs)} save pairs among {len(move_keys)} scored")
                else:
                    log.info(
                        "[pass_over_save] CANDIDATE-DROP: a save was legal but NO save "
                        f"pair reached the value head. chosen={chosen} | "
                        f"prefilter={'on' if prefilter else 'off'} "
                        f"scored={len(scored) if prefilter else 'n/a'} "
                        f"kept={len(move_keys)}")
        # -----------------------------------------------------------------------

        if self.enable_never_good:
            chosen = self._fix_never_good(chosen, board, player)
        return self._dedupe_save_pair(chosen)

    @staticmethod
    def _pair_has_save(pair):
        return any(isinstance(m, tuple) and len(m) == 3 and m[1] == 'save' for m in pair)

    def _dedupe_save_pair(self, pair):
        """A piece can be saved at most once per turn. If a pair tries to save
        the same piece-id twice (e.g. two halves that resolved to the same
        numbered piece), drop the second save to a pass. Defensive invariant
        enforced on every returned pair."""
        if not (isinstance(pair, tuple) and len(pair) == 2):
            return pair
        m1, m2 = pair
        def save_pid(m):
            if isinstance(m, tuple) and len(m) == 3 and m[1] == 'save':
                return m[0]
            return None
        p1, p2 = save_pid(m1), save_pid(m2)
        if p1 is not None and p1 == p2:
            return (m1, (0, 0, 0))
        return pair

    def _note_never_good(self, rule, before, after, fired=True):
        """Record (and log) a never-good correction. Always stashes the last
        event on self.last_never_good and increments never_good_counts (readable
        from any process). Emits a logging line so it shows up in the Flask
        server terminal during live play WITHOUT needing a debug flag set:
          - a correction that FIRED -> INFO
          - a pass-pair where no save was found -> DEBUG (only with debug flag)
        Set self.debug_never_good=True for the extra near-miss/no-op lines."""
        self.last_never_good = {'rule': rule, 'before': before, 'after': after, 'fired': fired}
        if fired:
            self.never_good_counts[rule] = self.never_good_counts.get(rule, 0) + 1
        import logging
        log = logging.getLogger('agent_gnn')
        if fired:
            log.info(f"[never_good] {rule} FIRED: {before} -> {after}")
        elif getattr(self, 'debug_never_good', False):
            log.info(f"[never_good] {rule}: pass pair {before} but no save found")

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
        # Normalize to tuples — pairs/moves may arrive as lists from some paths,
        # and `m == (0,0,0)` / isinstance(...tuple) checks silently fail on lists.
        def _t(x):
            return tuple(_t(e) for e in x) if isinstance(x, (list, tuple)) else x
        pair = _t(pair)

        if not (isinstance(pair, tuple) and len(pair) == 2):
            return pair
        m1, m2 = pair

        def is_pass(m):  return m == (0, 0, 0)
        def is_save(m):  return isinstance(m, tuple) and len(m) == 3 and m[1] == 'save'
        def is_num(pid): return isinstance(pid, tuple) and len(pid) == 2 and pid[1] <= 6

        # ---- Rule 1: numbered save dominates unnumbered save from same goal ----
        # `already_used` tracks numbered pieces already claimed by a save in THIS
        # pair, so an unnumbered save is never redirected onto a piece that is
        # already being saved (which would create a duplicate that then collapses
        # to a pass, dropping a legitimate save). Seed it with the numbered
        # pieces the pair already saves.
        already_used = set()
        for m in (m1, m2):
            if is_save(m) and is_num(m[0]):
                already_used.add(m[0])

        def upgrade(save_move):
            if not is_save(save_move) or is_num(save_move[0]):
                return save_move
            pl, _num = save_move[0]
            roll = save_move[2]
            piece = board.piece_lookup.get((pl, _num))
            if not piece or not piece.tile:
                return save_move
            for p in piece.tile.pieces:                 # same goal tile
                if (p.player == player and p.number <= 6 and p.number == roll
                        and (p.player, p.number) not in already_used):
                    already_used.add((p.player, p.number))
                    return ((p.player, p.number), 'save', roll)
            return save_move

        u1, u2 = upgrade(m1), upgrade(m2)
        if (u1, u2) != (m1, m2):
            fixed = self._dedupe_save_pair((u1, u2))
            self._note_never_good('rule1_upgrade_numbered', pair, fixed)
            return fixed

        # ---- Rule 2: never pass when a save is available ----
        def best_save(valid, exclude_pid=None):
            saves = [mv for mv in valid if is_save(mv) and mv[0] != exclude_pid]
            if not saves:
                return None
            saves.sort(key=lambda mv: 0 if is_num(mv[0]) else 1)   # prefer numbered
            return saves[0]

        # Identify which slot (if any) is a pass and what the non-pass move is.
        p1, p2 = is_pass(m1), is_pass(m2)

        result = pair
        rule = None
        if p1 and p2:
            s = best_save(board.get_valid_moves())
            if s is not None:
                result = (s, (0, 0, 0))
                rule = 'rule2_fill_double_pass'
        elif p1 ^ p2:
            real = m2 if p1 else m1            # the non-pass move
            base = len(board.moves)
            board.apply_move(real, switch_turn=False)
            # exclude the piece just moved/saved so we never re-save the same id
            exclude = real[0] if is_save(real) else None
            s = best_save(board.get_valid_moves(), exclude_pid=exclude)
            while len(board.moves) > base:
                board.undo_last_move()
            if s is not None:
                result = (real, s)             # canonical order: real move first
                rule = 'rule2_fill_pass'

        final = self._dedupe_save_pair(result)
        if rule is not None:
            self._note_never_good(rule, pair, final)
        elif (p1 or p2):
            # A pass slot existed but no save was found to fill it — log when
            # debugging so we can see "saw a pass pair, found no save" cases.
            self._note_never_good('rule2_no_save_found', pair, final, fired=False)
        return final

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
        # Ties by canonical move key, not enumeration order -- see _move_sort_key.
        scored = sorted(scored, key=lambda x: (-x[0], _pair_sort_key(x[1])))

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

        # Draw handled by direct comparison against its known value (0), never
        # simulated -- see the docstring in select_move_pair for why apply_move
        # can't safely be used for (1,1,1).
        draw_legal = (1, 1, 1) in moves
        draw_pair  = ((1, 1, 1), (0, 0, 0))

        moves_set = set(moves)

        # Handle pass-only case
        if moves_set == {(0, 0, 0)}:
            return ((0, 0, 0), (0, 0, 0))

        moves_set.discard((0, 0, 0))
        moves_set.discard((1, 1, 1))

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
            return draw_pair if draw_legal else ((0, 0, 0), (0, 0, 0))

        # Evaluate all first moves in one batch
        first_scores = self.model(first_encoded_list) * SCORE_SCALE  # [N]
        best_first_idx = first_scores.argmax().item()

        # A draw's true value is exactly 0 -- compare directly, no encoding.
        if draw_legal and 0.0 >= first_scores[best_first_idx].item():
            return draw_pair

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

        next_moves = set(board.get_valid_moves()) - {(0, 0, 0), (1, 1, 1)}

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
        second_scores = self.model(second_encoded_list) * SCORE_SCALE  # [M]
        best_second = second_move_keys[second_scores.argmax().item()]
        return (best_first, best_second)

    def select_move_pair_beam(self, moves, board, player, K=2):

        # Draw handled by direct comparison against its known value (0), never
        # simulated -- see the docstring in select_move_pair for why apply_move
        # can't safely be used for (1,1,1).
        draw_legal = (1, 1, 1) in moves
        draw_pair  = ((1, 1, 1), (0, 0, 0))

        valid_moves = [m for m in moves if m not in ((0, 0, 0), (1, 1, 1))]
        if not valid_moves:
            return draw_pair if draw_legal else ((0, 0, 0), (0, 0, 0))

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

        values = self.model(first_states).reshape(-1)

        # A draw's true value is exactly 0 -- compare directly against the
        # best encoded first-move value before running the (expensive)
        # beam search over second moves.
        if draw_legal and 0.0 >= values.max().item():
            return draw_pair

        topk = _top_indices(values, K)

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

            second_moves = list(set(board.get_valid_moves()) - {(0, 0, 0), (1, 1, 1)})

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

            vals2 = self.model(second_states).reshape(-1)

            best_idx = int(np.argmax(vals2))
            val = vals2[best_idx].item()

            if val > best_value:
                best_value = val
                best_pair = (m1, second_meta[best_idx])

            while len(board.moves) > initial_len:
                board.undo_last_move()

        return best_pair if best_pair else ((0,0,0),(0,0,0))
    # ------------------------------------------------------------------
    # Deep search: one opponent ply over the dice chance node
    # ------------------------------------------------------------------
    #
    # select_move_pair evaluates V(position after my pair) with the net.
    # select_move_pair_deep replaces that leaf value, for the top-k_me
    # candidate pairs, with a one-opponent-ply expectiminimax backup:
    #
    #     E[my value] = sum over the opponent's 21 possible dice rolls of
    #                   P(roll) * ( - V(position after the opponent's greedy
    #                                   reply pair, from the opponent's frame) )
    #
    # Frames/phases are chosen so every net evaluation is exactly the phase
    # the net was trained on ("mover's pair just completed, encoded from the
    # mover's perspective"): root leaves are (after my pair, my frame);
    # depth-2 leaves are (after opponent's reply, opponent's frame), negated.
    # Terminal positions use the training-target convention directly:
    # score/NUM_PIECES in the winner's frame (draws exactly 0), so learned
    # and exact values live on one comparable scale (raw net units).
    #
    # The opponent reply is a beam search over first moves (beam_k=3, full
    # batch) x full second-move batch per beam entry -- same structure as
    # select_move_pair_beam, i.e. a 1-ply shallow opponent -- deliberately
    # NOT a recursive deep opponent (cost) and NOT full pair enumeration
    # (measured prohibitive: ~42s/move at k_me=8; see CLAUDE.md).
    #
    # The LIVE path computes all 21 rolls at once via
    # _opp_reply_values_batched (per-die first-move dedup + dice-scalar
    # global-feature patching + a transposition cache on second-move
    # leaves + exactly two forward passes per candidate, instead of ~84
    # small ones). _opp_greedy_reply_raw is the sequential per-roll
    # REFERENCE ORACLE the batched path is validated against.
    #
    # An earlier version used select_move_pair_fast's two-stage GREEDY
    # (commit to the single best first move before ever looking at second
    # moves) to model the opponent here. That is explicitly documented on
    # select_move_pair_fast as unsuitable for interactive play (it can trap
    # on a first move whose own score is highest but whose forced follow-up
    # is poor) -- and using it to model the opponent made deep search
    # systematically underestimate the opponent's replies, which surfaced
    # as deep LOSING to plain shallow search at the same weights in a
    # 20-game match. Beam search (beam_k=3) fixed it -- see
    # _opp_greedy_reply_raw's docstring for the failure mode in detail.

    # The 21 unordered dice rolls with their probabilities (dice are
    # interchangeable in this game's move legality, so (a,b)~(b,a)).
    _DICE_ROLLS_21 = [(d1, d2, (1.0 if d1 == d2 else 2.0) / 36.0)
                      for d1 in range(1, 7) for d2 in range(d1, 7)]

    def _raw_value(self, board, player):
        """Net value of the current position in raw units from `player`'s
        frame (no SCORE_SCALE)."""
        return float(self.model(self.encoder.encode(board, player)))
    def _enter_opponent_turn_deterministic(self, board):
        """Reproduce switch_turn's side effects reversibly and WITHOUT
        rolling dice (caller sets dice per chance-node branch afterwards).
        Returns an opaque snapshot for _restore_turn_state. Must be called
        with the acting player's pair already applied.

        Side effects mirrored from switch_turn (game.py): cache clears,
        no-save/draw counter update, firstMove reset, player flip, and the
        last-piece rule. The last-piece rule PERMANENTLY renumbers a
        player's final numbered piece to unnumbered (piece.number ->
        NUM_PIECES+1) and rewrites piece_lookup -- there is no undo path
        for it in the engine, so we track exactly what it changed and
        reverse it by hand in _restore_turn_state."""
        saved = {
            'player': board.current_player,
            'dice': [(d.number, d.used) for d in board.dice],
            'firstMove': board.firstMove,
            'no_save_turns': board.no_save_turns,
            'half_turns': board._half_turns_since_round,
            'last_total_saved': board._last_total_saved,
            'draw_callable': board.draw_callable,
            'game_stages': dict(board.game_stages),
            'moves_len': len(board.moves),
        }
        board._distance_cache.clear()
        board._blot_cache.clear()
        board._reachable_cache.clear()
        board._blocked_key_cache.clear()
        board.update_no_save_counter()
        board.firstMove = None
        board.current_player = 'white' if board.current_player == 'black' else 'black'
        # last-piece rule, tracked for manual reversal
        before = {id(p): p.number for p in board.pieces if p.number <= 6}
        board.apply_last_piece_rule()
        saved['renumbered'] = [(p, before[id(p)]) for p in board.pieces
                               if id(p) in before and p.number != before[id(p)]]
        return saved

    def _restore_turn_state(self, board, saved):
        """Exactly reverse _enter_opponent_turn_deterministic. The caller
        must already have undone any opponent moves it applied (so
        len(board.moves) == saved['moves_len'] and the dice in play are the
        simulated opponent dice, about to be overwritten)."""
        assert len(board.moves) == saved['moves_len'], \
            "opponent moves not fully undone before turn-state restore"
        for p, old_num in saved['renumbered']:
            new_key = (p.player, p.number)
            if board.piece_lookup.get(new_key) is p:
                del board.piece_lookup[new_key]
            p.number = old_num
            board.piece_lookup[(p.player, old_num)] = p
        board.current_player = saved['player']
        for d, (num, used) in zip(board.dice, saved['dice']):
            d.number = num
            d.used = used
        board.firstMove = saved['firstMove']
        board.no_save_turns = saved['no_save_turns']
        board._half_turns_since_round = saved['half_turns']
        board._last_total_saved = saved['last_total_saved']
        board.draw_callable = saved['draw_callable']
        board.game_stages.update(saved['game_stages'])
        board._distance_cache.clear()
        board._blot_cache.clear()
        board._reachable_cache.clear()
        board._blocked_key_cache.clear()

    def _opp_greedy_reply_raw(self, board, opp, beam_k=3):
        """REFERENCE ORACLE (sequential, one roll per call) for the batched
        implementation below -- kept for validation, not used in the live
        deep-search path. With `board` at the start of the opponent's
        simulated turn (current_player == opp, dice set to one roll, both
        unused), find the opponent's best PAIR-WISE reply (beam over first
        moves, beam_k, full second-move batch per beam entry) and return
        the raw net value of the resulting position FROM THE OPPONENT'S
        FRAME. All applied moves are undone before returning.

        Semantics (mirrored exactly by _opp_reply_values_batched):
        - Terminal WINS for the mover short-circuit by depth: any winning
          first move makes the value the max such win; else any winning
          second move (across all expanded beam entries) does. Max, not
          first-found, so the result is enumeration-order-independent.
        - Otherwise: max over stand-pat (if passing is legal or no moves),
          stop-after-first (where legal), and second-move values.
        - A legal draw call floors the non-win result at exactly 0.
        - Stand-pat is evaluated PER ROLL with the roll's dice numbers set:
          the encoding includes the dice numbers (global feats [0]/[1]),
          so an unused 6 and an unused 1 are different positions to the
          net. (An earlier version computed stand-pat once for all 21
          rolls on the wrong assumption that only the used-flags are
          encoded.)"""
        base_len = len(board.moves)
        moves = board.get_valid_moves()
        draw_legal = (1, 1, 1) in moves
        cand = [m for m in moves if m not in ((0, 0, 0), (1, 1, 1))
                and isinstance(m, tuple) and len(m) == 3]
        pass_legal = (0, 0, 0) in moves

        def undo_to(n):
            while len(board.moves) > n:
                board.undo_last_move()

        best_v = None
        if pass_legal or not cand:
            # doing nothing is an option (or forced): value of standing pat
            best_v = self._raw_value(board, opp)
        if cand:
            # stage 1: apply every first move once; collect terminal wins
            # and encodings for the rest
            win1 = None
            encs, keys = [], []
            for m in cand:
                board.apply_move(m, switch_turn=False)
                w, s = board.check_game_over()
                if w == opp:
                    v = float(s) / NUM_PIECES
                    win1 = v if win1 is None else max(win1, v)
                else:
                    encs.append(self.encoder.encode(board, opp))
                    keys.append(m)
                undo_to(base_len)
            if win1 is not None:
                return win1
            v1 = self.model(encs).reshape(-1)
            topk = _top_indices(v1, beam_k)

            win2 = None
            for b1 in topk:
                v_after_first = float(v1[b1].item())
                board.apply_move(keys[b1], switch_turn=False)
                if all(d.used for d in board.dice):
                    # complete turns (block-saves consume both dice)
                    if best_v is None or v_after_first > best_v:
                        best_v = v_after_first
                    undo_to(base_len)
                    continue
                after_first = board.get_valid_moves()
                second = [m for m in after_first
                          if m not in ((0, 0, 0), (1, 1, 1))
                          and isinstance(m, tuple) and len(m) == 3]
                # Stopping after one move is only a real option if a pass is
                # legal AT THIS POINT (e.g. not with captured pieces still to
                # enter) -- mirror the shallow search's gating, don't assume.
                stop_allowed = (0, 0, 0) in after_first
                if stop_allowed or not second:
                    if best_v is None or v_after_first > best_v:
                        best_v = v_after_first
                if second:
                    encs2 = []
                    mid_len = len(board.moves)
                    for m2 in second:
                        board.apply_move(m2, switch_turn=False)
                        w, s = board.check_game_over()
                        if w == opp:
                            v = float(s) / NUM_PIECES
                            win2 = v if win2 is None else max(win2, v)
                        else:
                            encs2.append(self.encoder.encode(board, opp))
                        undo_to(mid_len)
                    if encs2:
                        v2 = self.model(encs2).reshape(-1)
                        v_pair = float(v2.max().item())
                        if best_v is None or v_pair > best_v:
                            best_v = v_pair
                undo_to(base_len)
            if win2 is not None:
                return win2
        if draw_legal and best_v < 0.0:
            best_v = 0.0    # opponent prefers the exact-0 draw
        return best_v

    def _position_key(self, board):
        """Cheap transposition key: sorted piece placements. Two board
        states with equal keys encode identically here (stages/blot/
        distance features are all functions of placement, and encode
        computes them fresh) -- dice are NOT in the key; callers add the
        dice context themselves."""
        ws, bs = board.white_saved, board.black_saved
        wu, bu = board.white_unentered, board.black_unentered
        parts = []
        for p in board.pieces:
            if p.tile is not None:
                loc = (p.tile.ring, p.tile.pos)
            elif p.rack is ws or p.rack is bs:
                loc = 's'
            elif p.rack is wu or p.rack is bu:
                loc = 'u'
            else:
                loc = 'h'
            parts.append((p.player, p.number, loc))
        parts.sort()
        return tuple(parts)

    def _dice_variant(self, enc, d1, d2, u1, u2):
        """Re-dice an encoding without re-encoding: the ONLY dice-dependent
        features are global feats [0]=die1/6, [1]=die2/6, [2]=die1_used,
        [3]=die2_used, so a variant shares every expensive tensor
        (tile/piece features, edges) by reference and clones only the
        11-float global vector."""
        g = enc['global_feats'].clone()
        g[0] = d1 / 6.0
        g[1] = d2 / 6.0
        g[2] = u1
        g[3] = u2
        out = dict(enc)
        out['global_feats'] = g
        return out

    def _opp_reply_values_batched(self, board, opp, beam_k=3):
        """All 21 rolls' opponent-reply values in one call -- semantically
        identical to calling _opp_greedy_reply_raw once per roll (the
        equivalence test enforces this), restructured around TWO batched
        forward passes instead of ~84 small ones:

        Phase A -- first moves are enumerated once PER DIE VALUE (a single
        move's legality and resulting position depend only on its own die,
        never the other's number) and base-encoded once; each roll's
        candidate set is then assembled as per_die[d1] u per_die[d2] (u
        block-saves and stand-pat, both roll-independent), with the dice
        scalars patched into a cloned global vector per roll
        (_dice_variant). One forward pass covers every (roll, first-move)
        variant, all 21 stand-pat variants, and all block-save variants:
        ~6x less enumeration/encoding than per-roll, and 1 dispatch.

        Phase B -- for each roll not already decided by a first-move win:
        take the top beam_k first moves by Phase A value, enumerate their
        second moves with the real engine (this part is genuinely
        per-roll: sum-move filtering depends on both dice), dedup
        resulting positions through a transposition cache keyed by
        (_position_key, roll) -- transposed (m1,m2)/(m2,m1) pairs collapse
        here -- and run ONE forward pass over the novel leaves.

        Caller must have the opponent's turn entered (current_player ==
        opp). Dice are left dirty; _restore_turn_state puts them back."""
        base_len = len(board.moves)
        dice = board.dice

        def undo_to(n):
            while len(board.moves) > n:
                board.undo_last_move()

        # ---- roll-independent facts + per-die enumeration ----
        # get_valid_moves adds (1,1,1) iff both dice unused and
        # draw_callable, and (0,0,0)/block-saves on conditions that don't
        # read dice numbers -- all roll-independent at turn start.
        draw_legal = bool(board.draw_callable)
        per_die = {}
        pass_legal = False
        block_saves = []
        for d in range(1, 7):
            dice[0].number = d; dice[0].used = False
            dice[1].number = d; dice[1].used = False
            mv = board.get_valid_moves()
            pass_legal = (0, 0, 0) in mv
            if not block_saves:
                block_saves = [m for m in mv
                               if isinstance(m, tuple) and len(m) == 3
                               and m != (0, 0, 0) and m != (1, 1, 1)
                               and m[1] == 0 and m[2] == 0]
            per_die[d] = [m for m in mv
                          if isinstance(m, tuple) and len(m) == 3
                          and m not in ((0, 0, 0), (1, 1, 1))
                          and m[2] == d]

        # ---- Phase A base encodes ----
        dice[0].used = False; dice[1].used = False
        sp_enc = self.encoder.encode(board, opp)          # stand-pat base
        fm_enc, fm_win = {}, {}
        for d in range(1, 7):
            dice[0].number = d; dice[0].used = False
            dice[1].number = d; dice[1].used = False
            for m in per_die[d]:
                board.apply_move(m, switch_turn=False)
                w, s = board.check_game_over()
                if w == opp:
                    fm_win[(d, m)] = float(s) / NUM_PIECES
                else:
                    fm_enc[(d, m)] = self.encoder.encode(board, opp)
                undo_to(base_len)
        bs_enc, bs_win = {}, {}
        for m in block_saves:
            board.apply_move(m, switch_turn=False)        # marks both dice
            w, s = board.check_game_over()
            if w == opp:
                bs_win[m] = float(s) / NUM_PIECES
            else:
                bs_enc[m] = self.encoder.encode(board, opp)
            undo_to(base_len)

        # ---- Phase A variant assembly + single forward ----
        A_encs = []
        roll_entries = [[] for _ in self._DICE_ROLLS_21]  # (A-index|None, kind, d, m)
        roll_win1 = [None] * len(self._DICE_ROLLS_21)
        sp_idx = [None] * len(self._DICE_ROLLS_21)
        for ri, (d1, d2, _w) in enumerate(self._DICE_ROLLS_21):
            sp_idx[ri] = len(A_encs)
            A_encs.append(self._dice_variant(sp_enc, d1, d2, 0.0, 0.0))
            slots = [(d1, 1.0, 0.0)] if d1 == d2 else \
                    [(d1, 1.0, 0.0), (d2, 0.0, 1.0)]
            for d, u1, u2 in slots:
                for m in per_die[d]:
                    if (d, m) in fm_win:
                        v = fm_win[(d, m)]
                        roll_win1[ri] = v if roll_win1[ri] is None \
                            else max(roll_win1[ri], v)
                    else:
                        roll_entries[ri].append((len(A_encs), 'fm', d, m))
                        A_encs.append(self._dice_variant(
                            fm_enc[(d, m)], d1, d2, u1, u2))
            for m in block_saves:
                if m in bs_win:
                    v = bs_win[m]
                    roll_win1[ri] = v if roll_win1[ri] is None \
                        else max(roll_win1[ri], v)
                else:
                    roll_entries[ri].append((len(A_encs), 'bs', 0, m))
                    A_encs.append(self._dice_variant(
                        bs_enc[m], d1, d2, 1.0, 1.0))
        A_vals = self.model(A_encs).reshape(-1)
        # ---- Phase B: expand beams, dedup leaves, single forward ----
        tcache = {}
        B_encs = []
        roll_best = [None] * len(self._DICE_ROLLS_21)     # running float max
        roll_pending = [[] for _ in self._DICE_ROLLS_21]  # B indices
        roll_win2 = [None] * len(self._DICE_ROLLS_21)
        for ri, (d1, d2, _w) in enumerate(self._DICE_ROLLS_21):
            if roll_win1[ri] is not None:
                continue                                   # decided at depth 1
            entries = roll_entries[ri]
            if pass_legal or not entries:
                roll_best[ri] = float(A_vals[sp_idx[ri]].item())
            if not entries:
                continue
            ev = [float(A_vals[i].item()) for i, _k, _d, _m in entries]
            order = sorted(range(len(entries)), key=lambda i: ev[i],
                           reverse=True)[:min(beam_k, len(entries))]
            for oi in order:
                idx, kind, d, m = entries[oi]
                v_first = ev[oi]
                if kind == 'bs':
                    # complete turn: the block-save IS the pair
                    if roll_best[ri] is None or v_first > roll_best[ri]:
                        roll_best[ri] = v_first
                    continue
                dice[0].number = d1; dice[0].used = False
                dice[1].number = d2; dice[1].used = False
                board.apply_move(m, switch_turn=False)
                after_first = board.get_valid_moves()
                second = [x for x in after_first
                          if x not in ((0, 0, 0), (1, 1, 1))
                          and isinstance(x, tuple) and len(x) == 3]
                stop_allowed = (0, 0, 0) in after_first
                if stop_allowed or not second:
                    if roll_best[ri] is None or v_first > roll_best[ri]:
                        roll_best[ri] = v_first
                mid_len = len(board.moves)
                for m2 in second:
                    board.apply_move(m2, switch_turn=False)
                    w, s = board.check_game_over()
                    if w == opp:
                        v = float(s) / NUM_PIECES
                        roll_win2[ri] = v if roll_win2[ri] is None \
                            else max(roll_win2[ri], v)
                    else:
                        key = (self._position_key(board), d1, d2)
                        slot = tcache.get(key)
                        if slot is None:
                            slot = len(B_encs)
                            tcache[key] = slot
                            B_encs.append(self.encoder.encode(board, opp))
                        roll_pending[ri].append(slot)
                    undo_to(mid_len)
                undo_to(base_len)
        if B_encs:
            B_vals = self.model(B_encs).reshape(-1)
        # ---- assemble the 21 values ----
        out = []
        for ri in range(len(self._DICE_ROLLS_21)):
            if roll_win1[ri] is not None:
                out.append(roll_win1[ri])
                continue
            if roll_win2[ri] is not None:
                out.append(roll_win2[ri])
                continue
            v = roll_best[ri]
            for slot in roll_pending[ri]:
                bv = float(B_vals[slot].item())
                if v is None or bv > v:
                    v = bv
            if draw_legal and v < 0.0:
                v = 0.0
            out.append(v)
        return out

    def select_move_pair_deep(self, moves, board, player, k_me=8,
                              return_scores=False):
        """Move selection with one opponent ply over the dice chance node.

        1. Rank all my move pairs with the existing shallow search.
        2. For the top k_me pairs: apply the pair, deterministically enter
           the opponent's turn, and for each of the 21 dice rolls let the
           opponent make its greedy reply; my deep value is the
           probability-weighted mean of the negated reply values.
        3. Pick the pair with the best deep value (draw compared at its
           exact 0, same rule as the shallow search), then run the same
           postprocessing tail (never-good fixes, save dedupe).

        Intended for PLAY/EVAL ONLY -- generation keeps the shallow path.
        """
        # game_stages hygiene: undo_last_move recomputes a player's stage
        # only when undoing save/rack moves, so unwinding a mixed pair (e.g.
        # field-move + save) can leave a slot reflecting an INTERMEDIATE
        # position's stage. Scores no longer read the dict (encode computes
        # stages fresh), but restore it anyway so a deep call is externally
        # side-effect-free.
        stages0 = dict(board.game_stages)
        try:
            ranked = self.select_move_pair(moves, board, player, return_scores=True)
            if isinstance(ranked, tuple):    # defensive: shallow returned a pair
                return ranked
            if not ranked:
                return ((0, 0, 0), (0, 0, 0))
            draw_pair = ((1, 1, 1), (0, 0, 0))
            if ranked[0][0] == float('inf'):  # guaranteed win found shallowly
                return ranked[0][1]
            draw_legal = any(p == draw_pair for _, p in ranked)
            cands = [p for _, p in ranked if p != draw_pair][:k_me]
            if not cands:
                return draw_pair if draw_legal else ((0, 0, 0), (0, 0, 0))
            if len(cands) == 1 and not draw_legal and not return_scores:
                chosen = cands[0]            # forced move: nothing to compare
                if self.enable_never_good:
                    chosen = self._fix_never_good(chosen, board, player)
                return self._dedupe_save_pair(chosen)

            opp = 'white' if player == 'black' else 'black'
            deep_scored = []
            root_len = len(board.moves)
            for pair in cands:
                for m in pair:
                    if m != (0, 0, 0):
                        board.apply_move(m, switch_turn=False)
                saved = self._enter_opponent_turn_deterministic(board)
                vals = self._opp_reply_values_batched(board, opp)
                exp_v = sum(w * (-v) for (_d1, _d2, w), v
                            in zip(self._DICE_ROLLS_21, vals))
                self._restore_turn_state(board, saved)
                while len(board.moves) > root_len:
                    board.undo_last_move()
                deep_scored.append((exp_v, pair))

            deep_scored.sort(key=lambda x: x[0], reverse=True)
            if return_scores:
                out = list(deep_scored)
                if draw_legal:
                    out.append((0.0, draw_pair))
                    out.sort(key=lambda x: x[0], reverse=True)
                return out
            best_v, chosen = deep_scored[0]
            if draw_legal and 0.0 >= best_v:
                return draw_pair
            if self.enable_never_good:
                chosen = self._fix_never_good(chosen, board, player)
            return self._dedupe_save_pair(chosen)
        finally:
            board.game_stages.update(stages0)
