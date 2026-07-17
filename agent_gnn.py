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
NUM_PIECES      = 12       # margin display unit (raw * NUM_PIECES = expected margin)
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
            print(f"GNNAgent ready on {next(self.model.parameters()).device}")
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
        with torch.no_grad():
            raw_score = self.model(encoded).item()

        final_score = raw_score * SCORE_SCALE
        return final_score, {'gnn_raw': raw_score, 'gnn_score': final_score, '_player': player}

    def select_move_pair(self, moves, board, player, return_scores=False):
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
        moves_set.discard((1, 1, 1))   # draw handled separately, see docstring

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
                while len(board.moves) > base:
                    board.undo_last_move()

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
        with torch.no_grad():
            scores = self.model(encoded_list)   # [N]

        final_scores = scores * SCORE_SCALE
        best_idx     = final_scores.argmax().item()

        if return_scores:
            ranked = list(zip(final_scores.tolist(), move_keys))
            if draw_legal:
                ranked.append((0.0, draw_pair))   # exact terminal value, not network-estimated
            ranked.sort(key=lambda x: x[0], reverse=True)
            return ranked

        # A draw's true value is exactly 0, in the same raw*SCORE_SCALE units
        # as final_scores -- compare directly, no encoding needed.
        if draw_legal and 0.0 >= final_scores[best_idx].item():
            return draw_pair

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
        with torch.no_grad():
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
        with torch.no_grad():
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

            # A draw's true value is exactly 0 -- compare directly against the
            # best encoded first-move value before running the (expensive)
            # beam search over second moves.
            if draw_legal and 0.0 >= values.max().item():
                return draw_pair

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

                vals2 = self.model(second_states).view(-1)

                best_idx = torch.argmax(vals2).item()
                val = vals2[best_idx].item()

                if val > best_value:
                    best_value = val
                    best_pair = (m1, second_meta[best_idx])

                while len(board.moves) > initial_len:
                    board.undo_last_move()

            return best_pair if best_pair else ((0,0,0),(0,0,0))