# "best_weights copy.json" is backup for a seemingly good set of weights
# 
import json
import os
import copy
import math


GAME_OVER_SCORE = 10000
LOG_TO_FILE = True

INITIAL_WEIGHTS = {
    'saved_bonuses': {'a': 18.0, 'b': 1.1}, # a = value for piece 1, b = exponent
    'goal_bonuses': {'a': 38.0, 'b': 1.05},
    'near_goal_bonuses': {'a': 18.0, 'b': 1.1},
    'captured_bonuses': {'a': 4.0, 'b': 1.5},
    'loose_piece_penalties': {'a': -18.0, 'b': 1.1},
    'blocked_piece_penalties': {'a': -16.0, 'b': 1.1},
    'game_stage_bonuses': {'midgame': 50, 'endgame': 100},
    'saved_piece': 60,
    'goal_piece': 36,
    'near_goal_piece': 4,
    'nearer_goal_piece': 3,
    'captured_opponent_piece': 5,
    'unentered_piece': -14,
    'loose_piece': -4,
    'blocked_piece': -6,
    'distance_penalty': -.2,
    'high_goal_penalty': -.3,
    'dice_roll_utilization': -2,
    'dice_spread': 3
}

def get_weights():
    if os.path.exists('best_weights.json'):
        import json
        with open('best_weights.json') as f:
            weights = json.load(f)
        for key in ['saved_bonuses', 'goal_bonuses', 'near_goal_bonuses',
                    'captured_bonuses', 'loose_piece_penalties', 'blocked_piece_penalties']:
            if key in weights and isinstance(weights[key], dict):
                # Use .isdigit() to safely check if the key is a number
                weights[key] = {int(k) if k.isdigit() else k: v for k, v in weights[key].items()}
        print("Loaded best_weights.json")
        return weights
    return INITIAL_WEIGHTS

class Agent():
    def __init__(self, board=None, weights=INITIAL_WEIGHTS, log_file='game_log.json', log_to_file=False):
        self.board = board

        raw_weights = weights if weights is not None else get_weights()
        self.weights = self._expand_weights(raw_weights)

        self.log = []
        self.log_file = log_file
        self.log_to_file = log_to_file

    def _expand_weights(self, raw):
            expanded = copy.deepcopy(raw)
            categories = [
                'saved_bonuses', 'goal_bonuses', 'near_goal_bonuses',
                'captured_bonuses', 'loose_piece_penalties', 'blocked_piece_penalties'
            ]
            
            for cat in categories:
                if isinstance(raw[cat], dict) and 'a' in raw[cat]:
                    a = raw[cat]['a']
                    b = raw[cat]['b']
                    # Create the 1-6 lookup dictionary
                    # Formula: y = a * (x^b)
                    expanded[cat] = {n: a * (math.pow(n, b)) for n in range(1, 7)}
                    expanded[cat][0] = 0.0  # Always 0 for non-existent piece 0
            return expanded

    def evaluate(self, board, player):
        winner, score = board.check_game_over()
        if winner:
            factor = 1 if winner == player else -1
            return factor * score * GAME_OVER_SCORE, {}

        # precompute distances once for both players
        distances = {piece: board.shortest_route_to_goal(piece) for piece in board.pieces}

        opponent = 'white' if player == 'black' else 'black'
        player_eval, player_components = self.evaluate_player(board, player, distances)
        opponent_eval, opponent_components = self.evaluate_player(board, opponent, distances)

        total_score = player_eval - opponent_eval
        score_components = {
            'player': player_components,
            'opponent': opponent_components,
            'total_score': f'{player}: {player_eval} - {opponent_eval} = {total_score}'
        }

        return total_score, score_components


    def evaluate_player(self, board, player, distances):
        opponent = 'white' if player == 'black' else 'black'
        save_rack = board.get_save_rack(player)
        unentered_rack = board.get_unentered_rack(player)
        opponent_unentered = board.get_unentered_rack(opponent)

        # Build piece subsets once
        player_pieces = [p for p in board.pieces if p.player == player]
        opponent_pieces = [p for p in board.pieces if p.player == opponent]
        player_board_pieces = [p for p in player_pieces if p.tile]
        opponent_board_pieces_list = [p for p in opponent_pieces if p.tile]

        # Saved pieces and bonus
        saved_pieces = len(save_rack)
        saved_bonus = sum(self.weights['saved_bonuses'].get(piece.number, 0) for piece in save_rack)

        # Goal pieces and bonus
        goal_pieces = [p for p in player_pieces if p.can_be_saved()]
        goal_bonus = sum(self.weights['goal_bonuses'].get(p.number, 0) for p in goal_pieces if p.number <= 6)

        # High goal penalty
        occupied_goals = [p.tile for p in goal_pieces if p.tile and p.number > 6]
        high_goal_penalty = sum(self.weights['goal_bonuses'].get(goal.number, 0) * self.weights['high_goal_penalty']
                                for goal in occupied_goals)

        # Pieces near goal and nearer goal with bonus
        board_pieces = player_board_pieces
        pieces_near_goal = [p for p in board_pieces if 1 <= distances[p] <= 6]
        pieces_nearer_goal = [p for p in board_pieces if p.number > 6 and 1 <= distances[p] <= 4]
        near_goal_bonus = sum(self.weights['near_goal_bonuses'].get(p.number, 0) for p in pieces_near_goal if p.number <= 6)

        # Off-goal and far-from-goal penalties
        numbered_off_goal = [p for p in player_pieces if p.number <= 6 and not p.can_be_saved()]
        off_goal_penalty = -sum(self.weights['goal_bonuses'].get(p.number, 0) for p in numbered_off_goal)
        numbered_far_from_goal = [p for p in numbered_off_goal if distances[p] > 6 and p.tile and p.tile.type in ['field', 'save']]
        far_from_goal_penalty = -sum(self.weights['goal_bonuses'].get(p.number, 0) for p in numbered_far_from_goal)

        # Total distance component
        pieces_not_near_goal = [p for p in player_pieces if distances[p] > 6]
        total_distance = min(sum(distances[p] for p in pieces_not_near_goal), 100)
        total_distance += sum(self.weights['goal_bonuses'].get(p.number, 0)
                            for p in pieces_not_near_goal if p.number <= 6) / 10

        # Blocked pieces
        blocked_pieces = [p for p in player_pieces if distances[p] > 1000]
        blocked_piece_bonus = sum(self.weights['blocked_piece_penalties'].get(p.number, 0)
                                for p in blocked_pieces if p.number <= 6)

        # Loose pieces
        loose_pieces = [p for p in board_pieces if p.tile.type == 'field' and len(p.tile.pieces) == 1]
        loose_piece_bonus = sum(self.weights['loose_piece_penalties'].get(p.number, 0)
                                for p in loose_pieces if p.number <= 6)
        opponent_board_piece_count = (len([p for p in opponent_board_pieces_list if p.tile.type in ['field', 'home']])
                                + min(1, len(opponent_unentered)))
        loose_piece_bonus *= (opponent_board_piece_count / 14)
        if board.game_stages[opponent] == 'endgame':
            loose_piece_bonus *= -1

        # Captured opponent pieces
        captured_pieces = [p for p in opponent_pieces if p.tile and p.tile.type == 'home']
        captured_bonus = sum(self.weights['captured_bonuses'].get(p.number, 0)
                            for p in captured_pieces if p.number <= 6)

        # Game stage bonus
        game_stage = board.game_stages[player]
        game_stage_bonus = self.weights['game_stage_bonuses'].get(game_stage, 0)

        # dice spread — reward having pieces at varied distances so any roll is useful
        goal_distances = set(distances[p] for p in player_board_pieces 
                            if 1 <= distances[p] <= 6
                            and not p.can_be_saved())
        dice_spread_bonus = len(goal_distances) * self.weights.get('dice_spread', 3)
    
        score_components = {
            'saved_pieces': saved_pieces * self.weights['saved_piece'],
            'saved_bonus': saved_bonus,
            'goal_pieces': len(goal_pieces) * self.weights['goal_piece'],
            'goal_bonus': goal_bonus,
            'captured_pieces': len(captured_pieces) * self.weights['captured_opponent_piece'],
            'captured_bonus': captured_bonus,
            'pieces_near_goal': len(pieces_near_goal) * self.weights['near_goal_piece'],
            'pieces_nearer_goal': len(pieces_nearer_goal) * self.weights['nearer_goal_piece'],
            'near_goal_bonus': near_goal_bonus,
            'blocked_pieces': len(blocked_pieces) * self.weights['blocked_piece'],
            'blocked_piece_bonus': blocked_piece_bonus,
            'loose_pieces': len(loose_pieces) * self.weights['loose_piece'],
            'loose_piece_bonus': loose_piece_bonus,
            'total_distance': total_distance * self.weights['distance_penalty'],
            'unentered_pieces': len(unentered_rack) * self.weights['unentered_piece'],
            'off_goal_penalty': off_goal_penalty,
            'far_from_goal_penalty': far_from_goal_penalty,
            'high_goal_penalty': high_goal_penalty,
            'game_stage_bonus': game_stage_bonus,
            'dice_spread_bonus': dice_spread_bonus
        }
        total_score = sum(score_components.values())
        score_components['_total_score'] = total_score
        score_components['_player'] = player
        score_components['_goal_pieces'] = [(piece.number, piece.player, distances[piece]) for piece in pieces_near_goal]

        return total_score, score_components


    def select_move_pair(self, moves, board, player):
        move_scores = dict()

        # Ensure moves is a set and does not contain integers
        if not isinstance(moves, (list, set)) or not all(isinstance(m, tuple) for m in moves):
            raise ValueError('Invalid moves format: expected a list or set of tuples.')

        # Evaluate the pass move if legal
        if (0, 0, 0) in moves:
            move_scores[((0, 0, 0), (0, 0, 0))] = self.evaluate(board, player)

        # Create a set of moves without the pass move
        moves = set(moves)
        moves.discard((0, 0, 0))

        for move in moves:
            if not isinstance(move, tuple) or len(move) != 3:
                raise ValueError('Invalid move format: each move should be a tuple of length 3.')

            initial_move_count = len(board.moves)

            board.apply_move(move, switch_turn=False)
            # only score pass as second move if it would be legal
            remaining_captured = [p for p in board.home_tile.pieces if p.player == board.current_player]
            if not remaining_captured:
                move_scores[(move, (0, 0, 0))] = self.evaluate(board, player)
     #       print(f"After {move}: dice={[(d.number, d.used) for d in board.dice]}, next_moves count={len(set(board.get_valid_moves()))}")


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
                    raise ValueError('Invalid next move format: each move should be a tuple of length 3.')

                board.apply_move(next_move, switch_turn=False)
                move_scores[(move, next_move)] = self.evaluate(board, player)
                board.undo_last_move()

            while len(board.moves) > initial_move_count:
                board.undo_last_move()



        best_move_pair = max(move_scores, key=lambda k: move_scores[k][0])
        best_move_score, best_move_components = move_scores[best_move_pair]

        # debug: check if we chose pass as second move when better options existed
#        if best_move_pair[1] == (0, 0, 0):
 #           non_pass_pairs = {k: v for k, v in move_scores.items() if k[1] != (0, 0, 0)}
  #          if non_pass_pairs:
   #             best_non_pass = max(non_pass_pairs, key=lambda k: non_pass_pairs[k][0])
    #            best_non_pass_score = non_pass_pairs[best_non_pass][0]
     #           print(f"Chose pass as second move (score={best_move_score:.1f}) over {best_non_pass} (score={best_non_pass_score:.1f})")
      #          print(f"First move: {best_move_pair[0]}, dice: {[(d.number, d.used) for d in board.dice]}")
       #         print(f"Chosen components: {best_move_components}")
        #        print(f"Best non-pass components: {non_pass_pairs[best_non_pass][1]}")
         #   else:
          #      print("Chose pass as second move, no non-pass pairs")


        # get top 4 move pairs by score
        top_moves = sorted(move_scores.items(), key=lambda x: x[1][0], reverse=True)[:4]

        log_entry = {
            'move': best_move_pair,
            'score': best_move_score,
            'components': best_move_components,
            'competitors': [
                {
                    'move': move_pair,
                    'score': score,
                    'components': components
                }
                for move_pair, (score, components) in top_moves[1:]  # skip best, already logged
            ]
        }

        self.log.append(log_entry)

        # keep only last N entries
        MAX_LOG_ENTRIES = 10
        if len(self.log) > MAX_LOG_ENTRIES:
            self.log = self.log[-MAX_LOG_ENTRIES:]

        if self.log_to_file:
            with open(self.log_file, 'w') as file:
                file.write(json.dumps(self.log, indent=4))
            #print(f"Log updated with move: {best_move_pair}")


        return best_move_pair




# agent tried to save a numbered piece when it wasn't in the midgame (but was one piece away from midgame)
# agent doesn't bring out its second captured piece but passes instead
# in endgame agent tries to save a piece that can't be saved when it can save a piece on a lower goal
# add "distance from endgame bonus"