import json
import os
import copy
import math
from collections import deque
import time
import logging
logger = logging.getLogger(__name__)

GAME_OVER_SCORE = 10000
# Components that count only through a fitted 'component_scale' (prefilter);
# the plain heuristic sum leaves them out. ROUTE_CAP caps a route length.
def _enemy_distance(start, enemy_at, walls, limit):
    """Steps from a tile to the nearest tile in enemy_at (index set), moving over
    field tiles not in walls; home and goal tiles end a route (not passed
    through). Larger than limit if none within it."""
    seen, frontier, d = {start}, [start], 0
    while frontier and d < limit:
        d += 1
        nxt = []
        for t in frontier:
            for n in t.neighbors:
                if n in seen or n.type == 'nogo':
                    continue
                seen.add(n)
                if n.index in enemy_at:
                    return d
                if n.type == 'field' and n.index not in walls:
                    nxt.append(n)
        frontier = nxt
    return limit + 1


SCALED_ONLY = ('numbered_route', 'blank_route', 'rack_route_n1', 'rack_route_n2', 'rack_route_n3',
               'blot_threat_near', 'blot_threat_far')
ROUTE_CAP = 16
LOG_TO_FILE = False

INITIAL_WEIGHTS = {
    'saved_bonuses': {'a': 18.0, 'b': 1.1},
    'goal_bonuses': {'a': 38.0, 'b': 1.05},
    'near_goal_bonuses': {'a': 18.0, 'b': 1.1},
    'captured_bonuses': {'a': 4.0, 'b': 1.5},
    'loose_piece_penalties': {'a': -18.0, 'b': 1.1},
    'blocked_piece_penalties': {'a': -16.0, 'b': 1.1},
    'enemy_blot_penalties': {'a': -6.0, 'b': 1.1},
    'high_goal_proximity_penalties': {'a': -0.25, 'b': 1.2},
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
    'dice_spread': 10,
    'permanent_block_bonus': 13, 
}

def get_weights(weights_file = 'best_weights.json'):
    if os.path.exists(weights_file):
        import json
        with open(weights_file) as f:
            weights = json.load(f)

        for key, default_value in INITIAL_WEIGHTS.items():
            if key not in weights:
                weights[key] = default_value
     #           print(f"Added missing weight: {key} = {default_value}")

        for key in ['saved_bonuses', 'goal_bonuses', 'near_goal_bonuses',
                    'captured_bonuses', 'loose_piece_penalties', 'blocked_piece_penalties',
                    'enemy_blot_penalties', 'high_goal_proximity_penalties']:
            if key in weights and isinstance(weights[key], dict):
                weights[key] = {int(k) if k.isdigit() else k: v for k, v in weights[key].items()}
       # print("Loaded", weights_file)
        return weights
    return INITIAL_WEIGHTS


class Agent():
    def __init__(self, board=None, weights=INITIAL_WEIGHTS, log_file='game_log.json', log_to_file=LOG_TO_FILE):
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
            'captured_bonuses', 'loose_piece_penalties', 'blocked_piece_penalties',
            'enemy_blot_penalties'
        ]
        for cat in categories:
            try:
                if cat not in raw:
                    continue
                if not isinstance(raw[cat], dict):
                    continue
                if 'a' not in raw[cat]:
                    continue
                a = raw[cat]['a']
                b = raw[cat]['b']
                expanded[cat] = {n: a * (math.pow(n, b)) for n in range(1, 7)}
                expanded[cat][0] = 0.0
            except Exception:
                pass
        return expanded

    def _extra_terms(self):
        if not hasattr(self, '_extra'):
            sc = self.weights.get('component_scale') or {}
            self._extra = (os.environ.get('HEUR_EXTRA') == '1' or
                           any(sc.get(k, 0) for k in SCALED_ONLY))
        return self._extra

    def evaluate(self, board, player):
        winner, score = board.check_game_over()
        if winner:
            factor = 1 if winner == player else -1
            return factor * score * GAME_OVER_SCORE, {}
        opponent = 'white' if player == 'black' else 'black'
        pieces_by_player = {
            player:   [p for p in board.pieces if p.player == player],
            opponent: [p for p in board.pieces if p.player == opponent],
        }
        distances = {piece: board.shortest_route_to_goal(piece)
                    for pieces in pieces_by_player.values()
                    for piece in pieces}
        player_eval, player_components = self.evaluate_player(board, player, distances, pieces_by_player)
        opponent_eval, opponent_components = self.evaluate_player(board, opponent, distances, pieces_by_player)
        total_score = player_eval - opponent_eval
        score_components = {
            'player': player_components,
            'opponent': opponent_components,
            'total_score': f'{player}: {player_eval} - {opponent_eval} = {total_score}'
        }
        return total_score, score_components

    def evaluate_player(self, board, player, distances, pieces_by_player):
        opponent = 'white' if player == 'black' else 'black'
        save_rack = board.get_save_rack(player)
        unentered_rack = board.get_unentered_rack(player)
        opponent_unentered = board.get_unentered_rack(opponent)

        player_pieces = pieces_by_player[player]
        opponent_pieces = pieces_by_player[opponent]
        player_board_pieces = [p for p in player_pieces if p.tile]
        opponent_board_pieces_list = [p for p in opponent_pieces if p.tile]

        saved_pieces = len(save_rack)
        saved_bonus = sum(self.weights['saved_bonuses'].get(piece.number, 0) for piece in save_rack)

        goal_pieces = [p for p in player_pieces if p.can_be_saved()]
        goal_bonus = sum(self.weights['goal_bonuses'].get(p.number, 0) for p in goal_pieces if p.number <= 6)

        occupied_goals = [p.tile for p in goal_pieces if p.tile and p.number > 6]
        high_goal_penalty = sum(self.weights['goal_bonuses'].get(goal.number, 0) * self.weights['high_goal_penalty']
                                for goal in occupied_goals)

        board_pieces = player_board_pieces
        pieces_near_goal = [p for p in board_pieces if 1 <= distances[p] <= 6]
        pieces_nearer_goal = [p for p in board_pieces if p.number > 6 and 1 <= distances[p] <= 4]
        near_goal_bonus = sum(self.weights['near_goal_bonuses'].get(p.number, 0) for p in pieces_near_goal if p.number <= 6)

        high_goal_proximity_penalty = 0
        if board.game_stages[player] != 'endgame':
            near_but_not_nearer = [p for p in pieces_near_goal if p.number > 6 and p not in pieces_nearer_goal]
            for piece in near_but_not_nearer:
                target_goal = board.get_target_goal_number(piece)
                if target_goal:
                    goal_index = target_goal - 1
                    penalty_value = self.weights['high_goal_proximity_penalties']['a'] * (goal_index ** self.weights['high_goal_proximity_penalties']['b'])
                    high_goal_proximity_penalty += penalty_value

        numbered_off_goal = [p for p in player_pieces if p.number <= 6 and not p.can_be_saved() and not (p.rack is save_rack)]
        off_goal_penalty = -sum(self.weights['goal_bonuses'].get(p.number, 0) for p in numbered_off_goal)
        numbered_far_from_goal = [p for p in numbered_off_goal if distances[p] > 6 and p.tile and p.tile.type in ['field', 'save']]
        far_from_goal_penalty = -sum(self.weights['goal_bonuses'].get(p.number, 0) for p in numbered_far_from_goal)

        pieces_not_near_goal = [p for p in player_pieces if distances[p] > 6]
        total_distance = min(sum(distances[p] for p in pieces_not_near_goal), 100)
        total_distance += sum(self.weights['goal_bonuses'].get(p.number, 0)
                            for p in pieces_not_near_goal if p.number <= 6) / 10

        # The new terms cost time; they are computed only when a fitted scale
        # uses them or HEUR_EXTRA=1 (prefilter_fit extraction). Else all 0.
        numbered_route = blank_route = blot_threat_near = blot_threat_far = 0
        rack_route = {1: 0, 2: 0, 3: 0}
        if self._extra_terms():
            # ROUTE LENGTHS (2026-10-29, prefilter only): what a wall costs the side
            # it blocks. Numbered pieces off their goal (own-goal route) and blanks on
            # the field (nearest goal), each capped at ROUTE_CAP (sealed = cap). They
            # count ONLY through a fitted component_scale: absent from the plain sum,
            # so the heuristic as a player is unchanged.
            def _rl(p):
                d = distances[p]
                return ROUTE_CAP if d > ROUTE_CAP else d
            numbered_route = sum(_rl(p) for p in player_pieces
                                 if p.number <= 6 and p.tile is not None and not p.can_be_saved())
            blank_route = sum(_rl(p) for p in player_pieces
                              if p.number > 6 and p.tile is not None and p.tile.type == 'field')

            # Rack pieces by place in line (owner, 2026-10-29): each numbered rack
            # piece counts route-from-home + n x its place (0 = front), capped at
            # 7 + 12n (sealed = cap) -- so a seal is worth 12n - n x place, most
            # against the front piece, falling evenly to ~0 at the back of the rack.
            # n = 1, 2, 3 stored; the fit picks.
            rack_num = [(i, distances[p]) for i, p in enumerate(unentered_rack) if p.number <= 6]
            rack_route = {n: sum(min(d + n * i, 7 + 12 * n) for i, d in rack_num) for n in (1, 2, 3)}
            # Threatened numbered blots: a lone numbered piece on the field with an
            # enemy piece (or the enemy's rack, entering from home) within 6 / 7-12
            # steps by a route the enemy may use (our walls block it).
            blot_threat_near = blot_threat_far = 0
            lone_numbered = [p for p in board_pieces if p.number <= 6 and p.tile.type == 'field'
                             and len(p.tile.pieces) == 1]
            if lone_numbered:
                enemy_at = {}
                for q in opponent_pieces:
                    if q.tile is not None:
                        enemy_at[q.tile.index] = True
                if opponent_unentered:
                    enemy_at[board.home_tile.index] = True
                walls = board._get_blocked_key(opponent)
                for p in lone_numbered:
                    d = _enemy_distance(p.tile, enemy_at, walls, 12)
                    if d <= 6:
                        blot_threat_near += 1
                    elif d <= 12:
                        blot_threat_far += 1

        blocked_pieces = [p for p in player_pieces if distances[p] > 1000]
        blocked_piece_bonus = sum(self.weights['blocked_piece_penalties'].get(p.number, 0)
                                for p in blocked_pieces if p.number <= 6)

        loose_pieces = [p for p in board_pieces if p.tile.type == 'field' and len(p.tile.pieces) == 1]
        loose_piece_penalty = len(loose_pieces) * self.weights['loose_piece']
        if board.game_stages[opponent] == 'endgame':
            loose_piece_penalty *= -1

        loose_piece_bonus = sum(self.weights['loose_piece_penalties'].get(p.number, 0)
                                for p in loose_pieces if p.number <= 6)
        opponent_board_piece_count = (len([p for p in opponent_board_pieces_list if p.tile.type in ['field', 'home']])
                                + min(1, len(opponent_unentered)))
        loose_piece_bonus *= (opponent_board_piece_count / 12)

        enemy_blot_penalty = 0
        if board.game_stages['white'] != 'endgame' and board.game_stages['black'] != 'endgame':
            for piece in player_board_pieces:
                if not piece.can_be_saved():
                    blots = board.count_enemy_blots_on_shortest_path(piece)
                    if blots != float('inf') and blots > 0:
                        piece_value = self.weights['enemy_blot_penalties'].get(piece.number, 1)
                        enemy_blot_penalty += blots * piece_value

        captured_pieces = [p for p in opponent_pieces if p.tile and p.tile.type == 'home']
        captured_bonus = sum(self.weights['captured_bonuses'].get(p.number, 0)
                            for p in captured_pieces if p.number <= 6)

        game_stage = board.game_stages[player]
        game_stage_bonus = self.weights['game_stage_bonuses'].get(game_stage, 0)

        # Count useful dice rolls (1-6) – at least one piece can be saved or can move onto its goal
        useful_rolls = set()
        for piece in player_pieces:
            if piece.can_be_saved():
                save_rolls = board.get_saving_die(piece)   # list of dice numbers that would save this piece
                for r in save_rolls:
                    useful_rolls.add(r)

        for piece in player_board_pieces:
            if not piece.can_be_saved() and not (piece.rack is save_rack):
                d = distances[piece]
                if 1 <= d <= 6:
                    useful_rolls.add(d)
        dice_spread_bonus = len(useful_rolls) * self.weights.get('dice_spread', 3)

        # Permanent block bonus
        permanent_block_bonus = 0
        if board.game_stages[opponent] != 'endgame':
            for tile in board.tiles:
                if tile.type == 'field':
                    friendly_pieces = [p for p in tile.pieces if p.player == player]
                    if len(friendly_pieces) >= 2 and all(p.number > 6 for p in friendly_pieces):
                        permanent_block_bonus += self.weights.get('permanent_block_bonus', 13)

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
            'loose_pieces': loose_piece_penalty,
            'loose_piece_bonus': loose_piece_bonus,
            'total_distance': total_distance * self.weights['distance_penalty'],
            'unentered_pieces': len(unentered_rack) * self.weights['unentered_piece'],
            'off_goal_penalty': off_goal_penalty,
            'far_from_goal_penalty': far_from_goal_penalty,
            'high_goal_penalty': high_goal_penalty,
            'high_goal_proximity_penalty': high_goal_proximity_penalty,
            'enemy_blot_penalty': enemy_blot_penalty,
            'game_stage_bonus': game_stage_bonus,
            'dice_spread_bonus': dice_spread_bonus,
            'permanent_block_bonus': permanent_block_bonus,
            'numbered_route': numbered_route,
            'blank_route': blank_route,
            'rack_route_n1': rack_route[1],
            'rack_route_n2': rack_route[2],
            'rack_route_n3': rack_route[3],
            'blot_threat_near': blot_threat_near,
            'blot_threat_far': blot_threat_far,
        }
        # Per-component scales (weights['component_scale'], prefilter only --
        # see agent_gnn): fitted 2026-10-05 so the net's favourite move
        # survives the prefilter (prefilter_fit.py). Absent = plain sum, so
        # the heuristic as a player is unchanged.
        scale = self.weights.get('component_scale')
        if scale:
            total_score = sum(v * scale.get(k, 0.0 if k in SCALED_ONLY else 1.0)
                              for k, v in score_components.items())
        else:
            total_score = sum(v for k, v in score_components.items() if k not in SCALED_ONLY)
        score_components['_total_score'] = total_score
        score_components['_player'] = player
        score_components['_goal_pieces'] = [(piece.number, piece.player, distances[piece]) for piece in pieces_near_goal]

        return total_score, score_components

    def select_move_pair(self, moves, board, player):

        start_time = time.time()

        if not board.dice[0].used and not board.dice[1].used:
            board.firstMove = None

        if not isinstance(moves, (list, set)) or not all(isinstance(m, tuple) for m in moves):
            raise ValueError('Invalid moves format: expected a list or set of tuples.')

        best_move_pair = None
        best_score = float('-inf')
        best_components = None

        top_moves = []

        def _keep_top(move_pair, score, components):
            top_moves.append((score, move_pair, components))
            top_moves.sort(key=lambda x: x[0], reverse=True)
            if len(top_moves) > 4:
                top_moves.pop()

        save_die_log = []
        board.log_callback = lambda entry: save_die_log.append(entry)

        if (0, 0, 0) in moves:
            score, components = self.evaluate(board, player)
            if score > best_score:
                best_score = score
                best_move_pair = ((0, 0, 0), (0, 0, 0))
                best_components = components
            _keep_top(((0, 0, 0), (0, 0, 0)), score, components)

        moves = set(moves)
        moves.discard((0, 0, 0))

        for move in moves:
            if not isinstance(move, tuple) or len(move) != 3:
                raise ValueError('Invalid move format: each move should be a tuple of length 3.')

            initial_move_count = len(board.moves)
            board.apply_move(move, switch_turn=False)

            remaining_captured = [p for p in board.home_tile.pieces if p.player == board.current_player]
            if not remaining_captured:
                score, components = self.evaluate(board, player)
                if score > best_score:
                    best_score = score
                    best_move_pair = (move, (0, 0, 0))
                    best_components = components
                _keep_top((move, (0, 0, 0)), score, components)

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
                score, components = self.evaluate(board, player)
                if score > best_score:
                    best_score = score
                    best_move_pair = (move, next_move)
                    best_components = components
                _keep_top((move, next_move), score, components)
                board.undo_last_move()

            while len(board.moves) > initial_move_count:
                board.undo_last_move()

        def serialize_first_move(fm):
            if fm is None:
                return None
            tile = fm['origin_tile']
            return {
                'piece': (fm['piece'].player, fm['piece'].number),
                'origin_tile': {'type': tile.type, 'ring': tile.ring, 'pos': tile.pos} if tile else None,
            }

        def snapshot(board):
            return {
                'firstMove': serialize_first_move(board.firstMove),
                'dice': [{'number': d.number, 'used': d.used} for d in board.dice],
            }

        def move_with_first(move):
            before = snapshot(board)
            board.apply_move(move, switch_turn=False)
            after = snapshot(board)
            board.undo_last_move()
            restored = snapshot(board)
            undo_mismatch = restored != before
            return {
                'move': move,
                'before': before,
                'after': after,
                'restored': restored,
                'undo_mismatch': undo_mismatch,
            }

        if self.log_to_file:
            self.log = [
                {
                    'move_pair': {
                        'first': move_with_first(move_pair[0]),
                        'second': move_with_first(move_pair[1]),
                    },
                    'score': score,
                    'components': components
                }
                for score, move_pair, components in top_moves
            ]
            with open(self.log_file, 'w') as file:
                file.write(json.dumps(self.log, indent=4))

        board.log_callback = None
        self.log = save_die_log + self.log

        if best_move_pair is None:
            if (0, 0, 0) in moves:
                return ((0, 0, 0), (0, 0, 0))
            return next(iter(moves)) if moves else ((0, 0, 0), (0, 0, 0))
        
        # --- Reordering logic for numbered piece saves: ensure numbered piece is saved first if possible to avoid frontend bug ---
        # this may no longer be needed if we're doing this correctly on the frontend
        def is_save_numbered_move(move):
            if move[1] != 'save':
                return False
            piece_id = move[0]
            if isinstance(piece_id, tuple) and len(piece_id) == 2:
                return piece_id[1] <= 6
            return False

        def is_bring_out_move(move, board):
            piece_id = move[0]
            if piece_id == 0 or piece_id == (0,0):
                return False
            piece = board.piece_lookup.get(piece_id)
            if not piece:
                return False
            if piece.tile and piece.tile.type == 'home':
                return True
            if piece.rack and piece.rack in (board.white_unentered, board.black_unentered):
                return True
            return False

        if best_move_pair != ((0,0,0), (0,0,0)):
            m1, m2 = best_move_pair
            save1 = is_save_numbered_move(m1)
            save2 = is_save_numbered_move(m2)
            if save1 != save2:
                save_move = m1 if save1 else m2
                other_move = m2 if save1 else m1
                if save_move[0] != other_move[0] and not is_bring_out_move(other_move, board):
                    best_move_pair = (save_move, other_move)

        elapsed = time.time() - start_time
        logger.info(f"select_move_pair took {elapsed:.3f} seconds")

        return best_move_pair