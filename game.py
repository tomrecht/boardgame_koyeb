import random
import os
from collections import deque
import json
import itertools

NUM_PIECES = 12
# Hard ceiling on the per-position caches, in entries. A busy midgame turn adds
# a few thousand, so this is far above anything one search needs; it exists so a
# pathological position can't grow the process without bound between resets.
CACHE_MAX = int(os.environ.get('BOARD_CACHE_MAX', '150000'))
# After this many two-player turns with no save (once both players are in
# midgame), either player may call a draw. Easily tunable.
NO_SAVE_TURNS_FOR_DRAW = 10

class Die:
    def __init__(self, board):
        self.board = board
        self.roll()

    def roll(self):
        self.number = random.randint(1, 6)  
        self.used = False

class Piece:
    def __init__(self, player, number, board):
        self.player = player
        self.number = number
        self.board = board
        self.tile = None
        self.rack = None
        self.reachable_tiles = None
        self.reachable_by_sum = None
        self.index = None

    def __repr__(self):
        return f'{self.player}({self.number})'
    
    def can_be_saved(self, already_saved_counts_as_saveable=True):
        if self.rack and (self.rack is self.board.white_saved or self.rack is self.board.black_saved):
            return True if already_saved_counts_as_saveable else False
        
        tile = self.tile
        if tile and tile.type == 'save':
            if self.number > 6 or (self.number == tile.number):
                return True
        return False

class Tile:
    def __init__(self, tile_type, ring, pos, board, number=None):
        self.type = tile_type
        self.ring = ring
        self.pos = pos
        self.pieces = []
        self.neighbors = []
        self.board = board
        self.number = number
        self.index = None

    def __repr__(self):
        return f"{self.type}({self.ring}, {self.pos})"
    
    def is_blocked(self, player=None):
        if not player:
            player = self.board.current_player
        return self.type == 'field' and len(self.pieces) > 1 and self.pieces[0].player != player

class Board:
    def __init__(self):
        self.players = ['white', 'black']
        self.dice = [Die(self), Die(self)] 
        self.pieces = []
        self.tiles = []
        self.tile_map = {}
        self.load_from_json('tile_neighbors.json')
        self.home_tile = self.get_tile(0, 0)
        self.current_player = 'white'
        self.white_unentered = []
        self.black_unentered = []
        self.white_saved = []
        self.black_saved = []
        self.assign_tile_indices()
        self.game_stages = {'white': 'opening', 'black': 'opening'}
        self.initialize_pieces()
        self.piece_lookup = {(p.player, p.number): p for p in self.pieces}
        self.firstMove = None
        self.moves = []
        # No-save draw rule (counts FULL ROUNDS = two player-turns). The
        # counter is maintained here in switch_turn so it works for every play
        # path including agent-vs-agent self-play and RL (no frontend needed).
        # A "save" is any increase in total saved pieces (own save or saving an
        # opponent's block); any save resets the streak. Counting only runs
        # once both players have left the opening (both_in_midgame).
        self.no_save_turns = 0          # completed rounds with no save
        self.draw_callable = False
        self._last_total_saved = 0      # snapshot of saved count at last turn boundary
        self._half_turns_since_round = 0  # 2 player-turns = 1 round
        self.draw_called = False
        self._distance_cache = {}
        self._blot_cache = {}          # NEW: cache for count_enemy_blots_on_shortest_path
        self._reachable_cache = {}
        self._blocked_key_cache = {}
        self._blot_key_cache = {}
        self.log_callback = None
        self.endgame_reward_applied = {'white': False, 'black': False}
        self.offgoals = {'white': 0, 'black': 0}

    def both_in_midgame(self):
        """True once neither player has pieces left in their unentered rack
        (i.e. both have left the opening). The no-save draw counter only runs
        from this point on."""
        return len(self.white_unentered) == 0 and len(self.black_unentered) == 0

    def total_saved(self):
        return len(self.white_saved) + len(self.black_saved)

    def update_no_save_counter(self):
        """Called once per player-turn at the real turn boundary (switch_turn).
        Resets the streak on any save; otherwise advances the round counter
        (2 player-turns = 1 round) once both players are in midgame. Robust to
        the agent's apply/undo search, which never calls switch_turn and leaves
        total_saved() unchanged across a probe+undo."""
        current_saved = self.total_saved()
        if current_saved > self._last_total_saved:
            # A piece was saved during the turn that just ended -> reset streak.
            self.no_save_turns = 0
            self._half_turns_since_round = 0
        elif self.both_in_midgame():
            self._half_turns_since_round += 1
            if self._half_turns_since_round >= 2:
                self._half_turns_since_round = 0
                self.no_save_turns += 1
        self._last_total_saved = current_saved
        self.draw_callable = self.no_save_turns >= NO_SAVE_TURNS_FOR_DRAW

    def __repr__(self):
        board_repr = "White unentered: " + str(self.white_unentered) + "\n"
        board_repr += "White saved: " + str(self.white_saved) + "\n"
        board_repr += "Black unentered: " + str(self.black_unentered) + "\n"
        board_repr += "Black saved: " + str(self.black_saved) + "\n"
        board_repr += "Pieces on board:\n"
        for piece in self.pieces:
            if piece.tile:
                board_repr += f"  {piece} on {piece.tile}\n"
        return board_repr

    def clear(self):
        self.white_unentered.clear()
        self.black_unentered.clear()
        self.white_saved.clear()
        self.black_saved.clear()
        self.pieces.clear()
        for tile in self.tiles:
            tile.pieces.clear()

    def add_tile(self, tile):
        self.tiles.append(tile)
        key = (tile.ring, tile.pos)
        self.tile_map[key] = tile

    def get_tile(self, ring, pos):
        return self.tile_map.get((ring, pos))

    def initialize_pieces(self):
        for player in self.players:
            pieces = [Piece(player, i + 1, self) for i in range(NUM_PIECES)]
            random.shuffle(pieces)
            if player == 'white':
                self.white_unentered.extend(pieces)
                for piece in pieces:
                    piece.rack = self.white_unentered
            else:
                self.black_unentered.extend(pieces)
                for piece in pieces:
                    piece.rack = self.black_unentered
            self.pieces.extend(pieces)

    def load_from_json(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        for key, value in data.items():
            ring, sector = map(int, key.replace('ring', '').replace('sector', '').split('_'))
            tile_type = value['type']
            number = value.get('number')
            tile = Tile(tile_type, ring, sector, self, number)
            self.add_tile(tile)
        for key, value in data.items():
            ring, sector = map(int, key.replace('ring', '').replace('sector', '').split('_'))
            tile = self.get_tile(ring, sector)
            if tile:
                for neighbor in value['neighbors']:
                    neighbor_tile = self.get_tile(neighbor['ring'], neighbor['sector'])
                    if neighbor_tile:
                        tile.neighbors.append(neighbor_tile)

    def _get_blocked_key(self, player):
        if player not in self._blocked_key_cache:
            self._blocked_key_cache[player] = frozenset(
                t.index for t in self.tiles
                if t.type == 'field'
                and len(t.pieces) > 1
                and t.pieces[0].player != player
            )
        return self._blocked_key_cache[player]

    def _get_blot_key(self, player):
        """Tiles holding exactly one enemy piece -- the other half of what a
        blot count depends on. Kept alongside _get_blocked_key and invalidated
        with it, so blot counts can be cached on a complete key and survive the
        apply/undo churn of a candidate search (a search that moves only my own
        pieces doesn't change this set at all, which is where the reuse comes
        from)."""
        if player not in self._blot_key_cache:
            self._blot_key_cache[player] = frozenset(
                t.index for t in self.tiles
                if t.type == 'field'
                and len(t.pieces) == 1
                and t.pieces[0].player != player
            )
        return self._blot_key_cache[player]

    def update_state(self, game_state_details):
        self.current_player = game_state_details['currentTurn']
        # BUGFIX: this was never invalidated here, so a reused Board (training
        # target/input encoding loops over records, and the Flask app between
        # requests) kept the PREVIOUS state's blocked-tile snapshot -- distance
        # features were then computed against stale blocking.
        self._blocked_key_cache.clear()
        self._blot_key_cache.clear()
        # The value caches are pure functions of their keys, so keeping them
        # across positions is CORRECT but unbounded -- and the app reuses one
        # Board per thread and never calls switch_turn, which is the only other
        # place they are cleared. Measured on a long game: _blot_cache reached
        # 90k entries and the worker 174 MB by move 45, still climbing ~2 MB a
        # move, which is what made moves take 30s and then get OOM-killed on a
        # 256 MB instance. Every entry retains the frozensets in its key, so the
        # cost is memory, not entries. Within one search they still do all the
        # work they were added for; only cross-request reuse is given up, and
        # that reuse is worthless because each request is a different position.
        self._distance_cache.clear()
        self._blot_cache.clear()
        self._reachable_cache.clear()
        self.clear()
        for die, die_details in zip(self.dice, game_state_details['dice']):
            die.number = die_details['value']
            die.used = die_details['used']

        def place_pieces_in_rack(rack, pieces_details, player):
            rack.clear()
            for piece_details in pieces_details:
                piece = Piece(player, piece_details['number'], self)
                self.pieces.append(piece)
                rack.append(piece)
                piece.rack = rack
        
        place_pieces_in_rack(self.white_unentered, game_state_details['racks']['whiteUnentered'], 'white')
        place_pieces_in_rack(self.white_saved, game_state_details['racks']['whiteSaved'], 'white')
        place_pieces_in_rack(self.black_unentered, game_state_details['racks']['blackUnentered'], 'black')
        place_pieces_in_rack(self.black_saved, game_state_details['racks']['blackSaved'], 'black')

        for piece_details in game_state_details['boardPieces']:
            player = piece_details['color']
            number = piece_details['number']
            ring = piece_details['tile']['ring']
            sector = piece_details['tile']['sector']
            tile = self.get_tile(ring, sector)
            piece = Piece(player, number, self)
            piece.tile = tile
            tile.pieces.append(piece)
            self.pieces.append(piece)
            if 'reachableBySum' in piece_details:
                piece.reachable_by_sum = [self.get_tile(t['ring'], t['sector']) for t in piece_details['reachableBySum']]
                self.firstMove = {'piece': piece, 'origin_tile': tile}

        self.assign_piece_indices()
        self.piece_lookup = {(p.player, p.number): p for p in self.pieces}
        self.game_stages['white'] = self.get_game_stage('white')
        self.game_stages['black'] = self.get_game_stage('black')
        self.apply_last_piece_rule()

        # In live play the frontend owns the live no-save counter (so it can
        # reset instantly on a save and be undo-sensitive). Adopt whatever it
        # reports if present; fall back to this board's own counter otherwise
        # (e.g. agent-vs-agent paths that never go through update_state).
        if 'noSaveTurns' in game_state_details:
            self.no_save_turns = int(game_state_details.get('noSaveTurns', 0))
            self.draw_callable = bool(game_state_details.get(
                'drawCallable', self.no_save_turns >= NO_SAVE_TURNS_FOR_DRAW))
            self._half_turns_since_round = 0
        # keep the saved-count snapshot consistent with the rebuilt board
        self._last_total_saved = self.total_saved()

    def assign_tile_indices(self):
        for i in range(len(self.tiles)):
            self.tiles[i].index = i

    def assign_piece_indices(self):
        self.pieces.sort(key=lambda piece: (piece.player != 'white', piece.number))
        for i in range(len(self.pieces)):
            self.pieces[i].index = i+1

    def get_game_stage(self, player):
        unentered_rack = self.white_unentered if player == 'white' else self.black_unentered
        if len(unentered_rack) > 0:
            return 'opening'
        player_pieces = [p for p in self.pieces if p.player == player]
        if all(p.can_be_saved() for p in player_pieces):
            return 'endgame'
        return 'midgame'
    
    def switch_turn(self):
        self._distance_cache.clear()
        self._blot_cache.clear()      # clear blot cache
        self._reachable_cache.clear()
        self._blocked_key_cache.clear()
        self._blot_key_cache.clear()
        self.update_no_save_counter()
        self.firstMove = None  
        for die in self.dice:
            die.roll()
        self.current_player = 'white' if self.current_player == 'black' else 'black'
        self.apply_last_piece_rule()
        

    def check_game_over(self):
        if self.draw_called:
            return 'draw', 0
        white_saved_count = len(self.white_saved)
        black_saved_count = len(self.black_saved)
        if white_saved_count == NUM_PIECES:
            black_unsaved_count = NUM_PIECES - black_saved_count
            return 'white', black_unsaved_count
        if black_saved_count == NUM_PIECES:
            white_unsaved_count = NUM_PIECES - white_saved_count
            return 'black', white_unsaved_count
        return None, None

    def get_unentered_piece(self):
        unentered_rack = self.white_unentered if self.current_player == 'white' else self.black_unentered
        if len(unentered_rack) == 0:
            return None
        return unentered_rack[0]

    def must_move_unentered(self):
        unentered_rack = self.white_unentered if self.current_player == 'white' else self.black_unentered
        if len(unentered_rack) == 0:
            return False
        if self.home_tile.pieces and any(piece.player == self.current_player for piece in self.home_tile.pieces):
            return False
        if self.firstMove:
            return False
        return True

    def get_saving_die(self, piece):
        if self.game_stages[piece.player] == 'opening':
            return []
        valid_dice = []
        current_tile = piece.tile
        if current_tile and current_tile.type == 'save' and (piece.number > 6 or piece.number == current_tile.number):
            if self.game_stages[piece.player] == 'endgame':
                if piece.number > 6:
                    highest_occupied_goal_number = max(
                        (tile.number for tile in self.tiles
                        if tile.type == 'save' and len(tile.pieces) > 0
                        and any(p.player == piece.player for p in tile.pieces)),
                        default=-1
                    )
                    valid_dice = [die for die in self.dice
                                if (not die.used) and
                                (die.number == current_tile.number or
                                (die.number > current_tile.number and current_tile.number == highest_occupied_goal_number))]
                else:
                    valid_dice = [die for die in self.dice if (not die.used) and die.number == current_tile.number]
            else:
                valid_dice = [die for die in self.dice if (not die.used) and die.number == current_tile.number]
        if valid_dice:
            return [die.number for die in valid_dice]
        else:
            return []

    def get_reachable_tiles(self, start_tile, steps):
        if start_tile is None:
            return []
        queue = deque([(start_tile, 0)])
        visited = set([start_tile])
        reachable_tiles = []
        while queue:
            current_tile, current_steps = queue.popleft()
            if current_steps < steps:
                for neighbor in current_tile.neighbors:
                    if neighbor not in visited and neighbor.type not in ['nogo', 'home'] and not neighbor.is_blocked():
                        queue.append((neighbor, current_steps + 1))
                        visited.add(neighbor)
                        if current_steps + 1 == steps:
                            reachable_tiles.append(neighbor)
            elif current_steps == steps:
                reachable_tiles.append(current_tile)
        return list(set(reachable_tiles))

    def get_reachable_tiles_by_dice(self, piece):  
        piece.reachable_tiles = None
        reachable_tiles = {self.dice[0].number: [], self.dice[1].number: []}
        if piece.rack and piece.rack in [self.white_unentered, self.black_unentered]:
            start_tile = self.home_tile
        else:
            start_tile = piece.tile
        if not self.dice[0].used:
            reachable_tiles[self.dice[0].number] = self.get_reachable_tiles(start_tile, self.dice[0].number)
            if self.firstMove and self.firstMove['piece'] == piece: 
                origin_tile = self.firstMove['origin_tile'] or self.home_tile
                reachable_by_sum = self.get_reachable_tiles(origin_tile, self.dice[0].number + self.dice[1].number)
                reachable_tiles[self.dice[0].number] = [tile for tile in reachable_tiles[self.dice[0].number] if tile in reachable_by_sum]
        if not self.dice[1].used:
            reachable_tiles[self.dice[1].number] = self.get_reachable_tiles(start_tile, self.dice[1].number)
            if self.firstMove and self.firstMove['piece'] == piece:
                origin_tile = self.firstMove['origin_tile'] or self.home_tile
                reachable_by_sum = self.get_reachable_tiles(origin_tile, self.dice[0].number + self.dice[1].number)     
                reachable_tiles[self.dice[1].number] = [tile for tile in reachable_tiles[self.dice[1].number] if tile in reachable_by_sum]
        if piece.tile and piece.tile.type == 'save' and self.game_stages[piece.player] != 'opening':
            save_rolls = self.get_saving_die(piece)
            for roll in save_rolls:
                if roll in reachable_tiles:
                    if 'save' not in reachable_tiles[roll]:
                        reachable_tiles[roll].append('save')
        piece.reachable_tiles = reachable_tiles

    def get_valid_moves(self, mask_offgoals=False):
        self.game_stages[self.current_player] = self.get_game_stage(self.current_player)  
        if self.dice[0].used and self.dice[1].used:
            return []
        captured_pieces = [piece for piece in self.home_tile.pieces if piece.player == self.current_player]
        if captured_pieces:
            for piece in captured_pieces:
                self.get_reachable_tiles_by_dice(piece)
            self.destinations_by_piece = {piece: piece.reachable_tiles for piece in captured_pieces}
        elif self.must_move_unentered():
            piece = self.get_unentered_piece()
            self.get_reachable_tiles_by_dice(piece)
            self.destinations_by_piece = {piece: piece.reachable_tiles}
        else:
            player_pieces = [p for p in self.pieces if p.player == self.current_player and p.tile and p.tile.type in ['field', 'save']]
            unentered_piece = self.get_unentered_piece()
            if unentered_piece:
                player_pieces.append(unentered_piece)

            # Deduplicate among unnumbered pieces (number > 6) on the same tile since they're interchangeable
            deduped = []
            seen_tiles = {}
            for piece in player_pieces:
                if piece.number > 6 and piece.tile is not None:
                    if piece.tile in seen_tiles:
                        continue
                    seen_tiles[piece.tile] = True
                deduped.append(piece)
            player_pieces = deduped

            for piece in player_pieces:
                self.get_reachable_tiles_by_dice(piece)
            self.destinations_by_piece = {piece: piece.reachable_tiles for piece in player_pieces}


        tuples_list = []
        for piece, moves in self.destinations_by_piece.items():
            for roll, destinations in moves.items():
                if destinations:
                    for destination in destinations:
                        if destination == 'save':
                            tuples_list.append(((piece.player, piece.number), destination, roll))
                        else:
                            tuples_list.append(((piece.player, piece.number), (destination.ring, destination.pos), roll))
        if not captured_pieces and not self.must_move_unentered():
            tuples_list.append((0, 0, 0))   # add the pass move
            # add single-piece block-save moves: peel ONE opponent piece off a
            # block (a 2+ opponent stack on a field tile) into their own saved
            # rack, at the cost of both dice. One move PER PIECE on the block --
            # the attacker chooses which piece to gift (and thus which piece
            # remains). Peeling one off a 2-stack leaves a blot (unblocks the
            # route); off a 3+ stack leaves a smaller stack (still blocks).
            if (not self.dice[0].used and not self.dice[1].used
                    and self.game_stages[self.current_player] != 'opening'
                    and not captured_pieces
                    and not self.must_move_unentered()):
                opponent_pieces_on_blocks = [p for p in self.pieces
                    if p.player != self.current_player
                    and p.tile
                    and p.tile.type == 'field'
                    and len(p.tile.pieces) > 1]
                for piece in opponent_pieces_on_blocks:
                    tuples_list.append(((piece.player, piece.number), 0, 0))
        if (not self.dice[0].used and not self.dice[1].used) and self.draw_callable:
            tuples_list.append((1, 1, 1))   # add calling a draw
        return tuples_list

    def save_move(self, move, origin_tile=None, origin_rack=None, captured_piece=None, firstMove_before=None):
        piece_id, destination, roll = move
        move_to_save = dict()
        move_to_save['piece'] = self.piece_lookup.get(piece_id)
        move_to_save['origin_tile'] = origin_tile
        move_to_save['origin_rack'] = origin_rack
        move_to_save['destination'] = destination
        move_to_save['captured_piece'] = captured_piece
        move_to_save['roll'] = roll
        move_to_save['firstMove_before'] = firstMove_before
        self.moves.append(move_to_save)

    def undo_last_move(self):
        if not self.moves:
            return
        last_move = self.moves.pop()
        # _distance_cache entries embed the blocked frozenset in their keys,
        # making them pure functions of (start, player, class, blocking) --
        # piece positions don't otherwise affect route distances. apply_move
        # already relies on exactly this purity (it too keeps the distance
        # cache), so keeping the entries lets the 2-ply search reuse BFS
        # results across all of a turn's candidates instead of recomputing
        # per undo. _blot_cache now gets the same treatment: its key carries
        # the enemy-blot set as well as the blocked set, so it too is a pure
        # function of the position and survives the undo. (Clearing it here is
        # what used to make the heuristic prefilter re-run a BFS per piece per
        # candidate -- 94% of a midgame move.)
        self._blocked_key_cache.clear()
        self._blot_key_cache.clear()
        piece = last_move['piece']
        origin_tile = last_move['origin_tile']
        origin_rack = last_move['origin_rack']
        destination = last_move['destination']
        captured_piece = last_move['captured_piece']
        roll = last_move['roll']
        if destination == 'save':
            saved_rack = self.white_saved if piece.player == 'white' else self.black_saved
            saved_rack.remove(piece)
            piece.rack = None
            piece.tile = origin_tile
            origin_tile.pieces.append(piece)
        elif destination == 0:
            saved_rack = self.white_saved if piece.player == 'white' else self.black_saved
            saved_rack.remove(piece)
            piece.rack = None
            piece.tile = origin_tile
            origin_tile.pieces.append(piece)
            # BUGFIX: block-saves marked BOTH dice used on apply, but their
            # roll is 0, which can never match a die number in the generic
            # unmark below -- so undoing a block-save left both dice
            # permanently 'used'. Every search that probed a block-save
            # candidate then ran the rest of its enumeration with corrupted
            # dice (dropping all two-move pairs and mis-encoding used-flags).
            # Block-saves are only legal when both dice are unused
            # (get_valid_moves gates on exactly that), so unconditionally
            # clearing both is exact; repeated clears while unwinding the
            # rest of the block's per-piece entries are harmless no-ops.
            self.dice[0].used = False
            self.dice[1].used = False
        else:
            new_tile = self.get_tile(*destination)
            new_tile.pieces.remove(piece)
            piece.tile = None
            if origin_tile:
                origin_tile.pieces.append(piece)
                piece.tile = origin_tile
            elif origin_rack is not None:
                origin_rack.insert(0, piece)
                piece.rack = origin_rack
            if captured_piece:
                self.home_tile.pieces.remove(captured_piece)
                captured_piece.tile = new_tile
                new_tile.pieces.append(captured_piece)
        if roll == self.dice[0].number and self.dice[0].used:
            self.dice[0].used = False
        elif roll == self.dice[1].number and self.dice[1].used:
            self.dice[1].used = False
        self.firstMove = last_move['firstMove_before']
        if destination == 0:
            # BUGFIX: block-save records carry the SAVED pieces' ids, which
            # belong to the mover's OPPONENT -- restoring current_player to
            # piece.player handed the turn to the wrong side after any
            # probed block-save was undone, so every search that enumerated
            # a block-save candidate then generated the wrong player's
            # moves as follow-ups (and could select an illegal pair).
            # Sibling of the block-save dice bug fixed above.
            self.current_player = 'white' if piece.player == 'black' else 'black'
        else:
            self.current_player = piece.player
        if destination == 'save' or origin_rack is not None:
            self.game_stages[self.current_player] = self.get_game_stage(self.current_player)

    def apply_move(self, move, switch_turn=True):
        piece_id, destination, roll = move
        captured_piece = None
        origin_tile = None
        origin_rack = None
        if move == (0, 0, 0):
            self.firstMove = None
            if switch_turn:
                self.current_player = 'white' if self.current_player == 'black' else 'black'
            return
        if move == (1, 1, 1):                    # call a draw (no piece involved)
            self.draw_called = True
            return
        firstMove_before = self.firstMove
        self._blocked_key_cache.clear()
        self._blot_key_cache.clear()
        piece = self.piece_lookup.get(piece_id)
        if not piece:
            print(f"No piece found for {piece_id}")
            return
        if destination == 0 and roll == 0:      # save ONE opponent piece from a block
            # Peel exactly the named piece off the block into the opponent's own
            # saved rack (a sacrifice by the mover). Costs both dice. Pushes a
            # single undo record; the destination==0 branch of undo_last_move
            # already restores exactly one piece to origin_tile.
            saved_rack = self.white_saved if piece.player == 'white' else self.black_saved
            origin_tile = piece.tile
            origin_tile.pieces.remove(piece)
            piece.tile = None
            piece.rack = saved_rack
            saved_rack.append(piece)
            self.save_move(((piece.player, piece.number), 0, 0), origin_tile, None, None, firstMove_before)
            for die in self.dice:
                die.used = True
            self.game_stages[self.current_player] = self.get_game_stage(self.current_player)
            return
        elif destination == 'save':             # save a piece
            saved_rack = self.white_saved if piece.player == 'white' else self.black_saved
            saved_rack.append(piece)
            if piece.tile:
                piece.tile.pieces.remove(piece)
                origin_tile = piece.tile
            piece.tile = None
            piece.rack = saved_rack
            self.game_stages[self.current_player] = self.get_game_stage(self.current_player)
        else:                                   # move a piece
            ring, pos = destination
            new_tile = self.get_tile(ring, pos)
            if piece.rack:
                origin_rack = piece.rack
                piece.rack.remove(piece)
                piece.rack = None
            if piece.tile:
                piece.tile.pieces.remove(piece)
                origin_tile = piece.tile
            if not self.firstMove:
                self.firstMove = {'piece': piece, 'origin_tile': origin_tile}
            if new_tile.type == 'field' and new_tile.pieces and new_tile.pieces[0].player != piece.player:
                captured_piece = new_tile.pieces.pop()
                captured_piece.tile = self.home_tile
                self.home_tile.pieces.append(captured_piece)
            new_tile.pieces.append(piece)
            piece.tile = new_tile
            if origin_rack is not None:
                self.game_stages[self.current_player] = self.get_game_stage(self.current_player)
        if roll == self.dice[0].number and not self.dice[0].used:
            self.dice[0].used = True
        elif roll == self.dice[1].number and not self.dice[1].used:
            self.dice[1].used = True
        self.save_move(move, origin_tile, origin_rack, captured_piece, firstMove_before)
        if switch_turn and all(die.used for die in self.dice):
            self.switch_turn()

    def get_save_rack(self, player):
        return self.white_saved if player == 'white' else self.black_saved
    
    def get_unentered_rack(self, player):
        return self.white_unentered if player == 'white' else self.black_unentered

    def shortest_route_to_goal(self, piece):
        if piece.can_be_saved():
            return 0
        start_tile = piece.tile if piece.tile else self.home_tile
        blocked_key = self._get_blocked_key(piece.player)
        cache_key = (start_tile.index, piece.player,
                     piece.number if piece.number <= 6 else 'any',
                     blocked_key)
        if cache_key in self._distance_cache:
            return self._distance_cache[cache_key]
        queue = deque([(start_tile, 0)])
        visited = set([start_tile])
        while queue:
            current_tile, distance = queue.popleft()
            for neighbor in current_tile.neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    if neighbor.type == 'save' and (piece.number > 6 or piece.number == neighbor.number):
                        self._distance_cache[cache_key] = distance + 1
                        return distance + 1
                    # blocked_key IS the frozenset of blocked tile indices for
                    # this player, so membership == is_blocked (and is O(1))
                    if neighbor.type not in ['nogo', 'home'] and neighbor.index not in blocked_key:
                        queue.append((neighbor, distance + 1))
        if len(self._distance_cache) > CACHE_MAX:
            self._distance_cache.clear()
        self._distance_cache[cache_key] = float('inf')
        return float('inf')

    def all_goal_distances(self, piece):
        """
        Return a dict {goal_number: distance} for all 6 goal tiles,
        accounting for current blocks. Distance is 0 if already saved
        or currently on that goal tile and can_be_saved.
        Uses a single BFS that collects all reachable goal distances at once.
        Unnumbered pieces (number > 6) can reach any goal tile.
        Numbered pieces can only reach their matching goal tile.
        """
        NUM_GOALS = 6
        goal_numbers = list(range(1, NUM_GOALS + 1))

        # Already saved: distance 0 to all goals (piece is done)
        if piece.rack and (piece.rack is self.white_saved or piece.rack is self.black_saved):
            return {n: 0 for n in goal_numbers}

        # Currently on a reachable goal tile
        if piece.can_be_saved():
            result = {n: float('inf') for n in goal_numbers}
            if piece.tile and piece.tile.type == 'save':
                result[piece.tile.number] = 0
            return result

        # Numbered piece can only reach its own goal. Cache the result dict
        # like the unnumbered path does (callers only read it) -- rebuilding
        # it per call was pure overhead at millions of calls per game.
        if piece.number <= 6:
            blocked_key = self._get_blocked_key(piece.player)
            cache_key = ('num_goals', piece.tile.index if piece.tile else -1,
                         piece.player, piece.number, blocked_key)
            cached = self._distance_cache.get(cache_key)
            if cached is not None:
                return cached
            dist = self.shortest_route_to_goal(piece)
            full_result = {n: (dist if n == piece.number else float('inf'))
                           for n in goal_numbers}
            self._distance_cache[cache_key] = full_result
            return full_result

        # Unnumbered piece: single BFS collecting all 6 goal distances
        blocked_key = self._get_blocked_key(piece.player)
        cache_key = ('all_goals', piece.tile.index if piece.tile else -1,
                     piece.player, blocked_key)
        if cache_key in self._distance_cache:
            return self._distance_cache[cache_key]

        start_tile = piece.tile if piece.tile else self.home_tile
        result = {}
        queue = deque([(start_tile, 0)])
        visited = set([start_tile])

        while queue and len(result) < NUM_GOALS:
            current_tile, distance = queue.popleft()
            for neighbor in current_tile.neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    if neighbor.type == 'save' and neighbor.number not in result:
                        result[neighbor.number] = distance + 1
                    if neighbor.type not in ['nogo', 'home'] and neighbor.index not in blocked_key:
                        queue.append((neighbor, distance + 1))

        # Fill unreachable goals with inf
        full_result = {n: result.get(n, float('inf')) for n in goal_numbers}
        self._distance_cache[cache_key] = full_result
        return full_result

    def get_target_goal_number(self, piece):
        if not hasattr(self, '_goal_cache'):
            self._goal_cache = {}
        cache_key = (piece.player, piece.number, piece.tile.index if piece.tile else -1)
        if cache_key in self._goal_cache:
            return self._goal_cache[cache_key]
        if piece.can_be_saved():
            return None
        start_tile = piece.tile if piece.tile else self.home_tile
        queue = deque([(start_tile, 0)])
        visited = set([start_tile])
        while queue:
            current, _ = queue.popleft()
            for neighbor in current.neighbors:
                if neighbor not in visited:
                    if neighbor.type == 'save' and (piece.number > 6 or piece.number == neighbor.number):
                        self._goal_cache[cache_key] = neighbor.number
                        return neighbor.number
                    visited.add(neighbor)
                    queue.append((neighbor, 0))
        self._goal_cache[cache_key] = None
        return None

    def count_enemy_blots_on_shortest_path(self, piece):
        if piece.can_be_saved():
            return 0
        start_tile = piece.tile if piece.tile else self.home_tile
        # Keyed on everything the count depends on: where we start, which tiles
        # are walled, and where the enemy blots are. That makes it a pure
        # function of the position, so the entry stays valid across the search's
        # apply/undo cycles (it used to be cleared on every undo).
        cache_key = (start_tile.index, piece.player,
                     piece.number if piece.number <= 6 else 'any',
                     self._get_blocked_key(piece.player),
                     self._get_blot_key(piece.player))
        if cache_key in self._blot_cache:
            return self._blot_cache[cache_key]

        queue = deque([(start_tile, 0, 0)])
        visited = set([start_tile])
        while queue:
            current_tile, distance, blot_count = queue.popleft()
            for neighbor in current_tile.neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_blot_count = blot_count
                    if (neighbor.type == 'field' and 
                        len(neighbor.pieces) == 1 and 
                        neighbor.pieces[0].player != piece.player):
                        new_blot_count += 1
                    if neighbor.type == 'save' and (piece.number > 6 or piece.number == neighbor.number):
                        if len(self._blot_cache) > CACHE_MAX:
                            self._blot_cache.clear()
                        self._blot_cache[cache_key] = new_blot_count
                        return new_blot_count
                    if neighbor.type not in ['nogo', 'home'] and not neighbor.is_blocked(piece.player):
                        queue.append((neighbor, distance + 1, new_blot_count))
        if len(self._blot_cache) > CACHE_MAX:
            self._blot_cache.clear()
        self._blot_cache[cache_key] = float('inf')
        return float('inf')

    def count_pieces_reaching_goals(self):
        reachable_counts = [0] * 6
        for piece in self.pieces:
            if not piece.tile or piece.tile == self.home_tile or piece.can_be_saved():
                continue
            for roll in range(1, 7):
                reachable_tiles = self.get_reachable_tiles(piece.tile, roll)
                if piece.number < 7:
                    matching_goal = next((tile for tile in reachable_tiles if tile.type == 'save' and tile.number == piece.number), None)
                    if matching_goal:
                        reachable_counts[roll - 1] += 1
                else:
                    if any(tile.type == 'save' for tile in reachable_tiles):
                        reachable_counts[roll - 1] += 1
        return reachable_counts

    def calculate_dice_roll_utilization_score(self):
        reachable_counts = self.count_pieces_reaching_goals()
        total_pieces = sum(reachable_counts)
        ideal_count = total_pieces / 6 if total_pieces > 0 else 0
        score = sum((count - ideal_count) ** 2 for count in reachable_counts)
        return score

    def apply_last_piece_rule(self):
        for player in ['white', 'black']:
            if self.game_stages[player] != 'endgame':
                continue
            save_rack = self.get_save_rack(player)
            unsaved = [p for p in self.pieces if p.player == player and p.rack != save_rack]
            if len(unsaved) == 1 and unsaved[0].number <= 6:
                piece = unsaved[0]
                old_key = (piece.player, piece.number)
                if old_key in self.piece_lookup:
                    del self.piece_lookup[old_key]
                piece.number = NUM_PIECES + 1
                new_key = (piece.player, piece.number)
                self.piece_lookup[new_key] = piece


def text_interface(board):
    while True:
        print("\nCurrent Board State:")
        print(board)
        valid_moves = board.get_valid_moves()
        print("\nValid moves:")
        for i, move in enumerate(valid_moves):
            piece_id, destination, roll = move
            piece_desc = f"{piece_id}" if piece_id != 0 else "Pass"
            dest_desc = f"{destination}" if destination != "save" else "Save"
            print(f"{i}: Move {piece_desc} to {dest_desc} with roll {roll}")
        choice = input("Enter the number of the move you want to make (or 'q' to quit): ")
        if choice.lower() == 'q':
            print("Exiting...")
            break
        try:
            choice = int(choice)
            if 0 <= choice < len(valid_moves):
                chosen_move = valid_moves[choice]
                board.apply_move(chosen_move)
                print("Move applied!")
            else:
                print("Invalid choice. Please select a valid move number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def random_play(self):
    while True:
        print('Dice:', [die.number for die in self.dice])
        print('Current player:', board.current_player)
        valid_moves = self.get_valid_moves()
        winner, score = self.check_game_over()
        if winner:
            print(f"   GAME OVER! {winner} wins with a score of {score}.")
            break
        save_moves = [move for move in valid_moves if move[1] == 'save']
        if save_moves:
            chosen_move = random.choice(save_moves)
        else:
            prioritized_moves = []
            for move in valid_moves:
                piece_id, destination, roll = move
                if isinstance(destination, tuple):
                    ring, pos = destination
                    tile = self.get_tile(ring, pos)
                    if tile and tile.type == 'save':
                        piece = next((p for p in self.pieces if (p.player, p.number) == piece_id), None)
                        if piece and (piece.number > 6 or piece.number == tile.number):
                            prioritized_moves.append(move)
            valid_moves = [
                move for move in valid_moves
                if move[1] == 'save' or 
                not (isinstance(move[1], tuple) and 
                     self.get_tile(*move[1]) and 
                     self.get_tile(*move[1]).type == 'save' and 
                     any(p.number > 6 or (p.number == self.get_tile(*move[1]).number) 
                         for p in self.get_tile(*move[1]).pieces))
            ]
            if prioritized_moves:
                chosen_move = random.choice(prioritized_moves)
            else:
                chosen_move = random.choice(valid_moves)
        self.apply_move(chosen_move)
        print(f"Applied move: {chosen_move}")
        print("\nCurrent Board State:")
        print(self)

if __name__ == '__main__':
    board = Board()
    while True:
        valid_moves = board.get_valid_moves()
        for i, move in enumerate(valid_moves):
            piece_id, destination, roll = move
            piece_desc = f"{piece_id}" if piece_id != 0 else "Pass"
            dest_desc = f"{destination}" if destination != "save" else "Save"
            print(f"{i}: Move {piece_desc} to {dest_desc} with roll {roll}")
        choice = input("Enter the number of the move you want to make (or 'u' to undo, 'q' to quit): ")
        if choice.lower() == 'q':
            print("Exiting...")
            break
        elif choice.lower() == 'u':
            board.undo_last_move()
            print("Last move undone!")
        else:
            try:
                choice = int(choice)
                if 0 <= choice < len(valid_moves):
                    chosen_move = valid_moves[choice]
                    board.apply_move(chosen_move)
                    print("Move applied!")
                else:
                    print("Invalid choice. Please select a valid move number.")
            except ValueError:
                print("Invalid input. Please enter a number.")