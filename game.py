import random
from collections import deque
import json
import itertools

NUM_PIECES = 12

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
        if self.rack and self.rack == self.board.white_saved or self.rack == self.board.black_saved:
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
        self.number = number  # for goal tiles
        self.index = None

    def __repr__(self):
        return f"{self.type}({self.ring}, {self.pos})"
        return f"Tile(type={self.type}, ring={self.ring}, pos={self.pos}, number={self.number})"
    
    def is_blocked(self, player = None):
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
        self._distance_cache = {}
        self._reachable_cache = {}

        self.endgame_reward_applied = {'white': False, 'black': False}
        self.offgoals = {'white': 0, 'black': 0}

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
            random.shuffle(pieces)  # Shuffle the pieces randomly

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
            number = value.get('number')  # Retrieve number if it's a save tile
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

    def update_state(self, game_state_details):
        # Set the current turn
        self.current_player = game_state_details['currentTurn']

        # Clear the board pieces
        self.clear()
        
        # Set dice values and used status
        for die, die_details in zip(self.dice, game_state_details['dice']):
            die.number = die_details['value']
            die.used = die_details['used']

        # Function to place pieces in their respective racks
        def place_pieces_in_rack(rack, pieces_details, player):
            rack.clear()
            for piece_details in pieces_details:
                piece = Piece(player, piece_details['number'], self)
                self.pieces.append(piece)
                rack.append(piece)
                piece.rack = rack
        
        # Place pieces in the unentered and saved racks
        place_pieces_in_rack(self.white_unentered, game_state_details['racks']['whiteUnentered'], 'white')
        place_pieces_in_rack(self.white_saved, game_state_details['racks']['whiteSaved'], 'white')
        place_pieces_in_rack(self.black_unentered, game_state_details['racks']['blackUnentered'], 'black')
        place_pieces_in_rack(self.black_saved, game_state_details['racks']['blackSaved'], 'black')

        
        # Place pieces on the board
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

       #     print('Placed piece:', piece, 'on tile:', tile)
            
            if 'reachableBySum' in piece_details:
                piece.reachable_by_sum = [self.get_tile(t['ring'], t['sector']) for t in piece_details['reachableBySum']]
                self.firstMove = {'piece': piece, 'origin_tile': tile}

        self.assign_piece_indices()
        self.piece_lookup = {(p.player, p.number): p for p in self.pieces}
        self.game_stages['white'] = self.get_game_stage('white')
        self.game_stages['black'] = self.get_game_stage('black')
        self.apply_last_piece_rule()

    def assign_tile_indices(self):
        for i in range(len(self.tiles)):
            self.tiles[i].index = i

    def assign_piece_indices(self):
        # Sort the pieces list by color (white then black) and then by their number
        self.pieces.sort(key=lambda piece: (piece.player != 'white', piece.number))
        # Assign the indices
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
        self._distance_cache = {}
        self._reachable_cache = {}
        self.firstMove = None  
        for die in self.dice:
            die.roll()
        self.current_player = 'white' if self.current_player == 'black' else 'black'
        self.apply_last_piece_rule()

    def check_game_over(self):
      #  TOTAL_PIECES = len(self.pieces) // 2 
        white_saved_count = len(self.white_saved)
        black_saved_count = len(self.black_saved)
        
        if white_saved_count == NUM_PIECES:
            black_unsaved_count = NUM_PIECES - black_saved_count
            return 'white', black_unsaved_count
        
        if black_saved_count == NUM_PIECES:
            white_unsaved_count = NUM_PIECES - white_saved_count
            return 'black', white_unsaved_count
        
        return None, None  # No winner yet

    def get_unentered_piece(self):
        unentered_rack = self.white_unentered if self.current_player == 'white' else self.black_unentered
        if len(unentered_rack) > 0:
            return unentered_rack[0]
        return None

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
            return False  # can't save pieces in the opening

        current_tile = piece.tile
        if current_tile and current_tile.type == 'save' and (piece.number > 6 or piece.number == current_tile.number):
            if self.game_stages[piece.player] == 'endgame':
                if piece.number > 6:

                    #these 2 lines are meant to fix the bug where agent tries to save with a too low roll; if it messes up anything, delete  and uncomment the following line
                    highest_occupied_goal_number = max((tile.number for tile in self.tiles if tile.type == 'save' and len(tile.pieces) > 0 and any(p.player == piece.player for p in tile.pieces)), default=-1)
                    valid_dice = [die for die in self.dice if (not die.used) and (die.number == current_tile.number or (die.number > current_tile.number and current_tile.number == highest_occupied_goal_number))]
                
                    #highest_occupied_goal_number = max((tile.number for tile in self.tiles if tile.type == 'save' and len(tile.pieces) > 0 and any(p.player == piece.player for p in tile.pieces)), default=0)
                    #valid_dice = [die for die in self.dice if (not die.used) and (die.number == current_tile.number or (die.number > current_tile.number and current_tile.number >= highest_occupied_goal_number))]
                else:
                    valid_dice = [die for die in self.dice if (not die.used) and die.number == current_tile.number]
            else:
                valid_dice = [die for die in self.dice if (not die.used) and die.number == current_tile.number]

            if valid_dice:
                matching_die = next((die for die in valid_dice if die.number == current_tile.number), None)
                if matching_die:
                    die = matching_die
                else:
                    die = max(valid_dice, key=lambda die: die.number)
                return die.number
            else:
                return False  # The piece cannot be saved with the current dice rolls

    def get_reachable_tiles(self, start_tile, steps):
        blocked_tiles = frozenset(
            t.index for t in self.tiles
            if t.type == 'field'
            and len(t.pieces) > 1
        )
        cache_key = (start_tile.index, steps, blocked_tiles)

        if cache_key in self._reachable_cache:
            return self._reachable_cache[cache_key]

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

        result = list(set(reachable_tiles))
        self._reachable_cache[cache_key] = result
        return result

    def get_reachable_tiles_by_dice(self, piece):   
        reachable_tiles = {self.dice[0].number: [], self.dice[1].number: []}
        
        if piece.rack and piece.rack in [self.white_unentered, self.black_unentered]:   # if an unentered piece, start from the home tile
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
            save_roll = self.get_saving_die(piece)
            if save_roll:             
                reachable_tiles[save_roll].append('save') 

        piece.reachable_tiles = reachable_tiles

    def get_valid_moves(self, mask_offgoals = False):
        self.game_stages[self.current_player] = self.get_game_stage(self.current_player)  
        if self.dice[0].used and self.dice[1].used:
            return []

        # if must move captured piece(s), do so
        captured_pieces = [piece for piece in self.home_tile.pieces if piece.player == self.current_player]
        if captured_pieces:
            for piece in captured_pieces:
                self.get_reachable_tiles_by_dice(piece)
            self.destinations_by_piece = {piece: piece.reachable_tiles for piece in captured_pieces}

        # if must move unentered piece, do so
        elif self.must_move_unentered():
            piece = self.get_unentered_piece()
            self.get_reachable_tiles_by_dice(piece)
            self.destinations_by_piece = {piece: piece.reachable_tiles}
            
        else:
            player_pieces = [p for p in self.pieces if p.player == self.current_player and p.tile and p.tile.type in ['field', 'save']]

            # check if there's an unentered piece which can enter, and if so add it to the list of pieces
            unentered_piece = self.get_unentered_piece()
            if unentered_piece:
                player_pieces.append(unentered_piece)

            for piece in player_pieces:
                self.get_reachable_tiles_by_dice(piece)
        
            self.destinations_by_piece = {piece: piece.reachable_tiles for piece in player_pieces}

        # transform the dictionary so that items are tuples of (piece, tile, roll)
        tuples_list = []
        for piece, moves in self.destinations_by_piece.items():
            for roll, destinations in moves.items():
                if destinations:  # Ignore empty destinations
                    for destination in destinations:
                        
                        if destination == 'save':
                            tuples_list.append(((piece.player, piece.number), destination, roll))
                     #   elif mask_offgoals and piece.can_be_saved() and (piece.number <=6 or roll != 4 or destination.type != 'save'):
                      #      continue   # don't include offgoal moves
                        else:
                            tuples_list.append(((piece.player, piece.number), (destination.ring, destination.pos), roll))

        if not captured_pieces and not self.must_move_unentered():
            tuples_list.append((0, 0, 0))  # add a pass move

        #   add tuples of form (piece, 0, 0) for saving opponent's piece 
            if (not self.dice[0].used and not self.dice[1].used 
                    and self.game_stages[self.current_player] != 'opening'
                    and not captured_pieces
                    and not self.must_move_unentered()):
                opponent_pieces_on_blocks = [p for p in self.pieces 
                    if p.player != self.current_player 
                    and p.tile 
                    and p.tile.type == 'field'
                    and len(p.tile.pieces) > 1]
                seen_tiles = set()
                for piece in opponent_pieces_on_blocks:
                    if piece.tile not in seen_tiles:
                        seen_tiles.add(piece.tile)
                        tuples_list.append(((piece.player, piece.number), 0, 0))
  
        return tuples_list
    
    def save_move(self, move, origin_tile = None, origin_rack = None, captured_piece = None):
        piece_id, destination, roll = move

        move_to_save = dict()
        move_to_save['piece'] = self.piece_lookup.get(piece_id)
        move_to_save['origin_tile'] = origin_tile
        move_to_save['origin_rack'] = origin_rack
        move_to_save['destination'] = destination
        move_to_save['captured_piece'] = captured_piece
        move_to_save['roll'] = roll

        self.moves.append(move_to_save)

    def undo_last_move(self):
        if not self.moves:
            return

        last_move = self.moves.pop()
        piece = last_move['piece']
        origin_tile = last_move['origin_tile']
        origin_rack = last_move['origin_rack']
        destination = last_move['destination']
        captured_piece = last_move['captured_piece']
        roll = last_move['roll']

        # Undo the move
        if destination == 'save':
            saved_rack = self.white_saved if piece.player == 'white' else self.black_saved
            saved_rack.remove(piece)
            piece.rack = None
            piece.tile = origin_tile
            origin_tile.pieces.append(piece)
        elif destination == 0:    # undo saving opponent's block
            saved_rack = self.white_saved if piece.player == 'white' else self.black_saved
            saved_rack.remove(piece)
            piece.rack = None
            piece.tile = origin_tile
            origin_tile.pieces.append(piece)
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

            if captured_piece:      # undo the capture
                new_tile.pieces.append(captured_piece)
                captured_piece.tile = new_tile

        # Mark the die as unused
        if roll == self.dice[0].number and self.dice[0].used:
            self.dice[0].used = False
        elif roll == self.dice[1].number and self.dice[1].used:
            self.dice[1].used = False

        # if exactly one die is now unused, this was the first move, so clear self.firstMove
        # changed 1 to 2 to see if this fixes the bug where agent moving out the first piece ignored the rule -- evaluate
        if sum(not die.used for die in self.dice) == 2:
            self.firstMove = None
        

        self.current_player = piece.player
      
        if destination == 'save' or origin_rack is not None:
            self.game_stages[self.current_player] = self.get_game_stage(self.current_player)

        self.check_game_over()
    
    def apply_move(self, move, switch_turn = True):
        piece_id, destination, roll = move

        captured_piece = None
        origin_tile = None
        origin_rack = None

        # Handle the pass move (0, 0, 0)
        if move == (0, 0, 0):
            self.firstMove = None  # Reset first move for the next turn
            if switch_turn:
                self.current_player = 'white' if self.current_player == 'black' else 'black'
            return

        # Find the piece object
        piece = self.piece_lookup.get(piece_id)
        if not piece:
            print(f"No piece found for {piece_id}")
            return

        # Handle saving opponent's piece

        if destination == 0 and roll == 0:
            saved_rack = self.white_saved if piece.player == 'white' else self.black_saved
            origin_tile = piece.tile
            block_pieces = origin_tile.pieces[:]  # copy since we're modifying the list
            
            for i, p in enumerate(block_pieces):
                origin_tile.pieces.remove(p)
                p.tile = None
                p.rack = saved_rack
                saved_rack.append(p)
                self.save_move(((p.player, p.number), 0, 0), origin_tile, None, None)
            
            for die in self.dice:
                die.used = True
            
            self.game_stages[self.current_player] = self.get_game_stage(self.current_player)
            return

        # Handle saving a piece
        elif destination == 'save':
            saved_rack = self.white_saved if piece.player == 'white' else self.black_saved
            saved_rack.append(piece)
            if piece.tile:
                piece.tile.pieces.remove(piece)
                origin_tile = piece.tile
            piece.tile = None
            piece.rack = saved_rack

            # did we move to endgame?
            self.game_stages[self.current_player] = self.get_game_stage(self.current_player)

        else:
            # Handle moving to a new tile
            ring, pos = destination
            new_tile = self.get_tile(ring, pos)

            # Remove the piece from its current location (rack or tile)
            if piece.rack:
                origin_rack = piece.rack
                piece.rack.remove(piece)
                piece.rack = None
            if piece.tile:
                piece.tile.pieces.remove(piece)
                origin_tile = piece.tile
            
            # Set the first move if not set already
            if not self.firstMove:
                self.firstMove = {'piece': piece, 'origin_tile': piece.tile}

            # Check if we are capturing an opponent piece (only on field tiles)
            if new_tile.type == 'field' and new_tile.pieces and new_tile.pieces[0].player != piece.player:
                captured_piece = new_tile.pieces.pop()
                captured_piece.tile = self.home_tile
                self.home_tile.pieces.append(captured_piece)

            # Move the piece to the new tile
            new_tile.pieces.append(piece)
            piece.tile = new_tile

            if origin_rack is not None: # did we move to midgame?
                self.game_stages[self.current_player] = self.get_game_stage(self.current_player)

        # Mark the die as used
        if roll == self.dice[0].number and not self.dice[0].used:
            self.dice[0].used = True
        elif roll == self.dice[1].number and not self.dice[1].used:
            self.dice[1].used = True

#        self.game_stages[self.current_player] = self.get_game_stage(self.current_player)

        self.save_move(move, origin_tile, origin_rack, captured_piece)

        # Switch to the next player if both dice are used
        if switch_turn and all(die.used for die in self.dice):
            print('switching turn')
            self.switch_turn()

    def get_save_rack(self, player):
        return self.white_saved if player == 'white' else self.black_saved
    
    def get_unentered_rack(self, player):
        return self.white_unentered if player == 'white' else self.black_unentered

    def shortest_route_to_goal(self, piece):
        if piece.can_be_saved():
            return 0

        start_tile = piece.tile if piece.tile else self.home_tile

        blocked_tiles = frozenset(
            t.index for t in self.tiles
            if t.type == 'field'
            and len(t.pieces) > 1
            and t.pieces[0].player != piece.player
        )
        cache_key = (start_tile.index, piece.player, piece.number if piece.number <= 6 else 'any', blocked_tiles)

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
                    if neighbor.type not in ['nogo', 'home'] and not neighbor.is_blocked(piece.player):
                        queue.append((neighbor, distance + 1))

        self._distance_cache[cache_key] = float('inf')
        return float('inf')
    
    def count_pieces_reaching_goals(self):
        # Initialize counters for each possible die roll (1-6)
        reachable_counts = [0] * 6
        
        # Iterate over all pieces on the board
        for piece in self.pieces:
            # Skip pieces on racks, on the home tile, or unnumbered pieces already on a goal
            if not piece.tile or piece.tile == self.home_tile or piece.can_be_saved():
                continue
            
            # Calculate reachable tiles for each roll from 1 to 6
            for roll in range(1, 7):
                reachable_tiles = self.get_reachable_tiles(piece.tile, roll)
                
                # Check if the piece can reach its goal with this roll
                if piece.number < 7:  # Numbered piece must reach its specific goal
                    matching_goal = next((tile for tile in reachable_tiles if tile.type == 'save' and tile.number == piece.number), None)
                    if matching_goal:
                        reachable_counts[roll - 1] += 1
                else:  # Unnumbered piece can reach any goal
                    if any(tile.type == 'save' for tile in reachable_tiles):
                        reachable_counts[roll - 1] += 1

        return reachable_counts

    def calculate_dice_roll_utilization_score(self):
        # Get the counts of pieces that can reach goals for each roll from 1 to 6
        reachable_counts = self.count_pieces_reaching_goals()

        # Calculate the ideal number of pieces per roll (assuming equal distribution)
        total_pieces = sum(reachable_counts)
        ideal_count = total_pieces / 6 if total_pieces > 0 else 0

        # Calculate the Dice Roll Utilization Score
        score = sum((count - ideal_count) ** 2 for count in reachable_counts)

        return score

    # if last piece is a numbered one on its goal, unnumber it
    def apply_last_piece_rule(self):
        for player in ['white', 'black']:
            if self.game_stages[player] != 'endgame':
                continue
            save_rack = self.get_save_rack(player)
            unsaved = [p for p in self.pieces if p.player == player and p.rack != save_rack]
            if len(unsaved) == 1 and unsaved[0].number <= 6:
                old_key = (unsaved[0].player, unsaved[0].number)
                unsaved[0].number = NUM_PIECES + 1
                new_key = (unsaved[0].player, unsaved[0].number)


    def get_all_possible_moves(self):
        destination_tiles = [tile.index for tile in self.tiles if tile.type in ['field','save']]
        pieces = range(len(self.pieces))
        all_possible_moves = list(itertools.product(pieces, destination_tiles))
        all_possible_moves.insert(0, (0, 0))  # Add the tuple (0,0,0) for passing
        for destination in destination_tiles:
            all_possible_moves.append((0, destination))  # Add an extra tuple for saving each tile: form (0, tile_index)
        return all_possible_moves
        
    def encode_state(self):

        def normalize(value, min_val, max_val):
            return (value - min_val) / (max_val - min_val)

        player = self.current_player
        opponent = 'white' if player == 'black' else 'black'
        player_saved_rack = self.white_saved if player == 'white' else self.black_saved
        opponent_saved_rack = self.white_saved if player == 'black' else self.black_saved
        player_unentered_rack = self.white_unentered if player == 'white' else self.black_unentered
        opponent_unentered_rack = self.white_unentered if player == 'black' else self.black_unentered
        player_pieces = [piece for piece in self.pieces if piece.player == player]
        opponent_pieces = [piece for piece in self.pieces if piece.player == opponent]

        state = []

        for piece in player_pieces:
            if piece in player_unentered_rack:
                rack_position = player_unentered_rack.index(piece)
                state.append(normalize(rack_position, 0, 100)) 
            elif piece in player_saved_rack:
                state.append(1)
            else:
                tile_position = piece.tile.index + 28           # offset by length of unentered racks
                state.append(normalize(tile_position, 0, 100))

        for piece in opponent_pieces:
            if piece in opponent_unentered_rack:
                rack_position = opponent_unentered_rack.index(piece) + NUM_PIECES   # offset by player's unentered rack
                state.append(normalize(rack_position, 0, 100)) 
            elif piece in opponent_saved_rack:
                state.append(1)
            else:
                tile_position = piece.tile.index + 28           
                state.append(normalize(tile_position, 0, 100))

        for rack in [player_saved_rack, opponent_saved_rack]:
            state.append(normalize(len(rack), 0, NUM_PIECES))

        for tile in self.tiles:
            if tile.type == 'field':
                state.append(int(tile.is_blocked() == True))

        for player in [player, opponent]:
            stage = self.game_stages[player]
            state.append(0 if stage == 'opening' else 0.5 if stage == 'midgame' else 1)

        for die in self.dice:
            state.append(normalize(die.number, 1, 6) if die.used else 0)

        return state

    def step(self, move_and_player, transition_factor=0.1):

        piece, destination, roll, player = move_and_player
        move = (piece, destination, roll)

        if move == (0, 0, 0):  # pass move
            self.apply_move(move)
            next_state = self.encode_state()
            reward = 0
            done = False
            return next_state, reward, done
        
        piece_object = next((p for p in self.pieces if (p.player, p.number) == piece), None)
        start_distance_to_goal = self.shortest_route_to_goal(piece_object)
        start_within_reach = True if start_distance_to_goal <= 6 else False

        # intermediate rewards: before move
        intermediate_reward = 0
        if destination == 'save':   # save pieces
            intermediate_reward += 5000             
            if piece_object.number <= 6:
                intermediate_reward += piece_object.number * 1000
        
        if isinstance(destination, tuple):   
            tile = self.get_tile(*destination)
            if piece_object.can_be_saved() and tile.type != 'save':  # don't move a piece that can be saved, except to another save tile
                self.offgoals[player] += 1
                print("Offgoal. Move:", move, self.game_stages[player], piece, piece_object.rack, piece_object.tile, tile, roll)
                intermediate_reward -= 30000
                if piece_object.number <= 6:
                    intermediate_reward -= piece_object.number * 6000
            
            if tile.pieces and tile.pieces[0].player != player:  # capture
                    intermediate_reward += 500
            elif tile.pieces and len(tile.pieces) == 1:   # create block
                    intermediate_reward += 500

        # apply move and check for game over
        
        self.apply_move(move)

        winner, score = self.check_game_over()
        next_state = self.encode_state()

        if score is None:
            score = 0  # Ensure score is numeric
        
        if winner:
            print(f"*** Game over! {winner} wins with a score of {score}.")
            done = True
            reward = score * 1000000 if winner == player else score * -1000000
            return next_state, reward, done

        # intermediate rewards: after move
        if isinstance(destination, tuple):   
            if piece_object.can_be_saved():  #  moved to goal
                intermediate_reward += 5000             
                if piece_object.number <= 6:
                    intermediate_reward += piece_object.number * 1000
                self.game_stages[player] = self.get_game_stage(player)
                if self.game_stages[player] == 'endgame' and not self.endgame_reward_applied[player]:   # enter endgame for first time
                    intermediate_reward += 50000
                    self.endgame_reward_applied[player] = True
            else:   # did piece move into/out of reach?
                end_distance_to_goal = self.shortest_route_to_goal(piece_object)
                end_within_reach = True if end_distance_to_goal <= 6 else False
                if not start_within_reach and end_within_reach:
                    intermediate_reward += 1000
                    if piece_object.number <= 6:
                        intermediate_reward -= piece_object.number * 200
                elif start_within_reach and not end_within_reach:
                    intermediate_reward -= 1000
                    if piece_object.number <= 6:
                        intermediate_reward -= piece_object.number * 200

        # Blend intermediate and final rewards
        reward = (1 - transition_factor) * intermediate_reward + transition_factor * score

        done = False   

        return next_state, reward, done
    



def text_interface(board):
    while True:
        # Display the current state of the board
        print("\nCurrent Board State:")
        print(board)

        # Get valid moves
        valid_moves = board.get_valid_moves()

        # List valid moves
        print("\nValid moves:")
        for i, move in enumerate(valid_moves):
            piece_id, destination, roll = move
            piece_desc = f"{piece_id}" if piece_id != 0 else "Pass"
            dest_desc = f"{destination}" if destination != "save" else "Save"
            print(f"{i}: Move {piece_desc} to {dest_desc} with roll {roll}")

        # Prompt the user for a choice
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
        # Get valid moves
        valid_moves = self.get_valid_moves()

        # Check for the end of the game
        winner, score = self.check_game_over()
        if winner:
            print(f"   GAME OVER! {winner} wins with a score of {score}.")
            break

        # Check for save moves and prioritize them
        save_moves = [move for move in valid_moves if move[1] == 'save']
        if save_moves:
            chosen_move = random.choice(save_moves)
        else:
            # Check for moves that place a piece on a save goal where it can be saved
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

            # Filter out moves that involve moving pieces already on save goals where they can be saved
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

        # Apply the move
        self.apply_move(chosen_move)
        print(f"Applied move: {chosen_move}")

        # Display the current state of the board
        print("\nCurrent Board State:")
        print(self)

import time



if __name__ == '__main__':
    board = Board()

    while True:
     #   print("\nCurrent Board State:")
     #   print(board)

        valid_moves = board.get_valid_moves()
     #   print("\nValid moves:")
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

# bug: agent sometimes passes when it shouldn't, ditto offgoals
# and passes when it has a captured piece so passing is illegal?
# bug ignoring shortest move rule
# rare bug moving completely blocked piece! agent moved its 6 from tile 3-11, where it was completely blocked and had no legal moves, to tile 2-8 on a roll of 2, when that tile is 4 tiles away so illegal twice!
# in two-player game, a player can offer to end the game with a proposed score