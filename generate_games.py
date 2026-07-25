"""
generate_games.py — Generate heuristic vs heuristic games for GNN pretraining.

Saves positions in the same schema as positions_with_moves.jsonl.
Uses multiprocessing for speed.

Usage:
    python generate_games.py --games 6000 --out training_data/generated_positions.jsonl
    python generate_games.py --games 3000 --out training_data/generated_positions.jsonl --workers 4
"""

import argparse
import json
import os
import random
import time
from multiprocessing import Pool, cpu_count

from game import Board
from agent import Agent, get_weights

MAX_TURNS = 120  # Hard cap — genuine games never reach this
SCHEMA_VERSION = 1


def get_game_stage(board, player):
    return board.game_stages.get(player, 'unknown')


def serialize_state(board):
    """Serialize board state in the same format as the JS client."""
    return {
        'currentTurn': board.current_player,
        'dice': [{'value': d.number, 'used': d.used} for d in board.dice],
        'racks': {
            'whiteUnentered': [{'color': 'white', 'number': p.number}
                               for p in board.white_unentered],
            'whiteSaved':     [{'color': 'white', 'number': p.number}
                               for p in board.white_saved],
            'blackUnentered': [{'color': 'black', 'number': p.number}
                               for p in board.black_unentered],
            'blackSaved':     [{'color': 'black', 'number': p.number}
                               for p in board.black_saved],
        },
        'boardPieces': [
            {
                'color':  p.player,
                'number': p.number,
                'tile':   {'ring': p.tile.ring, 'sector': p.tile.pos},
            }
            for p in board.pieces if p.tile is not None
        ],
    }


def play_and_record(seed):
    """
    Play one heuristic vs heuristic game and return a list of position records.
    Returns (positions, did_timeout) so callers can detect the 150-turn bug.
    """
    random.seed(seed)
    board = Board()
    white_agent = Agent(weights=get_weights())
    black_agent = Agent(weights=get_weights())
    agents = {'white': white_agent, 'black': black_agent}

    positions = []   # list of dicts with raw_state, player, game_stage, move_index
    turn = 0

    while turn < MAX_TURNS:
        winner, score = board.check_game_over()
        if winner:
            # Label all positions with outcome
            timestamp = int(time.time())
            total = len(positions)
            records = []
            for i, pos in enumerate(positions):
                ply_from_end = total - i
                player = pos['player']
                final_score = score if player == winner else -score
                records.append({
                    'schema_version': SCHEMA_VERSION,
                    'game_id':        f'gen_{seed}',
                    'player':         player,
                    'source':         'heuristic',
                    'game_stage':     pos['game_stage'],
                    'move_index':     pos['move_index'],
                    'move_pair':      pos['move_pair'],
                    'raw_state':      pos['raw_state'],
                    'final_score':    final_score,
                    'ply_from_end':   ply_from_end,
                    'timestamp':      timestamp,
                })
            return records, False

        player = board.current_player
        agent = agents[player]
        moves = board.get_valid_moves()

        # Snapshot state BEFORE moves are applied (turn-start state)
        raw_state = serialize_state(board)
        game_stage = get_game_stage(board, player)

        chosen = agent.select_move_pair(moves, board, player)

        # select_move_pair returns either a pair (m1, m2) or occasionally
        # a single move tuple in edge cases — normalise to a list
        if (isinstance(chosen, tuple)
                and len(chosen) == 2
                and isinstance(chosen[0], tuple)
                and len(chosen[0]) == 3):
            move_list = list(chosen)
        elif isinstance(chosen, tuple) and len(chosen) == 3:
            move_list = [chosen]
        else:
            move_list = list(chosen)

        for move in move_list:
            if move != (0, 0, 0):
                board.apply_move(move, switch_turn=False)

        positions.append({
            'player':     player,
            'game_stage': game_stage,
            'move_index': turn,
            'move_pair':  move_list,
            'raw_state':  raw_state,
        })

        board.switch_turn()
        turn += 1

    # Reached MAX_TURNS without finishing — report timeout
    return [], True


def worker(args):
    seed, = args
    records, timed_out = play_and_record(seed)
    return records, timed_out, seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games',   type=int, default=3000)
    parser.add_argument('--out',     type=str, default='training_data/generated_positions.jsonl')
    parser.add_argument('--workers', type=int, default=max(1, cpu_count() - 1))
    parser.add_argument('--seed',    type=int, default=42,
                        help='Base random seed (each game uses seed+i)')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    seeds = [(args.seed + i,) for i in range(args.games)]

    print(f"Generating {args.games} games using {args.workers} workers...")
    print(f"Output: {args.out}")
    print(f"MAX_TURNS cap: {MAX_TURNS} (any game hitting this is a bug)")

    start = time.time()
    total_positions = 0
    timeouts = 0
    games_done = 0

    with open(args.out, 'a') as f:
        with Pool(processes=args.workers) as pool:
            for records, timed_out, seed in pool.imap_unordered(worker, seeds, chunksize=1):
                games_done += 1
                if timed_out:
                    timeouts += 1
                    print(f"  WARNING: game seed={seed} hit MAX_TURNS cap — possible bug")
                else:
                    for rec in records:
                        f.write(json.dumps(rec) + '\n')
                    total_positions += len(records)

                if games_done == 1:
                    elapsed = time.time() - start
                    print(f"  First game done in {elapsed:.1f}s ({len(records) if not timed_out else 0} positions)")

                if games_done % 10 == 0:
                    elapsed = time.time() - start
                    rate = games_done / elapsed
                    remaining = (args.games - games_done) / rate
                    print(f"  {games_done}/{args.games} games, "
                          f"{total_positions} positions, "
                          f"{rate:.1f} games/s, "
                          f"~{remaining/60:.0f}m remaining, "
                          f"{timeouts} timeouts")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} minutes")
    print(f"  Games:     {games_done} ({timeouts} timeouts discarded)")
    print(f"  Positions: {total_positions}")
    print(f"  Rate:      {games_done/elapsed:.1f} games/s")
    if timeouts > 0:
        print(f"\n  WARNING: {timeouts} games timed out. Investigate before using this data.")


if __name__ == '__main__':
    main()