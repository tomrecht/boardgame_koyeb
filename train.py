# "best_weights copy.json" is backup for a seemingly good set of weights

import copy
import random
import math
import time
import json
import os
import glob
from multiprocessing import Pool
from game import Board
from agent import Agent, INITIAL_WEIGHTS
#from scipy import stats

import math  
# for pypy3 which can't use scipy
def ttest_greater(data):
    n = len(data)
    if n < 2:
        return 1.0
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / (n - 1)
    if variance == 0:
        return 0.0 if mean > 0 else 1.0
    t_stat = mean / math.sqrt(variance / n)
    # normal approximation — accurate enough for n > 20
    p_value = 0.5 * (1 - math.erf(t_stat / math.sqrt(2)))
    return p_value


# -------------------------
# CONFIG
# -------------------------

MAX_PAIRS = 24              # pairs per evaluation (= MAX_PAIRS*2 games)
MIN_PAIRS = 6               # minimum pairs before early stopping
OPPONENT_POOL_SIZE = 6      # number of past champions to keep
MUTATION_SIGMA = 0.08
EXPONENT_MUTATION_SIGMA = 0.02
BASE_MUTATION_SIGMA = 0.08
LARGE_MUTATION_PROB = 0.03
CORES = max(1, os.cpu_count() - 1)
SAVE_INTERVAL = 10


# -------------------------
# WEIGHT IO
# -------------------------

def load_weights(filename):
    with open(filename) as f:
        weights = json.load(f)
    for key, value in weights.items():
        if isinstance(value, dict):
            # Convert keys to int ONLY if they are numeric (1-6), leave 'a', 'b', 'midgame' as strings
            weights[key] = {int(k) if k.isdigit() else k: v for k, v in value.items()}
    return weights

def save_weights(weights, filename):
    with open(filename, 'w') as f:
        json.dump(weights, f, indent=2)


def get_best_weights():
    if os.path.exists('best_weights.json'):
        print("Loading best_weights.json")
        weights = load_weights('best_weights.json')
    else:
        files = sorted(glob.glob('weights_gen*.json'))
        if files:
            print(f"Loading {files[-1]}")
            weights = load_weights(files[-1])
        else:
            print("No saved weights found, using INITIAL_WEIGHTS")
            return copy.deepcopy(INITIAL_WEIGHTS)

    # ensure any new weights added to INITIAL_WEIGHTS are present
    for key, value in INITIAL_WEIGHTS.items():
        if key not in weights:
            weights[key] = value
        elif isinstance(value, dict):
            for k, v in value.items():
                if k not in weights[key]:
                    weights[key][k] = v

    return weights

def save_hof(hof, filename='hof.json'):
    # We only save the weights and the score, not the full objects
    serializable_hof = [{"win_rate": wr, "weights": w} for wr, w in hof]
    with open(filename, 'w') as f:
        json.dump(serializable_hof, f, indent=2)

def load_hof(filename='hof.json'):
    if not os.path.exists(filename):
        return []
    with open(filename, 'r') as f:
        data = json.load(f)
    return [(item["win_rate"], item["weights"]) for item in data]

# -------------------------
# MUTATION
# -------------------------

def mutate_value(v):
    if random.random() < LARGE_MUTATION_PROB:
        return v * math.exp(random.gauss(0, 0.5))
    return v * math.exp(random.gauss(0, MUTATION_SIGMA)) + random.gauss(0, 0.01)


def mutate_weights(weights):
    new = copy.deepcopy(weights)
    for key, value in new.items():
        if isinstance(value, dict):
            for k, v in value.items():
                new[key][k] = mutate_value(v)
        elif isinstance(value, (int, float)):
            new[key] = mutate_value(value)
    return new


# -------------------------
# GAME PLAY
# -------------------------

def play_game(agent1, agent2, max_turns=150):
    board = Board()
    agents = {'white': agent1, 'black': agent2}
    turns = 0
    consecutive_passes = 0

    while turns < max_turns:
        winner, score = board.check_game_over()
        if winner:
            return winner, score

        game_state = {
            'currentTurn': board.current_player,
            'dice': [{'value': d.number, 'used': d.used} for d in board.dice],
            'racks': {
                'whiteUnentered': [{'number': p.number, 'color': p.player} for p in board.white_unentered],
                'whiteSaved': [{'number': p.number, 'color': p.player} for p in board.white_saved],
                'blackUnentered': [{'number': p.number, 'color': p.player} for p in board.black_unentered],
                'blackSaved': [{'number': p.number, 'color': p.player} for p in board.black_saved],
            },
            'boardPieces': [{'color': p.player, 'number': p.number,
                            'tile': {'ring': p.tile.ring, 'sector': p.tile.pos}}
                           for p in board.pieces if p.tile]
        }
        board.update_state(game_state)

        current_agent = agents[board.current_player]
        moves = board.get_valid_moves()
        chosen = current_agent.select_move_pair(moves, board, board.current_player)

        if chosen == ((0, 0, 0), (0, 0, 0)):
            consecutive_passes += 1
            if consecutive_passes >= 6:
                white_saved = len(board.white_saved)
                black_saved = len(board.black_saved)
                if white_saved > black_saved:
                    return 'white', white_saved - black_saved
                elif black_saved > white_saved:
                    return 'black', black_saved - white_saved
                return None, 0
        else:
            consecutive_passes = 0

        for move in chosen:
            if move != (0, 0, 0):
                board.apply_move(move, switch_turn=False)
        board.switch_turn()
        turns += 1

    white_saved = len(board.white_saved)
    black_saved = len(board.black_saved)
    if white_saved > black_saved:
        return 'white', white_saved - black_saved
    elif black_saved > white_saved:
        return 'black', black_saved - white_saved
    return None, 0


# -------------------------
# MATCH (paired seeds)
# -------------------------

def play_pair(args):
    challenger_weights, opponent_weights, seed = args
    random.seed(seed)
    challenger = Agent(weights=challenger_weights)
    opponent = Agent(weights=opponent_weights)

    w1, s1 = play_game(challenger, opponent)
    random.seed(seed)
    w2, s2 = play_game(opponent, challenger)

    return w1, s1, w2, s2


# -------------------------
# EVALUATION
# -------------------------

def evaluate_challenger(challenger_weights, opponent_weights, gen_seed):
    rng = random.Random(gen_seed)
    seeds = [rng.randint(0, 2**31) for _ in range(MAX_PAIRS)]
    args = [(challenger_weights, opponent_weights, seed) for seed in seeds]

    challenger_wins = 0
    margins = []
    games_played = 0

    try:
        with Pool(processes=CORES) as pool:
            for w1, s1, w2, s2 in pool.imap_unordered(play_pair, args):
                s1 = s1 or 0
                s2 = s2 or 0
                games_played += 2

                # game 1: challenger=white
                if w1 == 'white':
                    challenger_wins += 1
                    margins.append(s1)
                elif w1 == 'black':
                    margins.append(-s1)
                else:
                    margins.append(0)

                # game 2: challenger=black
                if w2 == 'black':
                    challenger_wins += 1
                    margins.append(s2)
                elif w2 == 'white':
                    margins.append(-s2)
                else:
                    margins.append(0)

                print(f"  {challenger_wins}/{games_played}, mean margin={sum(margins)/len(margins):.2f}")

                if games_played >= MIN_PAIRS * 2:
                                    win_rate = challenger_wins / games_played
                                    mean_margin = sum(margins) / len(margins)
                                    max_possible_wins = challenger_wins + (MAX_PAIRS - games_played // 2) * 2
                                    p_value = ttest_greater(margins)

                                    # can't mathematically reach 55% — hopeless
                                    if max_possible_wins / (MAX_PAIRS * 2) < 0.55:
                                        print(f"  Early stop: can't reach 55% ({challenger_wins}/{games_played})")
                                        pool.terminate()
                                        break

                                    # clearly losing
                                    if win_rate < 0.5 and mean_margin < 0:
                                        print(f"  Early stop: clearly losing ({challenger_wins}/{games_played})")
                                        pool.terminate()
                                        break

                                    # clearly winning
                                    if win_rate > 0.60 and mean_margin > 0 and games_played >= MIN_PAIRS * 3:
                                        p_value_check = ttest_greater(margins)
                                        if p_value_check < 0.05:
                                            print(f"  Early stop: clearly winning ({challenger_wins}/{games_played})")
                                            pool.terminate()
                                        break

                                    # no significant signal
                                    if games_played >= MIN_PAIRS * 3 and p_value > 0.3:
                                        print(f"  Early stop: p={p_value:.3f} ({games_played} games)")
                                        pool.terminate()
                                        break

    except (BrokenPipeError, EOFError):
        pass

    if len(margins) > 1:
        p_value = ttest_greater(margins)
    else:
        p_value = 1.0

    win_rate = challenger_wins / games_played if games_played else 0
    mean_margin = sum(margins) / len(margins) if margins else 0
    # Square root rewards the first few points of margin more than the last few
    fitness = win_rate + 0.05 * (math.copysign(math.sqrt(abs(mean_margin)), mean_margin))

    print(f"  Final: {challenger_wins}/{games_played}, margin={mean_margin:.2f}, p={p_value:.3f}, fitness={fitness:.3f}")

    return challenger_wins, games_played, mean_margin, p_value, fitness

def benchmark_against_initial(current_weights, num_pairs=20):
    initial_weights = copy.deepcopy(INITIAL_WEIGHTS)
    rng = random.Random(42)  # fixed seed for comparability across generations
    seeds = [rng.randint(0, 2**31) for _ in range(num_pairs)]
    args = [(current_weights, initial_weights, seed) for seed in seeds]

    wins = 0
    total = 0
    margins = []

    try:
        with Pool(processes=CORES) as pool:
            for w1, s1, w2, s2 in pool.imap_unordered(play_pair, args):
                s1 = s1 or 0
                s2 = s2 or 0
                total += 2
                if w1 == 'white':
                    wins += 1
                    margins.append(s1)
                else:
                    margins.append(-s1)
                if w2 == 'black':
                    wins += 1
                    margins.append(s2)
                else:
                    margins.append(-s2)
    except (BrokenPipeError, EOFError):
        pass

    win_rate = wins / total if total else 0
    mean_margin = sum(margins) / len(margins) if margins else 0
    print(f"  Benchmark vs INITIAL_WEIGHTS: {wins}/{total} ({win_rate:.1%}), margin={mean_margin:.2f}")
    return win_rate, mean_margin

# -------------------------
# NEW: CMA-ES / MOMENTUM HELPERS
# -------------------------

def get_evolution_path(weights):
    """Initializes a zeroed-out structure matching the weights."""
    path = {}
    for k, v in weights.items():
        if isinstance(v, dict):
            path[k] = {sk: 0.0 for sk in v}
        else:
            path[k] = 0.0
    return path

# --- train.py ---

def mutate_weights_with_momentum(weights, path, learning_rate=0.2):
    new = copy.deepcopy(weights)
    for key, value in new.items():
        if isinstance(value, dict):
            for k, v in value.items():
                if k == 'b':
                    current_sigma = EXPONENT_MUTATION_SIGMA  
                elif k == 'a':
                    current_sigma = BASE_MUTATION_SIGMA
                else:
                    current_sigma = MUTATION_SIGMA
                
                noise = random.gauss(0, current_sigma)
                # Apply momentum as part of the exponent
                mutated_val = v * math.exp(noise + (learning_rate * path[key][k]))
                
                if k == 'b':
                    mutated_val = max(1.0, mutated_val)
                new[key][k] = mutated_val
        elif isinstance(value, (int, float)):
            noise = random.gauss(0, MUTATION_SIGMA)
            # The extra random.gauss at the end is for the additive 'jitter'
            new[key] = value * math.exp(noise + (learning_rate * path[key])) + random.gauss(0, 0.01)
    return new

def update_evolution_path(current_path, old_weights, new_weights):
    """Updates the path based on the log-ratio (percentage change) between old and new champion."""
    updated_path = copy.deepcopy(current_path)
    for key in old_weights:
        if isinstance(old_weights[key], dict):
            for k in old_weights[key]:
                # We use the log of the ratio to keep momentum scale-invariant
                if old_weights[key][k] != 0 and new_weights[key][k] != 0:
                    # abs() handles negative weights correctly
                    ratio = abs(new_weights[key][k] / old_weights[key][k])
                    diff = math.log(ratio)
                    # Decay of 0.8 and clipping prevents runaway values
                    updated_path[key][k] = 0.8 * updated_path[key][k] + 0.2 * diff
                    updated_path[key][k] = max(-2.0, min(2.0, updated_path[key][k]))
        elif isinstance(old_weights[key], (int, float)):
            if old_weights[key] != 0 and new_weights[key] != 0:
                ratio = abs(new_weights[key] / old_weights[key])
                diff = math.log(ratio)
                updated_path[key] = 0.8 * updated_path[key] + 0.2 * diff
                updated_path[key] = max(-2.0, min(2.0, updated_path[key]))
    return updated_path

# -------------------------
# TRAIN (Revised)
# -------------------------

def train():
    current_weights = get_best_weights()
    best_weights = copy.deepcopy(current_weights)
    
    # Initialize CMA-ES Path (Momentum) - Reset on each start
    evolution_path = get_evolution_path(current_weights)
    
    # Load Hall of Fame from disk
    hall_of_fame = load_hof() 
    
    BENCHMARK_INTERVAL = 20
    
    # Initialize pool with HOF agents (if any) and current weights
    opponent_pool = [h[1] for h in hall_of_fame] + [copy.deepcopy(current_weights)]
    # Keep pool within size limits
    if len(opponent_pool) > OPPONENT_POOL_SIZE:
        opponent_pool = opponent_pool[-OPPONENT_POOL_SIZE:]

    gen = 0
    start = time.time()

    print(f"Starting training: cores={CORES}, pairs={MAX_PAIRS}, sigma={MUTATION_SIGMA}")
    if hall_of_fame:
        print(f"Loaded {len(hall_of_fame)} HOF agents from disk.")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            gen_start = time.time()
            print(f"Generation {gen}:")

            # Mutate using the evolution path (momentum)
            challenger_weights = mutate_weights_with_momentum(current_weights, evolution_path)

            # pick a random opponent from the pool (includes HOF and recent winners)
            opponent_weights = random.choice(opponent_pool)

            wins, games_played, mean_margin, p_value, fitness = evaluate_challenger(
                challenger_weights, opponent_weights, gen_seed=gen)

            win_rate = wins / games_played if games_played else 0
            gen_time = time.time() - gen_start

            print(f"  Result: {wins}/{games_played} ({win_rate:.1%}), "
                  f"margin={mean_margin:.2f}, p={p_value:.3f}, fitness={fitness:.3f}, time={gen_time:.1f}s")

            # 1. PERIODIC BENCHMARK & HOF UPDATE
            if gen > 0 and gen % BENCHMARK_INTERVAL == 0:
                print("  --- Running Scheduled Benchmark vs INITIAL_WEIGHTS ---")
                wr, margin = benchmark_against_initial(current_weights)
                
                # Add current champion to HOF and sort by win rate
                hall_of_fame.append((wr, copy.deepcopy(current_weights)))
                hall_of_fame.sort(key=lambda x: x[0], reverse=True)
                hall_of_fame = hall_of_fame[:2] # Keep Top 2
                
                # Persist HOF to disk
                save_hof(hall_of_fame)
                
                # Refresh pool: ensure HOF agents are always in the mix
                latest_champs = [p for p in opponent_pool if p not in [h[1] for h in hall_of_fame]]
                opponent_pool = [h[1] for h in hall_of_fame] + latest_champs[-(OPPONENT_POOL_SIZE-2):]
                print(f"  HOF Updated. Best Benchmark WR: {hall_of_fame[0][0]:.1%}")

            # 2. PROMOTE CHALLENGER
            if (p_value < 0.05 and mean_margin > 0) or \
               (win_rate > 0.60 and mean_margin > 0 and games_played >= MIN_PAIRS * 3):
                
                print(f"  *** New champion! ***")
                
                # Update Momentum Path
                evolution_path = update_evolution_path(evolution_path, current_weights, challenger_weights)
                
                current_weights = challenger_weights
                opponent_pool.append(copy.deepcopy(current_weights))
                
                if len(opponent_pool) > OPPONENT_POOL_SIZE:
                    opponent_pool.pop(0)

                save_weights(current_weights, 'best_weights.json')
                best_weights = copy.deepcopy(current_weights)

            save_weights(current_weights, 'weights_current.json')

            if gen % SAVE_INTERVAL == 0:
                save_weights(current_weights, f'weights_gen{gen}.json')
                total_time = time.time() - start
                print(f"  Checkpoint saved. Total time: {total_time/3600:.1f}h")

            gen += 1
            print()

    except KeyboardInterrupt:
        print("\nTraining stopped by user")
        save_weights(best_weights, 'best_weights.json')
        # One last HOF save just in case
        if hall_of_fame:
            save_hof(hall_of_fame)

    return best_weights

if __name__ == '__main__':
    train()