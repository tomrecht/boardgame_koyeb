"""Champion arena: dynamic scheduling targeting under-played champions with margin-based Elo.

Prioritizes champions with fewer played games so their Elo ratings stabilize faster.
Maintains CRN (Common Random Numbers) and color swapping per match index.
"""
import os, sys, json, time, random, itertools
from collections import defaultdict
import multiprocessing as mp

REPO = os.path.dirname(os.path.abspath(__file__))
CHAMPIONS = {
    'aux_iter14':  f'{REPO}/td_champion_July21_aux_iter14.pt',
    'iter4':  f'{REPO}/td_champion_July17_iter4.pt',
    'iter10': f'{REPO}/td_champion_July18_iter10.pt',
    'iter14': f'{REPO}/td_champion_July19_iter14.pt',
    'symaug_iter6': f'{REPO}/symaug_champ_july27_iter6.pt',
    'symaug_almost11': f'{REPO}/symaug_almostchamp_july27_iter11.pt',
}
for kv in filter(None, os.environ.get('CHAMPS_EXTRA', '').split(',')):
    tag, path = kv.split('=', 1)
    CHAMPIONS[tag.strip()] = path.strip()

_only = [t.strip() for t in os.environ.get('CHAMPS', '').split(',') if t.strip()]
if _only:
    CHAMPIONS = {t: CHAMPIONS[t] for t in _only}

N_WORKERS = int(os.environ.get('N_WORKERS', '3'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', str(N_WORKERS * 2)))
RESULTS = os.environ.get('RESULTS', f'{REPO}/arena.jsonl')
STANDINGS = os.environ.get('STANDINGS', f'{REPO}/arena_standings.txt')
SEED_BASE = 4_400_000
MAX_TURNS, STUCK_LIMIT = 200, 60

# -------------------- worker (CPU-only) --------------------
_CACHE = {}

def _agent(path):
    if path not in _CACHE:
        import torch
        from network import BoardGNN
        from agent_gnn import GNNAgent
        m = BoardGNN()
        m.load_state_dict(torch.load(path, map_location='cpu'), strict=False)
        m.eval()
        _CACHE[path] = GNNAgent(model=m)
    return _CACHE[path]

def _worker_init():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['BOARDGAME_DEVICE'] = 'cpu'
    import torch
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    import network
    network.DEVICE = torch.device('cpu')

def _play(seed, wag, bag):
    from game import Board
    random.seed(seed)
    board = Board()
    agents = {'white': wag, 'black': bag}
    last_saved, since = 0, 0
    for _ in range(MAX_TURNS):
        winner, score = board.check_game_over()
        if winner:
            return winner, score
        if board.draw_callable:
            return None, 0
        cur = len(board.white_saved) + len(board.black_saved)
        if cur > last_saved:
            last_saved, since = cur, 0
        elif last_saved > 0:
            since += 1
        if last_saved > 0 and since >= STUCK_LIMIT:
            return None, 0
        player = board.current_player
        chosen = agents[player].select_move_pair(
            list(board.get_valid_moves()), board, player)
        if isinstance(chosen, tuple) and len(chosen) == 3:
            chosen = (chosen, (0, 0, 0))
        for m in chosen:
            if m != (0, 0, 0):
                board.apply_move(m, switch_turn=False)
        board.switch_turn()
    return None, 0

def arena_worker(task):
    a_tag, a_path, b_tag, b_path, seed, a_white = task
    wag = _agent(a_path if a_white else b_path)
    bag = _agent(b_path if a_white else a_path)
    winner, score = _play(seed, wag, bag)
    if winner is None:
        a_margin = 0
    else:
        a_won = (winner == 'white') == a_white
        a_margin = score if a_won else -score
    return {'a': a_tag, 'b': b_tag, 'seed': seed, 'a_white': a_white,
            'winner': winner, 'a_margin': a_margin}

# -------------------- scheduling & data --------------------
def load_rows():
    if not os.path.exists(RESULTS):
        return []
    with open(RESULTS) as f:
        return [json.loads(l) for l in f if l.strip()]

def canonical_pair(a, b):
    return (a, b) if a < b else (b, a)

def get_next_batch(rows, done_set, batch_size):
    """
    Focuses ALL batch capacity on the champion(s) with the fewest played games
    until they reach parity with the rest of the field.
    """
    game_counts = defaultdict(int)
    completed_pair_seeds = defaultdict(set)
    
    for a, b, seed, aw in done_set:
        cp = canonical_pair(a, b)
        completed_pair_seeds[cp].add((seed, aw))

    for r in rows:
        game_counts[r['a']] += 1
        game_counts[r['b']] += 1

    for c in CHAMPIONS:
        game_counts[c] += 0

    # Calculate completed CRN pairs
    pair_completed_matches = defaultdict(int)
    for cp, seed_aws in completed_pair_seeds.items():
        seeds = {s for s, _ in seed_aws}
        full_matches = sum(1 for s in seeds if (s, True) in seed_aws and (s, False) in seed_aws)
        pair_completed_matches[cp] = full_matches

    tasks = []
    pending_done_set = set(done_set)

    # Keep filling the batch focusing on the bottleneck champion(s)
    while len(tasks) < batch_size:
        # 1. Identify the champion with the absolute fewest games right now
        target = min(CHAMPIONS.keys(), key=lambda c: game_counts[c])
        
        # 2. Sort opponents by fewest games played against 'target' specifically,
        # then by their overall game count (spreads target's games evenly across the roster)
        opponents = sorted(
            [c for c in CHAMPIONS if c != target],
            key=lambda opp: (pair_completed_matches[canonical_pair(target, opp)], game_counts[opp])
        )
        
        task_added = False
        for opp in opponents:
            cp = canonical_pair(target, opp)
            match_idx = pair_completed_matches[cp]
            seed = SEED_BASE + match_idx
            
            needed_aws = [aw for aw in (True, False) if (target, opp, seed, aw) not in pending_done_set]
            
            if needed_aws:
                for aw in needed_aws:
                    tasks.append((target, CHAMPIONS[target], opp, CHAMPIONS[opp], seed, aw))
                    pending_done_set.add((target, opp, seed, aw))
                
                game_counts[target] += len(needed_aws)
                game_counts[opp] += len(needed_aws)
                pair_completed_matches[cp] += 1
                task_added = True
                break # Re-evaluate who the lowest champ is for the next task insertion
        
        # Guard clause in case all possible matchups for all active champs are somehow filled
        if not task_added:
            break

    return tasks[:batch_size]

def run():
    rows = load_rows()
    done = {(r['a'], r['b'], r['seed'], r['a_white']) for r in rows}
    
    print(f"{len(CHAMPIONS)} champions active. Resuming: {len(rows)} total games played. "
          f"Prioritizing lowest-game contenders. Ctrl-C to stop.", flush=True)
    if rows:
        print(standings_str(rows), flush=True)

    ctx = mp.get_context('spawn')
    batch_num = 0
    try:
        with ctx.Pool(N_WORKERS, initializer=_worker_init) as pool, \
                open(RESULTS, 'a') as f:
            while True:
                tasks = get_next_batch(rows, done, BATCH_SIZE)
                if not tasks:
                    time.sleep(1)
                    continue

                t0 = time.time()
                for r in pool.imap_unordered(arena_worker, tasks, chunksize=1):
                    f.write(json.dumps(r) + '\n')
                    f.flush()
                    rows.append(r)
                    done.add((r['a'], r['b'], r['seed'], r['a_white']))
                
                batch_num += 1
                print(f"\n=== after batch {batch_num} ({len(rows)} games total, {time.time()-t0:.0f}s) ===", flush=True)
                print(standings_str(rows), flush=True)
                
                with open(STANDINGS, 'w') as sf:
                    sf.write(f"after batch {batch_num} ({len(rows)} games)\n")
                    sf.write(standings_str(rows) + '\n')
                    
    except KeyboardInterrupt:
        print(f"\n[aborted at batch {batch_num}] {len(rows)} games saved; rerun to resume.\n"
              "Final standings:", flush=True)
        print(standings_str(rows), flush=True)

# -------------------- ratings & display --------------------
def _elo(scored, epochs=400, K=8, tags=None):
    r = {t: 1500.0 for t in (tags if tags is not None else CHAMPIONS)}
    rng = random.Random(0)
    for _ in range(epochs):
        rng.shuffle(scored)
        for a, b, s in scored:
            ea = 1.0 / (1.0 + 10 ** ((r[b] - r[a]) / 400.0))
            r[a] += K * (s - ea)
            r[b] += K * ((1 - s) - (1 - ea))
    return r

def _roster(rows):
    tags = list(CHAMPIONS)
    for r in rows:
        for t in (r['a'], r['b']):
            if t not in tags:
                tags.append(t)
    return tags

def compute_ratings(rows):
    tags = _roster(rows)
    idx = {t: i for i, t in enumerate(tags)}
    n = len(tags)
    wins = [[0] * n for _ in range(n)]
    games = [[0] * n for _ in range(n)]
    marg = [[0] * n for _ in range(n)]
    win_scored, marg_scored = [], []
    MAXM = 12.0
    for r in rows:
        a, b = r['a'], r['b']
        if a not in idx or b not in idx:
            continue
        m = r['a_margin']; ia, ib = idx[a], idx[b]
        games[ia][ib] += 1; games[ib][ia] += 1
        marg[ia][ib] += m; marg[ib][ia] += -m
        if r['winner'] is not None:
            aw = 1 if m > 0 else 0
            wins[ia][ib] += aw; wins[ib][ia] += (1 - aw)
            win_scored.append((a, b, float(aw)))
        marg_scored.append((a, b, min(1.0, max(0.0, 0.5 + m / (2 * MAXM)))))
    return (_elo(list(win_scored), tags=tags), _elo(list(marg_scored), tags=tags),
            wins, games, marg, idx)

def standings_str(rows):
    win_elo, marg_elo, wins, games, marg, idx = compute_ratings(rows)
    order = sorted(idx, key=lambda t: marg_elo[t], reverse=True)
    out = [f"{'champ':<18}{'margin-Elo':>12}{'win-Elo':>10}{'W-L-D':>10}{'avg-marg':>11}   (by margin-Elo)"]
    n = len(idx)
    for t in order:
        i = idx[t]
        g = sum(games[i])
        w = sum(wins[i])
        losses = sum(wins[j][i] for j in range(n))
        draws = g - w - losses
        am = sum(marg[i]) / g if g else 0.0
        display_name = t if t in CHAMPIONS else f"({t})"
        out.append(f"{display_name:<18}{marg_elo[t]:>12.0f}{win_elo[t]:>10.0f}"
                   f"{f'{w}-{losses}-{draws}':>10}{am:>+11.2f}")
    return "\n".join(out)

def analyze():
    rows = load_rows()
    win_elo, marg_elo, wins, games, marg, idx = compute_ratings(rows)
    order = sorted(idx, key=lambda t: marg_elo[t], reverse=True)
    gone = [t for t in idx if t not in CHAMPIONS]
    print(f"{len(rows)} games among {len(idx)} champions"
          + (f" (from results, not on current roster: {', '.join(gone)})" if gone else "")
          + "\n")
    print(standings_str(rows))

    disp = lambda t: t if t in CHAMPIONS else f"({t})"

    print("\nWIN% matrix (row beats col):")
    print(' ' * 12 + ''.join(f'{disp(t):>12}' for t in order))
    for a in order:
        cells = []
        for b in order:
            g = games[idx[a]][idx[b]]
            cells.append('   -   ' if a == b or g == 0
                         else f'{100*wins[idx[a]][idx[b]]/g:6.0f} ')
        print(f'{disp(a):<12}' + ''.join(f'{c:>12}' for c in cells))

    print("\nAVG MARGIN matrix (row's mean signed margin vs col):")
    print(' ' * 12 + ''.join(f'{disp(t):>12}' for t in order))
    for a in order:
        cells = []
        for b in order:
            g = games[idx[a]][idx[b]]
            cells.append('   -   ' if a == b or g == 0
                         else f'{marg[idx[a]][idx[b]]/g:+6.2f} ')
        print(f'{disp(a):<12}' + ''.join(f'{c:>12}' for c in cells))

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'analyze':
        analyze()
    else:
        run()