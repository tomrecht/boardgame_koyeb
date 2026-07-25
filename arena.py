"""Champion arena: round-robin tournament with margin-based Elo.

Every champion plays every other over paired, color-swapped seeds (CRN). Each
game yields a winner and a MARGIN (= loser's unsaved pieces, 1-12; the same
quantity the value net predicts as raw*12). We then rate the field two ways:

  win-Elo    : standard logistic Elo on win/draw/loss (robust ranking)
  margin-Elo : same iterative Elo but each game's score is the margin mapped to
               [0,1] (shutout win -> 1.0, win-by-1 -> ~0.54, draw -> 0.5), so
               decisive wins move the rating more. Margin is lower-variance than
               win/loss (measured), so this discriminates near-parity champs
               with fewer games.

Also prints the raw win% and avg-margin matrices (who beats whom, by how much).

Play is plain 2-ply shallow (the deployed policy). Parallel over a CPU pool.

Runs ROUNDS indefinitely (no count fixed in advance): each round is one paired,
color-swapped game per champion pair. After every round it prints and saves the
current standings (margin-Elo, win-Elo, W-L-D, avg-margin). Ctrl-C anytime;
state lives in arena.jsonl so rerunning resumes seamlessly (finished rounds are
skipped instantly). Add a champion (e.g. once aux promotes) and rerun -- only
the new matchups get played.

Usage:
  python -u arena.py                 # play rounds forever; Ctrl-C to stop
  python arena.py analyze            # full ratings + matrices from arena.jsonl
Env: N_WORKERS (default 8), CHAMPS_EXTRA="tag=path,..." to add champions,
     CHAMPS="tag,tag" to restrict the roster to a subset.
"""
import os, sys, json, time, random, itertools
import multiprocessing as mp

REPO = os.path.dirname(os.path.abspath(__file__))
CHAMPIONS = {
    'iter5':  f'{REPO}/best_iter5_m46.pt',
    'iter1':  f'{REPO}/td_champion_July17_iter1.pt',
    'iter4':  f'{REPO}/td_champion_July17_iter4.pt',
    'iter10': f'{REPO}/td_champion_July18_iter10.pt',
    'iter14': f'{REPO}/td_champion_July19_iter14.pt',
}
for kv in filter(None, os.environ.get('CHAMPS_EXTRA', '').split(',')):
    tag, path = kv.split('=', 1)
    CHAMPIONS[tag.strip()] = path.strip()
# CHAMPS="iter10,iter14" restricts the roster to a subset (comma-separated tags).
_only = [t.strip() for t in os.environ.get('CHAMPS', '').split(',') if t.strip()]
if _only:
    CHAMPIONS = {t: CHAMPIONS[t] for t in _only}

N_SEEDS = int(os.environ.get('N_SEEDS', '40'))
N_WORKERS = int(os.environ.get('N_WORKERS', '8'))
RESULTS = os.environ.get('RESULTS', f'{REPO}/arena.jsonl')
STANDINGS = os.environ.get('STANDINGS', f'{REPO}/arena_standings.txt')
SEED_BASE = 4_400_000
MAX_TURNS, STUCK_LIMIT = 200, 60

# -------------------- worker (CPU-only, per-process model cache) --------------
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


# -------------------- run (unbounded round loop) --------------------
def load_rows():
    if not os.path.exists(RESULTS):
        return []
    with open(RESULTS) as f:
        return [json.loads(l) for l in f if l.strip()]


def run():
    """Play round-robin ROUNDS forever until Ctrl-C. Each round = one paired,
    color-swapped game per champion pair (seed = SEED_BASE + round). After each
    round, recompute + print + save current standings. State lives entirely in
    arena.jsonl, so aborting and rerunning resumes seamlessly (finished rounds
    are skipped instantly). No round count is fixed in advance."""
    pairs = list(itertools.combinations(list(CHAMPIONS), 2))
    rows = load_rows()
    done = {(r['a'], r['b'], r['seed'], r['a_white']) for r in rows}
    max_round = max((r['seed'] - SEED_BASE for r in rows), default=-1)
    complete_rounds = sum(
        1 for k in range(max_round + 1)
        if pairs and all((a, b, SEED_BASE + k, aw) in done
                         for a, b in pairs for aw in (True, False)))
    print(f"{len(CHAMPIONS)} champions, {len(pairs)} pairs "
          f"({2*len(pairs)} games/round). Resuming: {len(rows)} games, "
          f"{complete_rounds} full rounds done. Ctrl-C to stop.", flush=True)
    if rows:
        print(standings_str(rows), flush=True)

    ctx = mp.get_context('spawn')
    round_num = 0
    try:
        with ctx.Pool(N_WORKERS, initializer=_worker_init) as pool, \
                open(RESULTS, 'a') as f:
            while True:                       # unbounded; Ctrl-C to abort
                seed = SEED_BASE + round_num
                tasks = [(a, CHAMPIONS[a], b, CHAMPIONS[b], seed, aw)
                         for a, b in pairs for aw in (True, False)
                         if (a, b, seed, aw) not in done]
                if tasks:
                    t0 = time.time()
                    for r in pool.imap_unordered(arena_worker, tasks, chunksize=1):
                        f.write(json.dumps(r) + '\n'); f.flush()
                        rows.append(r)
                        done.add((r['a'], r['b'], r['seed'], r['a_white']))
                    print(f"\n=== after round {round_num} "
                          f"({len(rows)} games total, {time.time()-t0:.0f}s) ===",
                          flush=True)
                    print(standings_str(rows), flush=True)
                    with open(STANDINGS, 'w') as sf:
                        sf.write(f"after round {round_num} ({len(rows)} games)\n")
                        sf.write(standings_str(rows) + '\n')
                round_num += 1
    except KeyboardInterrupt:
        print(f"\n[aborted at round {round_num}] {len(rows)} games saved to "
              f"{os.path.basename(RESULTS)}; rerun to resume.\n"
              "Final standings (also in " + os.path.basename(STANDINGS) + "):",
              flush=True)
        print(standings_str(rows), flush=True)


# -------------------- ratings --------------------
def _elo(scored, epochs=400, K=8):
    """scored: list of (a, b, a_score in [0,1]). Iterative symmetric Elo,
    mean-anchored at 1500. Returns {tag: rating}."""
    r = {t: 1500.0 for t in CHAMPIONS}
    rng = random.Random(0)
    for _ in range(epochs):
        rng.shuffle(scored)
        for a, b, s in scored:
            ea = 1.0 / (1.0 + 10 ** ((r[b] - r[a]) / 400.0))
            r[a] += K * (s - ea)
            r[b] += K * ((1 - s) - (1 - ea))
    return r


def compute_ratings(rows):
    """Aggregate rows -> (win_elo, marg_elo, wins, games, marg, idx). Only
    champions currently in CHAMPIONS are rated; unknown tags in the jsonl are
    ignored (safe when champions are added between runs)."""
    tags = list(CHAMPIONS)
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
    return (_elo(list(win_scored)), _elo(list(marg_scored)),
            wins, games, marg, idx)


def standings_str(rows):
    win_elo, marg_elo, wins, games, marg, idx = compute_ratings(rows)
    order = sorted(CHAMPIONS, key=lambda t: marg_elo[t], reverse=True)
    out = [f"{'champ':8}{'margin-Elo':>11}{'win-Elo':>9}{'  W-L-D':>9}"
           f"{'  avg-marg':>10}   (by margin-Elo)"]
    n = len(CHAMPIONS)
    for t in order:
        i = idx[t]
        g = sum(games[i])
        w = sum(wins[i])                          # t's wins
        losses = sum(wins[j][i] for j in range(n))  # opponents' wins over t
        draws = g - w - losses
        am = sum(marg[i]) / g if g else 0.0
        out.append(f"{t:8}{marg_elo[t]:11.0f}{win_elo[t]:9.0f}"
                   f"{f'{w}-{losses}-{draws}':>9}{am:+10.2f}")
    return "\n".join(out)


def analyze():
    rows = load_rows()
    win_elo, marg_elo, wins, games, marg, idx = compute_ratings(rows)
    order = sorted(CHAMPIONS, key=lambda t: marg_elo[t], reverse=True)
    print(f"{len(rows)} games among {len(CHAMPIONS)} champions\n")
    print(standings_str(rows))

    print("\nWIN% matrix (row beats col):")
    print('       ' + ''.join(f'{t:>8}' for t in order))
    for a in order:
        cells = []
        for b in order:
            g = games[idx[a]][idx[b]]
            cells.append('   -   ' if a == b or g == 0
                         else f'{100*wins[idx[a]][idx[b]]/g:6.0f} ')
        print(f'{a:7}' + ''.join(f'{c:>8}' for c in cells))

    print("\nAVG MARGIN matrix (row's mean signed margin vs col):")
    print('       ' + ''.join(f'{t:>8}' for t in order))
    for a in order:
        cells = []
        for b in order:
            g = games[idx[a]][idx[b]]
            cells.append('   -   ' if a == b or g == 0
                         else f'{marg[idx[a]][idx[b]]/g:+6.2f} ')
        print(f'{a:7}' + ''.join(f'{c:>8}' for c in cells))
    print("\nNote: margin-Elo rewards decisive wins; win-Elo is the plain "
          "who-beats-whom. Divergence between them flags a champ that wins "
          "often-but-narrowly (or rarely-but-big).")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'analyze':
        analyze()
    else:
        run()
