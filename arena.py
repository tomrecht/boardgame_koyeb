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
Resumable: appends to arena.jsonl, skips recorded games.

Usage:
  python -u arena.py                 # run the round-robin (writes arena.jsonl)
  python arena.py analyze            # ratings + matrices from arena.jsonl
Env: N_SEEDS (pairs/​champ-pair, x2 colors; default 40), N_WORKERS (default 8),
     CHAMPS_EXTRA="tag=path,tag=path" to add champions (e.g. aux champions).
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

N_SEEDS = int(os.environ.get('N_SEEDS', '40'))
N_WORKERS = int(os.environ.get('N_WORKERS', '8'))
RESULTS = os.environ.get('RESULTS', f'{REPO}/arena.jsonl')
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


# -------------------- run --------------------
def run():
    tags = list(CHAMPIONS)
    done = set()
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            for line in f:
                r = json.loads(line)
                done.add((r['a'], r['b'], r['seed'], r['a_white']))
    tasks = []
    for a, b in itertools.combinations(tags, 2):
        for k in range(N_SEEDS):
            seed = SEED_BASE + k
            for a_white in (True, False):
                key = (a, b, seed, a_white)
                if key not in done:
                    tasks.append((a, CHAMPIONS[a], b, CHAMPIONS[b], seed, a_white))
    print(f"{len(tags)} champions, {len(list(itertools.combinations(tags,2)))} pairs, "
          f"{len(tasks)} games to play ({len(done)} already done)", flush=True)
    if not tasks:
        print("nothing to do."); return
    t0 = time.time()
    ctx = mp.get_context('spawn')
    n_done = 0
    with ctx.Pool(N_WORKERS, initializer=_worker_init) as pool, open(RESULTS, 'a') as f:
        for r in pool.imap_unordered(arena_worker, tasks, chunksize=1):
            f.write(json.dumps(r) + '\n'); f.flush()
            n_done += 1
            if n_done % max(1, len(tasks) // 20) == 0:
                el = time.time() - t0
                print(f"  {n_done}/{len(tasks)} games ({el:.0f}s, "
                      f"{el/n_done:.1f}s/game)", flush=True)
    print(f"RUN COMPLETE: {n_done} games in {time.time()-t0:.0f}s", flush=True)


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


def analyze():
    rows = [json.loads(l) for l in open(RESULTS)]
    tags = list(CHAMPIONS)
    idx = {t: i for i, t in enumerate(tags)}
    n = len(tags)
    wins = [[0] * n for _ in range(n)]      # wins[a][b] = a's wins vs b
    games = [[0] * n for _ in range(n)]
    marg = [[0] * n for _ in range(n)]      # sum of a's signed margin vs b
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
        # margin score in [0,1] (draws -> 0.5); include draws for margin-Elo
        s = min(1.0, max(0.0, 0.5 + m / (2 * MAXM)))
        marg_scored.append((a, b, s))
    win_elo = _elo(list(win_scored))
    marg_elo = _elo(list(marg_scored))

    order = sorted(tags, key=lambda t: marg_elo[t], reverse=True)
    print(f"{len(rows)} games among {n} champions\n")
    print(f"{'champ':8}{'margin-Elo':>12}{'win-Elo':>10}   (sorted by margin-Elo)")
    for t in order:
        print(f"{t:8}{marg_elo[t]:12.0f}{win_elo[t]:10.0f}")

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
