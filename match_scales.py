"""match_scales.py -- do the fitted prefilter scales change playing strength?

Both sides are the shipped agent (same net, prefilter F=12 / K=40 / min 5);
the only difference is how the prefilter ranks candidates:

    NEW  component scales from prefilter_scales.json (keep-rate ~96%)
    OLD  the plain heuristic sum (keep-rate ~88%)

Paired seeds with colours swapped (common random numbers). Every game is
appended to match_scales.jsonl as it finishes; re-running resumes.

    python3 match_scales.py [pairs=400]
    python3 match_scales.py analyze
"""
import copy, json, math, os, random, statistics, sys
import multiprocessing as mp

REPO = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get('MODEL', f'{REPO}/model.onnx')
N_WORKERS = int(os.environ.get('N_WORKERS', '4'))
OUT = os.path.join(REPO, 'match_scales.jsonl')
SEED_BASE = 8_800_000
MAX_TURNS, STUCK_LIMIT = 200, 60

_AG = {}


def _agents():
    if not _AG:
        from agent_gnn import GNNAgent
        new = GNNAgent(weights_path=MODEL, use_prefilter=True, prefilter_top_k=40,
                       prefilter_min_k=5, first_move_prefilter=12)
        assert 'component_scale' in new.heuristic.weights, 'scales not loaded'
        old = GNNAgent(weights_path=MODEL, use_prefilter=True, prefilter_top_k=40,
                       prefilter_min_k=5, first_move_prefilter=12)
        old.backend, old.model, old.encoder = new.backend, new.backend, new.encoder
        old.heuristic.weights = copy.deepcopy(old.heuristic.weights)
        old.heuristic.weights.pop('component_scale', None)
        _AG['new'], _AG['old'] = new, old
    return _AG['new'], _AG['old']


def _play(seed, white, black):
    from game import Board
    random.seed(seed)
    board = Board()
    agents = {'white': white, 'black': black}
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
        chosen = agents[player].select_move_pair(list(board.get_valid_moves()), board, player,
                                                 difficulty=1.0)
        if isinstance(chosen, tuple) and len(chosen) == 3:
            chosen = (chosen, (0, 0, 0))
        for m in chosen:
            if m != (0, 0, 0):
                board.apply_move(m, switch_turn=False)
        board.switch_turn()
    return None, 0


def _worker(task):
    seed, new_white = task
    new, old = _agents()
    winner, score = _play(seed, new if new_white else old, old if new_white else new)
    m = 0 if winner is None else (score if (winner == 'white') == new_white else -score)
    return {'seed': seed, 'new_white': new_white, 'winner': winner, 'new_margin': m}


def analyze():
    rows = [json.loads(l) for l in open(OUT)]
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r['seed'], []).append(r['new_margin'])
    paired = [statistics.fmean(v) for v in by_seed.values() if len(v) == 2]
    dec = [r for r in rows if r['winner'] is not None]
    wins = sum(r['new_margin'] > 0 for r in dec)
    m = statistics.fmean(paired)
    se = statistics.stdev(paired) / math.sqrt(len(paired))
    print(f'{len(rows)} games, {len(paired)} complete pairs: new wins {wins}/{len(dec)} decisive '
          f'({100 * wins / max(1, len(dec)):.1f}%)')
    print(f'mean paired margin for NEW scales: {m:+.3f} (95% CI {m - 1.96 * se:+.3f} .. {m + 1.96 * se:+.3f})')


def main():
    if sys.argv[1:2] == ['analyze']:
        return analyze()
    pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    done = set()
    if os.path.exists(OUT):
        done = {(r['seed'], r['new_white']) for r in map(json.loads, open(OUT))}
    tasks = [(SEED_BASE + k, w) for k in range(pairs) for w in (True, False)
             if (SEED_BASE + k, w) not in done]
    print(f'{len(tasks)} games to play ({pairs} pairs), {N_WORKERS} workers', flush=True)
    ctx = mp.get_context('spawn')
    with ctx.Pool(N_WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, tasks, chunksize=1), 1):
            with open(OUT, 'a') as f:
                f.write(json.dumps(r) + '\n')
            if i % 20 == 0:
                print(f'  {i}/{len(tasks)} games', flush=True)
    analyze()


if __name__ == '__main__':
    main()
