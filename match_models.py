"""match_models.py -- two nets head to head in the APP's configuration (ONNX
net, prefilter F=12 / K=40 / min 5, hand rules on), paired colour-swapped seeds.
Appends to match_<tag>.jsonl; resumes.

    MODEL_A=blend50.onnx MODEL_B=model.onnx TAG=blend50 python3 match_models.py [pairs=500]
    TAG=blend50 python3 match_models.py analyze
"""
import json, math, os, random, statistics, sys
import multiprocessing as mp

REPO = os.path.dirname(os.path.abspath(__file__))
MODEL_A = os.environ.get('MODEL_A', f'{REPO}/blend50.onnx')
MODEL_B = os.environ.get('MODEL_B', f'{REPO}/model.onnx')
TAG = os.environ.get('TAG', 'models')
OUT = os.path.join(REPO, f'match_{TAG}.jsonl')
N_WORKERS = int(os.environ.get('N_WORKERS', '4'))
SEED_BASE = int(os.environ.get('SEED_BASE', '6600000'))
MAX_TURNS, STUCK_LIMIT = 200, 60
_AG = {}


def _agents():
    if not _AG:
        from agent_gnn import GNNAgent
        for k, path in (('A', MODEL_A), ('B', MODEL_B)):
            _AG[k] = GNNAgent(weights_path=path, use_prefilter=True, prefilter_top_k=40,
                              prefilter_min_k=5, first_move_prefilter=12)
    return _AG['A'], _AG['B']


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
        p = board.current_player
        chosen = agents[p].select_move_pair(list(board.get_valid_moves()), board, p, difficulty=1.0)
        if isinstance(chosen, tuple) and len(chosen) == 3:
            chosen = (chosen, (0, 0, 0))
        for m in chosen:
            if m != (0, 0, 0):
                board.apply_move(m, switch_turn=False)
        board.switch_turn()
    return None, 0


def _worker(task):
    seed, a_white = task
    A, B = _agents()
    winner, score = _play(seed, A if a_white else B, B if a_white else A)
    m = 0 if winner is None else (score if (winner == 'white') == a_white else -score)
    return {'seed': seed, 'a_white': a_white, 'winner': winner, 'a_margin': m}


def analyze():
    rows = [json.loads(l) for l in open(OUT)]
    by = {}
    for r in rows:
        by.setdefault(r['seed'], []).append(r['a_margin'])
    paired = [statistics.fmean(v) for v in by.values() if len(v) == 2]
    dec = [r for r in rows if r['winner'] is not None]
    wins = sum(r['a_margin'] > 0 for r in dec)
    m = statistics.fmean(paired)
    se = statistics.stdev(paired) / math.sqrt(len(paired))
    print(f'{TAG}: {len(rows)} games, {len(paired)} pairs: A ({os.path.basename(MODEL_A)}) wins '
          f'{wins}/{len(dec)} ({100 * wins / max(1, len(dec)):.1f}%), mean paired margin {m:+.3f} '
          f'(95% CI {m - 1.96 * se:+.3f} .. {m + 1.96 * se:+.3f})')


def main():
    if sys.argv[1:2] == ['analyze']:
        return analyze()
    pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    done = set()
    if os.path.exists(OUT):
        done = {(r['seed'], r['a_white']) for r in map(json.loads, open(OUT))}
    tasks = [(SEED_BASE + k, w) for k in range(pairs) for w in (True, False) if (SEED_BASE + k, w) not in done]
    print(f'{len(tasks)} games, A={MODEL_A} vs B={MODEL_B}', flush=True)
    with mp.get_context('spawn').Pool(N_WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, tasks, chunksize=1), 1):
            with open(OUT, 'a') as f:
                f.write(json.dumps(r) + '\n')
            if i % 50 == 0:
                print(f'  {i}/{len(tasks)}', flush=True)
    analyze()


if __name__ == '__main__':
    main()
