"""Self-play data for match awareness: the shipped net plays both sides (app
config: prefilter 12/40/5, hand rules, endgame lookahead) and every turn records
the net's value of the position it chose (afterstate, mover's view, in points)
with both sides' unsaved counts; the game's final margin closes it.
Run with the TRAINING branch's code on the path (layout-aware game.py):
    cd <train-features-v2 checkout> && GOAL_LAYOUT=sum7 N_WORKERS=4 \\
      python3 <this file> <net.onnx> out.jsonl <games> [seed_base]
One line per game: {seed, starter, winner, margin, turns: [[mover, v, my_unsaved, opp_unsaved], ...]}.
Resumes by seed."""
import json, os, random, sys
import multiprocessing as mp
sys.path.insert(0, os.getcwd())          # the training checkout (spawned workers too)
_AG = None
NET = None
MAX_TURNS, STUCK = 200, 60

def _agent():
    global _AG
    if _AG is None:
        from agent_gnn import GNNAgent
        _AG = GNNAgent(weights_path=NET, use_prefilter=True, prefilter_top_k=40,
                       prefilter_min_k=5, first_move_prefilter=12)
        _AG.endgame_lookahead = True
    return _AG

def _unsaved(b, p):
    return 12 - len(b.white_saved if p == 'white' else b.black_saved)

def play(seed):
    import numpy as np
    from game import Board
    from agent_gnn import NUM_PIECES
    random.seed(seed); np.random.seed(seed % (2 ** 31 - 1))
    ag = _agent()
    b = Board()
    starter = b.current_player
    rec, last, since = [], 0, 0
    for _ in range(MAX_TURNS):
        w, s = b.check_game_over()
        if w:
            return {'seed': seed, 'starter': starter, 'winner': w, 'margin': int(s), 'turns': rec}
        if b.draw_callable:
            break
        cur = len(b.white_saved) + len(b.black_saved)
        if cur > last: last, since = cur, 0
        elif last: since += 1
        if last and since >= STUCK:
            break
        p = b.current_player
        pair = ag.select_move_pair(list(b.get_valid_moves()), b, p)
        if isinstance(pair, tuple) and len(pair) == 3:
            pair = (pair, (0, 0, 0))
        for m in pair:
            if m != (0, 0, 0):
                b.apply_move(m, switch_turn=False)
        w, s = b.check_game_over()
        if w:
            v = float((1 if w == p else -1) * s)
        else:
            v = float(ag.model(ag._encode_after(b, p))) * NUM_PIECES
        rec.append([p[0], round(v, 3), _unsaved(b, p), _unsaved(b, 'black' if p == 'white' else 'white')])
        b.switch_turn()
    return {'seed': seed, 'starter': starter, 'winner': None, 'margin': 0, 'turns': rec}

def _init(net):
    global NET
    NET = net

def main(net, out, n, base=500_000_000):
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)['seed'] for l in open(out)}
    seeds = [base + k for k in range(n) if base + k not in done]
    ctx = mp.get_context('spawn')
    with ctx.Pool(int(os.environ.get('N_WORKERS', '4')), initializer=_init, initargs=(net,)) as pool, open(out, 'a') as f:
        for i, r in enumerate(pool.imap_unordered(play, seeds), 1):
            f.write(json.dumps(r) + '\n'); f.flush()
            if i % 25 == 0:
                print(i, 'games', flush=True)

if __name__ == '__main__':
    main(os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), int(sys.argv[3]),
         int(sys.argv[4]) if len(sys.argv) > 4 else 500_000_000)
