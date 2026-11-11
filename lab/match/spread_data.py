"""Training data for a margin-SPREAD output (match awareness, take 2).

The shipped net plays both sides (app config: prefilter 12/40/5, hand rules,
endgame lookahead). For each turn it records the net's 64-wide hidden layer
(readout[1], the layer every value output is read from) on the afterstate the
mover chose, the net's value there and the phase; the game's final margin closes
the record. On every CAND_EVERY-th turn it also records the same for the top
CAND_K candidates of the plain search, so we can ask whether a learned spread
tells near-equal moves apart. A torch checkpoint is needed (the hook).
    cd <train-features-v2 checkout> && GOAL_LAYOUT=sum7 N_WORKERS=4 \\
      python3 <this file> <ckpt.pt> out.jsonl.gz <games> [seed_base]
One line per game: {seed, starter, winner, margin,
  turns: [[mover, h, v, phase, cands|null], ...]}
h = base64 float16 x 64; v = net value in points, mover's view; phase = both
sides' unsaved after the move; cands = [[h, v], ...] (best first by the search).
A turn that ends the game is not recorded (its margin is exact)."""
import base64, gzip, json, os, random, sys
import multiprocessing as mp
sys.path.insert(0, os.getcwd())
_AG = None
NET = None
MAX_TURNS, STUCK = 200, 60
CAND_EVERY = int(os.environ.get('CAND_EVERY', '4'))
CAND_K = int(os.environ.get('CAND_K', '6'))
_H = {}


def _agent():
    global _AG
    if _AG is None:
        from agent_gnn import GNNAgent
        _AG = GNNAgent(weights_path=NET, use_prefilter=True, prefilter_top_k=40,
                       prefilter_min_k=5, first_move_prefilter=12)
        _AG.endgame_lookahead = True
        m = _AG.backend.model
        m.readout[1].register_forward_hook(lambda mod, i, o: _H.__setitem__('h', o.detach().cpu().numpy()))
    return _AG


def _probe(ag, b, p):
    """(h as base64 float16, value in points) of the current board, mover p."""
    import numpy as np
    from agent_gnn import NUM_PIECES
    v = float(np.asarray(ag.model([ag._encode_after(b, p)])).reshape(-1)[0]) * NUM_PIECES
    h = _H['h'].reshape(-1).astype(np.float16)
    return base64.b64encode(h.tobytes()).decode(), round(v, 3)


def _unsaved(b, p):
    return 12 - len(b.white_saved if p == 'white' else b.black_saved)


def _real(pair):
    return [m for m in pair if m != (0, 0, 0)]


def play(seed):
    import numpy as np
    from game import Board
    random.seed(seed); np.random.seed(seed % (2 ** 31 - 1))
    ag = _agent()
    b = Board()
    starter = b.current_player
    rec, last, since = [], 0, 0
    for t in range(MAX_TURNS):
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
        moves = list(b.get_valid_moves())
        cands = None
        if CAND_EVERY and t % CAND_EVERY == 0:
            ranked = ag.select_move_pair(moves, b, p, return_scores=True)
            cands = []
            for sc, pr in ranked:
                if len(cands) >= CAND_K: break
                if sc == float('inf') or not isinstance(pr, tuple): continue
                if isinstance(pr[0], int): pr = (pr, (0, 0, 0))
                if (1, 1, 1) in pr: continue
                done = 0
                try:
                    for m in _real(pr):
                        b.apply_move(m, switch_turn=False); done += 1
                    if not b.check_game_over()[0]:
                        cands.append(list(_probe(ag, b, p)))
                except Exception:
                    pass
                for _ in range(done):
                    b.undo_last_move()
            moves = list(b.get_valid_moves())
        pair = ag.select_move_pair(moves, b, p)
        if isinstance(pair, tuple) and len(pair) == 3 and isinstance(pair[0], int):
            pair = (pair, (0, 0, 0))
        if (1, 1, 1) in pair:
            break
        for m in _real(pair):
            b.apply_move(m, switch_turn=False)
        if not b.check_game_over()[0]:
            h, v = _probe(ag, b, p)
            opp = 'black' if p == 'white' else 'white'
            rec.append([p[0], h, v, _unsaved(b, p) + _unsaved(b, opp), cands])
        b.switch_turn()
    return {'seed': seed, 'starter': starter, 'winner': None, 'margin': 0, 'turns': rec}


def _init(net):
    global NET
    NET = net


def main(net, out, n, base=700_000_000):
    ctx = mp.get_context('spawn')
    seeds = [base + k for k in range(n)]
    with ctx.Pool(int(os.environ.get('N_WORKERS', '4')), initializer=_init, initargs=(net,)) as pool, \
            gzip.open(out, 'wt') as f:
        for i, r in enumerate(pool.imap_unordered(play, seeds), 1):
            f.write(json.dumps(r) + '\n'); f.flush()
            if i % 10 == 0:
                print(i, 'games', flush=True)


if __name__ == '__main__':
    main(os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), int(sys.argv[3]),
         int(sys.argv[4]) if len(sys.argv) > 4 else 700_000_000)
