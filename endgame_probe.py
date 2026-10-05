"""Does the net misjudge the endgame? Shallow 2-ply vs one-opponent-ply deep
search, by how close the opponent is to finishing.

Games are played forward by the deployed champion at full strength (shallow,
hand rules on -- exactly what ships). At every turn where the OPPONENT has
<= LATE_MAX unsaved pieces and an empty rack, and at a sparse sample of
midgame turns as a baseline, the same position is also put to
select_move_pair_deep (expectiminimax over all 21 opponent rolls, terminal
wins scored at their exact margin). Recorded per position: whether the deep
pick reaches a different position, and the deep search's own estimate of what
the shallow pick gives up, in margin points.

Usage: python3 endgame_probe.py            (resumes; results in endgame_probe.jsonl)
       python3 endgame_probe.py analyze
"""
import json, os, random, sys, time
from multiprocessing import Pool

REPO = os.path.dirname(os.path.abspath(__file__))
NET = os.environ.get('NET', f'{REPO}/symaug_champ_July27_iter6.pt')
N_GAMES = int(os.environ.get('N_GAMES', '60'))
N_WORKERS = int(os.environ.get('N_WORKERS', '3'))
RESULTS = os.environ.get('RESULTS', f'{REPO}/endgame_probe.jsonl')
LATE_MAX = 6          # opponent unsaved pieces at or below this = "late"
MID_MIN = 8           # baseline: opponent has at least this many left
MID_EVERY = 15        # sample one midgame turn in this many
K_ME = 16             # deep candidates (endgame is cheap, so wider than default 8)
SEED_BASE = 9_100_000
MAX_TURNS = 200

_AG = None


def _init():
    global _AG
    os.environ['BOARDGAME_DEVICE'] = 'cpu'
    import torch
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    import network
    network.DEVICE = torch.device('cpu')
    from network import BoardGNN
    from agent_gnn import GNNAgent
    m = BoardGNN()
    m.load_state_dict(torch.load(NET, map_location='cpu'), strict=False)
    m.eval()
    _AG = GNNAgent(model=m)


def _left(board, colour):
    rack = board.white_unentered if colour == 'white' else board.black_unentered
    on = sum(1 for p in board.pieces if p.player == colour and p.tile is not None)
    return on, len(rack)


def _apply_key(board, pair):
    from agent_gnn import _piece_locs
    n = len(board.moves)
    for m in pair:
        if m != (0, 0, 0):
            board.apply_move(m, switch_turn=False)
    k = _piece_locs(board)
    while len(board.moves) > n:
        board.undo_last_move()
    return k


def _saves(pair, player):
    return sum(1 for m in pair if isinstance(m, tuple) and len(m) == 3
               and m[1] == 'save' and isinstance(m[0], tuple) and m[0][0] == player)


def play(gi):
    import numpy as np
    from game import Board
    seed = SEED_BASE + gi
    random.seed(seed)
    np.random.seed(seed)
    board = Board()
    rows = []
    t0 = time.time()
    for turn in range(MAX_TURNS):
        w, s = board.check_game_over()
        if w or board.draw_callable:
            break
        player = board.current_player
        opp = 'black' if player == 'white' else 'white'
        moves = list(board.get_valid_moves())
        shallow = _AG.select_move_pair(moves, board, player, difficulty=1.0)
        if isinstance(shallow, tuple) and len(shallow) == 3:
            shallow = (shallow, (0, 0, 0))
        opp_on, opp_rack = _left(board, opp)
        my_on, my_rack = _left(board, player)
        late = opp_rack == 0 and opp_on <= LATE_MAX
        mid = opp_on + opp_rack >= MID_MIN and turn % MID_EVERY == 0
        if (late or mid) and len(moves) > 1:
            st = dict(board.game_stages)
            ts = time.time()
            deep = _AG.select_move_pair_deep(moves, board, player, k_me=K_ME,
                                             return_scores=True)
            board.game_stages.update(st)
            if isinstance(deep, list) and deep:
                sk = _apply_key(board, shallow)
                dv = {}
                for v, p in deep:
                    dv.setdefault(_apply_key(board, p), (v, p))
                best_v, best_p = deep[0]
                sh_v = dv.get(sk, (None,))[0]
                rows.append({
                    'game': gi, 'turn': turn, 'kind': 'late' if late else 'mid',
                    'opp_left': opp_on + opp_rack, 'my_left': my_on + my_rack,
                    'n_cands': len(deep),
                    'differ': _apply_key(board, best_p) != sk,
                    'gap': None if sh_v is None else round((best_v - sh_v) * 12, 4),
                    'shallow_in_k': sh_v is not None,
                    'sh_saves': _saves(shallow, player), 'dp_saves': _saves(best_p, player),
                    'shallow': repr(shallow), 'deep': repr(best_p),
                    'deep_v': round(best_v * 12, 3), 'secs': round(time.time() - ts, 2)})
        for m in shallow:
            if m != (0, 0, 0):
                board.apply_move(m, switch_turn=False)
        board.switch_turn()
    return {'game': gi, 'rows': rows, 'secs': round(time.time() - t0, 1)}


def analyze():
    rows = []
    for line in open(RESULTS):
        g = json.loads(line)
        rows.extend(g['rows'])
    print(f'{len(set(r["game"] for r in rows))} games, {len(rows)} probed positions\n')
    print(f'{"bucket":>16} {"n":>5} {"differ":>8} {"gap>0.05":>9} {"mean gap":>9} '
          f'{"sum gap":>8} {"deep saves more":>16} {"deep saves fewer":>17}')

    def line(label, rs):
        if not rs:
            return
        d = [r for r in rs if r['differ']]
        g = [r['gap'] for r in rs if r['gap'] is not None]
        big = sum(1 for x in g if x > 0.05)
        more = sum(1 for r in d if r['dp_saves'] > r['sh_saves'])
        fewer = sum(1 for r in d if r['dp_saves'] < r['sh_saves'])
        print(f'{label:>16} {len(rs):>5} {len(d)/len(rs):>7.1%} {big/len(rs):>8.1%} '
              f'{(sum(g)/len(g) if g else 0):>9.3f} {sum(g):>8.1f} {more:>16} {fewer:>17}')

    line('mid (baseline)', [r for r in rows if r['kind'] == 'mid'])
    late = [r for r in rows if r['kind'] == 'late']
    line('late (all)', late)
    for k in range(LATE_MAX, 0, -1):
        line(f'opp left = {k}', [r for r in late if r['opp_left'] == k])
    miss = sum(1 for r in rows if not r['shallow_in_k'])
    print(f'\nshallow pick outside deep top-{K_ME}: {miss}')
    print('\nlargest late gaps:')
    for r in sorted([r for r in late if r['gap']], key=lambda r: -r['gap'])[:12]:
        print(f"  g{r['game']} t{r['turn']} opp{r['opp_left']} me{r['my_left']} "
              f"gap {r['gap']:.3f}  sh {r['shallow']}  dp {r['deep']}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'analyze':
        return analyze()
    done = set()
    if os.path.exists(RESULTS):
        done = {json.loads(l)['game'] for l in open(RESULTS)}
    todo = [g for g in range(N_GAMES) if g not in done]
    print(f'{len(todo)} games to play', flush=True)
    with Pool(N_WORKERS, initializer=_init) as pool:
        for res in pool.imap_unordered(play, todo):
            with open(RESULTS, 'a') as f:
                f.write(json.dumps(res) + '\n')
            print(f"game {res['game']}: {len(res['rows'])} probes, {res['secs']}s", flush=True)


if __name__ == '__main__':
    main()
