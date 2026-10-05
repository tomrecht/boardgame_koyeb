"""Calibration benchmark: does a net's judgement of two moves track reality?

The 559 positions from owner's games (2026-10-04/05 overnight run) where his
move (A) and the champion's best (B) differ, each with the PLAYOUT difference
B - A from 20 paired playouts (net 1-ply both sides, common dice). For a net, its
own value difference V(after B) - V(after A) (margin points) is regressed on the
playouts: corr and slope. The deployed champion scores corr +0.09, slope 0.29 --
it is ~3x overconfident, i.e. its values between near-equal moves are mostly
noise. A better-trained net should raise both. Cheap: 2 evaluations a position.

    python3 calib_bench.py build      (needs rollout_disagree.jsonl + the game logs)
    python3 calib_bench.py <ckpt.pt> [...]
"""
import ast, json, math, os, sys

REPO = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(REPO, 'calib_bench.json')
LOGS = ['quahuru-games-apvo2h65-v2.jsonl', 'quahuru-games-f7in6olg.jsonl',
        'quahuru-games-apvo2h65-2026-10-05.jsonl']


def _serialize(board):
    return {
        'currentTurn': board.current_player,
        'dice': [{'value': d.number, 'used': d.used} for d in board.dice],
        'racks': {
            'whiteUnentered': [{'color': 'white', 'number': p.number} for p in board.white_unentered],
            'whiteSaved': [{'color': 'white', 'number': p.number} for p in board.white_saved],
            'blackUnentered': [{'color': 'black', 'number': p.number} for p in board.black_unentered],
            'blackSaved': [{'color': 'black', 'number': p.number} for p in board.black_saved],
        },
        'boardPieces': [{'color': p.player, 'number': p.number,
                         'tile': {'ring': p.tile.ring, 'sector': p.tile.pos}}
                        for p in board.pieces if p.tile is not None],
    }


def build():
    import weakness_probe as W
    recs = {}
    for f in LOGS:
        for line in open(os.path.join(REPO, f)):
            r = json.loads(line)
            recs[r['id'][:8]] = r
    rows = [r for r in map(json.loads, open(os.path.join(REPO, 'rollout_disagree.jsonl')))
            if 'error' not in r]
    by_game = {}
    for r in rows:
        by_game.setdefault(r['game'], {})[r['turn']] = r
    out = []
    for gid, turns in by_game.items():
        for t, b, played, b2 in W.turn_states(recs[gid]):
            if t not in turns:
                continue
            r = turns[t]
            item = {'game': gid, 'turn': t, 'mover': b.current_player}
            for arm, key in (('A', 'pair_A'), ('B', 'pair_B')):
                n0 = len(b.moves)
                for m in ast.literal_eval(r[key]):
                    if m not in ((0, 0, 0), (1, 1, 1)):
                        b.apply_move(m, switch_turn=False)
                item['state_' + arm] = _serialize(b)
                while len(b.moves) > n0:
                    b.undo_last_move()
            diffs = [y - x for x, y in zip(r['A'], r['B'])]
            n = len(diffs)
            m = sum(diffs) / n
            item['playout_diff'] = m
            item['playout_se'] = math.sqrt(sum((d - m) ** 2 for d in diffs) / (n - 1) / n)
            item['champion_gap'] = r.get('net_gap')
            out.append(item)
    json.dump(out, open(BENCH, 'w'))
    print(f'{len(out)} positions -> {BENCH}')


_ITEMS = None


def score_model(model, features=None):
    """(corr, slope, mean_gap, n) for a loaded BoardGNN on the benchmark."""
    global _ITEMS
    import torch
    import network
    from game import Board
    if _ITEMS is None:
        _ITEMS = json.load(open(BENCH))
    enc = network.BoardEncoder(features=features or getattr(model, 'features', 'v1'))
    board = Board()
    gaps, diffs = [], []
    model.eval()
    with torch.no_grad():
        for it in _ITEMS:
            v = {}
            for arm in ('A', 'B'):
                board.update_state(it['state_' + arm])
                v[arm] = float(model(enc.encode(board, it['mover']))) * 12
            gaps.append(v['B'] - v['A'])
            diffs.append(it['playout_diff'])
    n = len(gaps)
    mg, md = sum(gaps) / n, sum(diffs) / n
    cov = sum((x - mg) * (y - md) for x, y in zip(gaps, diffs))
    vg = sum((x - mg) ** 2 for x in gaps)
    vd = sum((y - md) ** 2 for y in diffs)
    return cov / math.sqrt(vg * vd), cov / vg, mg, n


def main():
    if sys.argv[1:2] == ['build']:
        return build()
    import torch
    import network
    network.DEVICE = torch.device('cpu')
    for path in sys.argv[1:]:
        m = network.model_from_state(torch.load(path, map_location='cpu'))
        c, s, g, n = score_model(m)
        print(f'{os.path.basename(path):<40} {m.features:<3} n {n}: corr {c:+.3f}  slope {s:+.3f}  mean gap {g:+.3f}')


if __name__ == '__main__':
    main()
