"""fit_input_scales.py -- per-column scale factors for the NEW inputs of A2/AB2.

Why (2026-10-14): fv2B's new inputs started at zero weight and stayed there --
after the run they carried ~1-3% of an old input's influence (weight x input
spread). Adam moves a weight by ~lr per step whatever its gradient, so an input
that barely varies (endgame-table value std 0.07, a rare blot flag 0.06) needs a
weight several times larger to matter and takes that much longer to get there.
Scaling each new column so its spread is ~TARGET (old inputs are mostly 0/1
flags, spread 0.2-0.5) fixes the speed without touching what it means.

Factors: TARGET / std, never below 1 (no shrinking) and capped at CAP (a rare
flag scaled further would put inputs far outside anything the net has seen).
Spread measured over owner's start-pool positions plus heuristic self-play (the
pool never shows the no-save counter moving). Only A2/AB2 (never trained before
this) are scaled, so no existing checkpoint changes meaning. encoder.js's twin of
features_v2 must apply the same factors -- it reads input_scales.json.

    python3 fit_input_scales.py   -> input_scales.json
"""
import json, os, random, sys
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
TARGET, CAP = 0.25, 5.0
N_POOL, N_GAMES = 3000, 20


def positions():
    from game import Board
    from agent import Agent
    recs = [json.loads(l) for l in open(os.path.join(REPO, 'start_pool.jsonl'))]
    random.seed(1)
    for r in random.sample(recs, min(N_POOL, len(recs))):
        b = Board(); b.update_state(r['state']); b.firstMove = None
        b.get_valid_moves()
        yield b
    ag = Agent(weights=None)
    for gi in range(N_GAMES):
        random.seed(9000 + gi); b = Board()
        for _ in range(200):
            w, _s = b.check_game_over()
            if w or b.draw_callable:
                break
            yield b
            pl = b.current_player
            ch = ag.select_move_pair(list(b.get_valid_moves()), b, pl)
            if isinstance(ch, tuple) and len(ch) == 3:
                ch = (ch, (0, 0, 0))
            for m in ch:
                if m != (0, 0, 0):
                    b.apply_move(m, switch_turn=False)
            b.switch_turn()


def main():
    import encoder as E
    enc = E.BoardEncoder(features='AB2', scaled=False)
    v1 = E.feature_dims('v1')
    T, P, G = [], [], []
    for b in positions():
        f = enc.encode(b, b.current_player)
        T.append(np.asarray(f['tile_feats'])[:, v1[0]:])
        P.append(np.asarray(f['piece_feats'])[:, v1[1]:])
        G.append(np.asarray(f['global_feats']).reshape(-1)[v1[2]:])
    T, P, G = np.concatenate(T), np.concatenate(P), np.stack(G)

    def fac(x):
        sd = x.std(0)
        return [round(float(min(CAP, max(1.0, TARGET / s))) if s > 0 else CAP, 2) for s in sd], \
               [round(float(s), 4) for s in sd]
    out = {'_doc': 'per-column factors for the NEW inputs (after v1\'s); see fit_input_scales.py',
           '_target': TARGET, '_cap': CAP, '_n_positions': len(G)}
    tf, ts = fac(T); pf, ps = fac(P); gf, gs = fac(G)
    out['AB2'] = {'tile': tf, 'piece': pf, 'global': gf}
    # A2 = AB2 without the threat columns (piece 1..4) and the 72 roll counts
    out['A2'] = {'tile': tf, 'piece': pf[:1], 'global': gf[:16]}
    out['_std_AB2'] = {'tile': ts, 'piece': ps, 'global': gs}
    with open(os.path.join(REPO, 'input_scales.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(f'{len(G)} positions; AB2 factors: tile {tf}, piece {pf}, global A2 {gf[:16]}, '
          f'rolls min/median/max {min(gf[16:])}/{sorted(gf[16:])[36]}/{max(gf[16:])}')


if __name__ == '__main__':
    main()
