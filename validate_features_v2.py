"""Check features_v2's approximations against ground truth on owner's games.

  threats : capture_prob / wall_prob against weakness_probe.threats (every legal
            opponent reply over all 21 rolls), for the mover's numbered field
            pieces in the position right after its move.
  turns   : side_summary's estimated turns to finish against the turns the
            WINNER actually still took (the loser never finishes).

Usage: python3 validate_features_v2.py [n_games=20] [every_kth_turn=3]
"""
import json, math, sys, time

import features_v2 as F
import weakness_probe as W

LOGS = ['quahuru-games-apvo2h65.jsonl', 'quahuru-games-f7in6olg.jsonl']


def corr(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb) if va and vb else float('nan')


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    every = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    recs = []
    for f in LOGS:
        for line in open(f):
            r = json.loads(line)
            if r.get('completed') and r['whiteIsAI'] != r['blackIsAI']:
                recs.append(r)
    recs = recs[:n_games]
    cap_x, cap_a, blk_x, blk_a = [], [], [], []
    t_exact = t_approx = 0.0
    turns_est, turns_true = [], []
    for rec in recs:
        n_turns = len(rec['turns'])
        winner = rec['result']['winner']
        for t, b, played, b2 in W.turn_states(rec):
            us = b.current_player
            # turns-to-finish: winner's estimate at the start of its own turns
            if us == winner:
                est = F.side_summary(b, us)[4]
                left = sum(1 for k in range(t, n_turns) if rec['turns'][k]['p'] == us[0])
                turns_est.append(est)
                turns_true.append(left)
            if t % every:
                continue
            t0 = time.time()
            exact = W.threats(b2, us)
            t_exact += time.time() - t0
            t0 = time.time()
            for p in b2.pieces:
                if p.player == us and p.number <= 6 and p.tile is not None and p.tile.type == 'field':
                    c, w = F.capture_prob(b2, p), F.wall_prob(b2, p)
                    ec, ew = exact[p.number]
                    cap_x.append(ec); cap_a.append(c); blk_x.append(ew); blk_a.append(w)
            t_approx += time.time() - t0
    n = len(cap_x)
    print(f'{len(recs)} games, {n} numbered field pieces compared')
    for name, x, a in (('capture', cap_x, cap_a), ('wall', blk_x, blk_a)):
        mae = sum(abs(p - q) for p, q in zip(x, a)) / n
        bias = sum(q - p for p, q in zip(x, a)) / n
        print(f'  {name:8} exact mean {sum(x)/n:.3f}  approx mean {sum(a)/n:.3f}  '
              f'corr {corr(x, a):.3f}  MAE {mae:.3f}  bias {bias:+.3f}')
        big = sorted(zip(x, a), key=lambda r: -abs(r[0] - r[1]))[:5]
        print(f'           worst (exact, approx): {[(round(p,2), round(q,2)) for p, q in big]}')
    print(f'  time: exact {1000*t_exact/max(1,n):.0f} ms/piece, approx {1000*t_approx/max(1,n):.1f} ms/piece')
    print(f'turns to finish (winner): corr {corr(turns_est, turns_true):.3f} over {len(turns_est)} positions, '
          f'mean est {sum(turns_est)/len(turns_est):.1f} vs true {sum(turns_true)/len(turns_true):.1f}')


if __name__ == '__main__':
    main()
