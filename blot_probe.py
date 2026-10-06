"""Does 'enemy blots on a numbered piece's shortest route' predict it getting
walled within the next few turns, beyond the 1-roll P(walled)?

Snapshot at every turn end (after the mover's moves, before the switch), for
every numbered piece on a field tile, both sides. Outcome: the piece's wall
penalty (route length with enemy walls minus without; None=cut counts) goes
positive at any of the next H snapshots while it is still a numbered field
piece. Only snapshots where the penalty is currently 0 are used.
"""
import sys, json, glob
import numpy as np
import replay_games as R
import features_v2 as F

H = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def goal_of(board, p):
    return next(g for g in board.tiles if g.type == 'save' and g.number == p.number)


def piece_obs(board):
    out = {}
    nofield = frozenset()
    for p in board.pieces:
        if p.number > 6 or p.tile is None or p.tile.type != 'field':
            continue
        goal = goal_of(board, p)
        blk = F._walled_for(board, p.player)
        base = F._route_len_raw(p.tile, goal, blk)
        free = F._route_len_raw(p.tile, goal, nofield)
        pen = None if base is None else base - free
        rec = {'pen': pen}
        if pen == 0:
            d_from = F._dist_from(board, p.tile, blk)
            d_to = F._dist_from(board, goal, blk)
            route = [t for t in board.tiles if t.type == 'field' and t is not p.tile
                     and t in d_from and t in d_to and d_from[t] + d_to[t] == base]
            enemy = F._other(p.player)
            rec['blots'] = sum(1 for t in route if len(t.pieces) == 1 and t.pieces[0].player == enemy)
            rec['route'] = base
            rec['pwall'] = F.wall_prob(board, p)
            rec['enemy_field'] = sum(1 for q in board.pieces if q.player == enemy
                                     and q.tile is not None and q.tile.type == 'field')
            rec['n_route_tiles'] = len(route)
        out[(p.player, p.number)] = rec
    return out


def run_game(rec):
    snaps = []
    orig_switch = R.Board.switch_turn

    def hooked(self, *a, **k):
        snaps.append(piece_obs(self))
        return orig_switch(self, *a, **k)
    R.Board.switch_turn = hooked
    try:
        ok, msg = R.replay(rec)
    finally:
        R.Board.switch_turn = orig_switch
    return snaps


def main():
    seen, recs = set(), []
    for f in sorted(glob.glob('quahuru-games-*.jsonl')):
        for l in open(f):
            r = json.loads(l)
            if r['id'] in seen:
                continue
            seen.add(r['id']); recs.append(r)
    rows = []
    for gi, rec in enumerate(recs):
        snaps = run_game(rec)
        for s, obs in enumerate(snaps):
            for key, o in obs.items():
                if o['pen'] != 0 or s + H >= len(snaps):
                    continue
                y = 0
                for s2 in range(s + 1, s + H + 1):
                    o2 = snaps[s2].get(key)
                    if o2 is None:
                        break           # left the field (goal, captured, saved)
                    if o2['pen'] is None or o2['pen'] > 0:
                        y = 1; break
                rows.append((gi, o['pwall'], o['blots'], o['route'], o['enemy_field'], y))
    a = np.array(rows, float)
    np.save('blot_rows_H%d.npy' % H, a)
    print(f'games {len(recs)}  obs {len(a)}  walled-within-{H} rate {a[:,5].mean():.3f}')


if __name__ == '__main__':
    main()
