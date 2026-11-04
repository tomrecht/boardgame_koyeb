"""SAVE vs BRING-TO-GOAL, general version (owner, 2026-10-16 idea; run 2026-11-04).

When a roll lets the mover save pieces, and some other pair gives up a save to
bring a piece onto a goal it can bank from, which is better?

  A = the net's best pair among those with the MOST own saves (>= 1)
  B = the net's best pair with FEWER saves that brings a piece onto a bankable
      goal: a numbered piece onto its own goal (not already on it at the turn
      start), or a blank onto any goal from a non-goal tile (owner: goal-to-goal
      is shuffling). Entering from the rack on a sum of 7 counts.

Stages (each resumable, files in this folder):
  screen   every start_pool position x 21 rolls, move generation only: keep
           (position, roll) where a save is legal and an arrival is reachable
           -> screen.jsonl
  prefs    a sample of those, scored by the net (all pairs), A and B picked
           -> prefs.jsonl (only rows where both exist)
  play     paired playouts from A and B (net 1-ply both sides, common dice per
           index) for a stratified sample -> play.jsonl
Env: NET (checkpoint, default distill_cand13.pt), N_WORKERS (8), PREFS_MAX
(3000), PLAY_POS (400), N (40 playouts per option).
"""
import sys, os, json, zlib, random
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(D)); sys.path.insert(0, D)
os.environ.setdefault('NET', os.path.join(os.path.dirname(D), 'distill_cand13.pt'))
from multiprocessing import Pool
from game import Board

ROLLS = [(a, b) for a in range(1, 7) for b in range(a, 7)]
NW = int(os.environ.get('N_WORKERS', '8'))


def setup(state, roll):
    b = Board(); b.update_state(state); b.firstMove = None
    b.dice[0].number, b.dice[1].number = roll
    for d in b.dice: d.used = False
    return b


def piece_tile(b, key):
    p = next((p for p in b.pieces if p.player == key[0] and p.number == key[1]), None)
    return None if p is None else p.tile


def is_arrival(start_tile, num, dest_tile):
    """dest_tile reached by a piece numbered `num` that began the turn on start_tile."""
    if dest_tile is None or dest_tile.type != 'save':
        return False
    if num <= 6:
        return dest_tile.number == num and not (start_tile is not None and start_tile is dest_tile)
    return not (start_tile is not None and start_tile.type == 'save')


def arrivals(b, pair, us):
    """Arrivals in a pair, judged from the turn-start board b: list of piece numbers."""
    out = []
    for m in pair:
        if not (isinstance(m, tuple) and len(m) == 3 and isinstance(m[0], tuple)
                and m[0][0] == us and isinstance(m[1], tuple) and m[1] != (0, 0)):
            continue
        if is_arrival(piece_tile(b, m[0]), m[0][1], b.get_tile(*m[1])):
            out.append(m[0][1])
    return out


def saves(pair, us):
    return [m[0][1] for m in pair if isinstance(m, tuple) and len(m) == 3 and m[1] == 'save'
            and isinstance(m[0], tuple) and m[0][0] == us]


# ---------------------------------------------------------------- screen
def screen_one(r):
    out = []
    for roll in ROLLS:
        b = setup(r['state'], roll); us = b.current_player
        mv = list(b.get_valid_moves())
        if not any(m[1] == 'save' for m in mv if isinstance(m, tuple) and len(m) == 3):
            continue
        arr = any(isinstance(m[1], tuple) and m[1] != (0, 0) and isinstance(m[0], tuple)
                  and is_arrival(piece_tile(b, m[0]), m[0][1], b.get_tile(*m[1])) for m in mv)
        if not arr:   # two halves on one piece (incl. a rack entry on a sum of 7)
            for m in mv:
                if not (isinstance(m[1], tuple) and isinstance(m[0], tuple)):
                    continue
                st0 = piece_tile(b, m[0]); n0 = len(b.moves)
                b.apply_move(m, switch_turn=False)
                arr = any(m2[0] == m[0] and isinstance(m2[1], tuple) and m2[1] != (0, 0)
                          and is_arrival(st0, m[0][1], b.get_tile(*m2[1])) for m2 in b.get_valid_moves())
                while len(b.moves) > n0: b.undo_last_move()
                if arr: break
        if arr:
            out.append({'game': r['game'], 'turn': r['turn'], 'roll': roll})
    return out


# ---------------------------------------------------------------- prefs
_AG = []
def agent():
    if not _AG:
        import weakness_probe as W
        _AG.append(W.agent())
    return _AG[0]


def pref_one(item):
    import weakness_probe as W
    r, roll = item
    b = setup(r['state'], roll); us = b.current_player
    st = dict(b.game_stages)
    sc = agent().select_move_pair(list(b.get_valid_moves()), b, us, return_scores=True)
    b.game_stages.update(st)
    if len(sc) == 1 and sc[0][0] == float('inf'):
        return None                       # a pair wins outright
    ns = [len(saves(p, us)) for _, p in sc]
    mx = max(ns)
    if mx == 0:
        return None
    A = next((s, p) for (s, p), n in zip(sc, ns) if n == mx)
    B = next(((s, p, arrivals(b, p, us)) for (s, p), n in zip(sc, ns) if n < mx and arrivals(b, p, us)), None)
    if B is None:
        return None
    opp = 'black' if us == 'white' else 'white'
    rank = lambda p: next(i for i, (_, q) in enumerate(sc) if q == p)
    top_n = ns[0]; top_arr = arrivals(b, sc[0][1], us)
    return {'game': r['game'], 'turn': r['turn'], 'roll': list(roll), 'us': us,
            'stage_us': b.get_game_stage(us), 'stage_opp': b.get_game_stage(opp),
            'unsaved_us': 12 - len(b.get_save_rack(us)), 'unsaved_opp': 12 - len(b.get_save_rack(opp)),
            'A': repr(A[1]), 'B': repr(B[1]), 'A_saves': saves(A[1], us), 'B_saves': saves(B[1], us),
            'A_arr': arrivals(b, A[1], us), 'B_arr': B[2],
            'valA': round(A[0] * W.M, 3), 'valB': round(B[0] * W.M, 3), 'rankA': rank(A[1]), 'rankB': rank(B[1]),
            'top': 'A' if top_n == mx else ('B' if top_arr else 'other'), 'n_pairs': len(sc)}


# ---------------------------------------------------------------- play
def play_one(t):
    import rollout_probe as RP
    p, which, i = t
    b = setup(POOL[(p['game'], p['turn'])]['state'], tuple(p['roll'])); us = b.current_player
    for m in eval(p[which]):
        if m != (0, 0, 0): b.apply_move(m, switch_turn=False)
    w, s = RP.rollout(b, seed=zlib.crc32(f"{p['game']}:{p['turn']}:{p['roll']}:{i}".encode()))
    return {'game': p['game'], 'turn': p['turn'], 'roll': p['roll'], 'opt': which, 'i': i,
            'margin': 0 if w is None else (s if w == us else -s)}


def load_pool():
    pool = {}
    for line in open(os.path.join(os.path.dirname(D), 'start_pool.jsonl')):
        r = json.loads(line); pool[(r['game'], r['turn'])] = r
    return pool

POOL = load_pool()


def kind(p):
    """Stratum: what B brings in, saves given up, late or not."""
    nb = 'num' if any(n <= 6 for n in p['B_arr']) else 'blank'
    forgo = len(p['A_saves']) - len(p['B_saves'])
    late = 'late' if p['unsaved_us'] <= 8 else 'early'
    return f"{nb}-{min(forgo, 2)}-{late}"


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'screen':
        rs = list(POOL.values())
        with Pool(NW) as pool, open(f'{D}/screen.jsonl', 'w') as f:
            n = 0
            for out in pool.imap_unordered(screen_one, rs, chunksize=20):
                for o in out: f.write(json.dumps(o) + '\n')
                n += 1
                if n % 2000 == 0: print('screened', n, flush=True)
        print('screen done', flush=True)
    elif mode == 'prefs':
        cand = [json.loads(l) for l in open(f'{D}/screen.jsonl')]
        rng = random.Random(1); rng.shuffle(cand)
        # spread over games: at most 2 rolls per position, 12 per game
        per_pos, per_game, pick = {}, {}, []
        for c in cand:
            k = (c['game'], c['turn'])
            if per_pos.get(k, 0) >= 2 or per_game.get(c['game'], 0) >= 12: continue
            per_pos[k] = per_pos.get(k, 0) + 1; per_game[c['game']] = per_game.get(c['game'], 0) + 1
            pick.append(c)
            if len(pick) >= int(os.environ.get('PREFS_MAX', '3000')): break
        done = set()
        if os.path.exists(f'{D}/prefs_done.txt'):
            done = set(open(f'{D}/prefs_done.txt').read().split())
        todo = [(POOL[(c['game'], c['turn'])], tuple(c['roll'])) for c in pick
                if f"{c['game']}:{c['turn']}:{c['roll'][0]}{c['roll'][1]}" not in done]
        print('prefs: picked', len(pick), 'todo', len(todo), flush=True)
        with Pool(NW, initializer=agent) as pool, open(f'{D}/prefs.jsonl', 'a') as f, \
                open(f'{D}/prefs_done.txt', 'a') as fd:
            for (r, roll), res in zip(todo, pool.imap(pref_one, todo, chunksize=4)):
                if res: f.write(json.dumps(res) + '\n'); f.flush()
                fd.write(f"{r['game']}:{r['turn']}:{roll[0]}{roll[1]}\n"); fd.flush()
        print('prefs done', flush=True)
    elif mode == 'play':
        prefs = [json.loads(l) for l in open(f'{D}/prefs.jsonl')]
        rng = random.Random(2); rng.shuffle(prefs)
        strata = {}
        for p in prefs: strata.setdefault(kind(p), []).append(p)
        want = int(os.environ.get('PLAY_POS', '400'))
        # equal share per stratum (all of a small one), filled round-robin
        pick, used_pos = [], set()
        while len(pick) < want and any(strata.values()):
            for k in sorted(strata):
                while strata[k]:
                    p = strata[k].pop()
                    if (p['game'], p['turn']) in used_pos: continue
                    pick.append(p); used_pos.add((p['game'], p['turn'])); break
                if len(pick) >= want: break
        json.dump(pick, open(f'{D}/play_pick.json', 'w'))
        from collections import Counter
        print('play: positions', len(pick), dict(Counter(kind(p) for p in pick)), flush=True)
        N = int(os.environ.get('N', '40'))
        done = set()
        if os.path.exists(f'{D}/play.jsonl'):
            for l in open(f'{D}/play.jsonl'):
                x = json.loads(l); done.add((x['game'], x['turn'], tuple(x['roll']), x['opt'], x['i']))
        tasks = [(p, o, i) for i in range(N) for p in pick for o in 'AB'
                 if (p['game'], p['turn'], tuple(p['roll']), o, i) not in done]
        print('play: tasks', len(tasks), flush=True)
        import weakness_probe as W
        with Pool(NW, initializer=W.agent) as pool, open(f'{D}/play.jsonl', 'a') as f:
            for k, res in enumerate(pool.imap_unordered(play_one, tasks, chunksize=2)):
                f.write(json.dumps(res) + '\n'); f.flush()
                if k % 2000 == 0: print('played', k, flush=True)
        print('play done', flush=True)
