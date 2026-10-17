import sys, os, json, zlib, random
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
import s56, sswap, splace, rollout_probe as RP, weakness_probe as W
N = int(os.environ.get('N', '40')); PER = int(os.environ.get('PER', '40'))
D = os.path.dirname(os.path.abspath(__file__))

def pref(item):
    r, n, A, B, swapped = item
    b = s56.setup(r); us = b.current_player
    st = dict(b.game_stages)
    sc = W.agent().select_move_pair(list(b.get_valid_moves()), b, us, return_scores=True)
    b.game_stages.update(st)
    rank = {repr(tuple(p)): (i, s) for i, (s, p) in enumerate(sc)}
    inf = lambda p: rank.get(repr(tuple(p)), (10**9, None))
    Bb = min(B, key=lambda p: inf(p)[0])
    opp = 'black' if us == 'white' else 'white'
    return {'game': r['game'], 'turn': r['turn'], 'n': n, 'us': us, 'swapped': swapped,
            'unsaved_us': 12 - len(b.get_save_rack(us)), 'unsaved_opp': 12 - len(b.get_save_rack(opp)),
            'A': repr(A), 'B': repr(Bb), 'rankA': inf(A)[0], 'rankB': inf(Bb)[0],
            'valA': round(inf(A)[1] * W.M, 3), 'valB': round(inf(Bb)[1] * W.M, 3),
            'best': round(sc[0][0] * W.M, 3), 'state': r['state']}

def job(t):
    p, which, i = t
    r = {'game': p['game'], 'turn': p['turn'], 'state': p['state']}
    b = s56.setup(r); us = b.current_player
    for m in eval(p[which]): b.apply_move(m, switch_turn=False)
    w, s = RP.rollout(b, seed=zlib.crc32(f"{p['game']}:{p['turn']}:{i}".encode()))
    return {'game': p['game'], 'turn': p['turn'], 'n': p['n'], 'opt': which, 'i': i,
            'margin': 0 if w is None else (s if w == us else -s)}

if __name__ == '__main__':
    ps = sswap.positions()
    late = lambda x: 12 - len(x[0]['state']['racks'][x[0]['state']['currentTurn'] + 'Saved']) <= 8
    # Late positions are scarce: keep them all. Otherwise prefer games not yet
    # used (positions from one game are correlated), never two at one game-turn.
    rng = random.Random(0); pick = [x for x in ps if late(x)]
    used = {x[0]['game'] for x in pick}; used_turn = {(x[0]['game'], x[0]['turn']) for x in pick}
    for n in (1, 2, 3, 4):
        pool_n = [x for x in ps if x[1] == n and not late(x)]
        rng.shuffle(pool_n)
        pool_n.sort(key=lambda x: x[0]['game'] in used)          # stable: unused games first
        k = 0
        for x in pool_n:
            if k == PER: break
            if (x[0]['game'], x[0]['turn']) in used_turn: continue
            pick.append(x); k += 1
            used.add(x[0]['game']); used_turn.add((x[0]['game'], x[0]['turn']))
    pick += [x for x in splace.positions() if late(x) and (x[0]['game'], x[0]['turn']) not in used_turn]
    print('positions', len(pick), 'late', sum(map(late, pick)), flush=True)
    with Pool(3, initializer=W.agent) as pool:
        prefs = pool.map(pref, pick)
        json.dump(prefs, open(f'{D}/swap_prefs.json', 'w'))
        print('prefs', len(prefs), flush=True)
        tasks = [(p, o, i) for i in range(N) for p in prefs for o in 'AB']
        with open(f'{D}/runswap.jsonl', 'a') as f:
            for res in pool.imap_unordered(job, tasks):
                f.write(json.dumps(res) + '\n'); f.flush()
    print('done', flush=True)
