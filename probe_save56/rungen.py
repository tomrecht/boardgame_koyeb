import sys, os, json, zlib
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
import s56, sgen, rollout_probe as RP, weakness_probe as W
N = int(os.environ.get('N', '40'))
D = os.path.dirname(os.path.abspath(__file__))
DONE = {('3c25f709', 1), ('506cc169', 1), ('48c9e1c7', 1)}

def pref(item):
    r, n, A, B = item
    b = s56.setup(r); us = b.current_player
    st = dict(b.game_stages)
    sc = W.agent().select_move_pair(list(b.get_valid_moves()), b, us, return_scores=True)
    b.game_stages.update(st)
    rank = {repr(tuple(p)): (i, s) for i, (s, p) in enumerate(sc)}
    inf = lambda p: rank.get(repr(tuple(p)), (10**9, None))
    Bb = min(B, key=lambda p: inf(p)[0])
    unsaved = lambda s: 12 - len(b.get_save_rack(s))
    opp = 'black' if us == 'white' else 'white'
    return {'game': r['game'], 'turn': r['turn'], 'n': n, 'us': us,
            'unsaved_us': unsaved(us), 'unsaved_opp': unsaved(opp),
            'A': repr(A), 'B': repr(Bb), 'rankA': inf(A)[0], 'rankB': inf(Bb)[0],
            'valA': round(inf(A)[1] * W.M, 3), 'valB': round(inf(Bb)[1] * W.M, 3),
            'best': round(sc[0][0] * W.M, 3)}

def job(t):
    r, n, which, pair, i = t
    b = s56.setup(r); us = b.current_player
    for m in pair: b.apply_move(m, switch_turn=False)
    w, s = RP.rollout(b, seed=zlib.crc32(f"{r['game']}:{i}".encode()))
    return {'game': r['game'], 'n': n, 'opt': which, 'i': i, 'margin': 0 if w is None else (s if w == us else -s)}

if __name__ == '__main__':
    items = [x for x in sgen.positions() if (x[0]['game'], x[1]) not in DONE]
    with Pool(3, initializer=W.agent) as pool:
        prefs = pool.map(pref, items)
        json.dump(prefs, open(f'{D}/gen_prefs.json', 'w'))
        print('prefs', len(prefs), flush=True)
        byk = {(x[0]['game'], x[1]): x[0] for x in items}
        tasks = []
        for i in range(N):
            for p in prefs:
                r = byk[(p['game'], p['n'])]
                tasks += [(r, p['n'], 'A', eval(p['A']), i), (r, p['n'], 'B', eval(p['B']), i)]
        with open(f'{D}/rungen.jsonl', 'a') as f:
            for res in pool.imap_unordered(job, tasks):
                f.write(json.dumps(res) + '\n'); f.flush()
    print('done', flush=True)
