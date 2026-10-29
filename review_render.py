"""Pick examples per theme from the verify file and render them with pos_image.mjs.
Writes review/img/<game8>_t<turn>.png and review/examples.json."""
import json, os, re, subprocess, sys
sys.path.insert(0, os.getcwd())
import review_games as RG

PER = int(os.environ.get('PER', '3'))
TOPN = int(os.environ.get('TOPN', '6'))
rows = [json.loads(l) for l in open('review/verify_cand13.jsonl')]
rows = [r for r in rows if 'deep_gap' in r and r['deep_gap'] >= RG.CONFIRM_MIN]
by = {}
for r in rows:
    diff = {k: r['d_best'][k] - r['d_own'][k] for k in r['d_best']}
    r['theme'] = RG.theme(diff, r['stage'])
    by.setdefault(r['theme'], []).append(r)
order = sorted(by, key=lambda t: -sum(r['deep_gap'] for r in by[t]))[:TOPN]


import ast
GAMES = {g['id']: g for g in RG.load_games()}
CHAMP, OWN = '#2e9e4f', '#7b4fe0'


def _loc(b, t):
    if t is None:
        return 'H'
    if t.type == 'save':
        return 'G%d' % t.number
    if t is b.home_tile:
        return 'H'
    return '%d.%d' % (t.ring, t.pos)


def arrows(r, key, col):
    """from>to:col per half of the pair, origins read off the rebuilt board."""
    b = RG._state_at(GAMES[r['game']], r['turn'])
    if b is None:
        return []
    out, moved = [], {}
    for m in ast.literal_eval(r[key]):
        if m in ((0, 0, 0), (1, 1, 1)):
            continue
        pid, dest, _ = m
        p = next((q for q in b.pieces if (q.player, q.number) == tuple(pid)), None)
        if p is None:
            continue
        frm = moved.get(tuple(pid)) or _loc(b, p.tile)
        to = '' if dest in ('save', 0) else RG._where(dest)
        out.append('%s>%s:%s' % (frm, to, col))
        if to:
            moved[tuple(pid)] = to
    return out


ex = []
env = dict(os.environ, PATH='/usr/local/bin:' + os.environ['PATH'])
for t in order:
    for r in sorted(by[t], key=lambda r: -r['deep_gap'])[:PER]:
        name = 'review/img/%s_t%d.png' % (r['game'][:8], r['turn'])
        arr = arrows(r, 'best_pair', CHAMP) + arrows(r, 'own_pair', OWN)
        args = ['node', 'pos_image.mjs', r['pos'], name]
        if arr:
            args += ['--arrows', ';'.join(arr)]
        if not os.path.exists(name):
            p = subprocess.run(args, env=env, capture_output=True, text=True)
            print(t, name, p.returncode, (p.stdout + p.stderr).strip()[-120:], flush=True)
        ex.append({k: r[k] for k in ('game', 'turn', 'stage', 'pos', 'own', 'best', 'gap',
                                     'deep_gap', 'theme', 'model', 'dice')} | {'img': name})
json.dump({'themes': order, 'examples': ex}, open('review/examples.json', 'w'), indent=1)
print('done', len(ex))
