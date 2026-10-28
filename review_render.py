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


def dests(s):
    out = []
    for m in re.finditer(r'>([^ (]+)\(', s):
        d = m.group(1)
        if d not in ('bank', 'H') and not d.startswith('call'):
            out.append(d)
    return out


ex = []
env = dict(os.environ, PATH='/usr/local/bin:' + os.environ['PATH'])
for t in order:
    for r in sorted(by[t], key=lambda r: -r['deep_gap'])[:PER]:
        name = 'review/img/%s_t%d.png' % (r['game'][:8], r['turn'])
        champ = dests(r['best'])
        mine = [d for d in dests(r['own'])]
        args = ['node', 'pos_image.mjs', r['pos'], name]
        if champ:
            args += ['--fill', ';'.join('%s:#5fae6a' % d for d in dict.fromkeys(champ))]
        if mine:
            args += ['--ring', ','.join(dict.fromkeys(mine))]
        if not os.path.exists(name):
            p = subprocess.run(args, env=env, capture_output=True, text=True)
            print(t, name, p.returncode, (p.stdout + p.stderr).strip()[-120:], flush=True)
        ex.append({k: r[k] for k in ('game', 'turn', 'stage', 'pos', 'own', 'best', 'gap',
                                     'deep_gap', 'theme', 'model', 'dice')} | {'img': name})
json.dump({'themes': order, 'examples': ex}, open('review/examples.json', 'w'), indent=1)
print('done', len(ex))
