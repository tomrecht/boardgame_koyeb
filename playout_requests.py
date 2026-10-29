"""playout_requests.py -- answer owner's "Play this turn out" requests.

The game (Settings > Game log > Play this turn out) writes
playouts/req/<id>.json to the private games repo: the turn's START and END
positions (getGameState format, which Board.update_state reads) and the mover.
This script, run by .github/workflows/playout-requests.yml on a schedule:
  1. lists requests without a result (playouts/res/<id>.json);
  2. scores every legal move from the start (net afterstate value), takes the
     champion's top TOP distinct moves, adds owner's move (the end position),
     and re-scores those with the 21-roll deep check;
  3. plays each out to the end N times with the champion's app search on both
     sides, the SAME dice for every move in playout i (common random numbers);
  4. writes playouts/res/<id>.json + .md and opens an issue with the report
     (GitHub then notifies owner).

    GAMES_TOKEN=... MODEL=<ckpt.pt> python3 playout_requests.py [max_requests=4]
    python3 playout_requests.py local <request.json> <out.md>     (no GitHub)
"""
import base64, copy, json, os, sys, time, urllib.error, urllib.request

os.environ.setdefault('GATE_PREFILTER', '12,40,5')
os.environ.setdefault('BOARDGAME_DEVICE', 'cpu')
REPO = os.environ.get('GAMES_REPO', 'tomrecht/quahuru-games')
N = int(os.environ.get('PLAYOUTS', '64'))
TOP = int(os.environ.get('TOP', '3'))
N_WORKERS = int(os.environ.get('N_WORKERS', str(os.cpu_count() or 4)))


def api(method, path, body=None):
    req = urllib.request.Request('https://api.github.com/repos/%s/%s' % (REPO, path), method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={'Authorization': 'Bearer ' + os.environ['GAMES_TOKEN'],
                                          'Accept': 'application/vnd.github+json',
                                          'User-Agent': 'quahuru-playouts'})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read() or b'null')
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def pending():
    req = api('GET', 'contents/playouts/req') or []
    res = api('GET', 'contents/playouts/res') or []
    done = {f['name'].rsplit('.', 1)[0] for f in res}
    return [f for f in req if f['name'].endswith('.json') and f['name'][:-5] not in done]


def _board(state, mover=None):
    from game import Board
    st = dict(state)
    if mover:
        st['currentTurn'] = mover
    b = Board()
    b.update_state(st)
    b._reachable_cache.clear()
    b.get_valid_moves()
    return b


def _arm_job(a):
    """One playout of one arm. a = (model, state, mover, pair_repr|None, key)."""
    import ast
    import quiz_playout as QP
    model, state, mover, pair, key = a
    ag = QP._agent(model)
    b = _board(state, mover)
    if pair:
        for m in ast.literal_eval(pair):
            if m not in ((0, 0, 0), (1, 1, 1)):
                b.apply_move(m, switch_turn=False)
    return QP._playout(ag, b, mover, key)


def solve(r, model):
    import numpy as np
    import review_games as RG
    import quiz_best as QB
    from agent_gnn import SCORE_SCALE, NUM_PIECES
    from multiprocessing import Pool
    t0 = time.time()
    mover = r['mover']
    b = _board(r['start'])
    ag = RG._agent(model)
    quick, rep = {}, {}
    for p in QB._pairs(b):
        k = QB._key(b, p)
        q = RG._shallow(ag, b, mover, p)
        if k not in quick or q > quick[k]:
            quick[k], rep[k] = q, p
    order = sorted(quick, key=lambda k: -quick[k])
    k_own = QB.canon(_board(r['end'], mover))
    arms = []
    for k in order[:TOP]:
        arms.append({'key': k, 'name': RG.pair_str(rep[k]), 'pair': repr(rep[k]), 'yours': k == k_own})
    if not any(a['yours'] for a in arms):
        arms.append({'key': k_own, 'name': RG.pair_str(rep[k_own]) if k_own in rep else '(your move)',
                     'pair': None, 'yours': True, 'legal': k_own in rep})
    import quiz_build as QBuild
    for a in arms:
        pr = a['pair'] or (repr(rep[a['key']]) if a['key'] in rep else None)
        a['moves'] = QBuild.moves(b, pr) if pr else []
        a['quick'] = round(quick.get(a['key'], float('nan')), 3)
        if a['key'] in rep:
            a['deep'] = round(ag._deep_value(b, mover, rep[a['key']]) / SCORE_SCALE * NUM_PIECES, 3)
    jobs = []
    for ai, a in enumerate(arms):
        state, pair = (r['end'], None) if (a['yours'] and a['pair'] is None) else (r['start'], a['pair'])
        for i in range(N):
            jobs.append((model, state, mover, pair, '%s:%d' % (r['id'], i)))
    with Pool(N_WORKERS) as pool:
        out = pool.map(_arm_job, jobs, chunksize=1)
    for ai, a in enumerate(arms):
        a['margins'] = out[ai * N:(ai + 1) * N]
    yours = next(a for a in arms if a['yours'])
    Y = np.array(yours['margins'], float)
    for a in arms:
        M = np.array(a['margins'], float)
        a['mean'] = round(float(M.mean()), 3)
        a['ci'] = round(float(1.96 * M.std(ddof=1) / np.sqrt(N)), 3)
        a['win'] = round(float((M > 0).mean()), 3)
        d = M - Y
        a['vs_yours'] = round(float(d.mean()), 3)
        a['vs_ci'] = round(float(1.96 * d.std(ddof=1) / np.sqrt(N)) if d.std() > 0 else 0.0, 3)
    import board_png
    for a in arms:
        a['png'] = board_png.render(b, a['moves'], '#7b4fe0' if a['yours'] else '#2b8a48')
    return {'id': r['id'], 'mover': mover, 'notation': RG.notation(b), 'n_moves': len(quick),
            'playouts': N, 'arms': arms, 'sec': round(time.time() - t0)}


def say(name):
    """'x>3.2(2) x>G1(5)' -> 'blank → 3.2 (2), blank → G1 (5)'."""
    import re
    out = []
    for h in name.split():
        m = re.match(r'^(\w+)>([^()]+)\((\d+)\)$', h)
        if h == 'pass':
            out.append('no move')
        elif not m:
            out.append(h)
        else:
            pc = 'blank' if m.group(1) == 'x' else m.group(1)
            out.append('block-save' if m.group(2) == '0' else
                       ('bank %s (%s)' % (pc, m.group(3)) if m.group(2) == 'bank' else '%s → %s (%s)' % (pc, m.group(2), m.group(3))))
    return ', '.join(out)


def report(res, img=lambda k: '%s-%d.png' % ('ID', k)):
    """Markdown report; img(k) is the URL of arm k's picture."""
    arms = sorted(enumerate(res['arms']), key=lambda ka: -ka[1]['mean'])
    best = max((a for a in res['arms'] if not a['yours']), key=lambda a: a['vs_yours'], default=None)
    if best is None or best['vs_yours'] - best['vs_ci'] <= 0:
        verdict = 'No move beats yours clearly when played out.'
    else:
        verdict = '**%s** beats your move by %+.2f ± %.2f points a game.' % (say(best['name']), best['vs_yours'], best['vs_ci'])
    lines = [verdict, '',
             '%s to move. Each move below was played to the end of the game %d times, the champion playing both sides; '
             'playout i throws the same dice for every move. ± = 95%% range.' % (res['mover'].capitalize(), res['playouts']), '']
    for rank, (k, a) in enumerate(arms, 1):
        tag = ' — **your move**' if a['yours'] else ''
        lines += ['### %d. %s%s' % (rank, say(a['name']), tag), '',
                  'Average final margin **%+.2f ± %.2f**, won %.0f%%%s. Champion\'s estimate %+.2f (quick) / %s (21-roll check).'
                  % (a['mean'], a['ci'], 100 * a['win'],
                     '' if a['yours'] else ', vs your move %+.2f ± %.2f' % (a['vs_yours'], a['vs_ci']),
                     a['quick'], ('%+.2f' % a['deep']) if 'deep' in a else '-'),
                  '', '![%s](%s)' % (say(a['name']), img(k)), '']
    lines += ['Position: `%s` · %d legal results · %d s' % (res['notation'], res['n_moves'], res['sec'])]
    return '\n'.join(lines)


def main(maxn=4):
    model = os.environ['MODEL']
    todo = pending()[:maxn]
    print(len(todo), 'request(s)', flush=True)
    for f in todo:
        rid = f['name'][:-5]
        meta = api('GET', 'contents/playouts/req/' + f['name'])
        r = json.loads(base64.b64decode(meta['content']).decode())
        r['id'] = rid
        res = solve(r, model)
        for k, a in enumerate(res['arms']):
            api('PUT', 'contents/playouts/res/%s-%d.png' % (rid, k),
                {'message': 'playout picture ' + rid, 'content': base64.b64encode(a.pop('png')).decode()})
        md = report(res, lambda k: '%s-%d.png' % (rid, k))
        print(md, flush=True)
        for name, body in (('%s.json' % rid, json.dumps(res)), ('%s.md' % rid, md)):
            api('PUT', 'contents/playouts/res/' + name,
                {'message': 'playout result ' + rid, 'content': base64.b64encode(body.encode()).decode()})
        page = 'https://github.com/%s/blob/main/playouts/res/%s.md' % (REPO, rid)
        body = report(res, lambda k: 'https://github.com/%s/blob/main/playouts/res/%s-%d.png?raw=true' % (REPO, rid, k))
        api('POST', 'issues', {'title': 'Playout: ' + say(next(a for a in res['arms'] if a['yours'])['name']),
                               'body': body + '\n\n[Report page with pictures](%s)' % page})


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'local':
        r = json.load(open(sys.argv[2]))
        r.setdefault('id', 'local')
        res = solve(r, os.environ.get('MODEL', 'distill_cand13.pt'))
        base = sys.argv[3].rsplit('.', 1)[0]
        for k, a in enumerate(res['arms']):
            open('%s-%d.png' % (base, k), 'wb').write(a.pop('png'))
        md = report(res, lambda k: '%s-%d.png' % (os.path.basename(base), k))
        open(sys.argv[3], 'w').write(md)
        print(md)
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
