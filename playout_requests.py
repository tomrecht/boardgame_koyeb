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
    for a in arms:
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
    return {'id': r['id'], 'mover': mover, 'notation': RG.notation(b), 'n_moves': len(quick),
            'playouts': N, 'arms': arms, 'sec': round(time.time() - t0)}


def report(res):
    lines = ['Position (%s to move, %d legal results): `%s`' % (res['mover'], res['n_moves'], res['notation']), '',
             '%d playouts per move, to the end of the game, the champion playing both sides; '
             'playout i throws the same dice for every move.' % res['playouts'], '',
             '| move | champion quick / deep | avg final margin | won | vs your move |',
             '|---|---|---|---|---|']
    for a in sorted(res['arms'], key=lambda a: -a['mean']):
        name = ('**%s** (yours)' % a['name']) if a['yours'] else a['name']
        qd = '%+.2f / %s' % (a['quick'], ('%+.2f' % a['deep']) if 'deep' in a else '-')
        vs = '-' if a['yours'] else '%+.2f ± %.2f' % (a['vs_yours'], a['vs_ci'])
        lines.append('| %s | %s | %+.2f ± %.2f | %.0f%% | %s |' % (name, qd, a['mean'], a['ci'], 100 * a['win'], vs))
    best = max((a for a in res['arms'] if not a['yours']), key=lambda a: a['vs_yours'], default=None)
    lines.append('')
    if best is None or best['vs_yours'] - best['vs_ci'] <= 0:
        lines.append('Verdict: no move beats yours clearly in the playouts.')
    else:
        lines.append('Verdict: %s beats your move by %+.2f ± %.2f points a game.' % (best['name'], best['vs_yours'], best['vs_ci']))
    lines.append('')
    lines.append('Quick / deep = the champion\'s own estimate (points, mover\'s side); ± = 95%% range. %d s.' % res['sec'])
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
        md = report(res)
        print(md, flush=True)
        for name, body in (('%s.json' % rid, json.dumps(res)), ('%s.md' % rid, md)):
            api('PUT', 'contents/playouts/res/' + name,
                {'message': 'playout result ' + rid, 'content': base64.b64encode(body.encode()).decode()})
        api('POST', 'issues', {'title': 'Playout result %s' % rid, 'body': md})


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'local':
        r = json.load(open(sys.argv[2]))
        r.setdefault('id', 'local')
        res = solve(r, os.environ.get('MODEL', 'distill_cand13.pt'))
        open(sys.argv[3], 'w').write(report(res))
        print(report(res))
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
