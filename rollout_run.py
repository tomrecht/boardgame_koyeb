"""rollout_run.py -- the rollout-teacher experiment, unattended (2026-10-26).

Does a DEEPER teacher distil into a better net than today's 1-turn close-call
search? Same decisions, same games, same training, three teachers:
  h1  est_1 = the 21-roll close-call value, re-run with the CURRENT champion
  h2  luck-adjusted rollouts 2 plies deep (opponent, then us)
  h4  4 plies deep (the main arm)
(rollout_label.py; ROLL_H=4 stores every depth, ROLL_N=4 playouts per candidate.)

Stages (state in rollout/state.json; rerun the same command to resume):
  relabel  GitHub blocks GH_BLOCKS of the decision file (remote_play 'relabel');
           this Mac labels from LOCAL_FROM up meanwhile (rollout_local.sh);
           whatever is still missing is then labelled here
  train    distill_train.py from the champion on each arm's labels
  confirm  each arm vs the champion, 2,000 pairs on GitHub (seeds 170M/172M/174M),
           then h4 vs h1 head to head (176M); local fallback if GitHub fails
Results: rollout/RESULT.txt (also printed).

    caffeinate -is nohup .venv/bin/python -u rollout_run.py >> rollout/run.log 2>&1 &
"""
import glob, json, math, os, subprocess, sys, time

import remote_play as RP

D = 'rollout'
PY = sys.executable
NET = os.environ.get('NET', 'distill/cand_13.pt')
DEC = os.path.join(D, 'decisions_A.jsonl.gz')
GH_BLOCKS = [(0, 1200), (1200, 1200), (2400, 1200)]
ARMS = [4, 1, 2]
TRAIN_ARGS = ['--epochs', '8', '--lr', '2e-5', '--td-w', '0.5', '--rank-w', '0']
PAIRS = int(os.environ.get('PAIRS', '2000'))
ENV = {'ROLL_H': '4', 'ROLL_N': '4'}
ST = os.path.join(D, 'state.json')


def load():
    return json.load(open(ST)) if os.path.exists(ST) else {}


def save(s):
    json.dump(s, open(ST, 'w'), indent=1)


def log(*a):
    print(time.strftime('%m-%d %H:%M'), *a, flush=True)


def n_decisions():
    import gzip
    return sum(1 for l in gzip.open(DEC, 'rt') if '"kind": "decision"' in l)


def have():
    got = set()
    for p in glob.glob(os.path.join(D, 'relabel_*.jsonl')):
        for l in open(p):
            try:
                got.add(json.loads(l)['idx'])
            except (ValueError, KeyError):
                pass
    return got


def paired(path):
    per = {}
    for l in open(path):
        r = json.loads(l)
        per.setdefault(r['seed'], {})[r['a_white']] = r['a_margin']
    u = [0.5 * (v[True] + v[False]) for v in per.values() if len(v) == 2]
    n = len(u)
    m = sum(u) / n
    se = math.sqrt(sum((x - m) ** 2 for x in u) / (n - 1) / n)
    return m, se, n


def sh(cmd, logf, env=None):
    log('$', ' '.join(cmd))
    with open(logf, 'a') as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=dict(os.environ, **(env or {})))
    if rc:
        raise SystemExit(f'failed ({rc}): {" ".join(cmd)} -- see {logf}')


def match(s, key, a, b, seed):
    out = os.path.join(D, f'confirm_{key}.jsonl')
    if not s.get('remote_off'):
        try:
            miss = RP.remote_play('confirm', [a, b], PAIRS, seed, out)
            if not miss:
                return paired(out)
        except RP.RemoteError as e:
            log('remote failed:', e)
            s['remote_fail'] = s.get('remote_fail', 0) + 1
            if s['remote_fail'] >= 2:
                s['remote_off'] = True
            save(s)
    sh([PY, '-u', 'confirm_match.py', a, b, str(PAIRS), out], os.path.join(D, f'confirm_{key}.log'),
       {'SEED_BASE': str(seed)})
    return paired(out)


def main():
    s = load()
    total = n_decisions()
    # 1. GitHub blocks
    for start, cnt in GH_BLOCKS:
        k = f'gh_{start}'
        if s.get(k) or s.get('remote_off'):
            continue
        out = os.path.join(D, f'relabel_gh_{start}.jsonl')
        try:
            miss = RP.remote_play('relabel', [NET, DEC], cnt, start, out, env=ENV)
            log(f'block {start}: {cnt - miss}/{cnt} labelled on GitHub')
        except RP.RemoteError as e:
            log(f'block {start}: remote failed: {e}')
            s['remote_fail'] = s.get('remote_fail', 0) + 1
            if s['remote_fail'] >= 2:
                s['remote_off'] = True
        s[k] = True
        save(s)
    # 2. wait for the local labeller, then fill whatever is missing here
    while not os.path.exists(os.path.join(D, 'local.done')):
        time.sleep(300)
    if not s.get('filled'):
        miss = total - len(have())
        log(f'{miss} decisions still missing; labelling them here')
        if miss:
            sh([PY, '-u', 'rollout_label.py', 'run', NET, DEC, os.path.join(D, 'relabel_fill.jsonl')],
               os.path.join(D, 'fill.log'),
               dict(ENV, SEED_BASE='0', DONE_GLOB=os.path.join(D, 'relabel_*.jsonl')))
        s['filled'] = True
        save(s)
    files = sorted(glob.glob(os.path.join(D, 'relabel_*.jsonl')))
    subprocess.call([PY, 'rollout_label.py', 'stats', *files])
    # 3. train each arm
    for h in ARMS:
        k = f'train_h{h}'
        if s.get(k):
            continue
        data = os.path.join(D, f'train_h{h}.jsonl.gz')
        sh([PY, 'rollout_label.py', 'build', DEC, data, str(h), *files], os.path.join(D, 'build.log'))
        sh([PY, '-u', 'distill_train.py', '--data', data, '--load', NET,
            '--save', os.path.join(D, f'stud_h{h}.pt'), '--log', os.path.join(D, 'train_log.jsonl'),
            *TRAIN_ARGS], os.path.join(D, f'train_h{h}.log'))
        s[k] = True
        save(s)
    # 4. matches
    res = s.setdefault('results', {})
    plan = [(f'h{h}_vs_champ', os.path.join(D, f'stud_h{h}.pt'), NET, 170_000_000 + 2_000_000 * i)
            for i, h in enumerate(ARMS)]
    plan.append(('h4_vs_h1', os.path.join(D, 'stud_h4.pt'), os.path.join(D, 'stud_h1.pt'), 176_000_000))
    for key, a, b, seed in plan:
        if key in res:
            continue
        m, se, n = match(s, key, a, b, seed)
        res[key] = {'mean': m, 'se': se, 'pairs': n}
        save(s)
        log(f'{key}: {m:+.3f} (95% CI {m - 1.96 * se:+.3f}..{m + 1.96 * se:+.3f}), {n} pairs')
    lines = [f'{k}: {v["mean"]:+.3f} (95% CI {v["mean"] - 1.96 * v["se"]:+.3f}..'
             f'{v["mean"] + 1.96 * v["se"]:+.3f}), {v["pairs"]} pairs' for k, v in res.items()]
    open(os.path.join(D, 'RESULT.txt'), 'w').write('\n'.join(lines) + '\n')
    log('DONE\n' + '\n'.join(lines))


if __name__ == '__main__':
    main()
