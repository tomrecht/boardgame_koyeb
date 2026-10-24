"""distill_run.py -- expert iteration: label with the close-call search, distil
into the net, keep it only if it beats the champion on fresh seeds.

Per iteration k:
  1. LABEL    close_call.py label <champion> T K  -> distill/labels_k.jsonl.gz
              (self-play, search on both sides; SEED_BASE 91M + 10k*k)
  2. TRAIN    distill_train.py on the last REPLAY files, warm start = champion
              -> distill/cand_k.pt
  3. CONFIRM  confirm_match.py cand_k champion CONFIRM_PAIRS, app config, fresh
              seed block (93M + 10k*k) -> promote iff paired mean > BAR
  The champion labels the next round, so a promotion compounds.

    BOARDGAME_DEVICE=cpu PYTHONHASHSEED=0 N_WORKERS=8 \
      nohup python -u distill_run.py >> distill/run.log 2>&1 &

Every step is resumable (each script skips finished games; a finished step is
recorded in distill/state.json), so rerunning the same command continues.

Knobs (env; defaults are a first guess):
    START=fv2C_it55.pt  ITERS=6  T=3  K=3  GAMES=300  REPLAY=2
    CONFIRM_PAIRS=300  BAR=0.0
    TRAIN_ARGS='--epochs 8 --lr 2e-5 --td-w 0.5 --rank-w 0'
    MAX_REPLIES (passed through to labelling; 0 = full 21-roll search)
    REMOTE=1  play LABEL and CONFIRM on GitHub Actions (remote_play.py; the
              branch must be pushed), TRAIN locally. REMOTE_SHARDS (20).
              Same seeds as a local run, so the two are interchangeable.
    START_POOL / START_FRAC (labelling games from owner's positions; default
    start_pool.jsonl at 0.3 if present)
"""
import json, math, os, shlex, subprocess, sys

D = 'distill'
PY = sys.executable
E = os.environ
START = E.get('START', 'fv2C_it55.pt')
ITERS = int(E.get('ITERS', '6'))
T, K = E.get('T', '3'), E.get('K', '3')
GAMES = E.get('GAMES', '300')
REPLAY = int(E.get('REPLAY', '2'))
CONFIRM_PAIRS = E.get('CONFIRM_PAIRS', '300')
BAR = float(E.get('BAR', '0.0'))
REMOTE = E.get('REMOTE', '0') == '1'
TRAIN_ARGS = shlex.split(E.get('TRAIN_ARGS', '--epochs 8 --lr 2e-5 --td-w 0.5 --rank-w 0'))


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


def run(cmd, env_extra, log):
    env = dict(E, **env_extra)
    print('$', ' '.join(cmd), ' '.join(f'{k}={v}' for k, v in env_extra.items()), flush=True)
    with open(log, 'a') as fh:
        rc = subprocess.call(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    if rc:
        sys.exit(f'step failed (exit {rc}); see {log}')


def try_remote(st, save, mode, ckpts, n, seed_base, out, env=None):
    """Play a step on GitHub (REMOTE=1). True = the cloud finished it. False =
    the local script must run, and it resumes from whatever landed in `out`
    (both resume by seed). ANY failure falls back to local, so an unattended run
    never stalls on GitHub; two failures in a row turn the cloud off for the
    rest of the run."""
    if not REMOTE or st.get('remote_off'):
        return False
    if os.path.exists(out):
        print(f'[remote] {out} exists from an earlier attempt; finishing it locally', flush=True)
        return False
    try:
        from remote_play import remote_play
        miss = remote_play(mode, ckpts, n, seed_base, out, env=env)
    except Exception as e:                       # incl. RemoteError, network, gh missing
        print(f'[remote] FAILED: {e!r}; running this step locally', flush=True)
        miss = None
    if miss == 0:
        st['remote_fails'] = 0
        save()
        return True
    st['remote_fails'] = st.get('remote_fails', 0) + 1
    if st['remote_fails'] >= 2:
        st['remote_off'] = True
        print('[remote] two failures in a row: local for the rest of this run '
              '(delete "remote_off" from state.json to retry the cloud)', flush=True)
    save()
    if miss:
        print(f'[remote] {miss} game(s) missing; the local run plays them', flush=True)
    return False


def main():
    os.makedirs(D, exist_ok=True)
    sp = os.path.join(D, 'state.json')
    st = json.load(open(sp)) if os.path.exists(sp) else \
        {'champion': os.path.abspath(START), 'iter': 0, 'done': {}, 'history': []}
    save = lambda: json.dump(st, open(sp, 'w'), indent=1)
    pool = E.get('START_POOL', 'start_pool.jsonl' if os.path.exists('start_pool.jsonl') else '')
    lab_env = {'START_POOL': pool, 'START_FRAC': E.get('START_FRAC', '0.3'),
               'MAX_REPLIES': E.get('MAX_REPLIES', '0')}
    while st['iter'] < ITERS:
        k = st['iter']
        done = st['done'].setdefault(str(k), {})
        champ = st['champion']
        print(f'=== iteration {k}: champion {champ}', flush=True)
        lab = os.path.join(D, f'labels_{k}.jsonl.gz')
        if not done.get('label'):
          if not try_remote(st, save, 'label', [champ], GAMES, 91_000_000 + 10_000 * k, lab,
                            env=dict(lab_env, T=T, K=K)):
            run([PY, '-u', 'close_call.py', 'label', champ, T, K, GAMES, lab],
                dict(lab_env, SEED_BASE=str(91_000_000 + 10_000 * k)), os.path.join(D, f'label_{k}.log'))
          done['label'] = True; save()
        cand = os.path.abspath(os.path.join(D, f'cand_{k}.pt'))
        if not done.get('train'):
            files = [os.path.join(D, f'labels_{j}.jsonl.gz') for j in range(max(0, k - REPLAY + 1), k + 1)]
            run([PY, '-u', 'distill_train.py', '--data', *files, '--load', champ, '--save', cand,
                 '--log', os.path.join(D, 'train_log.jsonl'), *TRAIN_ARGS],
                {}, os.path.join(D, f'train_{k}.log'))
            done['train'] = True; save()
        conf = os.path.join(D, f'confirm_{k}.jsonl')
        if not done.get('confirm'):
          if not try_remote(st, save, 'confirm', [cand, champ], CONFIRM_PAIRS,
                            93_000_000 + 10_000 * k, conf):
            run([PY, '-u', 'confirm_match.py', cand, champ, CONFIRM_PAIRS, conf],
                {'SEED_BASE': str(93_000_000 + 10_000 * k)}, os.path.join(D, f'confirm_{k}.log'))
          done['confirm'] = True; save()
        m, se, n = paired(conf)
        promote = m > BAR
        print(f'iteration {k}: cand vs champion {m:+.3f} (95% CI {m - 1.96 * se:+.3f} .. '
              f'{m + 1.96 * se:+.3f}, {n} pairs) -> {"PROMOTE" if promote else "keep champion"}', flush=True)
        st['history'].append({'iter': k, 'cand': cand, 'champion': champ, 'mean': m, 'se': se,
                              'pairs': n, 'promoted': promote})
        if promote:
            st['champion'] = cand
        st['iter'] = k + 1
        save()
    print('DONE', json.dumps(st['history'], indent=1))


if __name__ == '__main__':
    main()
