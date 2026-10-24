"""remote_play.py -- run the game-playing steps of a training loop on GitHub
Actions (free on this public repo: ~20 parallel 4-core jobs), keep training local.

    remote_play(mode, ckpts, n, seed_base, out, env)

  mode   'label'   -> close_call.py label <a> T K <games> <out>   (n = games)
         'confirm' -> confirm_match.py <a> <b> <pairs> <out>      (n = pairs)
  ckpts  local checkpoint paths, uploaded to the `cloud-scratch` release under
         a content-hashed name, so a rerun reuses them and nothing lands in git
  out    the local file the shards' outputs are concatenated into (label files
         are gzip with one member per game, so concatenation is valid gzip)
  env    passed to every shard (T, K, MAX_REPLIES, START_POOL, START_FRAC, ...)

The code itself comes from the branch REMOTE_REF (default: the current branch),
which must be pushed: the workflow checks it out. Shards play disjoint seed
blocks, SEED_BASE + shard * per, exactly the seeds the local run would use.
Workflow: .github/workflows/cloud-play.yml (on main, so it can be dispatched).

    python3 remote_play.py smoke      # 2 shards x 2 confirm pairs, end to end
"""
import glob, gzip, hashlib, json, os, re, shutil, subprocess, sys, tempfile, time, zlib

REPO = os.environ.get('REMOTE_REPO', 'tomrecht/boardgame_koyeb')
RELEASE = 'cloud-scratch'
WORKFLOW = 'cloud-play.yml'
SHARDS = int(os.environ.get('REMOTE_SHARDS', '20'))


class RemoteError(Exception):
    """Anything that stops the cloud step; the caller falls back to local."""


def _gh(*args, check=True, retries=1):
    for i in range(retries):
        try:
            r = subprocess.run(['gh', *args, '-R', REPO], capture_output=True, text=True, timeout=600)
        except (OSError, subprocess.TimeoutExpired) as e:          # no gh, or it hung
            r = None; err = repr(e)
        else:
            if not r.returncode or not check:
                return r
            err = r.stderr.strip()
        if i + 1 < retries:
            time.sleep(min(60, 10 * (i + 1)))
    raise RemoteError(f'gh {" ".join(args[:3])} failed: {err}')


def _ref():
    ref = os.environ.get('REMOTE_REF') or subprocess.run(
        ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], capture_output=True, text=True).stdout.strip()
    ahead = subprocess.run(['git', 'rev-list', '--count', f'origin/{ref}..{ref}'],
                           capture_output=True, text=True).stdout.strip()
    if ahead not in ('', '0'):
        raise RemoteError(f'branch {ref} has {ahead} unpushed commit(s); the cloud runs the PUSHED code')
    return ref


def upload(path):
    """Upload a checkpoint to the release (once per content); return its asset name."""
    h = hashlib.sha256(open(path, 'rb').read()).hexdigest()[:12]
    name = f'{h}-{os.path.basename(path)}'
    if _gh('release', 'view', RELEASE, check=False, retries=3).returncode:
        _gh('release', 'create', RELEASE, '--prerelease', '--title', 'cloud scratch',
            '--notes', 'Checkpoints for cloud-play.yml runs (remote_play.py). Safe to delete.')
    have = json.loads(_gh('release', 'view', RELEASE, '--json', 'assets', retries=3).stdout)['assets']
    if name not in {a['name'] for a in have}:
        tmp = os.path.join(tempfile.mkdtemp(), name)
        shutil.copy(path, tmp)
        _gh('release', 'upload', RELEASE, tmp, retries=3)
    return name


def _salvage(mode, path):
    """The complete records in one shard's output. A job killed mid-write leaves
    a truncated last line (confirm) or gzip member (label); a label game counts
    only once its closing {'kind': 'game'} line is there."""
    lines = []
    try:
        fh = gzip.open(path, 'rt') if mode == 'label' else open(path)
        for l in fh:
            lines.append(l)
    except (OSError, EOFError, UnicodeDecodeError, zlib.error):
        pass                                           # keep what was read before the damage
    good = []
    for l in lines:
        try:
            good.append(json.loads(l))
        except ValueError:
            break
    if mode == 'label':
        last = max((i for i, r in enumerate(good) if r.get('kind') == 'game'), default=-1)
        good = good[:last + 1]
    return good


def _missing(mode, recs, n, seed_base):
    if mode == 'label':
        have = {r['seed'] for r in recs if r.get('kind') == 'game'}
        return sum(seed_base + k not in have for k in range(n))
    have = {(r['seed'], r['a_white']) for r in recs}
    return sum((seed_base + k, w) not in have for k in range(n) for w in (True, False))


def remote_play(mode, ckpts, n, seed_base, out, env=None, shards=None):
    """Play on GitHub and write every COMPLETE record to `out` (fresh file).
    Returns the number of games still missing (0 = done). Raises RemoteError if
    the cloud could not be used at all. Either way the caller can run the local
    script on `out`: it resumes by seed and plays only what is missing."""
    shards = shards or SHARDS
    spec = {'mode': mode, 'ckpts': [upload(c) for c in ckpts], 'n': int(n),
            'shards': min(int(shards), int(n)), 'seed_base': int(seed_base), 'env': env or {}}
    ref = _ref()
    t0 = time.time()
    r = _gh('workflow', 'run', WORKFLOW, '--ref', 'main', '-f', f'ref={ref}',
            '-f', 'spec=' + json.dumps(spec), retries=3)
    m = re.search(r'/runs/(\d+)', r.stdout + r.stderr)
    run_id = m.group(1) if m else None
    while run_id is None:                              # older gh prints no URL: find the run
        if time.time() - t0 > 300:
            raise RemoteError('dispatched, but the run never appeared')
        time.sleep(10)
        runs = json.loads(_gh('run', 'list', '-w', WORKFLOW, '-L', '5', '--json',
                              'databaseId,event', retries=3).stdout)
        runs = [x for x in runs if x['event'] == 'workflow_dispatch']
        if runs:
            run_id = str(runs[0]['databaseId'])
    print(f'[remote] {mode} x{spec["n"]} on {spec["shards"]} shards, seeds {seed_base}.., '
          f'run https://github.com/{REPO}/actions/runs/{run_id}', flush=True)
    # Give up (and cancel) if it is still QUEUED after REMOTE_QUEUE_MIN -- GitHub
    # degraded -- or still going after REMOTE_TIMEOUT_MIN in all.
    queue_s = 60 * float(os.environ.get('REMOTE_QUEUE_MIN', '30'))
    total_s = 60 * float(os.environ.get('REMOTE_TIMEOUT_MIN', '400'))
    started = False
    while True:
        try:
            st = json.loads(_gh('run', 'view', run_id, '--json', 'status,conclusion,jobs',
                                retries=5).stdout)
        except RemoteError as e:                      # a long outage: stop waiting
            print(f'[remote] lost contact: {e}', flush=True)
            st = {'status': 'unknown'}
            break
        if st['status'] == 'completed':
            break
        started = started or any(j.get('status') in ('in_progress', 'completed')
                                 for j in st.get('jobs', []) if j.get('name', '').startswith('play'))
        el = time.time() - t0
        if (not started and el > queue_s) or el > total_s:
            print(f'[remote] giving up after {el / 60:.0f} min ({"queued" if not started else "too slow"}); '
                  f'cancelling run {run_id}', flush=True)
            _gh('run', 'cancel', run_id, check=False)
            time.sleep(60)                             # let the cancelled shards upload what they have
            break
        time.sleep(30)
    print(f'[remote] run {run_id} {st.get("conclusion") or st["status"]} after '
          f'{(time.time() - t0) / 60:.1f} min', flush=True)
    d = tempfile.mkdtemp()
    try:
        _gh('run', 'download', run_id, '-p', 'shard-*', '-D', d, check=False, retries=3)
    except RemoteError as e:
        print(f'[remote] could not download: {e}', flush=True)
    recs = []
    for p in glob.glob(os.path.join(d, 'shard-*', '*')):
        recs += _salvage(mode, p)
    if mode == 'label':
        with gzip.open(out, 'wt') as fo:              # fresh: never mixes with an older file
            for r in recs:
                fo.write(json.dumps(r) + '\n')
    else:
        with open(out, 'w') as fo:
            for r in recs:
                fo.write(json.dumps(r) + '\n')
    miss = _missing(mode, recs, spec['n'], seed_base)
    print(f'[remote] {out}: {("complete" if not miss else f"{miss} game(s) missing")}', flush=True)
    return miss


if __name__ == '__main__' and sys.argv[1:] == ['smoke']:
    out = 'remote_smoke.jsonl'
    if os.path.exists(out):
        os.remove(out)
    remote_play('confirm', ['distill_cand3.pt', 'fv2C_it55.pt'], 4, 140_000_000, out, shards=2)
    subprocess.call([sys.executable, 'confirm_match.py', 'analyze', out])
