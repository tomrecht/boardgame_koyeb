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
import glob, hashlib, json, os, re, shutil, subprocess, sys, tempfile, time

REPO = os.environ.get('REMOTE_REPO', 'tomrecht/boardgame_koyeb')
RELEASE = 'cloud-scratch'
WORKFLOW = 'cloud-play.yml'
SHARDS = int(os.environ.get('REMOTE_SHARDS', '20'))


def _gh(*args, check=True, capture=True):
    r = subprocess.run(['gh', *args, '-R', REPO], capture_output=capture, text=True)
    if check and r.returncode:
        sys.exit(f'gh {" ".join(args)} failed: {r.stderr.strip()}')
    return r


def _ref():
    ref = os.environ.get('REMOTE_REF') or subprocess.run(
        ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], capture_output=True, text=True).stdout.strip()
    ahead = subprocess.run(['git', 'rev-list', '--count', f'origin/{ref}..{ref}'],
                           capture_output=True, text=True).stdout.strip()
    if ahead not in ('', '0'):
        sys.exit(f'branch {ref} has {ahead} unpushed commit(s); the cloud runs the PUSHED code')
    return ref


def upload(path):
    """Upload a checkpoint to the release (once per content); return its asset name."""
    h = hashlib.sha256(open(path, 'rb').read()).hexdigest()[:12]
    name = f'{h}-{os.path.basename(path)}'
    if _gh('release', 'view', RELEASE, check=False).returncode:
        _gh('release', 'create', RELEASE, '--prerelease', '--title', 'cloud scratch',
            '--notes', 'Checkpoints for cloud-play.yml runs (remote_play.py). Safe to delete.')
    have = json.loads(_gh('release', 'view', RELEASE, '--json', 'assets').stdout)['assets']
    if name not in {a['name'] for a in have}:
        tmp = os.path.join(tempfile.mkdtemp(), name)
        shutil.copy(path, tmp)
        _gh('release', 'upload', RELEASE, tmp)
    return name


def remote_play(mode, ckpts, n, seed_base, out, env=None, shards=None):
    shards = shards or SHARDS
    spec = {'mode': mode, 'ckpts': [upload(c) for c in ckpts], 'n': int(n),
            'shards': min(int(shards), int(n)), 'seed_base': int(seed_base), 'env': env or {}}
    ref = _ref()
    t0 = time.time()
    r = _gh('workflow', 'run', WORKFLOW, '--ref', 'main', '-f', f'ref={ref}', '-f', 'spec=' + json.dumps(spec))
    m = re.search(r'/runs/(\d+)', r.stdout + r.stderr)
    run_id = m.group(1) if m else None
    while run_id is None:                              # older gh prints no URL: find the run
        time.sleep(5)
        runs = json.loads(_gh('run', 'list', '-w', WORKFLOW, '-L', '5', '--json',
                              'databaseId,createdAt,event').stdout)
        runs = [x for x in runs if x['event'] == 'workflow_dispatch']
        if runs:
            run_id = str(runs[0]['databaseId'])
    print(f'[remote] {mode} x{spec["n"]} on {spec["shards"]} shards, seeds {seed_base}.., '
          f'run https://github.com/{REPO}/actions/runs/{run_id}', flush=True)
    while True:
        st = json.loads(_gh('run', 'view', run_id, '--json', 'status,conclusion').stdout)
        if st['status'] == 'completed':
            break
        time.sleep(30)
    print(f'[remote] run {run_id} {st["conclusion"]} after {(time.time() - t0) / 60:.1f} min', flush=True)
    d = tempfile.mkdtemp()
    _gh('run', 'download', run_id, '-p', 'shard-*', '-D', d, check=False)
    parts = sorted(glob.glob(os.path.join(d, 'shard-*', '*')),
                   key=lambda p: int(re.search(r'shard-(\d+)', p).group(1)))
    got = len({re.search(r'shard-(\d+)', p).group(1) for p in parts})
    with open(out, 'wb') as fo:                      # fresh: a failed step is redone whole
        for p in parts:
            fo.write(open(p, 'rb').read())
    print(f'[remote] {got}/{spec["shards"]} shards returned output -> {out}', flush=True)
    if got < spec['shards']:
        sys.exit(f'{spec["shards"] - got} shard(s) produced nothing; see the run log, then rerun')
    return out


if __name__ == '__main__' and sys.argv[1:] == ['smoke']:
    out = 'remote_smoke.jsonl'
    if os.path.exists(out):
        os.remove(out)
    remote_play('confirm', ['distill_cand3.pt', 'fv2C_it55.pt'], 4, 140_000_000, out, shards=2)
    subprocess.call([sys.executable, 'confirm_match.py', 'analyze', out])
