"""rollout_run2.py -- round 2 of the rollout-teacher experiment (2026-10-26 night).

Round 1 (rollout_run.py, N=4 playouts per candidate): every student level with
cand_13 and with each other (h4 vs h1 -0.011 +- 0.049); the deep labels were
noise-dominated (within-decision spread 0.047 pts vs SE 0.067), and the h4/h2
students fit held-out labels WORSE than the warm start. So: same decisions, more
playouts. GitHub adds 8 per candidate on a disjoint range (ROLL_I0=8), the local
extra_i4 run adds 4 more where it got to; rollout_label pools them (N 12-16).
Then h4 and h2 students from cand_13, matched vs cand_13 and vs round 1's h1
student (the control: same decisions, the 1-ply teacher).

State rollout/state2.json; rerun to resume. Results rollout/RESULT2.txt.
    caffeinate -is nohup .venv/bin/python -u rollout_run2.py >> rollout/run2.log 2>&1 &
"""
import glob, json, os, subprocess, sys, time

import remote_play as RP
from rollout_run import D, PY, NET, DEC, TRAIN_ARGS, PAIRS, log, sh, paired

GH_BLOCKS = [(0, 1200), (1200, 1200), (2400, 1200), (3600, 1844)]
ENV = {'ROLL_H': '4', 'ROLL_N': '8', 'ROLL_I0': '8'}
ARMS = [4, 2]
ST = os.path.join(D, 'state2.json')


def load():
    return json.load(open(ST)) if os.path.exists(ST) else {}


def save(s):
    json.dump(s, open(ST, 'w'), indent=1)


def match(s, key, a, b, seed):
    out = os.path.join(D, f'confirm2_{key}.jsonl')
    try:
        miss = RP.remote_play('confirm', [a, b], PAIRS, seed, out)
        if not miss:
            return paired(out)
    except RP.RemoteError as e:
        log('remote failed:', e)
    sh([PY, '-u', 'confirm_match.py', a, b, str(PAIRS), out], os.path.join(D, f'confirm2_{key}.log'),
       {'SEED_BASE': str(seed)})
    return paired(out)


def main():
    s = load()
    for start, cnt in GH_BLOCKS:
        k = f'gh8_{start}'
        if s.get(k):
            continue
        out = os.path.join(D, f'extra_gh8_{start}.jsonl')
        try:
            miss = RP.remote_play('relabel', [NET, DEC], cnt, start, out, env=ENV)
            log(f'block {start}: {cnt - miss}/{cnt} extra-labelled on GitHub')
        except RP.RemoteError as e:
            log(f'block {start}: remote failed: {e} (skipped; pooling uses what exists)')
        s[k] = True
        save(s)
    if not s.get('local_stopped'):
        subprocess.call(['pkill', '-f', 'rollout_label.py run .*extra_i4'])
        time.sleep(5)
        s['local_stopped'] = True
        save(s)
    files = sorted(glob.glob(os.path.join(D, 'relabel_*.jsonl')) + glob.glob(os.path.join(D, 'extra_*.jsonl')))
    subprocess.call([PY, 'rollout_label.py', 'stats', *files])
    for h in ARMS:
        k = f'train_h{h}'
        if s.get(k):
            continue
        data = os.path.join(D, f'train2_h{h}.jsonl.gz')
        sh([PY, 'rollout_label.py', 'build', DEC, data, str(h), *files], os.path.join(D, 'build2.log'))
        sh([PY, '-u', 'distill_train.py', '--data', data, '--load', NET,
            '--save', os.path.join(D, f'stud2_h{h}.pt'), '--log', os.path.join(D, 'train2_log.jsonl'),
            *TRAIN_ARGS], os.path.join(D, f'train2_h{h}.log'))
        s[k] = True
        save(s)
    res = s.setdefault('results', {})
    plan = [('h4_vs_champ', os.path.join(D, 'stud2_h4.pt'), NET, 178_000_000),
            ('h4_vs_h1', os.path.join(D, 'stud2_h4.pt'), os.path.join(D, 'stud_h1.pt'), 180_000_000),
            ('h2_vs_champ', os.path.join(D, 'stud2_h2.pt'), NET, 182_000_000)]
    for key, a, b, seed in plan:
        if key in res:
            continue
        m, se, n = match(s, key, a, b, seed)
        res[key] = {'mean': m, 'se': se, 'pairs': n}
        save(s)
        log(f'round2 {key}: {m:+.3f} (95% CI {m - 1.96 * se:+.3f}..{m + 1.96 * se:+.3f}), {n} pairs')
    lines = [f'round2 {k}: {v["mean"]:+.3f} (95% CI {v["mean"] - 1.96 * v["se"]:+.3f}..'
             f'{v["mean"] + 1.96 * v["se"]:+.3f}), {v["pairs"]} pairs' for k, v in res.items()]
    open(os.path.join(D, 'RESULT2.txt'), 'w').write('\n'.join(lines) + '\n')
    log('DONE\n' + '\n'.join(lines))


if __name__ == '__main__':
    main()
