"""Replay recorded games through game.py, each under its own goal layout
(a game's `layout` field; absent = the original numbering). game.py reads the
layout from GOAL_LAYOUT at import, so each layout replays in its own process.
    python3 ci/replay_by_layout.py <training-branch dir> games.jsonl [...]
Exit 1 if any game diverges, or no game was replayed at all."""
import json, os, subprocess, sys, tempfile

def main(tdir, paths):
    by = {}
    for p in paths:
        for line in open(p):
            if line.strip():
                by.setdefault(json.loads(line).get('layout') or '', []).append(line)
    if not by:
        print('no games to replay'); return 1
    rc = 0
    for lay, lines in sorted(by.items()):
        f = tempfile.NamedTemporaryFile('w', suffix='.jsonl', delete=False)
        f.writelines(l if l.endswith('\n') else l + '\n' for l in lines); f.close()
        env = dict(os.environ)
        env.pop('GOAL_LAYOUT', None)
        if lay and lay != 'original':
            env['GOAL_LAYOUT'] = lay
        print(f'layout {lay or "original"}: {len(lines)} game(s)', flush=True)
        rc |= subprocess.call([sys.executable, 'replay_games.py', f.name], cwd=tdir, env=env)
    return 1 if rc else 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2:]))
