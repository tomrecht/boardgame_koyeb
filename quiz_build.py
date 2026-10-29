"""Build quiz items for the "Find the Move" artifact from review_games' verified
mistakes. Each confirmed turn (deep gap >= CONFIRM_MIN) becomes one item: the
position, both moves as text and as [from, to] tile pairs (origins read off the
replayed board), and the cost. Items go into chunk files of ~CHUNK items under
review/quiz/; an index (review/quiz/index.json) remembers which chunk holds
which item, so a rerun after new games only adds NEW chunks and never rewrites
an old one -- the page's progress is keyed by item id either way.

    python3 quiz_build.py review/verify_cand13.jsonl [more verify files]

Prints the chunk files written this run (the ones to push to the artifact's db
collection `positions`).
"""
import ast
import datetime
import json
import os
import sys

import review_games as RG

OUT = 'review/quiz'
CHUNK = 250


def _loc(b, t):
    if t is None or t is b.home_tile:
        return 'H'
    if t.type == 'save':
        return 'G%d' % t.number
    return '%d.%d' % (t.ring, t.pos)


def moves(b, pair_repr):
    """[[from, to], ...]; to '' = banked; a block-save is [wall tile, 'B']."""
    out, moved = [], {}
    for m in ast.literal_eval(pair_repr):
        if m in ((0, 0, 0), (1, 1, 1)):
            continue
        pid, dest, _ = m
        p = next((q for q in b.pieces if (q.player, q.number) == tuple(pid)), None)
        if p is None:
            continue
        frm = moved.get(tuple(pid)) or _loc(b, p.tile)
        if dest == 'save':
            to = ''
        elif dest == 0:
            to = 'B'
        else:
            to = RG._where(dest)
            moved[tuple(pid)] = to
        out.append([frm, to])
    return out


def main(paths):
    os.makedirs(OUT, exist_ok=True)
    ipath = os.path.join(OUT, 'index.json')
    index = json.load(open(ipath)) if os.path.exists(ipath) else {'items': {}, 'chunks': []}
    games = {g['id']: g for g in RG.load_games()}
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p)]
    seen, new = set(index['items']), []
    for r in rows:
        iid = '%s_%d' % (r['game'][:8], r['turn'])
        if 'deep_gap' not in r or r['deep_gap'] < RG.CONFIRM_MIN or iid in seen:
            continue
        seen.add(iid)
        g = games.get(r['game'])
        b = RG._state_at(g, r['turn']) if g else None
        if b is None:
            print('skip (no replay)', iid, file=sys.stderr)
            continue
        new.append({'i': iid, 'p': r['pos'], 'b': r['best'], 'o': r['own'],
                    'bm': moves(b, r['best_pair']), 'om': moves(b, r['own_pair']),
                    'g': round(r['deep_gap'], 2), 'st': r['stage'][0],
                    'm': r.get('model', ''), 'd': (g.get('at') or '')[:10]})
        if len(new) % 200 == 0:
            print('built', len(new), flush=True)
    stamp = datetime.date.today().strftime('%Y%m%d')
    written = []
    for k in range(0, len(new), CHUNK):
        cid = 'c%s_%02d' % (stamp, len(index['chunks']))
        part = new[k:k + CHUNK]
        path = os.path.join(OUT, cid + '.json')
        json.dump({'items': part, 'n': len(part)}, open(path, 'w'), separators=(',', ':'))
        index['chunks'].append(cid)
        for it in part:
            index['items'][it['i']] = cid
        written.append(path)
    json.dump(index, open(ipath, 'w'))
    print('new items %d in %d chunks; total %d' % (len(new), len(written), len(index['items'])))
    for w in written:
        print(w, os.path.getsize(w))


if __name__ == '__main__':
    main(sys.argv[1:])
