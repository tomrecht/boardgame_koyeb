"""quiz_merge.py -- fold the all-moves re-solve (quiz_best.py) and the playout
check (quiz_playout.py) into the quiz chunks in review/quiz/.

Per item (matched by id):
  re-solve  his move is the best of all  -> skip: 1 (out of play; kept for the record)
            else champion's move/key/arrows/gap from the re-solve (it may be a move the
            app's prefilter had culled), eb = eo + gap, and the outcome table
  playouts  pl = [champion - his, SE] in points over 32 common-dice games, only when
            the champion's move is still the one that was played out

    python3 quiz_merge.py best.jsonl playouts.jsonl     -> rewrites the chunk files
"""
import glob, json, os, sys

import numpy as np

QUIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'review', 'quiz')


def fnv(s):
    """32-bit FNV-1a of an ASCII string, base 36 -- twin of fnv() in the quiz page."""
    h = 0x811c9dc5
    for ch in s:
        h = ((h ^ ord(ch)) * 0x01000193) & 0xffffffff
    return _b36(h)


def _b36(n):
    d = '0123456789abcdefghijklmnopqrstuvwxyz'
    out = ''
    while True:
        n, r = divmod(n, 36)
        out = d[r] + out
        if not n:
            return out


def slim(table, keep=8):
    """Hash the outcome keys; keep every deep-checked entry + the top `keep` others."""
    out, others = [], 0
    for e in table:
        if 'd' not in e:
            if others >= keep:
                continue
            others += 1
        out.append(dict({'h': fnv(e['k']), 'q': round(e['q'], 2)}, **({'d': round(e['d'], 2)} if 'd' in e else {})))
    return out


def main(best_path, play_path):
    best = {}
    for l in open(best_path):
        r = json.loads(l)
        if 'k_best' in r:
            best[r['id']] = r
    play = {}
    for l in open(play_path):
        r = json.loads(l)
        A, B = np.array(r['A'], float), np.array(r['B'], float)
        play[r['id']] = [round(float(B.mean() - A.mean()), 2),
                         round(float(np.std(B - A, ddof=1) / np.sqrt(len(A))), 2)]
    n = dict(items=0, resolved=0, own_best=0, changed=0, played=0)
    for path in sorted(glob.glob(os.path.join(QUIZ, 'c*.json'))):
        doc = json.load(open(path))
        for it in doc['items']:
            n['items'] += 1
            r = best.get(it['i'])
            if r:
                n['resolved'] += 1
                if r['k_best'] == r['k_own']:
                    it['skip'] = 1
                    n['own_best'] += 1
                else:
                    it.pop('skip', None)
                    if r['k_best'] != it['kb']:
                        n['changed'] += 1
                    it.update(b=r['best'], bm=r['bm'], kb=r['k_best'], g=round(r['gap'], 2),
                              eb=round(it['eo'] + r['gap'], 2), table=slim(r['table']))
                    if r['k_best'] != r['k_old']:
                        it.pop('pl', None)
            if it['i'] in play and (not r or r['k_best'] == r['k_old']):
                it['pl'] = play[it['i']]
                n['played'] += 1
        json.dump(doc, open(path, 'w'), separators=(',', ':'))
    print(n)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
