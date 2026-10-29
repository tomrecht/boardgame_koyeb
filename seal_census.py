"""Which goal pair gets SEALED in owner's recorded games, by whom (owner, 2026-10-29:
"champ prefers to seal 2 and 4"). Replays every logged game (deduped by id) and,
after each turn, for each side X: field tiles with 2+ of X's pieces are walls; a
BFS from the home tile through the rest finds which goals the OTHER side can no
longer reach from home. A pair (2&4, 3&5, 1&6 -- partners 4 apart) is sealed when
both its goals are cut off.

    python3 seal_census.py quahuru-games-*.jsonl
"""
import sys, json, glob
from collections import deque, Counter, defaultdict
import replay_games as rg

PAIRS = [(2, 4), (3, 5), (1, 6)]
_orig_fp = rg.fingerprint
LOG = []


def cut_goals(board, sealer):
    walls = {t for t in board.tiles if t.type == 'field'
             and sum(1 for p in t.pieces if p.player == sealer) >= 2}
    seen = {board.home_tile}; q = deque([board.home_tile])
    while q:
        t = q.popleft()
        for n in t.neighbors:
            if n in seen or n.type in ('nogo', 'home') or n in walls:
                continue
            seen.add(n); q.append(n)
    return {t.number for t in board.tiles if t.type == 'save' and t not in seen}


def spy(board, shown=None):
    for sealer in ('white', 'black'):
        cut = cut_goals(board, sealer)
        LOG.append((board.current_player, sealer, frozenset(cut)))
    return _orig_fp(board, shown)


rg.fingerprint = spy


def main(paths):
    games = {}
    for p in paths:
        for line in open(p):
            if line.strip():
                g = json.loads(line); games.setdefault(g['id'], g)
    turns = defaultdict(Counter)    # who -> pair -> sealed turn-ends
    onsets = defaultdict(Counter)   # who -> pair -> times newly sealed
    goal_cut = defaultdict(Counter)
    games_with = defaultdict(Counter)
    n_ok = 0; total_turns = Counter()
    for g in games.values():
        LOG.clear()
        ok, _ = rg.replay(g)
        if not ok and not LOG:
            continue
        n_ok += ok
        who = {'white': 'computer' if g.get('whiteIsAI') else 'owner',
               'black': 'computer' if g.get('blackIsAI') else 'owner'}
        prev = {'white': set(), 'black': set()}
        seen_pairs = defaultdict(set)
        for mover, sealer, cut in LOG:
            w = who[sealer]
            if sealer == mover:
                total_turns[w] += 1
            for gl in cut:
                goal_cut[w][gl] += 1
            for pr in PAIRS:
                k = '%d&%d' % pr
                if set(pr) <= cut:
                    turns[w][k] += 1
                    seen_pairs[w].add(k)
                    if not set(pr) <= prev[sealer]:
                        onsets[w][k] += 1
            prev[sealer] = set(cut)
        for w, s in seen_pairs.items():
            for k in s:
                games_with[w][k] += 1
    print(f'{len(games)} games, {n_ok} replay fully (the rest counted up to their divergence)')
    for w in ('computer', 'owner'):
        print(f'\n{w} as SEALER  (its own turns: {total_turns[w]})')
        for pr in PAIRS:
            k = '%d&%d' % pr
            print(f'  {k}: seals built {onsets[w][k]:4d}   turn-ends sealed {turns[w][k]:5d}   games with one {games_with[w][k]:4d}')
        print('  single goals cut off from home (turn-ends): ' +
              '  '.join(f'G{n} {goal_cut[w][n]}' for n in range(1, 7)))


if __name__ == '__main__':
    main(sys.argv[1:] or glob.glob('quahuru-games-*.jsonl'))
