"""Capture-vs-goal turns from weakness_probe.jsonl, broken down by WHICH numbered
pieces: the opponent's numbers the mover could capture, its own numbers it could
put on their goals, and what it did. No net needed -- pure move enumeration."""
import json, collections
import weakness_probe as W

def options(b, us):
    """Every legal pair -> (captured opp numbers, own numbers onto own goal)."""
    out = []
    root = len(b.moves); fm0 = b.firstMove; st = dict(b.game_stages)
    first = [m for m in b.get_valid_moves() if m not in ((0, 0, 0), (1, 1, 1))]
    for m1 in first:
        b.apply_move(m1, switch_turn=False)
        second = [m for m in b.get_valid_moves() if m not in ((0, 0, 0), (1, 1, 1))] or [(0, 0, 0)]
        for m2 in second:
            out.append(effects(b, (m1, m2), us, already=1))
        b.undo_last_move(); b.firstMove = fm0; b.game_stages.update(st)
    return out

def effects(b, pair, us, already=0):
    opp = 'black' if us == 'white' else 'white'
    n0 = len(b.moves) - already
    home0 = {id(p) for p in b.home_tile.pieces} if not already else None
    caps, goals = set(), set()
    applied = []
    for m in pair[already:]:
        if m in ((0, 0, 0), (1, 1, 1)):
            continue
        b.apply_move(m, switch_turn=False); applied.append(m)
    for rec in b.moves[n0:]:
        cp = rec['captured_piece']
        if cp is not None and cp.player == opp and cp.number <= 6:
            caps.add(cp.number)
        p, d = rec['piece'], rec['destination']
        if p.player == us and p.number <= 6 and isinstance(d, tuple):
            t = b.get_tile(*d)
            if t.type == 'save' and t.number == p.number:
                goals.add(p.number)
    for _ in applied:
        b.undo_last_move()
    return frozenset(caps), frozenset(goals)

recs = {}
for f in ['quahuru-games-apvo2h65.jsonl', 'quahuru-games-f7in6olg.jsonl']:
    for l in open(f):
        r = json.loads(l); recs[r['id'][:8]] = r
want = collections.defaultdict(dict)
for l in open('weakness_probe.jsonl'):
    g = json.loads(l)
    for r in g['rows']:
        if r['cvg']:
            want[g['game']][r['turn']] = r
out = []
for gid, turns in want.items():
    for t, b, played, b2 in W.turn_states(recs[gid]):
        if t not in turns:
            continue
        us = b.current_player
        opts = options(b, us)
        cap_opts = sorted({n for c, g in opts if c and not g for n in c})
        goal_opts = sorted({n for c, g in opts if g and not c for n in g})
        pc, pg = effects(b, played, us)
        r = turns[t]
        out.append({'game': gid, 'turn': t, 'who': r['who'], 'cvg': r['cvg'], 'net': r['cvg_net'],
                    'cap_opts': cap_opts, 'goal_opts': goal_opts,
                    'captured': sorted(pc), 'goaled': sorted(pg)})
json.dump(out, open('cvg_breakdown.json', 'w'))
print(len(out), 'turns')
