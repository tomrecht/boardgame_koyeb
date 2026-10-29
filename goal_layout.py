"""goal_layout.py -- train / play on an ALTERNATIVE goal numbering (owner,
2026-10-29). Unset = today's board, and nothing anywhere changes.

    GOAL_LAYOUT=sum7   (1-6-2-5-3-4)      GOAL_LAYOUT=B   (1-4-5-2-3-6)
    GOAL_LAYOUT=1,6,2,5,3,4               (any numbering, ring order)

A layout = goal numbers at the six goal tiles in ring order (ring 7, sectors
2, 4, ..., 12; partners 4 apart are positions (0,1) (2,3) (4,5)). Today's is
4-2-5-3-6-1. Routes are untouched; only which number each goal carries.

Read at import by every place that knows goal numbers:
  game.Board            tile.number on the six goals
  encoder               tile input [6] (goal number / 6)
  features_v2           the endgame table (endgame_table_<name>.json, built by
                        `endgame_table.py --layout`) and the lone-blank bank times
  symmetry              the goal-number permutation of a 240-degree rotation
  league_run / close_call   start-pool positions RELABELLED: a numbered piece
                        keeps its goal TILE (piece n -> the number that tile
                        carries in the layout), as layout_probe.py does
The env var reaches multiprocessing workers and the GitHub shards (distill_run
passes it on), so set it once on the launch command.
"""
import json
import os

TODAY = (4, 2, 5, 3, 6, 1)
GOAL_SECTORS = (2, 4, 6, 8, 10, 12)          # ring 7
NAMED = {'sum7': (1, 6, 2, 5, 3, 4), 'B': (1, 4, 5, 2, 3, 6)}


def _parse(v):
    v = (v or '').strip()
    if not v or v == 'today':
        return None, None
    if v in NAMED:
        return v, NAMED[v]
    nums = tuple(int(x) for x in v.split(','))
    assert sorted(nums) == [1, 2, 3, 4, 5, 6], f'GOAL_LAYOUT {v!r} is not a numbering of 1-6'
    name = next((k for k, L in NAMED.items() if L == nums), 'L' + ''.join(map(str, nums)))
    return name, nums


NAME, NUMBERS = _parse(os.environ.get('GOAL_LAYOUT'))
if NUMBERS == TODAY:
    NAME, NUMBERS = None, None
ACTIVE = NUMBERS is not None


def number_at_sector(sector):
    """Goal number on the ring-7 goal tile at this sector, under the layout."""
    return NUMBERS[GOAL_SECTORS.index(sector)]


def patch_tile_info(raw):
    """tile_neighbors.json dict (key -> entry) with the goals renumbered, in place."""
    if not ACTIVE:
        return raw
    for key, val in raw.items():
        if val.get('type') == 'save':
            sector = int(key.replace('ring', '').replace('sector', '').split('_')[1])
            val['number'] = number_at_sector(sector)
    return raw


def table_path(repo_dir):
    return os.path.join(repo_dir, f'endgame_table_{NAME}.json' if ACTIVE else 'endgame_table.json')


def rotation_pi():
    """Goal-number permutation of one symmetry step (symmetry.PI for today):
    the goal at ring position i moves to position i+4."""
    L = NUMBERS if ACTIVE else TODAY
    return {L[i]: L[(i + 4) % 6] for i in range(6)}


def relabel_state(state):
    """A position recorded on TODAY's board, renumbered for the layout so every
    numbered piece keeps its goal tile. Blanks untouched. Identity if inactive."""
    if not ACTIVE:
        return state
    sig = {TODAY[i]: NUMBERS[i] for i in range(6)}
    s = json.loads(json.dumps(state))
    f = lambda n: sig[n] if n <= 6 else n
    for rack in s.get('racks', {}).values():
        for p in rack:
            p['number'] = f(p['number'])
    for p in s.get('boardPieces', []):
        p['number'] = f(p['number'])
    return s


assert {TODAY[i]: TODAY[(i + 4) % 6] for i in range(6)} == {1: 3, 2: 1, 3: 2, 4: 6, 5: 4, 6: 5}
