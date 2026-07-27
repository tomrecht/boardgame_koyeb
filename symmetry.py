"""D3 board automorphism (240-degree rotation) for symmetry augmentation.

The board has a validated D3 symmetry: a 240-degree rotation maps the tile graph
to itself and permutes the goal numbers 2->1->3->2 and 4->6->5->4. This module
builds that rotation as a reusable tile permutation (sigma) + number permutation
(pi), and transforms a serialized board position (raw_state) to its rotated
image -- so training can augment each position to a random one of the three
symmetric variants, balancing the goal pairs (2&4 / 1&6 / 3&5) by construction
and synthesizing the rare 1&6 / 3&5 experience iter10 barely generates.

CAVEAT (validated empirically before use, see validate_symmetry.py): dice have a
DUAL role -- movement distance (rotation-invariant) and save-goal matching
(goal-permuted) -- so the rotation is an exact symmetry of the board GRAPH but
only an approximate symmetry of the full game with dice. We leave dice untouched
and gate on a measured value-preservation check: V(pos) ~= V(rotate(pos)).
"""
import json, math

# goal-number permutation for ONE 240-degree step (2->1->3->2, 4->6->5->4)
PI = {1: 3, 2: 1, 3: 2, 4: 6, 5: 4, 6: 5}


def _tile_angles(numsegs=12, numrings=7):
    """Center angle of every (ring, sector) tile, ported from game.js createTiles
    (incl. the ring-5 half-tiles and outer-ring third-tiles)."""
    TWO = 2 * math.pi
    SEG = TWO / numsegs
    ang = {}

    def add(ring, sec, sa, ea):
        ang[(ring, sec)] = (sa + ea) / 2.0
    add(0, 0, 0, TWO)                       # home
    for r in range(numrings):
        for s in range(numsegs):
            sa, ea = s * SEG, s * SEG + SEG
            if r == numrings - 1:            # outer ring
                if s % 4 == 2:
                    sub = SEG / 3
                    for mt in range(3):
                        ms = sa + mt * sub
                        add(r + 1, (s + 4) * 3 + mt + 1, ms, ms + sub)
                else:
                    add(r + 1, s + 1, sa, ea)   # nogo (even s) or save (odd s)
            else:
                is_nogo = ((r == 0 and s % 4 == 0) or
                           ((r == 1 or r == 4) and (s + 2) % 4 == 0) or
                           ((r == 3 or r == 5) and s % 2 == 0))
                if r == 4 and s % 4 == 0 and not is_nogo:
                    sub = SEG / 2
                    for mt in range(2):
                        ms = sa + mt * sub
                        add(r + 1, (s + 6) * 2 + mt + 1, ms, ms + sub)
                    continue
                add(r + 1, s + 1, sa, ea)
    return ang


def build_sigma(tile_neighbors_path='tile_neighbors.json', degrees=240):
    """Tile permutation sigma: (ring, sector) -> (ring, sector) for a `degrees`
    rotation, matched by center angle. Restricted to REAL tiles (those in
    tile_neighbors.json: home/field/save; nogo tiles never hold pieces)."""
    real = set()
    for k in json.load(open(tile_neighbors_path)):
        r, s = k.replace('ring', '').replace('sector', '').split('_')
        real.add((int(r), int(s)))
    ang = _tile_angles()
    rot = math.radians(degrees)
    TWO = 2 * math.pi
    # index real tiles per ring by angle for matching
    sigma = {}
    for (r, s) in real:
        if r == 0:
            sigma[(r, s)] = (0, 0)           # home is fixed
            continue
        target = (ang[(r, s)] + rot) % TWO
        best, bestd = None, 1e9
        for (r2, s2) in real:
            if r2 != r:
                continue
            d = abs(((ang[(r2, s2)] - target + math.pi) % TWO) - math.pi)
            if d < bestd:
                bestd, best = d, (r2, s2)
        assert bestd < 0.02, f"no match for {(r,s)} -> angle {target} (best {best}, d {bestd})"
        sigma[(r, s)] = best
    return sigma


class Symmetry:
    def __init__(self, tile_neighbors_path='tile_neighbors.json'):
        self.sigma = build_sigma(tile_neighbors_path)          # one 240-deg step
        self.pi = dict(PI)
        # precompute powers 0,1,2
        self._sig = [self._identity_sigma(), self.sigma,
                     self._compose(self.sigma, self.sigma)]
        self._pin = [{n: n for n in range(1, 7)}, self.pi,
                     {n: self.pi[self.pi[n]] for n in range(1, 7)}]

    def _identity_sigma(self):
        return {k: k for k in self.sigma}

    def _compose(self, a, b):               # apply b then a
        return {k: a[b[k]] for k in b}

    def transform(self, raw_state, k):
        """Return raw_state rotated k*240 degrees (k in {0,1,2}). Remaps every
        piece's tile via sigma^k and every numbered (<=6) piece's number via
        pi^k. Dice / currentTurn / used-flags unchanged (see module caveat)."""
        if k % 3 == 0:
            return raw_state
        sig, pin = self._sig[k % 3], self._pin[k % 3]
        rs = json.loads(json.dumps(raw_state))   # deep copy
        for bp in rs['boardPieces']:
            r, s = bp['tile']['ring'], bp['tile']['sector']
            nr, ns = sig[(r, s)]
            bp['tile']['ring'], bp['tile']['sector'] = nr, ns
            if bp['number'] <= 6:
                bp['number'] = pin[bp['number']]
        for key in ('whiteUnentered', 'whiteSaved', 'blackUnentered', 'blackSaved'):
            for pc in rs['racks'][key]:
                if pc['number'] <= 6:
                    pc['number'] = pin[pc['number']]
        return rs
