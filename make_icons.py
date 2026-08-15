"""
make_icons.py — regenerate the app icons from the real board geometry.

The icon is the board drawn as a Q: goal 4 (30-60 degrees, exactly where a Q's
tail belongs) is replaced by a curling tail in the same colour, so the shape
reads as a Q without redrawing anything else. Same tiles and goal arcs as the
real board, no lines between tiles, a silhouette around the whole shape, and a
double-size hub.

Palette is the Plum Night theme (game.js THEMES.plum), so the icon matches a
theme the game actually has. Pieces are deliberately NOT drawn: at 96px they
read as specks, and on the dark ground they are nearly invisible.

Geometry comes from `board_geom.json`, dumped from the running board so the icon
cannot drift from the real thing:

    node scratchpad/geom.js          # writes board_geom.json (see CLAUDE.md)
    python make_icons.py board_geom.json

Writes icon-192.png, icon-512.png and icon-512-maskable.png (extra padding so a
circular OS mask cannot clip the board).
"""

import json
import math
import random
import sys

from PIL import Image, ImageDraw, ImageFilter

INK = (32, 26, 20)           # outline
PIECE = (24, 24, 24)

# Deep plum on the parchment ground (owner's pick from icon_options.png): the
# plum reads as the game's colour without the icon going dark, and the gold hub
# gives it a focal point at 96px where teal was too close in value.
BG, FIELD, GOAL, HUB = '#ece3d3', '#fffdf8', '#5c2a5e', '#d9a441'
TAIL_MID = math.radians(45)                      # goal 4's mid-angle
TAIL = ((300, 380), (495, 668), (742, 560), (652, 322))    # cubic Bezier, board units
# Curl variants. All start at the same point on the rim; they differ in how far
# the tail reaches before hooking back, and how tightly it hooks. A tighter curl
# buys board size without the tail LOOKING smaller, which a uniform tail_scale
# cannot do.
TAIL_SHAPES = {
    'current': TAIL,
    'tight':   ((300, 380), (470, 610), (660, 505), (566, 300)),
    'tighter': ((300, 380), (452, 566), (606, 470), (516, 296)),
    'coiled':  ((300, 380), (455, 560), (600, 430), (470, 330)),
}


def _hex(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def render(geom, size, pad_frac, pieces=0, seed=7, board_frac=None, ring=False,
           tail_scale=1.0, concentric=False, goal_extend=0.0, tail=None,
           drop_outer_field=False, no_tail=False, piece_tiles=None):
    """board_frac: the board's DIAMETER as a fraction of the padded box. The
    original fit sized the whole drawing -- board plus tail -- to that box, which
    made the board smaller than it needed to be and left a crescent of empty
    ground opposite the tail (owner: "a good bit of space at the 10 o'clock
    part"). Sizing the BOARD instead and letting the tail run into the
    bottom-right CORNER uses that space: a corner is sqrt(2) further from the
    centre than an edge, and the tail happens to point at one. The tail's SHAPE
    is untouched -- it scales with the board, as it must to stay attached.

    ring: colour the whole perimeter, not just the six goal wedges, so the Q's
    bowl is one unbroken band.

    tail_scale: shrinks the tail about the point where it meets the rim, so the
    bowl can grow into the room it gives up. The tail is what limits the board
    in a CONCENTRIC layout, since it reaches further from the centre than the
    rim does.

    goal_extend: widen each goal wedge by this fraction of its own angular width
    on EACH side. A middle road between six wedges and a solid band: the rim
    reads as more continuous without giving up the six goals the board actually
    has. At the point where the extensions meet, it becomes the band.

    drop_outer_field: drop the nine field tiles in the outermost field ring
    (ir 450, alongside the goals at ir 450-540). Once the wedges are extended
    those tiles sit in the gaps between them and read as confusing fragments
    rather than as board detail.

    no_tail: drop the curl entirely and draw all SIX goals. Without it the board
    is no longer limited by the tail's diagonal reach, so it fills the frame --
    which is most of the point of trying it.

    piece_tiles: [(ring, mid-angle degrees), ...] to place pieces deliberately
    rather than by `seed`. Rings are 1 (inner) to 6 (outer field); goals are
    ring 7 at 45 105 165 225 285 345 degrees (goals 4 2 5 3 6 1). A piece on
    ring 6 touches its wedge and one on ring 1 touches the hub.

    tail: an alternative cubic Bezier (four control points, board units) for the
    curl. TAIL_SHAPES holds the named ones.

    concentric: centre the BOARD on the canvas, rather than centring the
    board-plus-tail. An Android icon is always cropped to a shape centred on the
    canvas, so a board that is not concentric with it sits visibly off-centre
    inside the bubble -- which is only worth fixing this way if the bubble is
    there to stay."""
    bg, field, goal, hub_fill = _hex(BG), _hex(FIELD), _hex(GOAL), _hex(HUB)
    tiles = [t for t in geom['tiles'] if t['type'] not in ('nogo', 'home')]
    if drop_outer_field:
        rim = min(t['ir'] for t in tiles if t['type'] == 'save')
        tiles = [t for t in tiles if t['type'] == 'save' or t['ir'] < rim - 1]
    outer = max(t['or'] for t in tiles)
    hub_r = geom['homeRadius'] * 2          # deliberately twice the real hub

    S = size * 4                            # supersample, downscaled at the end
    im = Image.new('RGB', (S, S), bg)
    d = ImageDraw.Draw(im)
    pad = S * pad_frac

    # The tail sticks out on one side only, so fitting by radius would waste a
    # lot of frame. Fit to the actual drawn bounds instead.
    def dabs_u():
        (p0, p1, p2, p3) = tail or TAIL
        out = []
        for i in range(241):
            u = i / 240
            m = 1 - u
            out.append((m ** 3 * p0[0] + 3 * m * m * u * p1[0] + 3 * m * u * u * p2[0] + u ** 3 * p3[0],
                        m ** 3 * p0[1] + 3 * m * m * u * p1[1] + 3 * m * u * u * p2[1] + u ** 3 * p3[1],
                        86 - 58 * u ** 1.25))       # tapers toward the tip
        if tail_scale != 1.0:
            # About the ATTACHMENT point, so the tail stays welded to the rim
            # and only its reach and thickness change.
            ax, ay = (tail or TAIL)[0]
            out = [(ax + (x - ax) * tail_scale, ay + (y - ay) * tail_scale, w * tail_scale)
                   for x, y, w in out]
        return out

    if concentric:
        # Board on the canvas centre; scale so the furthest drawn point -- the
        # tail tip, always -- lands on the padded circle.
        reach = outer if no_tail else max([outer] + [math.hypot(x, y) + w
                                                     for x, y, w in dabs_u()])
        scale = (S / 2 - pad) / reach
        cx = cy = S / 2
    elif board_frac is None:
        # Historical fit: the whole drawing, tail included, inside the padded box.
        ux, uy = [], []
        for q in range(4):
            ux.append(outer * math.cos(q * math.pi / 2)); uy.append(outer * math.sin(q * math.pi / 2))
        for x, y, w in dabs_u():
            ux += [x - w, x + w]; uy += [y - w, y + w]
        x0, x1, y0, y1 = min(ux), max(ux), min(uy), max(uy)
        scale = (S - 2 * pad) / max(x1 - x0, y1 - y0)
        cx = S / 2 - (x0 + x1) / 2 * scale
        cy = S / 2 - (y0 + y1) / 2 * scale
    else:
        # Size the BOARD, centre it, then slide it just far enough up-left that
        # the tail still clears the padding -- into the space the tail's own
        # side does not use.
        scale = (S - 2 * pad) * board_frac / (2 * outer)
        cx = cy = S / 2
        tx1 = max(x + w for x, y, w in dabs_u()) * scale + cx
        ty1 = max(y + w for x, y, w in dabs_u()) * scale + cy
        lim = S - pad
        cx -= max(0.0, tx1 - lim)
        cy -= max(0.0, ty1 - lim)

    def pt(r, a):
        return (cx + r * scale * math.cos(a), cy + r * scale * math.sin(a))

    def tail_dabs():
        return [] if no_tail else [(cx + x * scale, cy + y * scale, w * scale)
                                   for x, y, w in dabs_u()]

    def is_goal4(t):
        return t['type'] == 'save' and abs((t['s'] + t['e']) / 2 - TAIL_MID) < 0.05

    def poly(t):
        a0, a1 = t['s'], t['e']
        steps = max(8, int((a1 - a0) / 0.02))
        return ([pt(t['or'], a0 + (a1 - a0) * i / steps) for i in range(steps + 1)] +
                [pt(t['ir'], a1 - (a1 - a0) * i / steps) for i in range(steps + 1)])

    ordered = sorted([t for t in tiles if no_tail or not is_goal4(t)],
                     key=lambda t: t['type'] == 'save')          # goals on top
    def sector(a0, a1, r0, r1):
        steps = max(8, int((a1 - a0) / 0.02))
        return ([pt(r1, a0 + (a1 - a0) * i / steps) for i in range(steps + 1)] +
                [pt(r0, a1 - (a1 - a0) * i / steps) for i in range(steps + 1)])

    def wide_goal(t):
        span = t['e'] - t['s']
        return sector(t['s'] - span * goal_extend, t['e'] + span * goal_extend,
                      t['ir'], t['or'])

    for t in ordered:
        if t['type'] == 'save' and goal_extend:
            d.polygon(wide_goal(t), fill=goal)
        else:
            d.polygon(poly(t), fill=goal if t['type'] == 'save' else field)

    # `ring`: a continuous band at the goals' own radii, all the way round, so
    # the bowl of the Q has no breaks. It cannot be done by recolouring tiles --
    # the goal wedges stick out PAST the field ring, so there is no tile at
    # those radii between them; the band has to be drawn.
    goals = [t for t in tiles if t['type'] == 'save']
    gir, gor = min(t['ir'] for t in goals), max(t['or'] for t in goals)

    def annulus():
        steps = 240
        outer_pts = [pt(gor, 2 * math.pi * i / steps) for i in range(steps + 1)]
        inner_pts = [pt(gir, 2 * math.pi * (steps - i) / steps) for i in range(steps + 1)]
        return outer_pts + inner_pts

    if ring:
        d.polygon(annulus(), fill=goal)
    for x, y, w in tail_dabs():
        d.ellipse([x - w, y - w, x + w, y + w], fill=goal)

    # Outline the UNION of the tiles: the edges of a solid mask are exactly the
    # silhouette, so no lines appear between neighbouring tiles.
    mask = Image.new('L', (S, S), 0)
    md = ImageDraw.Draw(mask)
    for t in ordered:
        md.polygon(wide_goal(t) if (t['type'] == 'save' and goal_extend) else poly(t), fill=255)
    if ring:
        md.polygon(annulus(), fill=255)
    for x, y, w in tail_dabs():
        md.ellipse([x - w, y - w, x + w, y + w], fill=255)
    edge = mask.filter(ImageFilter.FIND_EDGES)
    k = max(3, int(S / 300)) | 1
    edge = edge.filter(ImageFilter.MaxFilter(k)).point(lambda v: 255 if v > 40 else 0)
    im.paste(Image.new('RGB', (S, S), INK), (0, 0), edge)

    if pieces or piece_tiles:
        pr = 30 * scale
        if piece_tiles:
            rings = sorted({round(t['ir']) for t in tiles if t['type'] == 'field'})
            chosen = []
            for ring, deg in piece_tiles:
                ir = rings[ring - 1]
                hit = min((t for t in tiles
                           if t['type'] == 'field' and round(t['ir']) == ir),
                          key=lambda t: abs(math.degrees((t['s'] + t['e']) / 2) - deg))
                chosen.append(hit)
        else:
            rnd = random.Random(seed)
            outfield = [t for t in tiles if t['type'] == 'field' and t['ir'] >= 210]
            chosen = rnd.sample(outfield, pieces)
        for t in chosen:
            x, y = pt((t['ir'] + t['or']) / 2, (t['s'] + t['e']) / 2)
            d.ellipse([x - pr, y - pr, x + pr, y + pr], fill=PIECE)

    hr = hub_r * scale                       # hub last: covers the edges beneath it
    d.ellipse([cx - hr, cy - hr, cx + hr, cy + hr], fill=hub_fill,
              outline=INK, width=max(2, int(S / 110)))
    return im.resize((size, size), Image.LANCZOS)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'board_geom.json'
    geom = json.load(open(path))

    # The shipped design (owner's picks, 2026-08-15):
    #   concentric      -- an Android mask is centred on the canvas, so the board
    #                      must be too, or it sits visibly off-centre in the
    #                      circle. The bubble cannot be removed (see below), so
    #                      the board is made concentric with it instead.
    #   goal_extend .25 -- widens each wedge by a quarter of its own width on
    #                      each side: enough to read as a Q's bowl, not so much
    #                      that it becomes a solid band. Six wedges is the point.
    #   tighter curl    -- hooks back sooner, which buys board size (59% -> 69%
    #                      of the canvas) WITHOUT the tail looking shrunken, as
    #                      simply scaling it down did.
    #   drop_outer_field-- the nine field tiles at ir 450 sit between the
    #                      extended wedges and read as fragments.
    #   3 pieces        -- a touch of life at 512. They are 3px specks at 96 and
    #                      the icon reads fine without them, so this is the one
    #                      choice here that is taste rather than geometry.
    design = dict(concentric=True, goal_extend=0.25, drop_outer_field=True,
                  tail=TAIL_SHAPES['tighter'], pieces=3, seed=5)

    render(geom, 192, 0.03, **design).save('icon-192.png')
    render(geom, 512, 0.03, **design).save('icon-512.png')
    # The maskable pads more: with concentric fitting, the tail tip lands ON the
    # padded circle, and 0.03 would put it at 47% of the canvas -- inside a
    # circular mask (50%) but with nothing to spare on a launcher that crops
    # tighter. 0.08 brings it to 42%.
    render(geom, 192, 0.08, **design).save('icon-192-maskable.png')
    render(geom, 512, 0.08, **design).save('icon-512-maskable.png')
    print('wrote icon-192.png, icon-512.png, icon-192-maskable.png, icon-512-maskable.png')


if __name__ == '__main__':
    main()
