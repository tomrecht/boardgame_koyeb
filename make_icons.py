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


def _hex(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def render(geom, size, pad_frac, pieces=0, seed=7):
    bg, field, goal, hub_fill = _hex(BG), _hex(FIELD), _hex(GOAL), _hex(HUB)
    tiles = [t for t in geom['tiles'] if t['type'] not in ('nogo', 'home')]
    outer = max(t['or'] for t in tiles)
    hub_r = geom['homeRadius'] * 2          # deliberately twice the real hub

    S = size * 4                            # supersample, downscaled at the end
    im = Image.new('RGB', (S, S), bg)
    d = ImageDraw.Draw(im)
    pad = S * pad_frac

    # The tail sticks out on one side only, so fitting by radius would waste a
    # lot of frame. Fit to the actual drawn bounds instead.
    def dabs_u():
        (p0, p1, p2, p3) = TAIL
        out = []
        for i in range(241):
            u = i / 240
            m = 1 - u
            out.append((m ** 3 * p0[0] + 3 * m * m * u * p1[0] + 3 * m * u * u * p2[0] + u ** 3 * p3[0],
                        m ** 3 * p0[1] + 3 * m * m * u * p1[1] + 3 * m * u * u * p2[1] + u ** 3 * p3[1],
                        86 - 58 * u ** 1.25))       # tapers toward the tip
        return out

    ux, uy = [], []
    for q in range(4):
        ux.append(outer * math.cos(q * math.pi / 2)); uy.append(outer * math.sin(q * math.pi / 2))
    for x, y, w in dabs_u():
        ux += [x - w, x + w]; uy += [y - w, y + w]
    x0, x1, y0, y1 = min(ux), max(ux), min(uy), max(uy)
    scale = (S - 2 * pad) / max(x1 - x0, y1 - y0)
    cx = S / 2 - (x0 + x1) / 2 * scale
    cy = S / 2 - (y0 + y1) / 2 * scale

    def pt(r, a):
        return (cx + r * scale * math.cos(a), cy + r * scale * math.sin(a))

    def tail_dabs():
        return [(cx + x * scale, cy + y * scale, w * scale) for x, y, w in dabs_u()]

    def is_goal4(t):
        return t['type'] == 'save' and abs((t['s'] + t['e']) / 2 - TAIL_MID) < 0.05

    def poly(t):
        a0, a1 = t['s'], t['e']
        steps = max(8, int((a1 - a0) / 0.02))
        return ([pt(t['or'], a0 + (a1 - a0) * i / steps) for i in range(steps + 1)] +
                [pt(t['ir'], a1 - (a1 - a0) * i / steps) for i in range(steps + 1)])

    ordered = sorted([t for t in tiles if not is_goal4(t)],
                     key=lambda t: t['type'] == 'save')          # goals on top
    for t in ordered:
        d.polygon(poly(t), fill=goal if t['type'] == 'save' else field)
    for x, y, w in tail_dabs():
        d.ellipse([x - w, y - w, x + w, y + w], fill=goal)

    # Outline the UNION of the tiles: the edges of a solid mask are exactly the
    # silhouette, so no lines appear between neighbouring tiles.
    mask = Image.new('L', (S, S), 0)
    md = ImageDraw.Draw(mask)
    for t in ordered:
        md.polygon(poly(t), fill=255)
    for x, y, w in tail_dabs():
        md.ellipse([x - w, y - w, x + w, y + w], fill=255)
    edge = mask.filter(ImageFilter.FIND_EDGES)
    k = max(3, int(S / 300)) | 1
    edge = edge.filter(ImageFilter.MaxFilter(k)).point(lambda v: 255 if v > 40 else 0)
    im.paste(Image.new('RGB', (S, S), INK), (0, 0), edge)

    if pieces:
        rnd = random.Random(seed)
        outfield = [t for t in tiles if t['type'] == 'field' and t['ir'] >= 210]
        pr = 30 * scale
        for t in rnd.sample(outfield, pieces):
            x, y = pt((t['ir'] + t['or']) / 2, (t['s'] + t['e']) / 2)
            d.ellipse([x - pr, y - pr, x + pr, y + pr], fill=PIECE)

    hr = hub_r * scale                       # hub last: covers the edges beneath it
    d.ellipse([cx - hr, cy - hr, cx + hr, cy + hr], fill=hub_fill,
              outline=INK, width=max(2, int(S / 110)))
    return im.resize((size, size), Image.LANCZOS)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'board_geom.json'
    geom = json.load(open(path))
    render(geom, 192, 0.05).save('icon-192.png')
    render(geom, 512, 0.05).save('icon-512.png')
    # Maskable at BOTH launcher densities. Android crops a maskable icon to its
    # own shape (circle, squircle, ...), so these carry extra padding -- the
    # drawn board lands inside the central 80% "safe zone" and the parchment
    # runs to all four edges. The background must stay OPAQUE: a transparent
    # one is composited onto the system's own grey, which is the grey bubble it
    # would appear to be there to remove.
    render(geom, 192, 0.16).save('icon-192-maskable.png')
    render(geom, 512, 0.16).save('icon-512-maskable.png')
    print('wrote icon-192.png, icon-512.png, icon-192-maskable.png, icon-512-maskable.png')


if __name__ == '__main__':
    main()
