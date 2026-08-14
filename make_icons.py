"""
make_icons.py — regenerate the app icons from the real board geometry.

The icon is a miniature of the actual board: the same tiles, goal arcs and
palette, with no lines between tiles, a black silhouette around the whole shape
(the pale field is low-contrast against the parchment ground at 192px without
it), a double-size hub, and three black pieces for a bit of game character.

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

INK = (32, 26, 20)
PIECE = (24, 24, 24)


def _hex(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def render(geom, size, pad_frac, pieces=3, seed=7):
    theme = geom['theme']
    bg, field, goal = _hex(theme['bg']), _hex(theme['field']), _hex(theme['goal'])
    hub_fill = _hex(theme['hub'])
    tiles = [t for t in geom['tiles'] if t['type'] not in ('nogo', 'home')]
    outer = max(t['or'] for t in tiles)
    hub_r = geom['homeRadius'] * 2          # deliberately twice the real hub

    S = size * 4                            # supersample, downscaled at the end
    im = Image.new('RGB', (S, S), bg)
    d = ImageDraw.Draw(im)
    pad = S * pad_frac
    scale = (S - 2 * pad) / (2 * outer)
    cx = cy = S / 2

    def pt(r, a):
        return (cx + r * scale * math.cos(a), cy + r * scale * math.sin(a))

    def poly(t):
        a0, a1 = t['s'], t['e']
        steps = max(8, int((a1 - a0) / 0.02))
        return ([pt(t['or'], a0 + (a1 - a0) * i / steps) for i in range(steps + 1)] +
                [pt(t['ir'], a1 - (a1 - a0) * i / steps) for i in range(steps + 1)])

    ordered = sorted(tiles, key=lambda t: t['type'] == 'save')   # goals on top
    for t in ordered:
        d.polygon(poly(t), fill=goal if t['type'] == 'save' else field)

    # Outline the UNION of the tiles: the edges of a solid mask are exactly the
    # silhouette, so no lines appear between neighbouring tiles.
    mask = Image.new('L', (S, S), 0)
    md = ImageDraw.Draw(mask)
    for t in ordered:
        md.polygon(poly(t), fill=255)
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
    render(geom, 512, 0.16).save('icon-512-maskable.png')
    print('wrote icon-192.png, icon-512.png, icon-512-maskable.png')


if __name__ == '__main__':
    main()
