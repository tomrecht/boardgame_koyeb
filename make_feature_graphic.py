#!/usr/bin/env python3
"""Google Play feature graphic: 1024x500, written to feature-graphic.png.

The board mark is taken from the SHIPPED icon-512.png rather than re-rendered
from make_icons.py -- that guarantees the banner and the store icon are the same
drawing, and cannot drift if the icon script's SHIPPED design is ever switched.
The icon's ground is the same parchment as this canvas (#ece3d3, sampled and
asserted below), so the square icon composites onto it with no visible seam.

Layout is a HORIZONTAL LOCKUP CENTRED in the canvas, deliberately: Play crops
and overlays this graphic differently in different placements, so anything
pushed to an edge can be cut. Everything that matters sits in the middle.

No store badges, prices or fake UI -- all of which Play's asset policy forbids.
"""
from PIL import Image, ImageDraw, ImageFont

W, H = 1024, 500
BG = (236, 227, 211)          # #ece3d3, the icon's own ground
PLUM = (92, 42, 94)           # #5c2a5e
MUTED = (95, 86, 72)

NAME = 'QUAHURU'
TAGLINE = 'A dice game of racing and blocking'

FONT = '/System/Library/Fonts/Avenir Next.ttc'
DEMI, MEDIUM = 2, 5           # face indices within the .ttc


def main():
    icon = Image.open('icon-512.png').convert('RGB')
    assert icon.getpixel((4, 4)) == BG, 'icon ground no longer matches the banner ground'

    canvas = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(canvas)

    # Fit the whole lockup inside a safe width, shrinking it proportionally if
    # the strings above are edited into something longer. Without this the
    # tagline ran to within 17px of the canvas edge, which Play's cropping in
    # some placements would eat.
    SAFE_W = 900
    mark_px, name_px, tag_px, gap = 400, 104, 34, 44

    def measure(scale):
        nf = ImageFont.truetype(FONT, int(name_px * scale), index=DEMI)
        tf = ImageFont.truetype(FONT, int(tag_px * scale), index=MEDIUM)
        tw = max(d.textbbox((0, 0), NAME, font=nf)[2],
                 d.textbbox((0, 0), TAGLINE, font=tf)[2])
        return nf, tf, tw, int(mark_px * scale) + int(gap * scale) + tw

    scale = 1.0
    for _ in range(24):
        name_f, tag_f, text_w, total = measure(scale)
        if total <= SAFE_W:
            break
        scale *= 0.97

    mark = icon.resize((int(mark_px * scale),) * 2, Image.LANCZOS)
    gap = int(gap * scale)
    x = (W - total) // 2

    canvas.paste(mark, (x, (H - mark.height) // 2))

    tx = x + mark.width + gap
    # Optical centring: the two lines plus the space between them, centred as a
    # block against the mark rather than against the canvas.
    name_h = d.textbbox((0, 0), NAME, font=name_f)[3]
    tag_h = d.textbbox((0, 0), TAGLINE, font=tag_f)[3]
    block_h = name_h + 26 + tag_h
    ty = (H - block_h) // 2 - 8
    d.text((tx, ty), NAME, font=name_f, fill=PLUM)
    d.text((tx, ty + name_h + 26), TAGLINE, font=tag_f, fill=MUTED)

    canvas.save('feature-graphic.png')
    print('wrote feature-graphic.png  %dx%d' % canvas.size)


if __name__ == '__main__':
    main()
