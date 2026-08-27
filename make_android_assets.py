#!/usr/bin/env python3
"""Replace Capacitor's default launcher icons and splash screens with Quahuru's.

`npx cap add android` ships the Capacitor logo for both, so without this the app
installs with someone else's branding. Everything here is derived from the
SHIPPED web icons, so the launcher, the store listing and the PWA cannot drift:

  ic_launcher / ic_launcher_round  <- icon-512.png
  ic_launcher_foreground           <- icon-512-maskable.png, which was already
                                      drawn with the extra padding an adaptive
                                      icon's 72/108dp safe zone needs
  splash.png                       <- the board centred on the icon's ground

Re-run after changing the web icons, then `npx cap sync`. NOTE the launcher icon
is baked at install time, so seeing a change on a device needs a remove-and-
re-add, not just a reinstall.
"""
import os
from PIL import Image

RES = 'android/app/src/main/res'
GROUND = (236, 227, 211)          # #ece3d3, the icon's own parchment
DENSITIES = {'mdpi': 1, 'hdpi': 1.5, 'xhdpi': 2, 'xxhdpi': 3, 'xxxhdpi': 4}


def main():
    icon = Image.open('icon-512.png').convert('RGB')
    mask = Image.open('icon-512-maskable.png').convert('RGB')
    assert icon.getpixel((4, 4)) == GROUND, 'icon ground changed; update GROUND'

    n = 0
    for d, k in DENSITIES.items():
        mip = os.path.join(RES, 'mipmap-' + d)
        if not os.path.isdir(mip):
            continue
        # Legacy launcher icons: 48dp. The board is drawn concentric precisely
        # so a round mask cannot clip it, so round reuses the same art.
        px = int(48 * k)
        sq = icon.resize((px, px), Image.LANCZOS)
        sq.save(os.path.join(mip, 'ic_launcher.png'))
        sq.save(os.path.join(mip, 'ic_launcher_round.png'))
        # Adaptive foreground: 108dp, of which only the middle 72dp is safe.
        fpx = int(108 * k)
        mask.resize((fpx, fpx), Image.LANCZOS).save(
            os.path.join(mip, 'ic_launcher_foreground.png'))
        n += 3

    # The adaptive background sits behind the foreground; matching the ground
    # means any bleed at the mask's edge is invisible rather than white.
    with open(os.path.join(RES, 'values', 'ic_launcher_background.xml'), 'w') as fh:
        fh.write('<?xml version="1.0" encoding="utf-8"?>\n<resources>\n'
                 '    <color name="ic_launcher_background">#ECE3D3</color>\n'
                 '</resources>\n')

    # Splash: the board centred on the ground, sized off the SHORT edge so it is
    # the same size in portrait and landscape and never crops.
    for root, _dirs, files in os.walk(RES):
        if 'splash.png' not in files:
            continue
        path = os.path.join(root, 'splash.png')
        w, h = Image.open(path).size
        canvas = Image.new('RGB', (w, h), GROUND)
        side = int(min(w, h) * 0.42)
        art = icon.resize((side, side), Image.LANCZOS)
        canvas.paste(art, ((w - side) // 2, (h - side) // 2))
        canvas.save(path)
        n += 1

    print('rewrote %d Android assets' % n)


if __name__ == '__main__':
    main()
