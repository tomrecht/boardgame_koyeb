"""board_png.py -- draw a position (and a move as arrows) to PNG with PIL alone,
for reports made where no browser is available (the playout-request job).

Same look as the Find the Move quiz: tile geometry from board_geo.json
(extracted from game.js), pieces laid out per tile, a ring on each moving piece,
an arrow to its destination and a copy of the piece on the arrow.

    render(board, moves, colour) -> PNG bytes
      board  a game.Board
      moves  [[from, to, label], ...] (quiz_build.moves format: to '' = bank,
             'B' = block-save; label = number or 'x')
"""
import io, json, math, os

from PIL import Image, ImageDraw, ImageFont

GEO = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'board_geo.json')))
CX, CY = GEO['cx'], GEO['cy']
TILE = {t['loc']: t for t in GEO['tiles']}
X0, Y0, W = 290, -10, 1220          # the quiz's viewBox
SCALE = 0.62
SS = 2                              # supersample, then shrink: smooth edges

C = dict(ground='#e9e1d1', tile='#fdfaf3', line='#6b5a48', goal='#c4643a', home='#2c9e8f', ink='#28313b',
         wpiece='#f7f8fa', wline='#3a414b', bpiece='#16181b', btext='#ffffff')


def _font(px):
    for f in ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
              '/System/Library/Fonts/Supplemental/Arial Bold.ttf'):
        if os.path.exists(f):
            return ImageFont.truetype(f, int(px))
    return ImageFont.load_default()


def _xy(x, y):
    k = SCALE * SS
    return ((x - X0) * k, (y - Y0) * k)


def _pt(r, a):
    return CX + r * math.cos(a), CY + r * math.sin(a)


def _centre(t):
    return (CX, CY) if t['type'] == 'home' else _pt((t['r0'] + t['r1']) / 2, (t['a0'] + t['a1']) / 2)


def _sector(t, n=24):
    a0, a1, r0, r1 = t['a0'], t['a1'], t['r0'], t['r1']
    outer = [_pt(r1, a0 + (a1 - a0) * i / n) for i in range(n + 1)]
    inner = [_pt(r0, a1 - (a1 - a0) * i / n) for i in range(n + 1)]
    return [_xy(*p) for p in outer + inner]


def _layout(t, k):
    """Same as the quiz's layout(): positions (x, y, r) for k pieces in tile t."""
    if t['type'] == 'home':
        pr = 19 if k > 6 else 25
        return [(CX + (45 if k > 1 else 0) * math.cos(-math.pi / 2 + i * 2 * math.pi / k),
                 CY + (45 if k > 1 else 0) * math.sin(-math.pi / 2 + i * 2 * math.pi / k), pr) for i in range(k)]
    D, rm = t['r1'] - t['r0'], (t['r0'] + t['r1']) / 2
    L = rm * (t['a1'] - t['a0'])
    best = None
    for rows in (1, 2, 3):
        per = math.ceil(k / rows)
        pr = min(30 if t['type'] == 'save' else 25, (D - 4) / (2 * rows) - 0.5, (L - 4) / per / 2 - 0.5)
        if not best or pr > best[2]:
            best = (rows, per, pr)
    rows, per, pr = best
    out, idx = [], 0
    for r in range(rows):
        if idx >= k:
            break
        n = min(per, k - idx)
        rr = rm if rows == 1 else t['r0'] + 3 + (D - 6) * (r + 0.5) / rows
        step, am = 2 * pr + 2, (t['a0'] + t['a1']) / 2
        span = (n - 1) * step / rr
        for j in range(n):
            out.append((*_pt(rr, am - span / 2 + j * step / rr), pr))
            idx += 1
    return out


def _loc(b, t):
    if t is b.home_tile:
        return 'H'
    if t.type == 'save':
        return 'G%d' % t.number
    return '%d.%d' % (t.ring, t.pos)


def render(board, moves=(), colour='#2b8a48'):
    size = int(W * SCALE * SS)
    img = Image.new('RGB', (size, size), C['ground'])
    d = ImageDraw.Draw(img)
    lw = max(1, int(1.3 * SCALE * SS))
    for t in GEO['tiles']:
        if t['type'] == 'home':
            (x0, y0), (x1, y1) = _xy(CX - t['r1'], CY - t['r1']), _xy(CX + t['r1'], CY + t['r1'])
            d.ellipse([x0, y0, x1, y1], fill=C['home'], outline=C['line'], width=lw)
            continue
        d.polygon(_sector(t), fill=C['goal'] if t['type'] == 'save' else C['tile'], outline=C['line'], width=lw)
        if t['type'] == 'save':
            x, y = _xy(*_pt(t['r1'] + 32, (t['a0'] + t['a1']) / 2))
            d.text((x, y), str(t['number']), fill=C['ink'], font=_font(38 * SCALE * SS), anchor='mm')
    by = {}
    for p in board.pieces:
        if p.tile is not None:
            side = 'W' if p.player == 'white' else 'B'
            by.setdefault(_loc(board, p.tile), []).append((side, str(p.number) if p.number <= 6 else 'x'))
    coords = {}
    for loc, ps in by.items():
        t = TILE.get(loc)
        if not t:
            continue
        ps.sort(key=lambda s: s[0] + s[1])
        for (side, n), (x, y, r) in zip(ps, _layout(t, len(ps))):
            coords.setdefault(loc, []).append([side, n, x, y, r, False])
            _piece(d, x, y, r, side, n)
    mover = 'W' if board.current_player == 'white' else 'B'
    entries = 0
    for mv in moves:
        frm, to, lab = (list(mv) + [None])[:3]
        tf = TILE.get(frm)
        if not tf:
            continue
        who = ('B' if mover == 'W' else 'W') if to == 'B' else mover
        pc = next((c for c in coords.get(frm, []) if c[0] == who and c[1] == lab and not c[5]), None) if lab else None
        if pc:
            pc[5] = True
        from_rack = frm == 'H' and not pc and to not in ('', 'B')
        ax, ay = (pc[2], pc[3]) if pc else _centre(tf)
        rr = pc[4] + 8 if pc else (26 if from_rack else 34)
        if not from_rack:
            _ring(d, ax, ay, rr, colour)
        if to in ('', 'B'):
            x, y = _xy(ax, ay - rr - 16)
            d.text((x, y), 'bank' if to == '' else 'block-save', fill=colour, font=_font(30 * SCALE * SS),
                   anchor='mm', stroke_width=int(3 * SS), stroke_fill=C['tile'])
            continue
        tt = TILE.get(to)
        if not tt:
            continue
        bx, by_ = _centre(tt)
        L = math.hypot(bx - ax, by_ - ay) or 1
        ux, uy = (bx - ax) / L, (by_ - ay) / L
        x0, y0 = ax + ux * rr, ay + uy * rr
        x1, y1 = bx - ux * 22, by_ - uy * 22
        hx, hy = x1 - ux * 26, y1 - uy * 26
        d.line([_xy(x0, y0), _xy(hx, hy)], fill=C['tile'], width=int(14 * SCALE * SS))
        d.line([_xy(x0, y0), _xy(hx, hy)], fill=colour, width=int(8 * SCALE * SS))
        d.polygon([_xy(x1, y1), _xy(hx - uy * 16, hy + ux * 16), _xy(hx + uy * 16, hy - ux * 16)], fill=colour)
        if lab:
            k = entries if from_rack else 0
            entries += from_rack
            off = 0 if k == 0 else (1 if k % 2 else -1) * math.ceil(k / 2) * 52
            mx = x0 + ux * 34 - uy * off if from_rack else (x0 + hx) / 2
            my = y0 + uy * 34 + ux * off if from_rack else (y0 + hy) / 2
            _piece(d, mx, my, 25, who, lab, edge=colour)
    img = img.resize((size // SS, size // SS), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, 'PNG', optimize=True)
    return buf.getvalue()


def _piece(d, x, y, r, side, n, edge=None):
    (cx, cy), k = _xy(x, y), SCALE * SS
    w = side == 'W'
    d.ellipse([cx - r * k, cy - r * k, cx + r * k, cy + r * k], fill=C['wpiece'] if w else C['bpiece'],
              outline=edge or (C['wline'] if w else C['bpiece']), width=max(1, int((6 if edge else 3) * k)))
    if n != 'x':
        d.text((cx, cy), n, fill='#1d232b' if w else C['btext'], font=_font(r * 1.15 * k), anchor='mm')


def _ring(d, x, y, r, colour):
    (cx, cy), k = _xy(x, y), SCALE * SS
    d.ellipse([cx - r * k, cy - r * k, cx + r * k, cy + r * k], outline=colour, width=int(7 * k))
