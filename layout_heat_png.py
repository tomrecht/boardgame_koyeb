import json, math, statistics as st, sys
sys.path.insert(0, '/Users/tom/Game/BoardGame')
from PIL import Image, ImageDraw
import board_png as bp
from layout_probe import LAYOUTS, GOAL_SECTORS, TODAY, rot
from layout_heat import solve

def tloc(t):  # geo tile -> (ring,pos)
    if t['type'] == 'home': return (0, 0)
    if t['type'] == 'save': return (7, GOAL_SECTORS[TODAY.index(t['number'])])
    r, p = t['loc'].split('.'); return (int(r), int(p))

res = {}
for n, L in LAYOUTS.items():
    Vs = [solve(rot(L, r)) for r in range(3)]
    tiles = [k for k in Vs[0] if k[0] != 7]
    sd = {k: st.pstdev([V[k] for V in Vs]) for k in tiles}
    res[n] = (Vs[0], sd)
    print(f"{n:8s} {'-'.join(map(str,L))}  home {Vs[0][(0,0)]:.3f}  field mean {st.mean(Vs[0][k] for k in tiles):.3f}"
          f"  rotation SD per tile: mean {st.mean(sd.values()):.3f} max {max(sd.values()):.3f}", flush=True)

allv = [v for V, _ in res.values() for v in V.values()]
lo, hi = min(allv), max(allv)
def col(v):  # light yellow (fast) -> dark red (slow)
    x = (v - lo) / (hi - lo)
    stops = [(255, 247, 204), (253, 174, 97), (215, 48, 39), (110, 10, 30)]
    i = min(2, int(x * 3)); f = x * 3 - i
    return tuple(int(stops[i][j] + (stops[i + 1][j] - stops[i][j]) * f) for j in range(3))

panels = []
for n, L in LAYOUTS.items():
    V, sd = res[n]
    size = int(bp.W * bp.SCALE * bp.SS)
    img = Image.new('RGB', (size, size), (255, 255, 255)); d = ImageDraw.Draw(img)
    lw = max(1, int(1.3 * bp.SCALE * bp.SS))
    for t in bp.GEO['tiles']:
        v = V[tloc(t)]
        if t['type'] == 'home':
            (x0, y0), (x1, y1) = bp._xy(bp.CX - t['r1'], bp.CY - t['r1']), bp._xy(bp.CX + t['r1'], bp.CY + t['r1'])
            d.ellipse([x0, y0, x1, y1], fill=col(v), outline=bp.C['line'], width=lw)
        else:
            d.polygon(bp._sector(t), fill=col(v), outline=bp.C['line'], width=lw * (3 if t['type'] == 'save' else 1))
        x, y = bp._xy(*bp._centre(t))
        fs = 22 if t['type'] != 'field' else 15
        d.text((x, y), f'{v:.2f}', fill='black' if v < lo + 0.55 * (hi - lo) else 'white',
               font=bp._font(fs * bp.SCALE * bp.SS), anchor='mm')
        if t['type'] == 'save':
            num = L[GOAL_SECTORS.index(tloc(t)[1])]
            x, y = bp._xy(*bp._pt(t['r1'] + 34, (t['a0'] + t['a1']) / 2))
            d.text((x, y), str(num), fill=bp.C['ink'], font=bp._font(44 * bp.SCALE * bp.SS), anchor='mm')
    d.text((size // 2, int(30 * bp.SS)), f"{n}  {'-'.join(map(str, L))}   home {V[(0,0)]:.2f}   rotation SD {st.mean(sd.values()):.3f}",
           fill='black', font=bp._font(26 * bp.SCALE * bp.SS), anchor='mm')
    panels.append(img.resize((size // 2, size // 2), Image.LANCZOS))
w, h = panels[0].size
sheet = Image.new('RGB', (w * 3, h * 2 + 60), 'white')
for i, p in enumerate(panels): sheet.paste(p, ((i % 3) * w, (i // 3) * h))
d = ImageDraw.Draw(sheet)
for i in range(300):
    x = w * 3 // 2 - 300 + 2 * i
    d.rectangle([x, h * 2 + 10, x + 2, h * 2 + 30], fill=col(lo + (hi - lo) * i / 299))
d.text((w * 3 // 2 - 310, h * 2 + 20), f'{lo:.2f}', fill='black', font=bp._font(18), anchor='rm')
d.text((w * 3 // 2 + 310, h * 2 + 20), f'{hi:.2f} turns', fill='black', font=bp._font(18), anchor='lm')
sheet.save('/Users/tom/Game/BoardGame/layout_heat.png')
json.dump({n: {'tiles': {f'{k[0]}.{k[1]}': round(v, 4) for k, v in V.items()}, 'rot_sd_mean': st.mean(sd.values())}
           for n, (V, sd) in res.items()}, open('/Users/tom/Game/BoardGame/layout_heat.json', 'w'))
