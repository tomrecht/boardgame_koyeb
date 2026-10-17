import sys, os, json
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s56
def loc(t):
    if t.type == 'home': return 'H'
    if t.type == 'save': return f'G{t.number}'
    return f'{t.ring}.{t.pos}'
tok = lambda p: 'x' if p.number > 6 else str(p.number)
def notation(b):
    def side(pl):
        on = sorted(f'{tok(p)}@{loc(p.tile)}' for p in b.pieces if p.player == pl and p.tile)
        lst = lambda r: ','.join(map(tok, r)) or '-'
        un = b.white_unentered if pl == 'white' else b.black_unentered
        sv = b.white_saved if pl == 'white' else b.black_saved
        return ' '.join([pl[0].upper()] + on + ['r:' + lst(un), 's:' + lst(sv)])
    d = ''.join(str(x.number) for x in b.dice)
    return f'{b.current_player[0]} {d} | {side("white")} | {side("black")}'
if __name__ == '__main__':
  for r in s56.positions():
    b = s56.setup(r); A, B = s56.options(b)
    if not A: continue
    one = next(p for p in b.pieces if p.player == b.current_player and p.number == 1)
    print(json.dumps({'game': r['game'], 'turn': r['turn'], 'n': notation(b), 'one': loc(one.tile)}))
