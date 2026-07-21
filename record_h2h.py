"""Record iter10-vs-iter14 head-to-head games and emit standalone HTML replays.

Motivation (owner): watch HOW iter14 defeats iter10's goal-pair blocks -- does
it PREVENT them (2a) or ALLOW-and-BLOCK-SAVE them (2b, a sacrifice that hands
iter10 the blocking pieces but clears the absolute block)? So we prioritize
recording games in which iter10 actually forms a goal-pair block against iter14.

Each replay is a SELF-CONTAINED .html (board geometry + all frames + viewer JS
embedded) -- just open it in a browser; prev/next/play to step through every
move. No server, no dependencies. Board layout mirrors the real polar board
(concentric rings x 12 spokes; green = goal tiles, yellow = home).

Usage: python -u record_h2h.py [N_SEEDS] [MAX_REPLAYS]
Writes replay_<tag>.html into ./replays/.
"""
import os, sys, json, random
os.environ.setdefault('BOARDGAME_DEVICE', 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
torch.set_num_threads(1); torch.set_grad_enabled(False)
import network
network.DEVICE = torch.device('cpu')
from network import BoardGNN
from game import Board
from agent_gnn import GNNAgent

REPO = os.path.dirname(os.path.abspath(__file__))
A_TAG, A_NET = 'iter10', f'{REPO}/td_champion_July18_iter10.pt'
B_TAG, B_NET = 'iter14', f'{REPO}/td_champion_July19_iter14.pt'
OUTDIR = f'{REPO}/replays'
TILE_GEO = json.load(open(f'{REPO}/tile_neighbors.json'))
MAX_TURNS, STUCK_LIMIT = 200, 60
INF = float('inf')
PAIRS = [(2, 4), (1, 6), (3, 5)]
PAIR_NAME = {(2, 4): '2&4', (1, 6): '1&6', (3, 5): '3&5'}


def agent(path):
    m = BoardGNN(); m.load_state_dict(torch.load(path, map_location='cpu'),
                                      strict=False); m.eval()
    return GNNAgent(model=m)


def blocked_pairs(board, player):
    saved = board.white_saved if player == 'white' else board.black_saved
    saved_nums = {p.number for p in saved}
    denied = {p.number for p in board.pieces
              if p.player == player and p.number <= 6
              and p.number not in saved_nums
              and board.shortest_route_to_goal(p) == INF}
    return [PAIR_NAME[pr] for pr in PAIRS if pr[0] in denied and pr[1] in denied]


def snapshot(board, caption):
    pieces = [{'c': p.player, 'n': p.number, 'r': p.tile.ring, 's': p.tile.pos}
              for p in board.pieces if p.tile is not None]
    return {
        'cap': caption,
        'pieces': pieces,
        'wsaved': sorted(p.number for p in board.white_saved),
        'bsaved': sorted(p.number for p in board.black_saved),
        'wrack': sorted(p.number for p in board.white_unentered),
        'brack': sorted(p.number for p in board.black_unentered),
        'dice': [board.dice[0].number, board.dice[1].number],
        # blocks AGAINST each color (i.e. that color's goals denied by the other)
        'blk_w': blocked_pairs(board, 'white'),
        'blk_b': blocked_pairs(board, 'black'),
    }


def describe(m, player, tag):
    who = f"{tag}"
    if m in ((0, 0, 0), None):
        return f"{who}: pass (die unused)"
    pid, dest, roll = m
    nm = f"{'W' if pid[0] == 'white' else 'B'}{pid[1]}"
    if dest == 'save':
        return f"{who}: {nm} → SAVE (rolled {roll})"
    if dest == 0 and roll == 0:
        return f"{who}: {nm} BLOCK-SAVE — dissolves opponent block (sacrifice)"
    if isinstance(dest, (list, tuple)):
        return f"{who}: {nm} → ({dest[0]},{dest[1]})"
    return f"{who}: {nm} → {dest}"


def is_blocksave(m):
    return (isinstance(m, tuple) and len(m) == 3
            and m not in ((0, 0, 0), (1, 1, 1)) and m[1] == 0 and m[2] == 0)


def play_and_record(seed, white_tag, white_agent, black_tag, black_agent):
    random.seed(seed)
    board = Board()
    agents = {'white': (white_agent, white_tag), 'black': (black_agent, black_tag)}
    frames = [snapshot(board, "start")]
    last_saved, since = 0, 0
    iter10_blocked = False
    blocksaves = 0
    winner, score = None, 0
    for turn in range(MAX_TURNS):
        w, s = board.check_game_over()
        if w:
            winner, score = w, s
            break
        if board.draw_callable or (last_saved > 0 and since >= STUCK_LIMIT):
            break
        cur = len(board.white_saved) + len(board.black_saved)
        if cur > last_saved:
            last_saved, since = cur, 0
        elif last_saved > 0:
            since += 1
        player = board.current_player
        ag, tag = agents[player]
        chosen = ag.select_move_pair(list(board.get_valid_moves()), board, player)
        if isinstance(chosen, tuple) and len(chosen) == 3:
            chosen = (chosen, (0, 0, 0))
        for m in chosen:
            if is_blocksave(m):
                blocksaves += 1
            if m != (0, 0, 0):
                board.apply_move(m, switch_turn=False)
                frames.append(snapshot(board, f"turn {turn+1} — "
                                       + describe(m, player, tag)))
        board.switch_turn()
        # iter10 blocks iter14  <=>  iter14's goals are denied. iter14's color:
        i14_denied = frames[-1]['blk_b'] if black_tag == B_TAG else frames[-1]['blk_w']
        if i14_denied:
            iter10_blocked = True
    result = (f"{winner or 'draw'}"
              + (f" by {score}" if winner else ""))
    frames.append(snapshot(board, f"GAME OVER — {result}"))
    return frames, iter10_blocked, blocksaves, winner, score


def write_html(path, title, frames):
    html = HTML_TEMPLATE
    html = html.replace('/*__TITLE__*/', json.dumps(title))
    html = html.replace('/*__GEO__*/', json.dumps(TILE_GEO))
    html = html.replace('/*__FRAMES__*/', json.dumps(frames))
    with open(path, 'w') as f:
        f.write(html)


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    max_replays = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    os.makedirs(OUTDIR, exist_ok=True)
    agA, agB = agent(A_NET), agent(B_NET)
    print(f"playing up to {n_seeds} seeds (x2 colors), keeping up to "
          f"{max_replays} games where {A_TAG} blocks {B_TAG}...", flush=True)
    written = 0
    fallback = None
    for k in range(n_seeds):
        seed = 5_500_000 + k
        for gi, (wt, wa, bt, ba) in enumerate((
                (A_TAG, agA, B_TAG, agB), (B_TAG, agB, A_TAG, agA))):
            frames, blocked, bsaves, winner, score = play_and_record(
                seed, wt, wa, bt, ba)
            tag = f"{k}_{gi}_{wt}W_vs_{bt}B"
            print(f"  {tag}: winner={winner} score={score} "
                  f"iter10_blocked={blocked} blocksaves={bsaves} "
                  f"frames={len(frames)}", flush=True)
            if fallback is None:
                fallback = (tag, frames, winner, score)
            if blocked and written < max_replays:
                title = (f"{wt} (white) vs {bt} (black) — {winner or 'draw'}"
                         f"{(' +' + str(score)) if winner else ''} "
                         f"— iter10 blocked; {bsaves} block-saves")
                out = f"{OUTDIR}/replay_{tag}.html"
                write_html(out, title, frames)
                written += 1
                print(f"    -> wrote {out}", flush=True)
    if written == 0 and fallback:
        tag, frames, winner, score = fallback
        out = f"{OUTDIR}/replay_{tag}.html"
        write_html(out, f"{tag} — {winner or 'draw'} (no block observed)", frames)
        print(f"no blocking games found; wrote one sample: {out}", flush=True)
    print(f"DONE: {written} replay(s) in {OUTDIR}/", flush=True)


HTML_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>H2H replay</title><style>
 body{margin:0;background:#1b1b1f;color:#eee;font:14px/1.4 system-ui,sans-serif}
 #wrap{display:flex;flex-direction:column;align-items:center;padding:10px}
 canvas{background:#111;border-radius:8px}
 #cap{min-height:2.4em;margin:8px;text-align:center;max-width:760px;font-size:15px}
 #ctl{display:flex;gap:8px;align-items:center;margin:6px}
 button{background:#333;color:#eee;border:1px solid #666;border-radius:6px;
        padding:6px 12px;cursor:pointer;font-size:14px}
 button:hover{background:#444}
 #blk{color:#ff8;min-height:1.4em;font-weight:bold}
 input[type=range]{width:520px}
</style></head><body><div id="wrap">
 <div id="blk"></div>
 <canvas id="cv" width="760" height="760"></canvas>
 <div id="cap"></div>
 <div id="ctl">
   <button id="first">|&lt;</button><button id="prev">&lt; prev</button>
   <button id="play">play</button><button id="next">next &gt;</button>
   <button id="last">&gt;|</button>
   <span id="cnt"></span>
 </div>
 <input type="range" id="slider" min="0" value="0">
</div><script>
const TITLE=/*__TITLE__*/; const GEO=/*__GEO__*/; const FRAMES=/*__FRAMES__*/;
document.title=TITLE;
const CX=900,CY=640,STEP=60,HOME=90,SCALE=0.66,CCX=380,CCY=380;
function tilePos(r,s){ if(r===0)return {x:CX,y:CY};
  const rad=HOME+STEP*(r-0.5), ang=(s/12)*2*Math.PI - Math.PI/2;
  return {x:CX+rad*Math.cos(ang), y:CY+rad*Math.sin(ang)}; }
function toC(p){ return {x:CCX+(p.x-CX)*SCALE, y:CCY+(p.y-CY)*SCALE}; }
// precompute tile render list
const TILES=[]; for(const k in GEO){ const v=GEO[k];
  const m=k.match(/ring(\d+)_sector(\d+)/); const r=+m[1], s=+m[2];
  TILES.push({r,s,type:v.type,number:v.number, pos:toC(tilePos(r,s))}); }
const cv=document.getElementById('cv'), g=cv.getContext('2d');
function circle(x,y,rad,fill,stroke){ g.beginPath();g.arc(x,y,rad,0,7);g.fillStyle=fill;g.fill();
  if(stroke){g.lineWidth=2;g.strokeStyle=stroke;g.stroke();} }
function draw(i){ const f=FRAMES[i]; g.clearRect(0,0,760,760);
  // tiles
  for(const t of TILES){ let col='#2a2a30';
    if(t.type==='home')col='#6b6b1e'; else if(t.type==='save')col='#1e6b2e';
    circle(t.pos.x,t.pos.y,t.type==='home'?22:13,col,'#000');
    if(t.type==='save'){ g.fillStyle='#cfc';g.font='11px sans-serif';g.textAlign='center';
      g.textBaseline='middle';g.fillText(t.number,t.pos.x,t.pos.y); } }
  // group pieces by tile for clustering
  const byTile={}; for(const p of f.pieces){ const key=p.r+'_'+p.s;
    (byTile[key]=byTile[key]||[]).push(p); }
  for(const key in byTile){ const arr=byTile[key]; const [r,s]=key.split('_').map(Number);
    const base=toC(tilePos(r,s)); arr.forEach((p,j)=>{
      const off=arr.length>1? j*8-((arr.length-1)*4):0;
      const x=base.x+off, y=base.y+(arr.length>1? off*0.0:0);
      circle(x,y,9, p.c==='white'?'#eee':'#222', p.n<=6?'#e33':'#777');
      g.fillStyle=p.c==='white'?'#111':'#eee';g.font='bold 10px sans-serif';
      g.textAlign='center';g.textBaseline='middle';g.fillText(p.n,x,y); }); }
  // racks
  function rack(list,x,y,label,white){ g.fillStyle='#aaa';g.font='11px sans-serif';
    g.textAlign='left';g.fillText(label,x,y-12);
    list.forEach((n,j)=>{ const px=x+16*(j%8), py=y+16*Math.floor(j/8);
      circle(px,py,7,white?'#eee':'#222',n<=6?'#e33':'#777');
      g.fillStyle=white?'#111':'#eee';g.font='bold 9px sans-serif';g.textAlign='center';
      g.textBaseline='middle';g.fillText(n,px,py); }); }
  rack(f.wsaved,12,40,'W saved',true); rack(f.bsaved,12,110,'B saved',false);
  rack(f.wrack,12,700,'W rack',true); rack(f.brack,640,700,'B rack',false);
  g.fillStyle='#8cf';g.font='12px sans-serif';g.textAlign='right';
  g.fillText('dice '+f.dice.join(','),748,20);
  document.getElementById('cap').textContent=f.cap;
  const blocks=[]; if(f.blk_w.length)blocks.push('WHITE denied '+f.blk_w.join(','));
  if(f.blk_b.length)blocks.push('BLACK denied '+f.blk_b.join(','));
  document.getElementById('blk').textContent=blocks.length?('⛔ GOAL-PAIR BLOCK: '+blocks.join('   |   ')):'';
  document.getElementById('cnt').textContent=(i+1)+' / '+FRAMES.length;
  document.getElementById('slider').value=i;
}
let idx=0,playing=null; const sl=document.getElementById('slider');
sl.max=FRAMES.length-1;
function go(i){ idx=Math.max(0,Math.min(FRAMES.length-1,i)); draw(idx); }
document.getElementById('next').onclick=()=>go(idx+1);
document.getElementById('prev').onclick=()=>go(idx-1);
document.getElementById('first').onclick=()=>go(0);
document.getElementById('last').onclick=()=>go(FRAMES.length-1);
sl.oninput=()=>go(+sl.value);
document.getElementById('play').onclick=function(){ if(playing){clearInterval(playing);playing=null;this.textContent='play';return;}
  this.textContent='pause'; playing=setInterval(()=>{ if(idx>=FRAMES.length-1){clearInterval(playing);playing=null;document.getElementById('play').textContent='play';return;} go(idx+1); },700); };
document.onkeydown=e=>{ if(e.key==='ArrowRight')go(idx+1); if(e.key==='ArrowLeft')go(idx-1); };
draw(0);
</script></body></html>"""


if __name__ == '__main__':
    main()
