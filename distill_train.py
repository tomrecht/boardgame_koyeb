"""distill_train.py -- train an afterstate net toward close-call SEARCH values.

The net's measured weakness is value noise between near-equal moves (CLAUDE.md:
calibration corr ~0.2 vs playouts). close_call.py's search scores a candidate as
minus the opponent's expected best reply over its 21 rolls -- one turn deeper,
exact in the dice. Distillation trains the net to give that value directly, so
the gain arrives at today's phone cost, and the next net labels the next round
(expert iteration).

Data: `close_call.py label` files (gzip JSONL):
  kind 'cand'  a searched candidate's end-of-turn position, 'label' = its search
               value (mover frame, raw units, the net's own scale)
  kind 'turn'  each turn's chosen afterstate: an ordinary TD(lambda) chain,
               targets from a frozen copy of the net refreshed every epoch,
               exactly as train_td does -- the anchor to game outcomes
Loss = DISTILL_W x MSE(cand, label) + TD_W x MSE(turn, td_target)
     + RANK_W x listwise CE within each decision (softmax of the net's values
       vs softmax of the labels, temperature RANK_TAU raw units), since the
       ORDER of near-equal candidates is what picks the move.
Upweighting (train_td.UPW) applies to both kinds. Split by game.

    python3 distill_train.py --data cc_labels_*.jsonl.gz --load fv2C_it55.pt --save cand.pt

Knobs (flags; defaults are a first guess, not tuned):
    --distill-w 1.0 --td-w 0.5 --rank-w 0.0 --rank-tau 0.02
    --lr 2e-5 --epochs 8 --lam 0.9 --batch-size 64 --val-frac 0.15
"""
import argparse, copy, gzip, json, math, random
from collections import defaultdict

import torch
import torch.nn.functional as F

import network
from network import BoardEncoder, collate_batch
from game import Board
from train_td import split_by_game, upweight, compute_epoch_targets


def load_label_files(paths):
    cands, turns = [], []
    for p in paths:
        op = gzip.open if p.endswith('.gz') else open
        with op(p, 'rt') as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                k = r.get('kind')
                if k == 'cand':
                    cands.append(r)
                elif k == 'turn':
                    turns.append(r)
    return cands, turns


def _encode(recs, encoder, board, target_key):
    """Encode records, keeping only those that encode; returns (batch, targets,
    weights, kept records) aligned."""
    enc, tg, w, kept = [], [], [], []
    for r in recs:
        t = r.get(target_key)
        if t is None:
            continue
        try:
            board.update_state(r['raw_state'])
            e = encoder.encode(board, r['player'])
            f = upweight(board, r)
        except Exception:
            continue
        enc.append(e); tg.append(float(t)); w.append(f); kept.append(r)
    if not enc:
        return None, None, None, []
    dev = network.DEVICE
    return (collate_batch(enc), torch.tensor(tg, dtype=torch.float32, device=dev),
            torch.tensor(w, dtype=torch.float32, device=dev), kept)


def _batches(cands, turns, batch_size, shuffle):
    """Batches mixing whole DECISIONS (so the ranking term sees each group
    intact) with turn records, in proportion."""
    groups = defaultdict(list)
    for r in cands:
        groups[r['decision']].append(r)
    glist = list(groups.values())
    tlist = list(turns)
    if shuffle:
        random.shuffle(glist); random.shuffle(tlist)
    n_c = sum(len(g) for g in glist)
    n_b = max(1, math.ceil((n_c + len(tlist)) / batch_size))
    per_t = math.ceil(len(tlist) / n_b)
    gi = ti = 0
    for _ in range(n_b):
        bc = []
        while gi < len(glist) and len(bc) < batch_size - per_t:
            bc.extend(glist[gi]); gi += 1
        bt = tlist[ti:ti + per_t]; ti += per_t
        if bc or bt:
            yield bc, bt
    while gi < len(glist):                       # leftovers (rounding)
        bc = []
        while gi < len(glist) and len(bc) < batch_size:
            bc.extend(glist[gi]); gi += 1
        yield bc, []


def run_epoch(model, cands, turns, encoder, board, opt, args, training):
    model.train(training)
    tot = defaultdict(float)
    for bc, bt in _batches(cands, turns, args.batch_size, training):
        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            loss = torch.zeros((), device=network.DEVICE)
            if bc:
                b, y, w, kept = _encode(bc, encoder, board, 'label')
                if b is not None:
                    pred = model(b)
                    l_d = (w * (pred - y) ** 2).sum() / w.sum()
                    loss = loss + args.distill_w * l_d
                    tot['distill'] += l_d.item() * len(kept); tot['n_c'] += len(kept)
                    # agreement of the net's best candidate with the search's, per decision
                    idx = defaultdict(list)
                    for i, r in enumerate(kept):
                        idx[r['decision']].append(i)
                    l_r, n_g = torch.zeros((), device=network.DEVICE), 0
                    for ii in idx.values():
                        if len(ii) < 2:
                            continue
                        ii_t = torch.tensor(ii, device=network.DEVICE)
                        p_g, y_g = pred[ii_t], y[ii_t]
                        tot['top_agree'] += float(int(p_g.argmax()) == int(y_g.argmax()))
                        tot['n_g'] += 1
                        if args.rank_w > 0:
                            q = F.softmax(y_g / args.rank_tau, dim=0)
                            l_r = l_r - (q * F.log_softmax(p_g / args.rank_tau, dim=0)).sum()
                            n_g += 1
                    if n_g:
                        l_r = l_r / n_g
                        loss = loss + args.rank_w * l_r
                        tot['rank'] += l_r.item() * n_g
            if bt and args.td_w > 0:
                b, y, w, kept = _encode(bt, encoder, board, 'td_target')
                if b is not None:
                    pred = model(b)
                    l_t = (w * (pred - y) ** 2).sum() / w.sum()
                    loss = loss + args.td_w * l_t
                    tot['td'] += l_t.item() * len(kept); tot['n_t'] += len(kept)
            if training and loss.requires_grad:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
    out = {'distill': tot['distill'] / max(tot['n_c'], 1),
           'td': tot['td'] / max(tot['n_t'], 1),
           'rank': tot['rank'] / max(tot['n_g'], 1),
           'top_agree': tot['top_agree'] / max(tot['n_g'], 1),
           'n_cand': int(tot['n_c']), 'n_turn': int(tot['n_t']), 'n_dec': int(tot['n_g'])}
    out['total'] = args.distill_w * out['distill'] + args.td_w * out['td'] + args.rank_w * out['rank']
    return out


def baseline(model, cands, encoder, board, batch_size=256):
    """How far the UNTRAINED net is from the labels: MSE, the shallow-vs-label
    corr, and top-1 agreement -- what training has to improve on."""
    model.eval()
    preds, ys, decs = [], [], []
    with torch.no_grad():
        for i in range(0, len(cands), batch_size):
            b, y, _, kept = _encode(cands[i:i + batch_size], encoder, board, 'label')
            if b is None:
                continue
            preds += model(b).tolist(); ys += y.tolist(); decs += [r['decision'] for r in kept]
    n = len(ys)
    if n < 3:
        return {}
    mse = sum((p - y) ** 2 for p, y in zip(preds, ys)) / n
    by = defaultdict(list)
    for p, y, d in zip(preds, ys, decs):
        by[d].append((p, y))
    # within-decision corr of (pred - mean) vs (label - mean): what decides moves
    dp, dy, agree, ng = [], [], 0, 0
    for g in by.values():
        if len(g) < 2:
            continue
        mp_ = sum(p for p, _ in g) / len(g); my = sum(y for _, y in g) / len(g)
        dp += [p - mp_ for p, _ in g]; dy += [y - my for _, y in g]
        agree += max(range(len(g)), key=lambda i: g[i][0]) == max(range(len(g)), key=lambda i: g[i][1])
        ng += 1
    def corr(a, b):
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        va = sum((x - ma) ** 2 for x in a); vb = sum((x - mb) ** 2 for x in b)
        return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb) if va and vb else float('nan')
    return {'mse': mse, 'within_corr': corr(dp, dy) if dp else float('nan'),
            'top_agree': agree / max(ng, 1), 'decisions': ng}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', nargs='+', required=True)
    ap.add_argument('--load', required=True, help='warm start (an afterstate net)')
    ap.add_argument('--save', required=True)
    ap.add_argument('--distill-w', type=float, default=1.0)
    ap.add_argument('--td-w', type=float, default=0.5)
    ap.add_argument('--rank-w', type=float, default=0.0)
    ap.add_argument('--rank-tau', type=float, default=0.02, help='raw units (1 pt = 1/12)')
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--lam', type=float, default=0.9)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--log', default=None, help='append a JSON summary line here')
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    model = network.model_from_state(torch.load(args.load, map_location='cpu'), strict=False).to(network.DEVICE)
    assert getattr(model, 'afterstate', False), 'distillation targets afterstates: load an afterstate net'
    encoder = BoardEncoder(features=getattr(model, 'features', 'v1'), afterstate=True)
    board = Board()

    cands, turns = load_label_files(args.data)
    print(f'{len(cands)} candidate + {len(turns)} turn records from {len(args.data)} file(s)')
    # split by GAME across both kinds
    tr_ids = {r['game_id'] for r in split_by_game(turns or cands, args.val_frac, args.seed)[0]}
    c_tr = [r for r in cands if r['game_id'] in tr_ids]; c_va = [r for r in cands if r['game_id'] not in tr_ids]
    t_tr = [r for r in turns if r['game_id'] in tr_ids]; t_va = [r for r in turns if r['game_id'] not in tr_ids]
    print(f'  train {len(c_tr)} cand / {len(t_tr)} turn, val {len(c_va)} cand / {len(t_va)} turn')

    base = baseline(model, c_va, encoder, board)
    print(f'  BASELINE (warm start) on val candidates: {base}')
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, best_state, hist = float('inf'), None, []
    for ep in range(1, args.epochs + 1):
        if args.td_w > 0:
            t_tr_t = compute_epoch_targets(model, t_tr, encoder, board, args.lam, 1.0)
            t_va_t = compute_epoch_targets(model, t_va, encoder, board, args.lam, 1.0)
        else:
            t_tr_t, t_va_t = [], []
        tr = run_epoch(model, c_tr, t_tr_t, encoder, board, opt, args, True)
        va = run_epoch(model, c_va, t_va_t, encoder, board, opt, args, False)
        hist.append({'epoch': ep, 'train': tr, 'val': va})
        print(f'epoch {ep}: train distill {tr["distill"]:.5f} td {tr["td"]:.5f} | '
              f'val distill {va["distill"]:.5f} td {va["td"]:.5f} rank {va["rank"]:.4f} '
              f'top-1 agree {va["top_agree"]:.3f} ({va["n_dec"]} decisions)', flush=True)
        if va['total'] < best:
            best, best_state = va['total'], copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    after = baseline(model, c_va, encoder, board)
    print(f'  AFTER (best epoch) on val candidates: {after}')
    torch.save(model.state_dict(), args.save)
    print(f'saved {args.save}')
    if args.log:
        with open(args.log, 'a') as fh:
            fh.write(json.dumps({'save': args.save, 'load': args.load, 'data': args.data,
                                 'args': vars(args), 'baseline': base, 'after': after,
                                 'history': hist}) + '\n')


if __name__ == '__main__':
    main()
