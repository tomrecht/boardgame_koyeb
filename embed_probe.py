"""embed_probe.py -- does a v1 net already COMPUTE the v2 inputs internally?

fv2C (2026-10-19) won +0.26 vs the blend, but the new inputs stayed nearly unused
and input_probe found no information in them beyond the old inputs. One reason
would be redundancy: the net derives them itself. This fits each new input from
a v1 net's internal representation (it never saw them):

  NET     the trained net's final representation: per position the pooled vector
          the readout reads (3H + 11 v1 globals) plus the readout's two hidden
          layers; per piece its final embedding (H)
  RAW     the raw v1 inputs (positions: v1 globals + every piece row; pieces:
          own row + v1 globals + every piece row) -- what a linear read-off of
          the board gives with no computation
  RAND    the same architecture with random weights -- what message passing
          gives for free

Held-out R^2 (ridge, 5-fold CV grouped by game). A feature with NET near 1 and
well above RAW and RAND is computed by the net; one with NET near RAW/RAND is
not. Per-piece features (own-goal distance, P(captured), P(walled), enemy blots
>= 1 / >= 2) are fitted on pieces standing on field tiles.

Records come from DUMP_RECORDS (td_selfplay_loop); each is encoded turn-start v1
for the net and AB2 for the targets.

    python3 embed_probe.py <records.jsonl.gz> [net.pt=blend.pt] [n_positions=6000]
"""
import gzip, json, random, sys, time

import numpy as np
import torch
import torch.nn.functional as F

import network
import encoder as E
from network import BoardEncoder, BoardGNN, collate_batch
from game import Board

A_NAMES = ['no-save counter', 'opp saveable', 'my unentered', 'my on field',
           'my saved', 'opp unentered', 'opp on field', 'opp saved', 'my race',
           'opp race', 'my turns to finish', 'opp turns to finish',
           'my endgame table', 'opp endgame table', 'my numbered on own goal',
           'opp numbered on own goal']
ROLL_TYPES = ['saves numbered', 'saves blank', 'onto goal numbered',
              'onto goal blank', 'captures numbered', 'captures blank']
PIECE_NAMES = ['own-goal distance', 'P(captured)', 'P(walled)', 'blots >= 1', 'blots >= 2']


@torch.no_grad()
def representation(model, batch):
    """(position vectors [B, D], piece embeddings [B, P, H]) -- _forward_batch's
    internals, kept: the readout input plus its two hidden layers."""
    B, T, P = batch['B'], batch['T'], batch['P']
    dev = batch['tile_feats'].device
    tile_h = F.relu(model.tile_embed(batch['tile_feats']))
    piece_h = F.relu(model.piece_embed(batch['piece_feats']))
    piece_h_init = piece_h
    global_h = model.global_embed(torch.zeros(B, 1, dtype=torch.long, device=dev))
    for mp in model.mp_layers:
        tile_h, piece_h, global_h = mp(tile_h, piece_h, global_h, batch['tile_edge_index'],
                                       batch['piece_to_tile'], batch['tile_to_piece'], B, T, P)
    combined = torch.cat([tile_h.mean(1), (piece_h.mean(1) + piece_h_init.mean(1)) / 2,
                          global_h.squeeze(1), batch['global_feats']], dim=1)
    r = model.readout
    h1 = r[1](r[0](combined))
    h2 = r[3](r[2](h1))
    return torch.cat([combined, h1, h2], 1).numpy(), piece_h.numpy()


def collect(recs, nets, enc1, enc2):
    """Per position: targets (new globals), RAW, and each net's vector; per field
    piece: targets (new piece columns), RAW, and each net's embedding."""
    v1t, v1p, v1g = E.feature_dims('v1')
    board = Board()
    pos = {'y': [], 'RAW': [], 'game': [], **{k: [] for k in nets}}
    pc = {'y': [], 'RAW': [], 'game': [], **{k: [] for k in nets}}
    chunk = []

    def flush():
        if not chunk:
            return
        batch = collate_batch([c[0] for c in chunk])
        reps = {k: representation(m, batch) for k, m in nets.items()}
        for i, (e1, e2, field_idx, game) in enumerate(chunk):
            g1 = np.asarray(e1['global_feats']).ravel()
            p1 = np.asarray(e1['piece_feats'])
            p2 = np.asarray(e2['piece_feats'])
            pos['y'].append(np.asarray(e2['global_feats']).ravel()[v1g:])
            pos['RAW'].append(np.concatenate([g1, p1.ravel()]))
            pos['game'].append(game)
            for k in nets:
                pos[k].append(reps[k][0][i])
            for j in field_idx:
                pc['y'].append(p2[j, v1p:])
                pc['RAW'].append(np.concatenate([p1[j], g1, p1.ravel()]))
                pc['game'].append(game)
                for k in nets:
                    pc[k].append(reps[k][1][i, j])
        chunk.clear()

    for r in recs:
        try:
            board.update_state(r['raw_state'])
            e1 = enc1.encode(board, r['player'])
            e2 = enc2.encode(board, r['player'])
            _, pieces = E.encode_piece_features(board, enc1.tile_index, r['player'], None)
        except Exception:
            continue
        field_idx = [j for j, p in enumerate(pieces)
                     if p.tile is not None and p.tile.type == 'field']
        chunk.append((e1, e2, field_idx, r['game_id']))
        if len(chunk) >= 128:
            flush()
    flush()
    return ({k: np.asarray(v) for k, v in pos.items()},
            {k: np.asarray(v) for k, v in pc.items()})


def cv_r2(X, Y, groups, k=5):
    """Held-out R^2 per target column (ridge, alpha by inner CV, grouped folds)."""
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    pred = np.zeros_like(Y)
    for tr, te in GroupKFold(n_splits=k).split(X, Y, groups):
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 11)))
        m.fit(X[tr], Y[tr])
        pred[te] = m.predict(X[te])
    var = Y.var(0)
    out = np.full(Y.shape[1], np.nan)
    ok = var > 1e-9
    out[ok] = 1 - ((pred - Y) ** 2).mean(0)[ok] / var[ok]
    return out


def main():
    path = sys.argv[1]
    ckpt = sys.argv[2] if len(sys.argv) > 2 else 'blend.pt'
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
    network.DEVICE = torch.device('cpu')
    net = network.model_from_state(torch.load(ckpt, map_location='cpu'))
    if net.features != 'v1':
        raise SystemExit(f'{ckpt} is {net.features}; probe a v1 net (it never saw the new inputs)')
    torch.manual_seed(0)
    rand = BoardGNN(features='v1').eval()
    nets = {'NET': net, 'RAND': rand}
    enc1 = BoardEncoder(features='v1')
    enc2 = E.BoardEncoder(features='AB2', scaled=False)
    recs = [json.loads(l) for l in gzip.open(path, 'rt')]
    random.Random(0).shuffle(recs)
    recs = recs[:n]
    t0 = time.time()
    pos, pc = collect(recs, nets, enc1, enc2)
    print(f'{len(pos["y"])} positions, {len(pc["y"])} field pieces, '
          f'{len(set(pos["game"]))} games; net {ckpt}; {time.time() - t0:.0f}s')

    r_pos = {k: cv_r2(pos[k], pos['y'], pos['game']) for k in ('NET', 'RAW', 'RAND')}
    r_pc = {k: cv_r2(pc[k], pc['y'], pc['game']) for k in ('NET', 'RAW', 'RAND')}

    def row(name, i, R, Y):
        nz = 100 * np.mean(np.abs(Y[:, i]) > 1e-9)
        cells = ' '.join(f'{R[k][i]:6.3f}' if not np.isnan(R[k][i]) else '   n/a'
                         for k in ('NET', 'RAW', 'RAND'))
        print(f'  {name:28s} {cells}   nonzero {nz:5.1f}%')

    print(f'\n  {"held-out R^2":28s}    NET    RAW   RAND')
    print('per piece (field pieces):')
    for i, nm in enumerate(PIECE_NAMES):
        row(nm, i, r_pc, pc['y'])
    print('bookkeeping globals (A / A2):')
    for i, nm in enumerate(A_NAMES):
        row(nm, i, r_pos, pos['y'])
    print('roll opportunities (mean over dice 1-6):')
    for s, side in enumerate(('my', 'opp')):
        for t, tn in enumerate(ROLL_TYPES):
            cols = [16 + 36 * s + 6 * d + t for d in range(6)]
            m = {k: np.nanmean(r_pos[k][cols]) for k in r_pos}
            nz = 100 * np.mean(np.abs(pos['y'][:, cols]) > 1e-9)
            print(f'  {side + " " + tn:28s} ' + ' '.join(f'{m[k]:6.3f}' for k in ('NET', 'RAW', 'RAND'))
                  + f'   nonzero {nz:5.1f}%')
    print(f'\ntotal {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
