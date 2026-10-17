# Save the 5 and 6, or bring N to its goal? (owner, 2026-10-16)

Mover has the numbered 5 and 6 on their goals, a numbered N (1-4) exactly 11 from
goal N, and rolls 5-6. A = save the 5 and the 6; B = both dice on N, onto its goal
(the route the net's 2-ply search ranks highest). Paired playouts from the
position after each, common dice per index, the net's 1-ply policy for both sides
(`rollout_probe.rollout`). Margin from the mover's side; report B - A.

Positions come from owner's games (`start_pool.jsonl`), made to fit by:
* `sswap.py` -- swapping identities of the mover's OWN board pieces: the 5/6 with
  a blank already on goal 5/6, N with a blank on a field tile 11 from goal N.
  Occupancy unchanged (walls, blots, threats, the opponent all as played). N is
  only ever taken from a field tile, never off its own goal. 229 positions, 150
  games, at most 3 per (game, N), 4+ turns apart.
* `splace.py` -- N already 11 away; the 5/6 MOVED onto their goals from the
  field (changes occupancy). Used only for the scarce late positions.

`runswap.py` samples 40 per N from the swap set (mover > 8 unsaved) plus every
late position (mover <= 8 unsaved) from both sets, gets the net's 2-ply values
for A and B, then N=40 pair playouts each. ~5 h on 3 workers of the iMac.

    cd /Users/tom/Game/BoardGame
    BOARDGAME_DEVICE=cpu PYTHONHASHSEED=0 NET=$PWD/blend4way_Oct6.pt N=40 PER=40 \
      nohup python3 -u probe_save56/runswap.py > probe_save56/runswap.log 2>&1 &

Output: `swap_prefs.json` (positions, net values) and `runswap.jsonl` (one line
per playout) in this folder. Group the CI by game: positions from one game are
correlated.

Earlier results (same method, unmodified positions):
* the 3 positions with N = 1 first found, 100 pairs each: B - A = -0.50
  (95% CI -0.85..-0.15); -1.26 / -0.24 / -0.01; the net preferred B by ~0.4 in
  all three. Pictures `pos_*.png`.
* 33 more unmodified positions (N = 1-4), 40 pairs: see `rungen` in CLAUDE.md.

Sample size: paired SD ~3.0 per playout, real between-position SD ~0.5, so
80% power at 5% needs ~94 positions for a 0.2 effect, ~42 for 0.3.
