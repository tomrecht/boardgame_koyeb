# Quahuru strategy — notes toward a player guide

Raw material for the in-game Strategy panel (opened from How to Play; owner
chose a panel over a separate page, 2026-10-10). Owner's tips are the
backbone; measured facts (CLAUDE.md "Domain facts", ARCHIVE.md) are noted
where they support or qualify a tip.

## Tips, in order of importance, grouped (owner, 2026-10-10)

### Numbered pieces
1. **Get numbered pieces to their goal early** (the most important tip), or
   at least within one die of it — unless one is holding a wall, especially a
   wall against a higher-numbered piece.
2. **Don't leave numbered blots.** A lone BLANK on an outer ring is fine: it
   is often the first step of a wall there.
3. **Park numbered pieces behind a wall** so they can't be captured.
   - **Always save a numbered piece rather than a blank** when you can do
     either, **especially a high-numbered one** (owner): while it sits on a
     high goal, that goal is your highest occupied one, so in the endgame your
     blanks on lower goals lose the higher-die rule and need exact rolls.

### Every roll should count
4. **Aim for a position where every number 1-6 does something useful** --
   saves a piece, brings one to its goal, builds a wall, captures. Spread your
   pieces so that no roll is wasted.

### Walls
5. **Block goal pairs.**
   - **With a pair blocked, capture an opponent piece every turn you can**: a
     captured piece must re-enter first, so they can't spend the dice on
     block-saving your wall.
   - **When they have walled a pair against your numbered piece, don't
     block-save it while your dice have anything else useful to do.** They
     will eventually run out of moves and have to dissolve the wall or pass.
6. **Where to wall: rings 5 and 6 are best**, especially 5 (those tiles are
   more of a hub), **then ring 3.**
7. **All-blank walls beat walls containing a numbered piece.** If you have to
   build a wall with a numbered piece, try to swap it out for a blank later.

### Endgame
8. **Spread blanks among the goals, favouring the lower goals**, and **use a
   4 to drop a blank from a high goal to its low partner** (6->1, 5->3, 4->2
   are 4 steps apart), where it banks more cheaply.

### Captures
9. **A goal usually beats a capture**: over 330 positions taking the goal did
   +0.16 pts better on average. Capture when it's cheap, or to hold a pair
   block (tip 5).

### Closing note (owner)
It's largely a game of chance: you can play perfectly and still lose.

Rejected: "keep a numbered piece for last" -- with a numbered piece and a
blank on the same goal, banking either leaves the same position (the
last-piece rule makes the lone survivor a blank), so save the numbered piece
whenever you can.

## Shipped as the Strategy panel (2026-10-10)

`showStrategy` in game.js, wording in WORDING.md section 10. Pictures in
`assets/strategy/`, from these positions (`node pos_image.mjs "<pos>" out.png
<flags>`, then cropped to the board, 240,40,1040,832, and resized to 600 wide):

- park.webp: `b 44 | W 3@6.8 x@5.8 x@5.8 x@3.2 5@4.6 r:6,x,x s:1,2,4,x | B x@4.8 x@3.9 x@5.30 r:1,2,x,x s:3,4,5,6,x` `--ring 6.8 --fill 5.8:#e0b040`
- every-roll.webp: `w 36 | W x@G1 2@G2 3@4.8 x@5.13 x@2.6 r:- s:1,4,5,6,x,x,x | B x@3.11 r:3,4,5,6,x,x,x s:1,2,x,x` `--ring G1,G2,G3,G4,G5,3.11`
- pair-block.webp: `w 52 | W 2@3.4 4@4.2 x@3.8 x@2.10 r:1,3,x,x s:5,6,x,x | B x@5.2 x@5.2 x@5.4 x@5.4 6@G6 r:1,x,x s:2,3,4,5` `--fill 5.2,5.4:#e0b040 --ring G2,G4`
- hop.webp: `w 41 | W x@G1 x@G2 x@G5 r:- s:1,2,3,4,5,6,x,x,x | B x@G4 5@6.6 2@5.4 r:- s:1,3,4,6,x,x,x,x,x` `--ring G5,G3`

Claims in the captions were checked against the tile graph
(tile_neighbors.json), not the engine. Endgame table, blanks only: G1+G2+G5
banks out in 2.20 turns, G1+G2+G3 in 2.03 -- but all three on G1 takes 2.00,
so with NO numbered piece left, spreading buys nothing; it pays when a
numbered piece is still out (numbered 6: 3.46 spread vs 3.85 all on G1).

## What the measurements say about them

- **(1) Numbered pieces are the endgame bottleneck:** 3.27 turns to bank from
  their own goal (exact match, 11/36 a turn) against 1.00-1.64 for a blank.
  The net puts a numbered piece on its goal over a blank save in 871 of 1,002
  positions.
- **(5) Two walls seal a whole 4-apart pair** (6&1, 5&3, 4&2): the pair shares
  its approach corridors, so denying a pair costs no more than denying one
  goal. Sealing all six would take 12 pieces.
- **(6) Mostly consistent with the blocking DP** (one lone piece, no
  captures): the best single walls are on the outer rings in the approach
  corridors; against a numbered piece the value is entirely on the radial
  spoke to its goal, rising outward (+0.12 ... +0.40 turns). Against a blank,
  28 of 63 field tiles make a wall counterproductive — wall against numbered
  pieces. The DP does not model a wall's effect on several pieces at once or
  its hub value, which is owner's case for rings 5 and 3.
- **(8) Confirmed by exact DP (2026-10-10):** blanks on DISTINCT LOW goals;
  with a numbered 1 or 2, put blanks on its goal; with a 4, a mix (G1 + G4);
  with a 5 or 6, all under it. E.g. a numbered 6 + 3 blanks: G1/G2/G3 banks
  out in 3.46 turns, all on G1 3.85, all on G6 6.08. Details in CLAUDE.md.
- **(2) No measurement of blot danger exists.** The opening rollouts
  (-0.02 pts, CI -0.16..+0.11) measured how BLOCKABLE the computer left its
  numbered pieces, not capture exposure; `weakness_probe` found owner's and
  the computer's capture exposure indistinguishable, which says nothing
  about how much a blot costs.

## Untested, could be measured

- (2) blot cost by ring and piece type, from self-play.
- (5) capture-while-blocking: block-save frequency with vs without a capture
  in the previous turn, from self-play logs.
- (7) all-blank vs mixed walls: how much more often the mixed wall is broken
  or its numbered piece stranded.
