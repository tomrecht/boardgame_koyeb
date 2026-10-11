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

### Walls
4. **Block goal pairs.**
   - **With a pair blocked, capture an opponent piece every turn you can**: a
     captured piece must re-enter first, so they can't spend the dice on
     block-saving your wall.
   - **When they have walled a pair against your numbered piece, don't
     block-save it while your dice have anything else useful to do.** They
     will eventually run out of moves and have to dissolve the wall or pass.
5. **Where to wall: rings 5 and 6 are best**, especially 5 (those tiles are
   more of a hub), **then ring 3.**
6. **All-blank walls beat walls containing a numbered piece.** If you have to
   build a wall with a numbered piece, try to swap it out for a blank later.

### Endgame
7. **Spread blanks among the goals, favouring the lower goals**, and **use a
   4 to drop a blank from a high goal to its low partner** (6->1, 5->3, 4->2
   are 4 steps apart), where it banks more cheaply.

### Captures
8. **A goal usually beats a capture**: over 330 positions taking the goal did
   +0.16 pts better on average. Capture when it's cheap, or to hold a pair
   block (tip 4).

Rejected: "keep a numbered piece for last" -- with a numbered piece and a
blank on the same goal, banking either leaves the same position (the
last-piece rule makes the lone survivor a blank), so save the numbered piece
whenever you can.

## Illustrations planned

Rendered with `pos_image.mjs` from positions in the game's notation; owner
vets a contact sheet before anything ships: a numbered piece parked behind a
wall (3), a goal-pair block (4), waiting out a wall (4), the good wall rings
5 / 6 / 3 (5), blank spreading plus the 4-step hop (7). Tips 1, 2, 6, 8 as
text.

## What the measurements say about them

- **(1) Numbered pieces are the endgame bottleneck:** 3.27 turns to bank from
  their own goal (exact match, 11/36 a turn) against 1.00-1.64 for a blank.
  The net puts a numbered piece on its goal over a blank save in 871 of 1,002
  positions.
- **(4) Two walls seal a whole 4-apart pair** (6&1, 5&3, 4&2): the pair shares
  its approach corridors, so denying a pair costs no more than denying one
  goal. Sealing all six would take 12 pieces.
- **(5) Mostly consistent with the blocking DP** (one lone piece, no
  captures): the best single walls are on the outer rings in the approach
  corridors; against a numbered piece the value is entirely on the radial
  spoke to its goal, rising outward (+0.12 ... +0.40 turns). Against a blank,
  28 of 63 field tiles make a wall counterproductive — wall against numbered
  pieces. The DP does not model a wall's effect on several pieces at once or
  its hub value, which is owner's case for rings 5 and 3.
- **(7) Confirmed by exact DP (2026-10-10):** blanks on DISTINCT LOW goals;
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
- (4) capture-while-blocking: block-save frequency with vs without a capture
  in the previous turn, from self-play logs.
- (6) all-blank vs mixed walls: how much more often the mixed wall is broken
  or its numbered piece stranded.
