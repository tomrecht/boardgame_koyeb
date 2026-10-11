# Quahuru strategy — notes toward a player guide

Raw material for a strategy guide. Not shipped. Owner's tips are the
backbone; measured facts (CLAUDE.md "Domain facts", ARCHIVE.md) are noted
where they support or qualify a tip.

## Owner's tips (2026-10-10)

1. **Don't leave blots** — especially numbered blots, and especially on the
   outer rings.
2. **Get numbered pieces to their goal early**, or at least within one die of
   it — unless one is holding a wall, especially a wall against a
   higher-numbered piece.
3. **Park numbered pieces behind a wall** so they can't be captured.
4. **Block goal pairs.**
5. **Spread blanks among the goals, favouring the lower goals.**
6. **All-blank walls beat walls containing a numbered piece.** If you have to
   build a wall with a numbered piece, try to swap it out for a blank later.
7. **With a goal pair blocked, capture an opponent piece every turn you can**:
   a captured piece must re-enter first, so they can't spend the dice on
   block-saving your wall.
8. **Walls on rings 5 and 6 are best**, especially 5: those tiles are more of
   a hub.

## What the measurements say about them

- **(2) Numbered pieces are the endgame bottleneck:** 3.27 turns to bank from
  their own goal (exact match, 11/36 a turn) against 1.00-1.64 for a blank.
  The net puts a numbered piece on its goal over a blank save in 871 of 1,002
  positions.
- **(4) Two walls seal a whole 4-apart pair** (6&1, 5&3, 4&2): the pair shares
  its approach corridors, so denying a pair costs no more than denying one
  goal. Sealing all six would take 12 pieces.
- **(5) Confirmed by exact DP (2026-10-10):** blanks on DISTINCT LOW goals;
  with a numbered 1 or 2, put blanks on its goal; with a 4, a mix (G1 + G4);
  with a 5 or 6, all under it. E.g. a numbered 6 + 3 blanks: G1/G2/G3 banks
  out in 3.46 turns, all on G1 3.85, all on G6 6.08. Details in CLAUDE.md.
- **(8) Consistent with the blocking DP:** the best single walls are all on
  rings 5-7 in the approach corridors; the inner board is worth nothing to
  wall (too many routes near the hub). Against a numbered piece the value is
  entirely on the radial spoke to its goal, rising outward (+0.12 ... +0.40
  turns). Against a blank, 28 of 63 field tiles make a wall
  counterproductive — wall against numbered pieces.
- **Captures vs goals:** over 330 positions, taking the goal beat the capture
  by +0.16 pts on average — capture when it's cheap, not instead of progress.
- **Opening exposure** cost less than expected in rollouts (-0.02 pts, CI
  -0.16..+0.11) — tip (1) may matter more later in the game than in the
  opening. Worth testing.

## Untested, could be measured

- (6) all-blank vs mixed walls: how much more often the mixed wall is broken
  or its numbered piece stranded.
- (7) capture-while-blocking: block-save frequency with vs without a capture
  in the previous turn, from self-play logs.
- (1) blot cost by ring and piece type, from self-play.
