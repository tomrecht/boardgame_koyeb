# Quahuru: plan of action (written 2026-11-10)

From the 2026-11-10 review session. Numbers quoted here come from CLAUDE.md,
ARCHIVE.md and `train-features-v2`'s CLAUDE.md; nothing in this file was newly
measured. Not shipped (not in `build_web.py`'s list).

## Where things stand

- **Single-game strength is near a plateau on the current method.** Small
  distillation rounds are level on both layouts, rollout teachers (2-4 plies) are
  no better than the one-ply search labels, weight averaging and a 128-wide net
  gained little, deeper search came out null twice.
- **One open lead:** on sum7 the close-call search still beats the shipped net by
  +0.118 pts/game (95% CI +0.054..+0.182), so that teacher is not used up there.
  Round 4 (2,000-game rounds) is set up on the work iMac; no result on the branch
  yet.
- **Luck is about 88% of outcome variance; the first player is ahead by about
  +0.28 pts** (layout comparison, pass 2).
- **The sum7 net shipped without a recorded promotion checklist.** The prefilter
  scales, difficulty knots and hand-rule checks on file are for cand13 on the
  original numbering.

## Order of work

| # | Step | Effort | Depends on |
|---|---|---|---|
| A1 | Promotion checklist on the sum7 net | 1 day, GitHub | - |
| A2 | Read sum7 round 4; ship if it clears | hours | round 4 finishing |
| A3 | Total the games ever trained on | 1 hour | - |
| A4 | Parity fixtures and replay in CI | half a day | - |
| B1 | Luck-adjusted confirm matches as default | 1-2 days | - |
| B2 | Fixed sum7 panel for final checks | half a day | - |
| C1 | Exploiter run | about one distillation round | B1 |
| C2 | Luck-adjusted returns as value target | offline check 1 day, then one round | B1 |
| C3 | Transformer teacher | 1-2 weeks | C1 or C2 showing headroom |
| D | Match awareness | about a week | - (parallel) |
| E1 | Easy levels that think shallower, not noisier | 2-3 days | A1 |
| E2 | Five first-time players | owner | - |
| E3 | Android package on sum7 | hours | A1, A2 |
| F | Game-design experiments | owner's call | - |

A and B first: they are cheap and everything later is judged with them. C and D
are independent of each other. D is the item most likely to show up in play.

## A. Housekeeping

**A1. Promotion checklist on the sum7 net** (`GOAL_LAYOUT=sum7`, net
`distill_sum7/cand_13`; steps as in CLAUDE.md "PROMOTION CHECKLIST").
1. `offgoal_probe.py`.
2. `filter_probe.py`: keep-rate and deep-check cost of the prefilter. If the
   cost's CI clears 0, refit (`prefilter_fit.py extract 2` / `cv2` / `final2`),
   then `filter_ab.py` and a GitHub match with `SCALES_A` / `SCALES_B`.
   The scales are per layout from then on; `heuristic_weights.json` needs a
   per-layout `component_scale` if they differ.
3. `rule_probe.py`: the three hand rules on vs off.
4. Difficulty knots: `remote_play('difficulty', ...)` with
   `LEVELS=0.65,0.74,0.80,0.93,1.0`. Decide then whether Easiest drops to 0.60
   (it measured 9%, not the intended 4%, on cand13).
5. The close-call gauge is done (+0.118). `calib_bench` is skipped: its 559
   positions are original-layout games.

**A2. sum7 round 4.** When `sum7_r4_RESULT.txt` lands: if the final match's CI
clears 0, ship the exported `model_sum7.json` + hashed .onnx (sw.js SHELL entry,
CACHE bump, `REC_MODEL_TAG`, playout workflow asset), then re-run A1 steps 1-3
on the new net. If level, the search teacher is used up on sum7 too; C becomes
the only route to more strength.

**A3. Count the games.** Total self-play and labelled games per lineage from the
run logs. Under roughly 100,000 in all, a from-scratch run at 10-50x scale is a
real option (see C4); well above it, it is not.

**A4. CI.** A workflow on push to `main` / `testing` running `build_web.py`,
`agent_test.js`, `features_v2_test.js`, the lookahead and last-piece fixtures,
and `replay_games.py` on a small committed sample of games. The three engines
have diverged at least three times, each found by accident.

## B. Measurement

**B1. Luck-adjusted confirm matches.** `layout_play.py` already computes luck per
turn (thrown roll's value minus the mean over 21 rolls) and cut the skill CI from
+-0.089 to +-0.052, at about 120 core-seconds a game.
- Add it to `confirm_match.py` behind `LUCK=1`, plus a cheap variant whose 21
  values come from a one-ply look instead of the full agent.
- Check: replay a finished match's seeds (sum7 cand_13 vs distill_cand13, seeds
  204M, raw +0.148). The adjusted mean must sit inside the raw CI; mean luck per
  turn must be about 0.
- Adopt whichever variant gives the largest (variance cut) / (cost increase),
  if above 1.5. Otherwise keep raw scoring.

**B2. Panel.** `panel_gate.py` exists. Members for sum7: `distill_sum7` cand_7
(round 1), cand_13 (round 2), cand_22 (round 3), distill_cand13 playing sum7, and
the heuristic player. Used for final checks before shipping, not per iteration.

## C. Strength

**C1. Exploiter.** A copy of the champion trained only to beat the frozen
champion.
- Build on `league_run.py`: every generation game is exploiter vs frozen
  champion (`LEAGUE_FRAC=1.0`, panel = champion), opening exploration on, only
  the exploiter's positions trained on, close-call labels as in `distill_run.py`.
- Three or four rounds of 2,000 games on GitHub, training on the work iMac.
- Test: exploiter vs champion, 2,000 pairs, luck-adjusted.
- If it wins by a clear margin: check it on the panel. If it holds there it is
  simply a better net and ships; if it only beats the champion, mix its games
  into a champion training round and confirm the new champion against the old
  one, the exploiter and the panel.
- If it cannot win: record that as the first outside evidence the champion is
  near its ceiling, and stop spending on C except the cheap check in C2.

**C2. Luck-adjusted returns as the value target.** Today's value targets are
mostly raw outcomes (lambda 0.9). Target for the position at turn t becomes the
final margin minus the sum of luck from t onward (signed for the mover).
- Offline check first, on games that already carry per-turn luck (the layout
  comparison's pass-2 sum7 games, if the trajectories are still on the work
  iMac): SD of (target - net value) for both targets. Go on only if the
  luck-adjusted SD is under half.
- Then two arms from the same champion on the same 2,000 games: current targets
  vs luck-adjusted, same close-call labels. 2,000-pair confirm of each vs the
  parent, and head to head.

**C3. Transformer teacher.** Only if C1 or C2 shows there is headroom.
- 94 tile tokens + 24 piece tokens + 1 global token, 192-256 wide, 8 layers,
  4-6M parameters; attention bias learned from route distance between tiles;
  attention pooling; an extra output for the final-margin distribution.
  Offline only: no JS twin, no phone check.
- Train on the pooled sum7 label files, no new games.
- Test 1: held-out fit (ranking correlation, top-1) against the 64-wide net
  warm-started and trained on the same full data.
- Test 2, if it fits better: inside the same search vs the shipped net, 2,000
  pairs. Pass = it wins by more than the close-call gauge.
- If it passes, it labels games and the phone net is distilled from it with the
  existing loop.

**C4. Scale (conditional on A3).** A from-scratch afterstate run with tens of
times more games than any round so far, generation on GitHub. Not specified
further until the count is known.

## D. Match awareness

No retraining of the value net. Match formats in `game.js`: N games by total
score (default 6; ties by games won, then two more games or a draw) and race to
a target (default 21); only the winner scores, by the margin.

1. **Outcome distribution.** Winner and margin by starter from sum7 self-play
   (the layout comparison's games, or a fresh 5,000 on GitHub).
2. **Match equity table.** A DP giving P(win match) per state. Games mode:
   games left, score difference, wins difference, next starter. Race mode: both
   scores, next starter. Output `match_equity.json`, built by a Python script
   with a node check that both read the same numbers.
3. **Margin distribution per position.** Prototype: a lookup of the final-margin
   distribution by net value and pieces left, fitted on self-play, judged by
   held-out log-loss. Later, if worth it: a distribution head on the frozen
   trunk (value output unchanged).
4. **Move choice.** `local_agent.js` passes the match state; `agent.js` rescores
   the candidates the search keeps by sum over margins of P(margin) x table
   value. The endgame lookahead and the bank-the-most rule use the table with
   their exact probabilities. Outside a match nothing changes. The hint uses the
   same path.
5. **Python twin and test.** Match-aware vs plain, same net, whole matches on
   GitHub (5,000 per format), counting match wins. Pass = CI clears 0. JS = Python
   fixture as for the lookahead.
6. **Recorder.** Flag games played with it, so stats do not mix.
7. **Ship.** A major feature: stage on `testing`, then `main`.

Expected: a small overall gain, concentrated in the last game or two and in
race-to-21. Not yet measurable from anything on file.

## E. Players

**E1. Easy levels that think shallower.** Today's lower levels are the full net
with sampling noise. Replace noise at the bottom of the slider with real
handicaps: the heuristic player alone, then the fast one-stage selection (the
full agent beats it by about +1.5 pts/game), then full strength; lookahead and
hand rules off at the low end. Re-space the knots with the `difficulty` match
mode, then owner's feel. An imitation net is not planned: the only recorded human
is an expert, and beginners' games are deliberately not collected.

**E2. Five first-time players.** Owner watches five people who have never seen
the game do the tutorial and one game: where they stop, what they tap that does
nothing, whether they finish. No instruction during it.

**E3. Android package.** Rebuild on sum7 (next versionCode) once A1 and A2 are
done, as one batch.

## F. Game design (owner's call; none started)

Tested the same way: full strength vs the handicapped agent under each variant,
luck-adjusted, and see whether the skill gap widens (today about +1.5 pts).
- Dice deck: each player draws rolls from a shuffled deck of the 36
  combinations. Playable by today's net.
- Choosing which rack piece enters. Needs retraining.
- A decision in the numbered endgame (2 points for a saved numbered piece, or
  voluntary blanking at a cost). Needs a new net and tables.
- Doubling cube; longer default matches. No change to moves.
- First-player edge (+0.28 pts): matches already alternate starters; a single
  game could give the second player compensation.

## Dropped, with the reason on file

- Wider or deeper version of the same net: the 128-wide test gained little, depth
  nothing.
- Rollout teachers: two rounds, no better than search labels.
- More small distillation rounds: level on both layouts.
- Weight averaging, play-time ensembles: flat on this lineage.
- Policy head and tree search: the prefilter costs about 0.06 pts/game, and
  deeper search was null twice.
- More auxiliary heads: the July gain washed out.
- Endgame-solver labels: the loss they fix is about 0.003 pts/game.

## Decisions needed from owner

1. Easiest level: keep at 9% or lower the knot (after A1 step 4).
2. Whether to spend on C at all if the exploiter fails.
3. Whether E1 replaces the current slider behaviour or sits beside it.
4. Which, if any, of F to try.
