# Exploration / Coverage Spec — raising the strength ceiling past greedy self-play

Status: design, not yet implemented. To be run as its own experiment *after*
the TD(λ) run has produced a few corrected-feature iterations (do not fold
into the live TD run — it would confound the clean "what does TD alone do?"
question). Nothing here touches `td_selfplay_loop.py` until the current run
is done; build on a branch.

## Problem this addresses

Self-play generation is **fully greedy** — move selection is 2-ply `argmax`
over the value net (verified: no temperature, softmax, multinomial, epsilon,
or root noise in `agent_gnn.py` / `td_selfplay_loop.py`); the only
stochasticity in generation is the dice seeds. Consequence: the policy never
deliberately tries a move it currently undervalues, so any strong strategy
the net doesn't already rate highly is never generated, never trained on,
never discovered — a self-reinforcing coverage collapse (the policy's own
success narrows its state distribution). The 2&4-vs-1&6/3&5 goal-pair gap is
the one *visible* instance (symmetry exposes it); others without a symmetric
twin are produced by the same mechanism but are invisible to us.

This is **orthogonal to TD**. TD is credit assignment — learn the value of
*visited* states efficiently. Exploration is *coverage* — which states get
visited at all. TD polishes inside the box greedy self-play has found; it
cannot enlarge the box. Part of the standing ~55-60% gap to the human tester
is plausibly *undiscovered strategy*, not *mis-valued known strategy*, and
better credit assignment cannot reach that part.

## Key trap: value-guided exploration can't crack a severely-undervalued gap

Temperature/softmax over the net's own values only explores among moves the
net *already* rates highly. The 1&6/3&5 blocks are *undervalued* — that's the
problem — so softmax essentially never picks them. Value-guided exploration
rediscovers things near what the net already likes; it cannot reach a strong
strategy the net rates as bad. Therefore the effective tools are
**value-agnostic**: uniform-floor exploration, seeded starts, or (for known
invariances) augmentation.

## Interventions (in recommended order)

### 1. General stochastic exploration in generation — the primary engine

ε-greedy with a **uniform floor** over legal move-pairs, applied at **all
stages of the game** (NOT opening-only — the goal-pair block is a *midgame*
construction; opening-only exploration would never even attempt it), annealed
across iterations.

- ε-random pick samples **uniformly over legal move-pairs** (not top-k —
  uniform is the point; blunders are cheap and their outcomes are still
  informative). Both self-play sides explore (both sides' positions become
  training data).
- Suggested start: ε ≈ 0.15-0.25, held for the first ~5 iterations, then
  annealed toward 0 so late iterations produce clean near-greedy data for
  final value calibration. Tune against the diagnostic below.
- Finds *unknown* strategies (its unique value); slower/higher-variance for
  narrow, delayed-payoff targets.

**TD-correctness under exploration.** Exploratory moves make a naive λ-return
reflect the *behavior* policy (greedy+noise), biasing V slightly pessimistic.
Two versions:
- *Simple first cut:* modest ε everywhere, don't cut traces, accept the small
  bias (small if ε is small/annealed).
- *Correct version:* **cut the λ-trace at exploratory moves** (Watkins-style)
  — when a move was exploratory, truncate the λ-return there and bootstrap
  from V at that state instead of continuing the return through the random
  action. Keeps every target consistent with *greedy* continuation while
  exploring at every stage. Requires tagging each move greedy/exploratory and
  truncating the λ-return accordingly.
Start simple; add trace-cutting if the pessimistic bias shows up.

### 2. Seeded / synthetic "exploring starts" — the targeted, efficient tool

Construct positions in the known-desirable region (e.g. 1&6 or 3&5 already
blocked, or one move from it) and **seed self-play rollouts from them** —
letting the real outcome / λ-return supply the label. Do **not** hand-stamp
values (that hand-crafts the value function and teaches your errors). The net
learns the block's value from experience; you've only changed *where
trajectories begin*.

- Far more sample-efficient than blind ε-greedy for a narrow known target:
  you start *at* the block instead of hoping to stumble into it over many
  random moves against the coverage-collapse headwind.
- Doesn't need a proven automorphism (unlike §3) — only the block geometry
  (which tiles seal which goals).
- Generalizes: seed from *any* hypothesized position to teach a suspected new
  strategy, not just symmetric ones.
- Mechanism for fixing behavior: once V rates blocked states high, 2-ply
  greedy starts valuing moves that *head toward* them (value propagates back
  via generalization + TD bootstrap) → it constructs the blocks in real play.

Caveats: (a) **realism** — prefer positions built by locally transforming
*real* 2&4-block games into 1&6 blocks over fully manufactured ones, to stay
on-distribution; (b) **mix, don't replace** — blend seeded-start games into
normal self-play so V stays calibrated on the real distribution; (c) **clean
labels require the current net to convert the block in rollout** (usually
fine if the block is strong; muddy conversion → muddy label is the failure to
watch).

### 3. Symmetry augmentation — optional fallback for the one named gap

For each recorded `(position, target)`, also emit `(g(position), target)` for
the goal-pair symmetry generators `g1: 2&4↔1&6`, `g2: 2&4↔3&5`. Copies the
*already-learned* 2&4-block value onto its mirrors — instant and certain, no
stochastic rediscovery.

- Demoted to fallback: it only works for *true known symmetries* (cannot find
  unknowns), and it is the most setup-heavy (needs a proven board
  automorphism — the exact tile-index permutation — plus validation).
- **Validation (mandatory):** apply g to every position of a played game,
  verify the transformed game is legal and yields identical outcomes/values
  (same trace-diff method used elsewhere). A wrong automorphism silently
  teaches wrong values.
- Build only if §1+§2 haven't closed the goal-pair gap after a few
  iterations — then it's a targeted rescue, not upfront work.

## Scoping rules (non-negotiable)

- **Generation only. Eval stays fully greedy** — the promotion gate must
  measure the deployed (greedy) policy or the 0.55 ratchet stops meaning
  anything.
- **2-ply internals stay greedy.** Only the *actual move played* in self-play
  is sampled/seeded; the opponent-reply modeling *inside* the 2-ply value
  estimate remains argmax (it's a value computation, not a played move).
- **Forced-move short-circuit** (single candidate) already needs no sampling.

## Measurement

1. **Block-frequency diagnostic (fast, falsifiable, run first).** Instrument
   generation to count per game whether an absolute block of each pair
   {2&4, 1&6, 3&5} occurred. Baseline is known: 2&4 frequent, other two ≈ 0.
   Success = the other two rise toward 2&4's rate. Confirms an intervention
   works from a *single* generation batch, before any full strength eval, and
   is tied to a gap where we *know* the correct answer (all three equal).
2. **Strength (the verdict).** Fork from a fixed checkpoint (post-TD
   champion). Track A = continue greedy (control); Track B = intervention.
   Run K iterations each, then **greedy-vs-greedy** paired-seed eval (both
   eval greedy — testing which *training* produced the stronger deployed
   policy). Winner by win rate.

## Implementation plan

- New branch. Add an `explore_cfg` param (default `None` = greedy = current
  behavior *exactly*), plumbed `run_td_selfplay` → generation → move
  selection. Eval path hard-forces `None`.
- **Safety gate:** with `explore_cfg=None`, prove byte-identical to current
  via the 6-game trace-diff — the default path must be untouched.
- Then the block-frequency smoke test before committing to a full run.
- Sampling/seeding is ~free per move, so generation speed is unchanged; the
  only added cost is the comparison run(s).

## Inputs needed from owner (Tom)

- Board **block geometry** (which tiles seal which goals) — for §2, and for
  the diagnostic instrumentation.
- Board **automorphism** (goal-pair tile permutation) — only if §3 is built.
- Sensible per-game-phase ε shape, if the flat ε turns out too blunt.
