# Board Game GNN — TD(λ) Training Project

**Companion file: `ARCHIVE.md`** — the settled history this file used to carry
(the whole TD(λ) training programme, deep search, the arena measurements, and
the July–August 2026 frontend and port history). Verbatim, nothing deleted.
Read it for the reasoning behind a settled decision or a measured number; this
file keeps what a fresh session needs to act.

Context for Claude Code picking up this project. Owner (Tom) prefers terse
responses. Tom is the domain expert on the boardgame, you are the ML expert; make suggestions on training accordingly, but you may also suggest potential improvements to game rules and mechanics when relevant.

**KEEP THIS FILE CURRENT, BUT IN PROPORTION (owner, 2026-08-04, amended
2026-09-20).** Update CLAUDE.md as part of the work, not as a favour asked for
afterwards. Anything a fresh session would have to rediscover belongs here: a
rule change, a deployed model, a config the app depends on, a measured number
that settles a question, a bug whose cause was non-obvious, a decision and why.
Land it in the same commit as the change where practical. Prune too: when a note
is superseded, replace it rather than stacking a contradiction on top. Numbers
beat adjectives — record what was measured, over what sample, and what it rules
out.
**But not every tiny change needs an entry** (owner, 2026-09-20): a cosmetic
tweak, a label, a sound does not. Settled history belongs in **`ARCHIVE.md`**,
not here — this file had reached 4,273 lines and the re-reading cost was real.

**VERIFY IN PROPORTION TO RISK (owner, 2026-09-20).** The "measured, not
assumed" standard below earned its place on rules and input bugs; it earns
nothing on a chime.
  * **Drive it in a browser** for anything touching move legality, reachability,
    input handling, turn state, the agent, or layout geometry.
  * **Ship on the code path** — read it, reason it through, commit — for sounds,
    copy, colours, and anything that cannot change what the game accepts as a
    move.
  * Owner's words: *"Not every tiny change needs to be verified and
    documented."* Three sessions were lost to theorising without data, so the
    standard is not gone — it is targeted.

**BATCH THE CHERRY-PICKS TO `testing` (owner, 2026-09-20).** Fixes still reach
both live branches, but do the checkout / cherry-pick / push once at the END of
a session rather than after every fix — see "TWO DEPLOY TARGETS" below.

**WORKSPACE CONSTRAINT (owner, 2026-07-21): Ownder works on two machines, iMac and MacBook. 
Work only inside this folder (`/Users/tomrecht/game/boardgame_koyeb` on MacBook, `/Users/tom/Game/BoardGame` on iMac).** Reads/writes outside it trigger permission prompts. Do NOT create git worktrees or files in sibling dirs; run new experiments from a branch checked out *in this folder* (or keep files here).
When you must inspect another branch's file, use `git show <branch>:<path>`;
when you must read an out-of-folder artifact (e.g. a running worktree's log),
`cp` it in first. (Some existing runs — e.g. the symmetry-aug worktree — predate
this rule; monitor them by copying their logs in.) The symmetry-aug work is now
on origin as `symmetry-aug` / `symmetry-aug-main`, so it no longer lives only on
the MacBook; its checkpoints `symaug_champ_July27_iter6.pt` and
`symaug_almostchamp_July27_iter11.pt` are in this folder (iter6, the promoted
champion, is the one deployed).

**OVERNIGHT / AUTONOMOUS WORK RULES (owner, 2026-07-23, after a failure).**
The agent only acts *while a turn is running* (issuing tool calls). The moment
it ends a turn with a text message, it goes IDLE until the next user message or
a scheduled/background event — there is NO background execution otherwise.
A "continuing meanwhile…" sign-off followed by ending the turn = doing nothing.
So, for any "work through the night / while I sleep" task:
  1. NEVER promise to "keep going" and then end the turn. If work remains, keep
     issuing tool calls until it is actually done or at a genuine checkpoint.
  2. To truly run unattended, set up REAL continuation: launch long jobs as
     tracked background processes (`run_in_background`) AND schedule a self-
     wakeup (ScheduleWakeup / CronCreate) so the agent re-enters and continues;
     verify the wakeup actually fires. Chain wakeups until the job is complete.
  3. During an unattended window, do the DELEGATED autonomous task first (it is
     the only thing that can't happen while the owner is asleep). Don't let
     interactive polish requests displace it; run them in parallel via
     background if needed.
  4. Checkpoint durably every step (commit, update CLAUDE.md/memory) so progress
     survives an interruption and is visible on waking.

## Project background

Custom hub-and-spoke dice board game. Full stack built from scratch: game
engine (`game.py`), Flask backend (`app.py`), Phaser.js frontend (`game.js`).
Current AI approach: GNN value network (`network.py`) driving 2-ply move
selection (`agent_gnn.py`), trained via self-play. Strongest checkpoint before TD(λ): `best_iter5_m46.pt`.

An earlier heuristic evolutionary track (`train.py`) reached 90%+ vs.
baseline but was recognized as fundamentally limited without a neural
approach and is no longer the active line of work. The heuristic agent is still used to filter top-K moves for the neural network, for speed.


## Domain facts worth knowing (distilled; full analysis in ARCHIVE.md)

Board geometry and endgame arithmetic that has been measured and
engine-verified. These keep coming up, so they live here rather than in the
archive; the derivations, the verification counts and the caveats are in
ARCHIVE.md under "Current benchmark".

- **Every goal is exactly 7 from the home tile.** So an entering piece reaches a
  goal only on a dice sum of exactly 7 (6 rolls in 36), never on one die, and at
  sum 7 *all six* goals are legal destinations at once.
- **Goals are paired 4 APART — 6&1, 5&3, 4&2** (the other gaps are 7, 11, 14).
  This is why a lone piece can step from a high goal to its cheap partner, and
  why two walls can seal a whole pair.
- **Landing on a goal makes the stage `endgame` within the same turn**, so "step
  onto a goal with one die, bank with the other" is legal in one turn.
- **Expected turns to bank a lone blank, by goal:** 1.000 / 1.029 / 1.125 /
  1.303 / 1.462 / 1.644 for goals 1-6. **A numbered piece takes 3.273 turns from
  its own goal, the same for every number** — the endgame higher-die rule is
  blank-only, so it needs an exact match (11/36 a turn). Numbered pieces are the
  endgame bottleneck.
- **Stack low and extras ride out free; stack level and they queue.** A blank on
  goal 1 under a numbered-2 on goal 2 costs nothing (3.273 either way); the same
  blank *on* goal 2 costs 0.059, and two of them 0.353.
- **A goal tile can never be blocked** (`is_blocked` requires `type == 'field'`)
  — only the routes to it. Sealing a goal from home takes 2 walls (4 pieces),
  and those same 2 walls seal its whole 4-apart pair.
- **Walls are worth ~3x more against a numbered piece** (best +0.398 turns, all
  of it on the outer end of the radial spoke to that piece's goal) than against
  a blank (+0.128). Against a blank, **28 of 63 field tiles are
  counterproductive** — a wall can shorten their route into die range.
- **Endgame offgoaling is NOT a defect** — it tracks the banking odds exactly.
  The agent never leaves goals 1-3 (where the value gradient is steep) and
  sometimes leaves 4-6 (where standing on goal 6 is worth 0.039 turns more than
  standing beside it).

## AI and model — current status

- **Deployed model: `model.onnx` = `symaug_champ_July27_iter6.pt`**, the
  symmetry-aug run's promoted champion; owner reports it as his strongest
  opponent. Re-export with `onnx_export.py <ckpt> model.onnx`, which
  self-verifies.
- **Inference runs on the device.** `local_agent.js` + the ported stack
  (`route.js`, `encoder.js`, `infer.js`, `engine.js`, `heuristic.js`,
  `agent.js`) answer every move; there is no application server in play. The
  port was proven by trace-diff, 110/110 chosen pairs. `PORTING.md` has the
  account.
- **Training is not the active line of work.** The TD(λ) programme, the arena
  measurements, the aux-head and symmetry-aug runs, deep search and the
  interpretability probe are all in **ARCHIVE.md**. The headline that still
  matters if training resumes: promotions are correctly measured but **~76% of
  the certified local gain does not generalise** (non-transitivity from
  parent-only gating), and the fix is a fixed-panel gate.
- `?dev=1` unlocks debug / eval / setup modes and un-silences `console.log`.
  **Any harness that reads console output must pass it.**

## Traps that keep biting

Method notes that have each cost a session or more. The full stories are in
ARCHIVE.md; these are the rules that came out of them.

- **A measured MECHANISM is not a defect until the SYMPTOM is observed.** "The
  UI freezes" was written up twice from a profile alone and disproved both times
  by owner simply looking. Symmetrically, a felt symptom is only fixed when it
  stops being felt.
- **Believe the owner's observation over the instrument.** A harness artefact
  once "measured" the AI forfeiting half its turns.
- **Read the denominator, not the zero.** A "0 of 0" reads exactly like a pass.
  Check that a fixture actually exercises the branch you think it covers.
- **Build test positions by playing into them, or check them by hand.** A
  hand-mutated board has stale derived state — `Player.getGamePhase()` is a
  cached field, `mustMovePieces` has two writers, `tile.addPiece` does NOT set
  `piece.currentTile` (only `piece.move` does). Any of these makes a correct
  result look like a bug.
- **Derived state beats a flag** whenever another code path already mutates what
  the flag summarises.
- **Test the gate, not the legality.** A test of a toggle must ask whether the
  branch was reached, not whether the outcome was legal.
- **A short-circuited reference is invisible to a test that never enters the
  branch.** Check the branch, not the function.
- **Grep the file after any scripted edit, and never let a test enable the thing
  it tests.** Both have produced confident false passes.
- **Never pipe a build into `tail` and trust the exit code** — it reports
  `tail`'s.
- **Browser-harness specifics:** `page.evaluate` runs in an ISOLATED world and
  cannot see game.js's globals — inject with `page.addScriptTag` and pass
  results back through a DOM attribute. `_currentGame` is a FUNCTION (reading it
  as an object silently yields `{}`). Pieces live on the GAME (`game.pieces`,
  24), tiles are `game.tiles` (94). Wait for `_gameFrozen === false`, not just
  for the welcome card to go. Set **both sides human before the game starts** —
  a tap on the computer's turn reads as a dead tap, and flipping a role mid-game
  races two agent requests. `camera.worldView` is all zeros until a frame
  renders. A healthy boot: 94 tiles, 24 pieces, `renderer.type === 2`, one
  active scene, 0 failed requests; the Canvas2D `willReadFrequently` warnings
  are the board bake's readbacks and are expected.
- **HARNESS: DRIVE SYSTEM CHROME, NOT A DOWNLOADED CHROMIUM (fixed 2026-09-24).**
  The browser-automation skill resolves `patchright` from the newest CodeGPT
  extension and then looks for that patchright's own chromium build. **That build
  cannot be installed here** -- patchright 1.63 dropped macOS 13 support and this
  iMac is Darwin 22.6 (Ventura); the 3.24.69 extension that had `chromium-1193`
  on disk is **gone**, so the workaround recorded here on 2026-09-20 is dead.
  **What works: point patchright at the system Google Chrome.** No download, no
  extension archaeology:

        const req = createRequire('/Users/tom/.vscode/extensions/danielsanmedium.dscodegpt-3.24.74/standalone/')
        const { chromium } = req('patchright')
        await chromium.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
          headless: true, args: ['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'] })

  The swiftshader args are what get **WebGL** in headless, so `renderer.type === 2`
  and the board actually bakes; without them Phaser falls back and the geometry
  measurements are not the ones the player sees.
  **PATCHRIGHT RELAYS NO PAGE CONSOLE AT ALL.** It suppresses `Runtime.enable` to
  stay undetectable, so `page.on('console')` sees browser-level warnings (the
  Canvas2D readback ones) and **not a single `console.log`, `warn` or `error` from
  the page** -- verified against a bare `data:` URL: zero messages of any type.
  So `?dev=1` is still right and still necessary for a human reading the console,
  but **a harness must assert through the DOM**, never by reading log output. A
  test that "found no errors" in the console found nothing at all.

## Project direction: a real Android/iOS app (owner, 2026-08-14)

Standing goal — the web client should eventually ship as an actual app, with
possible store presence. Not being built yet; **future changes should be made
with it in mind.** Assessment and the concrete implications:

- **Packaging is the cheap part.** A PWA (manifest + service worker) gets a
  home-screen icon, no browser chrome and offline loading in hours, no store
  needed. A Capacitor wrapper adds real store presence in about a day plus store
  admin (Play $25 once; Apple $99/yr and review) and changes no game code — the
  WebView is Chrome on Android, WKWebView on iOS, so rendering is what we have.
- **The real work is on-device AI**, which is what removes the server, the cold
  starts, the 504s and MOVE_BUDGET juggling entirely. `model.onnx` (1.66 MB)
  runs as-is under **onnxruntime-web** (WASM); what needs porting to JS is
  `encoder.py` (pure numpy, mechanical) and `agent_gnn.py`'s 2-ply search +
  heuristic prefilter, plus whatever of `game.py` the search touches (game.js
  already has move generation/validation for human play). Days to a couple of
  weeks; the risk is correctness, so verify with the established standard —
  seeded trace-diff (`PYTHONHASHSEED=0`) asserting identical move traces against
  the Python agent. A native rewrite (Kotlin/Swift/Flutter) is ruled out: it
  would discard a polished Phaser UI to solve a problem we do not have.
- **DONE (2026-08-14): Phaser is served locally** (`phaser.min.js`, 1.04 MB, the
  same 3.55.2 build the CDN served). Measured: zero CDN requests, game boots,
  play unaffected. This was the app prerequisite AND the fix for the opaque
  `Script error.` traces — Phaser exceptions now carry a real stack.
- **DONE (2026-08-14): web-app manifest, icons, fullscreen toggle.**
  `manifest.json` (display `fullscreen` — see the edge-swipe note below for
  why not `standalone`; orientation `any`, theme `#b5623b`) plus
  `icon-192/512/512-maskable.png` (generated by a PIL script — board motif),
  a `<title>`, apple-* meta and apple-touch-icon, so Add to Home Screen launches
  chromeless on both platforms. Settings gains a **Fullscreen** row, phone-only
  and only where `document.fullscreenEnabled` (absent on iPhone Safari — there
  the manifest route IS the equivalent). Fullscreen needs a user gesture, so a
  saved preference is applied on the first pointerdown after load, not at
  start-up, and a `fullscreenchange` listener keeps the checkbox honest when the
  OS exits it. Measured: manifest + all three icons 200, row present on phone /
  absent on desktop, toggle calls requestFullscreen and persists the setting.
  **The app is named QUAHURU** and every shipped surface already carries it:
  `manifest.json`'s `name` and `short_name`, `<title>`, and the
  apple-mobile-web-app-title meta. (Confirmed 2026-08-15 — the earlier note here
  called it an unchosen placeholder, which was stale.)
  **Not done: a service worker** for true offline. Deliberately deferred — with
  deploys this frequent, a cache-first worker would serve a stale game.js, the
  exact class of bug that costs an afternoon. Add it with the packaging work,
  network-first for game.js/index.html.
- **Consequences for work being done now:**
  1. ~~Phaser is CDN-loaded~~ — done, see above; keep it local.
  2. **Zoom/pan belongs inside Phaser** (camera), not on browser `touch-action` /
     `visualViewport`. Already decided after the selection bug; the app goal is a
     second, independent reason — a WebView arbitrates pinch-zoom differently
     and often not at all.
  3. **Portrait layout is a prerequisite**, not a nicety: phone apps are expected
     to work held upright. It is last in the queue but gates the app work.
  4. **Don't add server-only game behaviour** without remembering it needs a JS
     twin later. Settings already live in localStorage, which a WebView keeps.

## Current state

- **THE DIFFICULTY SLIDER IS REMAPPED ONTO 0.8..1.0, BECAUSE THE BOTTOM OF ITS
  RANGE WAS NOT A DIFFICULTY SETTING AT ALL (measured 2026-09-25,
  `difficulty_arena.py`, 566 games).** The deployed champion played BOTH sides,
  one at difficulty d and the other always at 1.0, over paired colour-swapped
  seeds, plain 2-ply shallow. `d=1.0` was in the sweep as the control.

        d   slider  games  win%   (95% CI)   avg margin  shutouts  turns
       1.00   Max      96  50.0  (40-60)       +0.00        0%      56
       0.80   80%      94   0.0  ( 0-3.9)      -6.06        0%      61
       0.60   60%      94   0.0  ( 0-3.9)     -10.06       15%      56
       0.40   40%      94   0.0  ( 0-3.9)     -11.27       53%      53
       0.20   20%      94   0.0  ( 0-3.9)     -11.68       72%      51
       0.00  Easy      94   0.0  ( 0-3.9)     -11.89       90%      49

  **The control is clean, so the rest is trustworthy:** d=1.0 against itself gives
  exactly 50.0% (48/96), the colour-swapped halves are 26/48 and 22/48 at margin
  +0.21 / -0.21, and white takes 52 of 96. The harness is symmetric.
  **Even 80% -- the mildest weakening the slider offered -- never won a game** (0
  of 94; its BEST result in 94 was losing by 2), and at 40% and below it is SHUT
  OUT (beaten by the maximum margin of 12) in 53%, 72% and 90% of games. Every
  setting from 0.0 to 0.8 has the same 0% win rate, so **four fifths of the
  control's travel was undifferentiated and the "50%" on its label meant nothing.**
  Cause: at d=0.8 the ramps give temp 0.92 and top_p 0.40, and because the softmax
  in `_pick_move_index` is z-scored by the CANDIDATE SPREAD, a temperature near one
  std flattens it relative to the scores -- so the agent deviates from its best
  move on a large fraction of turns, which over ~56 turns is dozens of blunders.
  The ramps (`0.4 + (1-d)*2.6`, `0.25 + (1-d)*0.75`, mirrored in `agent.js`'s
  `pickMoveIndex`) are scaled wrong for this game's length.
  **THE FIX, and the two quantities it introduces.** `DIFFICULTY_FLOOR = 0.8`, and
  `getAIDifficulty()` now returns `0.8 + 0.2 * position`. Do not confuse them:
    * `getDifficultySetting()` -- where the SLIDER sits, 0..1, and what is
      persisted in `localStorage.aiDifficulty`;
    * `getAIDifficulty()` -- the EFFECTIVE number handed to the agent.
  **A saved setting is REINTERPRETED, not migrated.** `aiDifficulty` used to hold
  the effective value, so a player who had saved 0.5 is now read as position 0.5 ->
  effective 0.9 and their opponent gets stronger. Deliberate: every saved value
  below 0.8 was a setting that could not play the game, so there is no old position
  worth preserving. The default is unchanged -- no stored value means position 1.0
  means full strength.
  **THE LABEL IS ORDINAL WORDS AND DELIBERATELY CARRIES NO PERCENTAGE**
  (`difficultyLabel`): Max / Strong / Medium / Gentle / Easiest. A figure on that
  label reads as a win rate, and **nobody has measured inside 0.8..1.0** -- only
  the endpoints. The ORDERING is safe to claim, because both ramps move
  monotonically with the value; a number is not.
  **STILL TO DO: the fine sweep of the band** (0.99 / 0.97 / 0.95 / 0.92 / 0.90 /
  0.85), which is where the transition from 50% to 0% happens. Deferred at owner's
  request -- it ties up his machine. Those games are cheaper than the ones already
  run (178-267s against 300-345s at the bottom of the range), so ~2.5 h per pass of
  ten games per level. Put numbers on the label once it exists.
  **Cost of the run already done:** ~216s per game of throughput on 3 workers of
  this 4-core iMac, so ~3.6 h per pass of ten games per level across six levels.
  `difficulty_arena.jsonl` holds every game and the script skips what is already
  recorded, so a longer run resumes -- but **that file is GITIGNORED** (`*.jsonl`,
  the same convention as `arena.jsonl`), so the raw games are local to this iMac
  and the table above is the record.
  **A CLAIM MADE FROM WALL CLOCK AND DISPROVED BY THE TURNS COLUMN:** the first six
  games suggested weakened agents "play 2.3-2.5x longer games", inferred from
  seconds. Turns say otherwise -- 56 / 61 / 56 / 53 / 51 / 49, flat and if anything
  DECREASING. The inflation is **seconds per turn** (3.2 -> 6.6): the losing side
  banks nothing, so it keeps ~12 pieces on the board all game, its move generation
  is maximal every turn and the 2-ply search costs twice as much. Seconds are not a
  proxy for game length when the branching factor is what changed.
  **CAVEAT ON WHAT THIS MEASURES:** strength relative to the net at full strength,
  NOT relative to a human. Owner BEATS d=1.0 ~50-60%, so full strength is roughly
  an expert's equal -- which is why it crushes beginners, the complaint that
  started this. Where d=0.8 sits for a beginner is still unmeasured and needs a
  human, not an arena.

- **LEARNABILITY WORK, ON `testing` FOR TRIAL (owner, 2026-09-24).** Two testers
  reported the game still hard to learn after the tutorial, one asking for a hint
  mode. Four changes, all on `testing` and NOT yet on main, at owner's request.
  **1. A HINT PILL ("💡 Hint", bottom right beside the "?" legend).** Tapping it asks the SAME
  on-device agent that plays the computer's side what it would do in your
  position, and rings the piece and the destination tile. It is presentation, not
  new search -- `local_agent.js` has answered `selectMoves()` since the port, so a
  hint costs one inference batch, the same as one of the computer's turns.
  **Always at full strength**, whatever the difficulty slider says: a top-p
  sampled hint would sometimes recommend a move the agent itself ranks worse.
  **ONE MOVE AT A TIME, and that is not laziness** -- the agent picks a PAIR whose
  second half is chosen against the board as it stands AFTER the first, so that
  destination is frequently not a destination yet and marking both would ring a
  tile the player cannot legally tap. Default ON (`hintsEnabled`), with a Settings
  row; a lamp you never tap costs nothing, so there is no first-game-only state
  machine.
  **Two correctness traps, both real:**
  `getGameState` posts `reachableBySum` on a piece, and `engineState` reads that
  as the marker for "this piece already moved this turn" (the engine's
  `first_move`) -- but the frontend sets it on the SELECTED piece too. Harmless
  while only the computer's own turn asked (nothing is selected then), WRONG for a
  hint asked mid-turn on a human's. `_hintGameState` strips it and passes
  `gs.firstMove` explicitly, carrying the ORIGIN tile, which the marker cannot:
  it reports the piece's CURRENT tile. (`engineState` gained 13 lines for this and
  the old marker path is untouched, so the computer's own turns are unchanged.)
  And **the engine models the rack-entry obligation as applying to the turn's
  FIRST move only, while game.js also enforces it on the second** -- a difference
  the port never exercised, since the computer is only ever asked at the start of
  its turn. So `_renderHint` asks the LIVE game whether the half it is about to
  mark is playable (`_hintMoveIsLegalNow`) and **refuses rather than falling back
  to marking it anyway**: a hint pointing at an illegal move is worse than none.
  The marker is cleared by a 250ms poll against a signature (turn + dice + every
  piece's tile) rather than by hooks in movePiece / undo / switchTurn / the picker
  -- one place to be right instead of six to remember.
  **2. "why can't I move there?" NOW ANSWERS ITSELF.** The shortest-route rule is
  the one rule that contradicts what a player can see: a piece always travels its
  shortest route, so a longer path they have traced by hand is not a move and the
  tile just refuses the tap. `_noticeWhyUnreachable` (called from the ordinary
  refusal in `_noticeIfRouteWithheld`) **names the number** -- "That tile is 7
  steps away by the shortest route, so it takes a 7" -- which is what turns an
  arbitrary-feeling refusal into a rule. It also covers a wall on the tile, no
  route at all, and the mid-turn no-doubling-back half of the same rule.
  **Deliberately narrow:** it stays SILENT when the dice CAN make the distance,
  because then some other rule refused it and this function has nothing true to
  say -- a confident wrong explanation teaches a rule that does not exist.
  **3. THE TUTORIAL IS REACHABLE FROM HOW TO PLAY**, leading the panel above the
  first section. It was previously only on the welcome card (gone the moment you
  start playing) and in Settings (where nobody looks for a tutorial). It
  **confirms first when a game is in progress** -- the tutorial scripts positions
  into the live game and ends by restarting the scene to the welcome card, so
  launching it over a game throws that game away. "In progress" is asked of the
  board (anything entered or banked), because the frontend `Game` has no move
  history -- that lives on the ported engine.
  **4. THE CLOSING PANEL ASKS THE DIFFICULTY QUESTION AS TWO BUTTONS**, "Go easy"
  (sets `TUT_EASY_DIFFICULTY` = 0.5) and "Full strength", replacing the sentence
  that pointed at the slider. The sentence was the wrong instrument for the reason
  the 2026-09-13 entry gives: the phone card is capped and `#tutText` scrolls, so
  a sentence at the END of the text is the first thing below the fold. The buttons
  are pinned to the bottom of the card and cannot scroll away. Closing text is now
  190 characters, the SHORTEST of the eleven steps (step 7, at 450, is the tallest
  and still sets the pinned height), so nothing about the other steps moved.
  **The constant is `TUT_EASY_POSITION = 0.0`, a SLIDER POSITION** (= effective
  0.8 after the remap). It was `TUT_EASY_DIFFICULTY = 0.5`, a guess, then briefly
  0.8, and both were wrong the same way -- written as effective difficulties, so
  after the remap a value of 0.8 would have meant effective 0.96, nearly full
  strength and the opposite of the button's promise. See the difficulty-slider
  entry above for the measurement.
  **Measured in a browser** (5 viewports: desktop, phone portrait, phone
  portrait + `?safeinset=48,0,56,0`, phone landscape, phone landscape +
  `?safeinset=0,48,24,48`), against main's game.js served side by side as a
  baseline:

        every goal is 7 from home, so a front rack piece tapped at a goal is
        a distance-7 move -- which makes the rule testable with no fixture:
          dice 2+3 (sum 5): sum not offered, no move, 0 dice spent,
                            "That tile is 7 steps away ... so it takes a 7"
          dice 3+4 (sum 7): sum offered, LANDED ON GOAL 3, 2 dice spent,
                            no notice            <- the control that matters
          wall (2 black) 4 away with a die of 4: wall message, not distance
        hint: 2 markers at depth 76, one on the piece and one on the tile
              centre, markedMoveIsLegalNow TRUE, cleared 2 -> 0 when the board
              moved, "Hints are for your own turn." on the computer's turn
        lamp inside the viewport and clear of the legend at all 5, and it
              honours the insets (portrait bottom 903 -> 847, landscape right
              865 -> 817)
        tutorial card box BYTE-IDENTICAL to main on every step at every
              viewport except the closing one in landscape, which SHRANK
              314 -> 292; closing-step text overflow is LOWER than main's
              (portrait 56 vs 78; with insets 85 vs 107)
        0 console errors, 0 failed requests, 94 tiles, 24 pieces, WebGL

  **AMENDED AFTER OWNER'S TRIAL (2026-09-25). Three changes, all verified:**
  **(a) ONE PIECE MOVED TWICE IS NOW ONE HINT, ON THE DICE SUM.** The agent
  returns it as two halves because that is how it searched, but the player makes
  it in a single gesture, so hinting the intermediate tile and only revealing the
  real destination on a second tap was teaching the hard way. `_hintSumMove`
  collapses it: both halves the same piece, both ordinary tile moves, and the
  SECOND half's destination (pair order is application order -- select_move_pair
  chooses the second against the board after the first) actually in
  `reachableBySum`.
  **THE EXCEPTION IS THE GAME'S OWN, NOT A RE-IMPLEMENTATION:** a sum destination
  whose routes offer a choice of captures is withheld in `ambiguousSum` precisely
  so the player spends the dice one at a time to pick the capture, and there the
  two-step hint is correct -- so it falls through. A save half, a block-save half,
  two different pieces and a one-move pair all decline to collapse too.
  Measured, and with the PAIRED CONTROL that proves the gate is doing the work
  rather than the position: the same stubbed reader collapses when the tile is in
  `reachableBySum` (roll 7) and does NOT when the only difference is that the tile
  sits in `ambiguousSum`. On a real opening position the front rack piece offers
  6 sum destinations (the six goals, all exactly 7 from home), and the render puts
  a ring on the piece and on the FINAL tile, none on the intermediate, with
  "...to goal 4 — both dice on the one piece."
  **(b) THE LAMP IS A LABELLED PILL IN THE ACCENT COLOUR** -- owner: *"the hint
  lamp is tiny."* It was a 30px translucent dot matching the legend "?", which is
  wrong by intent: the legend is a reference you consult once and is deliberately
  faint, while this is an action to reach for mid-game. Now 77x34 (2606px2 against
  the legend's 900), accent ground, white glyph plus the word "Hint", which also
  removes the guesswork a bare emoji leaves. Measured inside the viewport with a
  10px gap to the legend at desktop, phone portrait, portrait + `?safeinset=
  48,0,56,0`, landscape + `?safeinset=0,48,24,48`, and a 320px-wide phone.
  **Watch-out:** the "thinking" state must rewrite only the GLYPH span --
  `btn.textContent = '…'` flattens the pill's two spans and the label never
  returns.
  **(c) THE TUTORIAL HANDS OVER WITH HINTS ON AND SAYS SO.** Both closing buttons
  go through `_tutFinish(position)`, which sets the difficulty, WRITES
  `hintsEnabled = '1'` (rather than relying on the default, so a player who had
  turned hints off still gets them back for this one game, and the notice is never
  a lie) and arms `_hintNudgePending`.
  **The notice cannot fire where the button is pressed:** `_tutEnd` restarts the
  scene to the WELCOME CARD, so a notice shown then would sit under it and be gone
  before the first game began. It is delivered from `_hintTick`, the 250ms poll the
  marker already uses, once a real unfrozen game exists with no card over it --
  one waiting place instead of hooking the four paths a game can start from.
  Measured from a start with hints explicitly OFF: "Go easy" gives position 0 /
  effective 0.8 / hints on / nudge armed; **no notice under the welcome card**, and
  the notice appears with the pill visible once the game starts.
  **It is a NOTICE and not another line in the card** for the same reason the
  difficulty sentence became buttons: the card is capped with `#tutText` scrolling
  inside it, so a sentence about hints would be the next thing below the fold.

  **Three fixture traps hit while measuring, each of which produced a convincing
  false pass** -- all three are the "read the denominator" rule again:
  selecting a rack piece **TENTATIVELY ENTERS it onto the home tile**, and a
  stranded one there makes two own pieces on home, which blanks `reachableBySum`
  -- so the "it should move" control silently could not move, and its `notice:
  null` looked like the code correctly staying quiet. Every case now gets a fresh
  page. `piece.move(tile)` **does not remove the piece from its rack**, so a
  hand-placed piece left the board empty and the "game in progress" confirm test
  passed against a fresh game; play into the position with the real handlers
  instead. And a gate test must pull the lever the gate actually reads:
  `currentPlayerIsHuman()` reads the **Player's `isAI`**, never the module-level
  `WHITE_IS_AI`, so setting the global left the predicate true and the hint went
  quietly ahead.

- **THE LAST GAME OF A MATCH SOUNDS FOR THE MATCH, NOT THE GAME (owner,
  2026-09-20).** A match is decided on TOTAL SCORE, so you can lose the final
  game and still take the match -- and the lose chime there read as having lost
  the whole thing. `endGame` now picks the chime from `matchTracker.winner` when
  `recordMatchGame` reports the match over, and from the game's own winner
  otherwise; the call had to MOVE below `recordMatchGame`, which is what sets
  that field. Earlier games in a match still sound for their own result.
  Untouched next to it: a `'tie'` game plays the LOSE chime for a human, because
  the condition special-cases `'draw'` and not `'tie'`. Pre-existing, unrelated
  to matches, not fixed.

- **THE TUTORIAL'S CLOSING PANEL NOW POINTS AT THE DIFFICULTY SLIDER (first
  tester, 2026-09-13).** `getAIDifficulty()` defaults to **1.0 = argmax, full
  strength** -- and OWNER beats it ~50-60% (corrected 2026-09-25; this file had
  it backwards, as the champion beating him), having learned to play it -- so it
  is roughly an expert's equal and will crush a beginner. A player who
  had just finished the tutorial had no idea the setting existed. The last step
  now ends: *"The computer plays at full strength by default. For a gentler
  first game, turn Difficulty down under the ⚙ settings."*
  **It had to be SHORT, and the first two drafts were not.** The phone card is
  capped to the band under the rack (`_tutFitBoard`), and `#tutText` scrolls
  inside it -- so a longer closing panel pushes the difficulty sentence, which
  is at the END of the text, below the fold. Measured: at 288 characters it
  scrolled on a phone WITH system-bar insets (the tester's own case, where the
  card is 256px rather than 270), at 226 it does not. **Do not lengthen this
  step without re-checking `scrollHeight > clientHeight` at
  `?safeinset=48,0,56,0`.** Verified no scroll and the Finish button on screen
  at phone portrait (with and without insets), phone landscape and desktop, and
  step 1's card box is byte-identical -- the bubble is pinned to the TALLEST
  step, so a longer final step would otherwise have shrunk the board on desktop
  for every step.
  **Not done, and the open question:** the default itself. Options discussed
  with owner -- lower it globally (but a silently weaker opponent is its own
  problem), ask once at the end of the tutorial with two buttons, or relabel
  the slider's ends ("Max"/"Easy"/"70%" says nothing about what it does).
  **ANSWERED, AND THE SLIDER HAS BEEN REMAPPED (2026-09-25) -- see the entry at
  the top of Current state.**

- **ANDROID 15/16 EDGE-TO-EDGE PUT THE STATUS BAR ON THE RACKS AND THE
  NAVIGATION BAR ON THE TUTORIAL'S BUTTONS (first tester, Pixel 10,
  2026-09-13).** The packaged app targets **SDK 36**; Android 15 enforces
  edge-to-edge for anything targeting 35+, and Android 16 removed
  `windowOptOutEdgeToEdgeEnforcement` entirely -- so the WebView renders BEHIND
  the system bars and there is no opt-out at that target. Owner's own Android
  never showed it (older OS), so this arrived with the first outside tester.
  **Capacitor 8.5 already hands the page the numbers, two different ways**
  (`SystemBars.java`, `insetsHandling: "css"` by default): on WebView **140+**
  with `viewport-fit=cover` it passes the insets through, so `env(safe-area-
  inset-*)` is correct AND it injects `--safe-area-inset-*` as custom
  properties; on anything older it pads the WebView instead. Either way the
  PAGE has to read them, and this one read only the bottom (for the iPhone home
  indicator) and nothing read the top at all.
  **index.html defines `--safe-t/r/b/l` as the `max()` of the two sources** and
  everything else reads only those four.
  **THE CANVAS IS NOW INSET BY THE SAFE AREA** (`_sizeCanvasToScreen` sets
  `--vx/--vy` alongside `--vw/--vh`). One change covers every piece of
  furniture the game draws -- board, racks, dice, arrows, score, HUD row -- on
  all four edges, because they are all inside the camera's frame; and it costs
  nothing elsewhere, since every world-to-CSS conversion in game.js already
  goes through the canvas's own bounding rect and **Phaser maps pointers
  through it too**. Measured: a tap aimed at the front rack piece selects that
  piece with a 48/56 inset in portrait, with 48/24/48 in landscape, and with no
  inset -- the offset canvas does not move the input.
  `_safeBottomWorld()` is **deleted**; the portrait bottom stack no longer
  subtracts anything, because the frame it sits in is already inset.
  **The DOM chrome is a separate job** -- it is positioned against the
  VIEWPORT, not the canvas, so it does not ride that inset: the gear and its
  panel, the legend button and popup, the flash notice, the first-run toast and
  the tutorial card all add the relevant `--safe-*` in `calc()`, and the five
  full-screen centred overlays (welcome, match setup, confirm, How to Play,
  coin flip) pad their centring box by all four.
  Measured on an emulated 412x915 phone, `?safeinset=48,0,56,0`:

        canvas 0..915 -> 48..859 (= 915 - 56)   tutorial rack top 15 -> 61
        tutorial card bottom 899 -> 843          gear top 10 -> 58

  and landscape `?safeinset=0,48,24,48`: canvas x 48..867, card right edge 851,
  gear right edge 855 -- all inside. **With no inset the layout is unchanged**
  (canvas, racks, HUD row, tutorial card and the desktop build all identical to
  the previous commit, measured against a stashed baseline).
  **`?safeinset` now takes `T,R,B,L` as well as a single bottom value, and it
  WRITES THE CSS VARIABLES rather than short-circuiting the JS read.** The
  first cut only overrode the JS side, so the gear -- which is CSS `calc()` --
  measured at its uninset position while the test passed: an override that only
  half the code can see tests only half the fix.
  **The insets can arrive late and without a resize**, because Capacitor
  injects them from a window-insets listener after the page is up, so
  `setupCameraControls` watches the reading itself every 250ms for the first
  six seconds and re-lays out when it moves; after that a resize covers it.
  When the BUFFER is unchanged but the canvas has MOVED, `scale.updateBounds()`
  has to be called by hand or Phaser keeps mapping pointers through the old
  rect.
  **The page is painted the theme's ground on phones** (`_paintPageGround`), so
  the strips behind the bars read as a continuation of the board rather than as
  a grey frame. Desktop keeps index.html's colour -- its letterbox bands have
  always been that colour.
  **Open, not established: whether Settings > Fullscreen actually hides the
  bars in the PACKAGED app.** It is on by default on phones, and if it worked
  there the tester would have seen no bars at all -- Capacitor's
  `onShowCustomView` may satisfy `requestFullscreen` without going immersive.
  Worth checking on a device; the layout fix stands either way, and in
  fullscreen the insets simply go to zero and the board gets the whole screen
  back.
  **Two harness traps, both of which produced a convincing wrong answer:**
  `camera.worldView` is all zeros until a frame RENDERS and headless Chrome
  does not always paint, so a world-to-CSS test must derive the view from
  `scrollX`/`zoom` (`worldView.x = scrollX + (camW - camW/zoom)/2`) instead;
  and a tap test that happens to run on the COMPUTER's turn reads as a dead tap
  -- `_inputLocked` is doing its job. Set both sides human before the game
  starts.

- **NARROW TILES KEEP THE OLD FORWARD-TAP-TO-TILE BEHAVIOUR (owner, 2026-09-11).**
  The face/halo split (tap the piece = take the selection, tap around it = move)
  needs somewhere to aim BESIDE the piece, and on a narrow tile the piece IS the
  tile. Owner reported it on rings 1-2. Measured, largest disc fitting inside the
  tile and outside the drawn piece, at rest zoom on a phone:

        field ring1  arc  63 ->  7.5 CSS px    field ring3  arc 126 -> 18.0
        field ring7  arc  84 -> 11.3           field ring4  arc 157 -> 18.1
        field ring5  arc  94 -> 13.8           field ring6  arc 220 -> 18.1
        field ring2  arc  94 -> 13.9           goal  ring7  arc 259 -> 27.1

  **The discriminator is the tile's ARC, not its ring** -- every field tile has
  the same 60 of radial extent. The OUTER field ring 7 is tighter than ring 2, so
  a ring-number rule would have fixed half the problem.
  **AND RING 5 IS NOT UNIFORM (owner's correction, and it settles the design):**
  its 12 tiles are 6 at arc 94 and 6 at arc 188, so no per-ring rule can express
  it at all. `_tileHasRoomBeside(tile, piece)` therefore asks the TILE:
  `arc >= pieceDiameter * TILE_ROOM_IN_PIECE_WIDTHS` (2.2, which falls in the gap
  between 1.88 at arc 94 and 2.52 at arc 126). Expressed in piece-widths so it
  holds for the larger goal pieces too.
  **Per-tile census of all 70** (`nogo` excluded): no room = ring1 x9 (arc 63),
  ring2 x9 (94), field ring7 x9 (84), ring5 x6 (94) = **33 tiles**; room = ring3
  x12, ring4 x6, ring5 x6 (188), ring6 x6, **goal x6 (all wide, 259)**, home = 37.
  Behaviour measured on both sides: a face tap on a lone piece on **field ring1
  (arc 63) MOVES** (reverted, as asked), on **field ring3 (arc 126) PASSES THE
  SELECTION**.
  **Measurement trap, hit twice in this session's harness:** `_tileHasRoomBeside`
  reads the PIECE's current radius, which shrinks as a tile fills. Evaluating it
  after the tap (when the tile holds two pieces) or against the wrong piece
  reports "room" on a tile that has none -- both misreads made a narrow tile look
  wide and a correct result look like a bug. In production it is only ever called
  with `_alone` true, so the radius is `tilePieceRadius(1)`; any test must use
  that too.
  **Second harness trap, same session: `tile.addPiece(piece)` does NOT set
  `piece.currentTile`** -- it only pushes onto `tile.pieces` and re-lays out. Only
  `piece.move(tile)` sets both. A position built with `addPiece` therefore has the
  tile and the piece disagreeing about where it is, and every reachability test
  run against it silently reports nonsense (here: a destination that "was not
  reachable", making a correct result look like a bug). Build test positions with
  `piece.move(tile, false)` after removing the piece from wherever it was.

- **TAPPING EMPTY SPACE DESELECTS (owner, 2026-09-11, both platforms).** A tap
  that lands on nothing -- the nogo surround, the background outside the board --
  now clears the selection. On a phone that was previously only possible by
  tapping the piece again; Esc is desktop-only.
  **"Nothing" is Phaser's own answer, not a geometry test.** The handler sits on
  the scene's `pointerup` and returns unless `currentlyOver` is EMPTY, so tiles,
  pieces, ghosts, the racks and every HUD button are excluded for free. That
  exclusion is load-bearing: a tap on your SAVED RACK means "bank the selected
  piece", and turning it into a deselect first would silently kill that gesture.
  nogo tiles never call `setInteractive` (`drawTile` returns before
  `buildTileChrome` for them), so the surround genuinely is empty.
  Guarded like `onTap`: not a ghost mouse event, not part of a pinch, and not a
  drag or camera pan -- releasing a one-finger pan over empty space would
  otherwise deselect every time the board was moved.
  Measured: empty tap -> deselected; a tap with any object under it -> still
  selected; a 9999px release -> still selected; mid-pinch -> still selected; the
  saved-rack tap still banks (0 -> 1 saved).
  **Consequence worth knowing: it also returns a TENTATIVELY ENTERED piece to its
  rack**, because `_clearSelection` does -- so a stray background tap undoes an
  entry. Same as Esc, and fully reversible: measured rack 12 -> tentative entry
  onto home -> empty tap -> back in **slot 0**, rack 12 again, **no die spent**.
  Wired in `setupDragging`, not `setupCameraControls`, because the latter is
  phone-only and this is wanted on both.

- **THE RACKS SHUFFLE BEFORE THE COIN FLIP, NOT AFTER (owner, 2026-09-11).** The
  shuffle lived in `createPieces`, which runs during the scene restart that
  FOLLOWS the flip -- so the order was: coin lands -> board rebuilds -> racks
  visibly reshuffle. Reversed.
  **It could not just be moved, because the shuffle has to happen to the game
  that is ON SCREEN.** `_shuffleRacksThen(cb)` shuffles the HELD game's two
  unentered racks in place (`shiftPiecesUp` re-lays out from the new order),
  waits `SHUFFLE_BEAT_MS` (500) so it reads as its own step rather than flashing
  under the overlay in the same frame, then runs the coin flip -- and hands the
  resulting order to the fresh game as `rackOrder`, which `init` stores and
  `createPieces` ADOPTS instead of drawing a new one. Without that carry-through
  the racks would shuffle twice and the order the player just watched settle
  would be thrown away.
  Gated on `_gameFrozen`, the same test `createPieces` already used to identify
  the held game: anything else (a finished game, the end-of-match card) has no
  meaningful rack to shuffle and falls straight through to the old behaviour.
  Measured on both entry points -- welcome "Single game" and a match's first game
  (via match setup's Start): rack goes 1..12 -> shuffled at frame 0, coin appears
  at 300/400ms, **shuffle before coin both times**, and the order after the game
  starts is byte-identical to the one shown before the flip (no reshuffle). Later
  New Games still draw a FRESH order -- four consecutive games gave four distinct
  orders, none of them sorted -- since they pass no `rackOrder` and fall back to
  the `!_gameFrozen` shuffle.
  `SHUFFLE_BEAT_MS` is declared immediately above the function it serves, not
  among unrelated constants: `RACK_TAP_WINDOW_MS` was once deleted along with the
  log block it happened to sit under.

- **TAPPING YOUR OWN PIECE NOW PASSES THE SELECTION TO IT, WHEN IT STANDS ALONE
  ON A REACHABLE TILE (owner, 2026-09-11, both platforms).** With a piece
  selected, a tap on one of your own pieces used to ALWAYS mean "move onto the
  tile it stands on" whenever that tile was a destination — so a piece within a
  die of the selected one **could not be selected at all**; you had to deselect
  first. Owner hit this repeatedly ("sometimes I've intended the former and
  gotten the latter").
  **The obvious split — piece = reselect, tile = move — is not implementable as
  hit areas, and measuring is what showed it.** A piece's tap target grows to
  half the distance to its nearest neighbour (up to 2.4r), so a piece ALONE on a
  tile swallows the tile whole: measured on a phone, drawn radius 25 against an
  **85** target, leaving a largest free tappable disc of **0 CSS px on five of
  the eight tile geometries** (ring4 0.9, ring6 15.2, goal 8.3) against a ~22px
  fingertip. Implemented literally, moving onto your own piece's tile would have
  become impossible by tap.
  **So the split is by WHERE inside the target the tap fell**, not by which
  object got it (`_tapOnPieceFace`): the visible disc is the piece, the ring
  around it is the tile. No hit area changes, so nothing else regresses, and the
  move keeps the generous target while reselect gets the piece you can see.
  **Restricted to a LONE occupant.** A crowded tile is the case the forwarding
  exists for — there the faces are most of the tile and the slivers between them
  are unhittable — so with 2+ pieces every tap still moves.
  Measured, identical on phone and desktop: lone piece, tap its face ->
  **selection passes**, no move; same piece, tap just outside its drawn edge ->
  **moves**; 3 occupants, tap a face -> **moves**; an opponent's piece alone on a
  destination, tap its face -> **moves and captures** (`selectable` is false for
  them, so they never take the selection); own piece off the reachable set ->
  selection passes, unchanged.
  **Desktop gets it for free and is not a special case**: `_applyHitArea` returns
  early off-phone, so a desktop piece's hit area IS its drawn disc — clicking the
  piece is always a face tap, and the tile is easy to hit with a mouse.

- **THE GAME PLAYED ON BEHIND THE NEW GAME / NEW MATCH CARD (owner, 2026-09-08,
  all platforms).** Pressing either mid-game asks for confirmation over the
  running board (and How to Play covers it outright), and the computer kept
  moving and the turn kept switching while the card sat there -- so closing it
  handed the player back a position they had not been watching. Now it freezes.
  **`_gamePausedByCard()`** (beside `_preGameCardUp`) is true while `#confirmDlg`,
  `#matchSetup` or `#welcomeScreen` is in the DOM. **Derived from the DOM, not
  stored**, for the same reason the settings gear's z-index is: these cards are
  opened and removed from several places and one missed call would strand the
  game paused for the session. The existing body MutationObserver -- already
  there for the dice -- resumes on the down edge, so every dismissal route
  (Cancel, Esc, the backdrop) is covered without each having to remember.
  **Two gates were needed, and the first alone was not enough.** Everything the
  computer does funnels through `getAgentMoves`, so gating it there covers the
  trigger, the second half of a pair and the retry (a reply already in flight is
  DROPPED, not held -- `_resumeHeldAgentTurn` re-asks from the live board, which
  cannot go stale). But `applyMovePair` plays the pair out through **chained
  1-second setTimeouts**, so up to two moves and a turn switch were already
  scheduled: measured, the board visibly kept moving behind the match-setup
  card. Each deferred step now goes through `later()`, which WAITS (re-polling
  at 250ms) while a card is up instead of firing under it.
  **Confirming is not a resumption.** `showConfirm`'s Yes handler and match
  setup's Start clear `_agentTurnHeld` BEFORE removing the card: the observer's
  resume runs as a microtask, ahead of a queued `scene.restart` building the new
  game, so it would otherwise re-ask the computer for a board about to be
  discarded (the reply would be binned by the `instanceId` guard, but only after
  spending an inference and hiding the new game's thinking icon). The `Game`
  constructor clears it too, as a backstop. A confirmed New Game also ends the
  `later()` chain through the existing `stillCurrent()` check.
  **Measured** (both sides computer, so the board never stops on its own; the
  fingerprint is turn + dice + every tile's pieces + saved counts): unchanged
  over 6s with the card up and moving again within 9s of Cancel, **6/6 trials
  with the card opened 300-3100ms into the pair's chain**; before the `later()`
  gate that same test was 0/1 for match setup. Confirm starts a fresh game that
  plays (new instanceId, flag cleared, not paused). **Nothing else regressed:**
  the risky-end-turn confirm shares `#confirmDlg`, and confirming it still hands
  over -- the computer played and gave the turn back, no deadlock -- and New
  Match Start still starts a match.
  **`#howToPlay` is in the set too (owner, 2026-09-08)**, added right after the
  first cut: it covers the board outright, so the same argument applies. Nothing
  extra was needed -- it opens and closes from one place and starts no game, so
  it takes no confirm-path clearing. Measured 3/3 frozen and resumed with it
  opened 500-2500ms into the pair's chain. It is deliberately NOT in
  `_preGameCardUp`, which blanks the dice: that rule is about a board nobody is
  playing, and How to Play sits over a live one -- verified the dice still draw
  (208/169 commands) while it is up.

- **THE BOARD ACCEPTED INPUT DURING THE COMPUTER'S TURN (owner, 2026-09-02).**
  Tapping or double-tapping pieces while the computer was thinking moved them
  and corrupted the position. `Piece.onClick` gates on `player === game.turn`,
  and on the computer's turn **the computer's own pieces satisfy that** -- so a
  tap selected one, lit its destinations, and a tile tap moved it and spent its
  die. A double-tap could bank one outright. **Reproduced before fixing** (AI
  stubbed so its turn hangs, live unfrozen game): black to move, tap black 7 ->
  selected, **18 destinations offered, piece moved to field 6,2, die 6
  consumed**. The turn/thinking pill, the hover highlight, the keyboard
  shortcuts and the end-turn arrow all already refused; the piece and tile
  handlers never did, and the line that would have (`if (this.player !==
  this.game.turn) return;` in `handleClick`) had been commented out.
  Fixed with one predicate, **`_inputLocked(game)`** (beside `_currentGame`),
  applied at every entry point that mutates state: `Piece.handleClick`,
  `Piece.handleDoubleClick` (reachable straight from the stack picker's
  opponent chips, so it needs its own guard), `Tile.onClick`, `Rack.onSaveTap`,
  `Rack.onEntryPanelTap`, the **undo arrow** (the twin of the guard the end-turn
  arrow already had) and `dragstart`.
  **Two deliberate exemptions.** Setup mode -- free placement is outside the
  turn rules by design. And the **tutorial**, which scripts both sides and never
  touches the `isAI` flags: without the exemption a stored "White = computer"
  would have frozen it. Verified with exactly that setting -- `whiteIsAI true`,
  `currentPlayerIsHuman false`, `_inputLocked false`, piece selected, 2
  destinations.
  Measured after the fix: computer's piece on its turn **not selected, 0
  destinations, not moved, no die spent**; human's piece on the computer's turn
  likewise inert; double-tap, saved-rack tap and entry-panel tap all no-ops;
  undo arrow leaves the dice untouched. **Normal play intact** -- own piece on
  own turn still selects, 15 destinations, moves, spends a die -- and setup mode
  still selects a piece during an "AI" turn.
  **The AI is unaffected because it never uses these paths**: `applyMovePair`
  goes through `game.movePiece` / `piece.save` directly. Confirmed by a 70s
  both-sides-computer game: 0 -> 15 pieces on the board, 1 saved.
  **Harness note for any future frontend test:** the browser-automation skill's
  `page.evaluate` runs in an **isolated world** and cannot see game.js's globals
  (`typeof window._currentGame` is `"undefined"` there). Inject with
  `page.addScriptTag` -- that lands in the main world -- and pass results back
  through a DOM attribute, which both worlds share. Also wait for
  `_gameFrozen === false`, not just for the welcome card to go: the coin flip
  and scene restart run after it is removed, and the held game is still frozen.
  **And get the accessors right before believing a probe (2026-09-13):
  `_currentGame` is a FUNCTION, not a variable** -- reading it as an object
  yields `{}` with no error, which looks exactly like "the game never built".
  Pieces live on the GAME (`game.pieces`, 24 of them), NOT on `Player`, which
  holds only `name`/`isAI`/`gamePhase`; tiles are `game.tiles` (94), an array.
  A healthy boot reads: **94 tiles, 24 pieces (12 per colour),
  `renderer.type === 2` (WebGL), exactly one active scene, 0 failed requests.**
  The Canvas2D `willReadFrequently` warnings are the RenderTexture bake's
  readbacks -- expected, not a fault.

- **SESSION UPDATE (2026-08-27) — LIVE ON CLOUDFLARE, AND THE STORE PREP HAS
  STARTED.** The static hosting step of the roadmap is DONE and verified.
  - **LIVE AT `https://quahuru.com`** (owner registered it through Cloudflare
    Registrar, so the domain was already in the account on Cloudflare
    nameservers — attaching it was Worker › Settings › Domains & Routes › Add
    custom domain, apex and www, with DNS and certificate automatic). The
    `quahuru.rechttom.workers.dev` URL still answers the same Worker.
    **NOTHING IN THE REPO NEEDED CHANGING** — no shipped file bakes in an
    origin, and the manifest's `start_url`/`scope` are relative (`./`), so the
    app works on any hostname. Verified on the custom domain exactly as below:
    headers, Brotli, service worker controlling at the new scope, 0 API calls,
    offline reload boots the board. **The Play privacy-policy URL is
    `https://quahuru.com/privacy`.**
    **This is the THIRD origin (Koyeb → workers.dev → quahuru.com) and must be
    the last before testers are recruited**: each move means a fresh service
    worker and cache, and orphans any home-screen install pointing at the old
    one. **Koyeb still tracks main and still deploys on push, but is now
    vestigial** and can be retired once the domain has been stable a few days.
  - **Deployed at `https://quahuru.rechttom.workers.dev`.** Note it is a
    **Worker with static assets, NOT Cloudflare Pages** — the dashboard now
    steers new projects to the Workers flow, whose form has a *deploy command*
    (`npx wrangler deploy`) and NO "build output directory" field. The output
    directory comes from `wrangler.jsonc` in the repo instead:
    `{ name, compatibility_date, assets: { directory: "./dist" } }`, with no
    Worker script at all (assets-only Workers are supported). Build command is
    `python3 build_web.py --out dist`; `build_web.py` imports only the stdlib, so
    the build needs no pip install.
  - **VERIFIED AGAINST THE LIVE SITE, not assumed:** `_headers` **is** honoured
    by Workers static assets (game.js and sw.js come back `no-cache`, ort/ and
    phaser long-lived) — this was the open question when choosing Workers over
    Pages. The wasm serves as `application/wasm` with **`content-encoding: br`**,
    so a cold visit pulls ~2.8 MB rather than 11 MB. Driven under CDP: service
    worker registered at scope `/` and **controlling**, on-device runtime
    `ready`, **0 API calls** during self-play, and an **offline reload boots the
    board** (94 tiles, 24 pieces). `/index.html` 307-redirects to `/`, which is
    normal canonicalisation.
  - **Koyeb is still up deliberately**, so reverting is just handing out the old
    URL. Anyone who home-screened the Koyeb URL is still pointed at it — a
    different origin with its own service worker — and must re-add from the new
    URL. **Settle the final URL BEFORE recruiting the 12 Play testers**, or they
    all have to re-add mid-clock.
  - **`privacy.html` shipped** (in `build_web.py`'s list, `no-cache`, and
    deliberately NOT in `sw.js`'s precache — it is not needed to play offline,
    and the build script only enforces that the worker's list is a SUBSET of the
    bundle). Required by both stores even though the app collects nothing.
    **It carries a `CONTACT_EMAIL` placeholder** — publishing an address is the
    owner's call, so it was not filled in.
  - **Capacitor scaffold in place**: `package.json`, `capacitor.config.json`
    (appId **`com.tomrecht.quahuru`**, which can NEVER change once published;
    `webDir: dist`), and `android/` from `npx cap add android` (Capacitor 8.5.0).
    Only **280 KB / 53 files** of it is tracked — `android/app/src/main/assets/
    public` is the 15 MB copy of `dist` and is gitignored, since `cap sync`
    regenerates it. `npm run sync` rebuilds `dist` and syncs in one step.
    **Gotcha:** `npm install` failed with EACCES on `~/.npm`; worked with
    `--cache ./.npm-cache` (also gitignored). The real fix is
    `sudo chown -R 501:20 ~/.npm`.
  - **THE ANDROID WRAPPER NO LONGER SHIPS CAPACITOR'S BRANDING (2026-08-27).**
    `npx cap add android` generates the Capacitor logo as BOTH the launcher icon
    and the splash screen; uploading that would have put someone else's mark on
    the store. `make_android_assets.py` rewrites all 26 of them from the SHIPPED
    web icons, so launcher, listing and PWA cannot drift: `ic_launcher` /
    `ic_launcher_round` from `icon-512.png` (the board is concentric precisely so
    a round mask cannot clip it, so round reuses the same art), the adaptive
    `ic_launcher_foreground` from `icon-512-maskable.png` (already drawn with the
    padding the 72-of-108dp safe zone needs), `ic_launcher_background` set to the
    parchment #ECE3D3 so bleed at the mask edge is invisible rather than white,
    and every `splash.png` as the board centred on that ground at 42% of the
    short edge. Re-run it after changing the web icons, then `npx cap sync`.
    **The launcher icon is baked at install time — seeing a change on a device
    needs a remove-and-re-add, not a reinstall.**
    **THE FOREGROUND MUST BE SCALED DOWN — THE WEB MASKABLE ICON IS FAR TOO BIG
    FOR ANDROID (owner, on the device, 2026-08-27).** An adaptive icon is 108dp
    but only the central **72dp — 66.7%** — survives the launcher's mask, much
    tighter than the web maskable spec. Measured: `icon-512-maskable.png` draws to
    **85.2%** of its canvas (and `icon-512.png` to 95.2%), so pasting it unscaled
    put the board's rim outside the visible circle on a real phone. The script now
    MEASURES the art's extent — as a radius about the centre, since the board is
    circular and a bounding box would overstate the corners — and scales it to
    **64%**, pasted onto the ground rather than resized to fill, so the mask can
    only ever cut parchment. Verified on the rebuilt asset: art spans **64.7%**,
    and simulating a 66.7% circular mask cuts **0** non-parchment pixels.
  - **RELEASE SIGNING is wired to `android/keystore.properties`**, which is
    GITIGNORED along with `*.jks` / `*.keystore`; `keystore.properties.example`
    shows the four keys. `app/build.gradle` applies the signing config ONLY when
    that file exists, so a fresh clone still builds debug and `bundleRelease`
    fails safely rather than producing an unsigned upload. Create the key with
    `keytool -genkey -v -keystore ~/quahuru-upload.jks -keyalg RSA -keysize 2048
    -validity 10000 -alias quahuru`. **Use Play App Signing and back the .jks and
    its passwords up** — the upload key is how Google knows an update is from
    the same developer.
  - **A SIGNED `.aab` HAS BEEN BUILT: 8.3 MB, `jarsigner` says "jar verified"**,
    at `android/app/build/outputs/bundle/release/app-release.aab`. Confirmed to
    contain the whole app — game.js, model.onnx, the 11 MB ort wasm and phaser
    under `base/assets/public/`. Project is targetSdk/compileSdk **36**, minSdk
    24, applicationId **`com.quahuru.game`**, label "Quahuru". **Every later
    upload needs a HIGHER versionCode** — Play rejects a repeat. Hit
    immediately: the first upload predated the adaptive-icon fix, so Play kept
    serving the clipped icon through an uninstall/reinstall, and replacing it
    needed versionCode 2. **A reinstall from Play does NOT pick up a local
    rebuild** — obvious in hindsight, easy to misread as the icon fix having
    failed.
    **CURRENT PACKAGE: versionCode 5, versionName 1.0.4, 8.3 MB, built
    2026-09-13** -- the safe-area fix for Android 15/16 edge-to-edge (see the
    entry at the top of Current state); it is the one to upload for the closed
    test, since versionCode 4 puts the status bar on the racks on any Android
    15+ device. Verified: `jar verified`, versionCode 5 / versionName 1.0.4 /
    com.quahuru.game read out of the bundle's protobuf manifest (an .aab's
    manifest is PROTOBUF, not binary XML -- `aapt2 dump xmltree` cannot read
    the bundle at all, and a UTF-16 string scan finds nothing; grep it as
    UTF-8), and `base/assets/public/game.js` and `index.html` hash-identical to
    the working tree.
    versionCode 4 / 1.0.3 (also 2026-09-13) carried
    the September input and pre-game work — the ghost-tap fix, the input lock
    during the computer's turn, the freeze behind New Game / New Match / How to
    Play, the tap-claims-the-gesture fix, tap-to-pass-the-selection, empty-space
    deselect, the shuffle-before-the-coin-flip order and the narrow-tile
    exemption. **`game.js` was the ONLY shipped file that changed** (542 diff
    lines over ten commits); index.html, sw.js, the manifest, the ported agent
    and the ort runtime are byte-identical, so `sw.js` keeps `quahuru-v8` — it
    is network-first for index.html and game.js, and no cache-first asset moved.
    **DO NOT REBUILD THE PACKAGE FOR EVERY CHANGE (owner, 2026-09-13).** Push
    commits as normal -- the web build at quahuru.com carries each one -- and
    package only when a BATCH is worth a new version; the versionCode bump
    belongs to that moment. Every upload costs a Play review cycle and a
    permanent versionCode, and the closed test's 14-day clock counts opted-in
    testers, not versions, so batching costs nothing.
    **Rebuilding the package (the whole recipe):**
    `python3 build_web.py --out dist` → bump `versionCode`/`versionName` in
    `android/app/build.gradle` → `npx --cache ./.npm-cache cap sync android` →
    `cd android && ./gradlew bundleRelease > /tmp/rel.log 2>&1; echo $?`.
    `npm run sync` does the first and third in one step.
    **Verify before uploading, and verify the BUNDLE, not the source:**
    `jarsigner -verify` (expect "jar verified"; the PKIX warning is normal for a
    self-signed upload key), the versionCode/versionName/package read out of
    `base/manifest/AndroidManifest.xml` in the .aab, and a hash of
    `base/assets/public/game.js` against the working tree's.
    **The package name was changed from `com.tomrecht.quahuru` to
    `com.quahuru.game` (owner, before any upload)** — it matches the domain he
    now owns, and it is PERMANENT once anything is uploaded to any track, so this
    was the last moment. It lives in seven places: `capacitor.config.json`,
    `android/app/build.gradle` (namespace AND applicationId), `strings.xml`
    (`package_name` and `custom_url_scheme`), the copied
    `android/app/src/main/assets/capacitor.config.json`, and the Java package
    line — plus the source DIRECTORY has to move to `java/com/quahuru/game/`.
    Verified by reading the string pool out of the built bundle's compiled
    manifest, not by grepping the source.
  - **THE BUILD NEEDS A JDK 21 EXACTLY — BOTH JDKs ON THE MACHINE ARE WRONG.**
    A narrow window, and it cost two failed builds:
    * system **JDK 17** is too OLD — Capacitor 8 wants 21+, giving
      **`invalid source release: 21`**;
    * Android Studio's bundled **JBR is Java 25**, too NEW — Gradle 8.14 cannot
      run on it and dies with **`Unsupported class file major version 69`**.
    **The JBR appeared to work twice and did not.** Those builds only succeeded
    because the compiled BUILD SCRIPTS were still cached; the moment
    `app/build.gradle` was edited (the package rename) and had to be recompiled,
    it failed. A cached artefact can hide a broken toolchain — re-verify after
    touching a build script, not just after touching code.
    So `./.jdk` holds a self-contained **Temurin 21** (gitignored, no sudo, no
    system install), pointed at by `org.gradle.java.home` in
    `android/gradle.properties`. **That path is ABSOLUTE and iMac-specific** —
    on the MacBook either re-download into `./.jdk` or, for a path that is the
    same on both machines, `brew install --cask temurin@21` and point at
    `/Library/Java/JavaVirtualMachines/temurin-21.jdk/Contents/Home`.
    `android/local.properties` (gitignored) carries `sdk.dir`; writing it by hand
    means the project never has to be opened in Android Studio at all — the SDK
    it installs is the only thing needed from it.
  - **METHOD NOTE: `./gradlew ... | tail` REPORTS EXIT 0 ON A FAILED BUILD.**
    The pipeline's status is `tail`'s, so the first failure was recorded as a
    success and only the log text gave it away. Redirect to a file and check
    `$?`, never pipe a build into `tail` and trust the code.
  - **`feature-graphic.png` (1024x500)** from `make_feature_graphic.py`, the
    Play listing's required banner. It composites the SHIPPED `icon-512.png`
    rather than re-rendering from `make_icons.py` — that cannot drift, and it
    asserts the icon's ground still equals the banner's (#ece3d3) so the square
    icon has no visible seam. The lockup is centred and auto-shrinks to a 900px
    safe width (Play crops this asset differently per placement); measured
    margins 77/72/80/80.
  - **Licence position, checked this session:** there is **no LICENSE file and
    that is correct** — no licence means all rights reserved, so nothing is
    foreclosed. Do NOT add MIT/Apache to this repo; it would let anyone ship
    Quahuru. The repo is PUBLIC, which grants no use rights but does make the
    code and model readable. **Open compliance item: `phaser.min.js` ships with
    NO MIT notice** (onnxruntime's is intact) — MIT requires the notice in
    copies, and both stores expect an attributions screen. Fix with a NOTICE
    file plus an Acknowledgements line in How to Play. Board-game specific:
    rules and mechanics are not copyrightable, only their expression — the
    NAME is the protectable asset, via trademark.
    **A COPYRIGHT LINE IS IN THE APP (owner, 2026-08-27)**, in How to Play >
    Credits: "Quahuru — the game, its rules, artwork and neural network — is
    © 2026 Tom Recht. All rights reserved." Copyright subsists without a notice,
    so this is not a legal necessity, but both stores expect one and it is what
    tells a reader whose game this is. `privacy.html` and `licenses.html` already
    carried the same line in their footers.
    **DONE (2026-08-27): `licenses.html`** carries the full MIT text for both,
    with the exact upstream copyright lines fetched rather than remembered —
    Phaser "Copyright (c) 2020 Richard Davey, Photon Storm Ltd." (from
    `unpkg.com/phaser@3.55.2/LICENSE.md`) and ONNX Runtime "Copyright (c)
    Microsoft Corporation" (from the runtime we actually ship). Reachable from
    **How to Play › Credits**, which also links the privacy policy — verified on
    both platforms, and the phone text still contains no "click".

- **BRANCHES PRUNED 14 -> 5 (owner, 2026-08-27).** Kept: **`main`** (Cloudflare /
  quahuru.com), **`testing`** (Koyeb), **`symmetry-aug-main`** (the deployed
  champion's lineage, and the ONLY home of `symmetry.py` — main does not have the
  symmetry work at all), and **`gnn` / `gnn2`** (the pre-TD training history, 67
  and 101 unique commits, kept pending owner's call). Deleted: seven branches
  with **zero** commits not already reachable from main (`app-packaging`,
  `deeper-search`, `fast-prefilter`, `frontend-overhaul`, `good_gnn`,
  `rule-single-piece-save`, `td-lambda`), plus `rule-numbered-win` (owner dropped
  the idea) and `symmetry-aug` — after carrying `symaug_smoke.py` onto
  `symmetry-aug-main`, the one file that existed nowhere else. `symmetry.py` was
  byte-identical on both.
  **Method note that nearly caused a wrong conclusion:** `git rev-list -n 1 main
  -- <file>` applies history simplification and reports NOTHING for files that
  only ever appear on a merged side-branch. Use `--full-history`. And the commit
  it names is the one that DELETED the file, so extract from its parent —
  `git show 782c43c0^:train.py`, not `782c43c0:train.py`.

- **REFERENCE: the interactive blocking tool** built 2026-08-27 lives at
  `https://claude.ai/code/artifact/b16ccf81-f0a9-4192-bf58-46a103eaaf64`
  ("Where to Build a Wall"): click a tile to pick the piece to slow, and every
  other tile shows the extra turns a wall there costs it, for a blank or any
  numbered piece. Data is the source x wall matrix from the DP; the generator
  scripts were scratchpad one-offs and are NOT committed, so re-deriving it means
  re-running the value iteration described above.

- **TWO DEPLOY TARGETS, TWO BRANCHES (owner, 2026-08-27).** `main` is what
  **quahuru.com** serves (Cloudflare builds on push) and must stay clean, because
  it is what testers install and play. **`testing` is tracked by KOYEB** and is
  where work-in-progress and instrumentation go, so a diagnostic never reaches
  the live site. Koyeb serves app.py from the repo root, so a branch works there
  with no build step.
  **EVERY BUG FIX GOES TO BOTH (owner, 2026-09-11).** The branches are two live
  deploy targets, and `testing` does NOT receive main's commits by itself -- it
  carries instrumentation on top of main -- so a fix left on main is simply
  absent from the Koyeb build owner is often the one actually playing. Commit on
  `main`, push, then `git checkout testing && git merge --ff-only origin/testing
  && git cherry-pick -x <sha>` and push. Keep it a cherry-pick, not a merge, so
  `testing` never drags its instrumentation back toward main. **Re-run the
  verification on the `testing` checkout too** -- it has extra code in the same
  files, so a clean auto-merge is not proof. Cheap audit that a fix reached both:
  `git diff main testing -- game.js` should show instrumentation and nothing else.
  **`testing` IS NO LONGER A PARALLEL HISTORY (2026-09-24).** It used to carry the
  same fixes as CHERRY-PICKS, so main was never an ancestor and every fix had to
  be applied twice by hand. It has now been **merged with main** (conflicts
  resolved by taking main's files wholesale, so the merge changed no shipped
  byte) and main IS an ancestor. A fix can therefore reach `testing` with
  `git merge main`, and only genuinely `testing`-only work needs care.
  **The tap log is RETIRED.** `_tapRecord` and Settings > Copy tap log existed to
  catch the single-tap-as-double report; that was found and fixed (`a945a9d`,
  "suppress the touch-typed ghost"), so the instrumentation was dropped in the
  same merge. The lesson worth keeping is the shape, not the code: **an
  investigation gets a throwaway branch that is deleted when the bug is found.**
  A `?dev=1`-style gate is NOT a substitute -- asking owner to set a query
  parameter before a bug he cannot predict has failed twice.
  **Living on `testing` now:** the learnability work of 2026-09-24 (hint lamp,
  the shortest-route explanation, the tutorial's new entry point and the closing
  panel's difficulty buttons) — see the entry at the top of Current state. It is
  there for owner to play before any of it reaches quahuru.com.
  **Watch-out already hit there:** the export button was first gated on the log
  being non-empty, but `createSettingsPanel` runs ONCE at start-up, so the
  condition was evaluated before any tap could have happened and the button never
  appeared — the same race as the first-run nudge.
