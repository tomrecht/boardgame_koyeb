# Overnight notes — GUI batch, analysis & roadmap

Everything below the "Analysis & suggestions" line is for your review/decision;
nothing there is wired in yet except where noted.

---

## What shipped tonight (all committed on `frontend-overhaul`, live at :10000)

1. **8 themes + dropdown** — added Rose, Ocean, Sand, Plum Night to Parchment/
   Slate/Forest/Dark. Unobtrusive top-right `<select>`, persisted to
   localStorage; `?theme=` still works.
2. **Difficulty slider** — top-right slider (Easy…Max). Sends a `difficulty`
   with each AI request; the GNN agent argmaxes at Max and otherwise top-p
   samples over a scale-invariant softmax (weaker but never silly). Draw
   decisions always use the true best move. *(backend: `agent_gnn._pick_move_index`,
   `app.py /select_moves`.)*
3. **Save by dragging** to your own saved rack (in addition to double-click).
4. **Match system** — a match is either N games (even, winner by total score)
   or a race to a target total. First game's starter is random, then alternates.
   Tie-break: total score → number of wins → then draw or extend by a pair
   (your choice; extend is default; default 6 games / race 21). The bottom line
   shows the *match's* running score+wins during a match and reverts to session
   totals for casual play. "New Match" HUD button + setup modal; EndGameScene is
   match-aware (Next Game / match result).
5. **Mobile** — pinch-zoom enabled (was blocked); `touch-action: pinch-zoom` so
   one finger drags pieces and two fingers zoom; single-finger touch-drag
   verified. Portrait shows a "rotate to landscape" hint (landscape is much
   bigger). Match modal is responsive.

Also fixed this session (earlier): shortest-path enforcement for two-dice
one-by-one moves (no backtracking), count-based piece sizing, popup-drag, smart
saving, drag polish, etc.

---

# Analysis & suggestions

## A. "How to Play" — review + rewrite

**Verdict:** accurate and complete, but it's one dense wall of ~14 paragraphs in
prose. New players will bounce off it. The *rules* are fine; the *presentation*
is the problem. Specific issues:

- Goal + score are conflated in the first sentence.
- "Entering" is under-explained — a newcomer doesn't know a piece leaves the rack
  onto the hub and then moves out using a die.
- The rarely-used rules (block-save gesture, endgame higher-roll save, "last
  numbered piece becomes unnumbered") sit at the same visual weight as the core
  loop, which buries the basics.
- No mention of the controls you can actually use (drag to move, drag-to-save,
  undo one die at a time, difficulty, themes, matches).

**Recommended structure:** short sections with headers, core loop first, edge
rules last, controls at the end. Draft below — say the word and I'll wire it in
(it also wants the instructions screen to render section headers, which is a
small change). I kept it tight and concrete:

> **Goal**
> Be the first to *save* all your pieces. Your score for a win is the number of
> pieces your opponent still had left — so winning big is worth more.
>
> **Your pieces**
> You have 12: six numbered (1–6) and six blank. They start on your side rack.
>
> **A turn**
> Roll two dice and move. Each die moves one piece a number of tiles equal to
> that die; you can move one piece with each die, or one piece with both (their
> sum). A piece always takes the shortest route to the tile you choose, and once
> it has moved with one die it can't double back with the other. You may skip a
> die (or the whole turn).
>
> **Getting on the board**
> Pieces enter through the yellow hub in the middle. Only the front piece on your
> rack can enter, and you must keep entering until your rack is empty before you
> do anything else.
>
> **Capturing & blocking**
> Land on a field tile holding a single enemy piece and you capture it — it goes
> back to the hub and its owner must re-enter it before doing anything else. A
> tile with **two or more** enemy pieces is a wall: you can't enter or pass
> through it.
>
> **Saving**
> The six coloured wedges on the rim are goals, numbered 1–6. To save a piece,
> get it onto a goal and roll that goal's number to lift it off the board. A
> numbered piece can only be saved from its own goal; a blank piece from any
> goal. (You can start saving once all your pieces are on the board.)
>
> **Endgame**
> When every piece you have left is saved or sitting on a goal it can be saved
> from, you're in the endgame: blank pieces can now be saved with a roll *higher*
> than their goal's number, as long as you have nothing waiting on a
> higher-numbered goal.
>
> **A couple of special moves**
> • Break a wall: past the opening and with no captured pieces, double-click (or
>   drag from the picker) one piece of an enemy two-stack to hand it back to
>   them — it costs both your dice but turns the wall into a lone piece.
> • Last piece: if you start a turn with a single piece left and it's a numbered
>   one sitting on its goal, it becomes blank (savable by any high roll).
>
> **Stalemate**
> If 10 full rounds pass with nobody saving a piece, either player may call a
> draw. Any save resets the counter.
>
> **Controls**
> Tap or drag a piece to move it; drag onto its goal — or double-click — to save.
> The ↶ arrow undoes one die at a time; ↷ ends your turn. On a crowded tile the
> **+N** badge opens a picker (drag a piece straight out of it). Set opponent
> strength with the Difficulty slider, pick a look from the theme menu, and use
> **New Match** for a multi-game match.

## B. Other GUI / UX ideas (rough priority order)

1. **Turn / thinking indicator.** There's no explicit "Your turn" vs "Computer
   thinking…" label. The dice recolour, but a small status line (or a spinner
   while the AI request is in flight) would remove ambiguity — especially on
   mobile where the AI delay looks like a freeze. *(cheap, high value)*
2. **Move animation.** Pieces teleport. A quick tween along the path (even just a
   150 ms slide to the destination) makes captures/saves readable and feels far
   more polished. *(medium)*
3. **Save/capture feedback.** A brief flash/scale-pop when a piece is captured or
   saved, and a soft sound (with a mute toggle). *(cheap–medium)*
4. **Progress at a glance.** A tiny "saved 3 / 12" per side, or filling the saved
   rack visibly, tells you who's ahead without counting. *(cheap)*
5. **Colour-blind check.** The teal/pink dice and the goal colours should be
   verified for deuteranopia/protanopia; offer a high-contrast/CB-safe palette
   as one of the themes. *(cheap)*
6. **Keyboard shortcuts on desktop.** `Z` = undo, `Enter`/`Space` = end turn,
   `Esc` = deselect. *(cheap)*
7. **Confirm risky end-turn.** If you press End Turn with an unused die that had a
   legal move, a small "You still have a move — end anyway?" guard. *(cheap)*
8. **Resign / new-game affordances during a match.** A way to abandon a match
   cleanly (currently New Game mid-match just… starts a game; decide whether that
   should end the match). *(cheap, needs a rule decision)*
9. **First-run nudge.** A one-time "Tap How to Play / try the tutorial" toast.
10. **Onboarding legend.** A tiny always-available legend (what the yellow hub,
    green/blue goals, and greyed dice mean) — a `?` that expands.

## C. Hosting the GNN champion cheaply

The blocker is **PyTorch's size**, not the model (it's ~1.6 MB). Full `torch`
CPU is ~200 MB on disk and ~250–400 MB RAM resident — right at the edge of the
smallest free tiers, and cold starts are slow. Two levers:

**Lever 1 — shrink the runtime (recommended regardless of host).**
Export the champion to **ONNX** and serve with `onnxruntime` instead of torch.
onnxruntime is ~40–60 MB, uses far less RAM, and does CPU inference *faster*
than eager torch. The 2-ply search already batches its forward passes, so it
maps cleanly onto a single `session.run`. This alone gets you comfortably inside
a 512 MB instance and cuts cold-start dramatically. (Cost: a one-time export +
a thin `onnxruntime` inference path parallel to `agent_gnn`.)

**Lever 2 — pick a host:**
- **Koyeb, bumped one size.** Your current eMicro is fine for the heuristic;
  the GNN wants a hair more RAM. With the ONNX runtime a 512 MB Koyeb instance
  should hold it, and it stays always-on (no cold starts). Cheapest continuity
  with what you have.
- **Fly.io** — a `shared-cpu-1x@512MB` VM is ~a couple dollars/month, always-on,
  simple Docker deploy. Good fit with ONNX.
- **Google Cloud Run** — scale-to-zero, genuinely cheap for low traffic (you
  pay per request), generous free tier. Trade-off: cold starts load the runtime
  (~seconds); with ONNX that's tolerable. Best "pay basically nothing when idle."
- **Hugging Face Spaces (CPU, free)** — 16 GB RAM, persistent, purpose-built for
  ML demos; you could even keep full torch. Downside: it's public and Gradio/
  Docker-flavoured, not your own domain.

**My recommendation:** ONNX-export the champ, then either stay on Koyeb (bump to
512 MB, always-on, least change) or move to Cloud Run if you want near-zero idle
cost and can accept cold starts. I can do the ONNX export + a drop-in inference
path as a self-contained task whenever you want; it's also useful locally
(faster moves, lower memory).

## D. Name ideas

Grouped by the image they lean on. My top picks: **Whirligig**, **Sundial**,
**Sixhaven**.

- *The spinning pinwheel look:* **Whirligig** (a spinning folk toy — memorable,
  playful), **Pinwheel**, **Vane**.
- *The sun-yellow hub + radial spokes:* **Sundial**, **Solstice**, **Sunspoke**,
  **Corona**.
- *The saving/rescue heart of the game:* **Sixhaven**, **Havens**, **Homeward**,
  **Sanctuary**, **Harbour**.
- *The hub-and-spoke structure:* **Spindle**, **Hubward**, **Roundhouse**,
  **Radial**.
- *Punchier/abstract:* **Halo**, **Nimbus**, **Cardinal** (six points like a
  compass), **Sixfold**.

If it were mine I'd shortlist **Whirligig** (fun, unique, matches the board),
**Sundial** (elegant, matches the hub), and **Sixhaven** (says what you do). Happy
to riff further once you pick a direction (whimsical vs. elegant vs. descriptive).

## E. Tutorial — roadmap (not started, per your note)

**Shape:** a short, guided, hands-on sequence — not a video, not a wall of text.
Each step loads a *pre-built mini-position*, shows a one-line instruction bubble,
highlights the one thing to do, and advances when you do it. This reuses the
setup-mode free-placement helpers to build positions and the existing move
engine to validate.

**Suggested steps (≈8, ~3–4 min total):**
1. Enter a piece from the rack through the hub.
2. Move with one die; then with two (show the shortest-path highlight).
3. Capture a lone enemy piece.
4. Run into a two-stack wall (feel the block), then go around.
5. Save a numbered piece from its matching goal.
6. Save a blank piece; then the endgame higher-roll save.
7. Break a wall with the double-click/drag-out gift move.
8. Free play vs. an easy AI to finish.

**Infra to build (when you greenlight):**
- A tiny *step runner*: an ordered list of
  `{ setupPosition, dice, instruction, allowedPieces, successWhen }`.
- An *instruction bubble* overlay (DOM, themed) with Next/Skip.
- *Input gating*: temporarily restrict selection to the tutorial's `allowedPieces`
  and fix the dice so the intended move is the natural one (both are easy given
  the current selection guards + `mustMovePieces` machinery).
- Disable the AI and the draw/impasse counters during the tutorial.
- Entry point: a "Tutorial" button next to How to Play; also the first-run nudge.

**Effort:** medium. The step runner + bubble is the bulk; positions are quick to
author in setup mode (I can capture them with the existing export). A lighter v1
("contextual tips during your first real game") is possible first if you'd rather
ship something small immediately.

---

*Questions when you're back:* (1) Should New Game mid-match abandon the match or
just play a one-off? (2) Name direction — whimsical / elegant / descriptive?
(3) Want me to (a) wire the rewritten How-to-Play, (b) do the ONNX export, and/or
(c) start the tutorial step-runner next?
