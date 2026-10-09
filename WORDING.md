# Quahuru — player-facing wording

A reference copy of the text the game shows players, gathered in one place so it
can be read and edited. **It is not shipped and the game does not read it** — the
live strings are in `game.js` (the function named under each heading says where).
Edit here and ask Claude to carry the changes into `game.js`.

Conventions: `{x}` is a value filled in at run time. `<b>`/`<i>` are bold/italic.
`{dbl}` is "double-click" on desktop and "double-tap" on a phone.

Snapshot taken 2026-09-30, on `testing`.

---

## 1. Tutorial (`_tutSteps`, `_tutStepHtml`, `_tutRender`)

Card header: `Tutorial · Step {n} of 12`
Buttons: **Exit** · **← Back** (from step 2 on; a bare **←** on a landscape phone) · **Skip →** (step 1: **Start →**) · closing panel: **← Back** · **Go easy** · **Full strength**
While Black replies: `Black plays…` · On completing a step: `✓ Nice!`

### Step 1 — What you’re playing for  *(no moves; a card on the board shows one sentence at a time, each replacing the last as the board acts it out)*
1. Quahuru is a race: the first to <b>save all twelve pieces</b> wins.
2. Each piece starts on your rack and comes out through the <b>home tile</b> in the centre…
3. …travels out onto the board, where it can <b>capture</b> enemy pieces on the way…
4. …reaches a <b>goal</b> on the rim…
5. …and is <b>saved</b> off it into your saved rack.

On a landscape phone the header reads `Step 1 of 12 · What you’re playing for`.

### Step 2 — Send two pieces out
You can start saving once you’ve brought all your pieces out. Send the front one out through home and spend the <b>5</b> on the highlighted tile near goal 5. Then bring a second piece out with the <b>3</b>. Pieces 1–6 each have one matching goal; blank pieces can use any.

### Step 3 — Numbered pieces head for their goal
Numbered pieces are the hardest to save — only their own goal will take them — so it pays to send them home early. Your next piece is the <b>6</b>, and goal 6 is exactly seven tiles away. Both dice can go on one piece: select the 6, then goal 6 (or drag it there).

### Step 4 — Take what’s exposed
A lone piece is exposed, and landing on it sends it back to the home tile. Black has left two. While you still have pieces on the rack, <b>one of your two moves must be that front rack piece</b>. Enter it with the <b>4</b> onto Black’s 5, and use the <b>2</b> to take the other with the piece already on the board. Black will have to move its captured pieces back out before doing anything else.

### Step 5 — Build a wall
Walls are how you slow your opponent down. Two of your pieces on one tile make a <b>wall</b> — enemy pieces can’t land on it or pass through. Black’s 5 must reach goal 5, and its short way in runs over a tile you hold. Your dice sum to 5: bring your next piece out to join it and shut that route.

### Step 6 — Saving
<b>⏩ A few turns later.</b> Saving is how you win, and it opens up once your rack is empty — as yours now is. A piece on a goal goes out on a die matching that goal’s number: your <b>6</b> is on goal 6 — double-click it, or drag it to your saved rack, and the 6 banks it for a point. Then do the same on <b>goal 1</b> with the 1 — a blank piece can be saved on any goal.

### Step 7 — The long way in
A wall doesn’t stop you, but it can make you pay. Black has walled the tile in front of goal 4. A piece always takes the shortest route to where you send it — and your 4’s shortest route was <b>five</b> tiles, so a single 5 would have done it. Now the only way in is <b>nine</b>, round through goal 2 — and your dice sum to 9, so move your 4 to its goal.

### Step 8 — Buy the door open
Sometimes the only way past a wall is to buy it down. The two walled tiles are the only ways into goals 2 and 4, so both are sealed — your <b>2</b> has no route home on any roll. Spend your dice on the door: double-click one of the two black pieces on the wall <b>in front of goal 2</b> to <b>save it for Black</b>. It costs both dice and hands Black a point, but the wall drops to a single piece — your 2 has a path again.

### Step 9 — The endgame
<b>⏩ Later.</b> Once every piece is on a goal, saving gets easier. Use the <b>1</b> to step your last one onto goal 3. That’s the <b>endgame</b>, and your saved rack lights up to show it: a blank now goes out on any die <i>bigger</i> than its goal’s number, if you hold no higher goal. Your highest is goal 3, so the <b>5</b> takes a blank off it. Numbered pieces always need their own number.

### Step 10 — Some dice do nothing
Not every roll can be used, and that’s fine. The <b>4</b> takes your last blank off goal 3. Your 2 can’t use the 5 — a numbered piece only ever goes out on its own number. Nothing else to do, so end your turn yourself: the right-hand arrow above the board (or the Enter key).

### Step 11 — Your last piece
The rules help a straggler: your 2 has <b>lost its number</b>. With one piece left at the start of your turn, a numbered piece on its goal becomes blank — so it no longer has to wait for a 2, and any die of 2 or more brings it in. Save it and the game is yours.

### Step 12 — You win!
All twelve saved — and you score the number of pieces your opponent still had out: four.<br><br>Now pick how hard your first real game should be. Either way you can change it later under ⚙ settings.

**Length limit:** on a phone the card is capped and the text scrolls inside it.
Step 8 is the tallest and sets the card's height on desktop, so keep any step
no longer than it.

After the tutorial (notice): `Hints are on for this game — tap Hint, bottom right, for a suggested move whenever you want one.`

---

## 2. Rule tips — first real game only (`_ruleTipScan`)

Each fires once, the first time **either side** does the thing. "You" wording is
used when the one human in a game against the computer did it. Otherwise the
second wording is used, with {Who} = "The computer" (or "White"/"Black" in a
two-player game) and {your} = "your" (or "White’s"/"Black’s").

**Opening** (the first turn of the game)
Opening: until a player’s rack is empty, one of their two moves each turn must bring the front rack piece out, unless they have a captured piece to bring out instead. Saving starts once the rack is empty.

**First capture**
- by you: Capture! Landing on a lone enemy piece sends it back to the home tile, and the computer must bring it out again before doing anything else.
- by the other side: {Who} captured {your} piece — a piece alone on a tile can be landed on. It goes back to the home tile, and you must bring it out again before doing anything else.

**First wall**
- by you: That’s a wall: two or more of your pieces on one tile. Enemy pieces can’t land on it or pass through it.
- by the other side: {Who} has built a wall — two pieces on one tile. Your pieces can’t land on it or pass through it, so the way round is longer.

**First rack emptied (saving starts)**
- you: Your rack is empty, so saving starts: a piece on a goal is saved with a die matching that goal’s number — {dbl} it, or drag it to the saved rack. Blank pieces can be saved from any goal, numbered pieces only from their own goal.
- other side: The computer’s rack is empty, so it can start saving: a piece on a goal is saved with a die matching that goal’s. Blank pieces can be saved from any goal, numbered pieces only from their own goal.

**First into the endgame**
- you: Endgame: every piece you have left is on a goal it can be saved from. A blank now also goes out on any die bigger than its goal’s number, as long as you hold no higher goal.
- other side: The computer is in the endgame: every piece left is on a goal it can be saved from. Now the computer’s blanks also go out on any die bigger than their goal’s number, as long as it holds no higher goal.

**First last piece losing its number**
- you: Your last piece has lost its number — with one piece left, a numbered piece on its goal becomes a blank, so it no longer has to wait for its own number.
- other side: The computer’s last piece has lost its number — with one piece left, a numbered piece on its goal becomes a blank, so it no longer has to wait for its own number.

**First block-save**
- by you: You saved an enemy piece off a wall: it costs both dice and gives the computer the point, but thins the wall — a wall of two becomes a single piece, which can be captured.
- by the other side: The computer spent both dice saving one of your pieces for you — that hands over the point to thin a wall. A wall of two becomes a single piece, which can be captured.

Settings row: `Explain rules as they come up (first game)`

---

## 3. Hints (`showHint`, `_renderHint`, `_renderSumHint`)

- Hint: move the marked piece to {goal N / the marked tile}.
- Hint: move the marked piece to {goal N / the marked tile} — both dice on the one piece.
- Hint: save the marked piece — {dbl} it, or drag it to your saved rack.
- Hint: {dbl} the marked enemy piece to save it for them. It costs both dice and hands them a point, but thins the wall — a wall of two becomes a single piece.
- Hint: you can call a draw — the button is bottom-left.
- Hint: nothing this roll can usefully do. End your turn with ↷.
- No hint for a half-finished turn — tap 💡 at the start of a turn instead.
- Hints are for your own turn.
- Both dice are spent — end your turn with ↷.
- No legal move with this roll — end your turn with ↷.
- Getting the computer ready…
- Couldn’t work out a hint just now.
- Hints need the on-device computer, which is switched off for this session.

Pill label: `💡 Hint` · Settings row: `Hint lamp (💡 bottom right)`

---

## 4. "Why can't I move there?" (`_noticeWhyUnreachable`, `_noticeIfRouteWithheld`)

The first two below appear on the first refused tap. The two marked *(on repeat)*
appear only when the player taps the same tile again with the board unchanged.

- Two or more enemy pieces hold that tile — that’s a wall. You can’t land on it or pass through it.
- There’s no route to that tile at all — enemy walls are blocking every way in.
- *(on repeat)* This piece has already moved this turn, so the other die has to carry it further on — it can’t double back.
- *(on repeat)* That tile is {d} steps away by the shortest route, so it takes a {d} — a piece always travels the shortest way, whichever path you had in mind.
- A capture is possible on the way — move one die at a time to choose the route.  *(auto en-route capture off)*
- More than one capture is possible on the way — move one die at a time to choose.  *(auto en-route capture on)*

Another piece has to move first (`_refuseForObligation`) — on the first try while first-game rule tips are on, otherwise when the same piece is tried again with the board unchanged:
- A captured piece has to come back out first — move it off the home tile before anything else.
- Your front rack piece still has to come out this turn, so keep a die for it.
- Rack pieces come out in order, from the front.  *(both dice unused)*
- Only the front piece on your rack can come out now.  *(a die already used)*

Double-click-to-goal declines (`sendToGoal`):
- A captured piece has to come back out first — move it off the home tile before anything else.
- The first piece on the rack must still enter this turn, so this one can’t use both dice.
- More than one goal is in reach, so move it by hand to choose.
- The {die} has another use, so it is left for you — double-tap again to bank this piece.
- More than one capture is possible on the way — move one die at a time to choose.

---

## 5. Other in-game notices and confirmations

- {White/Black} passed
- Getting the computer ready — retrying
- The computer couldn’t start on this device. Tap ↷ to try again.
- The on-device computer is switched off for this session.
- Turn pill: `Your turn` · `Computer thinking…` · `{White/Black}’s turn` · `Computer unavailable`
- Back again to leave — your game is saved.  *(an accidental back during a game: web back guard, and the Android app's back button)*
- Sorry, the saved game could not be restored.  *(Resume failed)*
- Browser's own "Leave site?" prompt during a game — its wording is the browser's, not ours

Confirmations (Cancel / confirm button):
- End your turn without using both dice?
- Start a new game?
- Abandon the current match and start a new one?
- Abandon this game and run the tutorial? — **Run tutorial**

---

## 6. Welcome card (`showWelcome`)

**QUAHURU**
Race your pieces out from the centre to the six goals and bank them all — while walling off your opponent’s routes. Play a single game or a multi-game match. New to it? Try the tutorial first.

Buttons: **Resume game** or **Resume match** (first, only when an unfinished game or match is saved) · **Single game** · **Play a match** · **How to Play** · **Interactive tutorial**

---

## 7. End card (`EndGameScene`)

- {White/Black} wins the game with a score of {n}
- {White/Black} calls a draw!
- The match is a draw! · {Winner} wins the match by {n} · {Winner} wins the match on games won
- White {s} ({w}W) · Black {s} ({w}W) · {n} games  /  race to {t}  /  game {k} of {t}
- Level after {n} games — match extended by 2
- Buttons: **New Game** · **New Match** · **Single Game** · **Next Game**

---

## 8. How to Play (`showInstructions`)

**Goal** — Be the first to <i>save</i> all your pieces. Your score for a win is the number of pieces your opponent still had left — so winning big is worth more.

**Your pieces** — You have 12: six numbered (1–6) and six blank. They start on your side rack.

**A turn** — Roll two dice and move. Each die moves one piece a number of tiles equal to that die; you can move one piece with each die, or one piece with both (their sum). A piece always takes the shortest route to the tile you choose, and once it has moved with one die it can’t double back with the other. You may skip a die (or the whole turn).

**Getting on the board** — Pieces enter through the home tile — the plain disc at the centre. Only the front piece on your rack can enter, and you must enter at least one piece per turn until your rack is empty (unless you have a captured piece, in which case you must enter that).

**Capturing & blocking** — Land on a field tile holding a single enemy piece and you capture it — it goes back to the home tile and its owner must re-enter it before doing anything else. A tile with <b>two or more</b> enemy pieces is a wall: you can’t enter or pass through it.

**Saving** — The six coloured wedges on the rim are goals, numbered 1–6. To save a piece, get it onto a goal and roll that goal’s number to lift it off the board. A numbered piece can only be saved from its own goal; a blank piece from any goal. (You can start saving once all your pieces are on the board.)

**Endgame** — When every piece you have left is saved or sitting on a goal it can be saved from, you’re in the endgame: blank pieces can now be saved with a roll <i>higher</i> than their goal’s number, as long as you have nothing waiting on a higher-numbered goal.

**A couple of special moves** — • Break a wall: past the opening and with no captured pieces, {dbl} (or drag from the picker) one piece of an enemy stack to save it for them — it costs both your dice and hands the opponent a piece, but thins the wall: a wall of two becomes a lone piece.<br>• Last piece: if you start a turn with a single piece left and it’s a numbered one sitting on its goal, it becomes blank (savable by any roll of that goal number or higher).

**Stalemate** — If 10 full rounds pass with nobody saving a piece, either player may call a draw. Any save resets the counter.

**Matches** — A match is several games, and it is won on <b>total score</b> — the sum of your winning margins — not on games won. Two formats: a set number of games (highest total score at the end wins), or a race to a target score. Starters alternate; if the scores finish level the match goes to whoever won more games, and if that is level too it is extended by a pair of games. The score line under the board tracks the match.

**Controls — phone** — Tap a piece, then tap where it should go — or just drag it there. Drag onto its goal, or double-tap, to save. The ↶ arrow undoes one die at a time; ↷ ends your turn. On a crowded tile the <b>+N</b> badge opens a picker (drag a piece straight out of it). Theme, difficulty and options live under the ⚙ settings, and <b>New Match</b> starts a multi-game match.<br>Pinch to zoom, and drag the board to move around it. While you are zoomed in, a piece you still have to enter hovers at the bottom left and the dice appear at the top right — the hovering piece can be tapped, or dragged straight onto the board.<br>
  then, where fullscreen exists: Settings › <b>Fullscreen</b> hides the browser bars, and stops a swipe from the edge of the screen going back a page.
  otherwise (iPhone): Add the game to your home screen to play without the browser bars.

**Controls — desktop** — Click a piece, then click where it should go — or just drag it there. Drag onto its goal, or double-click, to save. The ↶ arrow undoes one die at a time; ↷ ends your turn. On a crowded tile the <b>+N</b> badge opens a picker (drag a piece straight out of it). Theme, difficulty and options live under the ⚙ settings, and <b>New Match</b> starts a multi-game match.<br>Keyboard: <b>Z</b> undoes one die · <b>Enter</b> or <b>Space</b> ends your turn · <b>Esc</b> deselects the piece you’re holding.

**Credits** — Quahuru is built with Phaser and ONNX Runtime Web, both open source. The game collects no data — see the privacy policy.<br><br>Quahuru — the game, its rules, artwork and neural network — is © 2026 Tom Recht. All rights reserved.

---

## 9. Settings labels (`createSettingsPanel`)

Difficulty (slider: Max / Strong / Medium / Gentle / Easiest) · Sound effects ·
Fullscreen · Hint lamp (💡 bottom right) · Move & capture effects · End turn
automatically when both dice used · Confirm ending a turn with a move left ·
{Double-click} sends a piece to its goal · {Double-click} saves a piece in one move ·
Automatic en-route capture · Explain rules as they come up (first game) ·
**Interactive tutorial**
