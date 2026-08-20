/* Auto-detect environment for Koyeb or Localhost */
const IS_LOCAL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';

// This one variable now handles everything
const SERVER_URL = IS_LOCAL 
    ? 'http://localhost:10000' 
    : window.location.origin;

const DEBUG_MODE = false;

// NORMAL PLAY IS SILENT. Every console.log in this file is developer
// scaffolding -- move traces, agent state, save reasoning -- and on desktop it
// buries anything that matters. Suppressed unless the session enables the dev
// modes with ?dev=1, the same switch that unlocks debug / eval / setup, so one
// flag turns the whole developer surface on together.
// warn and error are deliberately LEFT ALONE: a real failure must still show,
// and several recovery paths (a refused agent reply, a disabled local agent)
// report through them.
// Note for test harnesses that read console output: pass ?dev=1.
const _DEV_CONSOLE = (function () {
    try { return new URLSearchParams(location.search).get('dev') === '1'; }
    catch (e) { return false; }
})();
if (!DEBUG_MODE && !_DEV_CONSOLE) {
    const _quiet = function () {};
    console.log = _quiet;
    console.info = _quiet;
    console.debug = _quiet;
}

// Who plays each colour. Defaults: you are White, the computer is Black. Both
// sides can be either, so you can watch the agent play itself or play both
// sides yourself. (playVsComputer is the old single-toggle key, still honoured
// as the default for Black so existing installs don't change behaviour.)
let WHITE_IS_AI = _boolSetting('whiteIsAI', false);
let BLACK_IS_AI = (function () {
    try {
        const nu = localStorage.getItem('blackIsAI');
        if (nu !== null) return nu === '1';
        const s = localStorage.getItem('playVsComputer'); return s === null ? true : s === '1'; }
    catch (e) { return true; }
})();

// Phone-only tweaks: a coarse pointer AND a small screen. Everything gated on
// this leaves the desktop browser exactly as it was. `?phone=0` turns every
// phone tweak off (so a phone can be compared against the plain build without a
// deploy), `?phone=1` forces them on for testing on a desktop.
// Declared HERE, above the first constant that calls it: as a `let` further
// down the file it sat in its temporal dead zone during top-level evaluation,
// the ReferenceError was swallowed by the catch, and every constant sized for a
// phone silently kept its desktop value.
let _phoneOverride;
function _isPhone() {
    try {
        if (_phoneOverride === undefined) {
            const q = new URLSearchParams(location.search).get('phone');
            _phoneOverride = (q === '0' || q === 'off') ? false
                           : (q === '1' || q === 'on') ? true : null;
        }
        if (_phoneOverride !== null) return _phoneOverride;
        return window.matchMedia('(pointer: coarse)').matches &&
               Math.min(window.innerWidth, window.innerHeight) <= 820;
    } catch (e) { return false; }
}

const PIECE_RADIUS_BASE = 20;
// On-board stacking: default board-piece radius. Pieces pack into a polar grid
// of slots sized from each tile's geometry; overflow folds into a "+K" badge
// with a tap-to-pick picker (ported from design/stack.html). On small tiles the
// per-tile radius shrinks (Tile.tilePieceRadius) so >=2 pieces fit un-stacked.
const STACK_PR = 20;       // default board-piece radius (used when it fits)
const STACK_MIN_R = 10;    // pieces shrink no smaller than this before stacking
const TILE_RADIUS_STEP = 60;
const CENTER_X = 900;
// Board vertical centre. 600 (canvas midpoint) keeps goal 2's number, at
// CENTER_Y + outerRadius + 26, inside the 1200-tall canvas instead of clipping.
const CENTER_Y = 600;
const HOME_TILE_RADIUS = TILE_RADIUS_STEP * 1.5;

const TOTAL_PIECES = 12;

// No-save draw rule: once both players are in midgame, if this many full
// rounds (a round = white turn + black turn) pass with no save by either
// player, either player may call a draw. Easily changed. Must match game.py.
const NO_SAVE_TURNS_FOR_DRAW = 10;

const DIE_1_POSITION= 400;   // (still used to place the undo/end-turn arrows)
const DIE_2_POSITION = 500;
// Dice: halfway between the original 80 and the 120 try. A phone grows them
// left and up, keeping the right edge where it is -- the HUD buttons end at
// x=220 so there is room on that side, whereas anything wider or lower runs
// into the board's upper-left arc. 120 at y=30 was the largest of the sizes
// tested against the real tile outlines that touches no tile and no text
// (39 CSS px on a landscape phone, up from 32.5).
const DIE_SIZE = _isPhone() ? 120 : 100;
const DICE_Y = _isPhone() ? 30 : 50;
const DICE_X2 = (_isPhone() ? 576 : 580) - DIE_SIZE;   // second (right) die
const DICE_X1 = DICE_X2 - DIE_SIZE - 20;               // first die, 20px gap
// Rack pieces render a touch larger than board pieces (mockup look), and larger
// again on a phone, where 22 is only ~14 CSS px. The rack panel grows with the
// piece radius (drawBackground derives its size from RACK_PR and spacing), so
// the two racks per side are re-centred on the board's midline to match.
const RACK_PR = _isPhone() ? 26 : 22;
const RACK_Y1 = _isPhone() ? 323 : 356;   // unentered rack
const RACK_Y2 = _isPhone() ? 625 : 622;   // saved rack, directly below it

// ── THEMES ──────────────────────────────────────────────────────────────
// Board palette. Chosen via the in-game dropdown (persisted in localStorage) or
// ?theme=<key> (default parchment). goalNum/accentCss/accentInk/bgInk are CSS
// strings for text; the rest are Phaser hex ints. nogo tiles blend into `bg`.
const THEMES = {
    parchment: { label:'Parchment',
                 bg:0xece3d3, field:0xfffdf8, border:0x43392c, goal:0xc8663f, goalNum:'#3a2418',
                 hub:0x2f9e8f, hubRing:0x1f7568, accent:0xb5623b, accentCss:'#b5623b', accentInk:'#ffffff',
                 highlight:0xe9c9b0, bgInk:'#3a2418' },
    slate:     { label:'Slate',
                 bg:0xdfe3ea, field:0xffffff, border:0x1e2733, goal:0x4a86c5, goalNum:'#0e1720',
                 hub:0xffcf5c, hubRing:0xe0a92e, accent:0x35618e, accentCss:'#35618e', accentInk:'#ffffff',
                 highlight:0xadd8e6, bgInk:'#28313b' },
    forest:    { label:'Forest',
                 bg:0xe1e8e0, field:0xffffff, border:0x20302a, goal:0x3f8a5c, goalNum:'#0e1f16',
                 hub:0xe8b23a, hubRing:0xc48f22, accent:0x2f7050, accentCss:'#2f7050', accentInk:'#ffffff',
                 highlight:0xbfe3cd, bgInk:'#20302a' },
    dark:      { label:'Dark Slate',
                 bg:0x232a33, field:0xeef2f7, border:0x0c1117, goal:0x3fb1c8, goalNum:'#ffffff',
                 hub:0xf0b44a, hubRing:0xc98f2c, accent:0x3fb1c8, accentCss:'#3fb1c8', accentInk:'#06222a',
                 highlight:0x9fc7d6, bgInk:'#e6edf3' },
    rose:      { label:'Rose',
                 bg:0xefe3e8, field:0xfffafc, border:0x3f2630, goal:0xb5566e, goalNum:'#3a1f27',
                 hub:0x6d8f8a, hubRing:0x4f716c, accent:0xa04d63, accentCss:'#a04d63', accentInk:'#ffffff',
                 highlight:0xf0cdd8, bgInk:'#3a1f27' },
    ocean:     { label:'Ocean',
                 bg:0xdce7ec, field:0xffffff, border:0x14313a, goal:0x2f7d95, goalNum:'#08222a',
                 hub:0xf2a65a, hubRing:0xcf8038, accent:0x2f6d85, accentCss:'#2f6d85', accentInk:'#ffffff',
                 highlight:0xbfe0ea, bgInk:'#123039' },
    sand:      { label:'Sand',
                 bg:0xe8dcc0, field:0xfffcf3, border:0x40331f, goal:0xc1666b, goalNum:'#3a2016',
                 hub:0x4d7c8a, hubRing:0x386070, accent:0xa15c4a, accentCss:'#a15c4a', accentInk:'#ffffff',
                 highlight:0xecd7a8, bgInk:'#40331f' },
    plum:      { label:'Plum Night',
                 bg:0x241f31, field:0xeee7f2, border:0x0f0b16, goal:0xc77dff, goalNum:'#ffffff',
                 hub:0xffd166, hubRing:0xcf9f33, accent:0xc77dff, accentCss:'#c77dff', accentInk:'#241432',
                 highlight:0xb28dc9, bgInk:'#efe6f6' },
    // Colour-blind-safe: Okabe-Ito blue/orange, distinguishable under
    // deuteranopia/protanopia/tritanopia (no red-green reliance).
    access:    { label:'High-Contrast (CB-safe)',
                 bg:0xe7eaee, field:0xffffff, border:0x101418, goal:0x0072b2, goalNum:'#101418',
                 hub:0xe69f00, hubRing:0xb37c00, accent:0x0072b2, accentCss:'#0072b2', accentInk:'#ffffff',
                 highlight:0x9ecae1, bgInk:'#101418' },
};
// Phaser hex int -> CSS colour string
function _cssHex(n) { return '#' + n.toString(16).padStart(6, '0'); }

const _themeKey = new URLSearchParams(location.search).get('theme')
    || (typeof localStorage !== 'undefined' && localStorage.getItem('boardTheme'))
    || 'parchment';
// A live-mutable COPY of the palette (never the shared THEMES entry, so switching
// themes at runtime can Object.assign into THEME without corrupting the map).
const THEME = Object.assign({}, THEMES[_themeKey] || THEMES.parchment);
let BACKGROUND_COLOR = THEME.bg;   // nogo tiles blend into this
let GOAL_COLOR = THEME.goal;        // all goal tiles a single colour
let TILE_BORDER = THEME.border;     // tile boundaries
// Recolor callbacks registered by themed Phaser objects (HUD buttons, score,
// AI toggle). Reset each time the scene is (re)created; run by applyThemeLive.
let _themedRedraws = [];
// True while the first-load welcome screen is up: the game is built but held —
// no AI moves — until the player presses Play (which starts a fresh game).
let _gameFrozen = false;

// Baked radial-gradient sphere texture for a piece (matches the mockup's CSS
// spheres, which Phaser vector circles can't reproduce). Cached per colour.
// Piece colors: light body + soft sheen for a glossy 3D read that stays
// visible on the white rack panels (a white gradient sphere on a white panel
// would vanish, so white pieces use a defined rim + slight off-white body).
const PIECE_WHITE_BODY = 0xf1f4f8;
const PIECE_WHITE_RIM  = 0x5b6472;
const PIECE_BLACK_BODY = 0x000000;   // solid black (no sheen highlight)
const PIECE_BLACK_RIM  = 0x0a0b0d;
const colorFirstDie = 0x40E0D0; // Turquoise
const colorSecondDie = 0xFFC0CB; 
const colorSum = 0xFFFF00; // Yellow
const FONT_FAMILY = 'Crimson Text';
// Clean-Modern HUD: system sans-serif + palette (matches design/cm.html).
// Single-quoted family names on purpose: these strings get interpolated into
// HTML style="..." attributes, and a double quote there ends the attribute and
// silently drops the whole declaration (which is what used to happen to the
// confirm dialog's buttons, the welcome sub-line, the How-to-Play body and the
// tutorial bubble). Single quotes are equally valid CSS and safe in both.
const HUD_FONT = "'Segoe UI', system-ui, -apple-system, sans-serif";
// Modal body text (option B: sans headings, serif body). System serif — no
// web font needed.
const BODY_FONT = "Georgia, 'Times New Roman', serif";
const HUD_ACCENT = THEME.accentCss;
const HUD_INK = '#28313b';
const HUD_PANEL_BORDER = 0xdbe1ea;

// Blend two 0xRRGGBB colours, t=0 -> a, t=1 -> b. Used to make the hover tint a
// weaker version of the selection colour rather than the same colour.
function _mixColor(a, b, t) {
    const ch = (v, sh) => (v >> sh) & 0xff;
    const m = (sh) => Math.round(ch(a, sh) + (ch(b, sh) - ch(a, sh)) * t) & 0xff;
    return (m(16) << 16) | (m(8) << 8) | m(0);
}

// AI difficulty (1 = full strength / argmax; lower = weaker via top-p sampling).
function getAIDifficulty() {
    let v = 1.0;
    try { const s = localStorage.getItem('aiDifficulty'); if (s !== null) v = parseFloat(s); } catch (e) {}
    return isFinite(v) ? Math.min(1, Math.max(0, v)) : 1.0;
}
// Boolean settings persisted in localStorage, with a default when unset.
function _boolSetting(key, dflt) {
    try { const s = localStorage.getItem(key); return s === null ? dflt : s === '1'; }
    catch (e) { return dflt; }
}
function getFeedbackEnabled()   { return _boolSetting('fxEnabled', true); }      // move/capture effects
function getAutoEndTurn()       { return _boolSetting('autoEndTurn', false); }   // end turn when both dice used
// confirm ending with a move left (never during the tutorial, which scripts a
// deliberate pass with a live-but-useless die)
function getConfirmRiskyEnd()   { return !_tut.active && _boolSetting('confirmRiskyEnd', true); }
// Optional gesture, OFF by default: double-click/double-tap sends a piece to a
// goal it can reach on the DICE SUM. Never during the tutorial, which scripts
// every move and would be walked off its rails by a shortcut.
function getSumToGoal()         { return !_tut.active && _boolSetting('sumToGoal', false); }
// How long after tentatively entering a rack piece a tap on the SAME slot still
// counts as the second half of that double-click. The mark is only live while
// the piece sits tentatively on home, so this is a backstop rather than the real
// test -- 400ms was too tight for a deliberate double-click, especially since
// the entry highlight is itself deferred 270ms.
// Lives HERE, beside the setting it serves. It previously sat next to a
// game-log constant and was deleted along with it when that log was removed,
// which threw a ReferenceError out of handleClick on every rack click and broke
// selection and dragging from the rack.
const RACK_TAP_WINDOW_MS = 1200;
// ON by default = today's behaviour: a sum move captures a lone enemy it passes,
// picking one by the numbered/higher-numbered rule. Turned OFF, a sum move whose
// ROUTE is ambiguous (see getReachableTilesByDice) is not offered at all, and the
// player moves one die at a time to say which way they meant to go.
function getAutoEnRouteCapture() { return _boolSetting('autoEnRoute', true); }

// A pre-game card is up: the welcome screen or match setup. Whatever board sits
// behind one is not being played, so its dice are not in play either.
function _preGameCardUp() {
    try {
        return !!(document.getElementById('welcomeScreen') || document.getElementById('matchSetup'));
    } catch (e) { return false; }
}

// Repaint the dice, so the rule above takes effect the moment a card appears or
// is dismissed rather than waiting for whatever would next have redrawn them.
function _redrawDice() {
    const g = _currentGame();
    if (!g || !g.dice) return;
    g.dice.forEach(d => { if (d.updateColor) d.updateColor(g.turn); else if (d.drawDie) d.drawDie(); });
}
// Driven off the DOM for the same reason the settings gear's z-index is: these
// cards are shown and removed from several places, and one missed call would
// leave the dice in the wrong state for the rest of the session.
try {
    let _wasUp = null;
    const _obs = new MutationObserver(() => {
        const up = _preGameCardUp();
        if (up === _wasUp) return;      // ignore unrelated body changes
        _wasUp = up;
        _redrawDice();
    });
    if (document.body) _obs.observe(document.body, { childList: true });
    else document.addEventListener('DOMContentLoaded', () => _obs.observe(document.body, { childList: true }));
} catch (e) { /* no MutationObserver: dice simply repaint on the next redraw */ }

// The live Game instance (for settings that act on the running game).
function _currentGame() {
    try { const sc = gameInstance.scene.getScene('MainGameScene'); return sc && sc.game; }
    catch (e) { return null; }
}

// ── TURN / THINKING INDICATOR ───────────────────────────────────────────
function turnStatusText(game) {
    // nothing to say before the player has started a game (welcome screen up)
    if (!game || game.gameOver || _gameFrozen) return '';
    const p = game.turn;
    // The tutorial says whose turn it is in its own card, and it is stripped
    // back to board, racks, dice and arrows -- no pill, gear or score line.
    if (_tut.active) return '';
    const isAI = (p === 'black' && BLACK_IS_AI) || (p === 'white' && WHITE_IS_AI);
    if (isAI) return 'Computer thinking…';
    // "Your turn" only makes sense when exactly one side is yours
    const humans = (WHITE_IS_AI ? 0 : 1) + (BLACK_IS_AI ? 0 : 1);
    return humans === 1 ? 'Your turn' : _cap(p) + '’s turn';
}
// A quick expanding ring at (x,y) — capture (red) / save (accent) feedback.
function fxBurst(scene, x, y, color) {
    if (!getFeedbackEnabled() || !scene || !scene.add) return;
    const ring = scene.add.circle(x, y, 14, color, 0).setStrokeStyle(4, color, 0.9).setDepth(70);
    scene.tweens.add({ targets: ring, scale: 3.2, alpha: 0, duration: 430, ease: 'Cubic.easeOut',
        onComplete: () => ring.destroy() });
}

// A refused move might have been refused ON PURPOSE: automatic en-route capture
// is off and this destination's ROUTE would decide the capture. Say so, or it
// reads as a dead board.
//
// Computed FRESH rather than read off `piece.reachableTiles`. That cache is
// cleared by undo (restoreState nulls it on every piece) and by a turn change,
// and movePiece bails out before ever reaching the message when it is empty --
// which is the likeliest reason owner saw the notice appear on a first attempt
// and not after a move-then-undo. Only runs on a move that is already refused,
// so the extra BFS costs nothing in normal play.
function _noticeIfRouteWithheld(game, piece, targetTile) {
    // Traced, because this failed to appear in one real sequence (move a piece,
    // undo, retry) that no harness has reproduced -- every branch says why.
    // console.log is silent unless ?dev=1, so this costs nothing in play.
    const no = (reason, extra) => { console.log('[route-notice] not shown:', reason, extra || ''); return false; };
    if (getAutoEnRouteCapture()) return no('the "Automatic en-route capture" setting is ON');
    if (!game || !piece || !targetTile) return no('missing game/piece/target');
    let r = null;
    try { r = game.getReachableTilesByDice(piece); } catch (e) { return no('reachability threw', e); }
    if (!r) return no('no reachable set (both dice used?)');
    if (!(r.ambiguousSum || []).includes(targetTile)) {
        return no('target is not a withheld route', {
            ambiguous: (r.ambiguousSum || []).length,
            sum: r.reachableBySum.length,
            dice: game.dice.map(d => d.value + (d.used ? '(used)' : '')),
            target: targetTile.type + ' ' + targetTile.ring + ',' + targetTile.sector,
        });
    }
    if (typeof flashNotice === 'function') {
        flashNotice('A capture is possible on the way — move one die at a time to choose the route.', 4500);
    }
    console.log('[route-notice] shown');
    return true;
}

// You tried to move a piece while a DIFFERENT one is obliged to move (a captured
// piece on the home tile, or the entry from the rack). Nothing happened, and
// without a cue that reads as the board ignoring you -- so pulse the piece that
// actually has to move, twice.
//
// Deliberately NOT gated on the move/capture effects toggle: this is not
// decoration, it is the answer to "why did my move do nothing". And deliberately
// an overlay ring rather than a tween on the piece itself -- a Piece owns three
// display objects (body, sheen, circle), and a tween interrupted by a move or a
// scene restart could strand one of them half-faded.
const MUST_FLASH_COLOR = 0xffb300;        // the amber that already means "must move"
function _flashMustMove(game) {
    const scene = _setupScene();
    if (!scene || !scene.add || !game) return;
    const must = (game.mustMovePieces || []).filter(p => p && p.x != null);
    if (!must.length) return;
    // One pulse per burst of refused taps, not one per tap.
    const now = Date.now();
    if (game._mustFlashUntil && now < game._mustFlashUntil) return;
    game._mustFlashUntil = now + 950;
    must.forEach(p => {
        const r = (p.radius || PIECE_RADIUS_BASE) * 1.45;
        const ring = scene.add.circle(p.x, p.y, r, 0, 0)
            .setStrokeStyle(5, MUST_FLASH_COLOR, 1).setDepth(75);
        scene.tweens.add({
            targets: ring,
            alpha: { from: 1, to: 0.15 },
            scale: { from: 0.82, to: 1.18 },
            duration: 210, yoyo: true, repeat: 1, ease: 'Sine.easeInOut',
            onComplete: () => ring.destroy(),
        });
    });
}

// The canvas keeps a fixed 3:2 shape, so on a phone it is letterboxed: bands of
// empty page above/below it in portrait, left/right of it in landscape. Put the
// pill in a band whenever one is big enough, so it never covers the board (it
// used to sit at the top of the *viewport*, which in landscape is the top of the
// board itself). Falls back to a compact overlay when there is no room anywhere.
// Phone-only tweaks: a coarse pointer AND a small screen. Everything gated on
// this leaves the desktop browser exactly as it was.
// `?phone=0` turns every phone tweak off (so a phone can be compared against
// the plain build without a deploy), `?phone=1` forces them on for testing on a
// desktop. Read once: this is called from hot paths like piece layout.
// Tapping a tile to pick the piece on it. On by default; `?tiletap=0` disables
// it. (It was suspected of breaking selection on a phone; the real cause was
// one-finger browser panning while zoomed -- see PANNING A ZOOMED BOARD.)
let _tileTapOverride;
function _tileTapEnabled() {
    try {
        if (_tileTapOverride === undefined) {
            const q = new URLSearchParams(location.search).get('tiletap');
            _tileTapOverride = !(q === '0' || q === 'off');
        }
        return _tileTapOverride;
    } catch (e) { return false; }
}

// Re-placing the pill needs the element, which only updateTurnStatus holds.
let _replaceTurnStatus = null;

function _placeTurnStatus(el) {
    const c = document.querySelector('canvas');
    if (!c) return;
    if (!_isPhone()) {   // desktop keeps the original top-centre pill
        el.style.cssText = el._base + 'left:50%; transform:translateX(-50%); top:10px;';
        return;
    }
    const r = c.getBoundingClientRect();
    const H = 34, GAP = 8;                       // pill height, and its clearance
    const above = r.top, below = window.innerHeight - r.bottom;
    const side = Math.max(r.left, window.innerWidth - r.right);
    const set = (css) => { el.style.cssText = el._base + css; };
    if (above >= H + GAP) {                      // portrait: band above the board
        set(`left:50%; transform:translateX(-50%); top:${Math.round(r.top - H - GAP / 2)}px;`);
    } else if (below >= H + GAP) {
        set(`left:50%; transform:translateX(-50%); top:${Math.round(r.bottom + GAP / 2)}px;`);
    } else if (side >= 104) {                    // landscape: band beside the board
        // Prefer the left band: the settings gear sits at the top of the right
        // one, so a pill there covers it (drop below the gear if left is too
        // narrow to use).
        const onLeft = r.left >= window.innerWidth - r.right - 24;
        const w = Math.round((onLeft ? r.left : window.innerWidth - r.right) - 16);
        set(`top:${onLeft ? 10 : 84}px; ${onLeft ? 'left' : 'right'}:8px; transform:none;` +
            `width:${w}px; font-size:12px; text-align:center; white-space:normal; line-height:1.25;`);
    } else if (_isPortrait()) {
        // The strip above the rack band is free apart from the gear, which owns
        // the top right.
        set('left:12px; top:14px; transform:none; font-size:12px; padding:4px 10px;');
    } else {
        // No band at all -- on a phone the canvas now fills the screen. Sit under
        // the settings gear on the right: the top left holds the HUD buttons and
        // the centre is the board.
        set('right:12px; top:84px; transform:none; font-size:12px; padding:4px 10px;');
    }
}

function updateTurnStatus(textOrGame) {
    const text = typeof textOrGame === 'string' ? textOrGame : turnStatusText(textOrGame);
    let el = document.getElementById('turnStatus');
    if (!el) {
        el = document.createElement('div'); el.id = 'turnStatus';
        el._base = 'position:fixed; z-index:40; box-sizing:border-box;' +
            'font-family:' + HUD_FONT + '; font-size:14px; font-weight:600; color:#28313b;' +
            'background:rgba(255,255,255,.8); padding:5px 15px; border-radius:20px;' +
            'box-shadow:0 2px 8px rgba(0,0,0,.14); pointer-events:none; transition:opacity .2s;' +
            'white-space:nowrap;';
        el.style.cssText = el._base;
        document.body.appendChild(el);
        // the board is re-fitted on rotate/resize, so the band moves with it
        window.addEventListener('resize', () => _placeTurnStatus(el));
        window.addEventListener('orientationchange', () => setTimeout(() => _placeTurnStatus(el), 250));
    }
    el.textContent = text || '';
    // Placement rewrites cssText wholesale, which would drop the opacity below
    // and leave an empty white pill on screen -- so place first, hide second.
    _placeTurnStatus(el);
    _replaceTurnStatus = () => _placeTurnStatus(el);
    el.style.opacity = text ? '1' : '0';
}

// ── MUST-ENTER GHOSTS ───────────────────────────────────────────────────
// Zooming in can leave the rack off screen, including the piece you are obliged
// to bring out -- with nothing on screen to tell you why nothing else will move.
// The enterable piece(s) are then echoed in a corner of whatever is visible,
// drawn translucent so they read as not-on-the-board, and the first is tappable.

// Scale.FIT keeps the whole canvas on screen, so nothing can be out of frame
// unless the page itself is pinch-zoomed. Checking this first also avoids
// trusting the canvas rect during start-up, before layout has settled -- which
// briefly reported the racks as off screen and flashed the ghosts up at zoom 1.
// PORTRAIT. The camera can frame any rectangle in world space, including one
// with negative coordinates -- so portrait does NOT move the board. It frames a
// taller, narrower box AROUND the board where it already is, and only the
// furniture (racks, dice, arrows, score, buttons) is repositioned into the bands
// above and below. Tile points, hit areas and goal-number text are all cached
// behind Tile._built/_points, so leaving the board alone avoids invalidating any
// of it -- and makes rotation a matter of moving a dozen objects, not a rebuild.
const PORTRAIT = { W: 1160, H: 2510, boardFromTop: 1180 };

function _isPortrait() {
    if (!_isPhone()) return false;
    try {
        const q = new URLSearchParams(location.search).get('portrait');
        if (q === '0') return false;
        if (q === '1') return true;
    } catch (e) {}
    // matchMedia is the orientation the browser actually reports. innerWidth/
    // innerHeight can be momentarily stale during load and while entering
    // fullscreen, and a game built on that reading kept the wrong layout.
    try {
        const m = window.matchMedia('(orientation: portrait)');
        if (m && typeof m.matches === 'boolean') return m.matches;
    } catch (e) {}
    return window.innerHeight > window.innerWidth;
}

// The world rectangle the camera frames.
// Portrait has the width for a much bigger rack: two panels of six fill it.
function _rackPR()  { return _isPortrait() ? 34 : RACK_PR; }
function _dieSize() { return _isPortrait() ? 150 : DIE_SIZE; }

// The tutorial hides the gear, the turn pill and the score stack, which frees a
// strip at the top of the portrait frame and a band at the bottom. Sliding the
// whole assembly up into that strip is what gives the card room to sit clear of
// black's racks.
//
// It takes BOTH halves to be a slide. `_fur()` subtracts the lift from every
// furniture y, which moves the furniture up in WORLD space -- i.e. up relative
// to the board, which does not move. On its own that walked the bottom rack
// into the board. The frame has to travel with it: dropping the same amount off
// boardFromTop moves the frame's origin down by the lift, so every `wd.y + k -
// lift` lands back at its original absolute world position and it is the BOARD
// that rises on screen. The bottom edge of the frame then sits `lift` further
// below the lowest furniture, which is the space the card gets.
//
// Bounded by the top rack: its panel starts 206 world px below the frame's top
// edge (rack y + 240, less the panel's own 34px overhang), so a lift beyond
// that pushes it off screen. 160 keeps ~15 CSS px of margin.
function _tutLift() {
    return (_isPortrait() && typeof _tut !== 'undefined' && _tut.active) ? 160 : 0;
}

// The lift alone is not enough. Measured at 390x844: it leaves a 203 CSS px
// band under the racks, and every one of the eleven steps wants 227-336 px at
// that width. A taller FRAME buys the rest -- the camera fits the whole
// rectangle, so adding empty world below the furniture scales the assembly
// down and turns into screen space at the bottom. 300 world px costs the board
// 363 -> ~325 CSS px (still far above the 234 it had before the portrait
// layout) and is only in force while the tutorial runs.
function _tutFrameExtra() { return _tutLift() ? 300 : 0; }

function _world() {
    if (_isPortrait()) {
        return { x: CENTER_X - PORTRAIT.W / 2,
                 y: CENTER_Y - (PORTRAIT.boardFromTop - _tutLift()),
                 w: PORTRAIT.W, h: PORTRAIT.H + _tutFrameExtra() };
    }
    return { x: 0, y: 0, w: WORLD_W, h: WORLD_H };
}

// Where the furniture sits. Landscape reproduces the historical literals
// exactly; portrait puts a rack band above and below the board, with the dice
// and arrows tucked immediately above it so they stay grouped with the board.
function _fur() {
    const wd = _world();
    if (!_isPortrait()) {
        return { diceX: [DICE_X1, DICE_X2], diceY: DICE_Y,
                 undoX: config.width - (_isPhone() ? 520 : DIE_2_POSITION),
                 endX:  config.width - (_isPhone() ? 330 : DIE_1_POSITION),
                 arrowY: _isPhone() ? 100 : 85,
                 cols: 3, rows: 4, dieSize: DIE_SIZE,
                 whiteUn: [75, RACK_Y1], whiteSv: [75, RACK_Y2],
                 blackUn: [1545, RACK_Y1], blackSv: [1545, RACK_Y2] };
    }
    const cols = 6, rows = 2;
    const lift = _tutLift();          // paired with _world()'s -- see _tutLift
    const pr = _rackPR(), ds = _dieSize();
    const spacing = pr * 2 + 12;
    const panelW = cols * spacing + pr;                // matches drawBackground
    const gap = 40;
    const x1 = wd.x + (wd.w - (2 * panelW + gap)) / 2 + pr;
    const x2 = x1 + panelW + gap;
    // The human's racks go in the top band; with two humans (or two AIs) white
    // does, matching landscape's white-left / black-right reading order.
    const topIsWhite = !WHITE_IS_AI || BLACK_IS_AI;
    const yTop = wd.y + 240 - lift, yBot = wd.y + 1790 - lift;
    const w = { un: [x1, topIsWhite ? yTop : yBot], sv: [x2, topIsWhite ? yTop : yBot] };
    const b = { un: [x1, topIsWhite ? yBot : yTop], sv: [x2, topIsWhite ? yBot : yTop] };
    // Arrows keep landscape's 190px spacing -- closer together they are easy to
    // mis-hit -- and sit against the right margin.
    return { diceX: [wd.x + 60, wd.x + 60 + ds + 20], diceY: wd.y + 455 - lift, dieSize: ds,
             undoX: wd.x + 855, endX: wd.x + 1045, arrowY: wd.y + 530 - lift,
             cols, rows,
             scoreAt: [wd.x + wd.w / 2, wd.y + 2040 - lift], scoreOrigin: [0.5, 0],
             impasseAt: [wd.x + wd.w / 2, wd.y + 2225 - lift],
             callDrawAt: [wd.x + wd.w / 2, wd.y + 2290 - lift],
             hudX: [wd.x + 230, wd.x + 550, wd.x + 870],
             hudY: wd.y + 2395 - lift - _safeBottomWorld(),
             whiteUn: w.un, whiteSv: w.sv, blackUn: b.un, blackSv: b.sv };
}

// Rotation changes which band each piece of furniture belongs in. Because the
// board itself never moves, this is a dozen setPositions rather than a rebuild.
let _lastPortrait = null;
// Portrait brings the rack band up under the gear, and at 64px it overlapped.
// 48 still clears the 44px touch-target guideline.
function _sizeGear(el) {
    const gear = el || document.getElementById('settingsGear');
    if (!gear) return;
    // Hidden for the whole tutorial, on every platform: the script owns the
    // screen, none of the settings apply to it, and in portrait the strip the
    // gear sits in is what the layout lifts into.
    const hide = typeof _tut !== 'undefined' && _tut.active;
    gear.style.display = hide ? 'none' : '';
    const px = _isPortrait() ? 48 : 64;
    gear.style.width = gear.style.height = px + 'px';
    gear.style.fontSize = Math.round(px * 0.53) + 'px';
}

// The three HUD buttons are a row along the bottom in portrait, and their
// widths depend on their labels, so space them from what they actually measure
// rather than from fixed centres -- at a bigger scale, fixed centres overlapped
// and ran off both edges.
// iPhone reserves the bottom strip for the home indicator (and the top/sides
// for the notch). viewport-fit=cover is already set, so env(safe-area-inset-*)
// is non-zero there; nothing read it, which would put the portrait button row
// (22 CSS px off the bottom) under the indicator. Measured from a probe element
// because env() is only available to CSS. ?safeinset=NN forces a value, which
// is the only way to exercise this without an iPhone.
let _safeProbe = null;
function _safeBottomCss() {
    try {
        const q = new URLSearchParams(location.search).get('safeinset');
        if (q !== null && q !== '' && !isNaN(+q)) return +q;
    } catch (e) {}
    if (!_safeProbe) {
        _safeProbe = document.createElement('div');
        _safeProbe.style.cssText = 'position:fixed; left:0; bottom:0; width:0; visibility:hidden;' +
            'pointer-events:none; height:env(safe-area-inset-bottom, 0px);';
        document.body.appendChild(_safeProbe);
    }
    return _safeProbe.getBoundingClientRect().height || 0;
}

// The same distance in world units, so a layout number can be shifted by it.
function _safeBottomWorld() {
    const css = _safeBottomCss();
    if (!css) return 0;
    // World units per CSS pixel -- NOT camera zoom, which is world units per
    // BUFFER pixel and is ~3x off on a device-pixel-ratio 3 screen.
    const cam = _mainCamera();
    const cv = gameInstance && gameInstance.canvas;
    const rect = cv && cv.getBoundingClientRect();
    if (cam && rect && rect.height && cam.worldView.height) {
        return css * (cam.worldView.height / rect.height);
    }
    // Before the first render: the frame fills the screen at base zoom.
    const vh = window.innerHeight || 0;
    return vh ? css * (_world().h / vh) : 0;
}

function _hudK()   { return _isPortrait() ? 2.6 : (_isPhone() ? 2 : 1); }
function _scoreK() { return _isPortrait() ? 4.0 : (_isPhone() ? 2.2 : 1); }

function _layoutHudRow(sc) {
    sc = sc || _setupScene();
    if (!sc || !sc._hudRow || !_isPortrait()) return;
    const wd = _world(), f = _fur();
    const vis = sc._hudRow.filter(b => b && b.visible !== false && b.getBounds);
    if (!vis.length) return;
    const ws = vis.map(b => b.getBounds().width);
    const total = ws.reduce((a, w) => a + w, 0);
    const room = wd.w - 80;
    const gap = vis.length > 1 ? Math.max(16, (room - total) / (vis.length - 1)) : 0;
    let x = wd.x + Math.max(40, (wd.w - (total + gap * (vis.length - 1))) / 2);
    vis.forEach((b, i) => { b.setHudPosition(x + ws[i] / 2, f.hudY); x += ws[i] + gap; });
}

function _hideRotateHint() {
    const el = document.getElementById('rotateHint');
    if (el) el.style.display = _isPortrait() ? 'none' : '';
}

function _relayoutFurniture() {
    _hideRotateHint();
    _sizeGear();
    const g = _currentGame();
    if (!g) return;
    // Deliberately no "nothing changed" guard: a transient reading at start-up
    // could otherwise leave the wrong layout stuck, and this is a couple of
    // dozen setPositions on an event that fires rarely.
    const p = _isPortrait();
    _lastPortrait = p;
    // The frame itself can change, not just what sits in it -- rotation, and
    // the tutorial's lift, both move it. Without this the camera keeps framing
    // the old rectangle and the furniture walks relative to the board.
    const sc0 = _setupScene();
    if (sc0) _fitCameraToWorld(sc0);
    const f = _fur();
    [[g.whiteUnenteredRack, f.whiteUn], [g.whiteSavedRack, f.whiteSv],
     [g.blackUnenteredRack, f.blackUn], [g.blackSavedRack, f.blackSv]].forEach(([r, xy]) => {
        if (!r) return;
        r.x = xy[0]; r.y = xy[1]; r.cols = f.cols; r.rows = f.rows;
        r.pr = _rackPR(); r.spacing = r.pr * 2 + 12;
        r.drawBackground();
        r.shiftPiecesUp();
    });
    (g.dice || []).forEach((d, i) => { d.x = f.diceX[i]; d.y = f.diceY; d.size = f.dieSize; d.drawDie(); });
    // drawDie() paints the DEFAULT colour, not the current player's -- so the
    // relayout has to re-apply it, or a black opening shows black dice that
    // flip to white when the settle pass runs 400ms after boot.
    if (g.updateDiceColors) g.updateDiceColors();
    if (g.undoButton) g.undoButton.setPosition(f.undoX, f.arrowY);
    if (g.switchTurnButton) g.switchTurnButton.setPosition(f.endX, f.arrowY);
    if (g.updateMustMoveHighlights) g.updateMustMoveHighlights();

    const sc = _setupScene();
    if (!sc) return;
    const H = config.height, phone = _isPhone();
    if (sc.scoreText) {
        sc.scoreText.setOrigin(p ? 0.5 : 0, p ? 0 : 1)
                    .setPosition(p ? f.scoreAt[0] : 24, p ? f.scoreAt[1] : H - 24);
        if (sc._fitScoreText) sc._fitScoreText();
    }
    if (sc.impasseText) {
        // The SIZE is per-orientation too, not just the position: it is baked
        // from _scoreK() at create time, so rotating out of portrait left this
        // line at portrait scale (84 world px against landscape's 46) and it
        // ran over the board.
        sc.impasseText.setFontSize(Math.round(21 * _scoreK()));
        sc.impasseText.setOrigin(p ? 0.5 : 0, p ? 0 : 1)
                      .setPosition(p ? f.impasseAt[0] : 24,
                                   p ? f.impasseAt[1] : (phone ? H - 148 : H - 58));
    }
    if (sc.callDrawButton) {
        sc.callDrawButton.setHudK(p ? 2.4 : _scoreK());
        sc.callDrawButton.setHudPosition(p ? f.callDrawAt[0] : (phone ? 190 : 85),
                                         p ? f.callDrawAt[1] : (phone ? H - 247 : H - 115));
    }
    (sc._hudRow || []).forEach((b, n) => {
        if (!b || !b.setHudPosition) return;
        b.setHudK(_hudK());
        b.setHudPosition(p ? f.hudX[n] : 150, p ? f.hudY : (phone ? 48 + n * 84 : 52 + n * 52));
    });
    _layoutHudRow(sc);
    if (_replaceTurnStatus) _replaceTurnStatus();
    if (sc.scoreText) {                      // the base size is per-orientation too
        sc._scoreBaseFs = Math.round(20 * _scoreK());
        if (sc._fitScoreText) sc._fitScoreText();
    }
}

// The zoom at which the whole world just fits the canvas. User zoom multiplies
// this, so "zoom 1" always means "everything visible" whatever the screen is.
function _baseZoom(scene) {
    if (!_isPhone()) return 1;
    const sz = scene.scale.gameSize;
    const wd = _world();
    return Math.min(sz.width / wd.w, sz.height / wd.h) || 1;
}

// Put the camera where the world is framed as asked: `left`/`top` are the world
// coordinates of the top-left of the visible area. Phaser centres a zoomed view
// on scroll + cameraSize/2, so worldView.x = scrollX + (camW - camW/zoom)/2.
function _setCameraView(cam, left, top) {
    // Round the scroll: a half-pixel camera offset renders every edge in the
    // board across two device pixels, which is exactly the residual softness
    // left after fixing the buffer resolution.
    cam.setScroll(Math.round(left - (cam.width - cam.width / cam.zoom) / 2),
                  Math.round(top - (cam.height - cam.height / cam.zoom) / 2));
}

// Draw at device resolution, display at CSS size: the buffer is the screen in
// real pixels, the canvas element covers the viewport, and Phaser maps pointer
// coordinates through the canvas rect, so input stays correct.
function _sizeCanvasToScreen() {
    if (!_isPhone() || !gameInstance || !gameInstance.scale) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 3);   // cap: 4x buffers cost more than they show
    const vw = Math.round(window.innerWidth), vh = Math.round(window.innerHeight);
    if (!vw || !vh) return;
    // CSS owns the displayed size, from these two custom properties, so Phaser
    // re-asserting its own inline width/height on resize cannot win.
    document.body.classList.add('fill-screen');
    document.documentElement.style.setProperty('--vw', vw + 'px');
    document.documentElement.style.setProperty('--vh', vh + 'px');
    let bw = vw * dpr, bh = vh * dpr;
    // Never let the camera shrink the world at RASTERISATION time. Tile outlines
    // are ~1.5 world px; drawn at a zoom below 1 they fall under a device pixel
    // and the whole board looks dusty and broken up. Enlarging the buffer so the
    // world renders at 1:1 or better puts the shrink back where Scale.FIT used
    // to do it -- a smooth image downsample by the browser.
    const wd = _world();
    const grow = Math.max(wd.w / bw, wd.h / bh, 1);
    bw *= grow; bh *= grow;
    // ?maxmp=N caps the buffer in megapixels, so a device that feels sluggish can
    // be A/B'd without a deploy. It matters more than it looks: in PORTRAIT the
    // "grow" above enlarges a 1170x2532 (2.96 MP) iPhone buffer to 1800x3895 =
    // 7.0 MP, and every frame pushes all of it. Lowering this trades the 1:1
    // rasterisation -- outlines soften -- for fill rate, which is the thing a
    // slower GPU/compositor runs out of first.
    const MAX_PX = (function () {
        try {
            const q = parseFloat(new URLSearchParams(location.search).get('maxmp'));
            if (isFinite(q) && q > 0) return q * 1e6;
        } catch (e) {}
        return 9e6;                                          // keep it sane on a phone
    })();
    const over = Math.sqrt((bw * bh) / MAX_PX);
    if (over > 1) { bw /= over; bh /= over; }
    bw = Math.round(bw); bh = Math.round(bh);
    // scale.resize() emits 'resize', so this must never be reached FROM that
    // event or it recurses until the stack blows -- which is what broke rotation
    // and left the camera controls half-wired (no panning).
    const sz = gameInstance.scale.gameSize;
    if (sz.width === bw && sz.height === bh) return;
    gameInstance.scale.resize(bw, bh);
}

function _fitCameraToWorld(scene) {
    if (!_isPhone()) return;
    const cam = scene.cameras.main;
    const sz = scene.scale.gameSize;
    if (!sz.width || !sz.height) return;
    // The camera must follow the new buffer size, or a rotation leaves it
    // rendering the old viewport -- blank, or a board sized for the old screen.
    cam.setSize(sz.width, sz.height);
    scene._camBase = _baseZoom(scene);
    cam.setZoom(scene._camBase * (scene._camUserZoom || 1));
    const vw = cam.width / cam.zoom, vh = cam.height / cam.zoom;
    const wd = _world();
    _setCameraView(cam, wd.x + (wd.w - vw) / 2, wd.y + (wd.h - vh) / 2);
    // A rotation or a resize changes the zoom, and the board texture is only as
    // crisp as the zoom it was baked at.
    _scheduleRebake(scene);
}

// ---------------------------------------------------------------------------
// BAKED BOARD
//
// The board does not change between moves, but every tile is its own Graphics,
// and a Graphics replays its ENTIRE command list every frame -- ~15,000 fill
// and stroke commands across ~107 objects, re-tessellated 60 times a second to
// produce an identical picture. Chrome absorbs it at ~25 fps; Safari's WebGL
// path does not (measured 7.8).
//
// So the resting board is drawn once into a RenderTexture, and a tile puts
// commands into its own Graphics only while it is actually a DIFFERENT colour
// from the baked copy (hover / reachable highlight). Every Graphics object
// stays exactly where it was -- they own the polygon hit areas, and an empty
// one still hit-tests -- it simply holds no commands at rest.
// ---------------------------------------------------------------------------

// A RenderTexture is a bitmap, so it is only as crisp as the scale it was baked
// at, and CLAUDE.md is emphatic that this board must never rasterise below 1:1
// (sub-pixel tile outlines are what made it look "dusty"). Camera zoom is
// exactly world-units -> buffer-pixels, so zoom IS the scale to match.
// Measured: an emulated phone zoomed to 3x asked for a 3465x3465 texture. RGBA
// makes that 48MB on a device that is also holding a 7MP canvas buffer, so the
// budget is deliberately below what deep zoom would like. Past the cap the
// texture is magnified rather than re-baked -- the outlines soften slightly at
// extreme zoom, which is a far better trade than dropping the whole board back
// to ~15,000 commands a frame for the duration.
const BAKE_MAX_PIXELS = 8e6;     // ~32MB RGBA

function _boardBounds(gm) {
    let r = HOME_TILE_RADIUS;
    for (const t of gm.tiles) if (t.outerRadius > r) r = t.outerRadius;
    r = Math.ceil(r) + 4;        // 1.7px stroke, drawn centred, plus headroom
    // INTEGER world bounds. The radii are fractional, and a texture whose
    // top-left lands on a half world-pixel is resampled across two device
    // pixels on every edge -- which shows up as every tile border differing
    // from the live-drawn board. Same failure as a fractional camera scroll.
    return { x: Math.round(CENTER_X - r), y: Math.round(CENTER_Y - r), w: 2 * r, h: 2 * r };
}

function _bakeScaleFor(scene, b) {
    const cam = scene.cameras && scene.cameras.main;
    let s = Math.ceil(((cam && cam.zoom) || 1) * 2) / 2;   // half steps -> fewer re-bakes
    // Floor of 2, deliberately. The live board rasterises its strokes straight
    // at the final device resolution; a texture baked at 1:1 rasterises them at
    // world resolution and is then resampled by Scale.FIT, which came out
    // measurably softer on the tile outlines. Baking at 2x makes that resample
    // a SUPERSAMPLE instead, so the baked board is at least as crisp as the
    // live one everywhere -- which is the bar, given how visible this board's
    // 1.7px outlines are (CLAUDE.md: "borders broken up", "dusty").
    s = Math.max(2, s);
    const cap = Math.sqrt(BAKE_MAX_PIXELS / (b.w * b.h));
    return Math.max(1, Math.min(s, cap));
}

// Draw the resting board into a texture and empty the tiles' command lists.
// Safe to call repeatedly; it replaces any previous bake.
function _bakeBoard(scene) {
    const gm = scene && scene.game;
    if (!gm || !gm.tiles || !gm.tiles.length || !scene.add) return;
    const b = _boardBounds(gm);
    const S = _bakeScaleFor(scene, b);

    // Draw every tile at its RESTING colours. A highlight is drawn live on TOP
    // of the texture, so baking one in would freeze it there for good.
    gm._boardBaked = false;
    for (const t of gm.tiles) t.drawTile('bake');

    if (scene._boardRT) { scene._boardRT.destroy(); scene._boardRT = null; }
    const rt = scene.add.renderTexture(b.x, b.y, Math.ceil(b.w * S), Math.ceil(b.h * S));
    // Below the tiles (depth 0) so a highlighted tile still draws over its own
    // baked copy, and below everything else that was already there.
    rt.setOrigin(0, 0).setScale(1 / S).setDepth(-1);
    for (const t of gm.tiles) {
        if (t.type === 'nogo') continue;         // draws nothing by design
        const g = t.graphics;
        // Tile geometry is in absolute world coordinates on an object sitting at
        // the origin, so scaling about that origin maps world (x,y) -> (x*S,y*S)
        // and the offset puts the board's top-left corner at texel (0,0).
        g.setScale(S); g.setPosition(-b.x * S, -b.y * S);
        rt.draw(g);
        g.setScale(1); g.setPosition(0, 0);
    }
    scene._boardRT = rt;
    scene._bakeScale = S;

    gm._boardBaked = true;
    for (const t of gm.tiles) t.drawTile();      // resting tiles now clear themselves
}

// Re-bake only when the resolution the camera wants has actually changed.
// Debounced, so a pinch does not re-bake every frame of the gesture.
function _scheduleRebake(scene) {
    if (!scene || !scene._boardRT || scene._rebakeTimer) return;
    scene._rebakeTimer = setTimeout(() => {
        scene._rebakeTimer = null;
        try {
            if (!scene._boardRT || !scene.game || scene.game.isDefunct) return;
            const want = _bakeScaleFor(scene, _boardBounds(scene.game));
            if (Math.abs(want - (scene._bakeScale || 0)) > 0.01) _bakeBoard(scene);
        } catch (e) {}
    }, 250);
}

// ?fpstest=1 -- an on-screen A/B of the bake, because the console snippet that
// measures this on a desktop is not pastable on a phone. Toggles the baked
// board off and on around two timed samples and prints fps and per-frame draw
// commands for each, so the comparison is same-page, same-game.
function _installFpsTest(scene) {
    let on = false;
    try { on = new URLSearchParams(location.search).get('fpstest') === '1'; } catch (e) {}
    if (!on || document.getElementById('fpsTestBtn')) return;

    const panel = document.createElement('div');
    panel.id = 'fpsTestBtn';
    panel.style.cssText =
        'position:fixed; left:8px; top:45%; z-index:60; max-width:46vw;' +
        'background:rgba(0,0,0,.82); color:#fff; font:600 13px/1.45 system-ui, sans-serif;' +
        'padding:10px 12px; border-radius:10px; cursor:pointer; white-space:pre;';
    panel.textContent = 'Tap: FPS A/B';
    document.body.appendChild(panel);

    const sample = async (secs) => {
        const g = gameInstance;
        await new Promise(r => setTimeout(r, 400));
        const f0 = g.loop.frame, t0 = performance.now();
        await new Promise(r => setTimeout(r, secs * 1000));
        const fps = (g.loop.frame - f0) / ((performance.now() - t0) / 1000);
        const cmds = scene.children.list.filter(o => o.type === 'Graphics')
            .reduce((n, o) => n + (o.commandBuffer || []).length, 0);
        return { fps: Math.round(fps * 10) / 10, cmds };
    };
    const setBake = (want) => {
        if (!scene._boardRT) return false;
        scene._boardRT.setVisible(want);
        scene.game._boardBaked = want;
        scene.game.tiles.forEach(t => t.drawTile());
        return true;
    };

    let running = false;
    panel.addEventListener('click', async () => {
        if (running) return;
        running = true;
        if (!scene._boardRT) { panel.textContent = 'no baked board\n(bake failed?)'; running = false; return; }
        // A moving piece or a rolling die would land inside a sample window, so
        // take both sides off the computer for the duration of the test.
        WHITE_IS_AI = false; BLACK_IS_AI = false;
        try { applyPlayerRoles(false); } catch (e) {}
        panel.textContent = 'measuring OFF…';
        setBake(false);
        const off = await sample(4);
        panel.textContent = 'measuring ON…';
        setBake(true);
        const onRes = await sample(4);
        const gain = off.fps > 0 ? Math.round((onRes.fps / off.fps - 1) * 100) : 0;
        panel.textContent =
            `OFF  ${off.fps} fps  ${off.cmds} cmd\n` +
            `ON   ${onRes.fps} fps  ${onRes.cmds} cmd\n` +
            `${gain >= 0 ? '+' : ''}${gain}%   (tap to redo)`;
        running = false;
    });
}

function _mainCamera() {
    const sc = _setupScene();
    return (sc && sc.cameras && sc.cameras.main) || null;
}

// A world y as a CSS y on the page, for laying DOM out against the board.
// Reads worldView, which is only correct AFTER a frame has rendered -- callers
// that have just changed the camera must wait one.
function _worldYToCss(wy) {
    const cam = _mainCamera(), cv = gameInstance && gameInstance.canvas;
    const rect = cv && cv.getBoundingClientRect();
    if (!cam || !rect || !rect.height || !cam.worldView.height) return null;
    return rect.top + (wy - cam.worldView.y) * (rect.height / cam.worldView.height);
}
function _pageZoomed() {
    const sc = _setupScene(), cam = _mainCamera();
    if (sc && cam) {
        const base = sc._camBase || _baseZoom(sc);
        if (cam.zoom > base * 1.02) return true;       // zoomed in past the fit
    }
    const vv = window.visualViewport;                  // browser pinch (desktop)
    return !!vv && vv.scale > 1.02;
}

// The world rectangle currently on screen. Under pinch-zoom that is the visual
// viewport; unzoomed it is the whole canvas.
function _visibleWorldRect() {
    const cv = gameInstance && gameInstance.canvas;
    if (!cv) return null;
    const r = cv.getBoundingClientRect();
    if (!r.width || !r.height) return null;
    // The camera already knows exactly which world rectangle is on screen -- no
    // viewport arithmetic needed. Always true on a phone, where the canvas fills
    // the viewport and the camera does the framing.
    const cam = _mainCamera();
    if (cam && (_isPhone() || cam.zoom > 1.02)) {
        const v = cam.worldView;
        return { x0: v.x, y0: v.y, x1: v.right, y1: v.bottom, cssW: r.width };
    }
    const vv = window.visualViewport;
    const l = vv ? vv.offsetLeft : 0, t = vv ? vv.offsetTop : 0;
    const w = vv ? vv.width : window.innerWidth, h = vv ? vv.height : window.innerHeight;
    const wx = (x) => (x - r.left) * (config.width / r.width);
    const wy = (y) => (y - r.top) * (config.height / r.height);
    return { x0: wx(l), y0: wy(t), x1: wx(l + w), y1: wy(t + h), cssW: w };
}

// How many unentered pieces could still be brought out THIS turn. Captured
// pieces sit on the home tile and must move first, one die each, so they crowd
// entries out: one captured leaves one entry, two leaves none, and a spent die
// costs one as well. Capped at two, which is all a turn can manage.
function _enterableUnentered(g) {
    if (!g || g.gameOver || _gameFrozen) return [];
    if (g.currentPlayerIsHuman && !g.currentPlayerIsHuman()) return [];
    const rack = g.turn === 'white' ? g.whiteUnenteredRack : g.blackUnenteredRack;
    if (!rack || !rack.pieces.length) return [];
    const colour = g.turn === 'white' ? 0xffffff : 0x000000;
    const home = g.tiles && g.tiles.find(t => t.type === 'home');
    const captured = home ? home.pieces.filter(p => p.color === colour).length : 0;
    const unused = g.dice.filter(d => !d.used).length;
    const n = Math.max(0, Math.min(unused - captured, rack.pieces.length, 2));
    return rack.pieces.slice(0, n);
}

let _ghosts = [];
function _updateMustEnterGhosts() {
    const scene = _setupScene(), g = _currentGame();
    if (!scene || !scene.add) return;
    const hide = () => _ghosts.forEach(gh => gh.setVisible(false));
    if (!_isPhone() || _tut.active || !_pageZoomed()) { hide(); return; }
    // Never re-place a ghost that is currently being dragged: this runs on
    // camera moves and viewport events, and would snatch it back to its corner
    // mid-gesture.
    if (scene._draggingGhost) return;

    const rect = _visibleWorldRect();
    const pieces = _enterableUnentered(g);
    if (!rect || !pieces.length) { hide(); return; }
    // If EITHER is out of frame, show both: one ghost appearing on its own,
    // while its neighbour is still visible on the rack, reads as a different
    // piece rather than the same pair.
    const anyOff = pieces.some(p => p.x < rect.x0 || p.x > rect.x1 || p.y < rect.y0 || p.y > rect.y1);
    if (!anyOff) { hide(); return; }
    const offscreen = pieces;

    // Constant apparent size: the ghost is a HUD affordance, so it should not
    // balloon with the zoom. ~44 CSS px across, the usual touch-target size.
    const worldPerCss = (rect.x1 - rect.x0) / rect.cssW;
    const r = 22 * worldPerCss, pad = 14 * worldPerCss;
    // The visible rect can extend past the canvas into the letterbox bands, and
    // a ghost placed out there is off the canvas: invisible and untappable.
    // Place within the visible part OF THE CANVAS.
    const _wd = _world();
    const px0 = Math.max(_wd.x, rect.x0), py1 = Math.min(_wd.y + _wd.h, rect.y1);
    offscreen.forEach((piece, i) => {
        let gh = _ghosts[i];
        if (!gh) {
            gh = scene.add.container(0, 0).setDepth(80);
            gh.body = scene.add.circle(0, 0, 1, 0xffffff);
            gh._draggable = true;
            gh.ring = scene.add.circle(0, 0, 1, 0x000000, 0).setStrokeStyle(2, THEME.accent, 1);
            gh.label = scene.add.text(0, 0, '', { fontFamily: HUD_FONT, fontStyle: 'bold' }).setOrigin(0.5);
            gh.add([gh.body, gh.ring, gh.label]);
            _ghosts[i] = gh;
        }
        const x = px0 + pad + r + i * (2 * r + pad);
        const y = py1 - pad - r;
        gh.setPosition(x, y).setVisible(true).setAlpha(i === 0 ? 0.78 : 0.68);
        // a white piece at low alpha on the pale board is just a faint ring, so
        // the body carries the same dark/light rim the real pieces use
        gh.body.setRadius(r).setFillStyle(piece.color, 1)
               .setStrokeStyle(2 * worldPerCss, piece.color === 0xffffff ? 0x2a2320 : 0xf2f2f2, 1);
        gh.ring.setRadius(r + 2 * worldPerCss).setStrokeStyle(2.5 * worldPerCss, THEME.accent, 1);
        gh.label.setText(piece.number <= 6 ? String(piece.number) : '')
                .setFontSize(Math.round(r * 1.2))
                .setColor(piece.color === 0xffffff ? '#000000' : '#ffffff');
        // Both are actionable now that either of the first two rack pieces may
        // enter -- the second used to be a dimmed preview because tapping it
        // could only ever have entered the first.
        gh.body.disableInteractive();
        {
            // Build the interactive object ONCE. setInteractive() replaces it
            // and drops the draggable flag with it, and this function runs on
            // every camera move and viewport event -- so calling it each time
            // would leave the ghost draggable only until the next refresh.
            if (!gh.body.input || !gh.body.input.hitArea) {
                gh.body.setInteractive(new Phaser.Geom.Circle(r, r, r + pad), Phaser.Geom.Circle.Contains);
            } else {
                gh.body.input.hitArea.setTo(r, r, r + pad);   // just re-shape it
                gh.body.input.enabled = true;
            }
            gh.body.off('pointerdown'); gh.body.off('pointerup');
            onTap(gh.body, () => {
                piece.handleClick({ rightButtonDown: () => false });
                _updateMustEnterGhosts();
            });
            // The ghost stands in for the piece, so it drags like one: drag it
            // onto a tile and the piece is entered and moved there in one go.
            gh.body.__ghost = { piece, ghost: gh };
            if (!gh.body.input.draggable) scene.input.setDraggable(gh.body);
        }
    });
    for (let i = offscreen.length; i < _ghosts.length; i++) _ghosts[i].setVisible(false);
}

// The dice matter every turn, and zooming in on the board scrolls them away.
// Same idea as the ghosts: when the real ones are out of frame, pin a small
// readout into the visible area -- top right, since the ghosts take bottom left.
let _hudDice = null;
function _updateHudDice() {
    const scene = _setupScene(), g = _currentGame();
    if (!scene || !scene.add) return;
    if (!_hudDice) _hudDice = scene.add.graphics().setDepth(80);
    _hudDice.clear();
    if (!_isPhone() || !g || g.gameOver || _gameFrozen || _tut.active || !_pageZoomed()) return;
    const rect = _visibleWorldRect();
    const dice = g.dice || [];
    if (!rect || dice.length < 2) return;
    // Only once a die is MEANINGFULLY out of frame: clipping a sliver off the
    // edge and then showing a full copy beside it just reads as duplication.
    // Half of either die hidden is the threshold.
    const hidden = (d) => {
        const vw = Math.max(0, Math.min(d.x + d.size, rect.x1) - Math.max(d.x, rect.x0));
        const vh = Math.max(0, Math.min(d.y + d.size, rect.y1) - Math.max(d.y, rect.y0));
        return 1 - (vw * vh) / (d.size * d.size);
    };
    if (!dice.some(d => hidden(d) >= 0.5)) return;

    const worldPerCss = (rect.x1 - rect.x0) / rect.cssW;
    const size = 34 * worldPerCss, gap = 9 * worldPerCss, pad = 14 * worldPerCss;
    // Clamp into the visible part OF THE CANVAS: the visible rect runs out into
    // the letterbox bands, and anything drawn there is off the canvas entirely.
    const _wd = _world();
    const vx0 = Math.max(_wd.x, rect.x0), vx1 = Math.min(_wd.x + _wd.w, rect.x1);
    const top = Math.max(0, rect.y0) + pad;
    const left = Math.max(vx0 + pad, vx1 - pad - (2 * size + gap));
    dice.forEach((d, i) => {
        paintDie(_hudDice, left + i * (size + gap), top, size, d.value, {
            dieColor: d.used ? 0x808080 : (g.turn === 'white' ? 0xffffff : 0x000000),
            dotColor: g.turn === 'white' ? 0x000000 : 0xffffff,
            borderColor: i === 0 ? colorFirstDie : colorSecondDie,
            bw: 5 * worldPerCss,
        });
    });
}

// A piece's touch target depends on where the OTHER pieces are, so it goes
// stale as soon as any of them moves -- and a stale one can overlap a
// neighbour, which is what makes taps land on the wrong piece. Recompute the
// whole set whenever anything settles. 24 pieces is nothing.
function _refreshHitAreas() {
    if (!_isPhone()) return;
    const g = _currentGame();
    if (!g || !g.pieces) return;
    g.pieces.forEach(p => { if (p._applyHitArea) p._applyHitArea(); });
}

// Everything that has to follow the viewport rather than the board.
function _updateViewportHud() { _refreshHitAreas(); _updateMustEnterGhosts(); _updateHudDice(); }

if (window.visualViewport) {
    visualViewport.addEventListener('resize', () => _updateViewportHud());
    visualViewport.addEventListener('scroll', () => _updateViewportHud());
}

// Phones: commit taps on pointer UP, not pointer DOWN.
// A pinch's first finger fires pointerdown before the second one lands, so a
// down-bound handler has already acted by the time the browser knows it is a
// zoom gesture -- which is how a pinch could move a piece or open a panel.
// Binding to up lets us check two things first: that no second finger joined,
// and that the pointer barely moved (a drag or pan is not a tap).
// Pieces deliberately keep pointerdown: dragging one relies on the selection
// being made there, and a stray selection is harmless anyway.
let _touchesDown = 0, _gestureWasMultiTouch = false;
function _multiTouchActive() { return _gestureWasMultiTouch || _touchesDown > 1; }
['touchstart', 'touchend', 'touchcancel'].forEach(type => {
    window.addEventListener(type, (e) => {
        _touchesDown = e.touches ? e.touches.length : 0;
        if (_touchesDown > 1) _gestureWasMultiTouch = true;
        // clear a little after the last finger lifts, so the second finger's own
        // pointerup cannot land as a tap straight after a pinch
        else if (_touchesDown === 0 && _gestureWasMultiTouch) {
            setTimeout(() => { if (_touchesDown === 0) _gestureWasMultiTouch = false; }, 140);
        }
    }, { capture: true, passive: true });
});

// ONE GESTURE, ONE ACTION. Pieces act on pointerdown (dragging depends on it)
// while tiles act on pointerup, so a single tap that lands on a PIECE was
// handled twice: the piece's handler forwards to its tile's onClick (making the
// move), and then the tile's own tap handler ran onClick again -- by which time
// selectedPiece was null, so the tile-tap-to-select branch picked up the piece
// that had just landed there. That is the "after moving, the piece is selected
// again" report: phone-only, and reproducible on retry because it depends on
// the destination being occupied and unambiguous, not on finger accuracy.
let _consumedGesture = null;
function _consumeGesture(pointer) {
    // downTime as well as id: Phaser REUSES pointer objects between gestures,
    // so identity alone would suppress a later, legitimate tap.
    _consumedGesture = (pointer && pointer.id !== undefined)
        ? { id: pointer.id, downTime: pointer.downTime } : null;
}
function _gestureConsumed(pointer) {
    return !!(pointer && _consumedGesture && pointer.id === _consumedGesture.id
              && pointer.downTime === _consumedGesture.downTime);
}

// pointer.getDistance() is in CANVAS BUFFER pixels, and on a phone the buffer is
// device pixels -- so a bare number here means something different on every
// screen. Measured on a DPR-3 phone: a 10 CSS px touch move reads back as 30.
// The old constants were therefore far tighter than they looked: a tap was
// rejected past 16 buffer px = 5.3 CSS px (less than a fingertip wobbles) while
// a drag only began past 34 = 11.3 CSS px. Anything in between was NEITHER, so
// the gesture did nothing at all -- the "double-tap often doesn't register"
// report -- and anything past 11.3 became a drag, which when dropped on the
// piece's own tile cancels the selection ("taken as a select and tiny drag").
// One slop value in CSS px, converted here, removes the dead zone by
// construction: below it is a tap, at or above it is a drag, nothing is neither.
// ?tapslop=N tunes it on a device without a deploy.
const TAP_SLOP_CSS = 14;
function _tapSlopCss() {
    try {
        const q = parseFloat(new URLSearchParams(location.search).get('tapslop'));
        if (isFinite(q) && q > 0) return q;
    } catch (e) {}
    return TAP_SLOP_CSS;
}
function _bufferPerCss() {
    try {
        const cv = gameInstance && gameInstance.canvas;
        const r = cv && cv.getBoundingClientRect();
        if (cv && r && r.width) return cv.width / r.width;
    } catch (e) {}
    return 1;
}
function _tapSlop() { return _tapSlopCss() * _bufferPerCss(); }

function onTap(obj, handler) {
    if (!_isPhone()) { obj.on('pointerdown', handler); return obj; }
    obj.on('pointerup', function (pointer, ...rest) {
        if (_multiTouchActive()) return;
        if (pointer && pointer.getDistance && pointer.getDistance() > _tapSlop()) return;  // a drag, not a tap
        if (_gestureConsumed(pointer)) return;      // a piece already acted on this tap
        return handler.call(this, pointer, ...rest);
    });
    return obj;
}

// A near miss on a crowded board usually lands on the WRONG tile rather than on
// nothing, and the move is then simply refused. If exactly ONE legal
// destination is within a fingertip of where you actually touched, that was
// plainly the one meant. Two candidates that close is ambiguous and left alone:
// guessing would be worse than refusing, especially with confirm-end-of-turn
// off, where a wrong move is hard to take back. Distance from the touch point,
// not adjacency -- adjacency does not know which side of the tile you touched,
// which made the pick look random when two destinations sat side by side.
function _tileDistance(tile, wx, wy) {
    const pts = tile.calculateAnnularSegmentPoints(
        CENTER_X, CENTER_Y, tile.innerRadius, tile.outerRadius, tile.startAngle, tile.endAngle);
    let best = Infinity;
    for (const p of pts) {
        const d = Math.hypot(p.x - wx, p.y - wy);
        if (d < best) best = d;
    }
    return best;
}

function _resolveDestination(game, tile, wx, wy) {
    if (!_isPhone() || !game || wx == null || wy == null) return tile;
    const piece = game.selectedPiece;
    const rt = piece && piece.reachableTiles;
    if (!rt) return tile;
    const reach = [...new Set([].concat(...Object.values(rt).filter(Array.isArray)))];
    if (!reach.length || (tile && reach.includes(tile))) return tile;   // aimed correctly
    // a fingertip, in world units at the current zoom
    const cam = _mainCamera();
    const cv = gameInstance && gameInstance.canvas;
    const rect = cv && cv.getBoundingClientRect();
    const worldPerCss = (cam && rect && rect.width) ? (cam.worldView.width / rect.width) : 1;
    const tol = 22 * worldPerCss;
    const near = reach.filter(t => _tileDistance(t, wx, wy) <= tol);
    return near.length === 1 ? near[0] : tile;      // exactly one candidate, or leave it
}

function getSoundEnabled()       { return _boolSetting('sound', true); }
// ON by default on phones: besides the screen it buys back, immersive fullscreen
// is the ONLY thing that stops Android's system back gesture eating edge drags
// (confirmed on a device -- no page-level mitigation touches it). An explicit
// opt-out still wins, since _boolSetting only falls back when nothing is stored.
function getFullscreenPref() {
    // ?fullscreen=0/1 overrides, so a device can be A/B'd without clearing storage
    try {
        const q = new URLSearchParams(location.search).get('fullscreen');
        if (q === '0' || q === '1') return q === '1';
    } catch (e) {}
    return _boolSetting('fullscreen', _isPhone());
}

// Fullscreen buys back the ~15% of a phone screen the browser's own bars take.
// It can only be entered from a user gesture, so the preference is applied
// on the first tap after load rather than at start-up. Not offered where the
// API is missing (notably Safari on iPhone, which has no element fullscreen --
// there the equivalent is Add to Home Screen, hence the manifest).
function _fullscreenSupported() {
    return !!(document.documentElement.requestFullscreen && document.fullscreenEnabled);
}
function _enterFullscreen() {
    if (!_fullscreenSupported() || document.fullscreenElement) return Promise.resolve();
    return (document.documentElement.requestFullscreen() || Promise.resolve()).catch(() => {});
}
function _exitFullscreen() {
    if (document.fullscreenElement && document.exitFullscreen) return document.exitFullscreen().catch(() => {});
    return Promise.resolve();
}
// On pointerUP, not down: entering fullscreen resizes the viewport, and doing
// that mid-gesture ate the first drag of a session (measured -- the drag simply
// did nothing). Waiting for the release lets the first gesture finish first.
function _armFullscreenOnFirstGesture() {
    if (!_isPhone() || !getFullscreenPref() || !_fullscreenSupported()) return;
    const go = () => { _enterFullscreen(); window.removeEventListener('pointerup', go, true); };
    window.addEventListener('pointerup', go, true);
}

// Segmented pill control -- two or three mutually exclusive choices, sized for
// a settings row. Returns the element with .value / .setValue / .setDisabled,
// so callers treat it like the <select> it replaces.
function makeSegmented(options, value, onChange) {
    const wrap = document.createElement('div');
    wrap.dataset.seg = '1';
    wrap.style.cssText = 'display:inline-flex; gap:2px; padding:2px; border-radius:999px;' +
        'background:#eef1f4; border:1px solid #dfe4ea;';
    const btns = [];
    const paint = () => btns.forEach(b => {
        const on = b.dataset.value === wrap.value;
        b.style.background = on ? THEME.accentCss : 'transparent';
        b.style.color = on ? '#fff' : '#5a6473';
        b.style.fontWeight = on ? '700' : '600';
    });
    options.forEach(([val, label]) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.dataset.value = val;
        b.textContent = label;
        b.style.cssText = 'border:none; border-radius:999px; cursor:pointer; padding:4px 12px;' +
            'font-family:' + HUD_FONT + '; font-size:12.5px; line-height:1.2; transition:background .12s;';
        b.onclick = () => { if (wrap.disabled) return; wrap.value = val; paint(); if (onChange) onChange(val); };
        wrap.appendChild(b); btns.push(b);
    });
    wrap.value = value;
    wrap.setValue = (v) => { wrap.value = v; paint(); };
    wrap.setDisabled = (d) => {
        wrap.disabled = d;
        wrap.style.opacity = d ? '.45' : '1';
        btns.forEach(b => b.style.cursor = d ? 'default' : 'pointer');
    };
    paint();
    return wrap;
}

// ── SOUND ────────────────────────────────────────────────────────────────
// Synthesised with WebAudio rather than shipped as files: a handful of short
// tones cost nothing to download, can't 404, and keep the deployment a single
// self-contained page. The context is created on the first sound (browsers
// refuse one before a user gesture) and reused.
const SFX = (() => {
    let ctx = null;
    function context() {
        if (ctx) return ctx;
        try {
            const AC = window.AudioContext || window.webkitAudioContext;
            if (AC) ctx = new AC();
        } catch (e) { ctx = null; }
        return ctx;
    }
    // one short enveloped tone; `slide` bends the pitch over the note
    function tone({ freq, dur = 0.09, type = 'sine', gain = 0.12, slide = 0, delay = 0 }) {
        const c = context(); if (!c) return;
        if (c.state === 'suspended') c.resume().catch(() => {});
        const t0 = c.currentTime + delay;
        const osc = c.createOscillator(), amp = c.createGain();
        osc.type = type;
        osc.frequency.setValueAtTime(freq, t0);
        if (slide) osc.frequency.exponentialRampToValueAtTime(Math.max(40, freq + slide), t0 + dur);
        // quick attack, smooth decay: a click without the click artefact
        amp.gain.setValueAtTime(0.0001, t0);
        amp.gain.exponentialRampToValueAtTime(gain, t0 + 0.008);
        amp.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
        osc.connect(amp).connect(c.destination);
        osc.start(t0); osc.stop(t0 + dur + 0.02);
    }
    const play = (fn) => { if (getSoundEnabled()) { try { fn(); } catch (e) {} } };
    return {
        // The save and capture effects carry far more perceived loudness than
        // their gain suggests -- a square wave and a two-note chime have much
        // more energy than one sine. On a phone the others were inaudible below
        // full volume, so the quiet ones are lifted rather than these cut.
        move:    () => play(() => tone({ freq: 320, dur: 0.08, type: 'triangle', gain: 0.22 })),
        capture: () => play(() => tone({ freq: 220, dur: 0.16, type: 'square', gain: 0.09, slide: -110 })),
        save:    () => play(() => { tone({ freq: 660, dur: 0.10, gain: 0.11 });
                                    tone({ freq: 990, dur: 0.14, gain: 0.09, delay: 0.08 }); }),
        win:     () => play(() => [523, 659, 784, 1047].forEach((f, i) =>
                                    tone({ freq: f, dur: 0.16, gain: 0.20, delay: i * 0.11 }))),
        lose:    () => play(() => [392, 330, 262].forEach((f, i) =>
                                    tone({ freq: f, dur: 0.20, type: 'triangle', gain: 0.20, delay: i * 0.13 }))),
    };
})();

// Brief centred notice under the status pill, for things that would otherwise
// happen invisibly (the computer passing its whole turn).
function flashNotice(text, ms = 2400) {
    let el = document.getElementById('flashNotice');
    if (!el) {
        el = document.createElement('div'); el.id = 'flashNotice';
        // Was 13px of grey on translucent white, which owner could barely see --
        // and these are the messages that explain why something did NOT happen,
        // so being missable defeats the point. Bigger, darker, opaque, with a
        // wrap width: the route-choice notice is a full sentence and used to run
        // off the edge as one line.
        el.style.cssText = 'position:fixed; top:44px; left:50%; transform:translateX(-50%);' +
            'z-index:31; font-family:' + HUD_FONT + '; font-size:17px; font-weight:700;' +
            'color:#28313b; background:rgba(255,255,255,.97); padding:11px 20px;' +
            'border-radius:14px; box-shadow:0 6px 22px rgba(0,0,0,.30); pointer-events:none;' +
            // text-wrap:balance so a two-line notice splits evenly instead of
            // stranding the last word or two on a line of their own.
            'max-width:min(560px, 92vw); text-align:center; line-height:1.35;' +
            'text-wrap:balance;' +
            'border:1px solid rgba(0,0,0,.10);' +
            'opacity:0; transition:opacity .2s;';
        document.body.appendChild(el);
    }
    el.textContent = text;
    el.style.opacity = '1';
    clearTimeout(el._t);
    el._t = setTimeout(() => { el.style.opacity = '0'; }, ms);
}

// ── KEYBOARD SHORTCUTS ──────────────────────────────────────────────────
// Esc backs out of whatever is on top, innermost first. Each one is dismissed
// through its OWN cancel path rather than by removing the element, so a
// callback like match-setup's "back to the welcome screen" still runs.
// Deliberately NOT dismissible: the welcome screen (there is nothing behind it
// -- the game underneath is frozen) and the coin flip (it closes itself).
// Returns true when it consumed the key, so the piece-deselect below cannot
// also fire underneath an open dialog.
function _escDismissTop() {
    const click = (box, sel) => {
        const btn = box.querySelector(sel);
        if (btn) btn.click(); else box.remove();
        return true;
    };
    const dlg = document.getElementById('confirmDlg');
    if (dlg) return click(dlg, '#cNo');                    // z70, above the rest
    const setup = document.getElementById('matchSetup');
    if (setup) return click(setup, '#mCancel');
    const howto = document.getElementById('howToPlay');
    if (howto) { howto.remove(); return true; }
    const panel = document.getElementById('settingsPanel');
    if (panel && panel.style.display !== 'none') { panel.style.display = 'none'; return true; }
    const legend = document.getElementById('legendPop');
    if (legend && legend.style.display === 'block') { legend.style.display = 'none'; return true; }
    return false;
}

// Z = undo one die, Enter/Space = end turn, Esc = close an overlay, else deselect.
document.addEventListener('keydown', (e) => {
    // Before the INPUT guard: Esc must still cancel match setup while the
    // caret is in its games field.
    if (e.key === 'Escape' && _escDismissTop()) { e.preventDefault(); return; }
    if (e.target && /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
    if (document.getElementById('matchSetup') || document.getElementById('howToPlay')) return;
    const g = _currentGame();
    if (!g || g.gameOver || !g.currentPlayerIsHuman || !g.currentPlayerIsHuman()) return;
    const k = e.key;
    if ((k === 'z' || k === 'Z') && !e.metaKey && !e.ctrlKey) {
        hideStackPicker(); g.undoOneMove(); clearMoveRecording(); e.preventDefault();
    } else if (k === 'Enter' || k === ' ') {
        if (g.dice.some(d => !d.used) && getConfirmRiskyEnd() && g.hasAnyLegalMove()) g.showConfirmationModal();
        else g.switchTurn();
        e.preventDefault();
    } else if (k === 'Escape') {
        hideStackPicker(); _clearSelection(g);
    }
});
// Difficulty is locked while a match is ongoing; reflect that in the panel.
function refreshSettingsMatchState() {
    const slider = document.querySelector('#settingsDiff input[type=range]');
    const note = document.getElementById('settingsDiffNote');
    const active = !!(matchTracker && !matchTracker.over);
    if (slider) { slider.disabled = active; slider.style.opacity = active ? '.45' : '1'; }
    if (note) note.style.display = active ? 'block' : 'none';
    // Same for who plays which colour: swapping sides mid-match would make the
    // running score meaningless.
    const prow = document.getElementById('settingsPlayers');
    if (prow) {
        prow.querySelectorAll('div[data-seg]').forEach(seg => seg.setDisabled(active));
        const pnote = document.getElementById('settingsPlayersNote');
        if (pnote) pnote.style.display = active ? 'block' : 'none';
    }
}

// Settings pills follow the globals (match setup can change them too).
function syncSettingsPlayers() {
    const prow = document.getElementById('settingsPlayers');
    if (!prow) return;
    const segs = prow.querySelectorAll('div[data-seg]');
    if (segs[0]) segs[0].setValue(WHITE_IS_AI ? 'computer' : 'human');
    if (segs[1]) segs[1].setValue(BLACK_IS_AI ? 'computer' : 'human');
}

// Push WHITE_IS_AI / BLACK_IS_AI onto the live game, and start the computer
// thinking if the change means it is now its move.
function applyPlayerRoles(triggerAI = true) {
    const g = _currentGame();
    if (!g) return;
    const w = g.players.find(p => p.name === 'white');
    const b = g.players.find(p => p.name === 'black');
    if (w) w.isAI = WHITE_IS_AI;
    if (b) b.isAI = BLACK_IS_AI;
    // Start pulling the runtime down as soon as we know a computer will need to
    // move -- including when the roles are set on the welcome card, before the
    // game begins. There is no server to cover the load any more.
    if (typeof _startLocalAIIfNeeded === 'function') _startLocalAIIfNeeded();
    if (typeof updateTurnStatus === 'function') updateTurnStatus(g);
    const cur = g.players.find(p => p.name === g.turn);
    if (triggerAI && cur && cur.isAI && !g.gameOver && !_gameFrozen && !window._tutorialActive) {
        const scene = _setupScene();
        if (scene && scene.showThinkingIcon) scene.showThinkingIcon();
        setTimeout(() => getAgentMoves(getGameState(g)), 400);
    }
}

// Apply a theme instantly — no reload, no new game. Mutate the live THEME palette
// in place and redraw every themed object (board tiles, background, goal numbers,
// HUD buttons, score line). Pieces and dice are theme-neutral, so they stay.
function applyThemeLive(key) {
    const pal = THEMES[key]; if (!pal) return;
    try { localStorage.setItem('boardTheme', key); } catch (e) {}
    Object.assign(THEME, pal);
    BACKGROUND_COLOR = THEME.bg; GOAL_COLOR = THEME.goal; TILE_BORDER = THEME.border;
    const scene = _setupScene(), game = _setupGame();
    if (scene && scene.cameras && scene.cameras.main) scene.cameras.main.setBackgroundColor(THEME.bg);
    if (game && game.tiles)  game.tiles.forEach(t => { if (t.applyThemeColors) t.applyThemeColors(); });
    if (game && game.pieces) game.pieces.forEach(p => { if (p.updateColor) p.updateColor(); });
    // The board texture holds the OLD palette, and applyThemeColors above only
    // repainted tiles that are drawing live -- so the bake has to be redone or
    // the whole board keeps the previous theme's colours.
    if (scene && scene._boardRT) { try { _bakeBoard(scene); } catch (e) {} }
    _themedRedraws.forEach(fn => { try { fn(); } catch (e) {} });
}

// A single unobtrusive Settings gear (top-right) holding theme, difficulty and
// the effects toggle, so they're not always on screen.
function createSettingsPanel() {
    if (document.getElementById('settingsGear')) return;
    const mk = (tag, css, txt) => { const e = document.createElement(tag);
        if (css) e.style.cssText = css; if (txt != null) e.textContent = txt; return e; };

    const gear = mk('button',
        'position:fixed; top:10px; right:12px; z-index:41; width:64px; height:64px;' +
        'border-radius:14px; border:1px solid rgba(0,0,0,.15); background:rgba(255,255,255,.75);' +
        'color:#28313b; font-size:34px; line-height:1; cursor:pointer; opacity:.6;' +
        'display:grid; place-items:center; transition:opacity .15s;', '⚙');
    gear.id = 'settingsGear'; gear.title = 'Settings';
    _sizeGear(gear);          // not yet in the DOM, so pass it directly
    gear.onmouseenter = () => gear.style.opacity = '1';
    gear.onmouseleave = () => gear.style.opacity = '.6';

    const panel = mk('div',
        'position:fixed; top:82px; right:12px; z-index:41; display:none;' +
        'background:#fff; color:#28313b; font-family:' + HUD_FONT + '; font-size:13px;' +
        'border:1px solid rgba(0,0,0,.15); border-radius:12px; padding:12px 14px; width:216px;' +
        // A landscape phone is ~390px tall, far shorter than this panel: without
        // a cap its lower half (sound, tutorial) sat off-screen and unreachable.
        'box-sizing:border-box; max-height:calc(100vh - 94px);' +
        'overflow-y:auto; overscroll-behavior:contain; -webkit-overflow-scrolling:touch;' +
        'box-shadow:0 12px 34px rgba(0,0,0,.22);');
    panel.id = 'settingsPanel';

    // Theme — applied live (no reload, no new game)
    const trow = mk('div', 'margin-bottom:12px;');
    trow.appendChild(mk('div', 'font-weight:600; margin-bottom:4px;', 'Theme'));
    const sel = mk('select', 'width:100%; padding:4px; border-radius:6px; border:1px solid #cfd6e0;');
    Object.keys(THEMES).forEach(k => { const o = mk('option', null, THEMES[k].label);
        o.value = k; if (k === _themeKey) o.selected = true; sel.appendChild(o); });
    sel.onchange = () => applyThemeLive(sel.value);
    trow.appendChild(sel); panel.appendChild(trow);

    // Difficulty
    const drow = mk('div', 'margin-bottom:12px;'); drow.id = 'settingsDiff';
    const dhead = mk('div', 'font-weight:600; margin-bottom:4px; display:flex; justify-content:space-between;');
    dhead.appendChild(mk('span', null, 'Difficulty'));
    const dval = mk('span', 'font-weight:400;'); dhead.appendChild(dval);
    drow.appendChild(dhead);
    const slider = mk('input', 'width:100%; cursor:pointer;');
    slider.type = 'range'; slider.min = '0'; slider.max = '100'; slider.step = '5';
    slider.value = String(Math.round(getAIDifficulty() * 100));
    const labelFor = (d) => d >= 0.99 ? 'Max' : d <= 0.01 ? 'Easy' : Math.round(d * 100) + '%';
    dval.textContent = labelFor(getAIDifficulty());
    slider.oninput = () => { const d = parseInt(slider.value) / 100; dval.textContent = labelFor(d);
        try { localStorage.setItem('aiDifficulty', String(d)); } catch (e) {} };
    drow.appendChild(slider);
    const dnote = mk('div', 'font-size:11px; color:#8b95a3; margin-top:2px; display:none;', 'Locked during a match');
    dnote.id = 'settingsDiffNote'; drow.appendChild(dnote);
    panel.appendChild(drow);

    // Who plays each colour. Locked during a match: swapping a side mid-match
    // would make the running score meaningless.
    const prow = mk('div', 'margin-bottom:10px;');
    prow.id = 'settingsPlayers';
    prow.appendChild(mk('div', 'font-weight:600; margin-bottom:4px;', 'Players'));
    const mkSide = (label, isAI, save) => {
        const row = mk('div', 'display:flex; align-items:center; gap:8px; margin:4px 0;');
        row.appendChild(mk('span', 'width:44px;', label));
        const seg = makeSegmented([['human', 'Human'], ['computer', 'Computer']],
                                  isAI() ? 'computer' : 'human',
                                  (v) => { save(v === 'computer'); applyPlayerRoles(); });
        row.appendChild(seg);
        return row;
    };
    prow.appendChild(mkSide('White', () => WHITE_IS_AI, (v) => {
        WHITE_IS_AI = v;
        try { localStorage.setItem('whiteIsAI', v ? '1' : '0'); } catch (e) {}
    }));
    prow.appendChild(mkSide('Black', () => BLACK_IS_AI, (v) => {
        BLACK_IS_AI = v;
        try { localStorage.setItem('blackIsAI', v ? '1' : '0'); } catch (e) {}
    }));
    prow.appendChild(mk('div', 'font-size:11px; color:#8b95a3; margin-top:2px; display:none;',
                        'Locked during a match')).id = 'settingsPlayersNote';
    panel.appendChild(prow);

    // Sound
    const srow = mk('label', 'display:flex; align-items:center; gap:8px; cursor:pointer; margin-bottom:8px;');
    const sfxBox = mk('input'); sfxBox.type = 'checkbox'; sfxBox.checked = getSoundEnabled();
    sfxBox.onchange = () => {
        try { localStorage.setItem('sound', sfxBox.checked ? '1' : '0'); } catch (e) {}
        if (sfxBox.checked) SFX.save();          // a sample of what you just enabled
    };
    srow.appendChild(sfxBox); srow.appendChild(mk('span', null, 'Sound effects'));
    panel.appendChild(srow);

    // Boolean toggles
    const toggle = (labelText, get, key, marginBottom) => {
        const row = mk('label', 'display:flex; align-items:center; gap:8px; cursor:pointer;' +
            (marginBottom ? ' margin-bottom:8px;' : ''));
        const cb = mk('input'); cb.type = 'checkbox'; cb.checked = get();
        cb.onchange = () => { try { localStorage.setItem(key, cb.checked ? '1' : '0'); } catch (e) {} };
        row.appendChild(cb); row.appendChild(mk('span', null, labelText));
        panel.appendChild(row);
    };
    // Phones only, and only where the API exists.
    if (_isPhone() && _fullscreenSupported()) {
        const frow = mk('label', 'display:flex; align-items:center; gap:8px; cursor:pointer; margin-bottom:8px;');
        const fsBox = mk('input'); fsBox.type = 'checkbox'; fsBox.id = 'settingsFullscreen';
        fsBox.checked = !!document.fullscreenElement || getFullscreenPref();
        fsBox.onchange = () => {
            try { localStorage.setItem('fullscreen', fsBox.checked ? '1' : '0'); } catch (e) {}
            fsBox.checked ? _enterFullscreen() : _exitFullscreen();
        };
        // the user can leave fullscreen with a system gesture; keep the box honest
        document.addEventListener('fullscreenchange', () => { fsBox.checked = !!document.fullscreenElement; });
        frow.appendChild(fsBox); frow.appendChild(mk('span', null, 'Fullscreen'));
        panel.appendChild(frow);
    }
    toggle('Move & capture effects', getFeedbackEnabled, 'fxEnabled', true);
    toggle('End turn automatically when both dice used', getAutoEndTurn, 'autoEndTurn', true);
    toggle('Confirm ending a turn with a move left', getConfirmRiskyEnd, 'confirmRiskyEnd', false);
    toggle('Double-click sends a piece to its goal', getSumToGoal, 'sumToGoal', false);
    toggle('Automatic en-route capture', getAutoEnRouteCapture, 'autoEnRoute', true);

    // Interactive tutorial launcher
    const tut = mk('button',
        'width:100%; margin-top:12px; padding:8px 0; border-radius:8px; border:none; cursor:pointer;' +
        'font-family:' + HUD_FONT + '; font-weight:700; font-size:13px; background:' + THEME.accentCss + '; color:#fff;',
        'Interactive tutorial');
    tut.onclick = () => {
        panel.style.display = 'none';
        // Settings is reachable from the welcome card now, and the tutorial
        // replaces the board underneath it -- so the card has to go, exactly as
        // it does when the tutorial is launched from the card's own button.
        const wel = document.getElementById('welcomeScreen'); if (wel) wel.remove();
        startTutorial();
    };
    panel.appendChild(tut);

    document.body.appendChild(gear); document.body.appendChild(panel);

    // The welcome card (z 56) and match setup (z 60) sit OVER the gear's own
    // z-index of 41, so on launch there was no way to reach settings before the
    // first game began -- and who plays which colour is exactly the thing you
    // want to set BEFORE playing. Raise the gear above those two while either
    // is up. NOT above How to Play, which opens from the welcome card and would
    // otherwise have a gear floating over it, and deliberately still below the
    // coin flip (65) and confirm (70), which are transient and modal.
    const SETTINGS_Z_BASE = '41', SETTINGS_Z_OVER = '61';
    const syncSettingsZ = () => {
        const over = !!(document.getElementById('welcomeScreen') || document.getElementById('matchSetup'))
                     && !document.getElementById('howToPlay');
        const z = over ? SETTINGS_Z_OVER : SETTINGS_Z_BASE;
        gear.style.zIndex = z; panel.style.zIndex = z;
    };
    // Driven off the DOM rather than from each show/hide site: the welcome card
    // is removed from four different places, and one missed restore would
    // strand the gear above everything for the rest of the session.
    try {
        new MutationObserver(syncSettingsZ).observe(document.body, { childList: true });
    } catch (e) { /* no MutationObserver: the gear just keeps its base z-index */ }
    syncSettingsZ();

    gear.onclick = (e) => { e.stopPropagation();
        const show = panel.style.display === 'none';
        panel.style.display = show ? 'block' : 'none';
        if (show) refreshSettingsMatchState();
    };
    document.addEventListener('pointerdown', (e) => {
        if (panel.style.display !== 'none' && !panel.contains(e.target) && e.target !== gear)
            panel.style.display = 'none';
    }, true);
    refreshSettingsMatchState();
}
// A small always-available "?" legend (bottom-right) explaining the few board
// symbols a newcomer can't name: the home tile, the goal wedges, greyed dice, +N.
function createLegendButton() {
    if (document.getElementById('legendBtn')) return;
    const mk = (tag, css, txt) => { const e = document.createElement(tag);
        if (css) e.style.cssText = css; if (txt != null) e.innerHTML = txt; return e; };

    const btn = mk('button',
        'position:fixed; bottom:12px; right:12px; z-index:41; width:30px; height:30px;' +
        'border-radius:50%; border:1px solid rgba(0,0,0,.15); background:rgba(255,255,255,.75);' +
        'color:#28313b; font-size:15px; font-weight:700; cursor:pointer; opacity:.55; transition:opacity .15s;', '?');
    btn.id = 'legendBtn'; btn.title = 'Legend';
    btn.onmouseenter = () => btn.style.opacity = '1';
    btn.onmouseleave = () => btn.style.opacity = '.55';

    const pop = mk('div',
        'position:fixed; bottom:50px; right:12px; z-index:41; display:none; width:250px;' +
        'background:#fff; color:#28313b; font-family:' + HUD_FONT + '; font-size:12.5px; line-height:1.45;' +
        'border:1px solid rgba(0,0,0,.15); border-radius:12px; padding:12px 14px;' +
        'box-shadow:0 12px 34px rgba(0,0,0,.22);');
    pop.id = 'legendPop';
    const dot = (c) => '<span style="display:inline-block; width:11px; height:11px; border-radius:50%;' +
        'background:' + c + '; border:1px solid rgba(0,0,0,.35); vertical-align:middle; margin-right:7px;"></span>';
    const cssHex = (n) => '#' + n.toString(16).padStart(6, '0');
    pop.innerHTML =
        '<div style="font-weight:700; margin-bottom:7px;">Legend</div>' +
        '<div style="margin-bottom:6px;">' + dot(cssHex(THEME.hub)) + '<b>Home tile</b> — the disc at the centre; pieces enter here.</div>' +
        '<div style="margin-bottom:6px;">' + dot(cssHex(THEME.goal)) + '<b>Goals</b> — the six numbered wedges on the rim; save pieces here.</div>' +
        '<div style="margin-bottom:6px;">' + dot('#c9ced6') + '<b>Greyed die</b> — already used this turn.</div>' +
        '<div>' + dot('#e7ebf1') + '<b>+N badge</b> — a stack; tap to pick a piece out of it.</div>';

    btn.onclick = () => { pop.style.display = pop.style.display === 'none' ? 'block' : 'none'; };
    document.addEventListener('pointerdown', (e) => {
        if (pop.style.display === 'block' && e.target !== btn && !pop.contains(e.target)) pop.style.display = 'none';
    }, true);
    document.body.appendChild(btn); document.body.appendChild(pop);
}

// One-time toast for brand-new visitors, pointing at How to Play.
function maybeShowFirstRunNudge() {
    let seen = false;
    try { seen = localStorage.getItem('seenNudge') === '1'; } catch (e) {}
    // The welcome screen already offers How to Play / Tutorial, so don't stack a
    // nudge on top of it.
    if (seen || document.getElementById('firstRunNudge') || document.getElementById('welcomeScreen')) return;
    try { localStorage.setItem('seenNudge', '1'); } catch (e) {}

    const t = document.createElement('div');
    t.id = 'firstRunNudge';
    t.style.cssText = 'position:fixed; left:50%; bottom:18px; transform:translateX(-50%) translateY(12px);' +
        'z-index:55; background:#28313b; color:#fff; font-family:' + HUD_FONT + '; font-size:13.5px;' +
        'padding:11px 16px; border-radius:11px; box-shadow:0 12px 30px rgba(0,0,0,.3);' +
        'display:flex; align-items:center; gap:12px; opacity:0; transition:opacity .3s, transform .3s; max-width:90vw;';
    const msg = document.createElement('span');
    msg.textContent = 'New here? ';
    const link = document.createElement('button');
    link.textContent = 'How to Play';
    link.style.cssText = 'background:' + THEME.accentCss + '; color:#fff; border:none; border-radius:7px;' +
        'padding:5px 11px; font-weight:700; font-size:13px; cursor:pointer; font-family:' + HUD_FONT + ';';
    const dismiss = () => { t.style.opacity = '0'; t.style.transform = 'translateX(-50%) translateY(12px)';
        setTimeout(() => t.remove(), 320); };
    link.onclick = () => { dismiss(); showInstructions(); };
    const x = document.createElement('button');
    x.textContent = '✕';
    x.style.cssText = 'background:none; border:none; color:#aab3bf; font-size:14px; cursor:pointer; padding:0 2px;';
    x.onclick = dismiss;
    t.appendChild(msg); t.appendChild(link); t.appendChild(x);
    document.body.appendChild(t);
    requestAnimationFrame(() => { t.style.opacity = '1'; t.style.transform = 'translateX(-50%) translateY(0)'; });
    setTimeout(() => { if (document.body.contains(t)) dismiss(); }, 11000);
}

// ── Guided interactive tutorial ─────────────────────────────────────────────
// A scripted walkthrough of one whole game, opening to win. Each step declares
// an exact position (board, racks, dice), the moves it will accept, and a
// success condition polled a few times a second. Anything the script didn't ask
// for is refused: _tutMoveOK is consulted inside getReachableTilesByDice, so
// off-script destinations are neither highlighted nor accepted, and _tutSaveOK /
// _tutBlockSaveOK gate the two save gestures. Black's replies are scripted
// slides rather than the AI (suppressed via window._tutorialActive), and
// switchTurn is short-circuited so the script owns the dice and the turn order.
const _tut = { active: false, step: 0, timer: null, bubble: null,
               turnEnded: false, busy: false, shake: null };

function _tutStep() { return _tut.active ? _tutSteps[_tut.step] : null; }
function _tutRack(game, player, kind) {
    if (kind === 'saved') return player === 'white' ? game.whiteSavedRack : game.blackSavedRack;
    return player === 'white' ? game.whiteUnenteredRack : game.blackUnenteredRack;
}
function _tutHub(game) { return game.tiles.find(t => t.type === 'home'); }
function _tutTile(game, r, s) { return game.tiles.find(t => t.ring === r && t.sector === s); }
function _tutPiece(game, player, n) { return game.pieces.find(p => p.player === player && p.number === n); }
function _tutGoal(game, n) { return game.tiles.find(t => t.type === 'save' && t.number === n); }
function _at(tile, r, s) { return !!tile && tile.ring === r && tile.sector === s; }
function _tutSavedCount(game, player) { return _tutRack(game, player, 'saved').pieces.length; }
// the piece a rack-entering move is really about (it sits on the home tile mid-entry)
function _tutIsFrontRack(game, piece) {
    const r = _tutRack(game, piece.player, 'unentered');
    return piece.rack === r ? r.pieces[0] === piece : piece.justMovedHome;
}

// Lay out a whole position from a declarative spec. board/saved/rack together
// must name all 12 pieces per side; rack order is the order given (front first).
function _tutApply(game, spec) {
    game.selectedPiece = null;
    if (game.unhighlightAllTiles) game.unhighlightAllTiles();
    ['white', 'black'].forEach(pl => {
        const s = spec[pl] || {};
        const unentered = _tutRack(game, pl, 'unentered');
        const saved = _tutRack(game, pl, 'saved');
        game.pieces.filter(p => p.player === pl).forEach(p => {
            p.isSelected = false; p.justMovedHome = false;
            _setupPlaceInRack(p, unentered, false);
        });
        (s.board || []).forEach(([n, rs]) => {
            const piece = _tutPiece(game, pl, n), tile = _tutTile(game, rs[0], rs[1]);
            if (piece && tile) _setupPlaceOnTile(piece, tile);
        });
        (s.saved || []).forEach(n => {
            const piece = _tutPiece(game, pl, n);
            if (piece) _setupPlaceInRack(piece, saved, false);
        });
        // re-place the rack pieces in the stated order (removes + appends)
        (s.rack || []).forEach(n => {
            const piece = _tutPiece(game, pl, n);
            if (piece) _setupPlaceInRack(piece, unentered, false);
        });
        game.pieces.filter(p => p.player === pl).forEach(p => p.updateColor && p.updateColor());
    });
}

// Re-derive both players' phases from the laid-out position, the same way the
// game does it (canBeSaved() reads the phase, so midgame has to be set first).
function _tutPhases(game) {
    [['white', 0xffffff, 0], ['black', 0x000000, 1]].forEach(([name, color, idx]) => {
        const player = game.players[idx];
        if (_tutRack(game, name, 'unentered').pieces.length) { player.setGamePhase('opening'); return; }
        player.setGamePhase('midgame');
        const mine = game.pieces.filter(p => p.color === color);
        if (mine.every(p => p.canBeSaved())) player.setGamePhase('endgame');
    });
}

function _tutSetDice(game, a, b) {
    game.dice[0].value = a; game.dice[0].used = false;
    game.dice[1].value = b; game.dice[1].used = false;
    game.dice.forEach(d => d.updateColor('white'));
}

// Re-derive per-turn state after a hand-built position so the board is playable.
function _tutRefresh(game) {
    game.turn = 'white';
    game.movedOnce = false;
    game.gameOver = false;
    _tut.turnEnded = false;
    game.undoStack = [];
    game._pendingPreMove = null;
    if (typeof clearMoveRecording === 'function') clearMoveRecording();
    game.pieces.forEach(p => { p.reachableTiles = null; p._turnStartTile = p.currentTile || null; });
    // switchTurn is short-circuited during the tutorial, so nothing else
    // refreshes game.state -- without this, undo would restore the stale
    // snapshot from the last real turn (the untouched opening position, with
    // the racks in their original order) instead of this step's start.
    game.state = game.captureState();
    if (game.updateMovablePieces) game.updateMovablePieces();
    if (typeof updateMustMoveHighlights === 'function') updateMustMoveHighlights(game);
    if (typeof updateTurnStatus === 'function') updateTurnStatus(game);
    game.dice.forEach(d => d.updateColor('white'));
}

// ── hooks called from the game itself (all no-ops outside the tutorial) ──────
function _tutMoveOK(game, piece, tile) {
    const step = _tutStep();
    if (!step) return true;
    try { return !!(step.move && step.move(game, piece, tile)); } catch (e) { return false; }
}
function _tutSaveOK(piece) {
    const step = _tutStep(); if (!step) return true;
    try { return !!(step.save && step.save(piece.game, piece)); } catch (e) { return false; }
}
function _tutBlockSaveOK(piece) {
    const step = _tutStep(); if (!step) return true;
    try { return !!(step.blockSave && step.blockSave(piece.game, piece)); } catch (e) { return false; }
}
function _tutFilterReach(game, piece, r) {
    const ok = t => _tutMoveOK(game, piece, t);
    return { reachableByFirstDie: r.reachableByFirstDie.filter(ok),
             reachableBySecondDie: r.reachableBySecondDie.filter(ok),
             reachableBySum: r.reachableBySum.filter(ok) };
}
// switchTurn during the tutorial: the script owns the turn, so only note that
// the player ended it (and only where the step asks them to).
function _tutTurnEnd() {
    const step = _tutStep();
    if (step && step.allowEndTurn) _tut.turnEnded = true;
    else _tutNudge();
}
// The instruction bubble would otherwise sit on top of the bottom of the board
// (goals 2 and 4 live down there, and most of the second half of the script
// happens around them). Shrink Phaser's fit box by the bubble's height instead,
// so the whole board stays visible — setParentSize is the supported path and
// keeps pointer mapping correct, unlike a CSS transform on the canvas.
// The scale manager's parent here is the window itself, so it re-reads the full
// window size every resizeInterval and would undo the override half a second
// later — hence parking the poll while the tutorial holds a smaller fit box.
function _tutLayout() {
    // Side-by-side only on a landscape phone, where the viewport is too short
    // to give any height away to the text and the card would sit on the board.
    // Anything with real height -- desktop, tablet, portrait phone -- stacks
    // the card under the board, which reads better and keeps the board wide.
    const w = window.innerWidth, h = window.innerHeight;
    return (h <= 560 && w / h >= 1.25) ? 'side' : 'bottom';
}

function _tutFitBoard() {
    const s = (typeof gameInstance !== 'undefined') && gameInstance.scale; if (!s) return;
    if (!_tut.active || !_tut.bubble) {
        s.resizeInterval = _tut.resizeInterval || 500;
        s.setParentSize(window.innerWidth, window.innerHeight);
        if (s.canvas) { s.canvas.style.marginLeft = ''; s.canvas.style.marginTop = ''; }
        return;
    }
    if (_tut.resizeInterval === undefined) _tut.resizeInterval = s.resizeInterval;
    s.resizeInterval = Number.MAX_SAFE_INTEGER;

    const W = window.innerWidth, H = window.innerHeight;
    const b = _tut.bubble;
    const mode = _tutLayout();
    const gap = 16;

    // Phones use Scale.NONE and the canvas always fills the viewport, so the
    // board CANNOT be shrunk to make room -- setParentSize does nothing here.
    // Stacking the bubble under the board therefore put it below the fold
    // (measured: top 860 on an 844-tall screen, so it never appeared). Overlay
    // it on the board instead, pinned to the bottom of the viewport; it has a
    // solid background and sits above the canvas.
    if (_isPhone()) {
        // A FIXED box, identical for every step: the size used to follow the
        // step's content, so later steps grew and crept over the board (owner
        // saw step 1 clear of it and step 2 onward covering a quarter).
        if (mode === 'side') {
            // Size it to the free column BESIDE the board rather than a
            // fraction of the screen: the board is centred, so the space either
            // side is (screen - board)/2, and anything wider necessarily covers
            // part of it. Measured from the camera so it follows any zoom.
            // Measure ONCE per tutorial and reuse it. worldView changes when a
            // step zooms the camera, so recomputing per step made the width
            // vary -- and with the right edge pinned, the left edge walked. That
            // is the sideways drift left after anchoring the top.
            if (!_tut._cardW) {
                let free = Math.round(W * 0.32);
                const cam = _mainCamera(), cv = gameInstance && gameInstance.canvas;
                const rect = cv && cv.getBoundingClientRect();
                if (cam && rect && cam.worldView.width) {
                    const boardCss = 1080 * (rect.width / cam.worldView.width);  // board is 1080 world px
                    free = Math.floor((W - boardCss) / 2) - 2 * gap;
                }
                _tut._cardW = Math.max(180, Math.min(300, free));
            }
            const bw = _tut._cardW;
            b.style.width = bw + 'px';
            b.style.left = 'auto';
            b.style.right = gap + 'px';
            // Anchor the TOP, not the centre. Vertically centring means a step
            // with more text grows both ways, so the card appears to move
            // between steps even though its box rules never changed -- which is
            // the drift owner still saw after the width was fixed.
            b.style.top = gap + 'px';
            b.style.bottom = 'auto';
            b.style.transform = _tut._xform = 'none';
            b.style.maxHeight = (H - 2 * gap) + 'px';
        } else {
            b.style.width = 'min(640px, ' + (W - 2 * gap) + 'px)';
            b.style.left = '50%';
            b.style.right = 'auto';
            b.style.top = 'auto';
            b.style.bottom = gap + 'px';
            b.style.transform = _tut._xform = 'translateX(-50%)';
            // Portrait: cap the card to the band BELOW the lower rack rather
            // than to a fraction of the screen, so it cannot cover black's
            // pieces. That band is what hiding the score stack, and the lift in
            // _tutLift, are for. Falls back to the fraction if the camera has
            // not rendered a frame yet.
            let cap = Math.round(H * 0.45);
            if (_isPortrait()) {
                const f = _fur(), pr = _rackPR(), spacing = pr * 2 + 12;
                // Rack panel bottom: drawBackground runs from y - pr for
                // rows*spacing + pr + verticalPadding.
                const below = Math.max(f.whiteUn[1], f.blackUn[1]) + f.rows * spacing + 22;
                const cssY = _worldYToCss(below);
                if (cssY !== null) cap = Math.max(120, Math.round(H - cssY - 2 * gap));
            }
            b.style.maxHeight = cap + 'px';
        }
        b.style.overflowY = 'hidden';   // #tutText scrolls instead, so the buttons stay put
        s.setParentSize(W, H);
        if (s.canvas) { s.canvas.style.marginLeft = ''; s.canvas.style.marginTop = ''; }
        return;
    }

    if (mode === 'side') {
        // Text in a column beside the board, board pinned to the other side.
        const bw = Math.min(380, Math.round(W * 0.34));
        b.style.width = bw + 'px';
        b.style.left = 'auto';
        b.style.right = gap + 'px';
        b.style.bottom = 'auto';
        b.style.top = '50%';
        b.style.transform = _tut._xform = 'translateY(-50%)';
        b.style.maxHeight = (H - 2 * gap) + 'px';
        b.style.overflowY = 'hidden';   // #tutText scrolls instead, so the buttons stay put
        _tutMeasureBubble(bw);
        s.setParentSize(Math.max(320, W - bw - 2 * gap), H);
        if (s.canvas) { s.canvas.style.marginLeft = '0px'; s.canvas.style.marginTop = ''; }
    } else {
        // Stacked: board on top, text under it. The pair is centred as a group,
        // so a short board doesn't leave a chasm between the two.
        const bw = Math.round(Math.min(640, W * 0.92));
        b.style.width = bw + 'px';                // explicit: clearing it would
        b.style.right = 'auto';                   // collapse the card to fit-content
        b.style.left = '50%';
        b.style.bottom = 'auto';
        b.style.transform = _tut._xform = 'translateX(-50%)';
        // On a tall narrow screen the text would otherwise eat most of the
        // height; cap it and let the longest steps scroll.
        const cap = Math.round(H * 0.45);
        b.style.maxHeight = cap + 'px';
        b.style.overflowY = 'hidden';   // #tutText scrolls instead, so the buttons stay put
        _tutMeasureBubble(bw);
        const bh = Math.min(cap, _tut.bubbleH || Math.round(b.getBoundingClientRect().height));
        s.setParentSize(W, Math.max(200, H - bh - 2 * gap));
        if (s.canvas) {
            const ch = s.canvas.getBoundingClientRect().height;
            const top = Math.max(0, Math.round((H - (ch + gap + bh)) / 2));
            s.canvas.style.marginTop = top + 'px';
            s.canvas.style.marginLeft = '';
            b.style.top = (top + ch + gap) + 'px';
        }
    }
    // Re-read the canvas bounds after moving it, or pointer positions stay
    // offset by the margins we just changed.
    if (s.canvas) s.updateBounds();
}
// The tutorial is stripped to board, racks, dice and arrows on every platform:
// no HUD buttons (they restart the scene out from under the runner), no score
// line, no impasse counter, no Call draw, no turn pill, no gear. None of it
// applies to a scripted game, and in portrait the bottom band it occupies is
// what the card needs.
function _tutHudVisible(on) {
    const scene = _setupScene();
    if (scene && scene.hudButtons) scene.hudButtons.forEach(b => b.setHudVisible && b.setHudVisible(on));
    if (scene) {
        if (scene.scoreText) scene.scoreText.setVisible(on);
        if (scene.impasseText && !on) scene.impasseText.setVisible(false);
        if (scene.callDrawButton && !on) scene.callDrawButton.setHudVisible(false);
    }
    _sizeGear();
    if (typeof updateTurnStatus === 'function') updateTurnStatus(_currentGame());
    // Coming back out, the counter's own rule decides whether it shows.
    if (on && typeof updateNoSaveDisplay === 'function') updateNoSaveDisplay();
}
window.addEventListener('resize', () => { if (_tut.active) { _tut._cardW = null; setTimeout(_tutFitBoard, 60); } });
window.addEventListener('orientationchange', () => { if (_tut.active) { _tut._cardW = null; setTimeout(_tutFitBoard, 250); } });

// The shake must be applied ON TOP of whatever transform the current layout
// uses, not on top of an assumed one. It used to hard-code translateX(-50%) --
// correct for the bottom-centred card, but the landscape phone card is pinned
// by its RIGHT edge with no transform at all, so one nudge moved it half its
// own width to the left and left it there. That is the sideways drift: it fires
// on an off-script move, which is why stepping through the script never showed
// it.
function _tutNudge() {
    const b = _tut.bubble; if (!b) return;
    clearInterval(_tut.shake);
    const base = _tut._xform || 'translateX(-50%)';
    const at = (dx) => (base === 'none' ? '' : base + ' ') + 'translateX(' + dx + 'px)';
    let n = 0;
    b.style.transition = 'transform .08s ease-in-out';
    _tut.shake = setInterval(() => {
        b.style.transform = at((n % 2) ? 7 : -7);
        if (++n > 3) { clearInterval(_tut.shake); _tut.shake = null; b.style.transform = base; }
    }, 80);
}

// ── the script ───────────────────────────────────────────────────────────────
// Tile shorthand: spokes run home -> ring1 .. ring7 goal. Spoke 2 -> goal 4,
// 4 -> goal 2, 6 -> goal 5, 8 -> goal 3, 10 -> goal 6, 12 -> goal 1. Goals sit
// at [7,2]=4 [7,4]=2 [7,6]=5 [7,8]=3 [7,10]=6 [7,12]=1.
const _TUT_BLACK_MID = [[1, [2, 9]], [2, [3, 10]], [3, [1, 8]], [7, [5, 13]], [12, [3, 1]]];

const _tutSteps = [
    {
        title: 'Send two pieces out',
        text: 'Your pieces wait on the rack. Send the front one out — it steps onto the home tile first, then out along a spoke — and spend the <b>5</b> on the highlighted tile near goal 5. Then bring a second piece out with the <b>3</b>. Pieces 1–6 each have one matching goal; blank pieces can use any.',
        dice: [5, 3],
        pos: { white: { rack: [7, 8, 6, 9, 10, 11, 12, 1, 2, 3, 4, 5] },
               black: { rack: [5, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 6] } },
        move: (g, p, t) => p.player === 'white' && (_at(t, 5, 6) || _at(t, 3, 10)),
        done: g => !!_tutTile(g, 5, 6).pieces.length && !!_tutTile(g, 3, 10).pieces.length,
        black: [{ n: 5, to: [4, 6] }],
    },
    {
        title: 'Numbered pieces head for their goal',
        text: 'Your next piece is the <b>6</b>. A numbered piece can only ever be saved on its own goal, so send it straight there — goal 6 is exactly seven tiles away, and both dice can go on one piece: select the 6, then goal 6 (or drag it there).',
        dice: [3, 4],
        pos: { white: { board: [[7, [5, 6]], [8, [3, 10]]], rack: [6, 9, 10, 11, 12, 1, 2, 3, 4, 5] },
               black: { board: [[5, [4, 6]]], rack: [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 6] } },
        move: (g, p, t) => p.player === 'white' && p.number === 6 && _at(t, 7, 10),
        done: g => _tutPiece(g, 'white', 6).currentTile === _tutGoal(g, 6),
        black: [{ n: 7, to: [5, 10] }],
    },
    {
        title: 'Take what’s exposed',
        text: 'A lone piece on a tile is exposed — land on it and it goes back to the home tile to start over. Black has left two. While you still have pieces on the rack, <b>one of your two moves must be that front rack piece</b> — either order. Enter it with the <b>4</b> onto Black’s 5, and use the <b>2</b> to take the other with the piece already on the board. Black will have to move its captured pieces back out before doing anything else.',
        dice: [4, 2],
        pos: { white: { board: [[7, [5, 6]], [8, [3, 10]], [6, [7, 10]]], rack: [9, 10, 11, 12, 1, 2, 3, 4, 5] },
               black: { board: [[5, [4, 6]], [7, [5, 10]]], rack: [8, 9, 10, 11, 12, 1, 2, 3, 4, 6] } },
        move: (g, p, t) => p.player === 'white' &&
            ((_tutIsFrontRack(g, p) && _at(t, 4, 6)) || (p.number === 8 && _at(t, 5, 10))),
        done: g => _tutHub(g).pieces.filter(p => p.player === 'black').length === 2,
        black: [{ n: 5, to: [4, 4] }, { n: 7, to: [2, 2] }],
    },
    {
        title: 'Build a wall',
        text: 'Two of your pieces on one tile make a <b>wall</b> — enemy pieces can’t land on it or pass through. Black’s 5 still has to come round to goal 5, and the short way in runs over a tile you already hold. Your dice sum to 5: bring your next piece all the way out to join it and shut that route down.',
        dice: [3, 2],
        pos: { white: { board: [[7, [5, 6]], [8, [5, 10]], [9, [4, 6]], [6, [7, 10]]], rack: [10, 11, 12, 1, 2, 3, 4, 5] },
               black: { board: [[5, [4, 4]], [7, [2, 2]]], rack: [8, 9, 10, 11, 12, 1, 2, 3, 4, 6] } },
        move: (g, p, t) => p.player === 'white' && _tutIsFrontRack(g, p) && _at(t, 5, 6),
        done: g => _tutTile(g, 5, 6).pieces.filter(p => p.player === 'white').length >= 2,
    },
    {
        title: 'Saving',
        fast: true,
        text: '<b>⏩ A few turns later.</b> Your rack is empty, so you’re out of the opening and can start saving. A piece on a goal goes out on a die matching that goal’s number: your <b>6</b> is on goal 6 — double-click it, or drag it to your saved rack, and the 6 banks it for a point. Then do the same on <b>goal 1</b> with the 1 — a blank piece can be saved on any goal.',
        dice: [6, 1],
        pos: { white: { board: [[6, [7, 10]], [10, [7, 12]], [4, [3, 3]], [2, [3, 4]],
                                [11, [5, 6]], [12, [3, 6]], [1, [3, 12]], [3, [3, 8]]],
                        saved: [5, 7, 8, 9] },
               black: { board: [[8, [5, 2]], [9, [4, 2]], [10, [5, 21]], [11, [5, 22]]].concat(_TUT_BLACK_MID),
                        saved: [4, 5, 6] } },
        save: (g, p) => p.player === 'white' && (p.number === 6 || p.number === 10),
        done: g => _tutSavedCount(g, 'white') >= 6,
        black: [{ n: 9, to: [6, 2] }, { n: 8, to: [6, 2] }],
    },
    {
        title: 'The long way in',
        text: 'Black has walled the tile in front of goal 4. A piece always takes the shortest route to where you send it — and your 4’s shortest route was <b>five</b> tiles, so a single 5 would have done it. Now the only way in is <b>nine</b>: up the far spoke, through goal 2 and round the outer arc. Luckily, your dice sum to 9, so move your 4 to its goal.',
        dice: [3, 6],
        pos: { white: { board: [[4, [3, 3]], [2, [3, 4]], [11, [5, 6]], [12, [3, 6]], [1, [3, 12]], [3, [3, 8]]],
                        saved: [5, 6, 7, 8, 9, 10] },
               black: { board: [[8, [6, 2]], [9, [6, 2]], [10, [5, 21]], [11, [5, 22]]].concat(_TUT_BLACK_MID),
                        saved: [4, 5, 6] } },
        move: (g, p, t) => p.player === 'white' && p.number === 4 && _at(t, 7, 2),
        done: g => _tutPiece(g, 'white', 4).currentTile === _tutGoal(g, 4),
        black: [{ n: 10, to: [6, 4] }, { n: 11, to: [6, 4] }],
    },
    {
        title: 'Buy the door open',
        text: 'Those two walled tiles are the only ways into goals 2 and 4, so those goals are now sealed — your <b>2</b> has no route home, on any roll, ever. Your dice can’t do anything useful this turn, so spend them on the door: double-click one of the two black pieces on the wall <b>in front of goal 2</b> to <b>save it for Black</b>. It costs both dice and hands Black a point, but the wall drops to a single piece — your 2 has a path again, with something to capture on the way.',
        dice: [3, 5],
        pos: { white: { board: [[4, [7, 2]], [2, [3, 4]], [11, [5, 6]], [12, [3, 6]], [1, [3, 12]], [3, [3, 8]]],
                        saved: [5, 6, 7, 8, 9, 10] },
               black: { board: [[8, [6, 2]], [9, [6, 2]], [10, [6, 4]], [11, [6, 4]]].concat(_TUT_BLACK_MID),
                        saved: [4, 5, 6] } },
        blockSave: (g, p) => p.player === 'black' && _at(p.currentTile, 6, 4),
        done: g => _tutSavedCount(g, 'black') >= 4,
    },
    {
        title: 'The endgame',
        fast: true,
        text: '<b>⏩ Later.</b> Everything you have left is on a goal but one — use the <b>1</b> to step it onto goal 3. Now every piece is on a goal it can be saved from: that’s the <b>endgame</b>, and blank pieces get easier to save — a blank goes out on any die <i>bigger</i> than its goal’s number, as long as you hold no higher goal. Your highest is goal 3, so the <b>5</b> takes a blank straight off it. Numbered pieces never get this; they always need their own number.',
        dice: [1, 5],
        pos: { white: { board: [[2, [7, 4]], [11, [7, 8]], [12, [6, 8]]],
                        saved: [1, 3, 4, 5, 6, 7, 8, 9, 10] },
               black: { board: [[1, [6, 4]], [2, [6, 4]], [3, [3, 10]], [7, [2, 9]]],
                        saved: [4, 5, 6, 8, 9, 10, 11, 12] } },
        move: (g, p, t) => p.player === 'white' && p.number === 12 && _at(t, 7, 8),
        save: (g, p) => p.player === 'white' && p.number > 6 && _at(p.currentTile, 7, 8),
        done: g => _tutSavedCount(g, 'white') >= 10,
        black: [{ n: 3, to: [3, 9] }],
    },
    {
        title: 'Some dice do nothing',
        text: 'The <b>4</b> takes your last blank off goal 3. Your 2 can’t use the 5 — a numbered piece only ever goes out on its own number, and you haven’t rolled a 2. Nothing else to do, so end your turn yourself: the right-hand arrow above the board (or the Enter key).',
        dice: [4, 5],
        pos: { white: { board: [[2, [7, 4]], [11, [7, 8]]],
                        saved: [1, 3, 4, 5, 6, 7, 8, 9, 10, 12] },
               black: { board: [[1, [6, 4]], [2, [6, 4]], [3, [3, 9]], [7, [2, 9]]],
                        saved: [4, 5, 6, 8, 9, 10, 11, 12] } },
        save: (g, p) => p.player === 'white' && p.number > 6 && _at(p.currentTile, 7, 8),
        allowEndTurn: true,
        done: g => _tutSavedCount(g, 'white') >= 11 && _tut.turnEnded,
        black: [{ n: 7, to: [2, 10] }],
    },
    {
        title: 'Your last piece',
        text: 'Your 2 has <b>lost its number</b>. With one piece left at the start of your turn, a numbered piece on its goal becomes blank — so it no longer has to wait for a 2, and any die of 2 or more brings it in. Save it and the game is yours.',
        dice: [5, 3],
        pos: { white: { board: [[2, [7, 4]]], saved: [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] },
               black: { board: [[1, [6, 4]], [2, [6, 4]], [3, [3, 9]], [7, [2, 10]]],
                        saved: [4, 5, 6, 8, 9, 10, 11, 12] } },
        after: g => { g.applyLastPieceRule(); },   // phase is endgame, so the 2 turns blank
        save: (g, p) => p.player === 'white' && !!p.currentTile && p.currentTile.type === 'save',
        done: g => _tutSavedCount(g, 'white') >= 12,
    },
    {
        title: 'You win!',
        text: 'All twelve saved. The game ends the moment your last piece is off the board, and you score the number of pieces your opponent still had out — four. That’s the whole game: enter, move, capture, wall, save. Ready for a real one?',
        finish: true,
        done: () => false,
    },
];

// ── runner ───────────────────────────────────────────────────────────────────
const _TUT_BUBBLE_CSS = 'position:fixed; left:50%; bottom:20px; transform:translateX(-50%);' +
    'z-index:58; width:min(640px,92vw); box-sizing:border-box; background:#fff; color:#28313b;' +
    'font-family:' + HUD_FONT + '; border-radius:14px; padding:15px 18px;' +
    'box-shadow:0 16px 44px rgba(0,0,0,.3); display:flex; flex-direction:column;';
function _tutBubble() {
    if (_tut.bubble) return _tut.bubble;
    const b = document.createElement('div');
    b.id = 'tutBubble';
    b.style.cssText = _TUT_BUBBLE_CSS;
    document.body.appendChild(b);
    _tut.bubble = b;
    return b;
}
function _tutStepHtml(step, idx) {
    return '<div style="font-size:12px; letter-spacing:.04em; text-transform:uppercase; color:#8b95a3; margin-bottom:3px;">' +
            'Tutorial · Step ' + (idx + 1) + ' of ' + _tutSteps.length + '</div>' +
        '<div style="font-weight:700; font-size:17px; margin-bottom:5px;">' + step.title + '</div>' +
        // The TEXT scrolls, not the card: with the card scrolling as a whole,
        // Exit/Skip sit at the end of the flex column and go below the fold on
        // any step taller than the cap -- which on a portrait phone is all of
        // them. min-height:0 is what lets a flex child shrink enough to scroll.
        '<div id="tutText" style="font-family:' + BODY_FONT + '; font-size:14.5px; line-height:1.5;' +
            'color:#33404b; overflow-y:auto; min-height:0; flex:1 1 auto;">' + step.text + '</div>' +
        '<div id="tutBtns" style="display:flex; gap:8px; margin-top:auto; padding-top:13px;' +
            'justify-content:flex-end; min-height:32px; align-items:center; flex:0 0 auto;"></div>';
}
// The board must not resize from step to step, so the bubble reserves the same
// height throughout: measure the tallest step once (off-screen, at the real
// width) and pin the bubble to it. Re-measured on resize, where text rewraps.
function _tutMeasureBubble(width) {
    const probe = document.createElement('div');
    probe.style.cssText = _TUT_BUBBLE_CSS + 'visibility:hidden; bottom:auto; top:0;' +
        (width ? 'width:' + width + 'px;' : '');
    document.body.appendChild(probe);
    let max = 0;
    _tutSteps.forEach((step, i) => {
        probe.innerHTML = _tutStepHtml(step, i);
        max = Math.max(max, probe.offsetHeight);
    });
    probe.remove();
    _tut.bubbleH = max;
}
function _tutRender() {
    const game = _setupGame(); if (!game) return;
    const step = _tutSteps[_tut.step];
    if (step.pos) {
        _tutApply(game, step.pos);
        _tutSetDice(game, step.dice[0], step.dice[1]);
        _tutPhases(game);
        _tutRefresh(game);
        if (step.after) { try { step.after(game); } catch (e) { console.warn('[TUTORIAL] after() failed:', e); } }
    }
    const b = _tutBubble();
    b.innerHTML = _tutStepHtml(step, _tut.step);
    const btns = b.querySelector('#tutBtns');
    const mkBtn = (label, primary, fn) => {
        const el = document.createElement('button');
        el.textContent = label;
        el.style.cssText = 'padding:7px 15px; border-radius:8px; cursor:pointer; font-family:' + HUD_FONT + ';' +
            'font-weight:700; font-size:13px; border:' + (primary ? 'none' : '1px solid #cfd6e0') + ';' +
            'background:' + (primary ? THEME.accentCss : '#fff') + '; color:' + (primary ? '#fff' : '#5a6473') + ';';
        el.onclick = fn; btns.appendChild(el); return el;
    };
    if (step.finish) {
        mkBtn('Finish', true, () => _tutEnd(true));       // Exit would do the same thing
        if (typeof updateTurnStatus === 'function') updateTurnStatus('');   // the game is over
    } else {
        mkBtn('Exit', false, () => _tutEnd(false));
        mkBtn('Skip →', true, _tutNext);
    }
}
function _tutNote(html) {
    const b = _tut.bubble; if (!b) return;
    const btns = b.querySelector('#tutBtns');
    if (btns) btns.innerHTML = html;
}
function _tutNext() {
    _tut.busy = false;
    if (_tut.step >= _tutSteps.length - 1) { _tutEnd(true); return; }
    _tut.step += 1;
    _tutRender();
}
// Black's scripted reply: slide each piece to its new tile, then carry on.
function _tutPlayBlack(game, moves, cb) {
    if (!moves || !moves.length) { cb(); return; }
    // "Black plays…" in the card is the only turn indication the tutorial gives;
    // the pill is suppressed throughout (see turnStatusText).
    _tutNote('<span style="color:#8b95a3; font-weight:700; font-size:13px;">Black plays…</span>');
    let i = 0;
    const next = () => {
        if (!_tut.active) { cb(); return; }
        if (i >= moves.length) { setTimeout(cb, 400); return; }
        const m = moves[i++];
        const piece = _tutPiece(game, 'black', m.n), tile = _tutTile(game, m.to[0], m.to[1]);
        if (piece && tile) {
            const ox = piece.x, oy = piece.y;
            _setupPlaceOnTile(piece, tile);
            if (piece.animateFrom) piece.animateFrom(ox, oy);
        }
        setTimeout(next, 620);
    };
    setTimeout(next, 300);
}
function _tutPoll() {
    if (!_tut.active || _tut.busy) return;
    const game = _setupGame(); if (!game) return;
    const step = _tutSteps[_tut.step];
    let ok = false;
    try { ok = !!(step.done && step.done(game)); } catch (e) { ok = false; }   // transient half-built state
    if (!ok) return;
    _tut.busy = true;
    _tutNote('<span style="color:#3a9e6a; font-weight:700; font-size:14px;">✓ Nice!</span>');
    setTimeout(() => {
        if (!_tut.active) return;
        _tutPlayBlack(game, step.black, _tutNext);
    }, 850);
}
function startTutorial() {
    if (_tut.active) return;
    _tut.active = true; window._tutorialActive = true;
    _tut.step = 0; _tut.busy = false; _tut.turnEnded = false;
    const welcome = document.getElementById('welcomeScreen');
    if (welcome) welcome.remove();     // reachable from the settings panel too
    // The tutorial runs on the welcome screen's held game, but it is a real
    // thing being played and it scripts its own dice -- which stay invisible
    // while the game counts as frozen. Its own guards (_tutorialActive) are
    // what keep the AI out, not this flag.
    _gameFrozen = false;
    _tutHudVisible(false);
    _tutBubble();
    _tutRender();
    // Layout first: the tutorial changes the world rect (see _tutLift), and the
    // card is sized against where the racks end up. worldView is only right
    // after a frame has rendered, so fit the card again on the next one.
    if (typeof _relayoutFurniture === 'function') { _lastPortrait = null; _relayoutFurniture(); }
    _tutFitBoard();
    _sizeGear();
    requestAnimationFrame(() => { if (_tut.active) _tutFitBoard(); });
    clearInterval(_tut.timer); _tut.timer = setInterval(_tutPoll, 300);
}
function _tutEnd(startGame) {
    _tut.active = false; window._tutorialActive = false;
    _tut.busy = false;
    clearInterval(_tut.timer); _tut.timer = null;
    clearInterval(_tut.shake); _tut.shake = null;
    if (_tut.bubble) { _tut.bubble.remove(); _tut.bubble = null; }
    _tut._cardW = null;
    _tutFitBoard();                       // give the board the full window back
    _tutHudVisible(true);
    const scene = _setupScene();
    if (scene && scene.scene) scene.scene.restart({ welcome: true });
}

// Defer to after the whole script has run (this file `defer`s, so the DOM is
// ready; setTimeout ensures later `let` globals like matchTracker are initialised
// before createSettingsPanel -> refreshSettingsMatchState touches them).
function _initChrome() { createSettingsPanel(); createLegendButton(); maybeShowFirstRunNudge();
                        _armFullscreenOnFirstGesture(); }
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', _initChrome);
else setTimeout(_initChrome, 0);

// Small themed pill shown on hover (undo / end-turn arrows). Phaser text can't
// round its own background, so — as with makeHudButton — a graphics rounded
// rect sits behind the label.
function makeHudTip(scene, cx, cy, label) {
    const txt = scene.add.text(cx, cy, label, {
        fontSize: '17px', fontFamily: HUD_FONT, fontStyle: 'bold',
        color: '#ffffff', padding: { x: 10, y: 5 }
    }).setOrigin(0.5).setDepth(6).setVisible(false);
    const b = txt.getBounds();
    const g = scene.add.graphics().setDepth(5).setVisible(false);
    g.fillStyle(0x28313b, 0.92);
    g.fillRoundedRect(b.x, b.y, b.width, b.height, 7);
    return { show: v => { txt.setVisible(v); g.setVisible(v); } };
}

// Rounded pill button with a soft shadow, matching the mockup .btn / .btn.ghost.
// Returns the interactive Text object (callers attach their own pointer handlers);
// a graphics background sits just behind it and tracks its bounds.
// `k` scales the whole button. The in-game HUD keeps k=1 (it has to share the
// corner with the dice and the racks); full-screen overlays pass a bigger k on
// a phone, where 19px of world font is barely 6 CSS px.
function makeHudButton(scene, cx, cy, label, { ghost = false, k = 1 } = {}) {
    const txt = scene.add.text(cx, cy, label, {
        fontSize: Math.round(19 * k) + 'px', fontFamily: HUD_FONT, fontStyle: 'bold',
        color: ghost ? HUD_INK : THEME.accentInk,
        padding: { x: Math.round(16 * k), y: Math.round(9 * k) }
    }).setOrigin(0.5).setDepth(2).setInteractive({ useHandCursor: true });
    const g = scene.add.graphics().setDepth(1);
    // The pill is drawn from the text's CURRENT bounds, so it has to be
    // repainted whenever the text moves or changes size -- moving the text
    // alone left the pill behind, which is what "the label overhangs its
    // button" looked like.
    const paint = () => {
        const b = txt.getBounds();
        const r = 9 * (txt._hudK || k);
        g.clear();
        g.fillStyle(0x000000, 0.12); g.fillRoundedRect(b.x, b.y + 2, b.width, b.height, r);
        if (ghost) {
            g.fillStyle(0xffffff, 1); g.fillRoundedRect(b.x, b.y, b.width, b.height, r);
            g.lineStyle(1, HUD_PANEL_BORDER, 1); g.strokeRoundedRect(b.x, b.y, b.width, b.height, r);
        } else {
            g.fillStyle(THEME.accent, 1); g.fillRoundedRect(b.x, b.y, b.width, b.height, r);
        }
    };
    txt._hudK = k;
    paint();
    txt.bg = g;                       // so callers can show/hide the whole button
    txt.setHudVisible = (v) => { txt.setVisible(v); g.setVisible(v);
        if (txt.input) txt.input.enabled = v; return txt; };
    txt.recolor = () => {             // re-apply theme colours in place (live theme switch)
        txt.setColor(ghost ? HUD_INK : THEME.accentInk);
        paint();
    };
    txt.setHudPosition = (x, y) => { txt.setPosition(x, y); paint(); return txt; };
    // Rotation changes how much room a button has, so it can be rescaled in
    // place; re-rendering the text keeps it crisp where setScale would blur it.
    txt.setHudK = (nk) => {
        if (nk === txt._hudK) return txt;
        txt._hudK = nk;
        txt.setFontSize(Math.round(19 * nk));
        txt.setPadding(Math.round(16 * nk), Math.round(9 * nk));
        paint();
        return txt;
    };
    _themedRedraws.push(txt.recolor);
    return txt;
}


const scoreTracker = {
    games_played: 0,
    white_wins: 0,
    black_wins: 0,
    draws: 0,
    total_score: 0
    };

// ── MATCH SYSTEM ────────────────────────────────────────────────────────
// null = casual single games. When a match is active the bottom score line
// shows this match's running score/wins (resets when a new match starts).
let matchTracker = null;
const MATCH_DEFAULT_GAMES = 6;
const MATCH_DEFAULT_RACE  = 21;

// Casual (non-match) play: the first game of a session picks its starter with a
// coin flip; subsequent New Games alternate from the previous starter.
let _lastGameStarter = null;
function otherPlayer(p) { return p === 'white' ? 'black' : 'white'; }
function nextCasualStarter() {
    return _lastGameStarter ? otherPlayer(_lastGameStarter) : (Math.random() < 0.5 ? 'white' : 'black');
}

function _cap(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

// cfg: { mode:'games'|'race', target:int, tieRule:'extra'|'draw' } -> returns game-1 starter
function startNewMatch(cfg) {
    const first = Math.random() < 0.5 ? 'white' : 'black';
    matchTracker = {
        mode: cfg.mode, target: cfg.target, tieRule: cfg.tieRule || 'extra',
        gamesPlayed: 0, whiteScore: 0, blackScore: 0,
        whiteWins: 0, blackWins: 0, draws: 0,
        firstStarter: first, over: false, winner: null,
    };
    if (typeof refreshSettingsMatchState === 'function') refreshSettingsMatchState();
    return first;
}
// Starter of the k-th game (0-indexed) in the match: alternates from firstStarter.
function matchStarterForGame(k) {
    if (!matchTracker) return (scoreTracker.games_played % 2 === 0) ? 'white' : 'black';
    return (k % 2 === 0) ? matchTracker.firstStarter
                         : (matchTracker.firstStarter === 'white' ? 'black' : 'white');
}
// Record a finished game into the active match; returns true if the match is over.
function recordMatchGame(winner, score) {
    const m = matchTracker; if (!m || m.over) return !!(m && m.over);
    m.gamesPlayed += 1;
    if (winner === 'white') { m.whiteScore += score; m.whiteWins += 1; }
    else if (winner === 'black') { m.blackScore += score; m.blackWins += 1; }
    else { m.draws += 1; }

    if (m.mode === 'race') {
        if (m.whiteScore >= m.target && m.whiteScore >= m.blackScore) { m.over = true; m.winner = 'white'; }
        else if (m.blackScore >= m.target) { m.over = true; m.winner = 'black'; }
    } else { // games: winner by total score after `target` games
        if (m.gamesPlayed >= m.target) {
            if (m.whiteScore > m.blackScore) { m.over = true; m.winner = 'white'; }
            else if (m.blackScore > m.whiteScore) { m.over = true; m.winner = 'black'; }
            // score tied -> break by number of wins
            else if (m.whiteWins > m.blackWins) { m.over = true; m.winner = 'white'; }
            else if (m.blackWins > m.whiteWins) { m.over = true; m.winner = 'black'; }
            // score AND wins tied -> draw or extend by a pair (same criteria)
            else if (m.tieRule === 'draw') { m.over = true; m.winner = 'draw'; }
            else {
                // Extending silently was confusing: you set a 6-game match and
                // suddenly it is showing game 7. Record it so the end-of-game
                // card can say what happened and why.
                m.extendedAt = m.target;
                m.target += 2;
                m.justExtended = true;
            }
        }
    }
    return m.over;
}
// Bottom score line while a match is active (null otherwise).
function matchScoreLine() {
    const m = matchTracker; if (!m) return null;
    const sep = '  \u00B7  ';
    const parts = [`White ${m.whiteScore} (${m.whiteWins}W)`,
                   `Black ${m.blackScore} (${m.blackWins}W)`];
    if (m.draws) parts.push(`Draws ${m.draws}`);
    parts.push(m.mode === 'race' ? `race to ${m.target}`
                                 : `game ${Math.min(m.gamesPlayed + (m.over ? 0 : 1), m.target)} of ${m.target}`);
    const prefix = m.over
        ? (m.winner === 'draw' ? 'Match drawn' : `${_cap(m.winner)} wins the match`)
        : 'Match';
    // On a phone this has to stay left of goal 2's arc (x=630), so break it in
    // two: heading and progress, then the two scores. Once the match is over the
    // progress ("game 4 of 4") is both redundant and too long to fit beside the
    // longer heading, so it is dropped.
    if (_isPhone()) {
        const head = m.over ? prefix : prefix + sep + parts.slice(2).join(sep);
        return head + '\n' + parts.slice(0, 2).join(sep);
    }
    return prefix + sep + parts.join(sep);
}

// Start a fresh game as the first game of a match (called after startNewMatch).
// The coin flip runs first, then the fresh game is started — so nothing (incl.
// a black/AI opener) moves until the flip resolves, mirroring the casual path.
function _startMatchFirstGame(starter) {
    clearMoveRecording();
    showCoinFlip(starter, () => {   // reveal who goes first, then start the game
        if (typeof gameInstance !== 'undefined' && gameInstance && gameInstance.scene) {
            // SceneManager.start() does not stop whatever is currently showing
            // (a scene's own scene.start() does). Starting a match from the
            // end-of-match screen therefore left that screen rendering on top
            // of the new game, so stop anything else that is running first.
            gameInstance.scene.getScenes(true).forEach(sc => {
                const key = sc.scene.key;
                if (key !== 'MainGameScene') gameInstance.scene.stop(key);
            });
            gameInstance.scene.start('MainGameScene', { startingPlayer: starter });
        }
    });
}

// First-load landing screen: a short greeting with Play / How to Play / Tutorial,
// so the session doesn't drop the player straight into the coin flip. Choosing
// Play runs the coin flip to reveal who starts.
function showWelcome(starter) {
    if (document.getElementById('welcomeScreen')) return;
    const raced = document.getElementById('firstRunNudge'); if (raced) raced.remove();
    const box = document.createElement('div');
    box.id = 'welcomeScreen';
    box.style.cssText = 'position:fixed; inset:0; z-index:56; display:grid; place-items:center;' +
        'background:rgba(0,0,0,.55); font-family:' + HUD_FONT + ';';
    const card = document.createElement('div');
    card.style.cssText = 'background:#fff; color:#28313b; border-radius:18px; padding:30px 34px;' +
        'width:min(420px,92vw); box-sizing:border-box; text-align:center; box-shadow:0 20px 55px rgba(0,0,0,.35);';
    card.innerHTML =
        '<div style="font-size:26px; font-weight:800; margin-bottom:6px;">Ready to play?</div>' +
        '<div style="font-family:' + BODY_FONT + '; font-size:15px; line-height:1.5; color:#5a6473; margin-bottom:22px;">' +
        'Race your pieces around the board and bring them all safely home. Play a single game or a multi-game match — new to it? Take a quick tour first.</div>' +
        '<div id="welBtns" style="display:flex; flex-direction:column; gap:10px;"></div>';
    box.appendChild(card); document.body.appendChild(box);
    const holder = card.querySelector('#welBtns');
    const mkBtn = (label, primary, fn) => {
        const el = document.createElement('button');
        el.textContent = label;
        el.style.cssText = 'padding:11px 0; border-radius:10px; cursor:pointer; font-family:' + HUD_FONT + ';' +
            'font-weight:700; font-size:15px; border:' + (primary ? 'none' : '1px solid #cfd6e0') + ';' +
            'background:' + (primary ? THEME.accentCss : '#fff') + '; color:' + (primary ? '#fff' : '#5a6473') + ';';
        el.onclick = fn; holder.appendChild(el);
    };
    // Single game: reveal the starter with the coin flip, then start a *fresh*
    // game (new dice + rack) for that starter — nothing about the game is
    // committed, and the AI never moves, until this point.
    mkBtn('Single game', true, () => {
        box.remove();
        showCoinFlip(starter, () => {
            clearMoveRecording();
            const sc = _setupScene();
            if (sc && sc.scene) sc.scene.restart({ startingPlayer: starter });
        });
    });
    // Match: configure a multi-game match; its own setup handles the coin flip
    // and the fresh first game.
    mkBtn('Play a match', true, () => { box.remove(); showMatchSetup(() => showWelcome(starter)); });
    mkBtn('How to Play', false, () => showInstructions());
    mkBtn('Interactive tutorial', false, () => { box.remove(); startTutorial(); });
}

// A quick coin-flip overlay landing on the player who goes first.
function showCoinFlip(starter, onDone) {
    const old = document.getElementById('coinFlip'); if (old) old.remove();
    const box = document.createElement('div');
    box.id = 'coinFlip';
    box.style.cssText = 'position:fixed; inset:0; z-index:65; display:grid; place-items:center;' +
        'background:rgba(0,0,0,.4); font-family:' + HUD_FONT + ';';
    const coin = document.createElement('div');
    coin.style.cssText = 'width:120px; height:120px; position:relative; transform-style:preserve-3d;';
    const faceCss = (bg, fg) => 'position:absolute; inset:0; border-radius:50%; backface-visibility:hidden;' +
        'display:grid; place-items:center; font-weight:700; font-size:19px;' +
        'box-shadow:0 6px 18px rgba(0,0,0,.4); background:' + bg + '; color:' + fg + ';';
    const white = document.createElement('div');
    white.style.cssText = faceCss('radial-gradient(circle at 35% 30%,#fff,#dfe3e8)', '#28313b'); white.textContent = 'White';
    const black = document.createElement('div');
    black.style.cssText = faceCss('radial-gradient(circle at 35% 30%,#555,#111)', '#fff') + 'transform:rotateY(180deg);'; black.textContent = 'Black';
    coin.appendChild(white); coin.appendChild(black);
    const caption = document.createElement('div');
    caption.style.cssText = 'position:absolute; bottom:34%; color:#fff; font-size:22px; font-weight:600; opacity:0; transition:opacity .3s;';
    box.appendChild(coin); box.appendChild(caption); document.body.appendChild(box);

    const total = 5 * 360 + (starter === 'white' ? 0 : 180);
    try { coin.animate([{ transform: 'rotateY(0deg)' }, { transform: 'rotateY(' + total + 'deg)' }],
        { duration: 1400, easing: 'cubic-bezier(.2,.8,.2,1)', fill: 'forwards' }); }
    catch (e) { coin.style.transform = 'rotateY(' + (starter === 'white' ? 0 : 180) + 'deg)'; }
    setTimeout(() => { caption.textContent = _cap(starter) + ' goes first'; caption.style.opacity = '1'; }, 1250);
    setTimeout(() => { box.style.transition = 'opacity .4s'; box.style.opacity = '0';
        setTimeout(() => { box.remove(); if (onDone) onDone(); }, 400); }, 2500);
}

// Small themed confirm dialog.
function showConfirm(message, onConfirm, confirmLabel) {
    const old = document.getElementById('confirmDlg'); if (old) old.remove();
    const box = document.createElement('div');
    box.id = 'confirmDlg';
    box.style.cssText = 'position:fixed; inset:0; z-index:70; display:grid; place-items:center;' +
        'background:rgba(0,0,0,.42); font-family:' + HUD_FONT + ';';
    const btn = 'font-family:' + HUD_FONT + '; font-weight:700; font-size:15px; padding:9px 18px;' +
        'border-radius:9px; border:none; cursor:pointer;';
    box.innerHTML =
        '<div style="background:#fff; color:#28313b; border-radius:16px; padding:22px 26px;' +
        'width:min(340px,90vw); box-sizing:border-box; box-shadow:0 18px 50px rgba(0,0,0,.3);">' +
        '<div style="font-size:16px; line-height:1.5; margin-bottom:16px;">' + message + '</div>' +
        '<div style="display:flex; gap:10px; justify-content:flex-end;">' +
        '<button id="cNo" style="' + btn + 'background:#eef1f4; color:#28313b;">Cancel</button>' +
        '<button id="cYes" style="' + btn + 'background:' + THEME.accentCss + '; color:#fff;">' +
        (confirmLabel || 'Yes') + '</button>' +
        '</div></div>';
    document.body.appendChild(box);
    box.querySelector('#cNo').onclick = () => box.remove();
    box.querySelector('#cYes').onclick = () => { box.remove(); onConfirm(); };
}

// How-to-Play as a scrollable, sectioned DOM overlay (sans headers, serif body).
function showInstructions() {
    const old = document.getElementById('howToPlay'); if (old) old.remove();
    const box = document.createElement('div');
    box.id = 'howToPlay';
    box.style.cssText = 'position:fixed; inset:0; z-index:60; display:grid; place-items:center;' +
        'background:rgba(0,0,0,.5); font-family:' + HUD_FONT + ';';
    // Two variants: a phone has no mouse and no keyboard, and it has gestures a
    // desktop does not, so the wording differs rather than covering both at once.
    const phone = _isPhone();
    const dbl = phone ? 'double-tap' : 'double-click';
    const sections = [
        ['Goal', 'Be the first to <i>save</i> all your pieces. Your score for a win is the number of pieces your opponent still had left — so winning big is worth more.'],
        ['Your pieces', 'You have 12: six numbered (1–6) and six blank. They start on your side rack.'],
        ['A turn', 'Roll two dice and move. Each die moves one piece a number of tiles equal to that die; you can move one piece with each die, or one piece with both (their sum). A piece always takes the shortest route to the tile you choose, and once it has moved with one die it can’t double back with the other. You may skip a die (or the whole turn).'],
        ['Getting on the board', 'Pieces enter through the home tile — the plain disc at the centre. Only the front piece on your rack can enter, and you must enter at least one piece per turn until your rack is empty (unless you have a captured piece, in which case you must enter that).'],
        ['Capturing &amp; blocking', 'Land on a field tile holding a single enemy piece and you capture it — it goes back to the home tile and its owner must re-enter it before doing anything else. A tile with <b>two or more</b> enemy pieces is a wall: you can’t enter or pass through it.'],
        ['Saving', 'The six coloured wedges on the rim are goals, numbered 1–6. To save a piece, get it onto a goal and roll that goal’s number to lift it off the board. A numbered piece can only be saved from its own goal; a blank piece from any goal. (You can start saving once all your pieces are on the board.)'],
        ['Endgame', 'When every piece you have left is saved or sitting on a goal it can be saved from, you’re in the endgame: blank pieces can now be saved with a roll <i>higher</i> than their goal’s number, as long as you have nothing waiting on a higher-numbered goal.'],
        ['A couple of special moves', '• Break a wall: past the opening and with no captured pieces, ' + dbl + ' (or drag from the picker) one piece of an enemy stack to save it for them — it costs both your dice and hands the opponent a piece, but turns the wall into a lone piece.<br>• Last piece: if you start a turn with a single piece left and it’s a numbered one sitting on its goal, it becomes blank (savable by any roll of that goal number or higher).'],
        ['Stalemate', 'If 10 full rounds pass with nobody saving a piece, either player may call a draw. Any save resets the counter.'],
        ['Matches', 'A match is several games, and it is won on <b>total score</b> — the sum of your winning margins — not on games won. Two formats: a set number of games (highest total score at the end wins), or a race to a target score. Starters alternate; if the scores finish level the match goes to whoever won more games, and if that is level too it is extended by a pair of games. The score line under the board tracks the match.'],
        ['Controls', phone
            ? 'Tap a piece, then tap where it should go — or just drag it there. Drag onto its goal, or double-tap, to save. The ↶ arrow undoes one die at a time; ↷ ends your turn. On a crowded tile the <b>+N</b> badge opens a picker (drag a piece straight out of it). Theme, difficulty and options live under the ⚙ settings, and <b>New Match</b> starts a multi-game match.'
              + '<br>Pinch to zoom, and drag the board to move around it. While you are zoomed in, a piece you still have to enter hovers at the bottom left and the dice appear at the top right — the hovering piece can be tapped, or dragged straight onto the board.'
              + '<br>Settings › <b>Fullscreen</b> hides the browser bars, and stops a swipe from the edge of the screen going back a page.'
            : 'Click a piece, then click where it should go — or just drag it there. Drag onto its goal, or double-click, to save. The ↶ arrow undoes one die at a time; ↷ ends your turn. On a crowded tile the <b>+N</b> badge opens a picker (drag a piece straight out of it). Theme, difficulty and options live under the ⚙ settings, and <b>New Match</b> starts a multi-game match.'
              + '<br>Keyboard: <b>Z</b> undoes one die · <b>Enter</b> or <b>Space</b> ends your turn · <b>Esc</b> deselects the piece you’re holding.'],
    ];
    // Wide two-column card so the whole thing is readable at a glance instead of
    // scrolled through; collapses to one scrolling column on a narrow screen.
    // The columns live in their own unconstrained div: given a fixed height, a
    // multi-column box overflows sideways (silently dropping the last sections)
    // rather than growing downwards.
    let html = '<style>' +
        '#howToPlay h2 { margin:0 0 10px; font-size:30px; }' +
        '#howToPlay h3 { margin:18px 0 4px; font-size:20px; }' +
        '#howToPlay h3:first-of-type { margin-top:0; }' +
        '#howToPlay p { margin:0; font-family:' + BODY_FONT + '; font-size:18px; line-height:1.55;' +
            'color:#33404b; }' +
        '</style><h2>How to Play</h2>';
    sections.forEach(([h, b]) => { html += '<h3>' + h + '</h3><p>' + b + '</p>'; });
    const card = document.createElement('div');
    card.style.cssText = 'position:relative; background:#fff; color:#28313b; border-radius:16px;' +
        'width:min(720px,94vw); max-height:90vh; overflow:hidden; box-sizing:border-box;' +
        'box-shadow:0 18px 50px rgba(0,0,0,.3);';
    const body = document.createElement('div');
    body.className = 'htpBody';
    body.style.cssText = 'padding:26px 30px; max-height:90vh; box-sizing:border-box;' +
        'overflow-y:auto; -webkit-overflow-scrolling:touch;';
    body.innerHTML = html;
    const close = document.createElement('button');
    close.setAttribute('aria-label', 'Close');
    close.textContent = '✕';   // ✕
    close.style.cssText = 'position:absolute; top:9px; right:11px; z-index:2; width:30px; height:30px;' +
        'border:none; border-radius:50%; cursor:pointer; background:rgba(0,0,0,.05); color:#6a7480;' +
        'font-size:17px; line-height:1; display:grid; place-items:center;';
    close.onmouseenter = () => close.style.background = 'rgba(0,0,0,.12)';
    close.onmouseleave = () => close.style.background = 'rgba(0,0,0,.05)';
    close.onclick = () => box.remove();
    card.appendChild(body); card.appendChild(close);
    box.appendChild(card); document.body.appendChild(box);
    box.addEventListener('pointerdown', (e) => { if (e.target === box) box.remove(); });
}

// DOM modal to configure and start a new match. onCancel (optional) runs when
// the user backs out — used by the welcome screen to return to it, since the
// first-load game is still frozen and not yet playable.
// Games are played in colour-swapped pairs, so the count must be even and at
// least 2. `step="2"` on a number input is only checked at validation and never
// while typing -- and a phone renders no spinner at all, so it was simply a
// free-text box. Coerce whatever arrives.
function _evenGames(v) {
    const n = parseInt(v, 10);
    if (!isFinite(n) || n < 2) return MATCH_DEFAULT_GAMES;
    return Math.max(2, n % 2 === 0 ? n : n + 1);
}

function _wireGamesStepper($) {
    const input = $('#mGames');
    if (!input || input._stepperWired) return;
    input._stepperWired = true;
    const step = (delta) => {
        input.value = String(Math.max(2, _evenGames(input.value) + delta * 2));
        input.dispatchEvent(new Event('change'));
    };
    $('#mGamesDown').onclick = () => step(-1);
    $('#mGamesUp').onclick = () => step(1);
    // typing is still allowed; normalise it when the field is left
    input.onblur = () => { input.value = String(_evenGames(input.value)); };
}

function showMatchSetup(onCancel) {
    const old = document.getElementById('matchSetup'); if (old) old.remove();
    const box = document.createElement('div');
    box.id = 'matchSetup';
    box.style.cssText = 'position:fixed; inset:0; z-index:60; display:grid; place-items:center;' +
        'background:rgba(0,0,0,.42); font-family:' + HUD_FONT + ';';
    const btnCss = 'font-family:' + HUD_FONT + '; font-weight:700; font-size:15px; padding:9px 18px;' +
        'border-radius:9px; border:none; cursor:pointer;';
    box.innerHTML =
        '<div style="background:#fff; color:#28313b; border-radius:16px; padding:22px 26px;' +
        'width:min(360px,90vw); box-sizing:border-box; box-shadow:0 18px 50px rgba(0,0,0,.3);">' +
          '<h2 style="margin:0 0 14px; font-size:22px;">New match</h2>' +
          '<label style="display:flex; gap:8px; align-items:center; margin:6px 0; font-size:15px;">' +
            '<input type="radio" name="mmode" value="games" checked> Set number of games (win by total score)</label>' +
          '<div id="gamesOpts" style="margin:2px 0 12px 26px; font-size:14px;">' +
            'Games: <span style="display:inline-flex; align-items:center; gap:6px;">' +
              '<button id="mGamesDown" type="button" style="width:34px; height:34px; font-size:20px; line-height:1;' +
                'border-radius:8px; border:1px solid #cfd6df; background:#f6f8fa; cursor:pointer;">\u2212</button>' +
              '<input id="mGames" type="number" min="2" step="2" value="' + MATCH_DEFAULT_GAMES + '"' +
                ' inputmode="numeric" style="width:56px; text-align:center;">' +
              '<button id="mGamesUp" type="button" style="width:34px; height:34px; font-size:20px; line-height:1;' +
                'border-radius:8px; border:1px solid #cfd6df; background:#f6f8fa; cursor:pointer;">+</button></span>' +
            '<div style="margin-top:8px;">On a tie: ' +
              '<label style="margin-left:4px;"><input type="radio" name="mtie" value="extra" checked> extra pair</label>' +
              '<label style="margin-left:10px;"><input type="radio" name="mtie" value="draw"> draw</label></div></div>' +
          '<label style="display:flex; gap:8px; align-items:center; margin:6px 0; font-size:15px;">' +
            '<input type="radio" name="mmode" value="race"> Race to a total score</label>' +
          '<div id="raceOpts" style="margin:2px 0 12px 26px; font-size:14px; opacity:.5;">' +
            'Target: <input id="mRace" type="number" min="1" value="' + MATCH_DEFAULT_RACE + '" style="width:56px;" disabled></div>' +
          '<div style="margin:14px 0 4px; font-size:15px; font-weight:600;">Players</div>' +
          '<div id="mPlayers" style="font-size:14px; margin-bottom:4px;"></div>' +
          '<div style="font-size:11px; color:#8b95a3; margin-bottom:8px;">Locked once the match starts</div>' +
          '<div style="display:flex; gap:10px; justify-content:flex-end; margin-top:12px;">' +
            '<button id="mCancel" style="' + btnCss + 'background:#eef1f4; color:#28313b;">Cancel</button>' +
            '<button id="mStart" style="' + btnCss + 'background:' + THEME.accentCss + '; color:#fff;">Start match</button>' +
          '</div></div>';
    document.body.appendChild(box);
    const $ = (s) => box.querySelector(s);
    const modeRadios = box.querySelectorAll('input[name=mmode]');
    const sync = () => {
        const mode = [...modeRadios].find(r => r.checked).value;
        $('#gamesOpts').style.opacity = mode === 'games' ? '1' : '.5';
        $('#mGames').disabled = mode !== 'games';
        $('#mGamesDown').disabled = $('#mGamesUp').disabled = mode !== 'games';
        _wireGamesStepper($);
        box.querySelectorAll('input[name=mtie]').forEach(r => r.disabled = mode !== 'games');
        $('#raceOpts').style.opacity = mode === 'race' ? '1' : '.5';
        $('#mRace').disabled = mode !== 'race';
    };
    modeRadios.forEach(r => r.addEventListener('change', sync)); sync();
    const segs = {};
    [['White', WHITE_IS_AI], ['Black', BLACK_IS_AI]].forEach(([side, isAI]) => {
        const row = document.createElement('div');
        row.style.cssText = 'display:flex; align-items:center; gap:8px; margin:4px 0;';
        const lab = document.createElement('span');
        lab.style.cssText = 'width:44px;'; lab.textContent = side;
        row.appendChild(lab);
        segs[side] = makeSegmented([['human', 'Human'], ['computer', 'Computer']],
                                   isAI ? 'computer' : 'human');
        row.appendChild(segs[side]);
        $('#mPlayers').appendChild(row);
    });
    $('#mCancel').onclick = () => { box.remove(); if (onCancel) onCancel(); };
    $('#mStart').onclick = () => {
        const mode = [...modeRadios].find(r => r.checked).value;
        let target, tieRule = 'extra';
        if (mode === 'games') {
            target = _evenGames($('#mGames').value);
            if (target % 2 !== 0) target += 1;                       // keep it even
            tieRule = [...box.querySelectorAll('input[name=mtie]')].find(r => r.checked).value;
        } else {
            target = Math.max(1, parseInt($('#mRace').value) || MATCH_DEFAULT_RACE);
        }
        WHITE_IS_AI = segs.White.value === 'computer';
        BLACK_IS_AI = segs.Black.value === 'computer';
        try {
            localStorage.setItem('whiteIsAI', WHITE_IS_AI ? '1' : '0');
            localStorage.setItem('blackIsAI', BLACK_IS_AI ? '1' : '0');
        } catch (e) {}
        syncSettingsPlayers();
        box.remove();
        const starter = startNewMatch({ mode, target, tieRule });
        _startMatchFirstGame(starter);
    };
}

let extraMoveRequested = false;

// ── HUMAN HALF-MOVE BOOKKEEPING ─────────────────────────────────────────
// What is left of the data-collection globals. The server-facing recording
// chain (game ids, /record_*, /start_game, /abort_game) is gone -- see the
// hosting audit in CLAUDE.md -- but the human move path still fills these as it
// goes, so they are kept and cleared as before.
let _pendingMoves = [];   // moves made this turn, in agent format
let _lastMovePair = null; // complete move pair for the turn

function clearMoveRecording() {
    _pendingMoves = [];
    _lastMovePair = null;
}

// Call before a human move executes; returns the die value that got used.
// Usage: const dieUsed = getDieUsedAfter(game, () => { /* execute move */ });
// Since moves are synchronous in the human path, we snapshot dice before and diff after.
function getDieUsedAfter(game, executeFn) {
    const before = game.dice.map(d => ({ value: d.value, used: d.used }));
    executeFn();
    for (let i = 0; i < game.dice.length; i++) {
        if (!before[i].used && game.dice[i].used) return before[i].value;
    }
    return 0; // both used (sum move) – return 0 as sentinel; caller handles
}

// Record one half-move in agent format and push to _pendingMoves.
// pieceColorNumber: [color_str, number]  e.g. ['white', 3]
// target: [ring, sector] | 'save' | 0  (0 = single-piece block-save)
// die: numeric die value used
function pushHumanMove(pieceColorNumber, target, die) {
    _pendingMoves.push([pieceColorNumber, target, die]);
}

// ── HIDDEN DEBUG TOGGLE ─────────────────────────────────────────────────
// Master switch for the hidden developer modes (triple-press D = debug,
// E = eval readout, S = setup/free-placement). Off by default so public/casual
// builds can never toggle them on; enable for a session with ?dev=1 in the URL.
const ALLOW_DEV_MODES = _DEV_CONSOLE;   // same ?dev=1 switch as the console above

window.debugMode = false;
(function() {
    var tapCount = 0;
    var tapTimer = null;
    document.addEventListener('keydown', function(e) {
        if (!ALLOW_DEV_MODES) return;
        if (e.key !== 'd' && e.key !== 'D') return;
        e.preventDefault();
        tapCount++;
        clearTimeout(tapTimer);
        tapTimer = setTimeout(function() { tapCount = 0; }, 1000);
        if (tapCount >= 3) {
            tapCount = 0;
            window.debugMode = !window.debugMode;
            if (!window.debugMode) hideDebugTip();
            console.log('[DEBUG] mode: ' + (window.debugMode ? 'ON' : 'OFF'));
        }
    });
})();

// ── EVAL DISPLAY TOGGLE (separate from debug) ───────────────────────────
// Triple-press 'E' to toggle a persistent on-board readout of both the GNN
// and heuristic evals for the current position. Also reveals the detailed
// "Evaluate Position" button/panel.
window.showEvals = false;
(function() {
    var tapCount = 0;
    var tapTimer = null;
    document.addEventListener('keydown', function(e) {
        if (!ALLOW_DEV_MODES) return;
        if (e.key !== 'e' && e.key !== 'E') return;
        e.preventDefault();
        tapCount++;
        clearTimeout(tapTimer);
        tapTimer = setTimeout(function() { tapCount = 0; }, 1000);
        if (tapCount >= 3) {
            tapCount = 0;
            window.showEvals = !window.showEvals;
            var scene = gameInstance && gameInstance.scene && gameInstance.scene.scenes && gameInstance.scene.scenes[0];
            if (scene && scene._debugEvalButton) {
                scene._debugEvalButton.setVisible(window.showEvals);
                scene._debugEvalLabel.setVisible(window.showEvals);
            }
            if (window.showEvals) {
                refreshEvalReadout();
            } else {
                hideEvalReadout();
            }
            console.log('[EVALS] display: ' + (window.showEvals ? 'ON' : 'OFF'));
        }
    });
})();

// ── SETUP / SANDBOX MODE ────────────────────────────────────────────────
// Triple-press 's' to toggle. Rearrange pieces, edit dice, set turn, and
// order the unentered racks, to build arbitrary positions for testing.
// Triple-press 'c' to copy the position JSON to clipboard + console.
window.setupMode = false;
let _setupSelected = null;     // piece currently picked up
let _setupBox = null;          // instructions HTML box

function _setupScene() {
    return gameInstance && gameInstance.scene && gameInstance.scene.scenes && gameInstance.scene.scenes[0];
}
function _setupGame() {
    const s = _setupScene();
    return s && s.game;
}
function _setupClearSelection() {
    if (_setupSelected) {
        _setupSelected.isSelected = false;
        _setupSelected.updateColor();
        _setupSelected = null;
    }
}
function setupSelectPiece(piece) {
    if (_setupSelected === piece) { _setupClearSelection(); return; }  // click again = drop selection
    _setupClearSelection();
    _setupSelected = piece;
    piece.isSelected = true;
    piece.updateColor();
}

// --- free placement helpers (bypass all game rules) ---
function _setupRemoveFromCurrent(piece) {
    if (piece.rack) {
        const r = piece.rack;
        r.removePiece(piece);
        if (r.shiftPiecesUp) r.shiftPiecesUp();
        piece.rack = null;
    }
    if (piece.currentTile) {
        piece.currentTile.removePiece(piece);   // updatePositions() repositions remaining pieces
        piece.currentTile = null;
    }
}
// Setup mode edits the position directly, so every value DERIVED from it has to
// be recomputed -- mustMovePieces above all. Leaving it stale meant that after
// sending the front rack piece to the back, the NEW front piece was not the
// obligatory one, and getReachableTilesByDice clears reachableBySum for a
// non-obligatory piece: it silently could not move on a sum (owner). Switching
// the turn with T was worse still, leaving the other player's obligation in
// place. The G handler already did this before sending a position to the agent,
// for exactly the same reason.
function _setupSyncDerived(game) {
    game = game || _setupGame();
    if (!game) return;
    game.updateMovablePieces();
    // The position IS the turn start in setup mode, so clear both the cached
    // destinations and the shortest-path anchor.
    game.pieces.forEach(p => { p.reachableTiles = null; p._turnStartTile = p.currentTile || null; });
}

function _setupPlaceOnTile(piece, tile) {
    _setupRemoveFromCurrent(piece);
    piece.currentTile = tile;
    piece.rack = null;
    piece.justMovedHome = false;
    if (piece.game && piece.game._reorderEntry === piece) piece.game._reorderEntry = null;
    tile.addPiece(piece);            // pushes + updatePositions() => positions & sizes the piece
    _setupSyncDerived(piece.game);
}
function _setupPlaceInRack(piece, rack, atFront) {
    _setupRemoveFromCurrent(piece);
    if (atFront && rack.addPieceToFirstPosition) {
        rack.addPieceToFirstPosition(piece);
    } else {
        rack.addPiece(piece);
    }
    rack.shiftPiecesUp();            // canonical re-layout of the whole rack
    piece.currentTile = null;
    piece.justMovedHome = false;
    if (piece.game && piece.game._reorderEntry === piece) piece.game._reorderEntry = null;
    _setupSyncDerived(piece.game);
}

// Double-click a piece: cycle board -> saved rack -> unentered rack -> board(home).
function setupCyclePieceLocation(piece) {
    const game = _setupGame();
    if (!game) return;
    const saved = piece.player === 'white' ? game.whiteSavedRack : game.blackSavedRack;
    const unentered = piece.player === 'white' ? game.whiteUnenteredRack : game.blackUnenteredRack;
    const home = game.tiles.find(t => t.type === 'home');
    if (piece.rack === saved) {
        _setupPlaceInRack(piece, unentered, false);
    } else if (piece.rack === unentered) {
        _setupPlaceOnTile(piece, home);
    } else {                          // on the board (any tile) or unplaced
        _setupPlaceInRack(piece, saved, false);
    }
    _setupClearSelection();
}

// Reorder the selected piece within whatever rack it is in (front / back).
function setupReorderInRack(piece, toFront) {
    if (!piece || !piece.rack) return;
    const rack = piece.rack;
    rack.removePiece(piece);          // filter out (no relayout needed yet)
    if (toFront) rack.pieces.unshift(piece);
    else rack.pieces.push(piece);
    piece.rack = rack;
    rack.shiftPiecesUp();             // relayout in new order
    _setupSyncDerived(piece.game);    // the FRONT piece changed -> so did the obligation
}

// --- instructions box ---
function _showSetupBox() {
    if (!_setupBox) {
        _setupBox = document.createElement('div');
        _setupBox.style.cssText =
            'position:fixed; top:12px; right:12px; z-index:9999;' +
            'background:rgba(20,20,20,0.9); color:#fff; font:13px/1.6 monospace;' +
            'padding:10px 12px; border:1px solid #888; border-radius:6px; max-width:360px;';
        _setupBox.innerHTML =
            '<b>SETUP MODE</b><br>' +
            '\u2022 Click a piece to pick it up; click a tile to drop it<br>' +
            '\u2022 Double-click a piece to cycle: board \u2192 saved \u2192 unentered \u2192 home<br>' +
            '\u2022 With a piece selected, <b>F</b> = move to front of its rack, <b>B</b> = to back<br>' +
            '\u2022 <b>1</b> / <b>2</b> change a die\u2019s value; <b>Shift+1</b> / <b>Shift+2</b> toggle \u201cused\u201d<br>' +
            '\u2022 <b>T</b> switches whose turn it is<br>' +
            '\u2022 <b>[</b> / <b>]</b> decrease / increase the no-save round counter<br>' +
            '\u2022 <b>G</b> send the current position to the agent (no dice re-roll)<br>' +
            '\u2022 Triple-press <b>C</b> to copy the position JSON (also logged)<br>' +
            '\u2022 Triple-press <b>S</b> to exit';
        document.body.appendChild(_setupBox);
    }
    _setupBox.style.display = 'block';
}
function _hideSetupBox() { if (_setupBox) _setupBox.style.display = 'none'; }
function enterSetupMode() { _showSetupBox(); if (typeof updateNoSaveDisplay === 'function') updateNoSaveDisplay(); }
function exitSetupMode() { _setupClearSelection(); _hideSetupBox(); if (typeof updateNoSaveDisplay === 'function') updateNoSaveDisplay(); }

// --- export current position ---
function exportSetupState() {
    const game = _setupGame();
    if (!game) return;
    const json = JSON.stringify(getGameState(game), null, 2);
    console.log('[SETUP] game state:\n' + json);
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(json)
            .then(() => console.log('[SETUP] copied to clipboard'))
            .catch(e => console.warn('[SETUP] clipboard copy failed (state is logged above):', e));
    }
}

// triple-press 's' : toggle setup mode
(function () {
    let n = 0, t = null;
    document.addEventListener('keydown', function (e) {
        if (!ALLOW_DEV_MODES) return;
        if (e.key !== 's' && e.key !== 'S') return;
        if (e.ctrlKey || e.metaKey) return;     // leave Ctrl/Cmd+S alone
        e.preventDefault();
        n++; clearTimeout(t); t = setTimeout(() => { n = 0; }, 1000);
        if (n >= 3) {
            n = 0;
            window.setupMode = !window.setupMode;
            if (window.setupMode) enterSetupMode(); else exitSetupMode();
            console.log('[SETUP] mode: ' + (window.setupMode ? 'ON' : 'OFF'));
        }
    });
})();

// triple-press 'c' : export position (works in or out of setup mode)
(function () {
    let n = 0, t = null;
    document.addEventListener('keydown', function (e) {
        if (e.key !== 'c' && e.key !== 'C') return;
        if (e.ctrlKey || e.metaKey) return;     // leave Ctrl/Cmd+C alone
        n++; clearTimeout(t); t = setTimeout(() => { n = 0; }, 1000);
        if (n >= 3) { n = 0; exportSetupState(); }
    });
})();

// setup-only keys: 1/2 die value, Shift+1/2 toggle used, t turn, f/b reorder in rack
document.addEventListener('keydown', function (e) {
    if (!window.setupMode) return;
    const game = _setupGame();
    if (!game) return;
    if (e.code === 'Digit1' || e.code === 'Digit2') {
        e.preventDefault();
        const die = game.dice[e.code === 'Digit1' ? 0 : 1];
        if (!die) return;
        if (e.shiftKey) die.used = !die.used;
        else die.value = (die.value % 6) + 1;
        die.updateColor(game.turn);
        _setupSyncDerived(game);      // reachable sets depend on the dice
        console.log(`[SETUP] die ${e.code === 'Digit1' ? 1 : 2}: value ${die.value}, used ${die.used}`);
    } else if (e.key === 't' || e.key === 'T') {
        e.preventDefault();
        game.turn = game.turn === 'white' ? 'black' : 'white';
        game.dice.forEach(d => d.updateColor(game.turn));
        _setupSyncDerived(game);      // the obligation belongs to whoever is to move
        console.log('[SETUP] turn ->', game.turn);
    } else if (e.key === 'f' || e.key === 'F') {
        e.preventDefault();
        if (_setupSelected) setupReorderInRack(_setupSelected, true);
    } else if (e.key === 'b' || e.key === 'B') {
        e.preventDefault();
        if (_setupSelected) setupReorderInRack(_setupSelected, false);
    } else if (e.key === '[' || e.key === ']') {
        e.preventDefault();
        const delta = e.key === ']' ? 1 : -1;
        game.noSaveTurns = Math.max(0, (game.noSaveTurns || 0) + delta);
        game._halfTurnsSinceRound = 0;
        game.lastTotalSaved = game.totalSaved();
        game.drawCallable = game.noSaveTurns >= NO_SAVE_TURNS_FOR_DRAW;
        updateNoSaveDisplay();
        console.log('[SETUP] no-save rounds ->', game.noSaveTurns, 'callable:', game.drawCallable);
    } else if (e.key === 'g' || e.key === 'G') {
        e.preventDefault();
        // Send the current position to the agent as-is (no dice re-roll).
        // Refresh derived per-position state first so the agent's reply isn't
        // rejected against a stale mustMovePieces list from before setup.
        game.updateMovablePieces();
        game.pieces.forEach(p => p.reachableTiles = null);
        console.log('[SETUP] sending current position to agent');
        game.scene.showThinkingIcon();
        getAgentMoves(getGameState(game));
    }
});

// ── ON-BOARD EVAL READOUT ───────────────────────────────────────────────
let _evalReadout = null;
function _getEvalReadout() {
    if (!_evalReadout) {
        _evalReadout = document.createElement('div');
        _evalReadout.style.cssText =
            'position:fixed; top:12px; left:12px; z-index:9998;' +
            'background:rgba(15,15,15,0.88); color:#e8e8e8;' +
            "font-family:'Courier New',monospace; font-size:15px; line-height:1.5;" +
            'border:1px solid #555; border-radius:6px; padding:8px 12px;' +
            'white-space:pre; pointer-events:none;';
        document.body.appendChild(_evalReadout);
    }
    return _evalReadout;
}
function hideEvalReadout() {
    if (_evalReadout) _evalReadout.style.display = 'none';
}
function _fmtAhead(player, value, decimals) {
    if (player === undefined || player === null || value === undefined || value === null) return '—';
    const v = Number(value);
    // value is from `player`'s (side-to-move) perspective; positive means that
    // side is ahead. Relabel to whichever side leads and show a positive magnitude.
    const ahead = v >= 0 ? player : (player === 'white' ? 'black' : 'white');
    const label = ahead.charAt(0).toUpperCase() + ahead.slice(1);
    return label + ' +' + Math.abs(v).toFixed(decimals);
}
function refreshEvalReadout() {
    if (!window.showEvals) return;
    const scene = gameInstance && gameInstance.scene && gameInstance.scene.scenes && gameInstance.scene.scenes[0];
    if (!scene || !scene.game) return;
    const el = _getEvalReadout();
    el.style.display = 'block';
    el.textContent = 'Evaluating…';
    evaluateBoard(getGameState(scene.game)).then(data => {
        if (!window.showEvals) { hideEvalReadout(); return; }
        if (!data) { el.textContent = 'eval unavailable'; return; }
        const margin = (data.gnn_raw == null) ? null : data.gnn_raw * TOTAL_PIECES;
        const best   = (data.gnn_best_margin == null) ? null : data.gnn_best_margin;
        el.textContent =
            'GNN best play: ' + _fmtAhead(data.gnn_player, best, 1) +
            '\nGNN current:   ' + _fmtAhead(data.gnn_player, margin, 1) +
            '\nHeur score:    ' + _fmtAhead(data.gnn_player, data.heur_score, 1);
    });
}

// Piece distance / blot count / saveability, from the device. Returns null if
// the local agent is unavailable, so callers fall back to their own reporting.
function _pieceDebugInfo(gameState, player, number) {
    if (typeof LocalAgent === 'undefined' || !LocalAgent.enabled()) return Promise.resolve(null);
    return LocalAgent.init({ serverUrl: SERVER_URL })
        .then(ok => (ok ? LocalAgent.pieceDebug(gameState, { player: player, number: number }) : null))
        .catch(e => { console.warn('local pieceDebug failed:', e); return null; });
}

// ── DEBUG HOVER TOOLTIP (follows cursor) ────────────────────────────────
let _dbgTip = null, _dbgMouseX = 0, _dbgMouseY = 0;
document.addEventListener('mousemove', function(e) {
    _dbgMouseX = e.clientX; _dbgMouseY = e.clientY;
    if (_dbgTip && _dbgTip.style.display === 'block') {
        _dbgTip.style.left = (_dbgMouseX + 14) + 'px';
        _dbgTip.style.top  = (_dbgMouseY + 14) + 'px';
    }
});
function showDebugTip(text) {
    if (!window.debugMode) return;
    if (!_dbgTip) {
        _dbgTip = document.createElement('div');
        _dbgTip.style.cssText =
            'position:fixed; pointer-events:none; z-index:10001;' +
            'background:rgba(15,15,15,0.9); color:#ffe08a;' +
            "font-family:'Courier New',monospace; font-size:13px;" +
            'padding:2px 7px; border-radius:4px; white-space:nowrap;';
        document.body.appendChild(_dbgTip);
    }
    _dbgTip.textContent = text;
    _dbgTip.style.left = (_dbgMouseX + 14) + 'px';
    _dbgTip.style.top  = (_dbgMouseY + 14) + 'px';
    _dbgTip.style.display = 'block';
}
function hideDebugTip() {
    if (_dbgTip) _dbgTip.style.display = 'none';
}

// ── STACK PICKER (tap-to-pick for overflowed tiles) ─────────────────────
// A tile that overflows hides its extra pieces behind a "+K" badge. Clicking
// such a tile opens this popover listing ALL pieces on the tile; clicking a
// chip selects that piece exactly as clicking it on the board would.
let _stackPicker = null, _pickerOpenedAt = 0;
function ensureStackPicker() {
    if (_stackPicker) return _stackPicker;
    const pop = document.createElement('div');
    pop.style.cssText =
        'position:absolute; display:none; z-index:50; background:#fff;' +
        'border:1px solid #cfd6e0; border-radius:14px; padding:12px;' +
        'box-shadow:0 12px 34px rgba(0,0,0,.22);';
    document.body.appendChild(pop);
    // any outside click dismisses (capture phase; skip the click that opened it).
    // Canvas clicks are left to the Phaser objects (tiles call hideStackPicker,
    // the badge toggles) so the badge can close a picker it opened.
    document.addEventListener('pointerdown', (e) => {
        if (!_stackPicker || _stackPicker.style.display === 'none') return;
        if (_stackPicker.contains(e.target)) return;
        if (e.target && e.target.tagName === 'CANVAS') return;
        if (performance.now() - _pickerOpenedAt < 60) return;
        hideStackPicker();
    }, true);
    _stackPicker = pop;
    return pop;
}
function hideStackPicker() { if (_stackPicker) _stackPicker.style.display = 'none'; }
function stackPickerOpen() { return !!(_stackPicker && _stackPicker.style.display !== 'none'); }
// Ring the obligatory-to-move pieces so the player can see them.
// The rack pieces that may enter, mirroring game.py's get_enterable_pieces:
// the first TWO, so the player chooses which of a turn's two entries goes
// first. Two is the cap because an entry costs one die. Unlike the engine we
// do NOT drop a second blank -- picking either gives the same result, so
// offering both is friendlier, and the engine only dedupes to keep its move
// set small.
function _entrantsOf(rack) {
    if (!rack || rack.type !== 'unentered') return [];
    // Reordering is a FIRST-MOVE privilege: once a die has been spent this turn
    // the rack hands out its front piece and nothing else, or the choice would
    // cascade and a piece could be deferred past the turn (which would change
    // the set of end-of-turn positions -- see game.py get_enterable_pieces).
    const g = _currentGame();
    const played = !!(g && g.dice && g.dice.some(d => d.used));
    return rack.pieces.slice(0, played ? 1 : 2);
}
// The second rack piece, when it is currently offered as an alternative opener.
function _secondEntrant(game) {
    const rack = game && (game.turn === 'white' ? game.whiteUnenteredRack : game.blackUnenteredRack);
    const ent = _entrantsOf(rack);
    return ent.length === 2 ? ent[1] : null;
}
function _isEntrant(piece) {
    return !!piece && !!piece.rack && piece.rack.type === 'unentered'
        && _entrantsOf(piece.rack).includes(piece);
}

function updateMustMoveHighlights(game) {
    if (!game || !game.pieces) return;
    const must = game.mustMovePieces || [];
    game.pieces.forEach(p => { if (p.setObligatory) p.setObligatory(must.includes(p)); });
}
// Clear any current board selection so a picker choice selects cleanly.
function _clearSelection(game) {
    if (game.selectedPiece) {
        const p = game.selectedPiece;
        // A piece that only tentatively entered (sitting on the home tile from
        // this click) belongs back on its rack when the selection is cleared —
        // e.g. pressing Esc — not stranded on the home tile.
        if (p.currentTile && p.currentTile.type === 'home' && p.justMovedHome) {
            p.returnToRack();
        }
        p.isSelected = false;
        p.reachableTiles = null;
        p.updateColor();
        game.unhighlightAllTiles();
        game.selectedPiece = null;
    }
}
function openStackPicker(tile) {
    const pop = ensureStackPicker();
    pop.innerHTML = '';
    const game = tile.game;
    const row = document.createElement('div');
    row.style.cssText = 'display:flex; gap:8px; flex-wrap:wrap; max-width:320px;';
    tile.pieces.forEach(piece => {
        const isW = piece.color === 0xffffff;
        const mine = piece.player === game.turn;   // only own pieces are selectable
        const chip = document.createElement('div');
        chip.style.cssText =
            'width:44px; height:44px; border-radius:50%; display:grid; place-items:center;' +
            'font-family:' + HUD_FONT + '; font-weight:700; font-size:16px; user-select:none;' +
            'transition:transform .08s, box-shadow .08s;' +
            'cursor:' + (mine ? 'pointer' : 'default') + '; opacity:' + (mine ? '1' : '0.45') + ';' +
            'background:' + (isW ? 'radial-gradient(circle at 34% 27%,#fff,#e7ebf0)'
                                 : 'radial-gradient(circle at 34% 27%,#5c5c5c,#171717)') + ';' +
            'color:' + (isW ? '#28313b' : '#fff') + ';' +
            'box-shadow:' + (isW ? 'inset 0 -2px 4px rgba(0,0,0,.12),0 2px 5px rgba(0,0,0,.2)'
                                 : 'inset 0 -2px 4px rgba(0,0,0,.4),0 2px 5px rgba(0,0,0,.3)') + ';';
        chip.textContent = piece.number <= 6 ? piece.number : '';   // unnumbered stay blank, as on the board
        if (mine) {
            const baseShadow = chip.style.boxShadow;
            chip.onmouseenter = () => { chip.style.transform = 'translateY(-3px)';
                chip.style.boxShadow = baseShadow + ',0 0 0 3px ' + THEME.accentCss; };
            chip.onmouseleave = () => { chip.style.transform = ''; chip.style.boxShadow = baseShadow; };
            // A plain click selects the piece; dragging the chip picks the (hidden)
            // piece up and lets you drop it straight onto a board tile.
            chip.onpointerdown = (ev) => {
                if (ev.button !== 0) return;
                ev.preventDefault(); ev.stopPropagation();
                const canvas = document.querySelector('canvas');
                const toGame = (cx, cy) => { const r = canvas.getBoundingClientRect();
                    return [(cx - r.left) * canvas.width / r.width, (cy - r.top) * canvas.height / r.height]; };
                const start = [ev.clientX, ev.clientY];
                let dragging = false;
                const selectPiece = () => {
                    _clearSelection(game);
                    piece.handleClick({ rightButtonDown: () => false });
                    return game.selectedPiece === piece;
                };
                const cleanup = () => {
                    document.removeEventListener('pointermove', onMove);
                    document.removeEventListener('pointerup', onUp);
                };
                const onMove = (e) => {
                    if (!dragging) {
                        if (Math.hypot(e.clientX - start[0], e.clientY - start[1]) < 5) return;
                        hideStackPicker();
                        if (!selectPiece()) { cleanup(); return; }
                        piece.setVisible(true); piece.setSize(STACK_PR);   // lift it out of the stack
                        dragging = true;
                    }
                    const [x, y] = toGame(e.clientX, e.clientY);
                    piece.setPosition(x, y);
                };
                const onUp = (e) => {
                    cleanup();
                    if (dragging) {
                        const [x, y] = toGame(e.clientX, e.clientY);
                        const target = game.tileAtPoint(x, y);
                        const before = piece.currentTile;
                        if (target) target.onClick();                 // move (or self-tile cancel)
                        if (piece.currentTile === before) _clearSelection(game);  // no move -> deselect
                        if (piece.currentTile) piece.currentTile.updatePositions();
                        else if (piece.rack) piece.rack.shiftPiecesUp();
                    } else {
                        hideStackPicker();
                        selectPiece();   // plain click -> select
                    }
                };
                document.addEventListener('pointermove', onMove);
                document.addEventListener('pointerup', onUp);
            };
        } else {
            // opponent piece: double-click supports the block-save gesture
            chip.ondblclick = (ev) => {
                ev.stopPropagation();
                hideStackPicker();
                if (piece.currentTile) piece.handleDoubleClick();
            };
        }
        row.appendChild(chip);
    });
    pop.appendChild(row);
    pop.style.display = 'block';
    _pickerOpenedAt = performance.now();

    // position centred above the tile
    const midR = (tile.innerRadius + tile.outerRadius) / 2;
    const midA = (tile.startAngle + tile.endAngle) / 2;
    const cxC = CENTER_X + midR * Math.cos(midA), cyC = CENTER_Y + midR * Math.sin(midA);
    const canvas = document.querySelector('canvas'), rect = canvas.getBoundingClientRect();
    const sx = rect.left + cxC * rect.width / canvas.width;
    const sy = rect.top + cyC * rect.height / canvas.height;
    const w = pop.offsetWidth, h = pop.offsetHeight;
    pop.style.left = Math.max(8, Math.min(window.innerWidth - w - 8, sx - w / 2)) + 'px';
    pop.style.top  = Math.max(8, sy - h - 14) + 'px';
}

function updateNoSaveDisplay() {
    const scene = gameInstance && gameInstance.scene && gameInstance.scene.scenes && gameInstance.scene.scenes[0];
    if (!scene || !scene.game || !scene.impasseText) return;
    const game = scene.game;

    // Show the counter once both players are past the opening, OR always when
    // in sandbox (which doesn't track game stages). Never during the tutorial --
    // its steps reach the endgame, where this would otherwise appear.
    const show = !_tut.active && (window.setupMode || game.bothInMidgame());
    if (!show) {
        scene.impasseText.setVisible(false);
        scene.callDrawButton.setHudVisible(false);
        return;
    }

    scene.impasseText
        .setText(`Turns with no save: ${game.noSaveTurns}`)
        .setVisible(true);

    // Offer the button on a human player's turn when callable.
    const humanCanCall = game.drawCallable && game.currentPlayerIsHuman() && !game.gameOver;
    scene.callDrawButton.setHudVisible(!!humanCanCall);
}

class Piece {
    constructor(scene, game, color, number, x, y, rack = null) {
        this.scene = scene;
        this.color = color;
        this.game = game;
        this.number = number;
        this.player = color === 0xffffff ? 'white' : 'black';
        this.originalColor = color;
        this.textColor = color === 0xffffff ? 0x000000 : 0xffffff;
        this.x = x;
        this.y = y;
        this.rack = rack;
        this.radius = PIECE_RADIUS_BASE;
        this.isSelected = false;
        this.isHovered = false;
        this.justMovedHome = false;
        this.reachableTiles = null;
        this.lastClickTime = null;
        this.borderColor = this.color === 0x000000 ? 0xffffff : 0x000000;
        
        this.drawPiece();
    }

    onHover() {
        if (stackPickerOpen()) return;   // suppress board hover while the picker is up
        if (this.game.selectedPiece && this.game.selectedPiece !== this) return;
        if (this.game.dice[0].used && this.game.dice[1].used) return;
        if (this.player !== this.game.turn) return;
        if (this.rack && this.rack.type === 'saved') {
            // A saved piece is part of the rack as far as aiming goes, so with a
            // piece selected this means "save it here" -- otherwise a filling
            // rack would shrink the target the player is told to click.
            if (this.game.selectedPiece && this.rack.onSaveTap) this.rack.onSaveTap();
            return;
        }
        if (this.rack && this.rack.type === 'unentered' && !_isEntrant(this)) return;
        if (!this.game.canSelectForMove(this)) return false;
        // A touch screen has no hover: a finger that leaves often sends no
        // pointerout at all, so setting this would strand the highlight on.
        if (_isPhone()) return;
        this.isHovered = true;
        this.updateColor();
    }

    onOut() {
        // Clearing hover is unconditional. The guards that used to sit here
        // (another piece selected, not your turn any more, ...) meant that a
        // piece could keep `isHovered` forever -- and since the highlight colour
        // is shared with "selected", it looked exactly like a piece that stayed
        // selected after moving. Owner hit this after a capture from the home
        // tile, intermittently, which is the giveaway: it depends on what the
        // state happened to be when the pointer left.
        this.isHovered = false;
        this.updateColor();
    }

    handleClick(pointer) {
        // see handleDoubleClick: a just-saved piece leaves its neighbours
        // shuffling under the finger
        if (this.game._saveGuardUntil && Date.now() < this.game._saveGuardUntil) return;
        hideStackPicker();   // clicking a piece dismisses an open overflow picker

        if (window.setupMode) {
            const now = Date.now();
            if (this._setupLastClick && now - this._setupLastClick < 300) {
                this._setupLastClick = 0;
                setupCyclePieceLocation(this);   // double-click: cycle board -> saved -> unentered -> home
            } else {
                this._setupLastClick = now;
                setupSelectPiece(this);          // single-click: pick up / drop selection
            }
            return;
        }

    // Check for right-click (button 2)
    if (pointer.rightButtonDown()) {
        console.log('🔍 Getting blot info for piece:', this.player, this.number);
        
        // Get current game state
        const gameState = getGameState(this.game);
        
        // Answered on the device (LocalAgent.pieceDebug); this used to need
        // /debug_piece_blots.
        _pieceDebugInfo(gameState, this.player, this.number).then(data => {
            if (!data || data.error) { console.log('   no debug info:', data && data.error); return; }
            console.log(`\u{1F4CA} Piece ${this.player}(${this.number}):`);
            console.log(`   Distance to goal: ${data.distance === null ? 'No path' : data.distance}`);
            console.log(`   Enemy blots on path: ${data.blot_count === null ? 'No path' : data.blot_count}`);
            console.log(`   Can be saved: ${data.can_be_saved}`);
        });
        
        return; // Stop here, don't select the piece
    }

        if (this.game.gameOver) return; 
        if (this.game.dice[0].used && this.game.dice[1].used) return;



        // Saved pieces are out of play: checked before the selection handover
        // below, which would otherwise make one the selected piece -- and with
        // that, draggable back onto the board.
        if (this.rack && this.rack.type === 'saved') {
            // A saved piece is part of the rack as far as aiming goes, so with a
            // piece selected this means "save it here" -- otherwise a filling
            // rack would shrink the target the player is told to click.
            if (this.game.selectedPiece && this.rack.onSaveTap) this.rack.onSaveTap();
            return;
        }

        // DOUBLE-TAP ON A RACK SLOT, for the send-to-goal gesture. The first tap
        // tentatively enters the piece onto the home tile and the rack then closes
        // the gap -- so the second tap of the gesture physically lands on the NEXT
        // piece, never on the one that moved, and per-piece lastClickTime never
        // sees a double-click at all. Owner saw it read as two single taps,
        // oscillating between the first two unentered pieces. Keyed to the rack
        // SLOT instead, which is what the player actually tapped twice.
        //
        // MUST come before the selection handover below: after the first tap the
        // entered piece IS game.selectedPiece, so the handover runs first and its
        // `justMovedHome` branch returns that piece to the rack -- which is the
        // oscillation itself, and it happens before any later check could see it.
        if (this.rack && this.rack.type === 'unentered' && getSumToGoal()) {
            const slot = this.rack.pieces.indexOf(this);
            const mark = this.game._rackSlotTap;
            this.game._rackSlotTap = null;
            const info = mark ? { slot: mark.slot, tappedSlot: slot, markPiece: mark.piece.number,
                                  thisPiece: this.number, age: Date.now() - mark.time,
                                  onHome: !!(mark.piece.currentTile && mark.piece.currentTile.type === 'home'),
                                  justMovedHome: mark.piece.justMovedHome }
                               : { tappedSlot: slot, thisPiece: this.number, markPiece: null };
            console.log('[send-to-goal] rack tap', info);
            if (mark && mark.rack === this.rack && mark.slot === slot &&
                Date.now() - mark.time < RACK_TAP_WINDOW_MS && mark.piece !== this &&
                mark.piece.justMovedHome && mark.piece.currentTile &&
                mark.piece.currentTile.type === 'home') {
                // Consume the gesture either way: this tap belongs to the piece
                // that just left, so it must not enter THIS one as a side effect
                // of the shortcut turning out not to apply.
                this.lastClickTime = null;
                // Kill the pending destination highlight, or it fires after the
                // piece has already gone and re-lights the board.
                clearTimeout(mark.piece._hlTimer);
                if (this.game.sendToGoal(mark.piece)) _clearSelection(this.game);
                else if (this.game.selectedPiece === mark.piece) mark.piece.highlightReachableTiles();
                return;
            }
        }

        if (this.game.selectedPiece && this.game.selectedPiece !== this) {

            // Could this piece take the selection instead? An opponent's never
            // can, and nor can one the obligation rules forbid moving. Checked
            // BEFORE the handover below, which used to run first and could leave
            // an unselectable piece as game.selectedPiece.
            const selectable = this.player === this.game.turn
                && !(this.rack && this.rack.type === 'unentered' && !_isEntrant(this))
                && this.game.canSelectForMove(this);

            // With another piece selected, a tap on a piece means "move onto the
            // tile it stands on" -- otherwise a crowded tile can only be reached
            // by hitting the slivers of empty space between its pieces. Any tile
            // type, since goals get crowded too.
            //
            // BUT only when that tile is actually a destination on offer. It used
            // to forward unconditionally, so tapping one of your own pieces
            // somewhere the selected piece cannot reach refused the move AND left
            // the tapped piece unselected -- the selection never moved, which is
            // what a player means by that tap.
            //
            // Asked of the selected piece's own reachable sets, NOT of the tiles'
            // `reachableColor`: the destination highlight is DEFERRED (see the
            // _hlTimer in onClick, which holds it back so a double-click save
            // shows no flash), so a fast second click lands while every
            // reachableColor is still null and a real destination would read as
            // unreachable. reachableTiles is set at selection time and already
            // accounts for the obligation rules that clear the sum set.
            const _sel = this.game.selectedPiece;
            const _r = _sel.reachableTiles || this.game.getReachableTilesByDice(_sel);
            const isDestination = !!(_r && this.currentTile &&
                [_r.reachableByFirstDie, _r.reachableBySecondDie, _r.reachableBySum]
                    .some(list => list && list.indexOf(this.currentTile) !== -1));

            if (this.currentTile && (!selectable || isDestination)) {
                // Claim the gesture: this tap has now been acted on, and the
                // tile's own pointerup handler must not run onClick a second
                // time (see onTap / _consumeGesture).
                _consumeGesture(pointer);
                this.currentTile.onClick();
                return;
            }
            if (!selectable) {         // nothing to hand the selection over to
                if (this.player === this.game.turn) _flashMustMove(this.game);
                return;
            }

            // Hold the outgoing piece in a local: returnToRack() clears
            // game.selectedPiece, so reading it again below threw (entering a
            // piece and then clicking any other rack piece hit this every time).
            const prev = this.game.selectedPiece;
            prev.isSelected = false;
            if (prev.currentTile && prev.currentTile.type === 'home' && prev.justMovedHome) {
                prev.returnToRack();}
            prev.updateColor();
            this.game.selectedPiece = this;
            this.game.unhighlightAllTiles();
            this.isSelected = false;
        }
        // if (this.player !== this.game.turn) return; 
        if (this.rack && this.rack.type === 'unentered' && !_isEntrant(this)) {
            _flashMustMove(this.game);   // the piece that IS enterable, instead
            return;
        }
        if (this.player === this.game.turn && !this.game.canSelectForMove(this)) {
            console.log("Must keep a die for the obligatory piece(s)");
            _flashMustMove(this.game);
            return false;
        }

        const currentTime = Date.now(); // Use system time
        if (this.lastClickTime === null) {
            this.lastClickTime = currentTime;
            this.onClick();
        } else {
            const timeSinceLastClick = currentTime - this.lastClickTime;
            this.lastClickTime = currentTime;
            if (timeSinceLastClick < 300) {
                this.handleDoubleClick();
                this.lastClickTime = null; // Reset after double click
            } else {
                this.onClick();
            }
        }
    }
    
    onClick() {
        if (this.rack && this.rack.type === 'unentered' && this.game.turn === this.rack.color) {
            // snapshot the pre-entry state (piece still on the rack) so undoing
            // the entry move returns the piece to the top of the rack, not home.
            this.game._pendingPreMove = this.game.captureState();
            // Remember whether this was the FRONT piece BEFORE it leaves the
            // rack. Once it sits on the home tile it is no longer in the rack,
            // so nothing downstream could otherwise tell a reordering entry
            // from an ordinary one -- which let the second piece keep its sum
            // destinations and come out on a dice sum.
            this.game._reorderEntry = (this.rack.pieces[0] !== this) ? this : null;
            // Which slot this came out of, so a second tap in the same place can
            // be recognised as the other half of a double-tap (see handleClick).
            this.game._rackSlotTap = getSumToGoal()
                ? { rack: this.rack, slot: this.rack.pieces.indexOf(this),
                    time: Date.now(), piece: this }
                : null;
            this.moveFromRack();
            this.justMovedHome = true;
            this.game.selectedPiece = this;
            this.reachableTiles = this.game.getReachableTilesByDice(this);
            // With the send-to-goal gesture on, a double-tap on a rack slot is a
            // real gesture, so hold the destinations back between the two taps --
            // exactly as a save double-click already does. Otherwise every goal
            // flashes lit in between: from the home tile all six are exactly 7
            // away, so a roll summing 7 legitimately lights all of them, which
            // reads as "it thinks this is ambiguous" (owner) even though a
            // numbered piece only ever targets its own goal.
            clearTimeout(this._hlTimer);
            if (getSumToGoal()) {
                this._hlTimer = setTimeout(() => {
                    if (this.game.selectedPiece === this) this.highlightReachableTiles();
                }, 270);
            } else {
                this.highlightReachableTiles();
            }
        }
        else if (this.currentTile && this.currentTile.type === 'home' && this.justMovedHome) {
                // Don't send it back yet: this fires on pointer DOWN, so
                // returning here made it impossible to drag a just-entered piece
                // off the home tile -- the press itself put it back on the rack.
                // A press that turns into a drag cancels this (see onDragStart);
                // a press that lifts without moving is a click, and returns it.
                this._pendingReturn = true;
        } else if (this.player === this.game.turn) {
            this.isSelected = !this.isSelected;
            this.updateColor();
            if (this.isSelected) {
                this.game._pendingPreMove = this.game.captureState();   // pre-move snapshot
                this.game.selectedPiece = this;
                this.reachableTiles = this.game.getReachableTilesByDice(this);
                // For a piece on its save tile a double-click means "save it", so
                // defer the destination highlight briefly and cancel it on the
                // second click — otherwise the destinations flash during the save.
                clearTimeout(this._hlTimer);
                if (this.canBeSaved()) {
                    this._hlTimer = setTimeout(() => {
                        if (this.isSelected && this.game.selectedPiece === this) this.highlightReachableTiles();
                    }, 270);
                } else {
                    this.highlightReachableTiles();
                }
            } else {
                clearTimeout(this._hlTimer);
                this.game.unhighlightAllTiles();
                this.game.selectedPiece = null;
                this.reachableTiles = null;
            }
        }
    }

    handleDoubleClick() {
        // cancel any pending (deferred) destination highlight from the first click
        // and clear highlights so a double-click save shows no destination flash.
        clearTimeout(this._hlTimer);
        this.game.unhighlightAllTiles();

        // A piece on a rack has nothing to double-click on. Reachable whenever a
        // tentatively-entered piece goes back to its rack between the two clicks
        // (a refused destination, Esc, undo) -- this used to throw.
        if (!this.currentTile) return;

        if (this.currentTile.type === 'save') {
            const saved = this.save(); // Save the piece if it can be saved
            // Not savable from THIS goal with this roll, but both dice can walk
            // it to another goal and save it there -- do both at once, exactly as
            // from a field tile.
            if (!saved && this.player === this.game.turn && this.game.sumSave(this)) return;
            // Still on a goal it cannot bank from, and cannot reach-and-bank
            // another either -- but the sum may reach a goal it CAN eventually
            // use (a numbered piece parked on the wrong goal, most usefully).
            if (!saved && this.player === this.game.turn && this.game.sendToGoal(this)) {
                _clearSelection(this.game); return;
            }
            // Saving removes this piece, and the tile then re-lays out what is
            // left -- so another piece slides into the spot under the finger and
            // a following pointer event selects it. Owner saw exactly that after
            // double-tapping a numbered piece off its goal. Ignore selections
            // for a moment; 250ms is well under a deliberate re-tap.
            this.game._saveGuardUntil = Date.now() + 250;
            _clearSelection(this.game);
        } else if (this.player === this.game.turn && this.game.sumSave(this)) {
            // not on a goal yet, but one die reaches the goal and the other saves
            // it this turn -> do both at once.
            return;
        } else if (this.player === this.game.turn && this.game.sendToGoal(this)) {
            // Optional gesture, and deliberately AFTER sumSave: both spend the
            // whole roll, and reaching a goal *and* banking beats parking on it,
            // since a banked piece is scored and out of play.
            _clearSelection(this.game);
            return;
        }

        // save a single opponent piece from a block, unless you're in the opening or have a captured piece
        const player = this.color === 0xffffff ? this.game.players[1] : this.game.players[0];
        if (this.player !== this.game.turn && this.currentTile && this.currentTile.type === 'field' && this.currentTile.pieces.length > 1 
            && player.getGamePhase() != 'opening' && this.game.dice.every(die => !die.used) ) {
                const homeTile = this.game.tiles.find(tile => tile.type === 'home');
                if (homeTile?.pieces.length && !(homeTile.pieces.every(piece => piece.player === this.player))) 
                    {
                    // can't save if you have a captured piece
                    homeTile.pieces.forEach(piece => console.log(piece.player));
                    console.log(this.player)
                    return;
                }
            if (_tut.active && !_tutBlockSaveOK(this)) { _tutNudge(); return; }
            console.log('Saving one opponent piece from block')
            const savedRack = this.color === 0xffffff ? this.game.whiteSavedRack : this.game.blackSavedRack;

            this.game.pushUndo();   // snapshot before the block-save

            // Peel ONLY the double-clicked piece into its own saved rack; the
            // rest of the block stays (a 2-stack becomes a blot). The attacker
            // chooses which piece to gift by which one they double-click.
            this.moveToRack(savedRack);
            this.game.registerSave();   // no-save streak resets immediately
            this.game.dice.forEach(die => die.setUsed())

            // Record human single-piece save as (save, pass) -- matches the agent
            // encoding ((piece,0,0), (0,0,0)).
            if (this.game.turn === 'white') {
                const rep = [this.player, this.number];
                _pendingMoves = [[rep, 0, 0], [0, 0, 0]];
            }

            // Check for the endgame condition
            const player = this.color === 0xffffff ? this.game.players[0] : this.game.players[1];
            this.game.checkEndgame(player);

            // Check for the win condition
            this.game.checkWinCondition();
            this.game.maybeAutoEndTurn();
        }
    }

    setPosition(x, y) {
        this.x = x;
        this.y = y;
        this.body.setPosition(x, y);
        this.circle.setPosition(x, y);
        this._layoutSheen();
        if (this.text) {
            this.text.setPosition(x, y);
        }
    }

    // Slide the piece from (ox,oy) to its current position (a quick move tween).
    // On completion it snaps to the tile/rack's exact layout spot so the visual
    // never drifts from where the piece logically belongs.
    animateFrom(ox, oy) {
        if (!getFeedbackEnabled() || !this.scene || !this.scene.tweens) return;
        const nx = this.x, ny = this.y;
        if (Math.hypot(nx - ox, ny - oy) < 2) return;
        if (this._moveTween) { this._moveTween.stop(); this._moveTween = null; }
        const snap = () => {
            this._moveTween = null;
            if (this.currentTile) this.currentTile.updatePositions();
            else if (this.rack) this.rack.shiftPiecesUp();
            else this.setPosition(nx, ny);
        };
        const proxy = { x: ox, y: oy };
        this.setPosition(ox, oy);
        this._moveTween = this.scene.tweens.add({
            targets: proxy, x: nx, y: ny, duration: 160, ease: 'Cubic.easeOut',
            onUpdate: () => this.setPosition(proxy.x, proxy.y),
            onComplete: snap
        });
    }

    setSize(size) {
        this.radius = size;
        this.body.setRadius(size);
        this.circle.setRadius(size);
        this._layoutSheen();
        if (this.text) {
            this.text.setFontSize(`${this._numberFontSize()}px`);
            if (_isPhone()) this.text.setStroke(this.text.style.stroke, Math.max(1, size * 0.1));
        }
        this._applyHitArea();
    }

    // Only the digits 1-6 are ever drawn, so a phone can afford a bigger one:
    // at 2.0r the cap height is about 1.4r inside a 2r circle.
    _numberFontSize() { return this.radius * (_isPhone() ? 2.0 : 1.7); }

    // Phones only. A piece is ~13px across on a landscape phone, well under the
    // 44px a fingertip wants, so grow the touch target into whatever space is
    // actually free around this piece: a piece alone on its tile has room, one
    // in a stack has almost none (slot centres are only 2r+4 apart, and an
    // overlapping hit area would quietly select the neighbour instead).
    // Untouched on desktop, where the default is the 2r bounding box.
    _applyHitArea() {
        const c = this.circle;
        if (!c || !c.input || !_isPhone()) return;      // desktop keeps Phaser's default box
        const r = this.radius;
        // Grow into whatever room the piece actually has: half the distance to
        // the nearest other piece, so two targets can never overlap (an overlap
        // goes to whichever sits higher in the display list, which on a rack is
        // usually not the piece you are allowed to move). An isolated piece ends
        // up with a target well over twice its size, which is what makes it
        // tappable while zoomed out.
        let nearest = Infinity;
        const all = (this.game && this.game.pieces) || [];
        for (const q of all) {
            if (q === this || q.hidden) continue;
            const d = Math.hypot((q.x || 0) - this.x, (q.y || 0) - this.y);
            if (d < nearest) nearest = d;
        }
        const room = Number.isFinite(nearest) ? nearest / 2 : r * 2.4;
        c.input.hitArea = new Phaser.Geom.Circle(r, r, Math.max(r, Math.min(r * 2.4, room)));
        c.input.hitAreaCallback = Phaser.Geom.Circle.Contains;
    }

    // Show/hide the whole piece. Overflow pieces on a stacked tile are hidden
    // (still selectable via the tile's tap-to-pick picker); hiding disables the
    // circle's pointer input so a hidden piece can't be clicked directly.
    setVisible(v) {
        this.hidden = !v;
        this.body.setVisible(v);
        this.sheen.setVisible(v);
        this.circle.setVisible(v);
        if (this.circle.input) this.circle.input.enabled = v;
        if (this.text) this.text.setVisible(v);
    }


    async debugShowPathInfo() {
        if (!this.currentTile) {
            console.log(`Piece ${this.player}(${this.number}) is not on the board (on rack or saved)`);
            return;
        }
        
        console.log(`\n=== DEBUG: Piece ${this.player}(${this.number}) ===`);
        console.log(`Location: Ring ${this.currentTile.ring}, Sector ${this.currentTile.sector} (${this.currentTile.type})`);
        
        // Get the current game state
        const gameState = getGameState(this.game);
        
        // Answered on the device. This block used to POST /debug_piece_info --
        // a route app.py has never defined, so it always 404'd and fell through
        // to localDebugInfo(). It works now.
        try {
            const data = await _pieceDebugInfo(gameState, this.player, this.number);
            if (data && !data.error) {
                console.log(`Distance to goal: ${data.distance === null ? 'No path' : data.distance} steps`);
                console.log(`Enemy blots on shortest path: ${data.blot_count === null ? 'No path' : data.blot_count}`);
                if (data.can_be_saved) console.log(`\u2713 Piece can be saved immediately!`);
            } else {
                this.localDebugInfo();
            }
        } catch (error) {
            console.error(`Error getting debug info: ${error}`);
            this.localDebugInfo();
        }
    }

    localDebugInfo() {
        // Simple local info without server call
        console.log(`Distance: Use shortest_route_to_goal() from backend`);
        console.log(`Can be saved: ${this.canBeSaved()}`);
        
        // Calculate reachable tiles for current dice
        const reachable = this.game.getReachableTilesByDice(this);
        if (reachable) {
            console.log(`Reachable with ${this.game.dice.filter(d => !d.used).map(d => d.value).join(',')}:`);
            if (reachable.reachableByFirstDie?.length) 
                console.log(`  Die 1 (${this.game.dice[0].value}): ${reachable.reachableByFirstDie.map(t => `${t.ring},${t.sector}`).join(', ')}`);
            if (reachable.reachableBySecondDie?.length) 
                console.log(`  Die 2 (${this.game.dice[1].value}): ${reachable.reachableBySecondDie.map(t => `${t.ring},${t.sector}`).join(', ')}`);
            if (reachable.reachableBySum?.length) 
                console.log(`  Sum (${this.game.dice[0].value + this.game.dice[1].value}): ${reachable.reachableBySum.map(t => `${t.ring},${t.sector}`).join(', ')}`);
        }
    }


    move(tile, checkMidgame = true) {
        if (this.rack) {
            this.moveFromRack();
        } else if (this.currentTile) {
            this.currentTile.removePiece(this);
        }
        this.currentTile = tile;
        tile.addPiece(this);
        this.rack = null;
        this.isSelected = false;
        this.justMovedHome = false;
        this.game.unhighlightAllTiles();
        if (checkMidgame) this.game.checkMidgame()

        const player = this.color === 0xffffff ? this.game.players[0] : this.game.players[1];
        this.game.checkEndgame(player);
        }
        

    

    _afterRackChange() {
        if (typeof _updateViewportHud === 'function') setTimeout(_updateViewportHud, 0);
    }

    moveFromRack() {
        const homeTile = this.game.tiles.find(tile => tile.type === 'home');
        this._rackIndexOnLeave = this.rack.pieces.indexOf(this);
        this.rack.removePiece(this);
        this.rack.shiftPiecesUp();
        this.rack = null;
        this.move(homeTile, false);
        this._turnStartTile = homeTile;   // an entering piece measures progress from home
        this.game.selectedPiece = this;
        this.isSelected = true;
        this._afterRackChange();          // one fewer piece left to bring out
    }

    moveToRack(rack, addToFront = false, index = null) {
        this.rack = rack;
        this.x = rack.nextX();
        this.y = rack.nextY();
        this.setSize(_rackPR());
        this.body.setPosition(this.x, this.y);
        this.circle.setPosition(this.x, this.y);
        this._layoutSheen();
        if (this.text) {
            this.text.setPosition(this.x, this.y);
        }
        this.setVisible(true);   // a piece hidden as tile overflow reappears in the rack
        if (index != null && rack.addPieceAt) {
            rack.addPieceAt(this, index);      // back to its own slot
        } else if (addToFront) {
        rack.addPieceToFirstPosition(this);
        } else {
        rack.addPiece(this);
         }
        if (this.currentTile) {
            this.currentTile.removePiece(this);
        }
        this.currentTile = null;
        this.isSelected = false;
        this.isHovered = false;
        this.game.selectedPiece = null;
        this.game.unhighlightAllTiles();
        this.updateColor();
    }

    returnToRack() {
        const unenteredRack = this.color === 0xffffff ? this.game.whiteUnenteredRack : this.game.blackUnenteredRack;
        // Back to the slot it left, not the front: taking the second piece out
        // and putting it back used to promote it to first place.
        const slot = (this._rackIndexOnLeave != null) ? this._rackIndexOnLeave : 0;
        this._rackIndexOnLeave = null;
        this.moveToRack(unenteredRack, true, slot);
        this.justMovedHome = false;
        this.reachableTiles = null;
        this.game.selectedPiece = null;
        // back on the rack means it is enterable again -- and if the rack is off
        // screen its ghost has to come back with it
        if (typeof _updateViewportHud === 'function') setTimeout(_updateViewportHud, 0);
        this.game.tiles.forEach(tile => {
            tile.unhighlight();
        })
    }

    updateColor() {
        if (!this.body) return;
        // Highlight recolors the body; the rim + sheen stay.
        //
        // SELECTED and HOVERED are deliberately DIFFERENT strengths. They used
        // to be the same colour, which made a piece under the cursor pixel-
        // identical to a selected one -- and clicking a destination tile leaves
        // the cursor exactly where the piece lands, so a completed move looked
        // like it had "stayed selected" or "got reselected". The move path is
        // clean (measured: isSelected, isHovered and selectedPiece all false
        // afterwards); it was only ever the shared colour. Hover is now a weak
        // tint toward the highlight, selection the full colour.
        const hi = this.color === 0xffffff ? 0x90ee90 : 0xee82ee;
        if (this.isSelected) {
            this.body.setFillStyle(hi);
        } else if (this.isHovered) {
            this.body.setFillStyle(_mixColor(this.bodyColor, hi, 0.4));
        } else {
            this.body.setFillStyle(this.bodyColor);
        }
    }

    // Mark this piece as obligatory-to-move with a bright ring (or clear it).
    setObligatory(v) {
        if (!this.circle) return;
        this.obligatory = v;
        const rim = this.color === 0xffffff ? PIECE_WHITE_RIM : PIECE_BLACK_RIM;
        if (v) this.circle.setStrokeStyle(4.5, 0xff8c1a, 1);   // bright orange ring
        else this.circle.setStrokeStyle(2.5, rim, 1);
    }


    highlightReachableTiles() {

        const reachableTiles = this.reachableTiles;
        if (!reachableTiles) return;

        const { reachableByFirstDie, reachableBySecondDie, reachableBySum } = reachableTiles;

        // Each die keeps its own colour (sets are empty for a used die); yellow
        // (sum) is non-empty only while both dice are unused.
        reachableByFirstDie.forEach(tile => { tile.reachableColor = colorFirstDie; tile.highlight(); });
        reachableBySecondDie.forEach(tile => { tile.reachableColor = colorSecondDie; tile.highlight(); });
        reachableBySum.forEach(tile => { tile.reachableColor = colorSum; tile.highlight(); });
    }

    canBeSaved() {
        if (this.rack === this.game.whiteSavedRack || this.rack === this.game.blackSavedRack) {
            return true;
        }

        const player = this.color === 0xffffff ? this.game.players[0] : this.game.players[1];
        if (player.getGamePhase() === 'opening') {
            return false;
        }
        if (!this.currentTile || this.currentTile.type !== 'save') {
            return false;
        }
        if (this.number > 6) {
            return true;
        } else {
            return this.currentTile.number === this.number;
        }
    }

    save() {
        if (_tut.active && !_tutSaveOK(this)) { _tutNudge(); _clearSelection(this.game); return false; }
        const player = this.color === 0xffffff ? this.game.players[0] : this.game.players[1];
        console.log(`Attempting to save piece ${this.number} for player ${player.name} in phase ${player.getGamePhase()}`);
        
        console.log(this.player, this.game.turn)

        if (player.getGamePhase() === 'opening') {
            console.log(`${player.name} is in the opening phase and cannot save pieces.`);
            return false;
        }



        if (this.player === this.game.turn && this.canBeSaved()) {
            const saveTileNumber = this.currentTile.number;
            const dice = this.game.dice.filter(die => !die.used);
            const isEndgame = player.getGamePhase() === 'endgame';
            const endgameHigherOK = isEndgame && !this.game.isHigherNumberedGoalOccupied(player, saveTileNumber);

            let dieToUse;
            if (this.number > 6) {
                // Unnumbered piece: savable by the matching die (== goal number) or,
                // in the endgame with no higher goal occupied, any higher die.
                // Smart saving: prefer a die NOT uniquely needed by a numbered piece
                // (a #N on goal N can ONLY be saved with an N), so we don't strand it.
                const candidates = dice.filter(d =>
                    d.value === saveTileNumber || (endgameHigherOK && d.value > saveTileNumber));
                const notReserved = candidates.filter(d =>
                    !this.game.numberedPieceNeedsDie(d.value, this.color));
                const pool = notReserved.length ? notReserved : candidates;
                // within the pool prefer the exact goal-number die, else the smallest
                dieToUse = pool.find(d => d.value === saveTileNumber)
                        || pool.sort((a, b) => a.value - b.value)[0];
            } else {
                // Numbered piece: only ever its own value. The endgame higher-die
                // rule is for blank pieces alone -- game.py's get_saving_die gates
                // it on number > 6, and sumSave below does too; this branch used to
                // allow it and let a numbered piece go out on any higher die.
                dieToUse = dice.find(d => d.value === saveTileNumber);
            }

            if (dieToUse) {
                console.log(`Using die ${dieToUse.value} to save piece ${this.number}`);
                const dieValue = dieToUse.value;
                this.game.pushUndo();   // snapshot before the save so undo reverts just it
                // Use the corresponding die
                dieToUse.setUsed();

                // Move the piece to the saved rack
                fxBurst(this.scene, this.x, this.y, THEME.accent);   // save flash on the goal
                const savedRack = this.color === 0xffffff ? this.game.whiteSavedRack : this.game.blackSavedRack;
                // The twelfth save ends the game, and the win/lose chime says so
                // better than a save chime landing on top of it.
                if (savedRack.pieces.length + 1 < TOTAL_PIECES) SFX.save();
                this.moveToRack(savedRack); // Move the piece to the saved rack
                this.game.registerSave();   // no-save streak resets immediately

                // Record human save move
                if (this.player === 'white') {
                    pushHumanMove([this.player, this.number], 'save', dieValue);
                }

                // Check for the endgame condition
                this.game.checkEndgame(player);

                // Check for the win condition
                this.game.checkWinCondition();

                refreshEvalReadout();  // update on-board eval after the save settles
                this.game.maybeAutoEndTurn();
                return true;
            } else {
                console.log(`No available die roll corresponds to the save tile's number ${saveTileNumber}, piece ${this.number} cannot be saved`);
                return false;
            }
        } else {
            return false;
        }
    }

    drawPiece() {
        const isWhite = this.color === 0xffffff;
        this.bodyColor = isWhite ? PIECE_WHITE_BODY : PIECE_BLACK_BODY;
        const rimColor = isWhite ? PIECE_WHITE_RIM : PIECE_BLACK_RIM;

        // flat sphere body + soft top-left sheen highlight for a glossy read
        // (native shapes render reliably on any background, unlike a baked
        // white-on-white gradient texture).
        this.body = this.scene.add.circle(this.x, this.y, this.radius, this.bodyColor);
        // white pieces get a soft sheen; black pieces are solid (no highlight).
        this.sheen = this.scene.add.circle(0, 0, this.radius * 0.42, 0xffffff, isWhite ? 0.55 : 0);
        this._layoutSheen();

        // transparent interactive circle carries the border + pointer events.
        this.circle = this.scene.add.circle(this.x, this.y, this.radius, 0xffffff, 0)
            .setStrokeStyle(2.5, rimColor, 1)
            .setInteractive()
            .on('pointerover', () => this.onHover())
            .on('pointerout', () => this.onOut())
            .on('pointerdown', (pointer) => { this._draggedSincePress = false; this.handleClick(pointer); });
        // drag-to-move (additive; click still works). The scene-level drag
        // handlers (Game.setupDragging) reach the piece via __piece.
        this.circle.__piece = this;
        this.scene.input.setDraggable(this.circle);
        const finishPress = () => {
            if (!this._pendingReturn) return;
            this._pendingReturn = false;
            if (!this._draggedSincePress) this.returnToRack();
        };
        this.circle.on('pointerup', finishPress);
        this.circle.on('pointerupoutside', finishPress);

        // Debug-mode tooltip: show the number of unnumbered pieces (numbered
        // pieces already display their number on the board).
        this.circle
            .on('pointerover', () => { if (window.debugMode && this.number > 6) showDebugTip(`${this.player} #${this.number}`); })
            .on('pointermove', () => { if (window.debugMode && this.number > 6) showDebugTip(`${this.player} #${this.number}`); })
            .on('pointerout',  () => { hideDebugTip(); });

        if (this.number <= 6 || DEBUG_MODE) {
            // Phaser's default font family is Courier -- a thin monospace with a
            // small x-height, which at 12px on a phone is the worst possible
            // choice. Phones get the bold UI sans instead, a bigger digit (only
            // 1-6 are ever drawn, so there is room inside the circle), and a
            // halo in the piece's own colour to lift it off the sheen.
            const st = {
                fontSize: `${this._numberFontSize()}px`,
                color: `#${this.textColor.toString(16).padStart(6, '0')}`,
                fontStyle: 'bold'
            };
            if (_isPhone()) st.fontFamily = HUD_FONT;
            this.text = this.scene.add.text(this.x, this.y, this.number, st).setOrigin(0.5, 0.5);
            if (_isPhone()) {
                this.text.setStroke(`#${this.bodyColor.toString(16).padStart(6, '0')}`,
                                    Math.max(1, this.radius * 0.1));
            }
        } else {
            this.text = null;
        }
    }

    _layoutSheen() {
        this.sheen.setPosition(this.x - this.radius * 0.32, this.y - this.radius * 0.32);
        this.sheen.setRadius(this.radius * 0.42);
    }
}

class Tile {

        constructor(scene, game, type, ring, sector, startAngle, endAngle, innerRadius, outerRadius, number) {
            this.scene = scene;
            this.game = game;
            this.type = type;
            this.ring = ring;
            this.sector = sector;
            this.number = number;
            this.pieces = [];
            this.startAngle = startAngle;
            this.endAngle = endAngle;
            this.innerRadius = innerRadius;
            this.outerRadius = outerRadius;
            this.neighbors = [];
            this.highlightColor = THEME.highlight;
            this.reachableColor = null;
            this.lastClickTime = null;

            this.lineColor = TILE_BORDER;
            this.graphics = scene.add.graphics();

            switch (type) {
                case "home":
                    this.fillColor = THEME.hub; this.lineColor = THEME.hubRing;
                    break;
                case "save":
                    this.fillColor = GOAL_COLOR; this.lineColor = TILE_BORDER;
                    break;
                case "nogo":
                    // nogo = "no board space" -> blend into the background,
                    // no visible border.
                    this.fillColor = BACKGROUND_COLOR;
                    this.lineColor = BACKGROUND_COLOR;
                    break;
                case "field":
                    this.fillColor = THEME.field; this.lineColor = TILE_BORDER;
                    break;
            }

            this.drawTile();

        }

    calculateAnnularSegmentPoints(cx, cy, innerRadius, outerRadius, startAngle, endAngle) {
        const points = [];
        const step = Math.min((endAngle - startAngle) / 20, Math.PI / 180); // Dynamic step for smoothness



        // Outer arc
        for (let angle = startAngle; angle <= endAngle; angle += step) {
            points.push({
                x: cx + outerRadius * Math.cos(angle),
                y: cy + outerRadius * Math.sin(angle)
            });
        }
        // Ensure last point of outer arc is exact
        points.push({
            x: cx + outerRadius * Math.cos(endAngle),
            y: cy + outerRadius * Math.sin(endAngle)
        });

        // Inner arc (reverse)
        for (let angle = endAngle; angle >= startAngle; angle -= step) {
            points.push({
                x: cx + innerRadius * Math.cos(angle),
                y: cy + innerRadius * Math.sin(angle)
            });
        }
        // Ensure first point of inner arc is exact
        points.push({
            x: cx + innerRadius * Math.cos(startAngle),
            y: cy + innerRadius * Math.sin(startAngle)
        });

  

        return points;
    }

    addNumberText(number, angle, radius) {
        if (this._numberText) { this._numberText.destroy(); this._numberText = null; }
        // radius is passed just OUTSIDE the goal tile's outer edge so the number
        // never gets obscured by pieces sitting on the goal. Black, large, bold.
        const x = CENTER_X + radius * Math.cos(angle);
        const y = CENTER_Y + radius * Math.sin(angle);
        const text = this.scene.add.text(x, y, number.toString(), {
            fontSize: '46px',
            color: THEME.goalNum,
            fontStyle: 'bold'
        }).setOrigin(0.5);
        // Halo in the board colour: the number sits just outside its goal wedge,
        // so on the darker themes it was reading as washed out against the
        // bright wedge beside it.
        text.setStroke(_cssHex(THEME.bg), 6);
        text.setDepth(50);
        text.setAngle(0);
        this._numberText = text;   // kept so a live theme switch can recolour it
    }

    // Re-derive this tile's colours from the current THEME and redraw (live theme
    // switch). Mirrors the type→colour switch in the constructor.
    applyThemeColors() {
        this.highlightColor = THEME.highlight;
        switch (this.type) {
            case 'home': this.fillColor = THEME.hub; this.lineColor = THEME.hubRing; break;
            case 'save': this.fillColor = GOAL_COLOR; this.lineColor = TILE_BORDER; break;
            case 'nogo': this.fillColor = BACKGROUND_COLOR; this.lineColor = BACKGROUND_COLOR; break;
            case 'field': this.fillColor = THEME.field; this.lineColor = TILE_BORDER; break;
        }
        if (this._numberText) {
            this._numberText.setColor(THEME.goalNum);
            this._numberText.setStroke(_cssHex(THEME.bg), 6);
        }
        this.drawTile();
    }

/*     onClick() {
        if (selectedPiece && this.type !== "nogo") {
            selectedPiece.move(this);
            selectedPiece.isSelected = false;
            selectedPiece.updateColor();
            selectedPiece = null;
        }
    } */

        onClick() {
            hideStackPicker();   // any tile click dismisses an open picker first
            if (window.setupMode) {
                if (_setupSelected && this.type !== 'nogo') {
                    _setupPlaceOnTile(_setupSelected, this);
                    _setupClearSelection();
                }
                return;
            }
            if (this.game.gameOver) return;
            // Phones: with nothing selected yet, tapping the tile selects the
            // piece on it, as long as there is no doubt which one is meant. A
            // tile is far easier to hit than a 13px piece. Delegating to the
            // piece's own handler keeps every rule and the double-tap-to-save
            // timing exactly as they are when you tap the piece itself.
            if (!this.game.selectedPiece && _isPhone() && _tileTapEnabled()) {
                const target = this._unambiguousPiece();
                if (target) { target.handleClick({ rightButtonDown: () => false }); return; }
            }
            // The overflow picker is opened only from the "+K" badge, never as a
            // side effect of a tile click (so moving a piece onto a tile that
            // tips into overflow doesn't pop the picker).
            if (this.game.selectedPiece && this.type !== "nogo") {
                const piece = this.game.selectedPiece;
                // clicking/dropping onto the piece's own tile cancels the
                // selection without consuming a die.
                if (piece.currentTile === this) {
                    piece.isSelected = false;
                    piece.updateColor();
                    this.game.unhighlightAllTiles();
                    this.game.selectedPiece = null;
                    return;
                }
                const diceBefore = this.game.dice.map(d => ({ value: d.value, used: d.used }));
                // Traced alongside _noticeIfRouteWithheld: if the notice never
                // appears AND this line never logs, the tap is not reaching
                // movePiece at all and the fault is upstream of the move logic.
                console.log('[route-notice] tile tapped with a piece selected ->',
                            this.type, this.ring + ',' + this.sector);
                if (this.game.movePiece(piece, this)) {
                    // Determine which die(s) were consumed
                    let dieUsed = 0;
                    for (let i = 0; i < this.game.dice.length; i++) {
                        if (!diceBefore[i].used && this.game.dice[i].used) {
                            dieUsed = dieUsed === 0 ? diceBefore[i].value : dieUsed + diceBefore[i].value;
                        }
                    }
                    pushHumanMove([piece.player, piece.number], [this.ring, this.sector], dieUsed);
                    piece.isSelected = false;
                    // Clear the piece's OWN selection flag and the board's
                    // highlights too, not just the game's selectedPiece: a
                    // capturing entry left isSelected true and the destinations
                    // lit, so the piece looked selected and swallowed the second
                    // die until it was manually deselected.
                    piece.isSelected = false;
                    piece.reachableTiles = null;
                    piece.updateColor();
                    this.game.unhighlightAllTiles();
                    this.game.selectedPiece = null;
                } else {
                    console.log('Move not possible');
                    // A refused move must leave the selection exactly as it was.
                    // Something downstream clears the board's highlights, so
                    // re-assert them rather than leave the piece selected with
                    // nothing lit -- which looks like the selection was lost.
                    if (this.game.selectedPiece === piece) {
                        piece.reachableTiles = this.game.getReachableTilesByDice(piece);
                        if (piece.highlightReachableTiles) piece.highlightReachableTiles();
                    }
                }
            }
        }
    

    // The piece a tap on this tile can only have meant: the current player's
    // single piece here, or -- when they are all unnumbered -- any of them,
    // since those are interchangeable. Numbered pieces each have their own goal,
    // so two of them on one tile stays ambiguous and the tap is ignored (tap the
    // piece itself, or use the stack picker).
    _unambiguousPiece() {
        const mine = (this.pieces || []).filter(p => p.player === this.game.turn && !p.hidden);
        if (mine.length === 1) return mine[0];
        if (mine.length > 1 && mine.every(p => p.number > 6)) return mine[0];
        return null;
    }

    onHover() {
        if (this.game.gameOver) return;
        if (this.type === "nogo") return;
        if (stackPickerOpen()) return;   // don't highlight tiles while the picker is up
        // Not the human's turn (computer thinking): no hover highlight.
        if (this.game.currentPlayerIsHuman && !this.game.currentPlayerIsHuman()) return;
        this.highlight();
        if (DEBUG_MODE) console.log(this.ring, this.sector)
    }

    // Recolour by REDRAWING the tile, never by appending another fill to the
    // same Graphics. These three run constantly -- unhighlightAllTiles touches
    // every tile on every selection -- and a Graphics object replays its whole
    // command list each frame, so appending here made the renderer do more work
    // every turn: measured 55 fps at the start of a game down to 4 fps by turn
    // 60, with object count and heap flat. A reload "fixed" it because it reset
    // the command lists.
    highlight() {
        this._fillOverride = this.reachableColor !== null ? this.reachableColor : this.highlightColor;
        this.drawTile();
    }

    unhighlight() {
        this._fillOverride = null;
        this.drawTile();
    }

    onOut() {
        this._fillOverride = this.reachableColor !== null ? this.reachableColor : null;
        this.drawTile();
    }


    addPiece(piece) {
        this.pieces.push(piece);
        this.updatePositions(); // Update positions whenever a piece is added
    }

    removePiece(piece) {
        this.pieces = this.pieces.filter(p => p !== piece);
        this.updatePositions(); // Update positions whenever a piece is removed
    }
    
 
 
 
    updatePositions() {
        if (this.type === "home") {
            const homeTileRadius = HOME_TILE_RADIUS - 30; // Adjust radius to fit pieces comfortably within the home tile
            const angularStep = Phaser.Math.DegToRad(360 / this.pieces.length); // Angular step between pieces
    
            this.pieces.forEach((piece, index) => {
                const angle = angularStep * index; // Calculate angle for each piece
                const x = CENTER_X + homeTileRadius * Math.cos(angle); // Calculate x position
                const y = CENTER_Y + homeTileRadius * Math.sin(angle); // Calculate y position

                piece.setSize(PIECE_RADIUS_BASE); // Set piece size
                piece.setPosition(x, y); // Set piece position
                piece.setVisible(true);
            });
        } else {
            // Capacity-based stacking, centred in the tile (both radially and
            // angularly). Pieces use this tile's adaptive radius (shrunk on small
            // tiles so >=2 fit). Numbered pieces (1-6) take visibility precedence;
            // overflow folds into a "+K" badge that the tap-to-pick picker expands.
            const n = this.pieces.length;
            const pr = this.tilePieceRadius(n);   // size just enough to fit n (down to the min)
            const cap = this._capacityAtSlot(pr * 2 + 4);
            const ord = [...this.pieces].sort((a, b) => (a.number > 6 ? 1 : 0) - (b.number > 6 ? 1 : 0));
            const over = ord.length > cap;
            const show = over ? cap - 1 : ord.length;
            const pos = this.stackPositions(over ? cap : ord.length, pr);
            ord.forEach((piece, i) => {
                if (i < show) {
                    const sl = pos[i] || pos[pos.length - 1];
                    piece.setSize(pr);
                    piece.setPosition(CENTER_X + sl.r * Math.cos(sl.a), CENTER_Y + sl.r * Math.sin(sl.a));
                    piece.setVisible(true);
                } else {
                    piece.setVisible(false);
                }
            });
            if (over) {
                const sl = pos[pos.length - 1];
                this.showBadge(CENTER_X + sl.r * Math.cos(sl.a), CENTER_Y + sl.r * Math.sin(sl.a),
                               ord.length - (cap - 1), pr);
            } else {
                this.hideBadge();
            }
        }
    }

    // Board-piece radius for this tile given `n` pieces to show: the default
    // STACK_PR, shrunk only as far as needed for all n to fit un-stacked (never
    // below STACK_MIN_R). So a lone piece keeps full size, a ring-3 tile holds 3
    // at a slightly smaller size, etc. — resize only when needed, prefer resizing
    // to stacking. Beyond the min radius the extra pieces fold into the badge.
    tilePieceRadius(n) {
        // On a phone, start bigger and shrink to fit, so a piece with a roomy
        // tile to itself is drawn larger rather than leaving the space empty.
        // The ceiling keeps it inside the tile's own radial band, since the
        // capacity test below only counts slots and would happily overflow it.
        const ext = this.outerRadius - this.innerRadius;
        let r = _isPhone() ? Math.max(STACK_PR, Math.floor(Math.min(STACK_PR * 1.5, (ext - 10) / 2)))
                           : STACK_PR;
        while (r > STACK_MIN_R && this._capacityAtSlot(r * 2 + 4) < n) r -= 1;
        return r;
    }

    _capacityAtSlot(slot) {
        const dth = this.endAngle - this.startAngle;
        const ext = this.outerRadius - this.innerRadius;
        const midR = (this.innerRadius + this.outerRadius) / 2;
        const maxRows = Math.max(1, Math.floor(ext / slot));
        let cap = 0;
        for (let k = 0; k < maxRows; k++) {
            const r = midR + (k - (maxRows - 1) / 2) * slot;
            cap += Math.max(1, Math.floor((r * dth) / slot));
        }
        return cap;
    }

    // Max pieces that fit for a given count n (at the radius chosen for n).
    stackCapacity(n = this.pieces.length) { return this._capacityAtSlot(this.tilePieceRadius(n) * 2 + 4); }

    // `count` piece positions, centred in the tile: rows straddle the mid radius
    // and each row is centred on the mid angle. One piece -> exact tile centre.
    stackPositions(count, pr = this.tilePieceRadius(count)) {
        const slot = pr * 2 + 4;
        const dth = this.endAngle - this.startAngle;
        const ext = this.outerRadius - this.innerRadius;
        const midR = (this.innerRadius + this.outerRadius) / 2;
        const midA = (this.startAngle + this.endAngle) / 2;
        const maxRows = Math.max(1, Math.floor(ext / slot));
        const rowInfo = (rows, k) => {
            const r = midR + (k - (rows - 1) / 2) * slot;
            return { r, cap: Math.max(1, Math.floor((r * dth) / slot)) };
        };
        // fewest rows that hold `count`
        let rows = 1;
        while (rows < maxRows) {
            let c = 0; for (let k = 0; k < rows; k++) c += rowInfo(rows, k).cap;
            if (c >= count) break;
            rows++;
        }
        const info = []; for (let k = 0; k < rows; k++) info.push(rowInfo(rows, k));
        // round-robin fill respecting per-row caps (keeps rows balanced)
        const sizes = new Array(rows).fill(0);
        let rem = count, guard = 0;
        while (rem > 0 && guard++ < 2000) {
            let placed = 0;
            for (let k = 0; k < rows && rem > 0; k++) {
                if (sizes[k] < info[k].cap) { sizes[k]++; rem--; placed++; }
            }
            if (!placed) break;
        }
        const positions = [];
        for (let k = 0; k < rows; k++) {
            const m = sizes[k], r = info[k].r, pitch = slot / r;
            for (let i = 0; i < m; i++) positions.push({ r, a: midA + (i - (m - 1) / 2) * pitch });
        }
        return positions;
    }

    showBadge(x, y, k, pr = STACK_PR) {
        if (!this.badgeCircle) {
            // interactive so the badge itself is a reliable tap target for the picker
            this.badgeCircle = this.scene.add.circle(x, y, pr, THEME.accent).setDepth(60)
                .setInteractive({ useHandCursor: true })
                // with a piece selected, the badge acts as the tile (drop/move here);
                // otherwise it toggles the overflow picker open/closed.
                ;
            onTap(this.badgeCircle, () => {
                    if (this.game.selectedPiece) { this.onClick(); return; }
                    if (stackPickerOpen()) hideStackPicker(); else openStackPicker(this);
                });
            this.badgeText = this.scene.add.text(x, y, '', {
                fontFamily: HUD_FONT, fontStyle: 'bold', color: '#ffffff'
            }).setOrigin(0.5).setDepth(61);
        }
        this.badgeCircle.setPosition(x, y).setRadius(pr).setVisible(true);
        if (this.badgeCircle.input) this.badgeCircle.input.enabled = true;
        this.badgeText.setPosition(x, y).setFontSize(`${pr}px`).setText('+' + k).setVisible(true);
    }

    hideBadge() {
        if (this.badgeCircle) {
            this.badgeCircle.setVisible(false); this.badgeText.setVisible(false);
            if (this.badgeCircle.input) this.badgeCircle.input.enabled = false;
        }
    }
    
    
    // mode === 'bake' draws the tile at its resting colours regardless of any
    // highlight, for _bakeBoard to capture into the board texture.
    drawTile(mode) {
        const baking = mode === 'bake';

        this.graphics.clear();
        // nogo = "no board space": draw nothing so the background shows through
        // and no nogo fill covers an adjacent field tile's border.
        if (this.type === 'nogo') return;
        // Once the board is baked, a tile at its resting colour is ALREADY in
        // the texture underneath, so it contributes nothing to the per-frame
        // command list. Only a highlighted tile draws over its baked copy.
        // (The Graphics object itself stays -- it owns the hit area, and an
        // empty one still hit-tests, which setVisible(false) would not.)
        if (!baking && this.game && this.game._boardBaked && this._fillOverride == null) return;
        this.graphics.lineStyle(1.7, this.lineColor, 1);
        this.graphics.fillStyle(!baking && this._fillOverride != null ? this._fillOverride : this.fillColor, 1);

        if (this.type === "home") {
            this.x = CENTER_X;
            this.y = CENTER_Y;
            this.graphics.fillCircle(CENTER_X, CENTER_Y, HOME_TILE_RADIUS);
            this.graphics.strokeCircle(CENTER_X, CENTER_Y, HOME_TILE_RADIUS);
        } else {
    
            const points = this._points || (this._points = this.calculateAnnularSegmentPoints(
                CENTER_X, CENTER_Y, this.innerRadius, this.outerRadius, this.startAngle, this.endAngle));



            this.graphics.beginPath();
            points.forEach((point, index) => {
                if (index === 0) {
                    this.graphics.moveTo(point.x, point.y);
                } else {
                    this.graphics.lineTo(point.x, point.y);
                }
            });
            this.graphics.closePath();
            this.graphics.fillPath();
            this.graphics.strokePath();

            // A ring-7 tile whose ring-6 neighbour is hidden nogo needs its own
            // inner edge closed off (see hideOuterNogoTiles). Part of the normal
            // draw so it survives every redraw, and so the bake captures it.
            if (this._innerArc) {
                this.graphics.lineStyle(1, 0x000000, 1);
                this.graphics.beginPath();
                const step = Math.PI / 180;
                for (let angle = this.startAngle; angle <= this.endAngle; angle += step) {
                    const x = CENTER_X + this.innerRadius * Math.cos(angle);
                    const y = CENTER_Y + this.innerRadius * Math.sin(angle);
                    if (angle === this.startAngle) this.graphics.moveTo(x, y);
                    else this.graphics.lineTo(x, y);
                }
                this.graphics.strokePath();
            }

            // Hit area, handlers and the goal number are built ONCE. drawTile is
            // now called every time a tile changes colour, and re-running this
            // block each time re-registered the pointer handlers and created a
            // fresh Text object per goal, every highlight.
            if (!this._built) this.buildTileChrome(points);
        }
    }

    buildTileChrome(points) {
        this._built = true;
        this.graphics.setInteractive(new Phaser.Geom.Polygon(points), Phaser.Geom.Polygon.Contains);
        onTap(this.graphics, (pointer) => {
            const t = pointer ? _resolveDestination(this.game, this, pointer.worldX, pointer.worldY) : this;
            t.onClick();
        });
        this.graphics
            .on('pointerover', () => this.onHover())
            .on('pointerout', () => this.onOut());

        // Debug-mode tooltip: show this tile's ring and sector.
        this.graphics
            .on('pointerover', () => { if (window.debugMode) showDebugTip(`ring ${this.ring}, sector ${this.sector}`); })
            .on('pointermove', () => { if (window.debugMode) showDebugTip(`ring ${this.ring}, sector ${this.sector}`); })
            .on('pointerout',  () => { hideDebugTip(); });

        // Add number to "save" tiles
        if (this.type === 'save' && this.number !== undefined) {
            this.addNumberText(this.number, (this.startAngle + this.endAngle) / 2, this.outerRadius + 26);
        }
    }
    
    
    
    
}

class Rack {
    constructor(scene, x, y, color, type, rows = 4, cols = 3) {
        this.scene = scene;
        this.x = x;
        this.y = y;
        this.color = color;
        this.type = type;
        this.pieces = [];
        this.rows = rows;
        this.cols = cols;
        this.pr = _rackPR();
        this.spacing = this.pr * 2 + 12;
        this.verticalPadding = 22;
        this.horizontalPadding = 18;
        this.background = scene.add.graphics();
        this.drawBackground();
    }

    addPiece(piece) {
        this.pieces.push(piece);
        piece.rack = this;
        // Sizing happens before this at game setup, and the touch target depends
        // on which rack the piece is in, so re-apply now that it knows.
        if (piece._applyHitArea) piece._applyHitArea();
    }

    removePiece(piece) {
        this.pieces = this.pieces.filter(p => p !== piece);
        // Close the gap HERE rather than trusting each caller to. While only
        // the front piece could leave, a caller that forgot left the hole at
        // the end where nothing showed; now a piece can leave from the middle,
        // and a forgotten relayout is a visible empty slot. Cheap and
        // idempotent, so the callers that already do it lose nothing.
        this.shiftPiecesUp();
    }

    shiftPiecesUp() {
        for (let i = 0; i < this.pieces.length; i++) {
            const piece = this.pieces[i];
            const newX = this.x + this.horizontalPadding + (i % this.cols) * this.spacing;
            const newY = this.y + this.verticalPadding + Math.floor(i / this.cols) * this.spacing;
            piece.setPosition(newX, newY);
            piece.setVisible(true);   // ensure a formerly-hidden overflow piece shows in the rack
                    // Force size reset when on rack
            if (this.type === 'unentered' || this.type === 'saved') {
                piece.setSize(this.pr);
            }
        }
    }

    // Put a piece back at a specific slot. returnToRack used to unshift it to
    // the front, which silently reordered the rack once the SECOND piece could
    // be taken out and put back.
    addPieceAt(piece, index) {
        const i = Math.max(0, Math.min(index | 0, this.pieces.length));
        this.pieces.splice(i, 0, piece);
        piece.rack = this;
        this.shiftPiecesUp();          // canonical re-layout of the whole rack
        if (piece._applyHitArea) piece._applyHitArea();
    }

    addPieceToFirstPosition(piece) {
        // Shift existing pieces down
        for (let i = this.pieces.length; i > 0; i--) {
            const currentPiece = this.pieces[i - 1];
            const newX = this.x + this.horizontalPadding + (i % this.cols) * this.spacing;
            const newY = this.y + this.verticalPadding + Math.floor(i / this.cols) * this.spacing;
            currentPiece.setPosition(newX, newY);
        }
    
        // Add the new piece to the first position of the array
        this.pieces.unshift(piece);
        piece.rack = this;
    
        // Set the position of the new piece
        const firstX = this.x + this.horizontalPadding;
        const firstY = this.y + this.verticalPadding;
        piece.setPosition(firstX, firstY);
    }
    

    
    nextX() {
        return this.x + this.horizontalPadding + (this.pieces.length % this.cols) * this.spacing;
    }

    nextY() {
        return this.y + this.verticalPadding + Math.floor(this.pieces.length / this.cols) * this.spacing;
    }

    /* Select a piece, then click anywhere in your saved rack to save it -- the
       same outcome as dragging it there, which is not a gesture everyone finds.
       Wired from drawBackground so it survives a relayout, and the hit area is
       RESHAPED in place rather than re-setInteractive'd: setInteractive replaces
       the interactive object outright, which is how ghost dragging silently lost
       its draggable flag. */
    // Second half of a rack double-click, caught by the PANEL rather than by a
    // piece -- so it works whether or not another piece slid into the slot.
    _wireEntryTap(bx, by, bw, bh) {
        if (!this.background.input) {
            this.background.setInteractive(new Phaser.Geom.Rectangle(bx, by, bw, bh),
                                           Phaser.Geom.Rectangle.Contains);
            onTap(this.background, () => this.onEntryPanelTap());
        } else if (this.background.input.hitArea && this.background.input.hitArea.setTo) {
            this.background.input.hitArea.setTo(bx, by, bw, bh);
        }
    }

    onEntryPanelTap() {
        const game = this.scene && this.scene.game;
        if (!game || game.gameOver || !getSumToGoal()) return;
        const mark = game._rackSlotTap;
        if (!mark || mark.rack !== this) return;
        // Only while that piece is still sitting tentatively on the home tile --
        // the state the gesture is about. No time window needed here: a piece
        // that has since moved, been returned, or had a die spent on it fails
        // this test on its own.
        const p = mark.piece;
        if (!p || !p.justMovedHome || !p.currentTile || p.currentTile.type !== 'home') return;
        game._rackSlotTap = null;
        clearTimeout(p._hlTimer);
        if (game.sendToGoal(p)) _clearSelection(game);
        else if (game.selectedPiece === p && p.highlightReachableTiles) p.highlightReachableTiles();
    }

    _wireSaveTap(bx, by, bw, bh) {
        if (!this.background.input) {
            this.background.setInteractive(new Phaser.Geom.Rectangle(bx, by, bw, bh),
                                           Phaser.Geom.Rectangle.Contains);
            onTap(this.background, () => this.onSaveTap());
        } else if (this.background.input.hitArea && this.background.input.hitArea.setTo) {
            this.background.input.hitArea.setTo(bx, by, bw, bh);
        }
    }

    onSaveTap() {
        const game = this.scene && this.scene.game;
        if (!game || game.gameOver) return;
        const piece = game.selectedPiece;
        if (!piece) return;
        const mySaved = piece.player === 'white' ? game.whiteSavedRack : game.blackSavedRack;
        if (this !== mySaved) return;              // only ever your own rack
        // Same order as dropping it here: save from where it stands, else the
        // two-dice walk-to-a-goal-and-save.
        if (piece.canBeSaved && piece.canBeSaved() && piece.save()) {
            game._saveGuardUntil = Date.now() + 250;
            _clearSelection(game);
            return;
        }
        if (game.sumSave(piece)) _clearSelection(game);
    }

    drawBackground() {
        // Clear first: this is redrawn on rotation now, and a Graphics replays
        // its entire command list every frame.
        this.background.clear();
        // Clean Modern (matches mockup): white rounded panel + soft shadow +
        // faint empty capacity slots. No text.
        const bx = this.x - this.pr, by = this.y - this.pr;
        const bw = this.cols * this.spacing + this.pr;
        const bh = this.rows * this.spacing + this.pr + this.verticalPadding;
        this.background.fillStyle(0x000000, 0.07);
        this.background.fillRoundedRect(bx, by + 5, bw, bh, 16);      // soft drop shadow
        this.background.fillStyle(0xffffff, 1);
        this.background.fillRoundedRect(bx, by, bw, bh, 16);
        this.background.lineStyle(1.5, 0xdbe1ea, 1);
        this.background.strokeRoundedRect(bx, by, bw, bh, 16);
        if (this.type === 'saved') this._wireSaveTap(bx, by, bw, bh);
        // The unentered panel takes taps as well, for the send-to-goal gesture.
        // The first tap of that double-click moves the piece OFF the rack, so
        // when it was the LAST piece the slot is left empty and the second tap
        // lands on bare panel: no piece, no handler, no gesture. That is the
        // whole bug owner kept reporting (the 5, the 6, the 2 -- each the last
        // piece; the 1 worked because others slid up behind it).
        if (this.type === 'unentered') this._wireEntryTap(bx, by, bw, bh);
        // faint slot circles show the rack's capacity (like the mockup)
        this.background.lineStyle(1.5, 0xdbe1ea, 0.85);
        for (let i = 0; i < this.cols * this.rows; i++) {
            const sx = this.x + this.horizontalPadding + (i % this.cols) * this.spacing;
            const sy = this.y + this.verticalPadding + Math.floor(i / this.cols) * this.spacing;
            this.background.strokeCircle(sx, sy, this.pr);
        }
    }
}



class Die {
    constructor(scene, x, y, isFirstDie, size) {
        this.scene = scene;
        this.value = Phaser.Math.Between(1, 6);
        this.x = x;
        this.y = y;
        this.size = size || DIE_SIZE;
        this.used = false;
        this.isFirstDie = isFirstDie;

        this.graphics = scene.add.graphics();
        this.drawDie();
    }

    roll() {
        this.value = Phaser.Math.Between(1, 6);
        this.used = false;
        this.drawDie();
    }

    setUsed() {
        this.used = true;
        this.drawDie();
        if (typeof _updateHudDice === 'function') _updateHudDice();
    }

    drawDie() {
        const color = this.used ? 0x808080 : 0xffffff;
        this.drawDieWithColor(color, 0x000000);
    }

    updateColor(turn) {
        const color = this.used ? 0x808080 : (turn === 'white' ? 0xffffff : 0x000000);
        const dotColor = turn === 'white' ? 0x000000 : 0xffffff;
        this.drawDieWithColor(color, dotColor);
    }

    drawDieWithColor(dieColor, dotColor) {
        this.graphics.clear();
        // Nothing to show before the player has actually started a game. The
        // board sits frozen behind the welcome screen with a rolled pair, but
        // that roll is discarded -- Play starts a fresh game -- so displaying it
        // just shows two values that are never used. The real game runs through
        // a new create(), which builds new dice with the flag already cleared.
        //
        // _gameFrozen only covers the FIRST load. Cancelling out of a game or a
        // match mid-session puts the same cards over a board whose dice are
        // equally moot, and there the flag is already false -- so ask what is on
        // screen as well (owner: "when cancelling game/match, dice should
        // disappear"). _redrawDice below repaints them when a card comes or goes.
        if (_gameFrozen || _preGameCardUp()) return;
        paintDie(this.graphics, this.x, this.y, this.size, this.value, {
            dieColor, dotColor,
            borderColor: this.isFirstDie ? colorFirstDie : colorSecondDie,
            // 5 world px is only ~1.6 CSS px on a phone -- the colour coding was
            // effectively invisible there.
            bw: _isPhone() ? 14 : 5,
        });
    }
}



// One die face, drawn into any Graphics. Shared by the board dice and by the
// pinned readout that appears when they are scrolled out of view.
function paintDie(gfx, x, y, size, value, { dieColor, dotColor, borderColor, bw }) {
    const r = size * 0.14;
    gfx.fillStyle(0x000000, 0.10);
    gfx.fillRoundedRect(x, y + size * 0.04, size, size, r);        // soft shadow
    // The colour-coded border is a slightly larger filled rounded rect BEHIND
    // the face rather than a stroke: a thick stroked path here left stray
    // coloured lines running across the board on some Android GPUs (the WebGL
    // line batch joining onto the next shape). Two fills have no path to leak.
    gfx.fillStyle(borderColor, 1);
    gfx.fillRoundedRect(x - bw / 2, y - bw / 2, size + bw, size + bw, r + bw / 2);
    gfx.fillStyle(dieColor, 1);
    gfx.fillRoundedRect(x, y, size, size, r);

    const dot = size * 0.11, off = size / 4, mid = size / 2;
    const drawDot = (dx, dy) => { gfx.fillStyle(dotColor, 1); gfx.fillCircle(x + dx, y + dy, dot); };
    if ([1, 3, 5].includes(value)) drawDot(mid, mid);
    if (value > 1) { drawDot(off, off); drawDot(size - off, size - off); }
    if (value > 3) { drawDot(off, size - off); drawDot(size - off, off); }
    if (value === 6) { drawDot(off, mid); drawDot(size - off, mid); }
}

class Game {
    constructor(scene, startingPlayer = 'white', debug = false) {
        this.scene = scene;
        this.players = [new Player('white', WHITE_IS_AI), new Player('black', BLACK_IS_AI)];
        this.startingPlayer = startingPlayer;
        this.turn = this.startingPlayer;
        const _f = _fur();
        this.dice = [new Die(scene, _f.diceX[0], _f.diceY, true, _f.dieSize),
                     new Die(scene, _f.diceX[1], _f.diceY, false, _f.dieSize)];
        this.gameOver = false;
        this.instanceId = ++_gameInstanceSeq;   // see getAgentMoves: drop stale replies
        this.score = { 'white': 0, 'black': 0 };
        this.selectedPiece = null;
        this.fullPassCounter = 0;

        // No-save draw rule state (frontend mirror; authoritative rule also
        // lives in game.py for agent-vs-agent play). Counts FULL ROUNDS with
        // no save once both players are past the opening. Resets to 0 the
        // instant any piece is saved; undo-sensitive via captureState.
        this.noSaveTurns = 0;            // completed rounds with no save
        this.drawCallable = false;
        this.lastTotalSaved = 0;         // saved-count snapshot at last turn boundary
        this._halfTurnsSinceRound = 0;   // 2 player-turns = 1 round


        // Initialize racks
        // saved rack sits flush beneath the unentered rack (panel height with
        // rows=4 is 240px), so each side's two racks touch like the mockup.
        // Two racks per side, stacked flush and vertically centred on the board
        // (panel height with rows=4 is 266px; block of two = 532, centred at 600).
        this.whiteUnenteredRack = new Rack(scene, _f.whiteUn[0], _f.whiteUn[1], 'white', 'unentered', _f.rows, _f.cols);
        this.whiteSavedRack = new Rack(scene, _f.whiteSv[0], _f.whiteSv[1], 'white', 'saved', _f.rows, _f.cols);
        this.blackUnenteredRack = new Rack(scene, _f.blackUn[0], _f.blackUn[1], 'black', 'unentered', _f.rows, _f.cols);
        this.blackSavedRack = new Rack(scene, _f.blackSv[0], _f.blackSv[1], 'black', 'saved', _f.rows, _f.cols);

        this.setupDragging(scene);
        this.setupCameraControls(scene);

        // Create buttons
        this.createSwitchTurnButton(scene);
        this.createUndoButton(scene);


        // Initialize game elements
        this.tiles = [];
        this.pieces = [];
        this.movedOnce = false;

        // Create tiles and pieces
        this.createTiles({ x: CENTER_X, y: CENTER_Y }, 7, 12, HOME_TILE_RADIUS, TILE_RADIUS_STEP);
        this.createPieces();

        // Roll dice and update movable pieces
        this.rollDice();
        this.updateMovablePieces();

        // Capture initial state
        this.state = this.captureState();
        this.undoStack = [];         // per-move undo snapshots for the current turn
        this._pendingPreMove = null; // pre-selection snapshot for the next move
        this.turnStartState = null;  // populated after each rollDice in switchTurn

                // Set players to endgame if debug mode is active
                this.debug = debug; // Add debug flag
                if (this.debug) {
                    this.players.forEach(player => player.setGamePhase('endgame'));
                }
    }



    createPieces() {
        let whitePieces = [];
        let blackPieces = [];
        for (let i = 1; i <= TOTAL_PIECES; i++) {
            whitePieces.push(new Piece(this.scene, this, 0xffffff, i, 0, 0, this.whiteUnenteredRack));
            blackPieces.push(new Piece(this.scene, this, 0x000000, i, 0, 0, this.blackUnenteredRack));
        }

        whitePieces = Phaser.Utils.Array.Shuffle(whitePieces);
        blackPieces = Phaser.Utils.Array.Shuffle(blackPieces);

        whitePieces.forEach(piece => {
            piece.setSize(piece.rack ? piece.rack.pr : _rackPR());
            piece.setPosition(this.whiteUnenteredRack.nextX(), this.whiteUnenteredRack.nextY());
            this.whiteUnenteredRack.addPiece(piece);
        });

        blackPieces.forEach(piece => {
            piece.setSize(piece.rack ? piece.rack.pr : _rackPR());
            piece.setPosition(this.blackUnenteredRack.nextX(), this.blackUnenteredRack.nextY());
            this.blackUnenteredRack.addPiece(piece);
        });

        this.pieces = whitePieces.concat(blackPieces);
    }

    createTiles(center, numRings, numSegments, innerRadius, segmentWidth) {

    
        // Central circle as 'home'
        this.tiles.push(new Tile(this.scene, this,  'home', 0, 0, 0, 2 * Math.PI, 0, innerRadius));
 
    
        const goalTileNumbers = [4, 2, 5, 3, 6, 1];

    
        for (let r = 0; r < numRings; r++) {
            let rInner = innerRadius + r * segmentWidth;
            let rOuter = rInner + segmentWidth;
            for (let s = 0; s < numSegments; s++) {
                let startAngle = s * (2 * Math.PI / numSegments);
                let endAngle = startAngle + (2 * Math.PI / numSegments);
    

    
                if (r === numRings - 1) { // Special handling for the outermost ring (Ring 7)

                    if (s % 4 === 2) {
                        let subSegmentAngle = (2 * Math.PI / numSegments) / 3;

                        for (let miniTile = 0; miniTile < 3; miniTile++) {
                            let miniStartAngle = startAngle + miniTile * subSegmentAngle;
                            let miniEndAngle = miniStartAngle + subSegmentAngle;

                            this.tiles.push(new Tile(this.scene, this, 'field', r + 1, (s + 4) * 3 + miniTile + 1, miniStartAngle, miniEndAngle, rInner, rInner + segmentWidth));
                        }
                    } else {
                        let tileType = s % 2 === 0 ? 'nogo' : 'save';
                        if (tileType === 'save') {
                            rOuter = rInner + (segmentWidth * 1.5);
                            let number = goalTileNumbers[Math.floor(s / 2) % goalTileNumbers.length];
  
                            this.tiles.push(new Tile(this.scene, this, tileType, r + 1, s + 1, startAngle, endAngle, rInner, rOuter, number));
                        } else {
        
                            this.tiles.push(new Tile(this.scene, this, tileType, r + 1, s + 1, startAngle, endAngle, rInner, rOuter));
                        }
                    }
                } else {
                    let tileType = 'field';
                    if (r === 0 && s % 4 === 0) { // Every 4th tile in Ring 1
                        tileType = 'nogo';
                 
                    } else if ((r === 1 || r === 4) && (s + 2) % 4 === 0) { // Every 4th tile offset by 2 in Ring 2
                        tileType = 'nogo';
                    
                    } else if ((r === 3 || r === 5) && s % 2 === 0) { // Every other tile in Rings 4 and 6
                        tileType = 'nogo';
                       
                    } else if (r === 4 && s % 4 === 0) { // Every 4th tile in Ring 5
                        let subSegmentAngle = (2 * Math.PI / numSegments) / 2; // Half-size tiles
                    
                        for (let miniTile = 0; miniTile < 2; miniTile++) {
                            let miniStartAngle = startAngle + miniTile * subSegmentAngle;
                            let miniEndAngle = miniStartAngle + subSegmentAngle;
                   
                            this.tiles.push(new Tile(this.scene, this, 'field', r + 1, (s + 6) * 2 + miniTile + 1, miniStartAngle, miniEndAngle, rInner, rOuter));
                        }
                        continue; // Skip adding the original tile
                    }
             
                    this.tiles.push(new Tile(this.scene, this, tileType, r + 1, s + 1, startAngle, endAngle, rInner, rOuter));
                }
            }
        }
        this.assignNeighbors(numSegments);
        this.assignHardcodedNeighbors(); 
        this.hideOuterNogoTiles()
        
    }
    
    hideOuterNogoTiles() {
        this.tiles.forEach(tile => {
            if (tile.type === 'nogo' && tile.ring === 6) {
                const coveredByRing7 = this.tiles.some(t => 
                    t.ring === 7 && 
                    t.type !== 'nogo' &&
                    t.startAngle < tile.endAngle && 
                    t.endAngle > tile.startAngle
                );
                if (!coveredByRing7) {
                    tile.graphics.clear();
                    tile.fillColor = BACKGROUND_COLOR;
                    tile.lineColor = BACKGROUND_COLOR;

                    // Mark the ring-7 tiles that need a closing arc along their
                    // inner edge, and let drawTile draw it. It used to be poked
                    // straight into the Graphics here, which meant ANY later
                    // redraw silently erased it -- hovering one of these tiles
                    // already lost the arc for good, and baking the board (which
                    // redraws every tile once) would have lost all of them.
                    this.tiles.forEach(t => {
                        if (t.ring === 7 && t.type !== 'nogo' &&
                            t.startAngle < tile.endAngle &&
                            t.endAngle > tile.startAngle) {
                            t._innerArc = true;
                            t.drawTile();
                        }
                    });
                }
            }
        });
    }

    assignNeighbors(numSegments) {

        this.tiles.forEach(tile => {
            // Skip 'nogo' tiles
            if (tile.type === 'nogo') return;

            // Identify neighbors in the same ring
            this.tiles.forEach(otherTile => {
                if (otherTile === tile || otherTile.type === 'nogo' || otherTile.ring === 0) return;
                if (otherTile.ring === tile.ring) {
                    // Check for adjacent sectors
                    if (Math.abs(otherTile.sector - tile.sector) === 1 || (tile.ring < 4 && Math.abs(otherTile.sector - tile.sector) === numSegments - 1)) {
                        tile.neighbors.push(otherTile);
                    }
                }
            });

            // Identify neighbors in the adjacent rings
            const adjacentRings = [tile.ring - 1, tile.ring + 1];
            adjacentRings.forEach(ring => {
                this.tiles.forEach(otherTile => {
                    if (otherTile === tile || otherTile.type === 'nogo' || otherTile.ring === 0) return;
                    if (otherTile.ring === ring) {
                        if (otherTile.sector === tile.sector){
                            tile.neighbors.push(otherTile);
                        }
                    }
                });
            });
        });

        // Special case for the 'home' tile
        const homeTile = this.tiles.find(tile => tile.type === 'home');
        if (homeTile) {
            this.tiles.forEach(tile => {
                if (tile.ring === 1 && tile.type !== 'nogo') {
                    homeTile.neighbors.push(tile);
                }
            });
        }
    }

    assignHardcodedNeighbors() {
        const hardcodedNeighbors = [
            { ring: 5, sector: 30, neighborSector: 10 },
            { ring: 5, sector: 29, neighborSector: 8 },
            { ring: 5, sector: 14, neighborSector: 2 },
            { ring: 5, sector: 4, neighborSector: 21 },
            { ring: 5, sector: 22, neighborSector: 6 },
            { ring: 7, sector: 33, neighborSector: 8 },
            { ring: 7, sector: 31, neighborSector: 6 },
            { ring: 7, sector: 4, neighborSector: 21 },
            { ring: 7, sector: 19, neighborSector: 2 },
            { ring: 7, sector: 12, neighborSector: 45 },
            { ring: 7, sector: 43, neighborSector: 10 }

        ];

        hardcodedNeighbors.forEach(tileData => {
            const tile = this.tiles.find(t => t.ring === tileData.ring && t.sector === tileData.sector);
            const neighbor = this.tiles.find(t => t.ring === tileData.ring && t.sector === tileData.neighborSector);

            if (tile && neighbor) {
                if (!tile.neighbors.includes(neighbor)) {
                    tile.neighbors.push(neighbor);
                }
                if (!neighbor.neighbors.includes(tile)) {
                    neighbor.neighbors.push(tile);
                }
            }
        });
    }



    isBlocked(tile) {
        const opponentPieces = tile.pieces.filter(p => p.player !== this.turn);
        const isBlocked = tile.type === 'field' && opponentPieces.length > 1;
        return isBlocked;
    }
    
    isHigherNumberedGoalOccupied(player, saveTileNumber) {
        const playerColor = player.name === 'white' ? 0xffffff : 0x000000;

        // Iterate over all save tiles
        for (const tile of this.tiles) {
            if (tile.type === 'save' && tile.number > saveTileNumber) {
                // Check if any piece on this tile belongs to the player
                for (const piece of tile.pieces) {
                    if (piece.color === playerColor) {
                        return true; // A higher-numbered goal is occupied by the player's piece
                    }
                }
            }
        }
        return false; // No higher-numbered goal occupied by the player's piece
    }

    // True if a numbered piece #dieValue of `color` is sitting on its own goal
    // (goal dieValue) still awaiting a save — that piece can ONLY be saved with a
    // die of exactly dieValue, so smart-saving reserves that value for it.
    numberedPieceNeedsDie(dieValue, color) {
        if (dieValue < 1 || dieValue > 6) return false;
        return this.pieces.some(p => p.color === color && p.number === dieValue &&
            p.currentTile && p.currentTile.type === 'save' && p.currentTile.number === dieValue);
    }

    checkEndgame(player) {
        if (this.debug) {
            player.setGamePhase('endgame');
            console.log(`${player.name} is in debug mode and will stay in the endgame phase.`);
            return;
        }

        const pieces = this.pieces.filter(piece => piece.color === (player.name === 'white' ? 0xffffff : 0x000000));
        const allCanBeSaved = pieces.every(piece => piece.canBeSaved());

        // Check if all pieces have been moved onto the board and can be saved
        if (player.getGamePhase() === 'midgame' && allCanBeSaved) {
            player.setGamePhase('endgame');
            console.log(`${player.name} has entered the endgame`);
        } else if (player.getGamePhase() === 'endgame' && !allCanBeSaved) {
            player.setGamePhase('midgame');
            console.log(`${player.name} has reverted to the midgame`);
        }
    }

    getReachableTiles(startTile, steps) {
        if (!startTile) {           // if piece is still on rack, pretend it's on the home square
            startTile = this.tiles.find(tile => tile.type === 'home');
        }

        const queue = [{ tile: startTile, stepsTaken: 0 }]; // Start with the current tile and 0 steps taken
        const visited = new Set();
        const reachableTiles = [];
    
        while (queue.length > 0) {
            const { tile: currentTile, stepsTaken: currentSteps } = queue.shift();
            if (currentSteps < steps) {
                currentTile.neighbors.forEach(neighbor => {
                    if (!visited.has(neighbor) && neighbor.type !== 'nogo' && neighbor.type !== 'home' && !this.isBlocked(neighbor)) {
                        queue.push({ tile: neighbor, stepsTaken: currentSteps + 1 });
                        visited.add(neighbor);
                        if (currentSteps + 1 === steps) {
                            reachableTiles.push(neighbor);
                        }
                    }
                });
            } else if (currentSteps === steps) {
                reachableTiles.push(currentTile);
            }
        }
    
        return [...new Set(reachableTiles)]; // Ensure unique tiles in the result
    }
    
    getReachableTilesByDice(piece) {
        if (!piece) return null;

        // Die-index aware: reachableByFirstDie is die[0] (teal), reachableBySecondDie
        // is die[1] (pink), each empty if that die is already used; reachableBySum
        // (yellow) exists only while BOTH dice are unused. Computed fresh every call
        // (no stale-reachableTiles filtering) so re-selecting or a second-die move
        // always sees the correct destinations.
        const d0 = this.dice[0], d1 = this.dice[1];
        if (d0.used && d1.used) return null;   // no available dice

        let reachableByFirstDie  = d0.used ? [] : this.getReachableTiles(piece.currentTile, d0.value);
        let reachableBySecondDie = d1.used ? [] : this.getReachableTiles(piece.currentTile, d1.value);
        let reachableBySum = (!d0.used && !d1.used)
            ? this.getReachableTiles(piece.currentTile, d0.value + d1.value) : [];

        const homeTile = this.tiles.find(tile => tile.type === 'home');
        if (homeTile.pieces.filter(p => p.color === piece.color).length > 1) {
            reachableBySum = [];   // >1 captured piece: no combined (sum) move
        }
        // A die has to be kept back for the obligatory piece(s), so a piece that
        // is not itself obligatory cannot spend both on a sum move. movePiece
        // already refuses it; without this the sum destinations still lit up,
        // offering moves that would then be rejected.
        const must = this.mustMovePieces || [];
        if (must.length > 0 && !must.includes(piece)) reachableBySum = [];
        // While a CAPTURED piece is on home nothing else may move at all, so no
        // other piece gets any destinations -- otherwise they light up and the
        // move is then refused, the same complaint that motivated the sum line
        // above.
        // `must.length > 0` as well as the derived check: belt and braces, so an
        // empty obligation list can never blank out every piece's destinations.
        if (must.length > 0 && !must.includes(piece) && this.hasCapturedOnHome()) {
            return { reachableByFirstDie: [], reachableBySecondDie: [], reachableBySum: [] };
        }

        // Taking the SECOND rack piece first is a reordering, not a deferral:
        // the front piece must still enter this turn, so a die may only go to
        // the second piece if the front can still enter with the OTHER one.
        // (No simulation needed -- entering never blocks a later entry, since
        // own pieces stack and a capture only removes an enemy.) Mirrors
        // game.py's _filter_second_entries.
        if (piece === _secondEntrant(this) || piece === this._reorderEntry) {
            const front = (this.turn === 'white' ? this.whiteUnenteredRack : this.blackUnenteredRack).pieces[0];
            const home = this.tiles.find(t => t.type === 'home');
            const frontBy0 = !d0.used && this.getReachableTiles(home, d0.value).length > 0;
            const frontBy1 = !d1.used && this.getReachableTiles(home, d1.value).length > 0;
            if (!frontBy1) reachableByFirstDie = [];    // would strand the front piece
            if (!frontBy0) reachableBySecondDie = [];
            reachableBySum = [];                        // it may not spend both dice
            void front;
        }

        // Shortest-path enforcement for a piece moved with both dice one-by-one:
        // once it has advanced from its turn-start tile, the remaining die must
        // keep going *forward* (to a tile whose shortest distance from the
        // turn-start tile equals the cumulative pips), never backtrack.
        const start = piece._turnStartTile;
        if (start && piece.currentTile && start !== piece.currentTile) {
            const dist = this._bfsDistances(start);
            const moved = dist.get(piece.currentTile);
            if (moved != null) {
                const keep = (tiles, v) => tiles.filter(t => dist.get(t) === moved + v);
                reachableByFirstDie  = keep(reachableByFirstDie,  d0.value);
                reachableBySecondDie = keep(reachableBySecondDie, d1.value);
                // reachableBySum is already [] here (both dice were needed to reach it)
            }
        }
        // AUTOMATIC EN-ROUTE CAPTURE, turned off: a sum move whose ROUTE decides
        // what gets captured is withheld, and the player moves one die at a time
        // to say which way they meant.
        //
        // The test is simply "two or more routes, at least one passing a lone
        // enemy". A two-die route has exactly ONE intermediate tile and the
        // capture happens there, so distinct routes have distinct intermediates
        // and therefore distinct outcomes -- there is no need to compare what
        // each route would take. With no capturable piece on any route every
        // route ends in the same position, so the single gesture stays honest.
        let ambiguousSum = [];
        if (!getAutoEnRouteCapture() && reachableBySum.length && !d0.used && !d1.used) {
            const from = piece.currentTile || this.tiles.find(t => t.type === 'home');
            const capturable = (t) => t && t.type !== 'save' && t.pieces.length === 1 &&
                                      t.pieces[0].color !== piece.color;
            // destination -> the set of intermediates that reach it. Built once
            // for the whole roll rather than per destination: 2 + |A| + |B| BFS
            // instead of a fresh pair for every sum target.
            const routes = new Map();
            const addRoutes = (mids, otherVal) => mids.forEach(m => {
                this.getReachableTiles(m, otherVal).forEach(dest => {
                    if (!routes.has(dest)) routes.set(dest, new Set());
                    routes.get(dest).add(m);
                });
            });
            addRoutes(this.getReachableTiles(from, d0.value), d1.value);
            addRoutes(this.getReachableTiles(from, d1.value), d0.value);   // doubles: same set, Set dedupes

            ambiguousSum = reachableBySum.filter(t => {
                const mids = routes.get(t);
                return mids && mids.size >= 2 && [...mids].some(capturable);
            });
            if (ambiguousSum.length) {
                const amb = new Set(ambiguousSum);
                reachableBySum = reachableBySum.filter(t => !amb.has(t));
            }
        }

        // The tutorial hard-blocks: off-script destinations are dropped here, so
        // they are neither highlighted nor accepted by movePiece.
        if (_tut.active) return _tutFilterReach(this, piece, { reachableByFirstDie, reachableBySecondDie, reachableBySum });
        // ambiguousSum rides along so movePiece can tell "withheld on purpose"
        // apart from "simply not reachable", and say so.
        return { reachableByFirstDie, reachableBySecondDie, reachableBySum, ambiguousSum };
    }

    // Shortest (BFS) distance from startTile to every reachable tile, respecting
    // the same blocked/nogo/home rules as movement.
    _bfsDistances(startTile) {
        const dist = new Map([[startTile, 0]]);
        const queue = [startTile];
        while (queue.length) {
            const t = queue.shift(), d = dist.get(t);
            t.neighbors.forEach(n => {
                if (n.type !== 'nogo' && n.type !== 'home' && !this.isBlocked(n) && !dist.has(n)) {
                    dist.set(n, d + 1); queue.push(n);
                }
            });
        }
        return dist;
    }
    
    
    
    movePiece(piece, targetTile, getReachableTiles = false) {
        if (!piece || !targetTile) return false;

        let reachableTiles = piece.reachableTiles;

        // The cache is empty (undo cleared it, or nothing selected this piece).
        // Still explain a deliberately withheld route before giving up.
        if (!reachableTiles && !getReachableTiles) {
            _noticeIfRouteWithheld(this, piece, targetTile);
            return false;
        }

        if (!reachableTiles) {  // this is called from AI agent's applyMove
            reachableTiles = this.getReachableTilesByDice(piece);
            piece.reachableTiles = reachableTiles}  

        if (!reachableTiles) return false;

        const { reachableByFirstDie, reachableBySecondDie, reachableBySum } = reachableTiles;

        const allReachableTiles = new Set([...reachableByFirstDie, ...reachableBySecondDie, ...reachableBySum]);

        // Withheld on purpose, as opposed to simply out of range.
        if (!allReachableTiles.has(targetTile)) {
            _noticeIfRouteWithheld(this, piece, targetTile);
            return false;
        }

        if (allReachableTiles.has(targetTile)) {

            if (this.isBlocked(targetTile)) {
                return false; // Can't move to a tile with more than one opposing piece
            }

            // Obligatory-move ordering: a non-obligatory move must leave a die for
            // every still-pending obligatory piece.
            if (this.mustMovePieces.length > 0 && !this.mustMovePieces.includes(piece)) {
                // Absolute while a captured piece is on home -- see canSelectForMove.
                if (this.hasCapturedOnHome()) {
                    console.log('A captured piece must move first');
                    _flashMustMove(this);
                    return false;
                }
                const unused = this.dice.filter(d => !d.used).length;
                const willUse = (reachableByFirstDie.includes(targetTile) ||
                                 reachableBySecondDie.includes(targetTile)) ? 1 : 2;
                if (unused - willUse < this.mustMovePieces.length) {
                    console.log('Must keep a die for the obligatory piece(s)');
                    _flashMustMove(this);
                    return false;
                }
            }

            // snapshot BEFORE this move so undo reverts just it. Prefer the
            // pre-selection snapshot (captured while an entering piece was still
            // on the rack) so undoing an entry returns it to the rack, not home.
            this.undoStack.push(this._pendingPreMove || this.captureState());
            this._pendingPreMove = null;

            if (targetTile.type === 'field' && targetTile.pieces.length === 1 && targetTile.pieces[0].color !== piece.color) {
                this.capturePiece(targetTile.pieces[0]); // Capture the opposing piece
            }
            
            

            const d0 = this.dice[0], d1 = this.dice[1];

            // check en route capture (only a genuine two-dice sum move)
            if (reachableBySum.includes(targetTile)) {
                this.checkEnRouteCapture(piece, targetTile);
            }

            const _ox = piece.x, _oy = piece.y;   // for the slide animation
            piece.isHovered = false;              // it is not under the pointer any more
            piece.move(targetTile);
            piece.animateFrom(_ox, _oy);
            SFX.move();

            const homeTile = this.tiles.find(tile => tile.type === 'home');
            if (homeTile.pieces.includes(piece)) homeTile.removePiece(piece);

            // reachableByFirstDie <-> die[0], reachableBySecondDie <-> die[1];
            // anything else reachable is the sum (both dice).
            if (reachableByFirstDie.includes(targetTile)) {
                d0.setUsed();
            } else if (reachableBySecondDie.includes(targetTile)) {
                d1.setUsed();
            } else {
                d0.setUsed();
                d1.setUsed();
            }
    
            if (!this.movedOnce) this.movedOnce = true;

                    // If the moved piece was in the mustMovePieces list, remove it
            if (this.mustMovePieces.includes(piece)) {
            this.mustMovePieces = this.mustMovePieces.filter(p => p !== piece);
            }
            if (typeof updateMustMoveHighlights === 'function') updateMustMoveHighlights(this);
            if (typeof _updateViewportHud === 'function') _updateViewportHud();

            // clear the now-stale reachability so the next selection recomputes
            // it fresh (a leftover set from before the move otherwise wrongly
            // filters the second-die reachable tiles down to nothing).
            piece.reachableTiles = null;

            refreshEvalReadout();  // update on-board eval after the move settles
            this.maybeAutoEndTurn();
            return true;
        }
    
        console.log('Target tile is not reachable by the available dice rolls');
        if (_tut.active) {
            // Shake the instructions, and drop the selection: a still-selected
            // piece turns the next click on a board piece into a tile click, so
            // the player could not then pick the piece the step asks for.
            _tutNudge();
            _clearSelection(this);
        }
        return false;
    }

    // If a piece (not yet on a goal) can reach its goal with one die AND be saved
    // with the other in the same turn, do both at once — so a single double-click
    // or a drag to the saved rack saves it without first parking it on the goal.
    // Returns true if it happened.
    // Reach a goal with one die and be saved from it with the other, in one
    // gesture. Also serves a piece that is ALREADY on a goal but cannot be saved
    // from it with this roll: goal pairs are 4 tiles apart, so a 4 walks it to
    // the other goal and the second die saves it there. That is two ordinary
    // moves, so it is a frontend affordance, not a rule change -- the engine has
    // always allowed the sequence and the agent already searches it.
    sumSave(piece) {
        if (!piece.currentTile) return false;
        if (piece.player !== this.turn) return false;
        if (this.dice[0].used || this.dice[1].used) return false;   // need both dice
        const player = piece.color === 0xffffff ? this.players[0] : this.players[1];
        if (player.getGamePhase() === 'opening') return false;

        const goals = this.tiles.filter(t => t.type === 'save' &&
            (piece.number > 6 ? true : t.number === piece.number));
        const r = this.getReachableTilesByDice(piece);
        if (!r) return false;
        piece.reachableTiles = r;

        // The move that lands this piece on `goal` can itself be what starts the
        // endgame (it's the last piece off the field), so the endgame rule has to
        // be judged against the position AFTER the move, not the current phase --
        // otherwise dragging your last field piece onto its goal and out in one
        // gesture is refused, while doing it in two steps works.
        const endgameAfterMove = this.pieces
            .filter(p => p.color === piece.color && p !== piece)
            .every(p => p.canBeSaved());
        const canSaveFrom = (goal, dieVal) => {
            if (dieVal === goal.number) return true;
            return piece.number > 6 &&
                (player.getGamePhase() === 'endgame' || endgameAfterMove) &&
                dieVal > goal.number && !this.isHigherNumberedGoalOccupied(player, goal.number);
        };
        for (const goal of goals) {
            if (goal === piece.currentTile) continue;   // already here: nothing to walk
            // movePiece consumes die[0] if the goal is reachable by it, else die[1];
            // the *other* die must then be able to save from the goal.
            const byFirst = r.reachableByFirstDie.includes(goal);
            const bySecond = r.reachableBySecondDie.includes(goal);
            const saveDieVal = byFirst ? this.dice[1].value : (bySecond ? this.dice[0].value : null);
            if (saveDieVal !== null && canSaveFrom(goal, saveDieVal)) {
                if (this.movePiece(piece, goal) && piece.save()) return true;
            }
        }
        return false;
    }

    // Optional gesture (settings, off by default): send a piece to a goal it can
    // reach on the DICE SUM. Deliberately sum-only -- a single-die route is one
    // ordinary move that the player can already make by tapping the destination,
    // and shortcutting it would take a die they might want elsewhere.
    //
    // A numbered piece only ever targets its OWN goal. A blank targets any goal,
    // but only when exactly one is reachable: with two there is no way to know
    // which one the player meant, so it does nothing rather than guess.
    //
    // Everything legality-related is delegated to getReachableTilesByDice, which
    // already encodes the entry obligations, the second-entrant reordering rule,
    // the ">1 captured piece" ban on sum moves, shortest-path enforcement and the
    // tutorial's hard block. And because the destination is in reachableBySum,
    // movePiece runs checkEnRouteCapture for us -- so en-route capture is
    // preserved by construction rather than by a second implementation.
    sendToGoal(piece) {
        // Traced: this declines for several legitimate reasons and they are
        // indistinguishable on screen. console.log is silent without ?dev=1.
        // SAY WHY. Every decline used to be silent -- the piece just sat there --
        // so "my dice cannot reach a goal" was indistinguishable from "the
        // feature is broken", and owner reported the latter three times when the
        // log shows it was always the former. Only for the declines a player can
        // act on; the internal ones (wrong turn, game over) stay quiet.
        const no = (why, extra, tell) => {
            console.log('[send-to-goal] not applied:', why, extra || '');
            if (tell && typeof flashNotice === 'function') flashNotice(tell, 3500);
            return false;
        };
        if (!getSumToGoal()) return no('the "Double-click sends a piece to its goal" setting is OFF');
        if (!piece || piece.player !== this.turn) return no('not this player\'s piece');
        if (this.gameOver) return no('game over');

        // NOT for a piece already standing on a goal it can use (owner). A
        // double-click there means "save", and if it cannot save with this roll
        // the answer is nothing -- not a wander off to another goal, which would
        // give up a banking square it was already on. `save()` and `sumSave()`
        // run BEFORE this in handleDoubleClick and are unaffected: the second
        // walks to another goal but BANKS there, which is always progress.
        //
        // A numbered piece on the WRONG goal is not on a goal it can use, so the
        // shortcut still applies and can carry it to its own -- which is exactly
        // the case owner wanted kept.
        const here = piece.currentTile;
        if (here && here.type === 'save' &&
            (piece.number > 6 || here.number === piece.number)) {
            return no('already on a goal it can use');
        }

        const r = this.getReachableTilesByDice(piece);
        if (!r) return no('no reachable set (both dice used?)');
        piece.reachableTiles = r;

        // Which goals may this piece target at all. Numbered pieces can only ever
        // match one, so the "exactly one" rule below bites for blanks.
        const eligible = (list) => [...new Set(list)].filter(t => t.type === 'save' &&
            (piece.number > 6 ? true : t.number === piece.number));

        // EVERY route counts toward ambiguity, not just the cheapest tier (owner
        // confirmed): one goal reachable by a single die and a DIFFERENT one by
        // the sum is exactly the case where there is no way to know which was
        // meant, so it does nothing. Only for blanks in practice -- a numbered
        // piece has one eligible goal, and the same goal can never appear in two
        // tiers, since movement is exact-distance and a tile sits at one BFS
        // depth.
        const goals = eligible([...r.reachableBySum,
                                ...r.reachableByFirstDie, ...r.reachableBySecondDie]);
        // The one decline worth explaining. Taking the SECOND rack piece first is
        // a reordering, so the front piece must still enter this turn and the
        // second may never spend both dice -- and every goal is exactly 7 from
        // the home tile, which always needs both. So the gesture simply cannot
        // apply to a second entrant, and without a word it looks broken: owner
        // hit it twice (a 2 and a 6, both not at the front) and read it as the
        // feature failing intermittently.
        if (goals.length === 0 && !this.dice[0].used && !this.dice[1].used &&
            (piece === this._reorderEntry || piece === _secondEntrant(this))) {
            if (typeof flashNotice === 'function') {
                flashNotice('The first piece on the rack must still enter this turn, so this one can’t use both dice.', 5000);
            }
            return no('second entrant: may not spend both dice');
        }
        if (goals.length !== 1) {
            const onHome = piece.currentTile && piece.currentTile.type === 'home';
            return no(goals.length === 0 ? 'no eligible goal in reach' : 'ambiguous: more than one eligible goal',
                      { piece: piece.number, eligibleGoals: goals.map(t => t.number),
                        dice: this.dice.map(d => d.value + (d.used ? '(used)' : '')),
                        sumTiles: r.reachableBySum.length },
                      goals.length === 0
                          // From the home tile every goal is exactly 7 away, so
                          // say the number -- that is the whole rule there.
                          ? (onHome ? 'No goal is 7 away on this roll — entering pieces need a total of exactly 7.'
                                    : 'No goal is within reach of this roll.')
                          : 'More than one goal is in reach, so move it by hand to choose.');
        }
        console.log('[send-to-goal] moving piece', piece.number, '-> goal', goals[0].number);

        // movePiece picks the die(s) itself: die[0] if the target is in its list,
        // else die[1], else both for a sum target -- which is also what makes
        // en-route capture fire on the sum route without asking for it.
        return this.movePiece(piece, goals[0]);
    }

    // Does the current player have any legal move left with the unused dice?
    // (Used to decide whether ending the turn is "risky".)
    hasAnyLegalMove() {
        if (this.dice.every(d => d.used)) return false;
        const color = this.turn === 'white' ? 0xffffff : 0x000000;
        const candidates = this.mustMovePieces.length > 0
            ? this.mustMovePieces.slice()
            : this.pieces.filter(p => p.color === color &&
                (p.currentTile || _isEntrant(p)));
        for (const p of candidates) {
            if (p.rack && p.rack.type === 'unentered' && !_isEntrant(p)) continue;
            const rt = this.getReachableTilesByDice(p);
            if (rt && (rt.reachableByFirstDie.length || rt.reachableBySecondDie.length || rt.reachableBySum.length)) return true;
            if (p.currentTile && p.currentTile.type === 'save' && p.canBeSaved && p.canBeSaved()) return true;
        }
        return false;
    }

    // If the setting is on and both dice are used, end the turn automatically
    // (after a short beat so the last move is visible).
    maybeAutoEndTurn() {
        if (!getAutoEndTurn() || this.gameOver) return;
        if (!this.dice.every(d => d.used)) return;
        // Obligatory pieces are pruned in movePiece, but a piece can also leave
        // the board by being saved; a stale entry here would block the auto-end
        // for the rest of the turn.
        if (this.mustMovePieces && this.mustMovePieces.length) {
            this.mustMovePieces = this.mustMovePieces.filter(
                p => p.currentTile || (p.rack && p.rack.type === 'unentered'));
        }
        // Deliberately NOT returning on a remaining obligation: every die is
        // already spent by this point, so an obligation that has not been met
        // can no longer be met, and blocking the auto-end just strands the
        // player on a finished turn. This is what made a dice-SUM entry (which
        // spends both dice at once, leaving the rack still non-empty and so
        // still "obligatory") never hand the turn over.
        if (this._autoEndScheduled) return;
        this._autoEndScheduled = true;
        const t = this.turn;
        setTimeout(() => {
            this._autoEndScheduled = false;
            if (!this.gameOver && this.turn === t && this.dice.every(d => d.used)) this.switchTurn();
        }, 550);
    }

    capturePiece(piece) {
        const homeTile = this.tiles.find(tile => tile.type === 'home');
        if (homeTile) {
            fxBurst(this.scene, piece.x, piece.y, 0xff5555);   // capture flash at the spot
            SFX.capture();
            piece.move(homeTile);
            piece.currentTile = homeTile;
            console.log(`Piece captured and sent to home tile: ${piece.color} ${piece.number}`);
        }
    }

    checkEnRouteCapture(piece, targetTile) {
        console.log('Checking en route capture');
    
        const diceValues = this.dice.filter(die => !die.used).map(die => die.value);
        if (diceValues.length < 2) return; // Ensure there are two dice values
    
        const [firstDieValue, secondDieValue] = diceValues;
    
        // Calculate reachable tiles using each die value separately
        const reachableWithFirstDie = this.getReachableTiles(piece.currentTile, firstDieValue);
        const reachableWithSecondDie = this.getReachableTiles(piece.currentTile, secondDieValue);
    
        // Find all intermediate tiles leading to the target tile
        const intermediateTiles1 = reachableWithFirstDie.filter(tile => this.getReachableTiles(tile, secondDieValue).includes(targetTile));
        const intermediateTiles2 = reachableWithSecondDie.filter(tile => this.getReachableTiles(tile, firstDieValue).includes(targetTile));

    
        // Combine the intermediate tiles
        const allIntermediateTiles = [...intermediateTiles1, ...intermediateTiles2];
    
        // Check if there's an opponent piece on any of the intermediate tiles and capture only one piece
        const captureConditionsMet = (tile) => tile && tile.pieces.some(p => p.player !== piece.player) && tile.pieces.length === 1  && tile.type !== 'save';

        // WHICH one, when the route passes more than one lone enemy. This used to
        // take the first tile the two die orders happened to yield, i.e. whatever
        // getReachableTiles returned first -- an arbitrary choice dressed up as a
        // rule. Owner's rule: prefer a NUMBERED piece (1-6, which is tied to one
        // matching goal and so costs its owner more), and among equals the higher
        // number. Scoring numbered pieces above every blank makes that one
        // comparison: blanks are 7-12 internally, so +1000 keeps 1-6 on top while
        // "higher number wins" still holds inside each class.
        const priority = (p) => (p.number <= 6 ? 1000 : 0) + p.number;
        let best = null, bestScore = -Infinity;
        for (const tile of allIntermediateTiles) {
            if (!captureConditionsMet(tile)) continue;
            const score = priority(tile.pieces[0]);
            if (score > bestScore) { bestScore = score; best = tile; }
        }
        if (best) {
            console.log('Capturing piece at intermediate tile:', best,
                        'number', best.pieces[0].number);
            this.capturePiece(best.pieces[0]);   // only ever one
        }
    }
    
    
    
    
    // Is the obligation ABSOLUTE (a captured piece of the current player's is
    // sitting on the home tile) or merely an ordering constraint (the entry from
    // the rack)? The two are not the same rule, and canSelectForMove used to
    // apply the weaker one to both.
    //
    // DERIVED, never stored. It was a field set by updateMovablePieces, and
    // movePiece edits `mustMovePieces` directly when an obligatory piece moves --
    // so the flag stayed true after the captured piece had left home, and the
    // guard in getReachableTilesByDice (whose `must` list was by then empty) gave
    // EVERY piece zero destinations. That is the "can't use my second die to do
    // anything at all" bug, and the AI hit the same thing one move earlier.
    hasCapturedOnHome() {
        const home = this.tiles && this.tiles.find(t => t.type === 'home');
        if (!home) return false;
        const color = this.turn === 'white' ? 0xffffff : 0x000000;
        // justMovedHome distinguishes a capture from a tentative entry: both put
        // a piece on home, but the latter is mid-entry and must not block the
        // rack-reordering privilege.
        return home.pieces.some(p => p.color === color && !p.justMovedHome);
    }

    updateMovablePieces() {
        this.mustMovePieces = [];

        const currentPlayerColor = this.turn === 'white' ? 0xffffff : 0x000000;
        const homeTile = this.tiles.find(tile => tile.type === 'home');
        const unenteredRack = currentPlayerColor === 0xffffff ? this.whiteUnenteredRack : this.blackUnenteredRack;

        // Check if there are pieces in the home tile (captured pieces)
        const homePieces = homeTile.pieces.filter(piece => piece.color === currentPlayerColor);
        if (homePieces.length > 0) {
            this.mustMovePieces = homePieces;
            // These used to be skipped by the early return, so the amber "must
            // move" rings were never refreshed for a capture.
            if (typeof updateMustMoveHighlights === 'function') updateMustMoveHighlights(this);
            if (typeof _updateViewportHud === 'function') _updateViewportHud();
            return; // If there are captured pieces, no other pieces may move
        }

        // Check if there's a piece in the unentered rack
        if (unenteredRack.pieces.length > 0) {
            // The amber ring means "this piece MUST move this turn", and only
            // the front piece is obliged: taking the second one first is a
            // reordering, not an alternative obligation -- the front still has
            // to enter. Selectability is a separate question (_entrantsOf).
            this.mustMovePieces = [unenteredRack.pieces[0]];
        }
        if (typeof updateMustMoveHighlights === 'function') updateMustMoveHighlights(this);
        if (typeof _updateViewportHud === 'function') _updateViewportHud();
    }

    // Obligatory-move ordering: an obligatory piece may always be selected. A
    // NON-obligatory piece may be moved first only if, after a (single-die) move,
    // a die still remains for every pending obligatory piece — so you can move a
    // free piece first, but are then locked to the obligatory one(s).
    canSelectForMove(piece) {
        if (this.mustMovePieces.length === 0 || this.mustMovePieces.includes(piece)) return true;
        // A CAPTURED piece is an absolute block: while one of yours sits on the
        // home tile nothing else may move at all, whatever the dice would allow.
        // game.py's get_valid_moves returns only captured-piece moves in that
        // state. The die-counting rule below belongs to the ENTRY obligation,
        // where moving another piece first IS legal as long as a die is left --
        // and applying it to captures let a single captured piece be ignored
        // whenever both dice were free, since (2 - 1) >= 1.
        if (this.hasCapturedOnHome()) return false;
        const unused = this.dice.filter(d => !d.used).length;
        return (unused - 1) >= this.mustMovePieces.length;
    }

    getTilesAndPieces(tiles, pieces) {
        this.tiles = tiles;
        this.pieces = pieces;  
        this.homeTile = tiles.find(tile => tile.type === 'home');
    }

    rollDice() {
        this.dice.forEach(die => die.roll());
        this.updateDiceColors();
    }

    setDiceUsed() {
        this.dice.forEach(die => die.setUsed());
    }

    updateDiceColors() {
        this.dice.forEach(die => die.updateColor(this.turn));
        if (typeof _updateHudDice === 'function') _updateHudDice();
    }

    saveOpponentPieces(tile, savedRack) {

        tile.pieces.forEach(piece => {
            piece.moveToRack(savedRack);
        });

        this.registerSave();        // no-save streak resets immediately
        this.setDiceUsed(); // Use up both dice
        this.updateMovablePieces(); // Update movable pieces
    }


    
switchTurn() {
        // The tutorial script owns the dice and the turn order: never roll, never
        // record, never hand over. The step's success poll advances instead.
        if (_tut.active) { _tutTurnEnd(); return; }
        const justFinished = this.turn;
        const playerObj = this.players.find(p => p.name === justFinished);
        const source = playerObj.isAI ? 'heuristic' : 'human';

        // No-save draw accounting happens at the real turn boundary.
        this.updateNoSaveCounter();
        // the other player's rack is a different set of enterable pieces
        if (typeof _updateViewportHud === 'function') setTimeout(_updateViewportHud, 0);

        // Human turns used to be posted to the backend here. Removed with the
        // rest of the recording chain (see the hosting audit in CLAUDE.md); the
        // local half-move bookkeeping is still cleared, since the human move
        // path fills it.
        if (!playerObj.isAI) clearMoveRecording();

        this.turn = this.turn === 'white' ? 'black' : 'white';

        // Unhighlight all pieces
        this.pieces.forEach(piece => {
            piece.isSelected = false;
            piece.isHovered = false;
            piece.updateColor();
        });

        this.unhighlightAllTiles();

        if (this.selectedPiece) {
            this.selectedPiece.isSelected = false;
            this.selectedPiece.updateColor();
            this.selectedPiece = null;
        }
        this.pieces.forEach(p => {
            if (p.justMovedHome) {
                p.returnToRack();
                p.justMovedHome = false;
            }
        });

        this.rollDice();
        this.movedOnce = false;
        this._autoEndScheduled = false;   // fresh latch each turn
        this.updateMovablePieces();
        // record each piece's position at the turn start so a piece moved with
        // both dice one-by-one must keep advancing (shortest path, no backtrack).
        this.pieces.forEach(piece => { piece.reachableTiles = null; piece._turnStartTile = piece.currentTile || null; });
        this.state = this.captureState();
        this.undoStack = [];         // fresh per-move undo history each turn
        this.applyLastPieceRule();
        this.turnStartState = getGameState(this);
        if (typeof updateMustMoveHighlights === 'function') updateMustMoveHighlights(this);
        if (typeof updateTurnStatus === 'function') updateTurnStatus(this);

        if (window.showEvals) refreshEvalReadout();

        // No-save draw display is owned by the frontend now; update directly.
        updateNoSaveDisplay();

        // If it's the agent's turn, ask the backend for its moves.
        const currentPlayerObject = this.players.find(player => player.name === this.turn);
        if (currentPlayerObject && currentPlayerObject.isAI && !window._tutorialActive && !_gameFrozen) {
            this.scene.showThinkingIcon();
            setTimeout(() => {
                getAgentMoves(getGameState(this));
            }, 1000);
        }
    }
    // ── NO-SAVE DRAW RULE ────────────────────────────────────────────────
    totalSaved() {
        return this.whiteSavedRack.pieces.length + this.blackSavedRack.pieces.length;
    }

    bothInMidgame() {
        // "past the opening" = unentered rack emptied (midgame or endgame).
        return this.players.every(p => p.getGamePhase() !== 'opening');
    }

    currentPlayerIsHuman() {
        const p = this.players.find(pl => pl.name === this.turn);
        return !!p && !p.isAI;
    }

    // Called the instant a piece is saved (own piece or an opponent block).
    // Resets the streak immediately and refreshes the display, without waiting
    // for the turn to end.
    registerSave() {
        this.noSaveTurns = 0;
        this._halfTurnsSinceRound = 0;
        this.lastTotalSaved = this.totalSaved();
        this.drawCallable = false;
        updateNoSaveDisplay();
    }

    // Called once per player-turn at the real turn boundary (switchTurn),
    // mirroring Board.update_no_save_counter in game.py.
    updateNoSaveCounter() {
        const current = this.totalSaved();
        if (current > this.lastTotalSaved) {
            this.noSaveTurns = 0;
            this._halfTurnsSinceRound = 0;
        } else if (this.bothInMidgame()) {
            this._halfTurnsSinceRound += 1;
            if (this._halfTurnsSinceRound >= 2) {
                this._halfTurnsSinceRound = 0;
                this.noSaveTurns += 1;
            }
        }
        this.lastTotalSaved = current;
        this.drawCallable = this.noSaveTurns >= NO_SAVE_TURNS_FOR_DRAW;
    }

    checkMidgame() {
        const unenteredRack = this.turn === 'white' ? this.whiteUnenteredRack : this.blackUnenteredRack;
        const player = this.turn === 'white' ? this.players[0] : this.players[1];
        if (unenteredRack.pieces.length === 0 && player.getGamePhase() === 'opening') {
            console.log('Entering midgame');
            player.setGamePhase('midgame');
        }
    }

    unhighlightAllTiles() {
        this.tiles.forEach(tile => tile.unhighlight());
        this.tiles.forEach(tile => tile.reachableColor = null);
    }

    checkWinCondition() {
        const whiteSavedAll = this.whiteSavedRack.pieces.length === TOTAL_PIECES;
        const blackSavedAll = this.blackSavedRack.pieces.length === TOTAL_PIECES;

        if (whiteSavedAll) {
            this.endGame('white');
        } else if (blackSavedAll) {
            this.endGame('black');
        }
    }

    applyLastPieceRule() {
        const player = this.turn;
        const playerObj = this.players.find(p => p.name === player);
        if (playerObj.getGamePhase() !== 'endgame') return;
        
        const savedRack = player === 'white' ? this.whiteSavedRack : this.blackSavedRack;
        const unsaved = this.pieces.filter(p => p.player === player && p.rack !== savedRack);
        
        if (unsaved.length === 1 && unsaved[0].number <= 6) {
            console.log(`Applying last piece rule to ${player}(${unsaved[0].number})`);
            unsaved[0].number = TOTAL_PIECES + 1;
            if (unsaved[0].text) {
                unsaved[0].text.destroy();
                unsaved[0].text = null;
            }
        }
    }

endGame(winner, score = null, impasse_caller = null) {
    // The tutorial ends on its own closing panel, not the end-game scene.
    if (_tut.active) return;
    // Determine the score (margin) first
    if (winner === 'tie') {
        score = 0;
    } else if (score === null) {
        score = winner === 'white'
            ? TOTAL_PIECES - this.blackSavedRack.pieces.length
            : TOTAL_PIECES - this.whiteSavedRack.pieces.length;
    }

    clearMoveRecording();

    this.gameOver = true;
    console.log(`${winner} wins with a score of ${score}!`);
    // From the human's point of view when exactly one side is human; two humans
    // (or two computers) just get the win chime.
    const humanSide = (!WHITE_IS_AI && BLACK_IS_AI) ? 'white'
                    : (WHITE_IS_AI && !BLACK_IS_AI) ? 'black' : null;
    if (winner === 'draw' || !humanSide || winner === humanSide) SFX.win(); else SFX.lose();
    if (winner === 'draw') {
        scoreTracker.draws += 1;
    } else if (winner === 'white') {
        scoreTracker.total_score += score;
        scoreTracker.white_wins += 1;
    } else if (winner === 'black') {
        scoreTracker.total_score -= score;
        scoreTracker.black_wins += 1;
    }
    scoreTracker.games_played += 1;
    if (typeof updateTurnStatus === 'function') updateTurnStatus('');   // hide during end screen
    // Fold this game into the active match (if any) before showing the result.
    const matchOver = matchTracker ? recordMatchGame(winner, score) : false;
    this.scene.updateScoreText();
    this.scene.scene.start('EndGameScene', {
        winner: winner, score: score, impasse_caller: impasse_caller,
        inMatch: !!matchTracker, matchOver: matchOver
    });
}

    captureState() {
        const state = {
            turn: this.turn,
            players: this.players.map(player => ({
                name: player.name,
                gamePhase: player.getGamePhase()
            })),
            pieces: this.pieces.map(piece => ({
                color: piece.color,
                number: piece.number,
                x: piece.x,
                y: piece.y,
                rack: piece.rack ? piece.rack.type : null,
                currentTile: piece.currentTile ? {
                    type: piece.currentTile.type,
                    ring: piece.currentTile.ring,
                    sector: piece.currentTile.sector
                } : null
            })),
            tiles: this.tiles.map(tile => ({
                type: tile.type,
                ring: tile.ring,
                sector: tile.sector,
                pieces: tile.pieces.map(p => ({
                    color: p.color,
                    number: p.number
                }))
            })),
            dice: this.dice.map(die => ({
                value: die.value,
                used: die.used
            })),
            racks: {
                whiteUnentered: this.whiteUnenteredRack.pieces.map(p => ({ color: p.color, number: p.number })),
                whiteSaved: this.whiteSavedRack.pieces.map(p => ({ color: p.color, number: p.number })),
                blackUnentered: this.blackUnenteredRack.pieces.map(p => ({ color: p.color, number: p.number })),
                blackSaved: this.blackSavedRack.pieces.map(p => ({ color: p.color, number: p.number })),
            },
            gameOver: this.gameOver,
            noSaveTurns: this.noSaveTurns,
            drawCallable: this.drawCallable,
            lastTotalSaved: this.lastTotalSaved,
            halfTurnsSinceRound: this._halfTurnsSinceRound,
            mustMove: this.mustMovePieces.map(p => ({ color: p.color, number: p.number })),
            movedOnce: this.movedOnce
        };

        return state;
    }
    

    restoreState(state = this.state) {
        if (!state) {
            console.error('No state to restore');
            return;
        }
    
        // Clear existing graphics
        this.tiles.forEach(tile => tile.graphics.clear());
        this.pieces.forEach(piece => {
            piece.body.destroy();
            piece.sheen.destroy();
            piece.circle.destroy();
            if (piece.text) {
                piece.text.destroy();
            }
        });
        this.dice.forEach(die => die.graphics.clear());
    
        this.turn = state.turn;
        this.players.forEach((player, index) => player.setGamePhase(state.players[index].gamePhase));
    
        this.tiles.forEach((tile, index) => {
            const tileState = state.tiles[index];
            if (tileState) {
                tile.type = tileState.type;
                tile.ring = tileState.ring;
                tile.sector = tileState.sector;
                tile.pieces = [];
                tile.drawTile();
            } else {
                console.warn(`Missing state for tile at index ${index}`);
            }
        });
    
        this.pieces.forEach((piece, index) => {
            const pieceState = state.pieces[index];
            if (pieceState) {
                piece.selected = false;
                piece.color = pieceState.color;
                piece.number = pieceState.number;
                // set coords directly (body/sheen/circle were just destroyed
                // above; drawPiece below recreates them at this position).
                piece.x = pieceState.x;
                piece.y = pieceState.y;
                piece.rack = pieceState.rack ? (pieceState.rack === 'unentered' ? (piece.color === 0xffffff ? this.whiteUnenteredRack : this.blackUnenteredRack) : (piece.color === 0xffffff ? this.whiteSavedRack : this.blackSavedRack)) : null;
                piece.currentTile = pieceState.currentTile ? this.tiles.find(tile => tile.ring === pieceState.currentTile.ring && tile.sector === pieceState.currentTile.sector) : null;
                piece.drawPiece();
            } else {
                console.warn(`Missing state for piece at index ${index}`);
            }
        });
    
        this.tiles.forEach((tile, index) => {
            const tileState = state.tiles[index];
            if (tileState) {
                tile.pieces = tileState.pieces.map(p => this.pieces.find(piece => piece.color === p.color && piece.number === p.number));
                tile.updatePositions();
            } else {
                console.warn(`Missing state for tile pieces at index ${index}`);
            }
        });
    
        this.dice.forEach((die, index) => {
            const dieState = state.dice[index];
            if (dieState) {
                die.value = dieState.value;
                die.used = dieState.used;
                die.drawDie();
            } else {
                console.warn(`Missing state for die at index ${index}`);
            }
        });

        const restoreRack = (rack, pieces) => {
            rack.pieces = pieces.map(pState => {
                const piece = this.pieces.find(piece => piece.color === pState.color && piece.number === pState.number);
                piece.rack = rack;
                return piece;
            });
            rack.shiftPiecesUp(); // Adjust positions after restoring
        };
    
        restoreRack(this.whiteUnenteredRack, state.racks.whiteUnentered);
        restoreRack(this.whiteSavedRack, state.racks.whiteSaved);
        restoreRack(this.blackUnenteredRack, state.racks.blackUnentered);
        restoreRack(this.blackSavedRack, state.racks.blackSaved);
    
    
        this.gameOver = state.gameOver;
        // Restore no-save draw counter (undo reverts the whole turn, so this
        // puts the streak back to whatever it was before the undone turn).
        if (state.noSaveTurns !== undefined) {
            this.noSaveTurns = state.noSaveTurns;
            this.drawCallable = state.drawCallable;
            this.lastTotalSaved = state.lastTotalSaved;
            this._halfTurnsSinceRound = state.halfTurnsSinceRound;
        }
        // restore the obligatory-move list and the moved-this-turn flag
        if (state.mustMove !== undefined) {
            this.mustMovePieces = state.mustMove.map(m =>
                this.pieces.find(p => p.color === m.color && p.number === m.number)).filter(Boolean);
        }
        if (state.movedOnce !== undefined) this.movedOnce = state.movedOnce;
        this.pieces.forEach(piece => piece.reachableTiles = null);
        this.updateDiceColors();
        this.unhighlightAllTiles();
        this.hideOuterNogoTiles()
        this.selectedPiece = null;
        console.log('Game state restored.');
        clearMoveRecording();
        refreshEvalReadout();  // update on-board eval after undo
        updateNoSaveDisplay(); // reflect restored no-save streak
    }

    // Undo one move at a time (one die), not the whole turn. Each committed move
    // pushes its pre-move snapshot; undo pops and restores the most recent one.
    undoOneMove() {
        // A piece that has only been PICKED UP -- tentatively entered onto the
        // home tile, no die spent -- is not a move yet and is not on the undo
        // stack. Popping the stack would therefore revert the previous REAL
        // move and drop this piece back along with it. Put it back and stop.
        const colour = this.turn === 'white' ? 0xffffff : 0x000000;
        const pending = this.pieces.find(p => p.justMovedHome && p.color === colour
                                              && p.currentTile && p.currentTile.type === 'home');
        if (pending) {
            pending.returnToRack();
            this._pendingPreMove = null;
            if (typeof updateMustMoveHighlights === 'function') updateMustMoveHighlights(this);
            if (typeof _updateViewportHud === 'function') _updateViewportHud();
            return;
        }
        if (this.undoStack && this.undoStack.length > 0) {
            this.restoreState(this.undoStack.pop());
        } else {
            this.restoreState();   // already at turn start -> revert to it (no-op-ish)
        }
        if (typeof updateMustMoveHighlights === 'function') updateMustMoveHighlights(this);
    }

    // Record the current state as an undo point (call BEFORE a move mutates).
    pushUndo() {
        if (!this.undoStack) this.undoStack = [];
        this.undoStack.push(this.captureState());
    }
    
    
    // Return the (non-nogo) tile whose annular wedge contains board point (x,y).
    tileAtPoint(x, y) {
        const dx = x - CENTER_X, dy = y - CENTER_Y;
        const r = Math.hypot(dx, dy);
        let a = Math.atan2(dy, dx); if (a < 0) a += 2 * Math.PI;
        for (const t of this.tiles) {
            if (t.type === 'nogo') continue;
            if (t.type === 'home') { if (r <= t.outerRadius) return t; continue; }
            if (r >= t.innerRadius && r <= t.outerRadius && a >= t.startAngle && a <= t.endAngle) return t;
        }
        return null;
    }

    // Return the rack whose panel contains point (x,y), or null (used by setup drag).
    rackAtPoint(x, y) {
        const racks = [this.whiteUnenteredRack, this.whiteSavedRack, this.blackUnenteredRack, this.blackSavedRack];
        for (const r of racks) {
            const bx = r.x - r.pr, by = r.y - r.pr;
            const bw = r.cols * r.spacing + r.pr, bh = r.rows * r.spacing + r.pr + r.verticalPadding;
            if (x >= bx && x <= bx + bw && y >= by && y <= by + bh) return r;
        }
        return null;
    }

    // Drag-to-move, additive with click. A piece is selected by the pointerdown
    // (its normal handleClick) before dragstart fires, so drag just moves the
    // already-selected piece and drops it on the tile under the pointer, exactly
    // as if that tile had been clicked. Invalid drops snap the piece back.
    // ── CAMERA PAN AND ZOOM (phones) ────────────────────────────────────
    // Panning and zooming happen INSIDE Phaser, on the camera, never by handing
    // gestures to the browser. That earlier attempt let Chrome claim one-finger
    // drags while zoomed, and it took the taps with them -- pieces stopped being
    // selectable. Here Phaser knows exactly what is under the finger, so a drag
    // that starts on a piece drags the piece and anything else pans; a pinch is
    // two pointers and can never be confused with a tap. `?cam=0` disables it.
    setupCameraControls(scene) {
        if (!_isPhone()) return;
        try {
            if (new URLSearchParams(location.search).get('cam') === '0') return;
        } catch (e) {}
        if (scene._camWired) return;
        scene._camWired = true;
        _sizeCanvasToScreen();
        _sizeGear();
        _fitCameraToWorld(scene);
        // Re-apply once the browser has settled: on a phone the viewport is
        // still moving at this point (URL bar, fullscreen), and a game built on
        // a stale orientation reading would otherwise keep the wrong layout.
        setTimeout(() => {
            _relayoutFurniture();
            _fitCameraToWorld(scene);
            // The pill measures the canvas rect, which is still stale mid-rotation
            // -- it stretched into a wide bar across the top when it was not.
            if (_replaceTurnStatus) _replaceTurnStatus();
        }, 400);
        // Two distinct jobs, deliberately not the same handler: the WINDOW
        // changing means re-measure the screen and resize the buffer; the SCALE
        // resizing (which our own resize triggers) only means re-frame.
        const onFrame = () => { _fitCameraToWorld(scene); _updateViewportHud(); };
        const onScreenChange = () => { _sizeCanvasToScreen(); _relayoutFurniture(); onFrame(); };
        scene.scale.on('resize', onFrame);
        window.addEventListener('resize', onScreenChange);
        const onOrient = () => setTimeout(onScreenChange, 250);
        window.addEventListener('orientationchange', onOrient);

        const cam = scene.cameras.main;
        const PAN_SLOP = 8, MAX_FACTOR = 4;
        scene.input.addPointer(2);                 // enough pointers for a pinch
        // The browser must not also zoom/scroll, or the two transforms compose.
        if (gameInstance.canvas) gameInstance.canvas.style.touchAction = 'none';

        // Zoom runs from "the whole world fits" up to 4x that. The visible world
        // rectangle is kept inside the world where it is smaller, and centred on
        // whichever axis has slack (a phone screen is never the world's shape).
        const clamp = () => {
            const base = scene._camBase || _baseZoom(scene);
            cam.zoom = Phaser.Math.Clamp(cam.zoom, base, base * MAX_FACTOR);
            scene._camUserZoom = cam.zoom / base;
            const vw = cam.width / cam.zoom, vh = cam.height / cam.zoom;
            // Derive the intended view from the CURRENT scroll, not from
            // cam.worldView: worldView is only recomputed at render, so reading
            // it here writes back the previous frame's position and silently
            // undoes the pan that just happened.
            let left = cam.scrollX + (cam.width - vw) / 2;
            let top  = cam.scrollY + (cam.height - vh) / 2;
            const wd = _world();
            left = vw >= wd.w ? wd.x + (wd.w - vw) / 2
                              : Phaser.Math.Clamp(left, wd.x, wd.x + wd.w - vw);
            top  = vh >= wd.h ? wd.y + (wd.h - vh) / 2
                              : Phaser.Math.Clamp(top, wd.y, wd.y + wd.h - vh);
            _setCameraView(cam, left, top);
        };

        let panFrom = null, panning = false, pinch = null;

        const pointers = () => [scene.input.pointer1, scene.input.pointer2]
            .filter(p => p && p.isDown);

        scene.input.on('pointerdown', (pointer, currentlyOver) => {
            const down = pointers();
            if (down.length >= 2) {                 // two fingers: zoom AND pan
                panning = false; panFrom = null;
                const [a, b] = down;
                const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
                const left = cam.scrollX + (cam.width - cam.width / cam.zoom) / 2;
                const top = cam.scrollY + (cam.height - cam.height / cam.zoom) / 2;
                pinch = {
                    dist: Phaser.Math.Distance.Between(a.x, a.y, b.x, b.y),
                    zoom: cam.zoom,
                    // the world point under the midpoint when the pinch began:
                    // keeping THIS under the moving midpoint gives both the zoom
                    // anchor and two-finger panning in one step
                    world: { x: left + mid.x / cam.zoom, y: top + mid.y / cam.zoom },
                };
                return;
            }
            // A drag that starts on anything draggable belongs to that thing, not
            // to the camera: pieces (__piece) AND the must-enter ghosts
            // (__ghost). Missing the ghosts meant the board panned out from
            // under a ghost drag, which reads as the drag simply not working.
            // Use the list Phaser hands us -- calling hitTestPointer() from
            // inside a pointer handler re-runs hit testing mid-update and
            // clobbers the drag state Phaser is setting up, which killed piece
            // dragging on a phone entirely.
            if ((currentlyOver || []).some(o => o && (o.__piece || o.__ghost))) { panFrom = null; return; }
            panFrom = { x: pointer.x, y: pointer.y, sx: cam.scrollX, sy: cam.scrollY };
        });

        scene.input.on('pointermove', (pointer) => {
            const down = pointers();
            if (pinch && down.length >= 2) {
                const [a, b] = down;
                const d = Phaser.Math.Distance.Between(a.x, a.y, b.x, b.y);
                if (pinch.dist > 0) {
                    const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
                    const base = scene._camBase || _baseZoom(scene);
                    // Dead zone on the finger separation. Two fingers dragged
                    // together still wobble a few pixels apart, and feeding that
                    // straight into the zoom made a pan shimmer in and out.
                    const ratio = d / pinch.dist;
                    if (Math.abs(ratio - 1) > 0.06) {
                        cam.zoom = Phaser.Math.Clamp(pinch.zoom * ratio, base, base * MAX_FACTOR);
                    }
                    // put the pinch's starting world point back under the CURRENT
                    // midpoint: that anchors the zoom and pans with the fingers.
                    _setCameraView(cam, pinch.world.x - mid.x / cam.zoom,
                                        pinch.world.y - mid.y / cam.zoom);
                    clamp();
                    _updateViewportHud();
                    // Zooming in past the baked scale would magnify the texture;
                    // debounced, so this re-bakes once the pinch settles rather
                    // than on every frame of the gesture.
                    _scheduleRebake(scene);
                }
                return;
            }
            if (!panFrom || !pointer.isDown || scene._draggingPiece || scene._draggingGhost) return;
            const dx = pointer.x - panFrom.x, dy = pointer.y - panFrom.y;
            if (!panning && Math.hypot(dx, dy) < PAN_SLOP) return;   // still might be a tap
            panning = true;
            cam.scrollX = panFrom.sx - dx / cam.zoom;
            cam.scrollY = panFrom.sy - dy / cam.zoom;
            clamp();
            _updateViewportHud();
        });

        // Dragging a piece to the edge of the view scrolls the board, so a
        // destination that is off screen can be reached without letting go.
        const edgePan = () => {
            const piece = scene._draggingPiece;
            if (!piece || cam.zoom <= (scene._camBase || 1) * 1.02) return;
            const p = scene.input.activePointer;
            if (!p || !p.isDown) return;
            const v = cam.worldView;
            const band = Math.min(v.width, v.height) * 0.14;
            const dx = p.worldX < v.x + band ? -1 : p.worldX > v.right - band ? 1 : 0;
            const dy = p.worldY < v.y + band ? -1 : p.worldY > v.bottom - band ? 1 : 0;
            if (!dx && !dy) return;
            const speed = 20 / cam.zoom;          // world px per frame
            cam.scrollX += dx * speed;
            cam.scrollY += dy * speed;
            clamp();
            _updateViewportHud();
        };
        scene.events.on('update', edgePan);

        const release = () => {
            if (pointers().length === 0) { pinch = null; panning = false; panFrom = null; }
        };
        scene.input.on('pointerup', release);
        scene.input.on('pointerupoutside', release);
        scene.events.once('shutdown', () => {
            scene._camWired = false;
            scene.events.off('update', edgePan);
            scene.scale.off('resize', onFrame);
            window.removeEventListener('resize', onScreenChange);
            window.removeEventListener('orientationchange', onOrient);
        });
    }

    setupDragging(scene) {
        // NB: Phaser clears scene.input listeners on shutdown/restart, so re-wire
        // every time create() runs. The guard is reset on 'shutdown' (below) so a
        // New Game (scene.restart) or end-game (scene.start) keeps pieces draggable.
        if (scene._dragWired) return;
        scene._dragWired = true;
        // The SAME slop onTap uses, so the two cannot disagree and leave a range
        // that is neither tap nor drag (see _tapSlop). In buffer pixels, which
        // is what Phaser compares against.
        scene.input.dragDistanceThreshold = _isPhone() ? _tapSlop() : 6;

        const onDragStart = (pointer, obj) => {
            if (obj.__ghost) {
                scene._draggingGhost = obj.__ghost;
                // Enter the piece NOW, not on drop: entering is what selects it
                // and lights up its destinations, so doing it at the end left
                // the whole drag with no highlights to aim at.
                const gp = obj.__ghost.piece;
                if (gp && gp.rack) gp.handleClick({ rightButtonDown: () => false });
                return;
            }
            const piece = obj.__piece; if (!piece) return;
            // a press that became a drag is not a click, so it must not send a
            // just-entered piece back to the rack when the finger lifts
            piece._draggedSincePress = true;
            piece._pendingReturn = false;
            hideStackPicker();
            piece._originTile = piece.currentTile;
            piece._originRack = piece.rack;
            if (window.setupMode) {
                // free placement: pointerdown already set _setupSelected via handleClick
                piece._dragOK = (_setupSelected === piece);
                piece._snapRack = false;
            } else {
                // draggable only if the pointerdown actually selected this piece
                piece._dragOK = (piece.game.selectedPiece === piece);
                piece._snapRack = !!(piece.justMovedHome && piece.currentTile && piece.currentTile.type === 'home');
            }
            scene._draggingPiece = piece._dragOK ? piece : null;
            // snap to the cursor immediately so an entering piece doesn't flash on
            // the home tile (where handleClick placed it) before the first drag move.
            if (piece._dragOK) piece.setPosition(pointer.worldX, pointer.worldY);
        };

        const onDrag = (pointer, obj) => {
            if (obj.__ghost) { obj.__ghost.ghost.setPosition(pointer.worldX, pointer.worldY); return; }
            const piece = obj.__piece; if (!piece || !piece._dragOK) return;
            // Centre the piece on the pointer. (dragX/dragY bake in the grab
            // offset from where the piece sat at dragstart — but rack pieces
            // teleport to home on pickup, making that offset huge, so the piece
            // would float far from the cursor. Following the pointer avoids that.)
            piece.setPosition(pointer.worldX, pointer.worldY);
        };

        // Also glue the dragged piece to the pointer each frame (after input is
        // processed, just before render) so it doesn't lag a frame behind on
        // fast moves — the batched 'drag' event alone feels laggy.
        const onUpdate = () => {
            const dp = scene._draggingPiece;
            if (dp && dp._dragOK) {
                const p = scene.input.activePointer;
                dp.setPosition(p.worldX, p.worldY);
            }
        };

        const onDragEnd = (pointer, obj) => {
            if (obj.__ghost) {
                // Enter the piece it stands for, then treat the drop as a tile
                // tap -- the same near-miss resolution as any other drop.
                const { piece: ghostPiece } = obj.__ghost;
                scene._draggingGhost = null;
                // normally entered at dragstart; this covers a drag that somehow
                // began without it
                if (ghostPiece.rack) ghostPiece.handleClick({ rightButtonDown: () => false });
                const t = ghostPiece.game.tileAtPoint(pointer.worldX, pointer.worldY);
                const drop = _resolveDestination(ghostPiece.game, t, pointer.worldX, pointer.worldY);
                if (drop) drop.onClick();
                _updateViewportHud();
                return;
            }
            const piece = obj.__piece; if (!piece || !piece._dragOK) return;
            piece._dragOK = false;
            scene._draggingPiece = null;
            const before = piece._originTile;
            const target = piece.game.tileAtPoint(pointer.worldX, pointer.worldY);

            if (window.setupMode) {
                // free placement: drop on a rack panel puts the piece in that rack
                // (to/from/between racks); drop on a tile places it there; an
                // off-board drop snaps it back to where it came from.
                const rack = piece.game.rackAtPoint(pointer.worldX, pointer.worldY);
                if (rack) {
                    _setupPlaceInRack(piece, rack, false);
                    _setupClearSelection();
                } else if (target && target.type !== 'nogo') {
                    target.onClick();
                } else {
                    if (piece._originTile) piece._originTile.updatePositions();
                    else if (piece._originRack) piece._originRack.shiftPiecesUp();
                    _setupClearSelection();
                }
                return;
            }

            // Dropping a savable piece on your own saved rack saves it (same as
            // the double-click gesture).
            const rackDrop = piece.game.rackAtPoint(pointer.worldX, pointer.worldY);
            const mySaved = piece.player === 'white' ? piece.game.whiteSavedRack : piece.game.blackSavedRack;
            if (rackDrop === mySaved) {
                // already on its goal -> save; otherwise try a reach-goal-and-save
                // (sum-save) in one drop.
                if (piece.canBeSaved && piece.canBeSaved() && piece.save()) return;
                if (piece.game.sumSave(piece)) return;
            }

            const drop = _resolveDestination(piece.game, target, pointer.worldX, pointer.worldY);
            if (drop) drop.onClick();              // moves the selected piece, with full rule checks
            if (piece.currentTile === before) {     // move didn't happen -> snap back + deselect
                if (piece._snapRack) piece.returnToRack();   // returns to rack (also deselects)
                else {
                    if (piece.currentTile) piece.currentTile.updatePositions();
                    // a piece dragged out of a rack has no tile to snap back to;
                    // without this it just stays wherever it was dropped
                    else if (piece.rack) piece.rack.shiftPiecesUp();
                    _clearSelection(piece.game);             // dropping cancels the selection
                }
            }
        };

        scene.input.on('dragstart', onDragStart);
        scene.input.on('drag', onDrag);
        scene.input.on('dragend', onDragEnd);
        scene.events.on('update', onUpdate);

        // On scene shutdown/restart, drop the guard + the update listener so the
        // next create() re-wires cleanly (input listeners are auto-removed).
        scene.events.once('shutdown', () => {
            scene._dragWired = false;
            scene._draggingPiece = null;
            scene.events.off('update', onUpdate);
        });
    }

    createUndoButton(scene) {
        // Phones get bigger arrows, further apart: at 64 world px they are ~21
        // CSS px with only 36px of world gap, which is easy to mis-hit.
        const buttonSize = _isPhone() ? 110 : 64;
        this.undoButton = scene.add.image(_fur().undoX, _fur().arrowY, 'leftWavyArrow')
            .setDisplaySize(buttonSize, buttonSize)
            .setInteractive();
        onTap(this.undoButton, () => {
            hideStackPicker();
            this.undoOneMove();   // one die / one move at a time
            clearMoveRecording();
        });

        const undoTooltip = makeHudTip(scene, this.undoButton.x, this.undoButton.y + buttonSize * 0.72, 'Undo');
        this.undoButton.on('pointerover', () => undoTooltip.show(true));
        this.undoButton.on('pointerout',  () => undoTooltip.show(false));
    }

    createSwitchTurnButton(scene) {
        const buttonSize = _isPhone() ? 110 : 64;
        this.switchTurnButton = scene.add.image(_fur().endX, _fur().arrowY, 'rightWavyArrow')
            .setDisplaySize(buttonSize, buttonSize)
            .setInteractive();
        onTap(this.switchTurnButton, () => {
                // Only the human whose turn it is may end the turn.
                if (this.gameOver || !this.currentPlayerIsHuman()) return;
                // Confirm only if the setting is on AND ending is actually risky
                // (a die is unused and a legal move still exists).
                if (this.dice.some(die => !die.used) && getConfirmRiskyEnd() && this.hasAnyLegalMove()) {
                    this.showConfirmationModal();
                } else {
                    this.switchTurn();
                }
            });

        const switchTurnTooltip = makeHudTip(scene, this.switchTurnButton.x,
                                             this.switchTurnButton.y + buttonSize * 0.72, 'End turn');
        this.switchTurnButton.on('pointerover', () => switchTurnTooltip.show(true));
        this.switchTurnButton.on('pointerout',  () => switchTurnTooltip.show(false));
    }
    
    // Ending a turn with a die still live: the shared DOM dialog (was a
    // hand-drawn Phaser rectangle with bright green/red Yes/No buttons).
    showConfirmationModal() {
        showConfirm('End your turn without using both dice?',
            () => this.switchTurn(), 'End turn');
    }

    hideConfirmationModal() {
        const dlg = document.getElementById('confirmDlg'); if (dlg) dlg.remove();
    }

    saveTileNeighborsToFile() {
        const tileNeighbors = {};

        this.tiles.forEach(tile => {
            if (tile.type !== 'nogo') {
                const key = `ring${tile.ring}_sector${tile.sector}`;
                tileNeighbors[key] = {
                    type: tile.type,
                    neighbors: tile.neighbors.map(neighbor => ({
                        ring: neighbor.ring,
                        sector: neighbor.sector
                    }))
                };
                if (tile.type === 'save') {
                    tileNeighbors[key].number = tile.number;
                }
            }
        });

        const json = JSON.stringify(tileNeighbors, null, 2);
        this.saveJSONToFile(json, 'tile_neighbors.json');
    }

    saveJSONToFile(json, filename) {
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    }
    

}

class Player {
    constructor(name, isAI = false) {
        this.name = name;
        this.isAI = isAI;
        this.gamePhase = 'opening'; // Initialize the game phase
    }

    // Method to set the game phase
    setGamePhase(phase) {
        this.gamePhase = phase;
        console.log(`${this.name}'s game phase set to: ${phase}`);
    }

    // Method to get the game phase
    getGamePhase() {
        return this.gamePhase;
    }
}

class MainGameScene extends Phaser.Scene {
    constructor() {
        super({ key: 'MainGameScene' });
        this.game = null;
        this.scoreText = null; 
        this.startingPlayer = 'white';
    }

    init(data) {
        if (data && data.welcome) {
            // back from the tutorial: same as a fresh page load — hold the game
            // behind the start screen until the player picks something.
            this.startingPlayer = Math.random() < 0.5 ? 'white' : 'black';
            this._coinFlipOnStart = true;
        } else if (data && data.startingPlayer) {
            this.startingPlayer = data.startingPlayer;
        } else {
            // initial page-load casual game: random first player, revealed by a coin flip
            this.startingPlayer = Math.random() < 0.5 ? 'white' : 'black';
            this._coinFlipOnStart = true;
        }
        _lastGameStarter = this.startingPlayer;
    }

    preload() {
        this.load.image('leftWavyArrow', 'assets/left-arrow.png');
        this.load.image('rightWavyArrow', 'assets/right-arrow.png');
        this.load.image('thinkingIcon', 'assets/thinking.png'); 

    }

    create() {
        const debugMode = false; // Set this to false to disable debug mode

        _themedRedraws = [];   // fresh registry each scene build (drop stale button refs)
        // On the first-load welcome, hold the game (no AI) until the player hits
        // Play; a real, freshly-rolled game is started then.
        _gameFrozen = !!(this._coinFlipOnStart && !matchTracker);
        this.game = new Game(this, this.startingPlayer, debugMode);
        // A restart destroys every game object, but this.game keeps pointing at
        // the old Game until create() runs again -- so anything deferred (the
        // agent's move animation) needs a positive "this game is over" mark, not
        // just an identity check. See stillCurrent() in the agent-move code.
        this.events.once('shutdown', (g => () => { g.isDefunct = true; })(this.game));
        // the ghosts are scene objects; a restart destroys them, so drop the pool
        this.events.once('shutdown', () => { _ghosts = []; _hudDice = null; });
        _ghosts = []; _hudDice = null;

        // Who plays each colour now lives in the settings panel; reflect the
        // persisted choice. checkInitialAIReady below starts the first move, so
        // don't also trigger it here.
        applyPlayerRoles(false);
        this.createEvalButton();

        const iconSize = 192;
        const xPosition = this.sys.game.config.width - iconSize / 2 - 100; 
        const yPosition = this.sys.game.config.height - iconSize / 2 - 100; 
        this.thinkingIcon = this.add.image(xPosition, yPosition, 'thinkingIcon')
        .setDisplaySize(iconSize, iconSize) 
        .setAlpha(0)
        .setVisible(true);


        // Add new game button (accent) + instructions button (ghost)
        const inMatch = !!(matchTracker && !matchTracker.over);

        // New Game is unavailable while a match is in progress.
        // The three HUD buttons are 19px of world font -- ~6 CSS px on a phone.
        // They get scaled and re-spaced there, into the corner box bounded by
        // the dice (x>=309) and the top of the racks (y>=297).
        const hudK = _hudK();
        const _hf = _fur();
        const hx = (n) => _isPortrait() ? _hf.hudX[n] : 150;
        const hy = (n) => _isPortrait() ? _hf.hudY
                                        : (_isPhone() ? 48 + n * 84 : 52 + n * 52);
        const newGameButton = makeHudButton(this, hx(0), hy(0), 'New Game', { ghost: true, k: hudK });
        onTap(newGameButton, () => {
            if (matchTracker && !matchTracker.over) return;
            this.showNewGameConfirmationModal();
        });
        if (inMatch) newGameButton.setHudVisible(false);

        // New Match sits where New Game would be during a match; starting one
        // mid-match asks for confirmation first.
        const newMatchButton = makeHudButton(this, hx(inMatch ? 0 : 1), hy(inMatch ? 0 : 1), 'New Match', { ghost: true, k: hudK });
        onTap(newMatchButton, () => {
            if (matchTracker && !matchTracker.over) {
                showConfirm('Abandon the current match and start a new one?', () => showMatchSetup());
            } else {
                showMatchSetup();
            }
        });

        const instructionsButton = makeHudButton(this, hx(inMatch ? 1 : 2), hy(inMatch ? 1 : 2), 'How to Play', { ghost: true, k: hudK });
        onTap(instructionsButton, () => { showInstructions(); });
        this._hudRow = [newGameButton, newMatchButton, instructionsButton];
        _layoutHudRow(this);
        // The tutorial hides these: New Game / New Match restart the scene, which
        // would leave the step runner talking to a board that no longer exists.
        this.hudButtons = [newGameButton, newMatchButton, instructionsButton];
        if (_tut.active) _tutHudVisible(false);
        if (typeof refreshSettingsMatchState === 'function') refreshSettingsMatchState();

        // Add save game state button
        if(DEBUG_MODE) {
        const saveGameStateButton = this.add.text(320, 104, 'Save Game', {
            fontSize: '24px',
            fontFamily: HUD_FONT,
            backgroundColor: '#87CEEB',
            padding: { x: 15, y: 7.5 },
            borderColor: '#000',
            borderWidth: 1.5,
            borderRadius: 3.75
        }).setOrigin(0.5).setInteractive();

        saveGameStateButton.on('pointerdown', () => {
            this.saveGameState(gameInstance.scene.scenes[0].game);
        });
    }

        // Add score display text box
        // Single-line counters directly on the background (no box), bottom-left.
        // Bottom-left status stack: score line, the no-save counter above it, and
        // the Call-draw button above that. All of it is world-space text, so on a
        // phone it renders at ~6.5 CSS px; k scales the stack and re-spaces it so
        // the three keep clear of each other. The corner is empty background
        // (the board is a circle), so there is room to grow into.
        const phone = _isPhone();
        const k = _scoreK();
        const H = this.sys.game.config.height;
        // Goal 2's arc starts at x=630 and dips to y=1140, so the enlarged score
        // line has to wrap rather than run underneath it. Origin (0,1) means it
        // grows upward from the bottom, so wrapping needs no repositioning; the
        // two lines above it are spaced for the 2-line worst case.
        this._scoreBaseFs = Math.round(20 * k);
        const scoreStyle = {
            fontSize: this._scoreBaseFs + 'px',
            fontFamily: HUD_FONT,
            color: THEME.bgInk
        };
        // NB: an explicitly undefined `wordWrap` is not the same as omitting it --
        // Phaser's GetValue treats the key as present and dereferences it, which
        // throws inside create() and leaves everything after this unbuilt. The
        // phone's line break is inserted into the text instead (see the setText
        // that assembles it), so no wrap width is needed at all.
        const _pf = _fur();
        this.scoreText = _isPortrait()
            ? this.add.text(_pf.scoreAt[0], _pf.scoreAt[1], '', Object.assign({}, scoreStyle, { align: 'center' }))
                  .setOrigin(_pf.scoreOrigin[0], _pf.scoreOrigin[1])
            : this.add.text(24, H - 24, '', scoreStyle).setOrigin(0, 1);
        _themedRedraws.push(() => this.scoreText.setColor(THEME.bgInk));

        this.updateScoreText();

        // No-save counter: a quiet HUD line (not a boxed red warning), with the
        // draw offer as a standard ghost pill underneath it when it applies.
        this.impasseText = this.add.text(
            _isPortrait() ? _pf.impasseAt[0] : 24,
            _isPortrait() ? _pf.impasseAt[1] : (phone ? H - 148 : H - 58), '', {
            fontSize: Math.round(21 * k) + 'px', fontFamily: HUD_FONT, color: THEME.bgInk
        }).setOrigin(_isPortrait() ? 0.5 : 0, _isPortrait() ? 0 : 1).setVisible(false).setAlpha(0.75);
        _themedRedraws.push(() => this.impasseText.setColor(THEME.bgInk));

        this.callDrawButton = makeHudButton(this,
            _isPortrait() ? _pf.callDrawAt[0] : (phone ? 190 : 85),
            _isPortrait() ? _pf.callDrawAt[1] : (phone ? H - 247 : H - 115),
            'Call draw', { ghost: true, k: _isPortrait() ? 2.4 : k });
        this.callDrawButton.setHudVisible(false);

            onTap(this.callDrawButton, () => {
                // The POST to /call_draw was fire-and-forget and the server did
                // nothing the client needs; the draw is ended right here.
                const g = gameInstance.scene.scenes[0].game;
                g.endGame('draw', null, g.turn);
            });

        this.checkInitialAIReady();


        // First casual game of a session: greet with a start screen, then the
        // coin flip (on Play) reveals the random starter.
        if (this._coinFlipOnStart && !matchTracker) {
            this._coinFlipOnStart = false;
            showWelcome(this.startingPlayer);
        }
        if (typeof updateTurnStatus === 'function') updateTurnStatus(this.game);

        // Bake LAST: every tile has its hit-area chrome built by now (that is
        // done on a tile's first full draw), and the camera has been framed, so
        // the scale the texture needs is known.
        _bakeBoard(this);
        _installFpsTest(this);
        this.events.once('shutdown', () => {
            if (this._rebakeTimer) { clearTimeout(this._rebakeTimer); this._rebakeTimer = null; }
            this._boardRT = null;
        });
    }

    // Keep the score row clear of goal 2's arc (x=630) whatever the numbers do.
    // Normal scores never trigger this; three- and four-digit ones shrink a
    // little rather than running under the board. Multi-line text reports the
    // widest line, which is exactly the constraint.
    _fitScoreText() {
        if (!_isPhone() || !this.scoreText) return;
        // Landscape: 630 (goal 2's arc) minus the 24px left margin. Portrait puts
        // the score in a clear band, so only the frame constrains it.
        const maxW = _isPortrait() ? _world().w - 140 : 582;
        this.scoreText.setFontSize(this._scoreBaseFs);
        if (this.scoreText.width > maxW) {
            const shrunk = Math.floor(this._scoreBaseFs * maxW / this.scoreText.width);
            this.scoreText.setFontSize(Math.max(30, shrunk));
        }
    }

    updateScoreText() {
            // During a match the line shows that match's running score/wins;
            // otherwise the session totals.
            const matchLine = matchScoreLine();
            if (matchLine) { this.scoreText.setText(matchLine); this._fitScoreText(); return; }

            // Single line, interpunct-separated, directly on the background.
            // Total score is signed (+ favours White), shown with a leader label.
            const total = scoreTracker.total_score;
            const totalStr = total === 0 ? '0'
                : `${total > 0 ? 'White' : 'Black'} +${Math.abs(total)}`;
            const sep = '  \u00B7  ';
            const parts = [`Games ${scoreTracker.games_played}`,
                           `White ${scoreTracker.white_wins}`,
                           `Black ${scoreTracker.black_wins}`,
                           `Draws ${scoreTracker.draws}`,
                           `Total score ${totalStr}`];
            // A phone breaks this in two deliberately -- games/white/black, then
            // draws/total. Left to wordWrap it split mid-item ("Black" / "0").
            this.scoreText.setText(_isPhone()
                ? parts.slice(0, 3).join(sep) + '\n' + parts.slice(3).join(sep)
                : parts.join(sep));
            this._fitScoreText();
        }

    createEvalButton() {
        const circleX = this.sys.game.config.width - 450;
        const circleY = this.sys.game.config.height - 100;

        const circle = this.add.circle(circleX, circleY, 15, 0xff6600)
            .setInteractive()
            .setVisible(false)
            .on('pointerdown', () => {
                evaluateBoard(getGameState(this.game)).then(data => {
                    if (data) this.showEvalPanel(data);
                });
            });

        const label = this.add.text(circleX + 25, circleY, 'Evaluate Position', {
            fontSize: '22px',
            fontFamily: HUD_FONT,
            color: '#cc4400'
        }).setOrigin(0, 0.5).setVisible(false);

        this._debugEvalButton = circle;
        this._debugEvalLabel  = label;
    }


        showEvalPanel(data) {
        if (this._evalPanel) this._evalPanel.remove();

        const panel = document.createElement('div');
        panel.style.cssText = `
            position:fixed; top:10px; right:10px; width:480px; max-height:90vh;
            overflow-y:auto; background:rgba(15,15,15,0.93); color:#e8e8e8;
            font-family:'Courier New',monospace; font-size:16px; line-height:1.6;
            border:1px solid #555; border-radius:6px; padding:14px 18px;
            z-index:9999; box-shadow:0 4px 18px rgba(0,0,0,0.6);
        `;

        // Close button
        const close = document.createElement('button');
        close.textContent = 'X';
        close.style.cssText = 'float:right;background:none;border:none;color:#aaa;font-size:16px;cursor:pointer;';
        close.onclick = () => panel.remove();
        panel.appendChild(close);

        // Title
        const title = document.createElement('div');
        title.textContent = 'Position Evaluation';
        title.style.cssText = 'font-weight:bold;font-size:15px;margin-bottom:6px;color:#ffcc55;';
        panel.appendChild(title);

        // GNN predicted value, phrased as "<Player> <raw>" e.g. "White 0.50".
        // The server agent IS the GNN, so total_score is the scaled value
        // (raw * SCORE_SCALE); gnn_raw is the model's direct output and
        // gnn_player is the side-to-move perspective the value is from.
        const gnnLabel = data.gnn_player
            ? data.gnn_player.charAt(0).toUpperCase() + data.gnn_player.slice(1)
            : null;
        if (data.gnn_raw !== undefined && data.gnn_raw !== null && gnnLabel) {
            const g = document.createElement('div');
            const margin = _fmtAhead(data.gnn_player, data.gnn_raw * TOTAL_PIECES, 1);
            g.textContent = `GNN expected margin: ${margin}  (raw ${Number(data.gnn_raw).toFixed(2)})`;
            g.style.cssText = 'color:#66ccff;margin-bottom:10px;font-weight:bold;font-size:17px;';
            panel.appendChild(g);
        } else if (data.total_score !== undefined || data.eval !== undefined) {
            // Fallback if the backend hasn't been updated to send gnn_raw yet.
            const tot = document.createElement('div');
            tot.textContent = `GNN value (scaled): ${data.total_score ?? data.eval}`;
            tot.style.cssText = 'color:#88ff88;margin-bottom:10px;';
            panel.appendChild(tot);
        }
        if (data.gnn_best_margin !== undefined && data.gnn_best_margin !== null && gnnLabel) {
            const b = document.createElement('div');
            b.textContent = `GNN best-play margin: ${_fmtAhead(data.gnn_player, data.gnn_best_margin, 1)}`;
            b.style.cssText = 'color:#88ffcc;margin-bottom:10px;font-weight:bold;font-size:17px;';
            panel.appendChild(b);
        }

        // Heuristic total, shown from the leading side's perspective.
        if (data.heur_score !== undefined && data.heur_score !== null && gnnLabel) {
            const h = document.createElement('div');
            h.textContent = `Heuristic: ${_fmtAhead(data.gnn_player, data.heur_score, 1)}`;
            h.style.cssText = 'color:#ffcc88;margin-bottom:10px;font-weight:bold;font-size:16px;';
            panel.appendChild(h);
        }

        const renderBlock = (heading, components) => {
            if (!components) return;
            const h = document.createElement('div');
            h.textContent = heading;
            h.style.cssText = `font-weight:bold;margin:8px 0 3px;
                color:${heading.includes('White') ? '#ffffff' : '#aaaaff'};
                border-bottom:1px solid #444;padding-bottom:2px;`;
            panel.appendChild(h);

            const entries = Object.entries(components)
                .filter(([k]) => !k.startsWith('_'))
                .sort(([,a],[,b]) => Math.abs(b) - Math.abs(a));

            const table = document.createElement('table');
            table.style.cssText = 'width:100%;border-collapse:collapse;';
            for (const [k, v] of entries) {
                const tr = document.createElement('tr');
                const num = typeof v === 'number' ? v.toFixed(2) : String(v);
                tr.innerHTML = `
                    <td style="padding:1px 4px;color:#ccc;">${k.replace(/_/g,' ')}</td>
                    <td style="padding:1px 4px;text-align:right;
                        color:${parseFloat(num)>=0?'#88ff88':'#ff8888'};
                        font-weight:${Math.abs(parseFloat(num))>50?'bold':'normal'}">
                        ${num}
                    </td>`;
                table.appendChild(tr);
            }
            panel.appendChild(table);
        };

        // Route player/opponent blocks to the right colour label
        const p = data.player, o = data.opponent;
        const whiteComponents = p?._player === 'white' ? p : o;
        const blackComponents = p?._player === 'black' ? p : o;
        renderBlock('White', whiteComponents);
        renderBlock('Black', blackComponents);

        var _g = this.game;
        if (_g && _g.switchTurn) { var _o = _g.switchTurn.bind(_g); _g.switchTurn = function() { panel.remove(); _g.switchTurn = _o; _o(); }; }

        document.body.appendChild(panel);
        this._evalPanel = panel;
    }
    showThinkingIcon() {
        this.thinkingIcon.setAlpha(0); // Ensure it starts from fully transparent
        this.thinkingIcon.setVisible(true);
    
        this.tweens.add({
            targets: this.thinkingIcon,
            alpha: { from: 0, to: 1 },
            duration: 1000, // Duration of fade in (in ms)
            yoyo: true, // Enable yoyo to reverse the tween
            repeat: -1, // Repeat indefinitely
            ease: 'Power1'
        });
    }
    
    hideThinkingIcon() {
        this.tweens.killTweensOf(this.thinkingIcon); // Stop all tweens related to thinkingIcon
        this.thinkingIcon.setVisible(false);
    }
    
    // Same shared dialog as every other confirmation in the app.
    showNewGameConfirmationModal() {
        showConfirm('Start a new game?', () => {
            clearMoveRecording();
            this.scene.restart({ startingPlayer: nextCasualStarter() });
        }, 'New game');
    }

    saveGameState(game) {
        const state = getGameState(game);
        const json = JSON.stringify(state, null, 2);
        this.saveJSONToFile(json, 'game_state.json');
    }

    saveJSONToFile(json, filename) {
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    }

    update() {
        // Update logic if needed
    }

    checkInitialAIReady() {
        if (_gameFrozen) return;   // welcome up: don't let the AI open behind it
        const isBlackAI = this.startingPlayer === 'black' && BLACK_IS_AI;
        const isWhiteAI = this.startingPlayer === 'white' && WHITE_IS_AI;

        if (isWhiteAI || isBlackAI) {
            this.time.delayedCall(500, () => {
                this.game.scene.showThinkingIcon();
                const gameState = getGameState(this.game);
                setTimeout(() => getAgentMoves(gameState), 1000);
            });
        }
    }
}





class EndGameScene extends Phaser.Scene {
    constructor() {
        super({ key: 'EndGameScene' });
    }

    init(data) {
        this._data = data;          // kept so a rotation can rebuild the card
        this.winner = data.winner;
        this.score = data.score;
        this.impasse_caller = data.impasse_caller;
        this.inMatch = data.inMatch;
        this.matchOver = data.matchOver;
    }

    create() {
        // This scene has its OWN camera, and on a phone the game size is the
        // device-pixel buffer (2597x1200 in landscape), not the world. Without
        // framing it the card, drawn at world centre, sat ~400px left of the
        // visible centre in landscape and high above it in portrait.
        _fitCameraToWorld(this);
        // The card's size and position are computed from _world() HERE, so a
        // rotation invalidates them -- a card built in landscape is 1640 wide
        // and the portrait frame is 1160, so it hangs off both sides. Re-framing
        // the camera cannot fix geometry that was baked in; rebuild instead.
        // Only when the frame actually changed shape, so ordinary resize
        // chatter does not restart the scene under the player.
        const worldKey = () => { const w = _world(); return [w.x, w.y, w.w, w.h].join(','); };
        this._worldKey = worldKey();
        this.scale.on('resize', () => {
            _fitCameraToWorld(this);
            if (worldKey() !== this._worldKey) this.scene.restart(this._data);
        });

        let message;
        if (this.winner === 'draw') {
            const caller = this.impasse_caller || 'A player';
            message = `${_cap(caller)} calls a draw!`;
        } else {
            message = `${_cap(this.winner)} wins the game with a score of ${this.score}`;
        }

        const abortAndClear = () => {
            clearMoveRecording();
        };
        const startGame = (starter) => { abortAndClear(); this.scene.start('MainGameScene', { startingPlayer: starter }); };
        updateTurnStatus('');   // the game is over: drop the turn/thinking pill

        // Card, headline, sub-line and pill buttons — the same vocabulary as the
        // welcome / match-setup overlays (this screen used to be bare text on the
        // background with square green and blue buttons).
        // This card is the whole screen -- there is nothing to collide with --
        // so on a phone it is simply drawn bigger. At K=1 the numbers below are
        // exactly the desktop layout. P() scales the offsets the call sites use.
        // The card is 820 world px wide at K=1, and portrait's frame is only
        // 1160 wide -- at a flat K=2 it hung 36px off BOTH sides. Cap the scale
        // so the card always fits the frame it is drawn into.
        const _wd = _world();
        const K = _isPhone() ? Math.min(2, (_wd.w - 80) / 820) : 1;
        const P = (n) => n * K;
        const card = (h) => {
            const w = 820 * K, x = _wd.x + _wd.w / 2 - w / 2, y = _wd.y + _wd.h / 2 - (h * K) / 2;
            const g = this.add.graphics();
            g.fillStyle(0x000000, 0.10); g.fillRoundedRect(x, y + P(6), w, h * K, P(22));
            g.fillStyle(0xffffff, 1);    g.fillRoundedRect(x, y, w, h * K, P(22));
            return y;
        };
        const headline = (y, text, size) => this.add.text(CENTER_X, y, text, {
            fontSize: P(size) + 'px', fontFamily: HUD_FONT, fontStyle: 'bold',
            color: HUD_INK, align: 'center', wordWrap: { width: P(720) }
        }).setOrigin(0.5);
        const subline = (y, text, size) => this.add.text(CENTER_X, y, text, {
            fontSize: P(size || 23) + 'px', fontFamily: HUD_FONT, color: '#5a6473',
            align: 'center', wordWrap: { width: P(720) }
        }).setOrigin(0.5);
        const button = (x, y, label, ghost, cb) => {
            // The card's buttons are its only controls and it covers the screen,
            // so they get a further step up beyond the card's own scale.
            const b = makeHudButton(this, x, y, label, { ghost, k: _isPhone() ? K * 1.35 : 1 });
            onTap(b, cb);
            return b;
        };

        if (this.inMatch && matchTracker) {
            const m = matchTracker;
            if (this.matchOver) {
                // The winner's MARGIN, not their running total -- a match is won
                // on total score, so the interesting number is the gap. It can be
                // zero: level on score is broken by games won, and "by 0" would
                // be nonsense, so say how it was actually won.
                const mDiff = Math.abs(m.whiteScore - m.blackScore);
                const mres = m.winner === 'draw' ? 'The match is a draw!'
                    : mDiff > 0 ? `${_cap(m.winner)} wins the match by ${mDiff}`
                                : `${_cap(m.winner)} wins the match on games won`;
                const top = card(340);
                subline(top + P(58), message, 21);
                headline(top + P(118), mres, 34);
                subline(top + P(182),
                    `White ${m.whiteScore} (${m.whiteWins}W)   ·   Black ${m.blackScore} (${m.blackWins}W)   ·   ${m.gamesPlayed} games`, 21);
                button(CENTER_X - P(105), top + P(262), 'New Match', false,
                    () => { abortAndClear(); matchTracker = null; refreshSettingsMatchState(); showMatchSetup(); });
                button(CENTER_X + P(105), top + P(262), 'Single Game', true,
                    () => { matchTracker = null; refreshSettingsMatchState(); startGame('white'); });
            } else {
                const status = m.mode === 'race' ? `race to ${m.target}`
                    : `game ${m.gamesPlayed + 1} of ${m.target}`;
                const extended = m.justExtended;
                m.justExtended = false;
                const top = card(extended ? 330 : 290);
                headline(top + P(78), message, 34);
                subline(top + P(142),
                    `White ${m.whiteScore} (${m.whiteWins}W)   ·   Black ${m.blackScore} (${m.blackWins}W)   ·   ${status}`, 21);
                if (extended) {
                    // Was 20 against the score line's 21, and in a lighter accent
                    // colour, which read as noticeably smaller than everything
                    // else on the card. Matched to the score line and given the
                    // weight to go with being a one-off announcement.
                    subline(top + P(186),
                        `Level after ${m.extendedAt} games — match extended by 2`, 21)
                        .setColor(THEME.accentCss).setFontStyle('bold');
                }
                button(CENTER_X, top + P(extended ? 258 : 218), 'Next Game', false,
                    () => startGame(matchStarterForGame(m.gamesPlayed)));
            }
            return;
        }

        // Casual single-game flow.
        const top = card(250);
        headline(top + P(80), message, 36);
        button(CENTER_X - P(105), top + P(176), 'New Game', false, () => startGame(nextCasualStarter()));
        button(CENTER_X + P(105), top + P(176), 'New Match', true, () => { abortAndClear(); showMatchSetup(); });
    }

}



function calculateAverageScore() {
    if (scoreTracker.games_played === 0) {
        return 0; // Avoid division by zero
    }
    return scoreTracker.total_score / scoreTracker.games_played;
}

// Ensure these functions are defined outside of any class or method

// The eval readout (E). Answered on the device, always -- /evaluate_board was
// the fallback while the runtime loaded and is gone with the rest of the server
// path. Verified bit-exact against that route over 25 real states before it
// went (gnn_raw, gnn_best_margin and heur_score all 25/25, worst diff 0.0).
function evaluateBoard(gameState) {
    if (typeof LocalAgent === 'undefined' || !LocalAgent.enabled()) return Promise.resolve(null);
    return LocalAgent.init({ serverUrl: SERVER_URL })
        .then(ok => (ok ? LocalAgent.evaluate(gameState) : null))
        .then(data => { console.log('Evaluation:', data); return data; })
        .catch(error => { console.error('Error:', error); return null; });
}


// Bumped for every Game built; an agent reply naming an older one is discarded.
let _gameInstanceSeq = 0;
let _agentRetries = 0;

// On-device inference (see local_agent.js and PORTING.md step 7). Kicked off as
// soon as a game starts with a computer player -- NOT on the first move request,
// which is when it used to load, because there is no server to answer while it
// does. Starting here puts the ~4.5 MB in flight while the human takes their
// first turn. A human-vs-human session still never pays for it.
function _startLocalAI() {
    if (typeof LocalAgent === 'undefined' || !LocalAgent.enabled()) return;
    LocalAgent.init({ serverUrl: SERVER_URL });
}

// Kick the loader off if either side is the computer. Safe to call repeatedly;
// LocalAgent.init is idempotent.
function _startLocalAIIfNeeded() {
    if (!WHITE_IS_AI && !BLACK_IS_AI) return;
    _startLocalAI();
}

// THE ONLY WAY THE COMPUTER MOVES. There is no server path any more (see
// local_agent.js's header): the app is a folder of static files, so if this
// cannot answer, nothing can. Waits for the runtime rather than asking
// elsewhere while it loads.
function _askLocalForMoves(gameState) {
    return LocalAgent.init({ serverUrl: SERVER_URL })
        .then(ok => {
            if (!ok) throw new Error('on-device AI unavailable: ' + (LocalAgent.state().error || 'load failed'));
            return LocalAgent.selectMoves(gameState);
        })
        // ?aicompare=1 still checks each answer against a dev server, which is
        // how the port was verified. It reaches the server itself; nothing on
        // the normal path does.
        .then(move => (LocalAgent.comparing() && move
                            ? LocalAgent.compareWithServer(gameState, move)
                            : move))
        .then(move => ({ message: 'Success', move: move, local: true }))
        .catch(err => {
            // A bad ANSWER is a port bug and is permanent; a failed LOAD is
            // retryable and local_agent.js has already decided which this was.
            if (LocalAgent.ready()) LocalAgent.disable(err);
            throw err;
        });
}

function getAgentMoves(gameState) {
    // difficulty 1 = full strength (argmax); lower = more top-p sampling (weaker)
    gameState = Object.assign({}, gameState, { difficulty: getAIDifficulty() });
    // Which game asked. Starting a new game while the computer is thinking used
    // to let the reply land on the board that replaced it, applying moves for
    // pieces that no longer exist.
    const askedBy = (_currentGame() || {}).instanceId;
    console.log('Sending game state to agent:', gameState);
    if (typeof LocalAgent === 'undefined' || !LocalAgent.enabled()) {
        // ?localai=0, or the port files never parsed. There is no server to ask.
        return Promise.resolve().then(() => {
            const scene = _setupScene(); if (scene && scene.hideThinkingIcon) scene.hideThinkingIcon();
            flashNotice('The on-device computer is switched off for this session.', 6000);
            if (typeof updateTurnStatus === 'function') updateTurnStatus('Computer unavailable');
        });
    }
    return _askLocalForMoves(gameState)
    .then(data => {
        const now = (_currentGame() || {}).instanceId;
        if (askedBy !== undefined && now !== askedBy) {
            console.log('Discarding agent reply for a game that has been replaced');
            const sc = _setupScene(); if (sc && sc.hideThinkingIcon) sc.hideThinkingIcon();
            return;
        }
        _agentRetries = 0;
        updateNoSaveDisplay();
        if (data.move) {
            console.log('Agent moves:', data.move, data.local ? '(on-device)' : '(server)');
            applyMovePair(data.move);
        } else {
            console.log('No move to apply:', data.message);
            gameInstance.scene.scenes[0].hideThinkingIcon();
            gameInstance.scene.scenes[0].game.switchTurn();
        }
        gameInstance.scene.scenes[0].hideThinkingIcon();
    })
    .catch(error => {
        console.error('Error:', error);
        const scene = gameInstance.scene.scenes[0];
        scene.hideThinkingIcon();
        // One quiet retry, then hand control back rather than leaving the player
        // staring at a board that will never move. The retry is worth more now
        // than it was against a server: the usual cause is a dropped fetch of
        // the runtime, and local_agent.js resets a failed LOAD so the second
        // attempt genuinely re-tries it.
        if (_agentRetries < 1) {
            _agentRetries += 1;
            flashNotice('Getting the computer ready — retrying', 3000);
            setTimeout(() => { scene.showThinkingIcon(); getAgentMoves(gameState); }, 1500);
        } else {
            _agentRetries = 0;
            flashNotice('The computer couldn’t start on this device. Tap ↷ to try again.', 6000);
            if (typeof updateTurnStatus === 'function') updateTurnStatus('Computer unavailable');
        }
    });
}






function applyMove(move) { 
    const game = gameInstance.scene.scenes[0].game;
    move = move.slice(0, -1);  // because the QDN model returns 4 elements including the player
    console.log('Applying move:', move);
    if (!Array.isArray(move) || move.length !== 3) {
        console.error('Invalid move format:', move);
        return;
    }

    const pieceColorNumber = move[0];
    const targetRingSector = move[1];
    const dieRoll = move[2];

    // Check for the (0, 0, 0) tuple
    if (pieceColorNumber === 0 && targetRingSector === 0 && dieRoll === 0) {
        console.log('Received (0, 0, 0) tuple, switching turn.');
        game.switchTurn();
        return;
    }        

    if (!Array.isArray(pieceColorNumber) || pieceColorNumber.length !== 2) {
        console.error('Invalid piece color and number format:', pieceColorNumber);
        return;
    }

    if (targetRingSector !== 'save' && (!Array.isArray(targetRingSector) || targetRingSector.length !== 2)) {
        console.error('Invalid target ring and sector format:', targetRingSector);
        return;
    }

    const piece = findPieceByColorAndNumber(pieceColorNumber[0], pieceColorNumber[1]);
    const targetTile = targetRingSector === 'save' ? 'save' : findTileByRingAndSector(targetRingSector[0], targetRingSector[1]);

    if (piece && targetTile) {
        // Highlight the piece
        piece.isSelected = true;
        piece.updateColor();
        if (targetTile !== 'save') targetTile.highlight();
        setTimeout(() => {
            if (targetTile === 'save') {
                piece.save();
                piece.isSelected = false;
                piece.updateColor();
                
                // Update game state after applying the move
                game.state = game.captureState();
                
                // Check if there are unused dice
                if (game.dice.some(die => !die.used)) {
                    // Call the agent again for additional moves
                    console.log('Unused dice found, calling agent for additional moves.');
                    const gameState = getGameState(game);
                    setTimeout(() => getAgentMoves(gameState), 1000); // Delay before making the next move
                } else {
                    // No unused dice, switch the turn
                    console.log('No unused dice, switching turn.');
                    game.switchTurn();
                }
            } else if (game.movePiece(piece, targetTile, true)) {
                console.log(`Piece ${pieceColorNumber[0]} ${pieceColorNumber[1]} moved to ring ${targetRingSector[0]}, sector ${targetRingSector[1]}`);
                piece.reachableTiles = game.getReachableTilesByDice(piece); // Update reachable tiles

                piece.isSelected = false;
                piece.updateColor();
                targetTile.unhighlight();

                // Update game state after applying the move
                game.state = game.captureState();
                
                // Check if there are unused dice
                if (game.dice.some(die => !die.used)) {
                    // Call the agent again for additional moves
                    console.log('Unused dice found, calling agent for additional moves.');
                    const gameState = getGameState(game);
                    setTimeout(() => getAgentMoves(gameState), 1000); // Delay before making the next move
                } else {
                    // No unused dice, switch the turn
                    console.log('No unused dice, switching turn.');
                    game.switchTurn();
                }
            } else {
                console.log('Move not valid according to game rules.');
                game.switchTurn();
            }
        }, 1000); // 1 second delay to highlight the piece before moving
    } else {
        console.log('Piece or target tile not found for move:', move);
        game.switchTurn();
    }
}

function applyMovePair(movePair) {
    const game = gameInstance.scene.scenes[0].game;

    console.log('Applying move pair:', movePair);
    if (!Array.isArray(movePair) || movePair.length !== 2) {
        console.error('Invalid move pair format:', movePair);
        return;
    }

if (movePair.some(m => Array.isArray(m) && m[0] === 1 && m[1] === 1 && m[2] === 1)) {
    const caller = game.turn; 
    console.log(`${caller} called a draw.`);
    setTimeout(() => {
        game.endGame('draw', null, caller);
    }, 2000);
    return;
}
    // Both dice declined: nothing moves, so say so -- otherwise the turn just
    // silently comes back to the player, which reads like a missed move.
    const isPass = (m) => Array.isArray(m) && m[0] === 0 && m[1] === 0 && m[2] === 0;
    if (movePair.every(isPass)) {
        flashNotice(_cap(game.turn) + ' passed');
    }

    let [move1, move2] = movePair;

    // Ensure a numbered-piece save is applied before its companion move.
    // A numbered save requires the die whose value equals the goal number, but
    // movePiece() consumes whichever unused die reaches the target -- so if the
    // companion runs first it can steal the save's die and the save then fails.
    // Mirror the backend guard: reorder only when exactly one move is a numbered
    // save, the companion is not a bring-out (which has its own die requirement),
    // and the two moves are different pieces. This lets the backend stay
    // agnostic about move order, for any move source (human, heuristic, GNN).
    const isNumberedSave = (mv) =>
        Array.isArray(mv) && mv[1] === 'save' &&
        Array.isArray(mv[0]) && mv[0].length === 2 && mv[0][1] <= 6;

    const isBringOut = (mv) => {
        if (!Array.isArray(mv) || !Array.isArray(mv[0]) || mv[0].length !== 2) return false;
        const p = findPieceByColorAndNumber(mv[0][0], mv[0][1]);
        if (!p) return false;
        if (p.currentTile && p.currentTile.type === 'home') return true;
        if (p.rack && p.rack.type === 'unentered') return true;
        return false;
    };

    const save1 = isNumberedSave(move1);
    const save2 = isNumberedSave(move2);
    if (save1 !== save2) {
        const saveMove  = save1 ? move1 : move2;
        const otherMove = save1 ? move2 : move1;
        const samePiece = Array.isArray(saveMove[0]) && Array.isArray(otherMove[0]) &&
                  String(saveMove[0][0]) === String(otherMove[0][0]) &&
                  Number(saveMove[0][1]) === Number(otherMove[0][1]);
        if (samePiece) {
            // Ensure move-to-goal precedes save regardless of which order they arrived in
            move1 = otherMove;   // move first
            move2 = saveMove;    // save second
        } else if (!isBringOut(otherMove)) {
            move1 = saveMove;
            move2 = otherMove;
        }
    }

    // The agent's two moves play out through chained setTimeouts, so a new game
    // started mid-animation (New Game / New Match, or the end-of-match screen)
    // used to keep applying them to pieces whose Phaser objects were already
    // destroyed. Every deferred step re-checks that this game is still the one
    // on screen.
    const stillCurrent = () => {
        if (game.isDefunct) return false;
        const live = gameInstance.scene.scenes[0] && gameInstance.scene.scenes[0].game;
        return !!live && live.instanceId === game.instanceId;
    };

    function processMove(move, callback) {
        if (!stillCurrent()) { console.log('Abandoning agent move: the game has been replaced'); return; }
        console.log('Applying move:', move);
        if (!Array.isArray(move) || move.length !== 3) {
            console.error('Invalid move format:', move);
            return;
        }

        const pieceColorNumber = move[0];
        const targetRingSector = move[1];
        const dieRoll = move[2];

        // Check for the (0, 0, 0) tuple (pass move)
        if (pieceColorNumber === 0 && targetRingSector === 0 && dieRoll === 0) {
            console.log('Received (0, 0, 0) tuple, switching turn.');
            game.switchTurn();
            return;
        }

        if (!Array.isArray(pieceColorNumber) || pieceColorNumber.length !== 2) {
            console.error('Invalid piece color and number format:', pieceColorNumber);
            return;
        }

        const piece = findPieceByColorAndNumber(pieceColorNumber[0], pieceColorNumber[1]);
        // Check for saving opponent's piece

        if (targetRingSector === 0 && dieRoll === 0) {
            console.log('Saving one opponent piece from block', pieceColorNumber);
            const piece = findPieceByColorAndNumber(pieceColorNumber[0], pieceColorNumber[1]);
            if (piece && piece.currentTile) {
                // Highlight only the single piece being peeled off the block
                piece.isSelected = true;
                piece.updateColor();
                piece.currentTile.highlight();
                setTimeout(() => {
                    if (!stillCurrent()) return;
                    const savedRack = piece.color === 0xffffff ? game.whiteSavedRack : game.blackSavedRack;
                    piece.moveToRack(savedRack);   // peel only the named piece; rest of the block stays
                    game.registerSave();   // no-save streak resets immediately
                    game.dice.forEach(die => die.setUsed());
                    game.checkWinCondition();
                    callback();
                }, 1000);
            } else {
                callback();
            }
            return;
        }

        if (targetRingSector !== 'save' && (!Array.isArray(targetRingSector) || targetRingSector.length !== 2)) {
            console.error('Invalid target ring and sector format:', targetRingSector);
            return;
        }

        const targetTile = targetRingSector === 'save' ? 'save' : findTileByRingAndSector(targetRingSector[0], targetRingSector[1]);
        console.log('Piece:', piece, 'Target tile:', targetTile);

        if (piece && targetTile) {
            // Highlight the piece
            piece.isSelected = true;
            piece.updateColor();
            if (targetTile !== 'save') targetTile.highlight();
            setTimeout(() => {
                if (!stillCurrent()) return;
                if (targetTile === 'save') {
                    piece.save();
                    console.log(`Piece ${pieceColorNumber[0]} ${pieceColorNumber[1]} saved`);

                    piece.isSelected = false;
                    piece.updateColor();
                    
                    callback();
                } else if (game.movePiece(piece, targetTile, true)) {
                    console.log(`Piece ${pieceColorNumber[0]} ${pieceColorNumber[1]} moved to ring ${targetRingSector[0]}, sector ${targetRingSector[1]}`);
                    piece.reachableTiles = game.getReachableTilesByDice(piece); // Update reachable tiles

                    piece.isSelected = false;
                    piece.updateColor();
                    targetTile.unhighlight();

                    callback();
                } else {
                    console.log('Move not valid according to game rules.');
                    game.switchTurn();
                }
            }, 1000); // 1 second delay to highlight the piece before moving
        } else {
            console.log('Piece or target tile not found for move:', move);
            game.switchTurn();
        }
    }

    // Apply the first move, then the second move in sequence
    processMove(move1, () => {
        processMove(move2, () => {
            const neitherMoveWasPass = !(move1[0] === 0 && move1[1] === 0 && move1[2] === 0) && 
                                       !(move2[0] === 0 && move2[1] === 0 && move2[2] === 0);

            if (neitherMoveWasPass && game.dice.some(die => !die.used) && !extraMoveRequested) {
                console.log('Requesting extra move.');
                extraMoveRequested = true;
                const gameState = getGameState(game);
                setTimeout(() => getAgentMoves(gameState), 1000);
            } else {
                console.log('Applied both moves, switching turn.');
                game.switchTurn();
            }
        });
    });
}

function findPieceByColorAndNumber(color, number) {
    return gameInstance.scene.scenes[0].game.pieces.find(piece => piece.player === color && piece.number === number);
}

function findTileByRingAndSector(ring, sector) {
    return gameInstance.scene.scenes[0].game.tiles.find(tile => tile.ring === ring && tile.sector === sector);
}

function findPieceById(id) {
    const game = gameInstance.scene.scenes[0].game;
    return game.pieces.find(piece => {
        const pieceId = piece.number + (piece.player === 'black' ? TOTAL_PIECES : 0);
        return pieceId === id;
    });
}

function getGameState(game) {
    console.log('Getting game state details');
    const gameStateDetails = {
        currentTurn: game.turn,
        dice: game.dice.map(die => ({
            value: die.value,
            used: die.used
        })),
        racks: {
            whiteUnentered: game.whiteUnenteredRack.pieces.map(piece => ({
                color: piece.player,
                number: piece.number
            })),
            whiteSaved: game.whiteSavedRack.pieces.map(piece => ({
                color: piece.player,
                number: piece.number
            })),
            blackUnentered: game.blackUnenteredRack.pieces.map(piece => ({
                color: piece.player,
                number: piece.number
            })),
            blackSaved: game.blackSavedRack.pieces.map(piece => ({
                color: piece.player,
                number: piece.number
            })),
        },
        boardPieces: game.pieces.filter(piece => piece.currentTile).map(piece => {
            const pieceDetails = {
                color: piece.player,
                number: piece.number,
                tile: {
                    ring: piece.currentTile.ring,
                    sector: piece.currentTile.sector
                }
            };
            if (piece.reachableTiles && piece.reachableTiles.reachableBySum) {
                pieceDetails.reachableBySum = piece.reachableTiles.reachableBySum.map(tile => ({
                    ring: tile.ring,
                    sector: tile.sector
                }));
            }
            return pieceDetails;
        })
    };
    gameStateDetails.noSaveTurns = game.noSaveTurns;
    gameStateDetails.drawCallable = game.drawCallable;
    gameStateDetails.bothMidgame = game.bothInMidgame();
    return gameStateDetails;
}

// The page-unload /abort_game POST went with the recording chain: there is
// no server-side game to abort (hosting audit, CLAUDE.md).


// WORLD_W/H stay the coordinate system everything is laid out in (the board is
// drawn at 1800x1200 whatever the screen is). On a phone the canvas RESIZEs to
// fill the viewport and the camera frames that world inside it -- otherwise FIT
// letterboxes the canvas, and zooming in then just enlarges the board inside the
// same small rectangle, leaving the grey bands untouched. Desktop keeps FIT.
const WORLD_W = 1800, WORLD_H = 1200;
const config = {
    type: Phaser.AUTO,
    width: WORLD_W,
    height: WORLD_H,
    backgroundColor: BACKGROUND_COLOR,
    // Phones: NONE, because the size is managed by _sizeCanvasToScreen below --
    // RESIZE sets the drawing buffer to CSS pixels, which on a 3x screen renders
    // the board at a third of the device resolution and visibly breaks up the
    // tile and piece outlines. Desktop keeps FIT (its buffer is already 1800x1200).
    scale: _isPhone()
        ? { mode: Phaser.Scale.NONE }
        : { mode: Phaser.Scale.FIT, autoCenter: Phaser.Scale.CENTER_BOTH },
    scene: [MainGameScene, EndGameScene],
};

const gameInstance = new Phaser.Game(config);

// Pinch-to-zoom on a phone. Phaser captures touch, i.e. preventDefault() on
// every touch event over the canvas, and Chrome suppresses zooming for a whole
// touch sequence whose touchstart was cancelled -- so the board couldn't be
// zoomed. Two details make letting it through safe:
//   - touchstart/touchmove uncancelled: the browser can pinch. It can't do
//     anything else, because `touch-action: pinch-zoom` (index.html) rules out
//     one-finger panning and double-tap zoom, so dragging a piece and
//     double-tap-to-save still belong to the canvas.
//   - touchend still cancelled: that is what suppresses the compatibility
//     mouse events. Without it every tap is handled twice -- the second pass
//     reads as a double-click and puts a just-entered piece back on the rack,
//     so taps appear to do nothing.
// The flag is flipped from a capture-phase listener, which runs before
// Phaser's own handler reads it.
// A lost WebGL context -- backgrounded tab, GPU process restart, memory
// pressure on the device -- leaves the board technically working but crawling,
// and Phaser 3.55 does not restore it by itself. Say so rather than leaving it
// looking like the game got slow for no reason. (preventDefault is what allows
// a restore event to fire at all, if the browser manages one.)
setTimeout(() => {
    const cv = gameInstance && gameInstance.canvas;
    if (!cv || !cv.addEventListener) return;
    cv.addEventListener('webglcontextlost', (e) => {
        e.preventDefault();
        console.warn('WebGL context lost');
        if (typeof flashNotice === 'function') flashNotice('Graphics stalled — reload the page', 20000);
    });
    cv.addEventListener('webglcontextrestored', () => {
        console.warn('WebGL context restored');
        if (typeof flashNotice === 'function') flashNotice('Graphics restored', 2500);
    });
}, 0);

['touchstart', 'touchmove', 'touchend', 'touchcancel'].forEach(type => {
    window.addEventListener(type, () => {
        const tm = gameInstance.input && gameInstance.input.touch;
        if (!tm) return;
        // A phone owns every gesture itself now (camera pan and pinch), so every
        // touch event is cancelled. That is also what stops Chrome's edge-swipe
        // "back", which overscroll-behavior cannot: the page has no scroll
        // container for that property to apply to, so the swipe went straight to
        // history navigation and swallowed drags aimed at the board.
        // Desktop keeps the old dance: leave touchstart/touchmove alone so the
        // browser can still pinch-zoom, cancel touchend to kill the duplicate
        // compatibility mouse events.
        tm.capture = _isPhone() ? true : (type === 'touchend' || type === 'touchcancel');
    }, { capture: true, passive: true });
});

// Cancel touch gestures ourselves, in the capture phase, before anything else
// can claim them. Phaser's own `capture` flag did not actually cancel here
// (measured: capture true, defaultPrevented false), and Chrome decides whether a
// drag is an edge-swipe "back" on the first uncancelled touchmove.
// CHROME'S EDGE-SWIPE "BACK": cancel touchmove, but ONLY for gestures that began
// within a whisker of the left or right screen edge -- the strip Chrome reserves
// for history navigation. Blunter versions were tried and reverted, each
// measured: `overscroll-behavior: none` had no effect (the page has no scroll
// container for it to apply to); cancelling touchstart as well stopped tapping
// working at all; cancelling EVERY touchmove stopped piece dragging. Phaser's
// own `capture` flag does not cancel here either (measured: capture true,
// defaultPrevented false), which is why this is done by hand.
const EDGE_STRIP = 24;
let _edgeGesture = false;
// Is this touch starting on something the player can drag? Own geometry, not
// Phaser's hit test -- calling that from a pointer handler corrupts drag state.
function _touchStartsOnDraggable(clientX, clientY) {
    const cam = _mainCamera(), cv = gameInstance && gameInstance.canvas;
    const g = _currentGame();
    if (!cam || !cv || !g) return false;
    const r = cv.getBoundingClientRect();
    if (!r.width || !r.height) return false;
    const v = cam.worldView;
    const wx = v.x + (clientX - r.left) * (v.width / r.width);
    const wy = v.y + (clientY - r.top) * (v.height / r.height);
    const slop = 30;
    const near = (o, rad) => o && Math.hypot((o.x || 0) - wx, (o.y || 0) - wy) <= rad + slop;
    if ((g.pieces || []).some(p => !p.hidden && near(p, p.radius || STACK_PR))) return true;
    return _ghosts.some(gh => gh && gh.visible && near(gh, (gh.body && gh.body.radius) || 40));
}

window.addEventListener('touchstart', (e) => {
    if (!_isPhone() || !e.touches || e.touches.length !== 1) return;
    const t = e.touches[0];
    const atEdge = t.clientX <= EDGE_STRIP || t.clientX >= window.innerWidth - EDGE_STRIP;
    // The racks sit inside the edge strip in portrait, so a gesture that starts
    // on a piece must never be treated as an edge swipe -- cancelling its moves
    // was measured to stop the drag happening at all.
    _edgeGesture = atEdge && !_touchStartsOnDraggable(t.clientX, t.clientY);
}, { capture: true, passive: true });

window.addEventListener('touchmove', (e) => {
    if (!_isPhone() || !_edgeGesture || !e.cancelable) return;
    // Stop interfering once a real drag is under way: the racks sit inside the
    // edge strip in portrait, and cancelling every move of a piece drag was
    // measured to break it. Chrome decides on the FIRST moves, which are still
    // cancelled here, so the back gesture is still suppressed.
    const sc = _setupScene();
    if (sc && (sc._draggingPiece || sc._draggingGhost)) return;
    e.preventDefault();
}, { capture: true, passive: false });

['touchend', 'touchcancel'].forEach(type => {
    window.addEventListener(type, (e) => {
        if (!e.touches || e.touches.length === 0) _edgeGesture = false;
    }, { capture: true, passive: true });
});

// ── PANNING A ZOOMED BOARD ──────────────────────────────────────────────
// Panning is NOT done by handing gestures to the browser. `touch-action` stays
// `pinch-zoom` (index.html) so one finger always belongs to the canvas.
// Allowing one-finger browser panning while zoomed broke selection outright on
// a real phone: Chrome claims the gesture, and the preventDefault meant to
// protect taps on pieces could not be trusted, because under pinch-zoom the
// touch's clientX and getBoundingClientRect() are not in the same coordinate
// space. Two-finger drag still pans (pinch-zoom permits multi-finger panning).
// In-canvas camera pan/zoom is the way to give one-finger panning back.