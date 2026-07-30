/* Auto-detect environment for Koyeb or Localhost */
const IS_LOCAL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';

// This one variable now handles everything
const SERVER_URL = IS_LOCAL 
    ? 'http://localhost:10000' 
    : window.location.origin;

const DEBUG_MODE = false;
// Every human turn used to be posted to the backend and appended to
// training_data/*.jsonl. The training data that mattered has been collected and
// the agent is trained from self-play now, so this is off; the server side is
// gated too (RECORD_TRAINING=1 there re-enables it for a local collection run).
const RECORD_TRAINING_DATA = false;

const WHITE_IS_AI = false;
let BLACK_IS_AI = (function () {
    try { const s = localStorage.getItem('playVsComputer'); return s === null ? true : s === '1'; }
    catch (e) { return true; }
})();

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
// Dice: halfway between the original 80 and the 120 try, anchored so the pair's
// top-right corner stays put (die 2 right edge = 580, top = 50).
const DIE_SIZE = 100;
const DICE_Y = 50;
const DICE_X2 = 580 - DIE_SIZE;         // second (right) die
const DICE_X1 = DICE_X2 - DIE_SIZE - 20; // first die, 20px gap
// Rack pieces render a touch larger than board pieces (mockup look).
const RACK_PR = 22;

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
    const isAI = (p === 'black' && BLACK_IS_AI) || (p === 'white' && WHITE_IS_AI);
    if (isAI) return 'Computer thinking…';
    return BLACK_IS_AI ? 'Your turn' : _cap(p) + '’s turn';
}
// A quick expanding ring at (x,y) — capture (red) / save (accent) feedback.
function fxBurst(scene, x, y, color) {
    if (!getFeedbackEnabled() || !scene || !scene.add) return;
    const ring = scene.add.circle(x, y, 14, color, 0).setStrokeStyle(4, color, 0.9).setDepth(70);
    scene.tweens.add({ targets: ring, scale: 3.2, alpha: 0, duration: 430, ease: 'Cubic.easeOut',
        onComplete: () => ring.destroy() });
}

function updateTurnStatus(textOrGame) {
    const text = typeof textOrGame === 'string' ? textOrGame : turnStatusText(textOrGame);
    let el = document.getElementById('turnStatus');
    if (!el) {
        el = document.createElement('div'); el.id = 'turnStatus';
        el.style.cssText = 'position:fixed; top:10px; left:50%; transform:translateX(-50%); z-index:30;' +
            'font-family:' + HUD_FONT + '; font-size:14px; font-weight:600; color:#28313b;' +
            'background:rgba(255,255,255,.8); padding:5px 15px; border-radius:20px;' +
            'box-shadow:0 2px 8px rgba(0,0,0,.14); pointer-events:none; transition:opacity .2s;';
        document.body.appendChild(el);
    }
    el.textContent = text || '';
    el.style.opacity = text ? '1' : '0';
}

// Brief centred notice under the status pill, for things that would otherwise
// happen invisibly (the computer passing its whole turn).
function flashNotice(text, ms = 2400) {
    let el = document.getElementById('flashNotice');
    if (!el) {
        el = document.createElement('div'); el.id = 'flashNotice';
        el.style.cssText = 'position:fixed; top:44px; left:50%; transform:translateX(-50%);' +
            'z-index:31; font-family:' + HUD_FONT + '; font-size:13px; font-weight:600;' +
            'color:#5a6473; background:rgba(255,255,255,.82); padding:4px 12px;' +
            'border-radius:20px; box-shadow:0 2px 8px rgba(0,0,0,.12); pointer-events:none;' +
            'opacity:0; transition:opacity .2s;';
        document.body.appendChild(el);
    }
    el.textContent = text;
    el.style.opacity = '1';
    clearTimeout(el._t);
    el._t = setTimeout(() => { el.style.opacity = '0'; }, ms);
}

// ── KEYBOARD SHORTCUTS ──────────────────────────────────────────────────
// Z = undo one die, Enter/Space = end turn, Esc = deselect.
document.addEventListener('keydown', (e) => {
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
    gear.onmouseenter = () => gear.style.opacity = '1';
    gear.onmouseleave = () => gear.style.opacity = '.6';

    const panel = mk('div',
        'position:fixed; top:82px; right:12px; z-index:41; display:none;' +
        'background:#fff; color:#28313b; font-family:' + HUD_FONT + '; font-size:13px;' +
        'border:1px solid rgba(0,0,0,.15); border-radius:12px; padding:12px 14px; width:216px;' +
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

    // Play vs computer toggle
    const crow = mk('label', 'display:flex; align-items:center; gap:8px; cursor:pointer; margin-bottom:8px;');
    const pc = mk('input'); pc.type = 'checkbox'; pc.checked = BLACK_IS_AI;
    pc.onchange = () => {
        BLACK_IS_AI = pc.checked;
        try { localStorage.setItem('playVsComputer', BLACK_IS_AI ? '1' : '0'); } catch (e) {}
        const g = _currentGame(); if (g && g.updateBlackPlayerAIStatus) g.updateBlackPlayerAIStatus(BLACK_IS_AI);
    };
    crow.appendChild(pc); crow.appendChild(mk('span', null, 'Play vs computer'));
    panel.appendChild(crow);

    // Boolean toggles
    const toggle = (labelText, get, key, marginBottom) => {
        const row = mk('label', 'display:flex; align-items:center; gap:8px; cursor:pointer;' +
            (marginBottom ? ' margin-bottom:8px;' : ''));
        const cb = mk('input'); cb.type = 'checkbox'; cb.checked = get();
        cb.onchange = () => { try { localStorage.setItem(key, cb.checked ? '1' : '0'); } catch (e) {} };
        row.appendChild(cb); row.appendChild(mk('span', null, labelText));
        panel.appendChild(row);
    };
    toggle('Move & capture effects', getFeedbackEnabled, 'fxEnabled', true);
    toggle('End turn automatically when both dice used', getAutoEndTurn, 'autoEndTurn', true);
    toggle('Confirm ending a turn with a move left', getConfirmRiskyEnd, 'confirmRiskyEnd', false);

    // Interactive tutorial launcher
    const tut = mk('button',
        'width:100%; margin-top:12px; padding:8px 0; border-radius:8px; border:none; cursor:pointer;' +
        'font-family:' + HUD_FONT + '; font-weight:700; font-size:13px; background:' + THEME.accentCss + '; color:#fff;',
        'Interactive tutorial');
    tut.onclick = () => { panel.style.display = 'none'; startTutorial(); };
    panel.appendChild(tut);

    document.body.appendChild(gear); document.body.appendChild(panel);
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

    if (mode === 'side') {
        // Text in a column beside the board, board pinned to the other side.
        const bw = Math.min(380, Math.round(W * 0.34));
        b.style.width = bw + 'px';
        b.style.left = 'auto';
        b.style.right = gap + 'px';
        b.style.bottom = 'auto';
        b.style.top = '50%';
        b.style.transform = 'translateY(-50%)';
        b.style.maxHeight = (H - 2 * gap) + 'px';
        b.style.overflowY = 'auto';
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
        b.style.transform = 'translateX(-50%)';
        // On a tall narrow screen the text would otherwise eat most of the
        // height; cap it and let the longest steps scroll.
        const cap = Math.round(H * 0.45);
        b.style.maxHeight = cap + 'px';
        b.style.overflowY = 'auto';
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
function _tutHudVisible(on) {
    const scene = _setupScene();
    if (scene && scene.hudButtons) scene.hudButtons.forEach(b => b.setHudVisible && b.setHudVisible(on));
}
window.addEventListener('resize', () => { if (_tut.active) setTimeout(_tutFitBoard, 60); });
window.addEventListener('orientationchange', () => { if (_tut.active) setTimeout(_tutFitBoard, 250); });

function _tutNudge() {
    const b = _tut.bubble; if (!b) return;
    clearInterval(_tut.shake);
    let n = 0;
    b.style.transition = 'transform .08s ease-in-out';
    _tut.shake = setInterval(() => {
        b.style.transform = 'translateX(-50%) translateX(' + ((n % 2) ? 7 : -7) + 'px)';
        if (++n > 3) { clearInterval(_tut.shake); _tut.shake = null; b.style.transform = 'translateX(-50%)'; }
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
        '<div style="font-family:' + BODY_FONT + '; font-size:14.5px; line-height:1.5; color:#33404b;">' + step.text + '</div>' +
        '<div id="tutBtns" style="display:flex; gap:8px; margin-top:auto; padding-top:13px;' +
            'justify-content:flex-end; min-height:32px; align-items:center;"></div>';
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
    _tutNote('<span style="color:#8b95a3; font-weight:700; font-size:13px;">Black plays…</span>');
    if (typeof updateTurnStatus === 'function') updateTurnStatus('Black’s turn');
    let i = 0;
    const next = () => {
        if (!_tut.active) { cb(); return; }
        if (i >= moves.length) {
            if (typeof updateTurnStatus === 'function') updateTurnStatus(game);
            setTimeout(cb, 400); return;
        }
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
    _tutHudVisible(false);
    _tutBubble();
    _tutRender();
    _tutFitBoard();
    clearInterval(_tut.timer); _tut.timer = setInterval(_tutPoll, 300);
}
function _tutEnd(startGame) {
    _tut.active = false; window._tutorialActive = false;
    _tut.busy = false;
    clearInterval(_tut.timer); _tut.timer = null;
    clearInterval(_tut.shake); _tut.shake = null;
    if (_tut.bubble) { _tut.bubble.remove(); _tut.bubble = null; }
    _tutFitBoard();                       // give the board the full window back
    _tutHudVisible(true);
    const scene = _setupScene();
    if (scene && scene.scene) scene.scene.restart({ welcome: true });
}

// Defer to after the whole script has run (this file `defer`s, so the DOM is
// ready; setTimeout ensures later `let` globals like matchTracker are initialised
// before createSettingsPanel -> refreshSettingsMatchState touches them).
function _initChrome() { createSettingsPanel(); createLegendButton(); maybeShowFirstRunNudge(); }
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
function makeHudButton(scene, cx, cy, label, { ghost = false } = {}) {
    const txt = scene.add.text(cx, cy, label, {
        fontSize: '19px', fontFamily: HUD_FONT, fontStyle: 'bold',
        color: ghost ? HUD_INK : THEME.accentInk, padding: { x: 16, y: 9 }
    }).setOrigin(0.5).setDepth(2).setInteractive({ useHandCursor: true });
    const b = txt.getBounds();
    const r = 9;
    const g = scene.add.graphics().setDepth(1);
    g.fillStyle(0x000000, 0.12); g.fillRoundedRect(b.x, b.y + 2, b.width, b.height, r);
    if (ghost) {
        g.fillStyle(0xffffff, 1); g.fillRoundedRect(b.x, b.y, b.width, b.height, r);
        g.lineStyle(1, HUD_PANEL_BORDER, 1); g.strokeRoundedRect(b.x, b.y, b.width, b.height, r);
    } else {
        g.fillStyle(THEME.accent, 1); g.fillRoundedRect(b.x, b.y, b.width, b.height, r);
    }
    txt.bg = g;                       // so callers can show/hide the whole button
    txt.setHudVisible = (v) => { txt.setVisible(v); g.setVisible(v);
        if (txt.input) txt.input.enabled = v; return txt; };
    txt.recolor = () => {             // re-apply theme colours in place (live theme switch)
        txt.setColor(ghost ? HUD_INK : THEME.accentInk);
        g.clear();
        g.fillStyle(0x000000, 0.12); g.fillRoundedRect(b.x, b.y + 2, b.width, b.height, r);
        if (ghost) {
            g.fillStyle(0xffffff, 1); g.fillRoundedRect(b.x, b.y, b.width, b.height, r);
            g.lineStyle(1, HUD_PANEL_BORDER, 1); g.strokeRoundedRect(b.x, b.y, b.width, b.height, r);
        } else {
            g.fillStyle(THEME.accent, 1); g.fillRoundedRect(b.x, b.y, b.width, b.height, r);
        }
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
            else { m.target += 2; }
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
    return prefix + sep + parts.join(sep);
}

// Start a fresh game as the first game of a match (called after startNewMatch).
// The coin flip runs first, then the fresh game is started — so nothing (incl.
// a black/AI opener) moves until the flip resolves, mirroring the casual path.
function _startMatchFirstGame(starter) {
    if (currentGameId) {
        fetch(`${SERVER_URL}/abort_game`, { method: 'POST',
            headers: { 'Content-Type': 'application/json' }, credentials: 'include' }).catch(() => {});
        currentGameId = null; moveCounter = 0; clearMoveRecording();
    }
    showCoinFlip(starter, () => {   // reveal who goes first, then start the game
        if (typeof gameInstance !== 'undefined' && gameInstance && gameInstance.scene) {
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
            if (currentGameId) {
                fetch(`${SERVER_URL}/abort_game`, { method: 'POST',
                    headers: { 'Content-Type': 'application/json' }, credentials: 'include' }).catch(() => {});
                currentGameId = null; moveCounter = 0; clearMoveRecording();
            }
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
    const sections = [
        ['Goal', 'Be the first to <i>save</i> all your pieces. Your score for a win is the number of pieces your opponent still had left — so winning big is worth more.'],
        ['Your pieces', 'You have 12: six numbered (1–6) and six blank. They start on your side rack.'],
        ['A turn', 'Roll two dice and move. Each die moves one piece a number of tiles equal to that die; you can move one piece with each die, or one piece with both (their sum). A piece always takes the shortest route to the tile you choose, and once it has moved with one die it can’t double back with the other. You may skip a die (or the whole turn).'],
        ['Getting on the board', 'Pieces enter through the home tile — the plain disc at the centre. Only the front piece on your rack can enter, and you must enter at least one piece per turn until your rack is empty (unless you have a captured piece, in which case you must enter that).'],
        ['Capturing &amp; blocking', 'Land on a field tile holding a single enemy piece and you capture it — it goes back to the home tile and its owner must re-enter it before doing anything else. A tile with <b>two or more</b> enemy pieces is a wall: you can’t enter or pass through it.'],
        ['Saving', 'The six coloured wedges on the rim are goals, numbered 1–6. To save a piece, get it onto a goal and roll that goal’s number to lift it off the board. A numbered piece can only be saved from its own goal; a blank piece from any goal. (You can start saving once all your pieces are on the board.)'],
        ['Endgame', 'When every piece you have left is saved or sitting on a goal it can be saved from, you’re in the endgame: blank pieces can now be saved with a roll <i>higher</i> than their goal’s number, as long as you have nothing waiting on a higher-numbered goal.'],
        ['A couple of special moves', '• Break a wall: past the opening and with no captured pieces, double-click (or drag from the picker) one piece of an enemy two-stack to save it for them — it costs both your dice and hands the opponent a piece, but turns the wall into a lone piece.<br>• Last piece: if you start a turn with a single piece left and it’s a numbered one sitting on its goal, it becomes blank (savable by any roll of that goal number or higher).'],
        ['Stalemate', 'If 10 full rounds pass with nobody saving a piece, either player may call a draw. Any save resets the counter.'],
        ['Matches', 'A match is several games, and it is won on <b>total score</b> — the sum of your winning margins — not on games won. Two formats: a set number of games (highest total score at the end wins), or a race to a target score. Starters alternate; if the scores finish level the match goes to whoever won more games, and if that is level too it is extended by a pair of games. The score line under the board tracks the match.'],
        ['Controls', 'Tap or drag a piece to move it; drag onto its goal — or double-click — to save. The ↶ arrow undoes one die at a time; ↷ ends your turn. On a crowded tile the <b>+N</b> badge opens a picker (drag a piece straight out of it). Theme, difficulty and options live under the ⚙ settings, and <b>New Match</b> starts a multi-game match.<br>On desktop: <b>Z</b> undoes one die · <b>Enter</b> or <b>Space</b> ends your turn · <b>Esc</b> deselects the piece you’re holding.'],
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
            '<input type="radio" name="mmode" value="games" checked> Set number of games (by total score)</label>' +
          '<div id="gamesOpts" style="margin:2px 0 12px 26px; font-size:14px;">' +
            'Games: <input id="mGames" type="number" min="2" step="2" value="' + MATCH_DEFAULT_GAMES + '" style="width:56px;">' +
            '<div style="margin-top:8px;">On a tie: ' +
              '<label style="margin-left:4px;"><input type="radio" name="mtie" value="extra" checked> extra pair</label>' +
              '<label style="margin-left:10px;"><input type="radio" name="mtie" value="draw"> draw</label></div></div>' +
          '<label style="display:flex; gap:8px; align-items:center; margin:6px 0; font-size:15px;">' +
            '<input type="radio" name="mmode" value="race"> Race to a total score</label>' +
          '<div id="raceOpts" style="margin:2px 0 12px 26px; font-size:14px; opacity:.5;">' +
            'Target: <input id="mRace" type="number" min="1" value="' + MATCH_DEFAULT_RACE + '" style="width:56px;" disabled></div>' +
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
        box.querySelectorAll('input[name=mtie]').forEach(r => r.disabled = mode !== 'games');
        $('#raceOpts').style.opacity = mode === 'race' ? '1' : '.5';
        $('#mRace').disabled = mode !== 'race';
    };
    modeRadios.forEach(r => r.addEventListener('change', sync)); sync();
    $('#mCancel').onclick = () => { box.remove(); if (onCancel) onCancel(); };
    $('#mStart').onclick = () => {
        const mode = [...modeRadios].find(r => r.checked).value;
        let target, tieRule = 'extra';
        if (mode === 'games') {
            target = Math.max(2, parseInt($('#mGames').value) || MATCH_DEFAULT_GAMES);
            if (target % 2 !== 0) target += 1;                       // keep it even
            tieRule = [...box.querySelectorAll('input[name=mtie]')].find(r => r.checked).value;
        } else {
            target = Math.max(1, parseInt($('#mRace').value) || MATCH_DEFAULT_RACE);
        }
        box.remove();
        const starter = startNewMatch({ mode, target, tieRule });
        _startMatchFirstGame(starter);
    };
}

let extraMoveRequested = false;

// ── DATA COLLECTION GLOBALS ─────────────────────────────────────────────
let currentGameId = null;
let moveCounter = 0;
let _pendingMoves = [];   // stores moves made this turn in agent format
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
const ALLOW_DEV_MODES = new URLSearchParams(location.search).get('dev') === '1';

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
function _setupPlaceOnTile(piece, tile) {
    _setupRemoveFromCurrent(piece);
    piece.currentTile = tile;
    piece.rack = null;
    piece.justMovedHome = false;
    tile.addPiece(piece);            // pushes + updatePositions() => positions & sizes the piece
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
        console.log(`[SETUP] die ${e.code === 'Digit1' ? 1 : 2}: value ${die.value}, used ${die.used}`);
    } else if (e.key === 't' || e.key === 'T') {
        e.preventDefault();
        game.turn = game.turn === 'white' ? 'black' : 'white';
        game.dice.forEach(d => d.updateColor(game.turn));
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
    // in sandbox (which doesn't track game stages).
    const show = window.setupMode || game.bothInMidgame();
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
        if (this.rack && this.rack.type === 'saved') return;
        if (this.rack && this.rack.type === 'unentered' && this.rack.pieces[0] !== this) return;
        if (!this.game.canSelectForMove(this)) return false;
        this.isHovered = true;
        this.updateColor();
    }

    onOut() {
        if (this.game.selectedPiece && this.game.selectedPiece !== this) return;
        if (this.player !== this.game.turn) return; 
        if (this.rack && this.rack.type === 'saved') return;
        if (this.rack && this.rack.type === 'unentered' && this.rack.pieces[0] !== this) return;
        
        this.isHovered = false;
        this.updateColor();
    }

    handleClick(pointer) {
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
        
        // Call the backend
        fetch(`${SERVER_URL}/debug_piece_blots`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                gameState: gameState,
                piece: {
                    player: this.player,
                    number: this.number
                }
            })
        })
        .then(response => response.json())
        .then(data => {
            console.log(`📊 Piece ${this.player}(${this.number}):`);
            console.log(`   Distance to goal: ${data.distance === Infinity ? 'No path' : data.distance}`);
            console.log(`   Enemy blots on path: ${data.blot_count === Infinity ? 'No path' : data.blot_count}`);
            console.log(`   Can be saved: ${data.can_be_saved}`);
        })
        .catch(error => {
            console.error('Error getting blot info:', error);
        });
        
        return; // Stop here, don't select the piece
    }

        if (this.game.gameOver) return; 
        if (this.game.dice[0].used && this.game.dice[1].used) return;



        // Saved pieces are out of play: checked before the selection handover
        // below, which would otherwise make one the selected piece -- and with
        // that, draggable back onto the board.
        if (this.rack && this.rack.type === 'saved') return;

        if (this.game.selectedPiece && this.game.selectedPiece !== this) {

            // If this piece is on a field tile, treat as tile click instead
            if (this.currentTile && this.currentTile.type === 'field') {
                this.currentTile.onClick();
                return;
            }

            this.game.selectedPiece.isSelected = false;
            if (this.game.selectedPiece.currentTile && this.game.selectedPiece.currentTile.type === 'home' && this.game.selectedPiece.justMovedHome) {
                this.game.selectedPiece.returnToRack();}
            this.game.selectedPiece.updateColor();
            this.game.selectedPiece = this;
            this.game.unhighlightAllTiles();
            this.isSelected = false;
        }
        // if (this.player !== this.game.turn) return; 
        if (this.rack && this.rack.type === 'unentered' && this.rack.pieces[0] !== this) return;
        if (this.player === this.game.turn && !this.game.canSelectForMove(this)) {
            console.log("Must keep a die for the obligatory piece(s)");
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
            this.moveFromRack();
            this.justMovedHome = true;
            this.game.selectedPiece = this;
            this.reachableTiles = this.game.getReachableTilesByDice(this);
            this.highlightReachableTiles();
        }
        else if (this.currentTile && this.currentTile.type === 'home' && this.justMovedHome) {
                this.returnToRack();
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
            this.save(); // Save the piece if it can be saved
        } else if (this.player === this.game.turn && this.game.sumSave(this)) {
            // not on a goal yet, but one die reaches the goal and the other saves
            // it this turn -> do both at once.
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
            this.text.setFontSize(`${size * 1.7}px`);
        }
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
        
        // Add a temporary endpoint to the server to get blot count
        // Or we can calculate it locally (more complex)
        
        // For now, let's request the evaluation which includes distance info
        try {
            const response = await fetch(`${SERVER_URL}/debug_piece_info`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    gameState: gameState,
                    piece: {
                        player: this.player,
                        number: this.number
                    }
                })
            });
            
            if (response.ok) {
                const data = await response.json();
                console.log(`Distance to goal: ${data.distance === Infinity ? 'No path' : data.distance} steps`);
                console.log(`Enemy blots on shortest path: ${data.blot_count === Infinity ? 'No path' : data.blot_count}`);
                if (data.path && data.path.length > 0) {
                    console.log(`Path length: ${data.path.length} tiles (including start and goal)`);
                    console.log(`Path: ${data.path.map(t => `${t.ring},${t.sector}`).join(' → ')}`);
                }
                if (data.can_be_saved) {
                    console.log(`✓ Piece can be saved immediately!`);
                }
            } else {
                console.log(`Could not get debug info from server`);
                // Fallback to local calculation
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
        

    

    moveFromRack() {
        const homeTile = this.game.tiles.find(tile => tile.type === 'home');
        this.rack.removePiece(this);
        this.rack.shiftPiecesUp();
        this.rack = null;
        this.move(homeTile, false);
        this._turnStartTile = homeTile;   // an entering piece measures progress from home
        this.game.selectedPiece = this;
        this.isSelected = true;
    }

    moveToRack(rack, addToFront = false) {
        this.rack = rack;
        this.x = rack.nextX();
        this.y = rack.nextY();
        this.setSize(RACK_PR);
        this.body.setPosition(this.x, this.y);
        this.circle.setPosition(this.x, this.y);
        this._layoutSheen();
        if (this.text) {
            this.text.setPosition(this.x, this.y);
        }
        this.setVisible(true);   // a piece hidden as tile overflow reappears in the rack
        if (addToFront) {
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
        this.moveToRack(unenteredRack, true);
        this.justMovedHome = false;
        this.reachableTiles = null;
        this.game.selectedPiece = null;
        this.game.tiles.forEach(tile => {
            tile.unhighlight();
        })
    }

    updateColor() {
        if (!this.body) return;
        // highlight (selected/hovered) recolors the body; the rim + sheen stay.
        if (this.isSelected || this.isHovered) {
            this.body.setFillStyle(this.color === 0xffffff ? 0x90ee90 : 0xee82ee);
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
            .on('pointerdown', (pointer) => this.handleClick(pointer));
        // drag-to-move (additive; click still works). The scene-level drag
        // handlers (Game.setupDragging) reach the piece via __piece.
        this.circle.__piece = this;
        this.scene.input.setDraggable(this.circle);

        // Debug-mode tooltip: show the number of unnumbered pieces (numbered
        // pieces already display their number on the board).
        this.circle
            .on('pointerover', () => { if (window.debugMode && this.number > 6) showDebugTip(`${this.player} #${this.number}`); })
            .on('pointermove', () => { if (window.debugMode && this.number > 6) showDebugTip(`${this.player} #${this.number}`); })
            .on('pointerout',  () => { hideDebugTip(); });

        if (this.number <= 6 || DEBUG_MODE) {
            this.text = this.scene.add.text(this.x, this.y, this.number, {
                fontSize: `${this.radius * 1.7}px`,
                color: `#${this.textColor.toString(16).padStart(6, '0')}`,
                fontStyle: 'bold'
            }).setOrigin(0.5, 0.5);
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
                    piece.updateColor();
                    this.game.selectedPiece = null;
                } else {
                    console.log('Move not possible');
                }
            }
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

    highlight() {   
        let color = this.reachableColor !== null ? this.reachableColor : this.highlightColor;
        this.graphics.fillStyle(color, 1); 
        this.graphics.fillPath();
    }

    unhighlight() {
        this.graphics.fillStyle(this.fillColor, 1);
        this.graphics.fillPath();
    }

    onOut() {
        let color = this.reachableColor !== null ? this.reachableColor : this.fillColor;
        this.graphics.fillStyle(color, 1);
        this.graphics.fillPath();
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
        let r = STACK_PR;
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
                .on('pointerdown', () => {
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
    
    
    drawTile() {
        
        this.graphics.clear();
        // nogo = "no board space": draw nothing so the background shows through
        // and no nogo fill covers an adjacent field tile's border.
        if (this.type === 'nogo') return;
        this.graphics.lineStyle(1.7, this.lineColor, 1);
        this.graphics.fillStyle(this.fillColor, 1);

        if (this.type === "home") {
            this.x = CENTER_X;
            this.y = CENTER_Y;
            this.graphics.fillCircle(CENTER_X, CENTER_Y, HOME_TILE_RADIUS);
            this.graphics.strokeCircle(CENTER_X, CENTER_Y, HOME_TILE_RADIUS);
        } else {
    
            const points = this.calculateAnnularSegmentPoints(CENTER_X, CENTER_Y, this.innerRadius, this.outerRadius, this.startAngle, this.endAngle);



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

            this.graphics.setInteractive(new Phaser.Geom.Polygon(points), Phaser.Geom.Polygon.Contains)
                .on('pointerdown', () => this.onClick())
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
    
    
    
    
}

class Rack {
    constructor(scene, x, y, color, type, rows = 4) {
        this.scene = scene;
        this.x = x;
        this.y = y;
        this.color = color;
        this.type = type;
        this.pieces = [];
        this.rows = rows;
        this.cols = 3;
        this.spacing = RACK_PR * 2 + 12;
        this.verticalPadding = 22;
        this.horizontalPadding = 18;
        this.background = scene.add.graphics();
        this.drawBackground();
    }

    addPiece(piece) {
        this.pieces.push(piece);
        piece.rack = this;
    }

    removePiece(piece) {
        this.pieces = this.pieces.filter(p => p !== piece);
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
                piece.setSize(RACK_PR);
            }
        }
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

    drawBackground() {
        // Clean Modern (matches mockup): white rounded panel + soft shadow +
        // faint empty capacity slots. No text.
        const bx = this.x - RACK_PR, by = this.y - RACK_PR;
        const bw = this.cols * this.spacing + RACK_PR;
        const bh = this.rows * this.spacing + RACK_PR + this.verticalPadding;
        this.background.fillStyle(0x000000, 0.07);
        this.background.fillRoundedRect(bx, by + 5, bw, bh, 16);      // soft drop shadow
        this.background.fillStyle(0xffffff, 1);
        this.background.fillRoundedRect(bx, by, bw, bh, 16);
        this.background.lineStyle(1.5, 0xdbe1ea, 1);
        this.background.strokeRoundedRect(bx, by, bw, bh, 16);
        // faint slot circles show the rack's capacity (like the mockup)
        this.background.lineStyle(1.5, 0xdbe1ea, 0.85);
        for (let i = 0; i < this.cols * this.rows; i++) {
            const sx = this.x + this.horizontalPadding + (i % this.cols) * this.spacing;
            const sy = this.y + this.verticalPadding + Math.floor(i / this.cols) * this.spacing;
            this.background.strokeCircle(sx, sy, RACK_PR);
        }
    }
}



class Die {
    constructor(scene, x, y, isFirstDie) {
        this.scene = scene;
        this.value = Phaser.Math.Between(1, 6);
        this.x = x;
        this.y = y;
        this.size = DIE_SIZE;
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
        this.graphics.fillStyle(0x000000, 0.10);
        this.graphics.fillRoundedRect(this.x, this.y + 4, this.size, this.size, 14);  // soft shadow
        // Colour-coded border preserved: die A vs die B. Drawn as a slightly
        // larger filled rounded rect *behind* the face rather than with
        // strokeRoundedRect: a thick stroked path here left stray coloured
        // lines running from the dice across the board on some Android GPUs
        // (the WebGL line batch joining onto the next shape). Two fills have
        // no path to leak.
        const borderColor = this.isFirstDie ? colorFirstDie : colorSecondDie;
        const bw = 5;
        this.graphics.fillStyle(borderColor, 1);
        this.graphics.fillRoundedRect(this.x - bw / 2, this.y - bw / 2,
                                      this.size + bw, this.size + bw, 14 + bw / 2);
        this.graphics.fillStyle(dieColor, 1);
        this.graphics.fillRoundedRect(this.x, this.y, this.size, this.size, 14);

        const dotSize = this.size * 0.11; // scales with the die
        const dotOffset = this.size / 4;

        const drawDot = (dx, dy) => {
            this.graphics.fillStyle(dotColor, 1);
            this.graphics.fillCircle(this.x + dx, this.y + dy, dotSize);
        };

        const midPoint = this.size / 2;

        // Dice faces based on value
        if ([1, 3, 5].includes(this.value)) drawDot(midPoint, midPoint);
        if (this.value > 1) {
            drawDot(dotOffset, dotOffset);
            drawDot(this.size - dotOffset, this.size - dotOffset);
        }
        if (this.value > 3) {
            drawDot(dotOffset, this.size - dotOffset);
            drawDot(this.size - dotOffset, dotOffset);
        }
        if (this.value === 6) {
            drawDot(dotOffset, midPoint);
            drawDot(this.size - dotOffset, midPoint);
        }
    }
}



class Game {
    constructor(scene, startingPlayer = 'white', debug = false) {
        this.scene = scene;
        this.players = [new Player('white', WHITE_IS_AI), new Player('black', BLACK_IS_AI)];
        this.startingPlayer = startingPlayer;
        this.turn = this.startingPlayer;
        this.dice = [new Die(scene, DICE_X1, DICE_Y, true), new Die(scene, DICE_X2, DICE_Y, false)];
        this.gameOver = false;
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
        this.whiteUnenteredRack = new Rack(scene, 75, 356, 'white', 'unentered');
        this.whiteSavedRack = new Rack(scene, 75, 622, 'white', 'saved');
        this.blackUnenteredRack = new Rack(scene, 1545, 356, 'black', 'unentered');
        this.blackSavedRack = new Rack(scene, 1545, 622, 'black', 'saved');

        this.setupDragging(scene);

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
            piece.setSize(RACK_PR);
            piece.setPosition(this.whiteUnenteredRack.nextX(), this.whiteUnenteredRack.nextY());
            this.whiteUnenteredRack.addPiece(piece);
        });

        blackPieces.forEach(piece => {
            piece.setSize(RACK_PR);
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

                    this.tiles.forEach(t => {
                        if (t.ring === 7 && t.type !== 'nogo' &&
                            t.startAngle < tile.endAngle &&
                            t.endAngle > tile.startAngle) {
                            t.graphics.lineStyle(1, 0x000000, 1);
                            t.graphics.beginPath();
                            const step = Math.PI / 180;
                            for (let angle = t.startAngle; angle <= t.endAngle; angle += step) {
                                const x = CENTER_X + t.innerRadius * Math.cos(angle);
                                const y = CENTER_Y + t.innerRadius * Math.sin(angle);
                                if (angle === t.startAngle) t.graphics.moveTo(x, y);
                                else t.graphics.lineTo(x, y);
                            }
                            t.graphics.strokePath();
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
        // The tutorial hard-blocks: off-script destinations are dropped here, so
        // they are neither highlighted nor accepted by movePiece.
        if (_tut.active) return _tutFilterReach(this, piece, { reachableByFirstDie, reachableBySecondDie, reachableBySum });
        return { reachableByFirstDie, reachableBySecondDie, reachableBySum };
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

        if (!reachableTiles && !getReachableTiles) return false;

        if (!reachableTiles) {  // this is called from AI agent's applyMove
            reachableTiles = this.getReachableTilesByDice(piece);
            piece.reachableTiles = reachableTiles}  

        if (!reachableTiles) return false;

        const { reachableByFirstDie, reachableBySecondDie, reachableBySum } = reachableTiles;

        const allReachableTiles = new Set([...reachableByFirstDie, ...reachableBySecondDie, ...reachableBySum]);
    
        if (allReachableTiles.has(targetTile)) {

            if (this.isBlocked(targetTile)) {
                return false; // Can't move to a tile with more than one opposing piece
            }

            // Obligatory-move ordering: a non-obligatory move must leave a die for
            // every still-pending obligatory piece.
            if (this.mustMovePieces.length > 0 && !this.mustMovePieces.includes(piece)) {
                const unused = this.dice.filter(d => !d.used).length;
                const willUse = (reachableByFirstDie.includes(targetTile) ||
                                 reachableBySecondDie.includes(targetTile)) ? 1 : 2;
                if (unused - willUse < this.mustMovePieces.length) {
                    console.log('Must keep a die for the obligatory piece(s)');
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
            piece.move(targetTile);
            piece.animateFrom(_ox, _oy);

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
    sumSave(piece) {
        if (!piece.currentTile || piece.currentTile.type === 'save') return false;
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

    // Does the current player have any legal move left with the unused dice?
    // (Used to decide whether ending the turn is "risky".)
    hasAnyLegalMove() {
        if (this.dice.every(d => d.used)) return false;
        const color = this.turn === 'white' ? 0xffffff : 0x000000;
        const candidates = this.mustMovePieces.length > 0
            ? this.mustMovePieces.slice()
            : this.pieces.filter(p => p.color === color &&
                (p.currentTile || (p.rack && p.rack.type === 'unentered' && p.rack.pieces[0] === p)));
        for (const p of candidates) {
            if (p.rack && p.rack.type === 'unentered' && p.rack.pieces[0] !== p) continue;
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
        if (this.mustMovePieces && this.mustMovePieces.length > 0) return;
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
    
        for (const tile of allIntermediateTiles) {
            if (captureConditionsMet(tile)) {
                console.log('Capturing piece at intermediate tile:', tile);
                this.capturePiece(tile.pieces[0]);
                break; // Capture only one piece and break out of the loop
            }
        }
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
            return; // If there are captured pieces, no other pieces may move
        }

        // Check if there's a piece in the unentered rack
        if (unenteredRack.pieces.length > 0) {
            this.mustMovePieces = [unenteredRack.pieces[0]]; // The first piece in the unentered rack must move
        }
        if (typeof updateMustMoveHighlights === 'function') updateMustMoveHighlights(this);
    }

    // Obligatory-move ordering: an obligatory piece may always be selected. A
    // NON-obligatory piece may be moved first only if, after a (single-die) move,
    // a die still remains for every pending obligatory piece — so you can move a
    // free piece first, but are then locked to the obligatory one(s).
    canSelectForMove(piece) {
        if (this.mustMovePieces.length === 0 || this.mustMovePieces.includes(piece)) return true;
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

        // Record human turns here
        if (!playerObj.isAI) {
            const preState = this.turnStartState || getGameState(this);
            const movePair = _pendingMoves.length > 0 ? _pendingMoves.slice() : null;
            recordTurnPosition(this, justFinished, source, movePair);
            if (movePair) {
                queryAndRecordContrastive(preState, movePair, justFinished, moveCounter);
            }
            clearMoveRecording();
        }

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

    // Flush any pending human move pair before the game result is recorded
    if (_pendingMoves.length > 0) {
        const playerObj = this.players.find(p => p.name === this.turn);
        const source = playerObj.isAI ? 'heuristic' : 'human';
        const preState = this.turnStartState || getGameState(this);
        const movePair = _pendingMoves.slice();
        recordTurnPosition(this, this.turn, source, movePair);
        if (!playerObj.isAI) {
            queryAndRecordContrastive(preState, movePair, this.turn, moveCounter);
        }
        clearMoveRecording();
    }

    // Notify backend of game result – this will flush all positions to disk
    notifyGameResult(winner, score);

    this.gameOver = true;
    console.log(`${winner} wins with a score of ${score}!`);
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
            const bx = r.x - RACK_PR, by = r.y - RACK_PR;
            const bw = r.cols * r.spacing + RACK_PR, bh = r.rows * r.spacing + RACK_PR + r.verticalPadding;
            if (x >= bx && x <= bx + bw && y >= by && y <= by + bh) return r;
        }
        return null;
    }

    // Drag-to-move, additive with click. A piece is selected by the pointerdown
    // (its normal handleClick) before dragstart fires, so drag just moves the
    // already-selected piece and drops it on the tile under the pointer, exactly
    // as if that tile had been clicked. Invalid drops snap the piece back.
    setupDragging(scene) {
        // NB: Phaser clears scene.input listeners on shutdown/restart, so re-wire
        // every time create() runs. The guard is reset on 'shutdown' (below) so a
        // New Game (scene.restart) or end-game (scene.start) keeps pieces draggable.
        if (scene._dragWired) return;
        scene._dragWired = true;
        scene.input.dragDistanceThreshold = 6;   // small moves stay clicks

        const onDragStart = (pointer, obj) => {
            const piece = obj.__piece; if (!piece) return;
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

            if (target) target.onClick();          // moves the selected piece, with full rule checks
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
        const buttonSize = 64; // Adjust the button size as needed
        this.undoButton = scene.add.image(config.width - DIE_2_POSITION, 85, 'leftWavyArrow')
            .setDisplaySize(buttonSize, buttonSize)
            .setInteractive()
            .on('pointerdown', () => {
                hideStackPicker();
                this.undoOneMove();   // one die / one move at a time
                clearMoveRecording();
            });

        const undoTooltip = makeHudTip(scene, this.undoButton.x, this.undoButton.y + 46, 'Undo');
        this.undoButton.on('pointerover', () => undoTooltip.show(true));
        this.undoButton.on('pointerout',  () => undoTooltip.show(false));
    }

    createSwitchTurnButton(scene) {
        const buttonSize = 64; // Adjust the button size as needed
        this.switchTurnButton = scene.add.image(config.width - DIE_1_POSITION, 85, 'rightWavyArrow')
            .setDisplaySize(buttonSize, buttonSize)
            .setInteractive()
            .on('pointerdown', () => {
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
    
        const switchTurnTooltip = makeHudTip(scene, this.switchTurnButton.x, this.switchTurnButton.y + 46, 'End turn');
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

    updateBlackPlayerAIStatus(isAI) {
        const blackPlayer = this.players.find(player => player.name === 'black');
        if (blackPlayer) {
            blackPlayer.isAI = isAI;
        }
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

        // "Play vs computer" now lives in the settings panel; make sure the game
        // reflects the persisted choice.
        this.game.updateBlackPlayerAIStatus(BLACK_IS_AI);
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
        const newGameButton = makeHudButton(this, 150, 52, 'New Game');
        newGameButton.on('pointerdown', () => {
            if (matchTracker && !matchTracker.over) return;
            this.showNewGameConfirmationModal();
        });
        if (inMatch) newGameButton.setHudVisible(false);

        // New Match sits where New Game would be during a match; starting one
        // mid-match asks for confirmation first.
        const newMatchButton = makeHudButton(this, 150, inMatch ? 52 : 104, 'New Match', { ghost: true });
        newMatchButton.on('pointerdown', () => {
            if (matchTracker && !matchTracker.over) {
                showConfirm('Abandon the current match and start a new one?', () => showMatchSetup());
            } else {
                showMatchSetup();
            }
        });

        const instructionsButton = makeHudButton(this, 150, inMatch ? 104 : 156, 'How to Play', { ghost: true });
        instructionsButton.on('pointerdown', () => { showInstructions(); });
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
        this.scoreText = this.add.text(24, this.sys.game.config.height - 24, '', {
            fontSize: '20px',
            fontFamily: HUD_FONT,
            color: THEME.bgInk
        }).setOrigin(0, 1);
        _themedRedraws.push(() => this.scoreText.setColor(THEME.bgInk));

        this.updateScoreText();

        // No-save counter: a quiet HUD line (not a boxed red warning), with the
        // draw offer as a standard ghost pill underneath it when it applies.
        this.impasseText = this.add.text(24, this.sys.game.config.height - 58, '', {
            fontSize: '21px', fontFamily: HUD_FONT, color: THEME.bgInk
        }).setOrigin(0, 1).setVisible(false).setAlpha(0.75);
        _themedRedraws.push(() => this.impasseText.setColor(THEME.bgInk));

        this.callDrawButton = makeHudButton(this, 85, this.sys.game.config.height - 115,
            'Call draw', { ghost: true });
        this.callDrawButton.setHudVisible(false);

            this.callDrawButton.on('pointerdown', () => {
                fetch(`${SERVER_URL}/call_draw`, { method: 'POST', credentials: 'include' })
                    .catch(e => console.warn('call_draw failed:', e));
                const g = gameInstance.scene.scenes[0].game;
                g.endGame('draw', null, g.turn);
            });

        this.checkInitialAIReady();

        // Call notifyStartGame when game is created
        notifyStartGame();

        // First casual game of a session: greet with a start screen, then the
        // coin flip (on Play) reveals the random starter.
        if (this._coinFlipOnStart && !matchTracker) {
            this._coinFlipOnStart = false;
            showWelcome(this.startingPlayer);
        }
        if (typeof updateTurnStatus === 'function') updateTurnStatus(this.game);
    }

    updateScoreText() {
            // During a match the line shows that match's running score/wins;
            // otherwise the session totals.
            const matchLine = matchScoreLine();
            if (matchLine) { this.scoreText.setText(matchLine); return; }

            // Single line, interpunct-separated, directly on the background.
            // Total score is signed (+ favours White), shown with a leader label.
            const total = scoreTracker.total_score;
            const totalStr = total === 0 ? '0'
                : `${total > 0 ? 'White' : 'Black'} +${Math.abs(total)}`;
            const sep = '  \u00B7  ';
            this.scoreText.setText(
                [`Games ${scoreTracker.games_played}`,
                 `White ${scoreTracker.white_wins}`,
                 `Black ${scoreTracker.black_wins}`,
                 `Draws ${scoreTracker.draws}`,
                 `Total score ${totalStr}`].join(sep)
            );
        }

    createRadioButton() {
        const circleX = this.sys.game.config.width - 350;
        const circleY = this.sys.game.config.height - 60;
        const textX = circleX + 30;
        const textY = circleY;
    
        const circle = this.add.circle(circleX, circleY, 15, BLACK_IS_AI ? THEME.accent : 0xD3D3D3)
            .setInteractive({ useHandCursor: true })
            .on('pointerdown', () => {
                BLACK_IS_AI = !BLACK_IS_AI;
                this.game.updateBlackPlayerAIStatus(BLACK_IS_AI);
                circle.setFillStyle(BLACK_IS_AI ? THEME.accent : 0xD3D3D3);
            });

        const text = this.add.text(textX, textY, 'Play Computer', {
            fontSize: '20px',
            fontFamily: HUD_FONT,
            color: THEME.bgInk
        }).setOrigin(0, 0.5);
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
            if (!this.game.gameOver && currentGameId) {
                fetch(`${SERVER_URL}/abort_game`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    credentials: 'include'
                }).catch(e => console.warn('abort_game failed:', e));
                currentGameId = null;
                moveCounter = 0;
                clearMoveRecording();
            }
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
        this.winner = data.winner;
        this.score = data.score;
        this.impasse_caller = data.impasse_caller;
        this.inMatch = data.inMatch;
        this.matchOver = data.matchOver;
    }

    create() {
        let message;
        if (this.winner === 'draw') {
            const caller = this.impasse_caller || 'A player';
            message = `${_cap(caller)} calls a draw!`;
        } else {
            message = `${_cap(this.winner)} wins the game with a score of ${this.score}`;
        }

        const abortAndClear = () => {
            if (currentGameId) {
                fetch(`${SERVER_URL}/abort_game`, { method: 'POST',
                    headers: { 'Content-Type': 'application/json' }, credentials: 'include' }).catch(() => {});
                currentGameId = null; moveCounter = 0; clearMoveRecording();
            }
        };
        const startGame = (starter) => { abortAndClear(); this.scene.start('MainGameScene', { startingPlayer: starter }); };
        updateTurnStatus('');   // the game is over: drop the turn/thinking pill

        // Card, headline, sub-line and pill buttons — the same vocabulary as the
        // welcome / match-setup overlays (this screen used to be bare text on the
        // background with square green and blue buttons).
        const card = (h) => {
            const w = 820, x = CENTER_X - w / 2, y = CENTER_Y - h / 2;
            const g = this.add.graphics();
            g.fillStyle(0x000000, 0.10); g.fillRoundedRect(x, y + 6, w, h, 22);
            g.fillStyle(0xffffff, 1);    g.fillRoundedRect(x, y, w, h, 22);
            return y;
        };
        const headline = (y, text, size) => this.add.text(CENTER_X, y, text, {
            fontSize: size + 'px', fontFamily: HUD_FONT, fontStyle: 'bold',
            color: HUD_INK, align: 'center', wordWrap: { width: 720 }
        }).setOrigin(0.5);
        const subline = (y, text, size) => this.add.text(CENTER_X, y, text, {
            fontSize: (size || 23) + 'px', fontFamily: HUD_FONT, color: '#5a6473',
            align: 'center', wordWrap: { width: 720 }
        }).setOrigin(0.5);
        const button = (x, y, label, ghost, cb) => {
            const b = makeHudButton(this, x, y, label, { ghost });
            b.on('pointerdown', cb);
            return b;
        };

        if (this.inMatch && matchTracker) {
            const m = matchTracker;
            if (this.matchOver) {
                const mScore = m.winner === 'white' ? m.whiteScore : m.blackScore;
                const mres = m.winner === 'draw' ? 'The match is a draw!'
                    : `${_cap(m.winner)} wins the match with a score of ${mScore}`;
                const top = card(340);
                subline(top + 58, message, 21);
                headline(top + 118, mres, 34);
                subline(top + 182,
                    `White ${m.whiteScore} (${m.whiteWins}W)   ·   Black ${m.blackScore} (${m.blackWins}W)   ·   ${m.gamesPlayed} games`, 21);
                button(CENTER_X - 105, top + 262, 'New Match', false,
                    () => { abortAndClear(); matchTracker = null; refreshSettingsMatchState(); showMatchSetup(); });
                button(CENTER_X + 105, top + 262, 'Single Game', true,
                    () => { matchTracker = null; refreshSettingsMatchState(); startGame('white'); });
            } else {
                const status = m.mode === 'race' ? `race to ${m.target}`
                    : `game ${m.gamesPlayed + 1} of ${m.target}`;
                const top = card(290);
                headline(top + 78, message, 34);
                subline(top + 142,
                    `White ${m.whiteScore} (${m.whiteWins}W)   ·   Black ${m.blackScore} (${m.blackWins}W)   ·   ${status}`, 21);
                button(CENTER_X, top + 218, 'Next Game', false,
                    () => startGame(matchStarterForGame(m.gamesPlayed)));
            }
            return;
        }

        // Casual single-game flow.
        const top = card(250);
        headline(top + 80, message, 36);
        button(CENTER_X - 105, top + 176, 'New Game', false, () => startGame(nextCasualStarter()));
        button(CENTER_X + 105, top + 176, 'New Match', true, () => { abortAndClear(); showMatchSetup(); });
    }

}



function calculateAverageScore() {
    if (scoreTracker.games_played === 0) {
        return 0; // Avoid division by zero
    }
    return scoreTracker.total_score / scoreTracker.games_played;
}

// Ensure these functions are defined outside of any class or method

function evaluateBoard(gameState) {
    return fetch(`${SERVER_URL}/evaluate_board`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(gameState)
    })
    .then(r => r.json())
    .then(data => { console.log('Evaluation:', data); return data; })
    .catch(error => { console.error('Error:', error); return null; });
}


function getAgentMoves(gameState) {
    // difficulty 1 = full strength (argmax); lower = more top-p sampling (weaker)
    gameState = Object.assign({}, gameState, { difficulty: getAIDifficulty() });
    console.log('Sending game state to agent:', gameState);
    return fetch(`${SERVER_URL}/select_moves`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        credentials: 'include',
        body: JSON.stringify(gameState)
    })
    .then(response => {
        console.log('Response status:', response.status);
        return response.json();
    })
    .then(data => {
        updateNoSaveDisplay();
        if (data.move) {
            console.log('Agent moves:', data.move);
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
        gameInstance.scene.scenes[0].hideThinkingIcon();
    });
}

function recordTurnPosition(game, player, source, movePair) {
    if (!RECORD_TRAINING_DATA) return;
    if (!currentGameId) {
        console.warn('No active game ID, skipping position recording');
        return;
    }
    const gameState = getGameState(game);
    moveCounter++;
    const playerObj = game.players.find(p => p.name === player);
    const gameStage = playerObj ? playerObj.getGamePhase() : 'unknown';
    fetch(`${SERVER_URL}/record_position`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            state: gameState,
            player: player,
            source: source,
            move_index: moveCounter,
            game_stage: gameStage,
            move_pair: movePair
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.message !== 'Position recorded') {
            console.warn('Failed to record position:', data);
        } else {
            console.log('Position recorded successfully');
        }
    })
    .catch(e => console.warn('record_position failed:', e));
}

async function queryAndRecordContrastive(preState, humanPair, player, moveIndex) {
    if (!RECORD_TRAINING_DATA) return;
    if (!currentGameId) return;
    try {
        const game = gameInstance.scene.scenes[0].game;
        const playerObj = game.players.find(p => p.name === player);
        const gameStage = playerObj ? playerObj.getGamePhase() : 'unknown';

        const response = await fetch(`${SERVER_URL}/query_agent_move`, {
            method: 'POST',
            credentials: 'include',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ state: preState, human_pair: humanPair }),
        });
        const data = await response.json();

        if (!data.differs) return;  // agent agreed, nothing to record

        await fetch(`${SERVER_URL}/record_contrastive_pair`, {
            method: 'POST',
            credentials: 'include',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                state: preState,
                player: player,
                game_stage: gameStage,
                move_index: moveIndex,
                human_pair: humanPair,
                agent_pair: data.agent_pair,
                agent_score: data.agent_score,
            }),
        });
        console.log('Contrastive pair recorded (agent disagreed)');
    } catch(e) {
        console.warn('queryAndRecordContrastive failed:', e);
    }
}

function notifyStartGame() {
    console.log('Starting new game, notifying backend...');
    fetch(`${SERVER_URL}/start_game`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
    })
    .then(response => response.json())
    .then(data => {
        currentGameId = data.game_id;
        moveCounter = 0;
        clearMoveRecording();
        console.log('Game started with ID:', currentGameId);
    })
    .catch(e => console.warn('start_game failed:', e));
}

function notifyGameResult(winner, score) {
    if (!currentGameId) {
        console.warn('No active game ID, cannot record result');
        return;
    }
    console.log(`Recording game result: ${winner} wins with score ${score}`);
    fetch(`${SERVER_URL}/record_game_result`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            winner: winner,
            score: score
        })
    })
    .then(() => {
        // Clear game ID after successful recording
        currentGameId = null;
        moveCounter = 0;
        clearMoveRecording();
    })
    .catch(e => console.warn('record_game_result failed:', e));
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

    function processMove(move, callback) {
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

// Page unload handler to abort game
window.addEventListener('beforeunload', function() {
    if (currentGameId) {
        fetch(`${SERVER_URL}/abort_game`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            keepalive: true
        }).catch(e => console.warn('abort_game failed:', e));
    }
});


const config = {
    type: Phaser.AUTO,
    width: 1800,
    height: 1200,
    backgroundColor: BACKGROUND_COLOR,
    scale: {
        mode: Phaser.Scale.FIT, 
        autoCenter: Phaser.Scale.CENTER_BOTH
    },
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
['touchstart', 'touchmove', 'touchend', 'touchcancel'].forEach(type => {
    window.addEventListener(type, () => {
        const tm = gameInstance.input && gameInstance.input.touch;
        if (tm) tm.capture = (type === 'touchend' || type === 'touchcancel');
    }, { capture: true, passive: true });
});