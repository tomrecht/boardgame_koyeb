/* Auto-detect environment for Koyeb or Localhost */
const IS_LOCAL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';

// This one variable now handles everything
const SERVER_URL = IS_LOCAL 
    ? 'http://localhost:10000' 
    : window.location.origin;

const DEBUG_MODE = false; 
const WHITE_IS_AI = false;
let BLACK_IS_AI = true;

const PIECE_RADIUS_BASE = 20; 
const TILE_RADIUS_STEP = 60; 
const CENTER_X = 900;
const CENTER_Y = 640; 
const HOME_TILE_RADIUS = TILE_RADIUS_STEP * 1.5; 

const TOTAL_PIECES = 12;

// No-save draw rule: once both players are in midgame, if this many full
// rounds (a round = white turn + black turn) pass with no save by either
// player, either player may call a draw. Easily changed. Must match game.py.
const NO_SAVE_TURNS_FOR_DRAW = 10;

const DIE_1_POSITION= 400;
const DIE_2_POSITION = 500;

const BACKGROUND_COLOR = 0xfdf6e3; // Cream
const colorFirstDie = 0x40E0D0; // Turquoise
const colorSecondDie = 0xFFC0CB; 
const colorSum = 0xFFFF00; // Yellow
const FONT_FAMILY = 'Crimson Text';


const scoreTracker = {
    games_played: 0,
    white_wins: 0,
    black_wins: 0,
    draws: 0,
    total_score: 0
    };

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
// target: [ring, sector] | 'save' | 0  (0 = block-save)
// die: numeric die value used
function pushHumanMove(pieceColorNumber, target, die) {
    _pendingMoves.push([pieceColorNumber, target, die]);
}

// ── HIDDEN DEBUG TOGGLE ─────────────────────────────────────────────────
window.debugMode = false;
(function() {
    var tapCount = 0;
    var tapTimer = null;
    document.addEventListener('keydown', function(e) {
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

function updateNoSaveDisplay() {
    const scene = gameInstance && gameInstance.scene && gameInstance.scene.scenes && gameInstance.scene.scenes[0];
    if (!scene || !scene.game || !scene.impasseText) return;
    const game = scene.game;

    // Show the counter once both players are past the opening, OR always when
    // in sandbox (which doesn't track game stages).
    const show = window.setupMode || game.bothInMidgame();
    if (!show) {
        scene.impasseText.setVisible(false);
        scene.callDrawButton.setVisible(false);
        return;
    }

    scene.impasseText
        .setText(`Turns with no save: ${game.noSaveTurns}`)
        .setVisible(true);

    // Offer the button on a human player's turn when callable.
    const humanCanCall = game.drawCallable && game.currentPlayerIsHuman() && !game.gameOver;
    if (humanCanCall) {
        const b = scene.impasseText.getBounds();
        scene.callDrawButton.setPosition(b.right + 10, b.bottom).setVisible(true);
    } else {
        scene.callDrawButton.setVisible(false);
    }
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
        if (this.game.selectedPiece && this.game.selectedPiece !== this) return;
        if (this.game.dice[0].used && this.game.dice[1].used) return;
        if (this.player !== this.game.turn) return; 
        if (this.rack && this.rack.type === 'saved') return;
        if (this.rack && this.rack.type === 'unentered' && this.rack.pieces[0] !== this) return;
        if (this.game.mustMovePieces.length > 0 && !this.game.mustMovePieces.includes(this)) {
            console.log("Must move a piece from the mustMovePieces list");
            return false;
        }
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
        if (this.rack && this.rack.type === 'saved') return;
        if (this.rack && this.rack.type === 'unentered' && this.rack.pieces[0] !== this) return;
        if (this.player === this.game.turn && this.game.mustMovePieces.length > 0 && !this.game.mustMovePieces.includes(this)) {
            console.log("Must move a piece from the mustMovePieces list");
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
                this.game.selectedPiece = this;
                this.reachableTiles = this.game.getReachableTilesByDice(this);
                this.highlightReachableTiles();
            } else {
                this.game.unhighlightAllTiles();
                this.game.selectedPiece = null;
                this.reachableTiles = null;
            }
        }
    }

    handleDoubleClick() {
        if (this.currentTile.type === 'save') {
            this.save(); // Save the piece if it can be saved
        }
        
        // save opponent's blocking piece, unless you're in the opening or have a captured piece
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
            console.log('Saving opponent block')
            const savedRack = this.color === 0xffffff ? this.game.whiteSavedRack : this.game.blackSavedRack;

            this.currentTile.pieces.forEach(piece => piece.moveToRack(savedRack));
            this.game.registerSave();   // no-save streak resets immediately
            this.game.dice.forEach(die => die.setUsed())

            // Record human block-save as a complete move pair (matches agent encoding)
            if (this.game.turn === 'white') {
                const rep = [this.player, this.number];
                _pendingMoves = [[rep, 0, 0], [rep, 0, 0]];
            }

            // Check for the endgame condition
            const player = this.color === 0xffffff ? this.game.players[0] : this.game.players[1];
            this.game.checkEndgame(player);

            // Check for the win condition
            this.game.checkWinCondition();
        }
    }

    setPosition(x, y) {
        this.x = x;
        this.y = y;
        this.circle.setPosition(x, y);
        if (this.text) {
            this.text.setPosition(x, y);
        }
    }
    
    setSize(size) {
        this.radius = size;
        this.circle.setRadius(size);
        if (this.text) {
            this.text.setFontSize(`${size * 1.5}px`);
        }
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
        this.game.selectedPiece = this;
        this.isSelected = true;
    }

    moveToRack(rack, addToFront = false) {
        this.rack = rack;
        this.x = rack.nextX();
        this.y = rack.nextY();
        this.setSize(PIECE_RADIUS_BASE);
        this.circle.setPosition(this.x, this.y);
        if (this.text) {
            this.text.setPosition(this.x, this.y);
        }
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
        if (this.isSelected || this.isHovered) {
            this.circle.fillColor = this.color === 0xffffff ? 0x90ee90 : 0xee82ee;
        } else {
            this.circle.fillColor = this.originalColor;
        }
        this.circle.setStrokeStyle(2, this.borderColor);
    }


    highlightReachableTiles() {

        const reachableTiles = this.reachableTiles;
        if (!reachableTiles) return;

        const { reachableByFirstDie, reachableBySecondDie, reachableBySum } = reachableTiles;

    
        // Highlight tiles and set the color attribute
        reachableByFirstDie.forEach(tile => {
            tile.reachableColor = colorFirstDie;
            tile.highlight();
        });
    
        reachableBySecondDie.forEach(tile => {
            tile.reachableColor = colorSecondDie;
            tile.highlight();
        });
    
        reachableBySum.forEach(tile => {
            tile.reachableColor = colorSum;
            tile.highlight();
        });
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
            let dieToUse = dice.find(die => die.value === saveTileNumber);


            if (!dieToUse && this.number > 6) {
                // If player is in the endgame they can save unnumbered pieces with higher die rolls
                const isEndgame = player.getGamePhase() === 'endgame';
                console.log(`Is player ${player.name} in endgame: ${isEndgame}`);

                if (isEndgame && !this.game.isHigherNumberedGoalOccupied(player, saveTileNumber)) {
                    console.log(`No higher-numbered goal occupied for player ${player.name}`);
                    // Find the highest die value that is greater than the save tile number
                    dieToUse = dice.filter(die => die.value > saveTileNumber)
                                   .sort((a, b) => b.value - a.value)[0];
                    console.log(`Die to use after endgame check: ${dieToUse ? dieToUse.value : 'None'}`);
                }
            }

            if (dieToUse) {
                console.log(`Using die ${dieToUse.value} to save piece ${this.number}`);
                const dieValue = dieToUse.value;
                // Use the corresponding die
                dieToUse.setUsed();

                // Move the piece to the saved rack
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
        this.circle = this.scene.add.circle(this.x, this.y, this.radius, this.color)
            .setInteractive()
            .on('pointerover', () => this.onHover())
            .on('pointerout', () => this.onOut())
            .on('pointerdown', (pointer) => this.handleClick(pointer));

        // Debug-mode tooltip: show the number of unnumbered pieces (numbered
        // pieces already display their number on the board).
        this.circle
            .on('pointerover', () => { if (window.debugMode && this.number > 6) showDebugTip(`${this.player} #${this.number}`); })
            .on('pointermove', () => { if (window.debugMode && this.number > 6) showDebugTip(`${this.player} #${this.number}`); })
            .on('pointerout',  () => { hideDebugTip(); });

        this.circle.setStrokeStyle(2, this.borderColor);

        if (this.number <= 6 || DEBUG_MODE) {
            this.text = this.scene.add.text(this.x, this.y, this.number, {
                fontSize: `${this.radius * 1.5}px`,
                color: `#${this.textColor.toString(16)}`
            }).setOrigin(0.5, 0.5);
        } else {
            this.text = null;
        }
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
            this.highlightColor = 0xadd8e6; // Light blue
            this.reachableColor = null;
            this.lastClickTime = null;
    
            this.lineColor = 0x000000;
            this.graphics = scene.add.graphics();

            switch (type) {
                case "home":
                    this.fillColor = 0xffff00;
                    break;
                case "save":
                    this.fillColor = 0x00ff00;
                    break;
                case "nogo":
                    this.fillColor = ring === 7 ? BACKGROUND_COLOR : 0x000000; 
                    this.lineColor = ring === 7 ? BACKGROUND_COLOR : 0x000000; // No border for 7th ring nogo tiles
                    break;
                case "field":
                    this.fillColor = 0xffffff;
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
        const textRadius = radius + 20; // Adjust the offset distance as needed
        const x = CENTER_X + textRadius * Math.cos(angle);
        const y = CENTER_Y + textRadius * Math.sin(angle);
        const text = this.scene.add.text(x, y, number.toString(), {
            fontSize: '36px',
            color: '#000',
            fontStyle: 'bold'
        }).setOrigin(0.5);
        

        
        text.setAngle(0);
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
            if (window.setupMode) {
                if (_setupSelected && this.type !== 'nogo') {
                    _setupPlaceOnTile(_setupSelected, this);
                    _setupClearSelection();
                }
                return;
            }
            if (this.game.gameOver) return; 
            if (this.game.selectedPiece && this.type !== "nogo") {
                const piece = this.game.selectedPiece;
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
                
                piece.setPosition(x, y); // Set piece position
                piece.setSize(PIECE_RADIUS_BASE); // Set piece size
            });
        } else {
            const innerRadius = TILE_RADIUS_STEP * (this.ring - 1) + HOME_TILE_RADIUS;
            const outerRadius = TILE_RADIUS_STEP * this.ring + HOME_TILE_RADIUS;
            const arcLength = (this.endAngle - this.startAngle) * outerRadius; // Arc length of the segment
            
            let padding = 10;
    
            let maxPieceSize = PIECE_RADIUS_BASE;
            let piecesPerArc = Math.floor(arcLength / (2 * maxPieceSize + padding));
    
            // Check if resizing is necessary
            if (this.pieces.length > piecesPerArc) {
                while (this.pieces.length > piecesPerArc && maxPieceSize > 10) { // Adjust minimum size threshold
                    maxPieceSize -= 1;
                    padding = 10 * (maxPieceSize / PIECE_RADIUS_BASE); // Adjust padding proportionally
                    piecesPerArc = Math.floor(arcLength / (2 * maxPieceSize + padding));
                }
            }
    
            if (this.pieces.length > piecesPerArc) {
                // If pieces are still too many, arrange in two rows
   
                const rowCount = 2; // Number of rows
                const piecesPerRow = Math.ceil(this.pieces.length / rowCount);
    
                let rowRadius = (innerRadius + outerRadius) / 2;
    
                // Adjust piece size for two-row arrangement
                while (true) {
                    const arcPieceCount = Math.floor(arcLength / (2 * maxPieceSize + padding));
                    if (arcPieceCount >= piecesPerRow || maxPieceSize <= 10) break; // Adjust minimum size threshold
                    maxPieceSize -= 1;
                    
                }
    
                this.pieces.forEach((piece, index) => {
                    const row = Math.floor(index / piecesPerRow);
                    const angularStep = (this.endAngle - this.startAngle) / piecesPerRow;
                    const angle = this.startAngle + angularStep * (index % piecesPerRow + 0.5);
                    const radius = row === 0 ? rowRadius - maxPieceSize - padding : rowRadius + maxPieceSize + padding;
    
                    const x = CENTER_X + radius * Math.cos(angle);
                    const y = CENTER_Y + radius * Math.sin(angle);
    
                    piece.setPosition(x, y);
                    piece.setSize(maxPieceSize); // Set piece size
                });
            } else {
                // Single row arrangement
                const angularStep = (this.endAngle - this.startAngle) / this.pieces.length; // Angular step between pieces
                this.pieces.forEach((piece, index) => {
                    const angle = this.startAngle + angularStep * (index + 0.5);
                    const x = CENTER_X + (innerRadius + outerRadius) / 2 * Math.cos(angle);
                    const y = CENTER_Y + (innerRadius + outerRadius) / 2 * Math.sin(angle);
                    
                    piece.setPosition(x, y);
                    piece.setSize(maxPieceSize); // Update the size of the piece
                });
            }
        }
    }
    
    
    drawTile() {
        
        this.graphics.clear();
        this.graphics.lineStyle(1, this.lineColor, 1);
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
            this.addNumberText(this.number, (this.startAngle + this.endAngle) / 2, this.outerRadius);
        }
        }
    }
    
    
    
    
}

class Rack {
    constructor(scene, x, y, color, type, rows = 5) {
        this.scene = scene;
        this.x = x;
        this.y = y;
        this.color = color;
        this.type = type;
        this.pieces = [];
        this.rows = rows;
        this.cols = 3;
        this.spacing = PIECE_RADIUS_BASE * 2 + 10;
        this.verticalPadding = 20;
        this.horizontalPadding = 15;
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
                    // Force size reset when on rack
            if (this.type === 'unentered' || this.type === 'saved') {
                piece.setSize(PIECE_RADIUS_BASE);
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
        this.background.fillStyle(0x008000, 1);
        this.background.fillRect(this.x - PIECE_RADIUS_BASE, this.y - PIECE_RADIUS_BASE, this.cols * this.spacing + PIECE_RADIUS_BASE, this.rows * this.spacing + PIECE_RADIUS_BASE + this.verticalPadding);
        this.background.lineStyle(2, 0x000000, 1);
        this.background.strokeRect(this.x - PIECE_RADIUS_BASE, this.y - PIECE_RADIUS_BASE, this.cols * this.spacing + PIECE_RADIUS_BASE, this.rows * this.spacing + PIECE_RADIUS_BASE + this.verticalPadding);
    }
}



class Die {
    constructor(scene, x, y, isFirstDie) {
        this.scene = scene;
        this.value = Phaser.Math.Between(1, 6);
        this.x = x;
        this.y = y;
        this.size = 80; // Smaller size
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
        this.graphics.fillStyle(dieColor, 1);
        const borderColor = this.isFirstDie ? colorFirstDie : colorSecondDie;
        this.graphics.lineStyle(5, borderColor, 1);
        this.graphics.fillRect(this.x, this.y, this.size, this.size);
        this.graphics.strokeRect(this.x, this.y, this.size, this.size);

        const dotSize = 8; // Adjusted dot size
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
        this.dice = [new Die(scene, DIE_1_POSITION, 50, true), new Die(scene, DIE_2_POSITION, 50, false)];
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
        this.whiteUnenteredRack = new Rack(scene, 75, 150, 'white', 'unentered');
        this.whiteSavedRack = new Rack(scene, 75, 600, 'white', 'saved');
        this.blackUnenteredRack = new Rack(scene, 1545, 150, 'black', 'unentered');
        this.blackSavedRack = new Rack(scene, 1545, 600, 'black', 'saved');

        this.confirmationModal = null;

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
            piece.x = this.whiteUnenteredRack.nextX();
            piece.y = this.whiteUnenteredRack.nextY();
            piece.circle.setPosition(piece.x, piece.y);
            if (piece.text) {
                piece.text.setPosition(piece.x, piece.y);
            }
            this.whiteUnenteredRack.addPiece(piece);
        });

        blackPieces.forEach(piece => {
            piece.x = this.blackUnenteredRack.nextX();
            piece.y = this.blackUnenteredRack.nextY();
            piece.circle.setPosition(piece.x, piece.y);
            if (piece.text) {
                piece.text.setPosition(piece.x, piece.y);
            }
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
    
        const dice = this.dice.filter(die => !die.used);
        const diceValues = dice.map(die => die.value);
    
        if (diceValues.length === 0) {
            console.log('No available dice');
            return null; // No available dice
        }
    
        const reachableByFirstDie = this.getReachableTiles(piece.currentTile, diceValues[0]);
        const reachableBySecondDie = diceValues.length > 1 ? this.getReachableTiles(piece.currentTile, diceValues[1]) : [];
        let reachableBySum = this.getReachableTiles(piece.currentTile, diceValues[0] + (diceValues[1] || 0));
    
        const homeTile = this.tiles.find(tile => tile.type === 'home');
        if (homeTile.pieces.filter(p => p.color === piece.color).length > 1) {
            console.log('Player has more than one captured piece');
            reachableBySum = [];
        }

        // Filter reachableByFirstDie and reachableBySecondDie based on piece.reachableTiles
        if (piece.reachableTiles)  {

            const reachableTilesSet = new Set(piece.reachableTiles.reachableBySum); // Assuming each tile has a unique id
    
            const filterTiles = (tiles) => tiles.filter(tile => reachableTilesSet.has(tile));
    
            const filteredReachableByFirstDie = filterTiles(reachableByFirstDie);
            const filteredReachableBySecondDie = filterTiles(reachableBySecondDie);
            const filteredReachableBySum = filterTiles(reachableBySum);
    
            return {
                reachableByFirstDie: filteredReachableByFirstDie,
                reachableBySecondDie: filteredReachableBySecondDie,
                reachableBySum: filteredReachableBySum
            };
        }
        return {
            reachableByFirstDie: reachableByFirstDie,
            reachableBySecondDie: reachableBySecondDie,
            reachableBySum: reachableBySum
        };
    }
    
    
    
    movePiece(piece, targetTile, getReachableTiles = false) {
        if (!piece || !targetTile) return false;

        if (this.mustMovePieces.length > 0 && !this.mustMovePieces.includes(piece)) {
            console.log("Must move a piece from the mustMovePieces list");
            return false;
        }
    
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
            
            

            if (targetTile.type === 'field' && targetTile.pieces.length === 1 && targetTile.pieces[0].color !== piece.color) {
                this.capturePiece(targetTile.pieces[0]); // Capture the opposing piece
            }
            
            

            const dice = this.dice.filter(die => !die.used);

            // check en route capture
            if (dice.length > 1 && reachableBySum.includes(targetTile)) {
                this.checkEnRouteCapture(piece, targetTile);
            }
    

            

            piece.move(targetTile);
    
            const homeTile = this.tiles.find(tile => tile.type === 'home');
            if (homeTile.pieces.includes(piece)) homeTile.removePiece(piece);

            if (reachableByFirstDie.includes(targetTile)) {
                dice[0].setUsed();
            } else if (reachableBySecondDie.includes(targetTile)) {
                dice[1].setUsed();
            } else {

                dice[0].setUsed();
                if (dice.length > 1) dice[1].setUsed();
            }
    
            if (!this.movedOnce) this.movedOnce = true;

                    // If the moved piece was in the mustMovePieces list, remove it
            if (this.mustMovePieces.includes(piece)) {
            this.mustMovePieces = this.mustMovePieces.filter(p => p !== piece);
            }

            refreshEvalReadout();  // update on-board eval after the move settles
            return true;
        }
    
        console.log('Target tile is not reachable by the available dice rolls');
        return false;
    }
    
    capturePiece(piece) {
        const homeTile = this.tiles.find(tile => tile.type === 'home');
        if (homeTile) {
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
        this.updateMovablePieces();
        this.pieces.forEach(piece => piece.reachableTiles = null);
        this.state = this.captureState();
        this.applyLastPieceRule();
        this.turnStartState = getGameState(this);  
        
        if (window.showEvals) refreshEvalReadout();

        // No-save draw display is owned by the frontend now; update directly.
        updateNoSaveDisplay();

        // If it's the agent's turn, ask the backend for its moves.
        const currentPlayerObject = this.players.find(player => player.name === this.turn);
        if (currentPlayerObject && currentPlayerObject.isAI) {
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
    this.scene.updateScoreText();
    this.scene.scene.start('EndGameScene', { winner: winner, score: score, impasse_caller: impasse_caller });
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
            halfTurnsSinceRound: this._halfTurnsSinceRound
        };

        return state;
    }
    

    restoreState() {
        const state = this.state;
        if (!state) {
            console.error('No state to restore');
            return;
        }
    
        // Clear existing graphics
        this.tiles.forEach(tile => tile.graphics.clear());
        this.pieces.forEach(piece => {
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
                piece.setPosition(pieceState.x, pieceState.y);
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
    
    
    createUndoButton(scene) {
        const buttonSize = 64; // Adjust the button size as needed
        this.undoButton = scene.add.image(config.width - DIE_2_POSITION, 85, 'leftWavyArrow')
            .setDisplaySize(buttonSize, buttonSize)
            .setInteractive()
            .on('pointerdown', () => {
                this.restoreState();
                clearMoveRecording();
            });
    
        // Add tooltip for undo button
        const undoTooltip = scene.add.text(this.undoButton.x, this.undoButton.y, 'UNDO', {
            fontSize: '22px',
            fontFamily: 'Arial, sans-serif',
            fill: '#000000',
            backgroundColor: 'rgba(0, 0, 0, 0)'
        }).setOrigin(0.5).setVisible(false);
    
        this.undoButton.on('pointerover', () => {
            undoTooltip.setVisible(true);
        });
    
        this.undoButton.on('pointerout', () => {
            undoTooltip.setVisible(false);
        });
    }

    createSwitchTurnButton(scene) {
        const buttonSize = 64; // Adjust the button size as needed
        this.switchTurnButton = scene.add.image(config.width - DIE_1_POSITION, 85, 'rightWavyArrow')
            .setDisplaySize(buttonSize, buttonSize)
            .setInteractive()
            .on('pointerdown', () => {
                // Only the human whose turn it is may end the turn.
                if (this.gameOver || !this.currentPlayerIsHuman()) return;
                if (this.dice.some(die => !die.used)) {
                    this.showConfirmationModal();
                } else {
                    this.switchTurn();
                }
            });
    
        // Add tooltip for switch turn button
        const switchTurnTooltip = scene.add.text(this.switchTurnButton.x, this.switchTurnButton.y, 'END TURN', {
            fontSize: '22px',
            fontFamily: 'Arial, sans-serif',
            fill: '#000000',
            backgroundColor: 'rgba(0, 0, 0, 0)'
        }).setOrigin(0.5).setVisible(false);
    
        this.switchTurnButton.on('pointerover', () => {
            switchTurnTooltip.setVisible(true);
        });
    
        this.switchTurnButton.on('pointerout', () => {
            switchTurnTooltip.setVisible(false);
        });
    }
    
    showConfirmationModal() {
        if (this.confirmationModal) {
            this.confirmationModal.destroy(true);
        }
        const modalWidth = 400;
        const modalHeight = 200;
        const modalX = CENTER_X - modalWidth / 2;
        const modalY = CENTER_Y - modalHeight / 2;

        this.confirmationModal = this.scene.add.container(0, 0); // Create a container for modal elements

        const modalBackground = this.scene.add.graphics();
        modalBackground.fillStyle(0xffffff, 1);
        modalBackground.fillRect(modalX, modalY, modalWidth, modalHeight);
        modalBackground.lineStyle(2, 0x000000, 1);
        modalBackground.strokeRect(modalX, modalY, modalWidth, modalHeight);
        this.confirmationModal.add(modalBackground);

        const text = this.scene.add.text(CENTER_X, CENTER_Y - 40, 'End your turn without using both dice?', {
            fontSize: '22px',
            fontFamily: FONT_FAMILY,
            color: '#000000',
            wordWrap: { width: modalWidth - 40 },
            align: 'center'
        }).setOrigin(0.5);
        this.confirmationModal.add(text);

        const confirmButton = this.scene.add.text(CENTER_X - 60, CENTER_Y + 40, 'Yes', {
            fontSize: '28px',
            fontFamily: FONT_FAMILY,
            backgroundColor: '#00ff00',
            padding: { x: 20, y: 10 },
            borderColor: '#000',
            borderWidth: 1.5,
            borderRadius: 3.75
        }).setOrigin(0.5).setInteractive();
        this.confirmationModal.add(confirmButton);

        const cancelButton = this.scene.add.text(CENTER_X + 60, CENTER_Y + 40, 'No', {
            fontSize: '28px',
            fontFamily: FONT_FAMILY,
            backgroundColor: '#ff0000',
            padding: { x: 20, y: 10 },
            borderColor: '#000',
            borderWidth: 1.5,
            borderRadius: 3.75
        }).setOrigin(0.5).setInteractive();
        this.confirmationModal.add(cancelButton);

        confirmButton.on('pointerdown', () => {
            this.hideConfirmationModal();
            this.switchTurn();
        });

        cancelButton.on('pointerdown', () => {
            this.hideConfirmationModal();
        });
    }

    hideConfirmationModal() {
        if (this.confirmationModal) {
            this.confirmationModal.destroy(true);
            this.confirmationModal = null;
        }
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
        this.startingPlayer = data.startingPlayer || 'white';
    }

    preload() {
        this.load.image('leftWavyArrow', 'assets/left-arrow.png');
        this.load.image('rightWavyArrow', 'assets/right-arrow.png');
        this.load.image('thinkingIcon', 'assets/thinking.png'); 

    }

    create() {
        const debugMode = false; // Set this to false to disable debug mode

        this.game = new Game(this, this.startingPlayer, debugMode);

        this.createRadioButton();
        this.createEvalButton();

        const iconSize = 192;
        const xPosition = this.sys.game.config.width - iconSize / 2 - 100; 
        const yPosition = this.sys.game.config.height - iconSize / 2 - 100; 
        this.thinkingIcon = this.add.image(xPosition, yPosition, 'thinkingIcon')
        .setDisplaySize(iconSize, iconSize) 
        .setAlpha(0)
        .setVisible(true);


        // Add instructions button
        const instructionsButton = this.add.text(150, 50, 'How to Play', {
            fontSize: '24px',
            fontFamily: FONT_FAMILY,
            backgroundColor: '#87CEEB',
            padding: { x: 15, y: 7.5 },
            borderColor: '#000',
            borderWidth: 1.5,
            borderRadius: 3.75
        }).setOrigin(0.5).setInteractive();

        instructionsButton.on('pointerdown', () => {
            this.scene.switch('InstructionsScene');
        });

                // Add new game button under the instructions button
                const newGameButton = this.add.text(150, 100, 'New Game', {
                    fontSize: '24px',
                    fontFamily: FONT_FAMILY,
                    backgroundColor: '#87CEEB',
                    padding: { x: 15, y: 7.5 },
                    borderColor: '#000',
                    borderWidth: 1.5,
                    borderRadius: 3.75
                }).setOrigin(0.5).setInteractive();
        
                newGameButton.on('pointerdown', () => {
                    this.showNewGameConfirmationModal();
                });

        // Add save game state button
        if(DEBUG_MODE) {
        const saveGameStateButton = this.add.text(300, 100, 'Save Game', {
            fontSize: '24px',
            fontFamily: FONT_FAMILY,
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
        this.scoreText = this.add.text(20, this.sys.game.config.height - 100, '', {
            fontSize: '24px',
            fontFamily: FONT_FAMILY,
            color: '#000',
            backgroundColor: '#ffffff',
            padding: { x: 10, y: 5 },
            borderColor: '#000',
            borderWidth: 1.5,
            borderRadius: 3.75
        }).setOrigin(0, 1);

        this.updateScoreText();

        // In MainGameScene.create(), replace the old impasseText and callDrawButton with:
        this.impasseText = this.add.text(20, this.sys.game.config.height - 280, '', {
            fontSize: '24px', fontFamily: FONT_FAMILY, color: '#a00',
            backgroundColor: '#fff', padding: { x: 8, y: 4 },
            borderColor: '#000', borderWidth: 1.5, borderRadius: 3.75
        }).setOrigin(0, 1).setVisible(false);

        this.callDrawButton = this.add.text(0, 0, 'Call draw', {
            fontSize: '24px', fontFamily: FONT_FAMILY, color: '#fff',
            backgroundColor: '#a00', padding: { x: 10, y: 5 },
            borderColor: '#000', borderWidth: 1.5, borderRadius: 3.75
        }).setOrigin(0, 1).setVisible(false).setInteractive({ useHandCursor: true });
        
        // ... rest of event listeners stay the same
            this.callDrawButton.on('pointerdown', () => {
                fetch(`${SERVER_URL}/call_draw`, { method: 'POST', credentials: 'include' })
                    .catch(e => console.warn('call_draw failed:', e));
                const g = gameInstance.scene.scenes[0].game;
                g.endGame('draw', null, g.turn);
            });

        this.checkInitialAIReady();

        // Call notifyStartGame when game is created
        notifyStartGame();

    }

    updateScoreText() {

            const averageScore = calculateAverageScore();
            const displayValue = Math.abs(averageScore).toFixed(2);

            let finalScoreLine;
            if (averageScore === 0) {
                // If the score is 0, just show the number
                finalScoreLine = `Average Score: ${displayValue}`;
            } else {
                // Otherwise, show the leader label
                const leaderLabel = averageScore > 0 ? "White" : "Black";
                finalScoreLine = `Average Score: ${leaderLabel}: ${displayValue}`;
            }

            this.scoreText.setText(
                `Games Played: ${scoreTracker.games_played}\n` +
                `White Wins: ${scoreTracker.white_wins}\n` +
                `Black Wins: ${scoreTracker.black_wins}\n` +
                `Draws: ${scoreTracker.draws}\n` +
                finalScoreLine
            );
        }

    createRadioButton() {
        const circleX = this.sys.game.config.width - 350;
        const circleY = this.sys.game.config.height - 60;
        const textX = circleX + 30;
        const textY = circleY;
    
        const circle = this.add.circle(circleX, circleY, 15, BLACK_IS_AI ? 0x87CEEB : 0xD3D3D3)
            .setInteractive()
            .on('pointerdown', () => {
                BLACK_IS_AI = !BLACK_IS_AI;
                this.game.updateBlackPlayerAIStatus(BLACK_IS_AI);
                circle.setFillStyle(BLACK_IS_AI ? 0x87CEEB : 0xD3D3D3);
            });
    
        const text = this.add.text(textX, textY, 'Play Computer', {
            fontSize: '24px',
            fontFamily: FONT_FAMILY,
            color: '#000'
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
            fontFamily: FONT_FAMILY,
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
    
    showNewGameConfirmationModal() {
        if (this.confirmationModal) {
            this.confirmationModal.destroy(true);
        }
        const modalWidth = 400;
        const modalHeight = 200;
        const modalX = CENTER_X - modalWidth / 2;
        const modalY = CENTER_Y - modalHeight / 2;

        this.confirmationModal = this.add.container(0, 0); // Create a container for modal elements

        const modalBackground = this.add.graphics();
        modalBackground.fillStyle(0xffffff, 1);
        modalBackground.fillRect(modalX, modalY, modalWidth, modalHeight);
        modalBackground.lineStyle(2, 0x000000, 1);
        modalBackground.strokeRect(modalX, modalY, modalWidth, modalHeight);
        this.confirmationModal.add(modalBackground);

        const text = this.add.text(CENTER_X, CENTER_Y - 40, 'Start a new game?', {
            fontSize: '22px',
            fontFamily: FONT_FAMILY,
            color: '#000000',
            wordWrap: { width: modalWidth - 40 },
            align: 'center'
        }).setOrigin(0.5);
        this.confirmationModal.add(text);

        const confirmButton = this.add.text(CENTER_X - 60, CENTER_Y + 40, 'Yes', {
            fontSize: '28px',
            fontFamily: FONT_FAMILY,
            backgroundColor: '#00ff00',
            padding: { x: 20, y: 10 },
            borderColor: '#000',
            borderWidth: 1.5,
            borderRadius: 3.75
        }).setOrigin(0.5).setInteractive();
        this.confirmationModal.add(confirmButton);

        const cancelButton = this.add.text(CENTER_X + 60, CENTER_Y + 40, 'No', {
            fontSize: '28px',
            fontFamily: FONT_FAMILY,
            backgroundColor: '#ff0000',
            padding: { x: 20, y: 10 },
            borderColor: '#000',
            borderWidth: 1.5,
            borderRadius: 3.75
        }).setOrigin(0.5).setInteractive();
        this.confirmationModal.add(cancelButton);

        confirmButton.on('pointerdown', () => {
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
            this.startingPlayer = (scoreTracker.games_played % 2 === 0) ? 'white' : 'black';
            this.scene.restart();
            this.hideConfirmationModal();
        });

        cancelButton.on('pointerdown', () => {
            this.hideConfirmationModal();
        });
    }

    hideConfirmationModal() {
        if (this.confirmationModal) {
            this.confirmationModal.destroy(true);
            this.confirmationModal = null;
        }
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
    }

    create() {
        let message;
        if (this.winner === 'draw') {
        const caller = this.impasse_caller || 'A player';
        const formattedCaller = caller.charAt(0).toUpperCase() + caller.slice(1);
        message = `${formattedCaller} calls a draw!`;
    } else {
        message = `${this.winner} wins with a score of ${this.score}!`;
    }
        this.add.text(CENTER_X, CENTER_Y - 50, message, {
            fontSize: '48px',
            fontFamily: FONT_FAMILY,
            color: '#ff0000'
        }).setOrigin(0.5);
        const restartButton = this.add.text(CENTER_X, CENTER_Y + 50, 'Restart', {
            fontSize: '32px',
            fontFamily: FONT_FAMILY,
            backgroundColor: '#008000',
            padding: { x: 20, y: 10 }
        }).setOrigin(0.5).setInteractive();
        restartButton.on('pointerdown', () => {
            // Abort current game before restarting
            if (currentGameId) {
                fetch(`${SERVER_URL}/abort_game`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    credentials: 'include'
                }).catch(e => console.warn('abort_game failed:', e));
                currentGameId = null;
                moveCounter = 0;
                clearMoveRecording();
            }
            const nextStarter = (scoreTracker.games_played % 2 === 0) ? 'white' : 'black';
            this.scene.start('MainGameScene', { startingPlayer: nextStarter });
        });
        const quitButton = this.add.text(CENTER_X, CENTER_Y + 120, 'Quit', {
            fontSize: '32px',
            fontFamily: FONT_FAMILY,
            backgroundColor: '#800000',
            padding: { x: 20, y: 10 }
        }).setOrigin(0.5).setInteractive();
        quitButton.on('pointerdown', () => {
            this.game.destroy(true);
        });
    }

}



class InstructionsScene extends Phaser.Scene {
    constructor() {
        super({ key: 'InstructionsScene' });
    }

    create() {
        const instructions = 'Win the game by saving all your pieces before your opponent does. Your score is the number of pieces your opponent has left. \n\n' +
            'Pieces begin on the side rack and enter the game through the yellow central home tile. Only the first piece on the side rack may enter the board. You may begin saving pieces once all your pieces have moved onto the board.  \n\n' +
            'To save a piece, move it to one of the green goal tiles and roll the number of that tile to move it off the board. ' +
            'Unnumbered pieces can be saved from any goal tile, but numbered pieces must be saved from the goal tile that matches their number. \n\n' +
            'If you land on a field tile (one that isn\'t a goal tile or the home tile) occupied by one of your opponent\'s pieces, you capture that piece and send it back to the home tile. \n\n' +
            'If a field tile is occupied by two or more of your opponent\'s pieces, that tile is blocked and you cannot move through or into it. \n\n' +
            'If you have one or more captured pieces, you must move them out of the home tile before moving any other pieces. ' +
            'Otherwise, if you have one or more pieces on the side rack, you must move the first of these onto the board before moving any other pieces. \n\n' +
            'A piece must take the shortest available route to its destination tile, both when using one die and when using two dice. \n\n' +
            'You may pass your turn without using one or both dice. \n\n' +
            'When all your pieces are either saved or on goal tiles from which they can be saved, you are in the endgame. In the endgame, you may save unnumbered pieces using a higher roll than the goal tile number, as long as you don\'t have any pieces on higher-numbered goals. \n\n' +
            'If you have only one piece left at the start of your turn, and it\'s a numbered piece which is on its goal, that piece becomes unnumbered.\n\n' +
            'Once both players have moved all their pieces onto the board, if ' + NO_SAVE_TURNS_FOR_DRAW + ' full rounds (each round being one turn for each player) pass without either player saving a piece, either player may call a draw on their turn. Saving any piece resets the count. \n\n' +
            'Click the back arrow to undo your moves or the right arrow to end your turn. \n\n' +
            'Good luck!'

        // Add instructions text
        this.add.text(CENTER_X, 50, 'How to Play', {
            fontSize: '48px',
            fontFamily: FONT_FAMILY,
            color: '#000000'
        }).setOrigin(0.5);

        // Create a text box for the instructions
        const textBox = this.add.text(CENTER_X, 300, instructions, {
            fontSize: '26px',
            fontFamily: FONT_FAMILY,
            color: '#000000',
            align: 'left',
            wordWrap: { width: config.width - 100 }
        }).setOrigin(0.5, 0);

        // Calculate the total height of the instructions text
        const instructionsHeight = textBox.height;

        // Adjust the position if the text exceeds the available space
        const maxY = config.height - 100 - 200; // Adjust maxY based on backButton position
        if (300 + instructionsHeight > maxY) {
            textBox.setY((config.height - instructionsHeight) / 2); // Center vertically if overflowing
        }

        // Add a button to go back to the main game
        const backButton = this.add.text(CENTER_X, config.height - 70, 'Back to Game', {
            fontSize: '32px',
            fontFamily: FONT_FAMILY,
            backgroundColor: '#ffcc00',
            padding: { x: 20, y: 10 },
            borderColor: '#000',
            borderWidth: 1.5,
            borderRadius: 3.75
        }).setOrigin(0.5).setInteractive();

        backButton.on('pointerdown', () => {
            this.scene.switch('MainGameScene'); // Resume the MainGameScene
        });
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
            console.log('Saving opponent block', pieceColorNumber);
            const piece = findPieceByColorAndNumber(pieceColorNumber[0], pieceColorNumber[1]);
            if (piece && piece.currentTile) {
                // Highlight all pieces in the block
                piece.currentTile.pieces.forEach(p => {
                    p.isSelected = true;
                    p.updateColor();
                });
                piece.currentTile.highlight();
                setTimeout(() => {
                    const savedRack = piece.color === 0xffffff ? game.whiteSavedRack : game.blackSavedRack;
                    [...piece.currentTile.pieces].forEach(p => p.moveToRack(savedRack));
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
    scene: [MainGameScene, InstructionsScene, EndGameScene],
};

const gameInstance = new Phaser.Game(config);