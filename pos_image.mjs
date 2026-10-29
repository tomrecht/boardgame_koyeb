// Render a position, written in the game's notation, to a PNG -- using the game
// itself, so the picture is exactly what a player would see.
//
//   node pos_image.mjs "<notation>" out.png [--phone] [--caption "text"]
//        [--fill "5.4,5.6:#e0b040;G2:#7fb069"] [--ring "6.4,G3"] [--labels]
//        [--arrows "5.4>G2:#5fae6a;H>3.2:#8a5cc9;G4>:#5fae6a"]
//
// --fill tints tiles (the tutorial's target fill; default colour amber), --ring
// draws a static ring round tiles (hint colour), --labels writes every tile's
// location on it -- a drafting aid for writing positions, not for publication.
//
// The notation is documented beside positionToNotation in game.js. Settings >
// Position (dev) > Copy gives you one from a live game; ?dev=1&pos=<notation>
// opens one as a playable board.
//
// Serves this folder on a free local port and drives the SYSTEM Chrome through
// patchright (the downloaded Chromium cannot be installed on this macOS; see
// CLAUDE.md, "HARNESS: DRIVE SYSTEM CHROME"). The swiftshader flags are what give
// headless Chrome WebGL, so the board bakes as it does for a player.
// Needs node on PATH (on the iMac: PATH="/usr/local/bin:$PATH").
import http from 'http';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { fileURLToPath } from 'url';
import { createRequire } from 'module';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const flag = (name) => { const i = args.indexOf(name); if (i < 0) return null; const v = args[i + 1]; args.splice(i, 2); return v; };
const phone = args.includes('--phone'); if (phone) args.splice(args.indexOf('--phone'), 1);
const caption = flag('--caption');
const fillArg = flag('--fill'), ringArg = flag('--ring'), arrowArg = flag('--arrows');
const labels = args.includes('--labels'); if (labels) args.splice(args.indexOf('--labels'), 1);
const [notation, out] = args;
if (!notation || !out) {
    console.error('usage: node pos_image.mjs "<notation>" out.png [--phone] [--caption "text"]');
    process.exit(2);
}

// patchright ships inside the CodeGPT VS Code extension; take the newest.
const extDir = path.join(os.homedir(), '.vscode', 'extensions');
const cg = fs.readdirSync(extDir).filter(d => d.startsWith('danielsanmedium.dscodegpt-')).sort().pop();
if (!cg) { console.error('patchright not found (no CodeGPT extension under ' + extDir + ')'); process.exit(1); }
const { chromium } = createRequire(path.join(extDir, cg, 'standalone') + '/')('patchright');
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json',
                '.png': 'image/png', '.wasm': 'application/wasm', '.onnx': 'application/octet-stream', '.css': 'text/css' };
const server = http.createServer((req, res) => {
    const p = path.join(ROOT, decodeURIComponent(new URL(req.url, 'http://x').pathname));
    if (!p.startsWith(ROOT) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, { 'Content-Type': TYPES[path.extname(p)] || 'application/octet-stream' });
    fs.createReadStream(p).pipe(res);
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const port = server.address().port;

const browser = await chromium.launch({ executablePath: CHROME, headless: true,
    args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
let code = 0;
try {
    const ctx = await browser.newContext(phone
        ? { viewport: { width: 412, height: 915 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true }
        : { viewport: { width: 1280, height: 900 } });
    const page = await ctx.newPage();
    // A first visit would seed hints and rule tips; keep the picture clean.
    await page.addInitScript(() => { try { localStorage.setItem('hintsEnabled', '0'); localStorage.setItem('ruleTips', '0');
                                           localStorage.setItem('seenNudge', '1'); } catch (e) {} });
    const url = `http://127.0.0.1:${port}/index.html?dev=1&posshot=1&pos=${encodeURIComponent(notation)}`;
    await page.goto(url);
    await page.waitForFunction(() => document.body.getAttribute('data-pos'), null, { timeout: 30000 });
    const status = await page.evaluate(() => document.body.getAttribute('data-pos'));
    if (status !== 'ok') { console.error('position not loaded: ' + status); code = 1; }
    else {
        if (caption) await page.evaluate((text) => {
            const c = document.createElement('div');
            c.style.cssText = 'position:fixed;left:50%;transform:translateX(-50%);bottom:14px;max-width:90vw;' +
                'background:rgba(255,255,255,.95);color:#28313b;font:600 16px/1.4 system-ui,sans-serif;' +
                'padding:9px 14px;border-radius:10px;box-shadow:0 4px 18px rgba(0,0,0,.2);z-index:999;text-align:center';
            c.textContent = text; document.body.appendChild(c);
        }, caption);
        if (fillArg || ringArg || labels || arrowArg) {
            // game.js globals live in the page's MAIN world; page.evaluate cannot
            // see them, so the drawing goes in through a script tag.
            const fills = [];
            for (const grp of (fillArg || '').split(';').filter(Boolean)) {
                const [locs, col] = grp.split(':');
                for (const l of locs.split(',')) fills.push([l.trim(), col || '#e0b040']);
            }
            const rings = (ringArg || '').split(',').map(x => x.trim()).filter(Boolean);
            // --arrows "from>to:#col;from>:#col" -- a ring on the moving piece's tile and an
            // arrow to its destination (no destination = ring only, e.g. a piece banked).
            const arrows = (arrowArg || '').split(';').filter(Boolean).map(g => {
                const [mv, col] = g.split(':'); const [a, b] = mv.split('>');
                return [a.trim(), (b || '').trim(), col || '#5fae6a'];
            });
            await page.addScriptTag({ content: `(() => {
                const game = _currentGame(), scene = _setupScene();
                const fills = ${JSON.stringify(fills)}, rings = ${JSON.stringify(rings)}, labels = ${labels};
                const arrows = ${JSON.stringify(arrows)};
                const bad = [];
                for (const [loc, col] of fills) {
                    const t = _posTileFor(game, loc); if (!t) { bad.push(loc); continue; }
                    t._tutTargetColor = parseInt(col.replace('#', ''), 16); t._tutTarget = true; t.drawTile();
                }
                for (const loc of rings) {
                    const t = _posTileFor(game, loc), c = _hintTileCentre(t); if (!c) { bad.push(loc); continue; }
                    const depth = (t.outerRadius != null) ? t.outerRadius - t.innerRadius : 80;
                    scene.add.circle(c.x, c.y, Math.max(20, Math.min(62, depth * 0.42)), 0, 0)
                         .setStrokeStyle(6, HINT_COLOR, 1).setDepth(76);
                }
                const cols = [...new Set(arrows.map(a => a[2]))];
                for (const [from, to, col] of arrows) {
                    const tf = _posTileFor(game, from), a = _hintTileCentre(tf);
                    if (!a) { bad.push(from); continue; }
                    const c = parseInt(col.replace('#', ''), 16);
                    const off = cols.length > 1 ? (cols.indexOf(col) - (cols.length - 1) / 2) * 9 : 0;
                    scene.add.circle(a.x, a.y, 30, 0, 0).setStrokeStyle(5, c, 1).setDepth(77);
                    if (!to) continue;
                    const b = _hintTileCentre(_posTileFor(game, to));
                    if (!b) { bad.push(to); continue; }
                    const dx = b.x - a.x, dy = b.y - a.y, L = Math.hypot(dx, dy) || 1;
                    const ux = dx / L, uy = dy / L, px = -uy * off, py = ux * off;
                    const x0 = a.x + ux * 30 + px, y0 = a.y + uy * 30 + py;
                    const x1 = b.x - ux * 26 + px, y1 = b.y - uy * 26 + py;
                    const g = scene.add.graphics().setDepth(78);
                    g.lineStyle(10, 0xffffff, 0.8).lineBetween(x0, y0, x1 - ux * 14, y1 - uy * 14);
                    g.lineStyle(6, c, 1).lineBetween(x0, y0, x1 - ux * 14, y1 - uy * 14);
                    g.fillStyle(c, 1).fillTriangle(x1, y1, x1 - ux * 24 - uy * 13, y1 - uy * 24 + ux * 13,
                                                   x1 - ux * 24 + uy * 13, y1 - uy * 24 - ux * 13);
                }
                if (labels) for (const t of game.tiles) {
                    if (t.type === 'nogo') continue;
                    const c = _hintTileCentre(t); if (!c) continue;
                    scene.add.text(c.x, c.y, _posLoc(t), { font: 'bold 13px sans-serif', color: '#000',
                        backgroundColor: 'rgba(255,255,255,.75)' }).setOrigin(0.5).setDepth(90);
                }
                document.body.setAttribute('data-marks', bad.length ? 'bad: ' + bad.join(' ') : 'ok');
            })();` });
            const marks = await page.evaluate(() => document.body.getAttribute('data-marks'));
            if (marks !== 'ok') { console.error('marks: ' + marks); code = 1; }
        }
        // Headless Chrome's canvas lags the DOM; give the bake and the slides time.
        await page.waitForTimeout(3000);
        await page.screenshot({ path: out });
        console.log('wrote ' + out);
    }
} finally {
    await browser.close();
    server.close();
}
process.exit(code);
