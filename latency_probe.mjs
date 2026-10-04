// Latency of the shipped JS agent under different prefilter settings, real model,
// real browser (onnxruntime-web WASM). Plays computer-vs-computer self-play; every
// time the game asks the agent for a move, the same engine position is ALSO
// searched under each alternative config and timed. The game plays the shipped
// config's answer, so the games are ordinary.
//
//   node latency_probe.mjs [n_moves=120]      -> latency_probe.json
//
// Desktop Chrome on the iMac, single-threaded WASM. A phone is slower by a
// roughly constant factor, so read RATIOS to the shipped config, not the ms.
import http from 'http';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { fileURLToPath } from 'url';
import { createRequire } from 'module';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const N_MOVES = +(process.argv[2] || 120);
const extDir = path.join(os.homedir(), '.vscode', 'extensions');
const cg = fs.readdirSync(extDir).filter(d => d.startsWith('danielsanmedium.dscodegpt-')).sort().pop();
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
try {
    const page = await (await browser.newContext({ viewport: { width: 1280, height: 900 } })).newPage();
    await page.addInitScript(() => { try {
        localStorage.setItem('whiteIsAI', '1'); localStorage.setItem('blackIsAI', '1');
        localStorage.setItem('hintsEnabled', '0'); localStorage.setItem('ruleTips', '0');
        localStorage.setItem('seenNudge', '1'); localStorage.setItem('sound', '0');
    } catch (e) {} });
    await page.goto(`http://127.0.0.1:${port}/index.html?dev=1`, { waitUntil: 'load' });
    await page.waitForFunction(() => typeof window.selectMovePair === 'function', null, { timeout: 60000 });
    await page.addScriptTag({ content: `
      (function () {
        const CONFIGS = {
          shipped:   {},
          K80:       { prefilterTopK: 80 },
          F24_K80:   { firstMovePrefilter: 24, prefilterTopK: 80 },
          onestage:  { firstMovePrefilter: 0 },
          off:       { prefilter: false },
        };
        const orig = window.selectMovePair;
        const rows = [];
        window.selectMovePair = async function (engine, W, moves, player, opts) {
          if (opts && opts.returnScores) return orig.apply(this, arguments);
          const row = { nMoves: moves.length };
          let shipped;
          for (const [name, cfg] of Object.entries(CONFIGS)) {
            const t0 = performance.now();
            const pair = await orig(engine, W, moves, player, Object.assign({}, opts, cfg));
            row[name] = Math.round(performance.now() - t0);
            row[name + '_pair'] = JSON.stringify(pair);
            if (name === 'shipped') shipped = pair;
          }
          rows.push(row);
          document.body.setAttribute('data-lat', JSON.stringify(rows));
          return shipped;
        };
      })();` });
    await page.getByText('Single game', { exact: true }).click();
    const t0 = Date.now();
    let n = 0;
    while (n < N_MOVES && Date.now() - t0 < 4 * 3600e3) {
        await page.waitForTimeout(15000);
        const s = await page.evaluate(() => document.body.getAttribute('data-lat'));
        const rows = s ? JSON.parse(s) : [];
        if (rows.length !== n) { n = rows.length; fs.writeFileSync(path.join(ROOT, 'latency_probe.json'), JSON.stringify(rows)); console.log(n, 'moves'); }
        // a finished game: start another
        const again = page.getByText(/New Game|Play again/i).first();
        if (await again.isVisible().catch(() => false)) await again.click().catch(() => {});
    }
} finally {
    await browser.close();
    server.close();
}
