// Latency of the CLOSE-CALL search (agent.js selectWithCloseCall), real model,
// real browser (onnxruntime-web WASM) -- latency_probe.mjs with the close-call
// settings as the alternatives. Computer-vs-computer self-play; every move the
// game asks for is ALSO searched under each config and timed, with the close-call
// stats (triggered, rounds of rolls done, changed). The game plays the shipped
// (close-call off) answer, so the games are ordinary.
//
//   node close_call_latency.mjs [n_moves=120]      -> close_call_latency.json
//
// Read per config: median / p90 / max ms over ALL moves, and over triggered
// moves only; rounds per trigger (21 = the full search) under each budget.
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
// patchright: from $PATCHRIGHT_FROM if set, else the CodeGPT extension's copy
function pwBase() {
    if (process.env.PATCHRIGHT_FROM) return process.env.PATCHRIGHT_FROM;
    const extDir = path.join(os.homedir(), '.vscode', 'extensions');
    const cg = fs.readdirSync(extDir).filter(d => d.startsWith('danielsanmedium.dscodegpt-')).sort().pop();
    return path.join(extDir, cg, 'standalone');
}
const { chromium } = createRequire(pwBase() + '/')('patchright');
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
        const CC = (ms, extra) => Object.assign({ closeCall: true, closeCallBudgetMs: ms }, extra || {});
        // 'shipped' = the plain search (close-call is ON by default since
        // 2026-10-22, so it must be switched off explicitly); cc_r3 = what ships:
        // one round of rolls, whatever the device.
        const CONFIGS = {
          shipped:   { closeCall: false },
          cc_r3:     CC(Infinity, { closeCallMaxReplies: 3 }),
        };
        const orig = window.selectMovePair;
        const rows = [];
        window.selectMovePair = async function (engine, W, moves, player, opts) {
          // agent.js's functions are page globals, so the close-call search's OWN
          // inner calls (the ranking pass, the opponent replies) land here too:
          // pass them straight through, or every config re-runs inside every other.
          if (opts && (opts.returnScores || opts._rankOut || opts.closeCall === false))
            return orig.apply(this, arguments);
          const row = { nMoves: moves.length };
          let shipped;
          for (const [name, cfg] of Object.entries(CONFIGS)) {
            const t0 = performance.now();
            const st = {};
            const pair = await orig(engine, W, moves, player, Object.assign({}, opts, cfg,
                                                                            cfg.closeCall ? { closeCallStats: st } : {}));
            row[name] = Math.round(performance.now() - t0);
            if (cfg.closeCall) row[name + '_cc'] = { t: st.triggered || 0, r: st.rounds || 0, c: st.changed || 0 };
            row[name + '_pair'] = JSON.stringify(pair);
            if (name === 'shipped') shipped = pair;
          }
          rows.push(row);
          document.body.setAttribute('data-lat', JSON.stringify(rows));
          return shipped;
        };
        // The end card's buttons are canvas objects; start the next game directly.
        setInterval(() => {
          try {
            const g = (typeof gameInstance !== 'undefined') ? gameInstance : null;
            const sm = g && g.scene;
            if (sm && sm.isActive && sm.isActive('EndGameScene')) startGame(nextCasualStarter());
          } catch (e) {}
        }, 3000);
      })();` });
    await page.getByText('Single game', { exact: true }).click();
    const t0 = Date.now();
    let n = 0;
    while (n < N_MOVES && Date.now() - t0 < 4 * 3600e3) {
        await page.waitForTimeout(15000);
        const s = await page.evaluate(() => document.body.getAttribute('data-lat'));
        const rows = s ? JSON.parse(s) : [];
        if (rows.length !== n) { n = rows.length; fs.writeFileSync(path.resolve(ROOT, process.env.OUT || 'close_call_latency.json'), JSON.stringify(rows)); console.log(n, 'moves'); }
    }
} finally {
    await browser.close();
    server.close();
}
