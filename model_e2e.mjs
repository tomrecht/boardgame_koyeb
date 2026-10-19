// model_e2e.mjs -- end-to-end check of the JS agent with a chosen model, in the
// real page (system Chrome, swiftshader WebGL), for shipping a new net:
//   MODEL=<path.onnx> node model_e2e.mjs <label> <n_moves> [scores.json]
// MODEL's sidecar <path>.json is served as model.json (none -> v1). scores.json
// (optional): [{model, kind, player, snapshot, py}] reference scores from Python
// (OnnxBackend), compared with Infer.score. Then both sides play; every move is
// timed. Result -> e2e_<label>.json + stdout.
import http from 'http';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { createRequire } from 'module';

const ROOT = path.dirname(new URL(import.meta.url).pathname);
const [label = 'run', nMovesArg = '40', scoresPath = ''] = process.argv.slice(2);
const N_MOVES = +nMovesArg;
const MODEL = process.env.MODEL || '';
// patchright: from $PATCHRIGHT_FROM (a dir whose node_modules has it) if set,
// else the CodeGPT extension's copy, as on the iMac.
function pwBase() {
    if (process.env.PATCHRIGHT_FROM) return process.env.PATCHRIGHT_FROM;
    const extDir = path.join(os.homedir(), '.vscode', 'extensions');
    const cg = fs.readdirSync(extDir).filter(d => d.startsWith('danielsanmedium.dscodegpt-')).sort().pop();
    return path.join(extDir, cg, 'standalone');
}
const { chromium } = createRequire(pwBase() + '/')('patchright');
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json',
                '.png': 'image/png', '.webp': 'image/webp', '.wasm': 'application/wasm',
                '.onnx': 'application/octet-stream', '.css': 'text/css' };
const served = { model: 0, sidecar: 0, table: 0 };
const server = http.createServer((req, res) => {
    const u = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    let p = path.join(ROOT, u);
    if (u === '/model.onnx' && MODEL) { p = MODEL; served.model++; }
    if (u === '/model.json' && MODEL) {
        const side = MODEL.replace(/\.onnx$/, '.json');
        served.sidecar++;
        if (!fs.existsSync(side)) { res.writeHead(404); res.end(); return; }
        p = side;
    }
    if (u === '/endgame_table.json') served.table++;
    if (!fs.existsSync(p) || fs.statSync(p).isDirectory()) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, { 'Content-Type': TYPES[path.extname(p)] || 'application/octet-stream' });
    fs.createReadStream(p).pipe(res);
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const port = server.address().port;
const result = { label, model: MODEL || '(repo model.onnx)' };
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
    // the agent loads lazily; force it and wait until it is ready
    await page.addScriptTag({ content: `(async () => { try {
        const ok = await LocalAgent.init();
        document.body.setAttribute('data-init', String(ok));
      } catch (e) { document.body.setAttribute('data-init', 'error: ' + e.message); } })();` });
    await page.waitForFunction(() => document.body.getAttribute('data-init'), null, { timeout: 120000 });
    result.init = await page.evaluate(() => document.body.getAttribute('data-init'));
    await page.addScriptTag({ content: `document.body.setAttribute('data-cfg', JSON.stringify(window.Infer.config()));` });
    result.config = JSON.parse(await page.evaluate(() => document.body.getAttribute('data-cfg')));
    // 1. score parity
    if (scoresPath) {
        const ref = JSON.parse(fs.readFileSync(scoresPath, 'utf8'))
            .filter(r => r.model === (result.config.features === 'v1' ? 'v1' : 'AB_after'));
        await page.addScriptTag({ content: `(async () => {
            const ref = ${JSON.stringify(ref.map(r => ({ s: r.snapshot, p: r.player })))};
            const out = [];
            for (const r of ref) out.push((await window.Infer.score(window.ort, [r.s], r.p))[0]);
            document.body.setAttribute('data-scores', JSON.stringify(out));
          })();` });
        await page.waitForFunction(() => document.body.getAttribute('data-scores'), null, { timeout: 120000 });
        const js = JSON.parse(await page.evaluate(() => document.body.getAttribute('data-scores')));
        const diffs = js.map((v, i) => Math.abs(v - ref[i].py));
        const lastIdx = ref.map((r, i) => r.kind === 'last' ? i : -1).filter(i => i >= 0);
        result.parity = { n: js.length, maxAbsDiff: Math.max(...diffs),
                          within1e4: diffs.filter(d => d < 1e-4).length,
                          lastPieceCases: lastIdx.length,
                          lastPieceMaxDiff: lastIdx.length ? Math.max(...lastIdx.map(i => diffs[i])) : null };
    }
    // 2. self-play timing
    await page.addScriptTag({ content: `(function () {
        const orig = window.selectMovePair;
        const rows = [];
        window.selectMovePair = async function (engine, W, moves, player, opts) {
          if (opts && opts.returnScores) return orig.apply(this, arguments);
          const t0 = performance.now();
          const pair = await orig.apply(this, arguments);
          rows.push({ ms: Math.round(performance.now() - t0), n: moves.length });
          document.body.setAttribute('data-lat', JSON.stringify(rows));
          return pair;
        };
        setInterval(() => { try {
            const sm = gameInstance && gameInstance.scene;
            if (sm && sm.isActive('EndGameScene')) {
              document.body.setAttribute('data-games', String(+(document.body.getAttribute('data-games') || 0) + 1));
              startGame(nextCasualStarter());
            }
        } catch (e) {} }, 2000);
      })();` });
    await page.getByText('Single game', { exact: true }).click();
    const t0 = Date.now();
    let rows = [];
    while (rows.length < N_MOVES && Date.now() - t0 < 40 * 60e3) {
        await page.waitForTimeout(10000);
        const s = await page.evaluate(() => document.body.getAttribute('data-lat'));
        rows = s ? JSON.parse(s) : [];
    }
    const ms = rows.map(r => r.ms).sort((a, b) => a - b);
    const q = (f) => ms.length ? ms[Math.min(ms.length - 1, Math.floor(f * ms.length))] : null;
    result.moves = rows.length;
    result.latency = { median: q(0.5), p90: q(0.9), max: ms[ms.length - 1] };
    result.gamesFinished = +(await page.evaluate(() => document.body.getAttribute('data-games') || 0));
    result.served = served;
} finally {
    await browser.close();
    server.close();
}
fs.writeFileSync(path.join(ROOT, `e2e_${label}.json`), JSON.stringify(result, null, 1));
console.log(JSON.stringify(result));
