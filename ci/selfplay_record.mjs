/* Play N full games in the real page, computer on both sides, with the game
 * recorder on, and write the recorded games (one JSON per line) for
 * replay_games.py. Exits 1 on a page error or a game that does not finish.
 *     python3 -m http.server 8765 &   (from the repo root)
 *     node ci/selfplay_record.mjs http://localhost:8765/ out.jsonl [games=2] [layout]
 * Driven only through the DOM and localStorage: no console reading (see CLAUDE.md). */
import { createRequire } from 'module';
import fs from 'fs';
const require = createRequire(import.meta.url);
let pw;
try { pw = require('playwright'); } catch (e) { pw = require(process.env.PLAYWRIGHT_PATH || 'playwright'); }
const [url, out, nGames = '2', layout = ''] = process.argv.slice(2);
const N = +nGames;
const GAME_MS = +(process.env.GAME_TIMEOUT_MS || 900000);

const browser = await pw.chromium.launch({
    args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const ctx = await browser.newContext({ viewport: { width: 1280, height: 860 } });
await ctx.addInitScript(() => {
    try {
        if (sessionStorage.getItem('ci-init')) return;
        sessionStorage.setItem('ci-init', '1');
        localStorage.clear();
        for (const [k, v] of Object.entries({ whiteIsAI: '1', blackIsAI: '1', recEnabled: '1',
            hintsEnabled: '0', ruleTips: '0', seenNudge: '1', sound: '0' })) localStorage.setItem(k, v);
    } catch (e) {}
});
const page = await ctx.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(String(e && e.stack || e)));
const full = url + (url.includes('?') ? '&' : '?') + 'dev=1' + (layout ? '&layout=' + layout : '');

// recGames = a JSON array of JSON lines, one per finished game (game.js REC_KEY).
const recLines = () => page.evaluate(() => {
    try { return JSON.parse(localStorage.getItem('recGames') || '[]'); } catch (e) { return []; }
});
const recCount = async () => (await recLines()).length;
let done = 0;
for (let g = 0; g < N; g++) {
    // Each game on a fresh load (the recorder buffer survives in localStorage),
    // started from the welcome card's Single game button.
    await page.goto(full);
    const single = page.getByText('Single game', { exact: true });
    await single.first().waitFor({ timeout: 60000 });
    await single.first().click();
    const before = await recCount();
    const start = Date.now();
    while (await recCount() <= before) {
        if (errors.length) break;
        if (Date.now() - start > GAME_MS) { errors.push(`game ${g + 1} did not finish in ${GAME_MS} ms`); break; }
        await page.waitForTimeout(2000);
    }
    if (errors.length) break;
    done++;
    console.log(`game ${g + 1} recorded after ${Math.round((Date.now() - start) / 1000)} s`);
}
const lines = await recLines();
fs.writeFileSync(out, lines.join('\n') + (lines.length ? '\n' : ''));
await browser.close();
console.log(`${done}/${N} games played, ${lines.length} recorded -> ${out}; page errors: ${errors.length}`);
for (const e of errors.slice(0, 5)) console.log('ERROR', e);
process.exit(errors.length || done < N ? 1 : 0);
