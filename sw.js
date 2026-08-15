/* Quahuru service worker.
 *
 * Deliberately network-FIRST for the two files that change every deploy
 * (index.html and game.js). A cache-first worker would serve a stale game.js
 * after a push, which is exactly the class of bug that costs an afternoon --
 * see CLAUDE.md, where adding a worker was deferred for this reason.
 *
 * Everything else (Phaser, icons, the manifest) is content that only changes
 * when its name does, so it is cache-first with a background refresh.
 *
 * Bump CACHE when the shell list changes; activate() drops every other cache.
 */
const CACHE = 'quahuru-v6';

// The shell is what a cold, offline start needs to render a board.
const SHELL = [
    './',
    './index.html',
    './game.js',
    './phaser.min.js',
    './manifest.json',
    './icon-192.png',
    './icon-512.png',
    './icon-192-maskable.png',
    './icon-512-maskable.png',
    // Phaser preloads these; they MUST be precached rather than left to
    // cache-first. On the very first visit the worker is not controlling the
    // page yet, so the requests Phaser makes during that load are never seen
    // and never cached -- and the next (offline) load has no arrows.
    './assets/left-arrow.png',
    './assets/right-arrow.png',
    './assets/thinking.png',
    // The inference runtime. Big (10.5MB raw, 2.8MB gzipped), so it is
    // precached deliberately: it is what makes offline PLAY possible, and
    // fetching it mid-game would be worse than paying for it up front.
    './ort/ort.wasm.min.js',
    './ort/ort-wasm-simd-threaded.mjs',
    './ort/ort-wasm-simd-threaded.wasm',
    './model.onnx',
];

// Files whose content changes under a fixed name -- always try the network.
// That is ALL of our own JavaScript, not just game.js: route.js, encoder.js and
// infer.js change on every deploy too, and leaving them cache-first served a
// stale copy during development (it cost an hour chasing a fix that had already
// been made). Third-party pinned files are excluded -- phaser.min.js and
// everything under /ort/ only change when their version does.
const VENDORED = /\/(phaser\.min\.js|ort\/)/;
const ALWAYS_FRESH = (path) =>
    !VENDORED.test(path) && /(\/|\.html|\.js|\.json)$/.test(path);

self.addEventListener('install', (e) => {
    e.waitUntil((async () => {
        const cache = await caches.open(CACHE);
        // addAll fails the whole install if any one file 404s, which would
        // leave no worker at all; take them one at a time instead.
        await Promise.all(SHELL.map(async (url) => {
            try { await cache.add(new Request(url, { cache: 'reload' })); }
            catch (err) { console.warn('[sw] precache skipped', url, err); }
        }));
        await self.skipWaiting();
    })());
});

self.addEventListener('activate', (e) => {
    e.waitUntil((async () => {
        const names = await caches.keys();
        await Promise.all(names.filter(n => n !== CACHE).map(n => caches.delete(n)));
        await self.clients.claim();
    })());
});

self.addEventListener('fetch', (e) => {
    const req = e.request;
    if (req.method !== 'GET') return;                 // /select_moves etc. are POSTs
    const url = new URL(req.url);
    if (url.origin !== self.location.origin) return;  // never touch third parties
    // API calls must never be served from cache.
    if (/^\/(select_moves|evaluate_board|call_draw|start_game|abort_game|update_impasse|query_agent_move|debug_piece_blots|training_data_stats|record_)/.test(url.pathname)) return;

    const fresh = req.mode === 'navigate' || ALWAYS_FRESH(url.pathname);
    e.respondWith(fresh ? networkFirst(req) : cacheFirst(req));
});

async function networkFirst(req) {
    const cache = await caches.open(CACHE);
    try {
        const res = await fetch(req);
        if (res && res.ok) cache.put(req, res.clone());
        return res;
    } catch (err) {
        const hit = await cache.match(req, { ignoreSearch: true });
        if (hit) return hit;
        // A navigation with nothing cached still needs something to render.
        const shell = await cache.match('./index.html', { ignoreSearch: true });
        if (shell) return shell;
        throw err;
    }
}

async function cacheFirst(req) {
    const cache = await caches.open(CACHE);
    const hit = await cache.match(req, { ignoreSearch: true });
    if (hit) {
        // Refresh in the background so a changed asset is picked up next load.
        fetch(req).then(res => { if (res && res.ok) cache.put(req, res.clone()); }).catch(() => {});
        return hit;
    }
    const res = await fetch(req);
    if (res && res.ok) cache.put(req, res.clone());
    return res;
}
