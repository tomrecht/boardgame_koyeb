/* KILL SWITCH -- not served by default.
 *
 * A service worker persists on a device: deleting sw.js does NOT remove an
 * installed one (a 404 on the update check leaves the existing worker in
 * place). To retire it, COPY THIS OVER sw.js and deploy:
 *
 *     cp sw-kill.js sw.js && git commit -am "sw: kill switch" && git push
 *
 * Browsers byte-compare sw.js on navigation, so the next visit installs this,
 * which unregisters itself and drops every cache. Once the fleet has picked it
 * up, sw.js can be deleted for real.
 */
self.addEventListener('install', () => self.skipWaiting());

self.addEventListener('activate', (e) => {
    e.waitUntil((async () => {
        const names = await caches.keys();
        await Promise.all(names.map(n => caches.delete(n)));
        await self.registration.unregister();
        const clients = await self.clients.matchAll({ type: 'window' });
        clients.forEach(c => c.navigate(c.url));       // reload onto the plain network
    })());
});
