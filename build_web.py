#!/usr/bin/env python3
"""Assemble the static web bundle into dist/.

The repo is 79 MB tracked, and 45 MB of that is training data (positions_with_
moves.jsonl alone is 34 MB) plus ~10 MB of checkpoints and design PNGs. None of
it belongs on a public host, so a static deploy must NOT point at the repo root
-- it points at what this script produces.

The list below is the whole runtime surface, and it is checked rather than
trusted: every file must exist, and it must agree with sw.js's precache list,
because a file the worker precaches but the deploy omits is an app that installs
and then cannot start offline.

Usage:  python build_web.py [--out dist] [--check]
        --check verifies the manifest without writing anything.
"""
import argparse
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# The page itself.
SHELL = [
    'index.html',
    'manifest.json',
    'sw.js',
    'game.js',
    'local_agent.js',
    'phaser.min.js',
    # Required by both app stores, at a public URL, even for an app that
    # collects nothing. Deliberately NOT precached by sw.js: it is not needed to
    # play offline, and the worker's list is what the check below enforces.
    'privacy.html',
]

# Phaser preloads these; the service worker precaches them for the same reason.
ASSETS = [
    'assets/left-arrow.png',
    'assets/right-arrow.png',
    'assets/thinking.png',
]

ICONS = [
    'icon-192.png',
    'icon-512.png',
    'icon-192-maskable.png',
    'icon-512-maskable.png',
]

# The ported agent (local_agent.js's FILES) and the data it reads.
PORT = [
    'route.js',
    'encoder.js',
    'infer.js',
    'engine.js',
    'heuristic.js',
    'agent.js',
    'encoder_static.json',
    'heuristic_weights.json',
]

# The inference runtime. The .wasm is 10.5 MB raw; a host that serves it
# compressed sends ~2.8 MB. The pre-made .wasm.gz is NOT shipped -- any CDN
# worth using compresses on the fly, and shipping both invites serving the
# wrong one.
RUNTIME = [
    'ort/ort.wasm.min.js',
    'ort/ort-wasm-simd-threaded.mjs',
    'ort/ort-wasm-simd-threaded.wasm',
    'model.onnx',
]

BUNDLE = SHELL + ASSETS + ICONS + PORT + RUNTIME


def sw_precache_list():
    """The paths sw.js precaches, so the two lists can be compared."""
    src = open(os.path.join(HERE, 'sw.js')).read()
    block = re.search(r'const SHELL = \[(.*?)\];', src, re.S)
    if not block:
        return None
    return {m.group(1) for m in re.finditer(r"'\./([^']+)'", block.group(1))}


def check():
    problems = []

    missing = [f for f in BUNDLE if not os.path.exists(os.path.join(HERE, f))]
    if missing:
        problems.append('missing from the repo: ' + ', '.join(missing))

    sw = sw_precache_list()
    if sw is None:
        problems.append("could not parse sw.js's SHELL list")
    else:
        bundle = set(BUNDLE)
        # './' in the worker's list is index.html under another name.
        sw = {p for p in sw if p}
        not_shipped = sorted(sw - bundle)
        if not_shipped:
            problems.append('precached by sw.js but NOT in the bundle (the app '
                            'would install and fail offline): ' + ', '.join(not_shipped))
    return problems


def build(out):
    problems = check()
    if problems:
        for p in problems:
            print('ERROR: ' + p, file=sys.stderr)
        return 1

    if os.path.exists(out):
        shutil.rmtree(out)
    total = 0
    for rel in BUNDLE:
        src = os.path.join(HERE, rel)
        dst = os.path.join(out, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        total += os.path.getsize(src)

    # Cache policy, for hosts that read _headers (Cloudflare Pages, Netlify).
    # index.html/game.js change on every deploy under a fixed name, so they must
    # not be held in the HTTP cache -- the service worker is network-first for
    # exactly the same reason. Everything else is content that changes only when
    # its name does.
    with open(os.path.join(out, '_headers'), 'w') as fh:
        fh.write(
            "/*\n"
            "  X-Content-Type-Options: nosniff\n"
            "\n"
            "/index.html\n"
            "  Cache-Control: no-cache\n"
            "/privacy.html\n"
            "  Cache-Control: no-cache\n"
            "/sw.js\n"
            "  Cache-Control: no-cache\n"
            "/game.js\n"
            "  Cache-Control: no-cache\n"
            "/local_agent.js\n"
            "  Cache-Control: no-cache\n"
            "/route.js\n"
            "  Cache-Control: no-cache\n"
            "/encoder.js\n"
            "  Cache-Control: no-cache\n"
            "/infer.js\n"
            "  Cache-Control: no-cache\n"
            "/engine.js\n"
            "  Cache-Control: no-cache\n"
            "/heuristic.js\n"
            "  Cache-Control: no-cache\n"
            "/agent.js\n"
            "  Cache-Control: no-cache\n"
            "\n"
            "/ort/*\n"
            "  Cache-Control: public, max-age=31536000\n"
            "/model.onnx\n"
            "  Cache-Control: public, max-age=604800\n"
            "/phaser.min.js\n"
            "  Cache-Control: public, max-age=31536000\n"
            "/assets/*\n"
            "  Cache-Control: public, max-age=604800\n"
        )

    print('%s: %d files, %.1f MB' % (out, len(BUNDLE), total / 1048576.0))
    print('  (the .wasm is %.1f MB of that and compresses to ~2.8 MB in transit)'
          % (os.path.getsize(os.path.join(HERE, 'ort/ort-wasm-simd-threaded.wasm')) / 1048576.0))
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'dist'))
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    if args.check:
        problems = check()
        for p in problems:
            print('ERROR: ' + p, file=sys.stderr)
        print('bundle OK: %d files' % len(BUNDLE) if not problems else 'bundle has problems')
        sys.exit(1 if problems else 0)
    sys.exit(build(args.out))
