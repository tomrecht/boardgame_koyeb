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
import json
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
    # MIT requires the notice to travel with the software, and phaser.min.js
    # ships with its banner stripped. Linked from How to Play > Credits.
    'licenses.html',
]

# Phaser preloads these; the service worker precaches them for the same reason.
ASSETS = [
    'assets/left-arrow.png',
    'assets/right-arrow.png',
    'assets/thinking.png',
]

# Strategy tips pictures (showStrategy). Not precached by sw.js: they are only
# needed when the panel is opened, and cache-first after that. Served with a
# week's max-age, so a changed picture needs a new file name.
STRATEGY = [
    'assets/strategy/park.webp',
    'assets/strategy/every-roll.webp',
    'assets/strategy/pair-block.webp',
    'assets/strategy/hop.webp',
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
    'features_v2.js',
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
    # The net itself ships under the name the sidecar gives it (MODEL_FILE
    # below), not as model.onnx -- see there.
    # The model's SIDECAR (onnx_export.py writes it): which encoder the net
    # needs -- feature set v1/A/AB, and whether it was trained on end-of-turn
    # positions. infer.js reads it; absent would mean v1 turn start.
    'model.json',
]


def _model_meta():
    try:
        return json.load(open(os.path.join(HERE, 'model.json')))
    except Exception:
        return {}


# THE NET SHIPS UNDER A CONTENT-HASHED NAME (2026-10-19). sw.js serves .onnx
# cache-first, so a returning player's OLD service worker -- the one that answers
# the first load after a deploy -- handed back the cached old model.onnx beside
# the fresh model.json, and an AB2 sidecar on a v1 net made the first inference
# fail ("on-device computer is switched off"). A name the old worker has never
# seen goes to the network. The repo keeps model.onnx for the Python tools; the
# sidecar's "onnx" names a byte-identical copy, model-<sha256[:8]>.onnx.
def _model_file():
    return _model_meta().get('onnx') or 'model.onnx'


MODEL_FILE = _model_file()


# An A/AB net reads the exact endgame table (features_v2); a v1 net does not,
# so it ships only then -- 1.3 MB raw, ~0.4 MB compressed.
NEEDS_TABLE = _model_meta().get('features', 'v1') != 'v1'
BUNDLE = SHELL + ASSETS + STRATEGY + ICONS + PORT + RUNTIME + [MODEL_FILE] + \
    (['endgame_table.json'] if NEEDS_TABLE else [])


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

    if MODEL_FILE == 'model.onnx':
        problems.append("model.json names no hashed model file ('onnx'): a returning "
                        "player's old service worker would serve its cached model.onnx "
                        "with the new sidecar")
    elif os.path.exists(os.path.join(HERE, MODEL_FILE)):
        import hashlib
        data = open(os.path.join(HERE, MODEL_FILE), 'rb').read()
        if MODEL_FILE != 'model-%s.onnx' % hashlib.sha256(data).hexdigest()[:8]:
            problems.append(MODEL_FILE + ' is not named by its content hash')
        if data != open(os.path.join(HERE, 'model.onnx'), 'rb').read():
            problems.append(MODEL_FILE + ' differs from model.onnx (the Python tools '
                            'and the app would run different nets)')

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
        # The table is fetched at start-up by an A/AB net, so offline it must
        # come from the cache: refuse to build an A/AB model without it there.
        if NEEDS_TABLE and 'endgame_table.json' not in sw:
            problems.append("model.json says an A/AB net but sw.js does not precache "
                            "'./endgame_table.json' (offline the computer could not "
                            "move) -- add it to SHELL and bump CACHE")
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
            "/licenses.html\n"
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
            # no-cache (revalidate) rather than a week's max-age: a new champion
            # must reach returning players on their next visit (2026-10-06)
            "/model.onnx\n"
            "  Cache-Control: no-cache\n"
            # the hashed name never changes content
            "/model-*.onnx\n"
            "  Cache-Control: public, max-age=31536000, immutable\n"
            "/model.json\n"
            "  Cache-Control: no-cache\n"
            "/features_v2.js\n"
            "  Cache-Control: no-cache\n"
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
