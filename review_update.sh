#!/bin/zsh
# New phone games -> new quiz positions. Pulls the synced games, scans and
# verifies only games/turns not seen before (both steps resume), then writes
# NEW chunk files under review/quiz/ for Claude to push to the quiz artifact's
# `positions` collection (ArtifactData set, one doc per chunk file).
# review/quiz/index.json is the record of what is already in the quiz: keep it.
set -e
cd "$(dirname "$0")"
git -C games_sync pull -q
export BOARDGAME_DEVICE=cpu GATE_PREFILTER=12,40,5
python3 review_games.py scan distill_cand13.pt review/scan_cand13.jsonl
python3 review_games.py verify distill_cand13.pt review/scan_cand13.jsonl review/verify_cand13.jsonl
python3 quiz_build.py review/verify_cand13.jsonl
