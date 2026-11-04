#!/bin/zsh
# screen -> prefs -> play, each resumable. Log: probe_savebring/run.log
set -e
cd "$(dirname "$0")/.."
export BOARDGAME_DEVICE=cpu PYTHONHASHSEED=0 N_WORKERS=${N_WORKERS:-8}
[ -s probe_savebring/screen.jsonl ] || .venv/bin/python -u probe_savebring/sb.py screen
.venv/bin/python -u probe_savebring/sb.py prefs
.venv/bin/python -u probe_savebring/sb.py play
