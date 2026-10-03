#!/usr/bin/env bash
# cloud_setup.sh -- set up a fresh Linux VM (Ubuntu/Debian) for a panel-league
# training run, and optionally launch it. CPU only: generation and gating are
# many small single-threaded 2-ply searches, so cores are what matter.
#
#   curl -fsSL https://raw.githubusercontent.com/tomrecht/boardgame_koyeb/train-panel-league/cloud_setup.sh | bash
#   curl -fsSL ... | LAUNCH=1 bash          # set up AND start the full run
#
# Env: REPO_URL, BRANCH (train-panel-league), DEST (~/boardgame_koyeb),
#      LAUNCH=1 to start the run under nohup, SMOKE=1 to launch a smoke run
#      instead. Any league_run.py knob (see CLAUDE.md RUNBOOK) passes through.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/tomrecht/boardgame_koyeb.git}"
BRANCH="${BRANCH:-train-panel-league}"
DEST="${DEST:-$HOME/boardgame_koyeb}"

echo "== system packages"
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y git python3 python3-venv python3-pip
fi

echo "== clone $BRANCH"
if [ -d "$DEST/.git" ]; then
  git -C "$DEST" fetch origin
  git -C "$DEST" checkout "$BRANCH"
  git -C "$DEST" pull --ff-only origin "$BRANCH"
else
  git clone --branch "$BRANCH" "$REPO_URL" "$DEST"
fi
cd "$DEST"

echo "== python deps (CPU-only torch wheel, numpy<2, scipy)"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip install "numpy<2" scipy

CORES="$(nproc)"
WORKERS=$(( CORES > 1 ? CORES - 1 : 1 ))
echo "== $CORES cores -> N_WORKERS=$WORKERS (one core left for the main process)"

echo "== checks (fast, no games)"
ls -1 symaug_iter6.pt td_champion_July17_iter4.pt td_champion_July18_iter10.pt \
      td_champion_July19_iter14.pt td_champion_July21_aux_iter14.pt
python td_returns.py
python test_explore.py
python test_panel_gate.py 20000
PYTHONHASHSEED=0 python test_hand_rules.py

export BOARDGAME_DEVICE=cpu PYTHONHASHSEED=0 N_WORKERS="$WORKERS"
if [ "${SMOKE:-0}" = "1" ]; then
  export PREFIX="${PREFIX:-smoke}"
fi
LOG="${PREFIX:-league}_run.log"

if [ "${LAUNCH:-0}" = "1" ] || [ "${SMOKE:-0}" = "1" ]; then
  nohup python -u league_run.py >> "$LOG" 2>&1 &
  echo "== launched PID $! -> $DEST/$LOG   (follow: tail -f $LOG)"
else
  cat <<EOF

Setup done. To launch the full run (survives logout):

  cd $DEST && source .venv/bin/activate
  BOARDGAME_DEVICE=cpu PYTHONHASHSEED=0 N_WORKERS=$WORKERS nohup python -u league_run.py >> league_run.log 2>&1 &
  tail -f league_run.log

Resume after a stop: run the same command; it continues from league_live.pt.
Copy back when done: league_iter*.pt league_champion.pt league_live.pt
  league_gate_log.jsonl league_panel_cache.json league_config.json league_run.log
EOF
fi
