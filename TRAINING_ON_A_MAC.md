# Training run on an Apple-silicon Mac (e.g. the work iMac M4)

Written 2026-10-05 for branch `train-features-v2`. Assumes a Mac with nothing
installed. Every command goes into **Terminal** (Applications > Utilities >
Terminal). Lines starting with `#` are comments; don't type them.

## 1. Developer tools (gives you `git`)

    git --version

If it prints a version, skip to step 2. If a dialog offers to install the
"command line developer tools", click **Install** (a few minutes, ~1 GB). Or run:

    xcode-select --install

## 2. Python 3.11

Download the **macOS 64-bit universal2 installer** for Python **3.11** from
https://www.python.org/downloads/macos/ and run it (defaults are fine). Then
close Terminal, open a new window, and check:

    python3.11 --version        # should print Python 3.11.x

## 3. Get the code

    git clone --branch train-features-v2 https://github.com/tomrecht/boardgame_koyeb.git ~/quahuru-train
    cd ~/quahuru-train

The clone carries the warm starts (`symaug6_AB.pt`, `iter10_AB.pt`, and the A
versions), the panel nets, the endgame table and the calibration benchmark.

## 4. Python environment

    cd ~/quahuru-train
    python3.11 -m venv .venv
    source .venv/bin/activate          # your prompt now starts with (.venv)
    pip install --upgrade pip
    pip install torch "numpy<2" scipy

`source .venv/bin/activate` is needed again in every NEW Terminal window before
running anything below.

## 5. Copy the start-position pool (not in git: it is your games)

From the home iMac copy `start_pool.jsonl` (about 10 MB) into `~/quahuru-train/`
on the work iMac -- AirDrop, iCloud Drive, a USB stick or email all work. Check:

    ls -lh ~/quahuru-train/start_pool.jsonl

Without it the run still works; it just plays every game from the opening.

## 6. Quick checks (a minute)

    cd ~/quahuru-train && source .venv/bin/activate
    python td_returns.py                 # ends "All td_returns tests passed."
    python test_panel_gate.py            # ends "ALL OK"
    python -c "import torch; print('MPS GPU available:', torch.backends.mps.is_available())"

## 7. Smoke run (measures this Mac; ~30-60 minutes)

Keep the Mac awake for it (`caffeinate -i`). First with the GPU for training and
CPU workers:

    mkdir -p smoke
    caffeinate -i env BOARDGAME_DEVICE=mps PYTHONHASHSEED=0 SMOKE=1 PREFIX=smoke/cpu \
        WARM_START=symaug6_AB.pt PRESCREEN_BAR=-12 python -u league_run.py > smoke/cpu.log 2>&1

When it returns, look at the end:

    grep -E "Feature set|Start pool|Calibration|s/game|PANEL GATE|Traceback|Error" smoke/cpu.log

You want: `Feature set: AB`, the start pool line, a `PANEL GATE ... CAP` line,
and NO `Traceback`. Note the `s/game` numbers.

Then the same with the GPU for the workers too, to compare:

    caffeinate -i env BOARDGAME_DEVICE=mps WORKER_DEVICE=mps PYTHONHASHSEED=0 SMOKE=1 \
        PREFIX=smoke/mps WARM_START=symaug6_AB.pt PRESCREEN_BAR=-12 python -u league_run.py > smoke/mps.log 2>&1
    grep -E "s/game|PANEL GATE|Traceback|Error" smoke/mps.log

Use `WORKER_DEVICE=mps` in the real run only if its `s/game` is clearly lower and
it finished without errors. If a run dies with an `MPSNDArray...` assertion, MPS
is not usable there: use `BOARDGAME_DEVICE=cpu` instead.

## 8. The real run (arm B: AB features, warm start = deployed champion)

    cd ~/quahuru-train && source .venv/bin/activate
    mkdir -p runs
    nohup caffeinate -is env BOARDGAME_DEVICE=mps PYTHONHASHSEED=0 PREFIX=runs/fv2B \
        WARM_START=symaug6_AB.pt python -u league_run.py >> runs/fv2B.log 2>&1 &

(Add `WORKER_DEVICE=mps` after `BOARDGAME_DEVICE=mps` if step 7 said so.) It
picks one worker per core minus one automatically. You can close Terminal.

The iter10 arm is the same with `PREFIX=runs/fv2B10 WARM_START=iter10_AB.pt` --
run the two one after the other, not at the same time (they would share cores).

## 9. Watching it

    tail -f ~/quahuru-train/runs/fv2B.log          # Ctrl-C stops watching, not the run
    grep -E "PANEL GATE|calibration benchmark|promoted" ~/quahuru-train/runs/fv2B.log

Each iteration ends with a `PANEL GATE ... outcome:` line (PROMOTE / REJECT / CAP
/ GUARD_FAIL) and a `calibration benchmark itN: corr ... slope ...` line
(deployed champion: corr +0.090, slope 0.297 -- higher is better).

## 10. Stopping, resuming, sleep

* **Stop:** `pkill -f league_run.py`
* **Resume:** run the step-8 command again, unchanged. It resumes from the last
  finished iteration (`runs/fv2B_live.pt`); at most the iteration in progress is
  lost.
* **Sleep:** `caffeinate` keeps the Mac awake while the run lasts. Locking the
  screen is fine. Logging out, restarting or shutting down stops the run --
  just resume afterwards.
* In System Settings > Energy (or Displays > Advanced), "Prevent automatic
  sleeping when the display is off" is a useful belt-and-braces setting.

## 11. Bringing results back

Everything is in `~/quahuru-train/runs/`: `fv2B.log`, `fv2B_gate_log.jsonl`,
`fv2B_champion.pt` (the best net so far) and `fv2B_iter*.pt`. Copy the folder
back to the home iMac (AirDrop/iCloud), or just send the log and the
`_champion.pt`.

## Expected speed (estimate, not measured)

M4 performance core ~90 s per AB game, efficiency core ~250 s; about 230 games
an hour on all cores, so ~3.5 h per iteration and ~3 days of compute for 20
iterations -- roughly 5-6 days if it only runs when the Mac is idle. The smoke
run's `s/game` lines replace these guesses with real numbers.
