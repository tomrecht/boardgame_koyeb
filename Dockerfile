# CPython, not PyPy: onnxruntime ships CPython wheels only. (Neither did torch,
# which the old pypy base could never actually have installed.)
FROM python:3.11-slim

WORKDIR /app

# Requirements first, for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The agent runs off model.onnx (see _opponent_model_path in app.py), so the
# image needs neither torch nor any .pt checkpoint.
ENV OPPONENT_MODEL=model.onnx
# Smaller feature cache than training uses: ~25 MB less resident per worker.
ENV ROW_CACHE_MAX=20000
# Sized for a fraction-of-a-vCPU instance: locally a move is p50 0.4s / p99 3s,
# so on eMicro the tail lands around 15-20s. Past this the agent chooses from
# the candidates it has rather than making the player wait.
ENV MOVE_BUDGET=10

# ONE worker by default: each is a separate process with its own copy of the
# net (~90 MB), and two of them on a small instance ran the box out of memory --
# gunicorn killed a worker mid-request (WORKER TIMEOUT then SIGKILL) and the app
# was unreachable while it restarted. Threads share that memory, so a second
# player's request can start while one is mid-move; the GIL means they take
# turns on the CPU, which is the right trade on a fractional vCPU. Raise
# WEB_WORKERS only on an instance with the memory to spare.
#
# The timeout is 60s, not 120s: a request that overruns should recycle quickly
# rather than hold the worker. MOVE_BUDGET (app.py) keeps move selection well
# under it, so the timeout is a backstop, not the mechanism.
CMD gunicorn --bind 0.0.0.0:$PORT --timeout ${WEB_TIMEOUT:-60} --graceful-timeout 20 \
    --workers ${WEB_WORKERS:-1} --threads ${WEB_THREADS:-2} \
    --max-requests 2000 --max-requests-jitter 200 app:app
