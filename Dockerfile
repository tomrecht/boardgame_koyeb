# CPython, not PyPy: onnxruntime ships CPython wheels only. (Neither did torch,
# which the old pypy base could never actually have installed.)
FROM python:3.11-slim

WORKDIR /app

# Requirements first, for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Precompress the inference runtime: 10.5MB -> 2.8MB. Done once here rather
# than per request, which would burn CPU on a small instance for every new
# client. app.py serves the .gz when the client accepts it.
RUN gzip -9 -k ort/ort-wasm-simd-threaded.wasm && ls -la ort/

# The agent runs off model.onnx (see _opponent_model_path in app.py), so the
# image needs neither torch nor any .pt checkpoint.
ENV OPPONENT_MODEL=model.onnx
# Smaller feature cache than training uses: ~25 MB less resident per worker.
ENV ROW_CACHE_MAX=20000
# Prefer strong-and-slow: give move selection as long as it can have without
# tripping a timeout. On eMicro the heaviest midgame positions land around
# 15-20s (p50 is nearer 2s), so 40s means they are searched in full and the
# budget effectively never bites -- while still leaving room under the 60s
# worker timeout and the platform's own request timeout.
#
# The ceiling here is the PLATFORM's request timeout, not this: Koyeb returns a
# 504 at 60s unless the service is configured otherwise. Raise that first, then
# MOVE_BUDGET and WEB_TIMEOUT together.
ENV MOVE_BUDGET=40
ENV WEB_TIMEOUT=90

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
