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

# Move selection is CPU-bound Python, so real parallelism comes from workers
# (separate processes, ~90 MB each); threads only let a second player's request
# start while another is mid-move. Both are safe now that the board is
# per-thread and difficulty is passed per call. Raise WEB_WORKERS if the
# instance has the memory.
CMD gunicorn --bind 0.0.0.0:$PORT --timeout 120 --workers ${WEB_WORKERS:-2} --threads 2 app:app
