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

# One worker, one thread on purpose: app.py keeps a single shared Board and a
# single agent instance across requests, so concurrent handlers would corrupt
# each other's state. A move takes ~0.1-0.5s, so requests just queue.
CMD gunicorn --bind 0.0.0.0:$PORT --timeout 120 --workers 1 --threads 1 app:app
