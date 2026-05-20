FROM pypy:3.10-slim

WORKDIR /workspace

# Install system dependencies 
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Run gunicorn (adjust timeout to 60-120 seconds for safety)
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--timeout", "60", "app:app"]