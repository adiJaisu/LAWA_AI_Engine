# Use a stable Python base image
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for OpenCV and ffmpeg
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip || true
RUN for i in 1 2 3 4 5; do \
    pip install --no-cache-dir --default-timeout=100 \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    -r requirements.txt && break || \
    (echo "Retrying $i..." && sleep 5); \
    done

# Copy application code
COPY . /app

# Create directory for evidence storage
RUN mkdir -p /app/evidence_storage && chmod 777 /app/evidence_storage

# Expose WebSocket port
EXPOSE 9004

CMD ["python", "main.py"]
