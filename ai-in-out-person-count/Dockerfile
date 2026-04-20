FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHON_VERSION=3.10

# Install Python and minimal system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3-pip \
        ffmpeg \
        ca-certificates \
        libturbojpeg && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    python3.10 -m pip install --upgrade pip setuptools wheel && \
    rm -rf /var/lib/apt/lists/*
    
# Create working directory
WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Install Python dependencies before copying entire project
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files after dependencies are installed
COPY . /app/

ENV RABBITMQ_HOST=rabbitmq
ENV RABBITMQ_PORT=5672
ENV EXCHANGE_NAME=LAWA_exchange
ENV EXCHANGE_TYPE=direct
ENV HOST=0.0.0.0
ENV PORT=5002

EXPOSE 5002

# Start application
CMD ["python3", "app.py", "--host", "$HOST", "--port", "$PORT"]
