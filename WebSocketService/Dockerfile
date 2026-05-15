FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip || true
RUN for i in 1 2 3 4 5; do \
    pip install --no-cache-dir --default-timeout=100 \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    -r requirements.txt && break || \
    (echo "Retrying $i..." && sleep 5); \
    done

COPY . .

EXPOSE 9004

CMD ["python", "server.py"]