FROM python:3.11-slim
WORKDIR /app
COPY requirements-coordinator.txt .
RUN pip install --no-cache-dir -r requirements-coordinator.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu
COPY . .
