# Stage 1: Build environment
FROM python:3.11-slim AS builder

RUN apt update && apt install -y --no-install-recommends \
    build-essential libxml2-dev libfreetype6-dev libpng-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./ ./

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U pip && \
    pip install .[nogpu]

# Stage 2: Clean final image
FROM python:3.11-slim

WORKDIR /app

# Copy python packages
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY ./ ./

CMD ["python", "your_script.py"]
