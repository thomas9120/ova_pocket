FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# --- System dependencies + Python 3.13 ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.13 python3.13-venv python3.13-dev curl && \
    ln -sf /usr/bin/python3.13 /usr/bin/python3 && \
    rm -rf /var/lib/apt/lists/*

# --- uv package manager ---
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# --- Install Python dependencies (cached layer) ---
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# --- Copy application code ---
COPY . .
RUN uv sync --frozen

EXPOSE 5173 8000

# Start both the web UI (port 8000) and the API backend (port 5173).
# Models are downloaded on first launch and cached in the hf-cache volume.
CMD uv run python3 -m http.server --bind 0.0.0.0 --directory /app 8000 & \
    exec uv run uvicorn ova.api:app --host 0.0.0.0 --port 5173
