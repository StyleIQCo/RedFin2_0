# syntax=docker/dockerfile:1.6
# Multi-stage build: small runtime image (< 500 MB) with baked-in models.
#
# Stage 1: builder — install deps, train models, write artifacts to registry/
# Stage 2: runtime — slim python + only the compiled deps + registry/

FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# System deps needed for LightGBM native build
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

COPY . .

# Pre-train so the runtime container starts with a production model in place.
# In real infra we'd instead pull a vetted model from the registry bucket
# by SHA; pre-training here keeps the demo single-command.
ENV PYTHONPATH=/app
RUN python -m data.generate --n 20000 \
 && python -m src.avm.train \
 && python -m src.recommender.train

# --- runtime ---
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 redfin

COPY --from=builder /install /usr/local
COPY --from=builder /app /app
RUN chown -R redfin:redfin /app

USER redfin

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
