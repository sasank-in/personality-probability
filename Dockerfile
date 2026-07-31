# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first (better layer caching). CPU-only torch keeps the image small.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install ".[mistral]"

# Model artifacts
COPY artifacts ./artifacts

EXPOSE 8000

# Container-friendly defaults; override via env at runtime.
ENV PERSONALITY_LOG_LEVEL=INFO \
    HOST=0.0.0.0 \
    PORT=8000

# Non-root user
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"

CMD ["personality-serve"]
