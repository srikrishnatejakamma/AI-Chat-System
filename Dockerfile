FROM node:22-alpine AS frontend
WORKDIR /frontend
COPY frontend/package*.json ./
RUN --mount=type=cache,target=/root/.npm npm ci --no-fund --no-audit
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.10.2 /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ANONYMIZED_TELEMETRY=False \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    DATA_DIR=/app/data

WORKDIR /app
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/uv UV_HTTP_TIMEOUT=1800 uv pip install --system -r requirements.txt
COPY . .
COPY --from=frontend /frontend/dist ./frontend/dist
RUN useradd --create-home --uid 10001 appuser && mkdir -p /app/data && chown appuser:appuser /app/data
USER appuser
EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/api/health', timeout=3)"
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8501"]
