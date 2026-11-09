# ==============================================================================
# BoliBench — Production Dockerfile for Railway Deployment
# ==============================================================================

# 1. Base Image: Official minimal Python 3.13 slim image
FROM python:3.13-slim

# 2. Prevent Python bytecode writing and enable unbuffered standard I/O
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    APP_ENV=production \
    BOLIBENCH_DATA_DIR=/data

# 3. Install FFmpeg for audio processing, curl for health checks, and gosu for root privilege drop
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
        gosu && \
    rm -rf /var/lib/apt/lists/*

# 4. Set working directory
WORKDIR /app

# 5. Create non-root runtime user (UID 10001) and persistent data mount point
RUN useradd -m -u 10001 -s /bin/bash appuser && \
    mkdir -p /data && \
    chown -R 10001:10001 /data /app

# 6. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy application code and entrypoint (excluding files listed in .dockerignore)
COPY app/ ./app/
COPY entrypoint.sh ./entrypoint.sh

# Ensure entrypoint is executable and set ownership of application files to non-root user
RUN chmod +x ./entrypoint.sh && \
    chown -R 10001:10001 /app

# 8. Expose default port
EXPOSE 8000

# 9. Container Health Check using /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://127.0.0.1:${PORT:-8000}/health || exit 1

# 10. Entrypoint: fixes volume permissions if needed, verifies write access, drops to appuser (UID 10001)
ENTRYPOINT ["/app/entrypoint.sh"]

# 11. Start service: single Uvicorn worker bound to 0.0.0.0 and dynamic runtime PORT
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
