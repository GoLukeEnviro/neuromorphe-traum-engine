# Neuromorphe Traum-Engine v2.0 - Docker Image
# Multi-stage build für optimierte Produktions-Images

# Build Stage
FROM python:3.11-slim as builder

# Build-Abhängigkeiten installieren
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    pkg-config \
    libffi-dev \
    libssl-dev \
    libsndfile1-dev \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python-Abhängigkeiten installieren
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production Stage
FROM python:3.11-slim as production

# System-Abhängigkeiten für Runtime
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Non-root User erstellen
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Python-Pakete von Build Stage kopieren
COPY --from=builder /root/.local /home/appuser/.local

# Arbeitsverzeichnis erstellen
WORKDIR /app

# Anwendungscode kopieren
COPY src/ ./src/
COPY run.py .
COPY README.md .

# Verzeichnisse für Daten erstellen
RUN mkdir -p \
    /app/data/audio_input \
    /app/data/audio_output \
    /app/data/processed_database \
    /app/data/generated_tracks \
    /app/data/database \
    /app/logs \
    && chown -R appuser:appuser /app

# Environment Variables
ENV PYTHONPATH=/app/src
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default-Konfiguration
ENV DATABASE_URL=sqlite:///app/data/database/neuromorphe_engine.db
ENV AUDIO_INPUT_DIR=/app/data/audio_input
ENV AUDIO_OUTPUT_DIR=/app/data/audio_output
ENV PROCESSED_DATABASE_DIR=/app/data/processed_database
ENV GENERATED_TRACKS_DIR=/app/data/generated_tracks
ENV LOG_DIR=/app/logs
ENV ENVIRONMENT=production

# User wechseln
USER appuser

# Health Check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Port freigeben
EXPOSE 8000

# Startup Command
CMD ["python", "-m", "src.main", "--mode", "prod", "--host", "0.0.0.0", "--port", "8000"]

# Development Stage
FROM production as development

# Development-Tools installieren
USER root
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    flake8 \
    mypy \
    jupyter \
    ipython

# Development-Konfiguration
ENV ENVIRONMENT=development
ENV LOG_LEVEL=DEBUG
ENV DATABASE_ECHO=true

# Development-User
USER appuser

# Development Command
CMD ["python", "-m", "src.main", "--mode", "dev", "--host", "0.0.0.0", "--port", "8000", "--log-level", "DEBUG"]