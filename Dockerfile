# ================================================================
# Crypto Trading App - Production Dockerfile
# ================================================================
# Multi-stage build for remote deployment
# Optimized for coinstardon/crypto-trading-engine:latest
# Includes .env, ML models, and historical data for full functionality
# ================================================================

# Stage 1: Base Python Image with System Dependencies
# ================================================================
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Create non-root user for security
RUN groupadd --gid 1000 trader && \
    useradd --uid 1000 --gid trader --shell /bin/bash --create-home trader

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# ================================================================
# Stage 2: Dependencies Installation
# ================================================================
FROM base as dependencies

# Set work directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# ================================================================
# Stage 3: Application Build
# ================================================================
FROM dependencies as application

# Set work directory
WORKDIR /app

# Create necessary directories with proper permissions
RUN mkdir -p /app/trading/logs \
    /app/historical_exports \
    /app/binance_exports \
    /app/ml_results \
    /app/data \
    /app/backups \
    && chown -R trader:trader /app

# Copy ALL application files including .env and supporting files
COPY --chown=trader:trader . .

# Verify .env file was included in the copy
RUN test -f .env && echo "✅ .env file included" || echo "⚠️  No .env file found - using environment variables"

# ================================================================
# VERIFICATION: Ensure ALL files are included
# ================================================================

# Verify that ML model files are included
RUN echo "🔍 Checking for ML models..." && \
    find /app/ml_results -name "*.h5" -type f | head -10 && \
    find /app/ml_results -name "*.pkl" -type f | head -10 && \
    echo "📊 ML models check complete"

# Verify that historical data files are included  
RUN echo "🔍 Checking for historical data..." && \
    find /app/historical_exports -name "*.csv" -type f | head -5 && \
    echo "📈 Historical data check complete"

# Verify .env and configuration files
RUN echo "🔍 Checking configuration files..." && \
    ls -la /app/.env* 2>/dev/null || echo "No .env files found" && \
    ls -la /app/config/ && \
    echo "⚙️  Configuration check complete"

# Verify all supporting files are present
RUN echo "🔍 Checking supporting files..." && \
    ls -la /app/*.py /app/*.sh /app/*.md 2>/dev/null | head -10 && \
    echo "📁 Supporting files check complete"

# Set proper file permissions for ALL data directories
RUN chown -R trader:trader /app

# Set executable permissions for all scripts
RUN find /app -name "*.sh" -type f -exec chmod +x {} \; && \
    find /app -name "*.py" -type f -exec chmod +x {} \; && \
    chmod +x /app/start.sh /app/docker_process_manager.py /app/healthcheck.py

# Create data persistence volumes for runtime data only
VOLUME ["/app/trading/logs"]

# ================================================================
# Stage 4: Production Image
# ================================================================
FROM application as production

# Install additional production dependencies
RUN pip install --no-cache-dir gunicorn supervisor

# Switch to non-root user
USER trader

# Set Python path
ENV PYTHONPATH=/app

# Set production environment variables
ENV FLASK_ENV=production \
    PYTHONPATH=/app \
    TRADING_ENV=production \
    LOG_LEVEL=INFO \
    MAX_WORKERS=4

# Expose port for web interface
EXPOSE 8080

# Set default command
CMD ["/app/start.sh", "trading"]

# Health check with better endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || python /app/healthcheck.py || exit 1

# ================================================================
# Build Information & Labels
# ================================================================
LABEL maintainer="Coinstardon <coinstardon@example.com>" \
      version="1.0.0" \
      description="Production Crypto Trading Engine with ML Models and Historical Data" \
      org.opencontainers.image.title="crypto-trading-engine" \
      org.opencontainers.image.description="Automated Crypto Trading Application with ML Predictions - Production Ready" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.vendor="Coinstardon" \
      org.opencontainers.image.url="https://hub.docker.com/r/coinstardon/crypto-trading-engine" \
      org.opencontainers.image.documentation="https://github.com/coinstardon/crypto-trading-engine" \
      org.opencontainers.image.source="https://github.com/coinstardon/crypto-trading-engine" \
      org.opencontainers.image.licenses="MIT"