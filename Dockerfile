# ================================================================
# Crypto Trading App - Production Dockerfile
# ================================================================
# Multi-stage Docker build for the crypto trading application
# Optimized for production deployment with proper security and
# performance considerations.
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
    && chown -R trader:trader /app

# Copy application code
COPY --chown=trader:trader . .

# Ensure .env file is available for configuration
COPY --chown=trader:trader .env .env

# Copy and set permissions for startup and health check scripts
COPY --chown=trader:trader start.sh /app/start.sh
COPY --chown=trader:trader healthcheck.py /app/healthcheck.py
RUN chmod +x /app/start.sh /app/healthcheck.py

# Create data persistence volumes (excluding historical_exports to use image data)
VOLUME ["/app/trading/logs", "/app/ml_results"]

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

# Expose port for potential web interface (future enhancement)
EXPOSE 8080

# Set default command
CMD ["/app/start.sh", "trading"]

# Health check
HEALTHCHECK --interval=5m --timeout=30s --start-period=2m --retries=3 \
    CMD python /app/healthcheck.py

# ================================================================
# Build Information
# ================================================================
LABEL maintainer="Crypto Trading Team" \
      version="1.0.0" \
      description="Automated Crypto Trading Application with ML Predictions" \
      org.opencontainers.image.source="https://github.com/your-repo/crypto-trading" \
      org.opencontainers.image.documentation="https://github.com/your-repo/crypto-trading/README.md"