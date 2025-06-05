#!/bin/bash
# ================================================================
# Production Deployment Script for Crypto Trading Engine
# ================================================================
# This script builds and runs the production-ready crypto trading
# container with all necessary files and configurations included.
# ================================================================

set -e

echo "🚀 Crypto Trading Engine - Production Deployment"
echo "================================================="

# Build the production container
echo "🔨 Building production container..."
docker build --no-cache -t coinstardon/crypto-trading-engine:production .

# Verify the build was successful
if [ $? -eq 0 ]; then
    echo "✅ Production container built successfully"
else
    echo "❌ Container build failed"
    exit 1
fi

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker stop crypto-trading-engine 2>/dev/null || true
docker rm crypto-trading-engine 2>/dev/null || true

# Verify .env file is included
echo "🔍 Verifying .env file is included..."
ENV_CHECK=$(docker run --rm coinstardon/crypto-trading-engine:production test -f /app/.env && echo "FOUND" || echo "MISSING")
if [ "$ENV_CHECK" = "FOUND" ]; then
    echo "✅ .env file is properly included in container"
else
    echo "❌ .env file is missing from container"
    exit 1
fi

# Verify historical data is included
echo "🔍 Verifying historical data..."
DATA_COUNT=$(docker run --rm coinstardon/crypto-trading-engine:production find /app/historical_exports -name "*.csv" -type f | wc -l)
echo "📊 Found $DATA_COUNT CSV data files"

if [ "$DATA_COUNT" -gt 0 ]; then
    echo "✅ Historical data is properly included"
else
    echo "❌ No historical data found"
    exit 1
fi

# Verify ML models are included
echo "🔍 Verifying ML models..."
MODEL_COUNT=$(docker run --rm coinstardon/crypto-trading-engine:production find /app/ml_results -name "*.h5" -type f | wc -l)
echo "🤖 Found $MODEL_COUNT ML models"

if [ "$MODEL_COUNT" -gt 0 ]; then
    echo "✅ ML models are properly included"
else
    echo "⚠️  No pre-trained ML models found - will train on first run"
fi

echo ""
echo "🎯 Production container ready for deployment!"
echo "================================================="

# Create production deployment commands
echo "📝 Production Deployment Commands:"
echo ""
echo "# Run in production mode with persistent logs:"
echo "docker run -d --name crypto-trading-engine \\"
echo "  --restart unless-stopped \\"
echo "  -p 8080:8080 \\"
echo "  -v crypto-trading-logs:/app/trading/logs \\"
echo "  coinstardon/crypto-trading-engine:production"
echo ""
echo "# Monitor container logs:"
echo "docker logs -f crypto-trading-engine"
echo ""
echo "# Check container status:"
echo "docker ps | grep crypto-trading-engine"
echo ""
echo "# Stop container:"
echo "docker stop crypto-trading-engine"
echo ""

# Ask if user wants to run the container now
read -p "🚀 Do you want to start the production container now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Starting production container..."
    
    # Create a persistent volume for logs
    docker volume create crypto-trading-logs 2>/dev/null || true
    
    # Run the production container
    docker run -d \
        --name crypto-trading-engine \
        --restart unless-stopped \
        -p 8080:8080 \
        -v crypto-trading-logs:/app/trading/logs \
        coinstardon/crypto-trading-engine:production
    
    echo "✅ Production container started successfully!"
    echo "📊 Container ID: $(docker ps -q --filter name=crypto-trading-engine)"
    echo ""
    echo "📋 Monitor with: docker logs -f crypto-trading-engine"
    echo "🔍 Check status: docker ps | grep crypto-trading-engine"
    
    # Show initial logs
    echo ""
    echo "📊 Initial container logs:"
    echo "=========================="
    sleep 3
    docker logs crypto-trading-engine
else
    echo "📝 Container built and ready - use the commands above to deploy"
fi

echo ""
echo "🎉 Production deployment script completed!"
