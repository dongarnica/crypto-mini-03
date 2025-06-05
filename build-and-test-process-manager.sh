#!/bin/bash
# Quick Build and Test Script for Process Manager Integration

echo "🚀 Building Docker image with Process Manager..."
echo "================================================="

# Build the image
docker build -t crypto-trading-app .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed!"
    exit 1
fi

echo "✅ Docker build successful!"
echo ""

# Test the container for 30 seconds to see countdown timers
echo "🧪 Testing container with countdown timers (30 seconds)..."
echo "========================================================="

# Create a temporary .env file for testing if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating temporary .env file for testing..."
    cat > .env << EOF
BINANCE_API_KEY=test_key
BINANCE_API_SECRET=test_secret
ALPACA_API_KEY=test_key
ALPACA_SECRET_KEY=test_secret
TRADING_MODE=demo
EOF
fi

# Run container for testing
docker run --rm \
    --name crypto-trading-test \
    --env-file .env \
    -v $(pwd)/historical_exports:/app/historical_exports \
    crypto-trading-app &

CONTAINER_PID=$!

echo "Container started with PID $CONTAINER_PID"
echo "Waiting 5 seconds for startup..."
sleep 5

echo ""
echo "📊 Showing countdown timers for 20 seconds..."
echo "=============================================="

# Show logs for 20 seconds
timeout 20 docker logs -f crypto-trading-test 2>/dev/null || true

echo ""
echo "🛑 Stopping test container..."
docker stop crypto-trading-test 2>/dev/null || true
docker rm crypto-trading-test 2>/dev/null || true

echo ""
echo "✅ Test completed!"
echo ""
echo "🚀 To run in production:"
echo "  docker run -d --name crypto-trading --env-file .env crypto-trading-app"
echo ""
echo "📊 To view countdown timers:"
echo "  docker logs -f crypto-trading"
echo ""
echo "🔍 To check health:"
echo "  docker exec crypto-trading python healthcheck.py"
