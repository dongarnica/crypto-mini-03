#!/bin/bash
set -e

echo "🚀 Starting Crypto Trading Application"
echo "======================================"

# Check if .env file exists
if [ ! -f /app/.env ]; then
    echo "❌ Error: .env file not found"
    echo "💡 Please ensure .env file is mounted or copied to /app/.env"
    exit 1
fi

# Source environment variables from .env file
set -a
source /app/.env
set +a

# Validate required environment variables
if [ -z "$BINANCE_API_KEY" ] || [ -z "$BINANCE_API_SECRET" ]; then
    echo "❌ Error: Binance API credentials not found in .env file"
    echo "💡 Please ensure BINANCE_API_KEY and BINANCE_API_SECRET are set in .env"
    exit 1
fi

if [ -z "$ALPACA_API_KEY" ] || [ -z "$ALPACA_SECRET_KEY" ]; then
    echo "⚠️ Warning: Alpaca API credentials not found in .env file"
    echo "💡 Trading will use Binance-only mode"
fi

# Create logs directory if it does not exist
mkdir -p /app/trading/logs

# Set proper permissions
chown -R trader:trader /app/trading/logs /app/historical_exports /app/ml_results 2>/dev/null || true

echo "✅ Environment validated"
echo "🔑 Using API keys from .env file"
echo "📊 Starting trading engine..."

# Start based on the command provided
if [ "$1" = "trading" ]; then
    exec python /app/trading/strategy_engine_refactored.py
elif [ "$1" = "bash" ]; then
    exec /bin/bash
else
    # Default: Start trading engine
    exec python /app/trading/strategy_engine_refactored.py
fi
