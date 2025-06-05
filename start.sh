#!/bin/bash
set -e

echo "🚀 Starting Crypto Trading Application - Production Mode"
echo "======================================================="

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

# Set production environment variables
export FLASK_ENV=production
export PYTHONPATH=/app
export TRADING_ENV=production
export LOG_LEVEL=${LOG_LEVEL:-INFO}

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

# Create necessary directories
mkdir -p /app/trading/logs
mkdir -p /app/data/trading-logs
mkdir -p /app/data/ml-results
mkdir -p /app/data/historical-exports

# Set proper permissions
chown -R trader:trader /app/trading/logs /app/historical_exports /app/ml_results 2>/dev/null || true

echo "✅ Environment validated"
echo "🔑 Using API keys from .env file"

# ================================================================
# CRITICAL: Validate ML models and data before starting
# ================================================================
echo "🧪 Validating ML pipeline..."

# Check for ML models
MODEL_COUNT=$(find /app/ml_results -name "*_3class_enhanced.h5" -type f 2>/dev/null | wc -l)
SCALER_COUNT=$(find /app/ml_results -name "*_scaler.pkl" -type f 2>/dev/null | wc -l)
DATA_COUNT=$(find /app/historical_exports -name "*.csv" -type f 2>/dev/null | wc -l)

echo "📊 Found $MODEL_COUNT ML models, $SCALER_COUNT scalers, $DATA_COUNT data files"

echo "📊 Found $MODEL_COUNT ML models, $SCALER_COUNT scalers, $DATA_COUNT data files"

if [ "$MODEL_COUNT" -eq 0 ]; then
    echo "⚠️  No pre-trained ML models found - will train on first run"
fi

if [ "$DATA_COUNT" -eq 0 ]; then
    echo "❌ No historical data files found!"
    echo "💡 Please ensure historical data is included in the Docker image"
    exit 1
fi

echo "✅ ML pipeline validation complete"

# ================================================================
# Production Web Server Setup
# ================================================================
echo "🌐 Setting up production web server..."

# Start health check endpoint in background
echo "💓 Starting health check endpoint..."
python /app/health_endpoint.py &
HEALTH_PID=$!

# Create a simple web interface for monitoring
cat > /app/web_interface.py << 'EOF'
from flask import Flask, render_template_string, jsonify
import os
import json
from datetime import datetime
import requests

app = Flask(__name__)

MONITORING_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Crypto Trading Engine - Production Monitor</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .status-card { background: #2d2d2d; border-radius: 8px; padding: 20px; margin: 10px 0; }
        .status-healthy { border-left: 4px solid #4CAF50; }
        .status-warning { border-left: 4px solid #FF9800; }
        .status-error { border-left: 4px solid #f44336; }
        .metric { display: inline-block; margin: 10px 20px; }
        .refresh-btn { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        .logs { background: #000; color: #0f0; padding: 15px; border-radius: 4px; font-family: monospace; max-height: 400px; overflow-y: auto; }
    </style>
    <script>
        function refreshData() { location.reload(); }
        setInterval(refreshData, 30000); // Auto-refresh every 30 seconds
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Crypto Trading Engine</h1>
            <h2>Production Monitoring Dashboard</h2>
            <button class="refresh-btn" onclick="refreshData()">🔄 Refresh</button>
        </div>
        
        <div class="status-card status-healthy">
            <h3>📊 System Status</h3>
            <div class="metric">🕒 Uptime: {{ uptime }}</div>
            <div class="metric">🔄 Status: Running</div>
            <div class="metric">📈 Version: 1.0.0</div>
            <div class="metric">🐳 Container: crypto-trading-engine</div>
        </div>
        
        <div class="status-card">
            <h3>💾 Data & Models</h3>
            <div class="metric">📊 Data Files: {{ data_count }}</div>
            <div class="metric">🤖 ML Models: {{ model_count }}</div>
            <div class="metric">📁 Log Files: Available</div>
        </div>
        
        <div class="status-card">
            <h3>🔗 API Endpoints</h3>
            <div class="metric"><a href="/health" style="color: #4CAF50;">/health</a> - Full Health Check</div>
            <div class="metric"><a href="/health/simple" style="color: #4CAF50;">/health/simple</a> - Simple Status</div>
            <div class="metric"><a href="/health/ready" style="color: #4CAF50;">/health/ready</a> - Readiness Probe</div>
            <div class="metric"><a href="/health/live" style="color: #4CAF50;">/health/live</a> - Liveness Probe</div>
        </div>
        
        <div class="status-card">
            <h3>📋 Recent Activity</h3>
            <div class="logs">
                📅 {{ timestamp }}: Production container started<br>
                🔧 Environment: Production Mode<br>
                🔑 Authentication: Configured<br>
                🤖 ML Pipeline: Initialized<br>
                🌐 Web Interface: Active on port 8080<br>
                💓 Health Checks: Enabled<br>
            </div>
        </div>
    </div>
</body>
</html>
'''

@app.route('/')
def dashboard():
    # Get system info
    data_count = len([f for f in os.listdir('/app/historical_exports') if f.endswith('.csv')]) if os.path.exists('/app/historical_exports') else 0
    model_count = len([f for f in os.listdir('/app/ml_results') if f.endswith('.h5')]) if os.path.exists('/app/ml_results') else 0
    
    return render_template_string(MONITORING_TEMPLATE,
        uptime="Running",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        data_count=data_count,
        model_count=model_count
    )

@app.route('/api/status')
def api_status():
    return jsonify({
        'status': 'running',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0',
        'environment': 'production'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
EOF

# Start web interface
echo "🌐 Starting production web interface on port 8080..."
python /app/web_interface.py &
WEB_PID=$!

echo "✅ Production web server started (PID: $WEB_PID)"
echo "💓 Health endpoint running (PID: $HEALTH_PID)"
echo "🌐 Web interface available at http://localhost:8080"

# ================================================================
# Start Trading Application
# ================================================================
echo "🐳 Starting Docker Process Manager with enhanced output..."

# Start based on the command provided
if [ "$1" = "trading" ]; then
    echo "🚀 Starting full trading system..."
    exec python /app/docker_process_manager.py --output-dir /app/historical_exports
elif [ "$1" = "web-only" ]; then
    echo "🌐 Running in web-only mode..."
    wait $WEB_PID
elif [ "$1" = "bash" ]; then
    exec /bin/bash
elif [ "$1" = "trading-only" ]; then
    # Legacy mode: Start only trading engine
    exec python /app/trading/strategy_engine_refactored.py
elif [ "$1" = "original" ]; then
    # Use original process manager
    exec python /app/process_manager.py --output-dir /app/historical_exports
else
    # Default: Start enhanced Docker process manager
    exec python /app/docker_process_manager.py --output-dir /app/historical_exports
fi
