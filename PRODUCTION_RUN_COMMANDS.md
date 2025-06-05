# Docker Production Run Commands
## coinstardon/crypto-trading-engine:latest

## 🚀 Full Production Mode (Recommended)

### Basic Production Start
```bash
docker run -d \
  --name crypto-trading-production \
  --restart unless-stopped \
  -p 8080:8080 \
  coinstardon/crypto-trading-engine:latest
```

### Enhanced Production Mode (With Persistent Logs)
```bash
docker run -d \
  --name crypto-trading-production \
  --restart unless-stopped \
  -p 8080:8080 \
  -v $(pwd)/logs:/app/trading/logs \
  -v $(pwd)/exports:/app/historical_exports \
  coinstardon/crypto-trading-engine:latest
```

### Full Production Mode (With Environment Override)
```bash
docker run -d \
  --name crypto-trading-production \
  --restart unless-stopped \
  -p 8080:8080 \
  -v $(pwd)/logs:/app/trading/logs \
  -v $(pwd)/exports:/app/historical_exports \
  -e BINANCE_API_KEY="your_binance_api_key" \
  -e BINANCE_API_SECRET="your_binance_secret" \
  -e ALPACA_API_KEY="your_alpaca_key" \
  -e ALPACA_SECRET_KEY="your_alpaca_secret" \
  coinstardon/crypto-trading-engine:latest
```

### Maximum Production Mode (With Resource Limits)
```bash
docker run -d \
  --name crypto-trading-production \
  --restart unless-stopped \
  -p 8080:8080 \
  -v $(pwd)/logs:/app/trading/logs \
  -v $(pwd)/exports:/app/historical_exports \
  -v $(pwd)/ml_results:/app/ml_results \
  --memory=4g \
  --cpus=2.0 \
  --health-cmd="python /app/healthcheck.py" \
  --health-interval=30s \
  --health-timeout=10s \
  --health-retries=3 \
  coinstardon/crypto-trading-engine:latest
```

## 🔧 Command Components Explained

### Core Parameters
```bash
-d                          # Run in detached mode (background)
--name crypto-trading-production  # Container name
--restart unless-stopped    # Auto-restart on failure/reboot
-p 8080:8080               # Expose port 8080
```

### Volume Mounts (Optional but Recommended)
```bash
-v $(pwd)/logs:/app/trading/logs           # Persistent trading logs
-v $(pwd)/exports:/app/historical_exports  # Historical data exports
-v $(pwd)/ml_results:/app/ml_results      # ML model results
```

### Environment Variables (Optional - .env file included)
```bash
-e BINANCE_API_KEY="your_key"
-e BINANCE_API_SECRET="your_secret"
-e ALPACA_API_KEY="your_key"
-e ALPACA_SECRET_KEY="your_secret"
```

### Resource Limits (Production Recommended)
```bash
--memory=4g          # Limit memory to 4GB
--cpus=2.0          # Limit to 2 CPU cores
```

### Health Checks
```bash
--health-cmd="python /app/healthcheck.py"
--health-interval=30s
--health-timeout=10s
--health-retries=3
```

## 🎯 What Full Production Mode Includes

✅ **Trading Engine**: Real-time crypto trading with ML predictions
✅ **Historical Data Export**: Every 2 hours, 7-day retention
✅ **ML Training**: Periodic model retraining
✅ **Health Monitoring**: Built-in health checks
✅ **Auto-restart**: Container restarts on failure
✅ **Persistent Logs**: Trading logs saved to host
✅ **Resource Management**: Memory and CPU limits

## 📊 Monitoring Commands

### Check Status
```bash
docker ps | grep crypto-trading-production
docker logs crypto-trading-production
docker logs -f crypto-trading-production  # Follow logs
```

### Health Check
```bash
docker inspect crypto-trading-production --format='{{.State.Health.Status}}'
```

### Resource Usage
```bash
docker stats crypto-trading-production
```

### Execute Commands
```bash
docker exec -it crypto-trading-production bash
```

## 🛑 Stop/Restart

### Stop
```bash
docker stop crypto-trading-production
```

### Restart
```bash
docker restart crypto-trading-production
```

### Remove
```bash
docker stop crypto-trading-production
docker rm crypto-trading-production
```

## 🔄 Update to Latest

### Pull and Restart
```bash
docker pull coinstardon/crypto-trading-engine:latest
docker stop crypto-trading-production
docker rm crypto-trading-production
# Then run with your preferred command above
```

## 📋 Current Status
- **Container Name**: crypto-trading-full (currently running)
- **Image**: coinstardon/crypto-trading-engine:latest
- **Status**: All 3 modes active (Trading, Export, ML Training)
- **Port**: 8080 exposed
- **Health**: Healthy
