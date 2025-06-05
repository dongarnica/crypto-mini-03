# 🚀 Running coinstardon/crypto-trading-engine:latest

## Quick Start Commands

### 1. Basic Run (Uses built-in .env file)
```bash
docker run -d --name crypto-trading coinstardon/crypto-trading-engine:latest
```

### 2. Run with Custom Environment Variables
```bash
docker run -d --name crypto-trading \
  -e BINANCE_API_KEY="your_binance_api_key" \
  -e BINANCE_API_SECRET="your_binance_secret_key" \
  -e ALPACA_API_KEY="your_alpaca_api_key" \
  -e ALPACA_SECRET_KEY="your_alpaca_secret_key" \
  coinstardon/crypto-trading-engine:latest
```

### 3. Run with Volume Mounts (Recommended for production)
```bash
docker run -d --name crypto-trading \
  -v $(pwd)/logs:/app/trading/logs \
  -v $(pwd)/custom.env:/app/.env \
  coinstardon/crypto-trading-engine:latest
```

### 4. Run with Port Exposure (for web interface - future feature)
```bash
docker run -d --name crypto-trading \
  -p 8080:8080 \
  coinstardon/crypto-trading-engine:latest
```

## Available Run Modes

The container supports different startup modes:

### Default Mode (Recommended)
```bash
docker run -d --name crypto-trading coinstardon/crypto-trading-engine:latest
# Starts enhanced Docker process manager with full functionality
```

### Trading Only Mode
```bash
docker run -d --name crypto-trading coinstardon/crypto-trading-engine:latest trading-only
# Starts only the trading engine without data collection
```

### Interactive Shell Mode
```bash
docker run -it --name crypto-trading coinstardon/crypto-trading-engine:latest bash
# Opens a bash shell inside the container for debugging
```

### Original Process Manager
```bash
docker run -d --name crypto-trading coinstardon/crypto-trading-engine:latest original
# Uses the original process manager (legacy mode)
```

## Monitoring and Management

### Check Container Status
```bash
# Check if container is running
docker ps | grep crypto-trading

# Check container health
docker inspect crypto-trading | grep Health -A 10

# View container logs
docker logs crypto-trading

# Follow logs in real-time
docker logs -f crypto-trading
```

### Interactive Commands
```bash
# Execute commands inside running container
docker exec -it crypto-trading bash

# Check application status inside container
docker exec crypto-trading python /app/healthcheck.py

# View ML models inside container
docker exec crypto-trading find /app/ml_results -name "*.h5"
```

## Production Deployment

### Using Docker Compose (Recommended)
Create a `docker-compose.yml` file:

```yaml
version: '3.8'
services:
  crypto-trading:
    image: coinstardon/crypto-trading-engine:latest
    container_name: crypto-trading-prod
    restart: unless-stopped
    environment:
      - BINANCE_API_KEY=${BINANCE_API_KEY}
      - BINANCE_API_SECRET=${BINANCE_API_SECRET}
      - ALPACA_API_KEY=${ALPACA_API_KEY}
      - ALPACA_SECRET_KEY=${ALPACA_SECRET_KEY}
    volumes:
      - ./logs:/app/trading/logs
      - ./custom-env:/app/.env
    healthcheck:
      test: ["CMD", "python", "/app/healthcheck.py"]
      interval: 5m
      timeout: 30s
      retries: 3
      start_period: 2m
```

Then run:
```bash
docker-compose up -d
```

## Environment Variables Required

The container needs these environment variables (either in .env file or passed as -e flags):

### Required for Trading
- `BINANCE_API_KEY` - Your Binance API key
- `BINANCE_API_SECRET` - Your Binance API secret

### Optional (for dual-broker trading)
- `ALPACA_API_KEY` - Your Alpaca API key  
- `ALPACA_SECRET_KEY` - Your Alpaca secret key

### Optional Configuration
- `TRADING_MODE` - Set to 'paper' for paper trading (default: 'live')
- `LOG_LEVEL` - Set logging level (default: 'INFO')

## Troubleshooting

### Container Won't Start
```bash
# Check detailed logs
docker logs crypto-trading

# Common issues:
# 1. Missing .env file - mount one or use environment variables
# 2. Invalid API credentials - check your keys
# 3. Port conflicts - change port mapping
```

### Performance Monitoring
```bash
# Check container resource usage
docker stats crypto-trading

# Check disk usage
docker exec crypto-trading du -sh /app/*

# Monitor trading logs
docker exec crypto-trading tail -f /app/trading/logs/trading.log
```

### Data Backup
```bash
# Backup ML models and results
docker cp crypto-trading:/app/ml_results ./backup-ml-results

# Backup trading logs
docker cp crypto-trading:/app/trading/logs ./backup-logs
```

## What Happens When You Run It?

1. **Startup Validation** - Checks for .env file and API credentials
2. **ML Pipeline Check** - Validates ML models and historical data
3. **Process Manager Launch** - Starts the Docker process manager
4. **Trading Engine** - Begins crypto trading with ML predictions
5. **Health Monitoring** - Continuous health checks every 5 minutes

## Container Contents Verification

The container includes everything needed:
- ✅ All Python trading code
- ✅ Pre-trained ML models (15 .h5 files)
- ✅ Historical market data (33 .csv files)
- ✅ Data scalers (21 .pkl files)
- ✅ Configuration files
- ✅ .env file template

Ready to trade! 🚀📈
