# Crypto Trading Engine - Docker Deployment Guide

## 🐳 Docker Hub Repository

The crypto trading engine is available as a Docker image on Docker Hub:

**Repository**: `codespacesdev/crypto-trading-engine`

### Available Tags
- `latest` - Latest stable version
- `v1.0.0` - Version 1.0.0

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Alpaca API credentials (for live trading)
- Environment configuration

### 1. Pull the Image
```bash
docker pull codespacesdev/crypto-trading-engine:latest
```

### 2. Create Environment File
Create a `.env` file with your configuration:

```env
# Alpaca API Configuration
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # or https://api.alpaca.markets for live

# Trading Configuration
SYMBOLS=BTCUSDT,ETHUSDT,DOTUSDT
LOG_LEVEL=INFO
SAVE_TRADES=true
MIN_CONFIDENCE=0.7
RISK_PERCENTAGE=0.02
LOOKBACK_PERIOD=60
PREDICTION_HORIZON=24

# ML Configuration
USE_BINARY_CLASSIFICATION=true
RETRAIN_INTERVAL_HOURS=24
```

### 3. Create Docker Compose File
Create a `docker-compose.yml` file:

```yaml
services:
  crypto-trader:
    image: codespacesdev/crypto-trading-engine:latest
    container_name: crypto-trading-engine
    restart: unless-stopped
    env_file:
      - .env
    volumes:
      - ./logs:/app/trading/logs
      - ./ml_results:/app/ml_results
      - ./historical_exports:/app/historical_exports
    ports:
      - "8080:8080"
    networks:
      - crypto-trading-network
    healthcheck:
      test: ["CMD", "python", "/app/healthcheck.py"]
      interval: 5m
      timeout: 30s
      retries: 3
      start_period: 2m

networks:
  crypto-trading-network:
    driver: bridge

volumes:
  trading_logs:
  ml_results:
  historical_data:
```

### 4. Run the Application
```bash
# Start the application
docker-compose up -d

# Check logs
docker-compose logs -f crypto-trader

# Check status
docker-compose ps
```

## 📊 Features

### Core Trading Features
- ✅ **Automated Crypto Trading** - BTCUSDT, ETHUSDT, DOTUSDT support
- ✅ **ML-Powered Predictions** - LSTM neural networks for price forecasting
- ✅ **Risk Management** - Position sizing and stop-loss management
- ✅ **Real-time Monitoring** - Live portfolio tracking and P&L calculation
- ✅ **Paper Trading** - Safe testing environment with Alpaca Paper API

### Technical Features
- ✅ **Docker Containerized** - Portable and scalable deployment
- ✅ **Health Monitoring** - Built-in health checks and logging
- ✅ **Data Persistence** - Volume mounting for logs and ML models
- ✅ **Auto-restart** - Resilient operation with restart policies
- ✅ **Security** - Non-root user execution and secure secrets management

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ALPACA_API_KEY` | Alpaca API key | - | ✅ |
| `ALPACA_SECRET_KEY` | Alpaca secret key | - | ✅ |
| `ALPACA_BASE_URL` | Alpaca API endpoint | paper-api.alpaca.markets | ✅ |
| `SYMBOLS` | Trading symbols (comma-separated) | BTCUSDT,ETHUSDT,DOTUSDT | ❌ |
| `LOG_LEVEL` | Logging level | INFO | ❌ |
| `SAVE_TRADES` | Enable trade logging | true | ❌ |
| `MIN_CONFIDENCE` | Minimum ML confidence | 0.7 | ❌ |
| `RISK_PERCENTAGE` | Position size percentage | 0.02 | ❌ |

### Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./logs` | `/app/trading/logs` | Trading logs and activity |
| `./ml_results` | `/app/ml_results` | ML models and predictions |
| `./historical_exports` | `/app/historical_exports` | Historical price data |

## 📈 Monitoring

### Application Logs
```bash
# Follow real-time logs
docker-compose logs -f crypto-trader

# View recent logs
docker-compose logs --tail=100 crypto-trader
```

### Health Check
```bash
# Check container health
docker-compose ps

# Manual health check
docker exec crypto-trading-engine python /app/healthcheck.py
```

### Portfolio Status
The application logs portfolio status every 5 minutes:
```
🏆 ACTIVE POSITIONS (2)
------------------------------------------------------------
      BTCUSDT:   0.013711 @ $107139.06
              Current: $105789.44
              P&L: 🔴 $  -18.50 (-1.3%)
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. Authentication Errors
```bash
# Check API credentials
docker-compose logs crypto-trader | grep -i "auth\|credential\|api"
```

#### 2. ML Model Issues
```bash
# Check ML pipeline status
docker-compose logs crypto-trader | grep -i "ml\|model\|prediction"
```

#### 3. Container Restart Loop
```bash
# Check container status
docker-compose ps

# View error logs
docker-compose logs crypto-trader | tail -50
```

### Performance Tuning

#### Memory Usage
The application uses approximately 3.7GB of disk space and 500MB-1GB RAM during operation.

#### CPU Usage
ML training can be CPU-intensive. Consider:
- Limiting CPU usage: `cpus: 2.0` in docker-compose.yml
- Scheduling training during low-traffic periods

## 🔒 Security

### Production Recommendations
1. **API Keys**: Use Docker secrets or external secret management
2. **Network**: Restrict network access with firewall rules
3. **Updates**: Regularly update to latest image versions
4. **Monitoring**: Implement log aggregation and alerting

### Secret Management
```yaml
# Using Docker secrets (recommended for production)
services:
  crypto-trader:
    image: codespacesdev/crypto-trading-engine:latest
    secrets:
      - alpaca_api_key
      - alpaca_secret_key
    environment:
      ALPACA_API_KEY_FILE: /run/secrets/alpaca_api_key
      ALPACA_SECRET_KEY_FILE: /run/secrets/alpaca_secret_key

secrets:
  alpaca_api_key:
    file: ./secrets/api_key.txt
  alpaca_secret_key:
    file: ./secrets/secret_key.txt
```

## 📋 Deployment Checklist

- [ ] Docker and Docker Compose installed
- [ ] Alpaca API credentials obtained
- [ ] Environment file configured
- [ ] Volume directories created
- [ ] Network ports available (8080)
- [ ] Sufficient disk space (5GB+)
- [ ] Sufficient RAM (2GB+)

## 🔄 Updates

### Updating to Latest Version
```bash
# Pull latest image
docker-compose pull

# Restart with new image
docker-compose up -d

# Verify update
docker-compose logs crypto-trader | head -20
```

### Backup Before Updates
```bash
# Backup ML models and logs
tar -czf backup_$(date +%Y%m%d).tar.gz logs/ ml_results/
```

## 📞 Support

For issues and support:
1. Check the application logs first
2. Review this deployment guide
3. Check Docker Hub repository for updates
4. File issues with detailed logs and configuration

---

**⚠️ Risk Warning**: This software is for educational and research purposes. Cryptocurrency trading involves significant financial risk. Always test with paper trading before using real money.
