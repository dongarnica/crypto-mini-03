# 🐳 Docker Deployment Guide - Crypto Trading App

This guide provides comprehensive instructions for deploying the crypto trading application using Docker and Docker Compose.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Deployment Options](#deployment-options)
- [Management Commands](#management-commands)
- [Monitoring & Logs](#monitoring--logs)
- [Data Persistence](#data-persistence)
- [Security Considerations](#security-considerations)
- [Troubleshooting](#troubleshooting)

## 🔧 Prerequisites

### System Requirements
- **Docker**: Version 20.10.0 or higher
- **Docker Compose**: Version 2.0.0 or higher (or legacy 1.29.0+)
- **System Memory**: Minimum 4GB RAM (8GB recommended)
- **Disk Space**: Minimum 10GB free space
- **Network**: Stable internet connection for API access

### Installation

#### Ubuntu/Debian
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose (if not included)
sudo apt-get update
sudo apt-get install docker-compose-plugin
```

#### CentOS/RHEL
```bash
# Install Docker
sudo yum install -y yum-utils
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo yum install docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo systemctl start docker
sudo systemctl enable docker
```

#### macOS
```bash
# Install Docker Desktop
brew install --cask docker

# Or download from: https://docs.docker.com/desktop/mac/install/
```

#### Windows
Download Docker Desktop from: https://docs.docker.com/desktop/windows/install/

## 🚀 Quick Start

### 1. Clone and Setup
```bash
# Clone the repository (if not already done)
git clone <your-repo-url>
cd crypto-mini-03

# Make deployment script executable
chmod +x deploy.sh

# Run complete setup
./deploy.sh setup
```

### 2. Configure API Credentials
```bash
# Edit the .env file with your actual API credentials
nano .env

# Required variables:
# ALPACA_API_KEY=your_alpaca_api_key
# ALPACA_SECRET_KEY=your_alpaca_secret_key
# BINANCE_API_KEY=your_binance_api_key
# BINANCE_API_SECRET=your_binance_secret
```

### 3. Start Trading
```bash
# Start the application
./deploy.sh start

# Check status
./deploy.sh status

# View logs
./deploy.sh logs
```

## ⚙️ Configuration

### Environment Variables

Create or modify the `.env` file with your configuration:

```bash
# API Credentials
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_secret

# Trading Configuration
PAPER_TRADING=true              # Set to false for live trading
LOG_LEVEL=INFO                  # DEBUG, INFO, WARNING, ERROR
CRYPTO_SYMBOLS=BTCUSDT,ETHUSDT,ADAUSDT,DOTUSDT,LINKUSDT

# Risk Management
MAX_POSITION_SIZE=0.05          # 5% max per position
MAX_PORTFOLIO_RISK=0.20         # 20% max total risk
STOP_LOSS_PCT=0.02              # 2% stop loss
TAKE_PROFIT_PCT=0.04            # 4% take profit
```

### Docker Compose Profiles

The application supports multiple deployment profiles:

- **default**: Production trading environment
- **development**: Development environment with source code mounting
- **monitoring**: Includes log viewer interface
- **backup**: Data backup utilities

## 🛠️ Deployment Options

### Production Deployment
```bash
# Standard production deployment
./deploy.sh start

# With monitoring interface
docker compose --profile monitoring up -d
```

### Development Deployment
```bash
# Start development environment
./deploy.sh dev

# Access development container
docker exec -it crypto-trading-dev bash
```

### High Availability Deployment
```bash
# Deploy with restart policies and resource limits
docker compose up -d --scale crypto-trader=1
```

## 📊 Management Commands

### Using the Deployment Script

```bash
# Setup and configuration
./deploy.sh setup          # Complete initial setup
./deploy.sh build          # Build Docker image

# Application management
./deploy.sh start          # Start the application
./deploy.sh stop           # Stop the application
./deploy.sh restart        # Restart the application
./deploy.sh status         # Show application status

# Monitoring and debugging
./deploy.sh logs           # View application logs
./deploy.sh monitor        # Start log monitoring interface

# Data management
./deploy.sh backup         # Create data backup

# Development
./deploy.sh dev            # Start development environment

# Cleanup
./deploy.sh clean          # Remove containers and images
```

### Manual Docker Commands

```bash
# Build image
docker build -t crypto-trader:latest .

# Run container manually
docker run -d \
  --name crypto-trading-engine \
  --env-file .env \
  -v $(pwd)/data/trading-logs:/app/trading/logs \
  -v $(pwd)/data/historical-exports:/app/historical_exports \
  crypto-trader:latest

# View logs
docker logs -f crypto-trading-engine

# Execute commands in container
docker exec -it crypto-trading-engine bash

# Stop and remove container
docker stop crypto-trading-engine
docker rm crypto-trading-engine
```

## 📈 Monitoring & Logs

### Log Viewer Interface
```bash
# Start log monitoring interface
./deploy.sh monitor

# Access at: http://localhost:9999
```

### Manual Log Access
```bash
# View live logs
docker compose logs -f crypto-trader

# View specific service logs
docker compose logs -f log-viewer

# View logs from host system
tail -f data/trading-logs/trading_*.log
```

### Health Checks
```bash
# Check container health
docker inspect crypto-trading-engine --format='{{.State.Health.Status}}'

# View health check logs
docker inspect crypto-trading-engine --format='{{range .State.Health.Log}}{{.Output}}{{end}}'
```

## 💾 Data Persistence

### Volume Mapping

The application uses the following persistent volumes:

- **trading-logs**: `/app/trading/logs` → `./data/trading-logs`
- **historical-data**: `/app/historical_exports` → `./data/historical-exports`
- **ml-models**: `/app/ml_results` → `./data/ml-results`
- **binance-data**: `/app/binance_exports` → `./data/binance-exports`

### Backup Strategy

```bash
# Create backup
./deploy.sh backup

# Manual backup
docker run --rm \
  -v crypto-mini-03_trading-logs:/data/logs:ro \
  -v $(pwd)/backups:/backup \
  alpine:latest \
  tar czf /backup/trading-backup-$(date +%Y%m%d-%H%M%S).tar.gz -C /data .

# Restore from backup
tar xzf backups/trading-backup-*.tar.gz -C data/
```

## 🔒 Security Considerations

### Environment Security
- Never commit `.env` files to version control
- Use Docker secrets for production deployments
- Rotate API keys regularly
- Use read-only volumes where possible

### Container Security
```bash
# Run with limited privileges
docker run --user 1000:1000 --read-only crypto-trader:latest

# Use security profiles
docker run --security-opt=no-new-privileges crypto-trader:latest
```

### Network Security
```bash
# Isolate container network
docker network create --driver bridge crypto-trading-network
```

## 🐛 Troubleshooting

### Common Issues

#### Container Won't Start
```bash
# Check logs for errors
docker compose logs crypto-trader

# Verify environment variables
docker compose config

# Check resource usage
docker stats
```

#### API Connection Issues
```bash
# Test API credentials
docker exec crypto-trading-engine python -c "
from alpaca.alpaca_client import AlpacaCryptoClient
client = AlpacaCryptoClient()
print('Account:', client.get_account())
"
```

#### Memory Issues
```bash
# Check memory usage
docker stats crypto-trading-engine

# Increase memory limits in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 8G
```

#### Permission Issues
```bash
# Fix volume permissions
sudo chown -R 1000:1000 data/
chmod -R 755 data/
```

### Debug Mode
```bash
# Start in debug mode
LOG_LEVEL=DEBUG ./deploy.sh start

# Access container for debugging
docker exec -it crypto-trading-engine bash
```

### Performance Optimization
```bash
# Monitor resource usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"

# Optimize image size
docker images crypto-trader:latest

# Clean up unused resources
docker system prune -f
```

## 📞 Support

### Getting Help
- Check logs: `./deploy.sh logs`
- Verify status: `./deploy.sh status`
- Review configuration: `docker compose config`

### Common Commands Reference
```bash
# Quick status check
./deploy.sh status

# Restart if issues
./deploy.sh restart

# View recent logs
docker compose logs --tail=50 crypto-trader

# Emergency stop
docker kill crypto-trading-engine
```

---

## 🎯 Production Checklist

Before deploying to production:

- [ ] Update `.env` with production API credentials
- [ ] Set `PAPER_TRADING=false` for live trading
- [ ] Configure appropriate log levels
- [ ] Set up monitoring and alerting
- [ ] Configure automatic backups
- [ ] Test disaster recovery procedures
- [ ] Review security settings
- [ ] Set up log rotation
- [ ] Configure resource limits
- [ ] Test health checks

---

**⚠️ Important**: Always test with paper trading before enabling live trading with real funds!
