# Docker Integration with Process Manager
## Dual Countdown Timer Setup

This document describes the enhanced Docker setup that includes both the trading engine and historical data export service with visual countdown timers.

## 🚀 **Quick Start**

### Build and Run the Container

```bash
# Build the Docker image
docker build -t crypto-trading-app .

# Run with dual countdown timers (default)
docker run -d \
  --name crypto-trading \
  --env-file .env \
  -v $(pwd)/trading/logs:/app/trading/logs \
  -v $(pwd)/ml_results:/app/ml_results \
  crypto-trading-app

# View the countdown timers
docker logs -f crypto-trading
```

### Run Modes

```bash
# Default: Process manager with dual countdown timers
docker run crypto-trading-app

# Trading engine only (legacy mode)
docker run crypto-trading-app trading-only

# Interactive bash shell
docker run -it crypto-trading-app bash
```

## 📊 **Process Manager Features**

### Dual Countdown Timers
- **Trading Engine**: Shows cycle progress (every 5 minutes)
- **Data Export**: Shows time until next export (every 2 hours)

### Visual Display
```
================================================================================
🚀 CRYPTO TRADING SYSTEM - PROCESS MANAGER
================================================================================

🕐 Current Time: 2025-06-05 14:30:25

📈 TRADING ENGINE STATUS
----------------------------------------
   Status: ✅ RUNNING (PID: 1234)
   Uptime: 0:15:30
   Next Cycle: ⏳ 04:30

📊 DATA EXPORT STATUS
----------------------------------------
   Status: ✅ SCHEDULED
   Last Export: 14:20:06
   Next Export: ⏳ 01:49:35
   Scheduled At: 16:20:06

💾 DATA STATISTICS
----------------------------------------
   Total Files: 18
   Symbols: 18
   Data Size: 0.76 MB

📊 CYCLE PROGRESS
----------------------------------------
   Trading:  [████████████████████████████████████████████████████] 85.0%
   Export:   [████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 25.0%

================================================================================
Press Ctrl+C to stop
================================================================================
```

## 🔧 **Configuration**

### Environment Variables
All existing environment variables are supported:
- `BINANCE_API_KEY`
- `BINANCE_API_SECRET`
- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`

### Process Manager Settings
Edit `process_manager.py` to customize:
```python
self.trading_cycle_minutes = 5  # Trading engine cycle time
self.export_cycle_hours = 2     # Data export cycle time
```

## 🏥 **Health Monitoring**

### Docker Health Check
The container includes comprehensive health checks:
- Environment validation
- Process manager status
- ML models availability
- Historical data presence
- Configuration validation

### Check Container Health
```bash
# Check health status
docker inspect --format='{{.State.Health.Status}}' crypto-trading

# View health check details
docker inspect --format='{{json .State.Health}}' crypto-trading | jq
```

## 📁 **File Structure**

### New Files Added
- `process_manager.py` - Main process manager with countdown timers
- Updated `start.sh` - Enhanced startup script
- Updated `Dockerfile` - Includes process manager
- Updated `healthcheck.py` - Monitors process manager

### Log Files
```
historical_exports/logs/
├── process_manager.log     # Process manager logs
├── scheduled_data.log      # Data export logs
└── data_export_service.log # Service logs
```

## 🚀 **Production Deployment**

### Docker Compose
```yaml
version: '3.8'
services:
  crypto-trading:
    build: .
    container_name: crypto-trading
    restart: unless-stopped
    env_file: .env
    volumes:
      - ./trading/logs:/app/trading/logs
      - ./ml_results:/app/ml_results
    healthcheck:
      test: ["CMD", "python", "/app/healthcheck.py"]
      interval: 5m
      timeout: 30s
      retries: 3
      start_period: 2m
```

### Systemd Service (Alternative)
```bash
# Create systemd service for direct process manager
python data_export_service.py --create-systemd
```

## 🔍 **Monitoring & Debugging**

### View Live Countdown
```bash
# Follow container logs to see countdown timers
docker logs -f crypto-trading

# Execute into running container
docker exec -it crypto-trading bash
```

### Check Process Status
```bash
# Inside container
python process_manager.py --output-dir historical_exports &
ps aux | grep python
```

### Manual Data Export
```bash
# Inside container
python scheduled_data_manager.py --manual
```

## 🛠 **Troubleshooting**

### Common Issues

**Process Manager Not Starting**
```bash
# Check logs
docker logs crypto-trading

# Verify environment
docker exec crypto-trading env | grep -E "(BINANCE|ALPACA)"
```

**Data Export Failing**
```bash
# Check export logs
docker exec crypto-trading cat /app/historical_exports/logs/scheduled_data.log

# Test manual export
docker exec crypto-trading python scheduled_data_manager.py --manual
```

**Trading Engine Issues**
```bash
# Check if trading engine exists
docker exec crypto-trading ls -la /app/trading/

# Run in trading-only mode
docker run crypto-trading-app trading-only
```

## 📊 **Performance Notes**

- **Memory Usage**: ~200-300MB baseline
- **CPU Usage**: Low (periodic spikes during exports)
- **Disk I/O**: Minimal (log rotation, data exports)
- **Network**: API calls to Binance/Alpaca

## 🔄 **Upgrade Path**

### From Previous Version
1. Stop existing container
2. Build new image with updated code
3. Start with same environment variables
4. Logs and data will be preserved via volumes

### Configuration Migration
All existing `.env` files are compatible with the new setup.

---

**Last Updated**: June 5, 2025  
**Version**: 2.0.0 (Process Manager Integration)
