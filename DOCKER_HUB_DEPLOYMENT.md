# Docker Hub Deployment Guide
## Enhanced Crypto Trading Application

This guide shows you how to push your crypto trading app to Docker Hub and then pull and run it anywhere.

## 🚀 Quick Start

### 1. Push to Docker Hub

```bash
# Set your Docker Hub username (replace with your actual username)
export DOCKER_HUB_USERNAME="your-dockerhub-username"

# Run the publish script
./publish-to-dockerhub.sh --username your-dockerhub-username
```

### 2. Pull and Run from Docker Hub

```bash
# Pull the image
docker pull your-dockerhub-username/crypto-trading-enhanced:latest

# Create .env file with your API keys
cat > .env << EOF
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets
CRYPTO_SYMBOLS=BTCUSDT,ETHUSDT,DOGEUSDT,SOLUSDT,AVAXUSDT,LTCUSDT,UNIUSDT,LINKUSDT,XRPUSDT,DOTUSDT,SHIBUSDT,AAVEUSDT,BCHUSDT,SUSHIUSDT,YFIUSDT
MODE=paper
EOF

# Run the container
docker run -d \
  --name crypto-trading-enhanced \
  -p 8080:8080 -p 8081:8081 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  your-dockerhub-username/crypto-trading-enhanced:latest
```

### 3. Monitor the Application

```bash
# Check container status
docker ps

# View real-time logs with enhanced output
docker logs -f crypto-trading-enhanced

# Check container health
docker inspect crypto-trading-enhanced | grep Health -A 10
```

---

## 📋 Detailed Instructions

### Prerequisites

1. **Docker Hub Account**: Sign up at [hub.docker.com](https://hub.docker.com)
2. **Docker Installed**: On your local machine
3. **API Keys**: From Binance and Alpaca Markets

### Step 1: Prepare Your Environment

```bash
# Navigate to your project directory
cd /path/to/crypto-mini-03

# Ensure you have the latest changes
git pull  # if using git

# Verify required files exist
ls -la docker_process_manager.py scheduled_data_manager.py .env
```

### Step 2: Build and Push to Docker Hub

#### Option A: Using the Automated Script (Recommended)

```bash
# Make script executable
chmod +x publish-to-dockerhub.sh

# Run with your Docker Hub username
./publish-to-dockerhub.sh --username your-dockerhub-username

# Follow the prompts to enter your Docker Hub password
```

#### Option B: Manual Build and Push

```bash
# 1. Build the image
docker build -t your-dockerhub-username/crypto-trading-enhanced:latest .

# 2. Login to Docker Hub
docker login

# 3. Push the image
docker push your-dockerhub-username/crypto-trading-enhanced:latest
```

### Step 3: Deploy on Any Server

#### On a New Server/Machine:

```bash
# 1. Install Docker (Ubuntu/Debian example)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 2. Create project directory
mkdir crypto-trading-app
cd crypto-trading-app

# 3. Create environment file
nano .env
# Add your API keys as shown above

# 4. Pull and run the container
docker pull your-dockerhub-username/crypto-trading-enhanced:latest

docker run -d \
  --name crypto-trading-enhanced \
  --restart unless-stopped \
  -p 8080:8080 -p 8081:8081 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  your-dockerhub-username/crypto-trading-enhanced:latest
```

### Step 4: Access the Application

The enhanced crypto trading app will be running with:

- **Trading Engine**: Analyzing crypto markets every 5 minutes
- **Data Export**: Scheduled every 2 hours  
- **ML Training**: Scheduled every 6 hours
- **Enhanced Dashboard**: Beautiful terminal output with progress bars and countdowns

#### Monitor Real-Time Activity:

```bash
# View live logs with enhanced visual output
docker logs -f crypto-trading-enhanced

# You'll see:
# 🚀 TRADING ENGINE STATUS with countdown timers
# 📊 DATA EXPORT STATUS with progress bars  
# 🤖 ML TRAINING STATUS with scheduling info
# 💾 Live trading signals and recommendations
```

---

## 🔧 Configuration Options

### Environment Variables

Create a `.env` file with these variables:

```bash
# Required API Keys
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key

# Trading Configuration
MODE=paper                    # paper or live
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Crypto Symbols (comma-separated)
CRYPTO_SYMBOLS=BTCUSDT,ETHUSDT,DOGEUSDT,SOLUSDT,AVAXUSDT,LTCUSDT,UNIUSDT,LINKUSDT,XRPUSDT,DOTUSDT,SHIBUSDT,AAVEUSDT,BCHUSDT,SUSHIUSDT,YFIUSDT

# Optional: Logging and ML Configuration
LOG_LEVEL=INFO
ENABLE_ML_RETRAINING=true
EXPORT_INTERVAL_HOURS=2
ML_RETRAIN_INTERVAL_HOURS=6
```

### Volume Mounts

```bash
# Mount data directory for persistence
-v $(pwd)/data:/app/data

# Mount logs directory for external access
-v $(pwd)/logs:/app/trading/logs

# Mount ML results for model persistence
-v $(pwd)/ml_results:/app/ml_results
```

### Port Mappings

```bash
# Standard ports
-p 8080:8080  # Main application port
-p 8081:8081  # Secondary/monitoring port (future use)
```

---

## 🛠️ Troubleshooting

### Common Issues

1. **Container Exits Immediately**
   ```bash
   # Check logs for errors
   docker logs crypto-trading-enhanced
   
   # Verify .env file has correct API keys
   cat .env
   ```

2. **Permission Denied**
   ```bash
   # Fix data directory permissions
   sudo chown -R 1000:1000 data/
   ```

3. **Out of Memory**
   ```bash
   # Run with memory limit
   docker run --memory=2g ... your-dockerhub-username/crypto-trading-enhanced:latest
   ```

4. **Can't Connect to APIs**
   ```bash
   # Test network connectivity
   docker exec crypto-trading-enhanced ping api.binance.com
   ```

### Logs and Debugging

```bash
# View all logs
docker logs crypto-trading-enhanced

# Follow logs in real-time
docker logs -f crypto-trading-enhanced

# Execute commands inside container
docker exec -it crypto-trading-enhanced bash

# Check process status inside container
docker exec crypto-trading-enhanced ps aux
```

---

## 🔄 Updates and Maintenance

### Updating the Application

```bash
# 1. Pull latest image
docker pull your-dockerhub-username/crypto-trading-enhanced:latest

# 2. Stop current container
docker stop crypto-trading-enhanced
docker rm crypto-trading-enhanced

# 3. Run new container
docker run -d \
  --name crypto-trading-enhanced \
  --restart unless-stopped \
  -p 8080:8080 -p 8081:8081 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  your-dockerhub-username/crypto-trading-enhanced:latest
```

### Backup Data

```bash
# Backup trading data
tar czf crypto-trading-backup-$(date +%Y%m%d).tar.gz data/

# Backup ML models
tar czf ml-models-backup-$(date +%Y%m%d).tar.gz ml_results/
```

---

## 📊 Features Included

✅ **Enhanced Docker Process Manager** - Beautiful visual output with progress bars  
✅ **Automated Trading Engine** - 5-minute cycle analysis with live countdowns  
✅ **Scheduled Data Export** - 2-hour cycles with progress tracking  
✅ **ML Model Training** - 6-hour automated retraining cycles  
✅ **Real-time Monitoring** - Live trading signals and recommendations  
✅ **Health Checks** - Automatic container health monitoring  
✅ **Volume Persistence** - Data and models saved between container restarts  
✅ **Multi-Exchange Support** - Binance and Alpaca integration  
✅ **Paper Trading Mode** - Safe testing environment  

---

## 📞 Support

For issues with the Docker deployment:

1. Check the logs: `docker logs crypto-trading-enhanced`
2. Verify your .env file has valid API keys
3. Ensure ports 8080 and 8081 are available
4. Check Docker daemon is running: `docker info`

The application includes comprehensive error handling and will display helpful error messages in the container logs.
