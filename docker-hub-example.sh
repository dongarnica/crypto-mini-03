#!/bin/bash

# =================================================================
# Complete Docker Hub Deployment Example
# =================================================================
# This script demonstrates the complete workflow for pushing your
# crypto trading app to Docker Hub and running it anywhere.
# =================================================================

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🚀 Complete Docker Hub Deployment Example${NC}"
echo "=========================================="

# Step 1: Set your Docker Hub username
echo -e "${YELLOW}Step 1: Configure Docker Hub username${NC}"
echo "Replace 'yourusername' with your actual Docker Hub username:"
DOCKER_HUB_USERNAME="yourusername"  # ← CHANGE THIS
echo "DOCKER_HUB_USERNAME=$DOCKER_HUB_USERNAME"
echo ""

# Step 2: Build and push to Docker Hub
echo -e "${YELLOW}Step 2: Build and push to Docker Hub${NC}"
echo "Command to run:"
echo "./publish-to-dockerhub.sh --username $DOCKER_HUB_USERNAME"
echo ""
echo "This will:"
echo "- Build the enhanced crypto trading image"
echo "- Test the container functionality"
echo "- Push to Docker Hub as $DOCKER_HUB_USERNAME/crypto-trading-enhanced:latest"
echo ""

# Step 3: Pull and run from Docker Hub
echo -e "${YELLOW}Step 3: Pull and run from Docker Hub (on any machine)${NC}"

cat << EOF

# Pull the image
docker pull $DOCKER_HUB_USERNAME/crypto-trading-enhanced:latest

# Create environment file with your API keys
cat > .env << 'ENVEOF'
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
CRYPTO_SYMBOLS=BTCUSDT,ETHUSDT,DOGEUSDT,SOLUSDT,AVAXUSDT,LTCUSDT,UNIUSDT,LINKUSDT,XRPUSDT,DOTUSDT,SHIBUSDT,AAVEUSDT,BCHUSDT,SUSHIUSDT,YFIUSDT
MODE=paper
LOG_LEVEL=INFO
ENVEOF

# Create data directory
mkdir -p data

# Run the enhanced crypto trading container
docker run -d \\
  --name crypto-trading-enhanced \\
  --restart unless-stopped \\
  -p 8080:8080 -p 8081:8081 \\
  --env-file .env \\
  -v \$(pwd)/data:/app/data \\
  $DOCKER_HUB_USERNAME/crypto-trading-enhanced:latest

# Monitor the enhanced output with beautiful countdowns and progress bars
docker logs -f crypto-trading-enhanced

EOF

echo ""
echo -e "${YELLOW}Step 4: What you'll see${NC}"
cat << EOF

The enhanced crypto trading app will display:

🐳 CRYPTO TRADING SYSTEM - DOCKER CONTAINER
==========================================================================================
🕐 Current Time: $(date '+%Y-%m-%d %H:%M:%S')
⏱️  System Uptime: 0:00:15

🚀 TRADING ENGINE STATUS
--------------------------------------------------
   Status: ✅ RUNNING (PID: 88)
   Uptime: 0:00:15
   Next Cycle: ⏳ 04:45

📊 DATA EXPORT STATUS  
--------------------------------------------------
   Status: ⏰ SCHEDULED
   Next Export: ⏳ 01:45
   Scheduled At: $(date -d '+2 hours' '+%H:%M:%S')

🤖 ML TRAINING STATUS
--------------------------------------------------
   Status: 🧠 SCHEDULED  
   Next Training: ⏳ 05:45
   Scheduled At: $(date -d '+6 hours' '+%H:%M:%S')

📊 PROCESS CYCLES
--------------------------------------------------
   Trading:  [██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 5.0%
   Export:   [████████████████████████████████████░░░░] 90.0%
   ML Train: [████████████████████████████░░░░░░░░░░░░] 70.0%

💡 All processes running with enhanced output | Press Ctrl+C to stop
==========================================================================================

EOF

echo -e "${GREEN}🎉 Your crypto trading app is now deployable to Docker Hub!${NC}"
echo ""
echo -e "${BLUE}Summary:${NC}"
echo "1. Update DOCKER_HUB_USERNAME in this script"
echo "2. Run: ./publish-to-dockerhub.sh --username yourusername"
echo "3. On any machine: docker pull yourusername/crypto-trading-enhanced:latest"
echo "4. Create .env file with your API keys"
echo "5. Run the container and enjoy the enhanced visual output!"
echo ""
echo -e "${YELLOW}📖 Full documentation: DOCKER_HUB_DEPLOYMENT.md${NC}"
