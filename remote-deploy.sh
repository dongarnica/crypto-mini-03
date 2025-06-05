#!/bin/bash
# ================================================================
# Remote Server Deployment Script for Crypto Trading Engine
# ================================================================
# This script handles Docker Hub publishing and remote deployment
# commands for production servers.
# ================================================================

set -e

# Configuration
DOCKER_HUB_USERNAME="coinstardon"
IMAGE_NAME="crypto-trading-engine"
VERSION="1.0.0"
CONTAINER_NAME="crypto-trading-engine"
PRODUCTION_PORT="8080"
LOG_VOLUME="crypto-trading-logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}"
    echo "================================================================"
    echo "  🚀 Crypto Trading Engine - Remote Deployment Setup"
    echo "  📦 Docker Hub Integration & Remote Server Commands"
    echo "================================================================"
    echo -e "${NC}"
}

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header

# ================================================================
# STEP 1: Build Production Image
# ================================================================
print_step "Building production Docker image..."

docker build --no-cache -t ${DOCKER_HUB_USERNAME}/${IMAGE_NAME}:${VERSION} .
docker build --no-cache -t ${DOCKER_HUB_USERNAME}/${IMAGE_NAME}:latest .

if [ $? -eq 0 ]; then
    print_info "✅ Production image built successfully"
else
    print_error "❌ Image build failed"
    exit 1
fi

# ================================================================
# STEP 2: Verify Image Contents
# ================================================================
print_step "Verifying image contents..."

# Check .env file
ENV_CHECK=$(docker run --rm ${DOCKER_HUB_USERNAME}/${IMAGE_NAME}:latest test -f /app/.env && echo "FOUND" || echo "MISSING")
if [ "$ENV_CHECK" = "FOUND" ]; then
    print_info "✅ .env file included"
else
    print_error "❌ .env file missing"
    exit 1
fi

# Check data files
DATA_COUNT=$(docker run --rm ${DOCKER_HUB_USERNAME}/${IMAGE_NAME}:latest find /app/historical_exports -name "*.csv" -type f 2>/dev/null | wc -l)
print_info "📊 Found $DATA_COUNT CSV data files"

# Check ML models
MODEL_COUNT=$(docker run --rm ${DOCKER_HUB_USERNAME}/${IMAGE_NAME}:latest find /app/ml_results -name "*.h5" -type f 2>/dev/null | wc -l)
print_info "🤖 Found $MODEL_COUNT ML models"

# ================================================================
# STEP 3: Docker Hub Authentication Check
# ================================================================
print_step "Checking Docker Hub authentication..."

if ! docker info | grep -q "Username"; then
    print_info "⚠️  Not logged into Docker Hub"
    echo ""
    echo "Please login to Docker Hub first:"
    echo "docker login"
    echo ""
    read -p "Press Enter after logging in, or 'q' to quit: " -r
    if [[ $REPLY == "q" ]]; then
        exit 0
    fi
fi

# ================================================================
# STEP 4: Push to Docker Hub
# ================================================================
print_step "Pushing images to Docker Hub..."

echo "Pushing version ${VERSION}..."
docker push ${DOCKER_HUB_USERNAME}/${IMAGE_NAME}:${VERSION}

echo "Pushing latest tag..."
docker push ${DOCKER_HUB_USERNAME}/${IMAGE_NAME}:latest

if [ $? -eq 0 ]; then
    print_info "✅ Images pushed successfully to Docker Hub"
else
    print_error "❌ Push failed"
    exit 1
fi

# ================================================================
# STEP 5: Generate Remote Deployment Commands
# ================================================================
print_step "Generating remote deployment commands..."

echo ""
echo "================================================================"
echo "🚀 REMOTE SERVER DEPLOYMENT COMMANDS"
echo "================================================================"
echo ""

echo "# 1. PULL IMAGE ON REMOTE SERVER:"
echo "docker pull ${DOCKER_HUB_USERNAME}/${IMAGE_NAME}:latest"
echo ""

echo "# 2. CREATE PERSISTENT VOLUME FOR LOGS:"
echo "docker volume create ${LOG_VOLUME}"
echo ""

echo "# 3. STOP EXISTING CONTAINER (if running):"
echo "docker stop ${CONTAINER_NAME} 2>/dev/null || true"
echo "docker rm ${CONTAINER_NAME} 2>/dev/null || true"
echo ""

echo "# 4. RUN PRODUCTION CONTAINER:"
echo "docker run -d \\"
echo "  --name ${CONTAINER_NAME} \\"
echo "  --restart unless-stopped \\"
echo "  -p ${PRODUCTION_PORT}:${PRODUCTION_PORT} \\"
echo "  -v ${LOG_VOLUME}:/app/trading/logs \\"
echo "  -e TZ=UTC \\"
echo "  ${DOCKER_HUB_USERNAME}/${IMAGE_NAME}:latest"
echo ""

echo "# 5. MONITORING COMMANDS:"
echo "docker logs -f ${CONTAINER_NAME}                    # Follow logs"
echo "docker ps | grep ${CONTAINER_NAME}                  # Check status"
echo "docker exec -it ${CONTAINER_NAME} /bin/bash         # Access container"
echo "docker stats ${CONTAINER_NAME}                      # Resource usage"
echo ""

echo "# 6. MAINTENANCE COMMANDS:"
echo "docker restart ${CONTAINER_NAME}                    # Restart container"
echo "docker stop ${CONTAINER_NAME}                       # Stop container"
echo "docker system prune -f                              # Cleanup unused images"
echo ""

# ================================================================
# STEP 6: Create Remote Deployment Script
# ================================================================
print_step "Creating remote deployment script file..."

cat > remote-server-deploy.sh << 'EOF'
#!/bin/bash
# ================================================================
# Remote Server Deployment Script
# Run this script on your remote production server
# ================================================================

set -e

DOCKER_HUB_USERNAME="coinstardon"
IMAGE_NAME="crypto-trading-engine"
CONTAINER_NAME="crypto-trading-engine"
PRODUCTION_PORT="8080"
LOG_VOLUME="crypto-trading-logs"

echo "🚀 Deploying Crypto Trading Engine on Remote Server"
echo "=================================================="

# Pull latest image
echo "📦 Pulling latest image from Docker Hub..."
docker pull ${DOCKER_HUB_USERNAME}/${IMAGE_NAME}:latest

# Create volume for logs
echo "📁 Creating persistent volume for logs..."
docker volume create ${LOG_VOLUME} 2>/dev/null || true

# Stop existing container
echo "🛑 Stopping existing container..."
docker stop ${CONTAINER_NAME} 2>/dev/null || true
docker rm ${CONTAINER_NAME} 2>/dev/null || true

# Run new container
echo "🚀 Starting production container..."
docker run -d \
  --name ${CONTAINER_NAME} \
  --restart unless-stopped \
  -p ${PRODUCTION_PORT}:${PRODUCTION_PORT} \
  -v ${LOG_VOLUME}:/app/trading/logs \
  -e TZ=UTC \
  ${DOCKER_HUB_USERNAME}/${IMAGE_NAME}:latest

if [ $? -eq 0 ]; then
    echo "✅ Container deployed successfully!"
    echo "📊 Container ID: $(docker ps -q --filter name=${CONTAINER_NAME})"
    echo ""
    echo "🔍 Access your application at: http://YOUR_SERVER_IP:${PRODUCTION_PORT}"
    echo "📋 Monitor logs with: docker logs -f ${CONTAINER_NAME}"
    echo ""
    echo "Initial logs:"
    echo "============="
    sleep 3
    docker logs ${CONTAINER_NAME}
else
    echo "❌ Deployment failed!"
    exit 1
fi
EOF

chmod +x remote-server-deploy.sh

print_info "✅ Created remote-server-deploy.sh for remote deployment"

# ================================================================
# STEP 7: Create Docker Compose for Remote Deployment
# ================================================================
print_step "Creating docker-compose for remote deployment..."

cat > docker-compose.remote.yml << EOF
version: '3.8'

services:
  crypto-trading-engine:
    image: ${DOCKER_HUB_USERNAME}/${IMAGE_NAME}:latest
    container_name: ${CONTAINER_NAME}
    restart: unless-stopped
    ports:
      - "${PRODUCTION_PORT}:${PRODUCTION_PORT}"
    volumes:
      - crypto-trading-logs:/app/trading/logs
    environment:
      - TZ=UTC
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:${PRODUCTION_PORT}/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'

volumes:
  crypto-trading-logs:
    external: true
EOF

print_info "✅ Created docker-compose.remote.yml"

# ================================================================
# STEP 8: Create Environment Setup Guide
# ================================================================
print_step "Creating deployment documentation..."

cat > REMOTE_DEPLOYMENT_GUIDE.md << 'EOF'
# Remote Server Deployment Guide

## Prerequisites

1. **Remote server with Docker installed**
   ```bash
   # Install Docker on Ubuntu/Debian
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker $USER
   ```

2. **Server requirements**
   - Minimum 2GB RAM
   - 10GB available disk space
   - Open port 8080 for web interface

## Deployment Methods

### Method 1: Using Deployment Script (Recommended)

1. **Download deployment script to your server:**
   ```bash
   wget https://raw.githubusercontent.com/YOUR_REPO/remote-server-deploy.sh
   chmod +x remote-server-deploy.sh
   ```

2. **Run deployment:**
   ```bash
   ./remote-server-deploy.sh
   ```

### Method 2: Manual Docker Commands

1. **Pull the image:**
   ```bash
   docker pull coinstardon/crypto-trading-engine:latest
   ```

2. **Create volume:**
   ```bash
   docker volume create crypto-trading-logs
   ```

3. **Run container:**
   ```bash
   docker run -d \
     --name crypto-trading-engine \
     --restart unless-stopped \
     -p 8080:8080 \
     -v crypto-trading-logs:/app/trading/logs \
     -e TZ=UTC \
     coinstardon/crypto-trading-engine:latest
   ```

### Method 3: Using Docker Compose

1. **Download docker-compose file:**
   ```bash
   wget https://raw.githubusercontent.com/YOUR_REPO/docker-compose.remote.yml
   ```

2. **Deploy:**
   ```bash
   docker-compose -f docker-compose.remote.yml up -d
   ```

## Monitoring & Management

### View Logs
```bash
docker logs -f crypto-trading-engine
```

### Check Status
```bash
docker ps | grep crypto-trading-engine
```

### Access Container
```bash
docker exec -it crypto-trading-engine /bin/bash
```

### Resource Monitoring
```bash
docker stats crypto-trading-engine
```

### Restart Container
```bash
docker restart crypto-trading-engine
```

## Application Access

- **Web Interface:** http://YOUR_SERVER_IP:8080
- **API Endpoints:** http://YOUR_SERVER_IP:8080/api/
- **Health Check:** http://YOUR_SERVER_IP:8080/health

## Troubleshooting

### Container Won't Start
```bash
docker logs crypto-trading-engine
```

### Port Already in Use
```bash
sudo netstat -tulpn | grep :8080
docker ps -a | grep 8080
```

### Update Deployment
```bash
docker pull coinstardon/crypto-trading-engine:latest
docker stop crypto-trading-engine
docker rm crypto-trading-engine
# Run container command again
```

## Security Considerations

1. **Firewall Configuration**
   ```bash
   sudo ufw allow 8080
   sudo ufw enable
   ```

2. **SSL/TLS Setup**
   - Use nginx proxy for HTTPS
   - Configure Let's Encrypt certificates

3. **Environment Variables**
   - Store sensitive data in Docker secrets
   - Use .env files for configuration

## Backup & Recovery

### Backup Logs
```bash
docker run --rm -v crypto-trading-logs:/data -v $(pwd):/backup alpine tar czf /backup/logs-backup.tar.gz /data
```

### Restore Logs
```bash
docker run --rm -v crypto-trading-logs:/data -v $(pwd):/backup alpine tar xzf /backup/logs-backup.tar.gz -C /
```
EOF

print_info "✅ Created REMOTE_DEPLOYMENT_GUIDE.md"

echo ""
echo "================================================================"
echo "🎉 REMOTE DEPLOYMENT SETUP COMPLETED!"
echo "================================================================"
echo ""
echo "📁 Files created:"
echo "  - remote-server-deploy.sh      (Remote deployment script)"
echo "  - docker-compose.remote.yml    (Docker Compose configuration)"
echo "  - REMOTE_DEPLOYMENT_GUIDE.md   (Complete deployment guide)"
echo ""
echo "🚀 Docker Hub Images:"
echo "  - ${DOCKER_HUB_USERNAME}/${IMAGE_NAME}:${VERSION}"
echo "  - ${DOCKER_HUB_USERNAME}/${IMAGE_NAME}:latest"
echo ""
echo "🔗 Quick deployment on remote server:"
echo "  wget YOUR_REPO_URL/remote-server-deploy.sh && chmod +x remote-server-deploy.sh && ./remote-server-deploy.sh"
echo ""
echo "📋 Next steps:"
echo "  1. Verify images are available on Docker Hub"
echo "  2. Test deployment on your remote server"
echo "  3. Configure SSL/domain if needed"
echo "  4. Set up monitoring and backups"
echo ""
