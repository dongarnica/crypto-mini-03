#!/bin/bash

# ================================================================
# Production Remote Server Deployment Script
# Crypto Trading Engine - Remote Server Setup
# ================================================================
# This script deploys the crypto trading engine to a remote production server
# Usage: ./deploy-remote.sh [server_ip] [ssh_user] [options]

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="crypto-trading-engine"
DOCKER_IMAGE="coinstardon/crypto-trading-engine:1.0.0"
COMPOSE_FILE="docker-compose.prod.yml"

# Default values
SERVER_IP="${1:-}"
SSH_USER="${2:-root}"
DEPLOY_PATH="/opt/crypto-trading"
PUSH_TO_REGISTRY="${PUSH_TO_REGISTRY:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${PURPLE}[STEP]${NC} $1"
}

# Usage information
show_usage() {
    cat << EOF
Usage: $0 [server_ip] [ssh_user] [options]

Arguments:
  server_ip     IP address of the remote server
  ssh_user      SSH username (default: root)

Environment Variables:
  PUSH_TO_REGISTRY   Push image to registry instead of transferring (true/false)

Examples:
  $0 192.168.1.100 ubuntu
  PUSH_TO_REGISTRY=true $0 192.168.1.100 deploy_user

Prerequisites:
  - Docker image built locally: $DOCKER_IMAGE
  - SSH key-based authentication configured
  - Required configuration files present
  - Docker Hub credentials configured (if using registry)

EOF
}

# Check prerequisites
check_prerequisites() {
    log_step "Checking prerequisites..."
    
    # Check if Docker image exists locally
    if ! docker images --format "table {{.Repository}}:{{.Tag}}" | grep -q "$DOCKER_IMAGE"; then
        log_error "Docker image $DOCKER_IMAGE not found locally"
        log_info "Please build the image first with: docker build -t $DOCKER_IMAGE ."
        exit 1
    fi
    
    # Check required files
    local required_files=(
        "$COMPOSE_FILE"
        "config/trading_config.json"
        "config/ml_config.json"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_error "Required file not found: $file"
            exit 1
        fi
    done
    
    # Check for .env.production file
    if [[ ! -f ".env.production" ]]; then
        log_warning ".env.production file not found"
        log_info "Creating template .env.production file..."
        cat > .env.production << 'ENV_EOF'
# Production Environment Variables
ENVIRONMENT=production
LOG_LEVEL=INFO
PAPER_TRADING=false

# API Keys (REPLACE WITH ACTUAL VALUES)
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here

# Redis Configuration
REDIS_PASSWORD=your_redis_password_here

# Grafana Configuration
GRAFANA_PASSWORD=your_grafana_password_here

# Security Settings
JWT_SECRET=your_jwt_secret_here

# Trading Configuration
MAX_TRADES_PER_DAY=20
MIN_TIME_BETWEEN_TRADES=300
ENV_EOF
        log_warning "Please edit .env.production with your actual configuration before deploying!"
        read -p "Press Enter to continue or Ctrl+C to exit and edit the file..."
    fi
    
    log_success "Prerequisites check passed"
}

# Validate server connection
validate_server() {
    if [[ -z "$SERVER_IP" ]]; then
        log_error "Server IP address is required"
        show_usage
        exit 1
    fi
    
    log_step "Testing connection to $SSH_USER@$SERVER_IP..."
    if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$SSH_USER@$SERVER_IP" "echo 'Connection test successful'" > /dev/null 2>&1; then
        log_error "Cannot connect to server $SERVER_IP"
        log_info "Please ensure:"
        log_info "1. Server is accessible via SSH"
        log_info "2. SSH key-based authentication is configured"
        log_info "3. User $SSH_USER has sudo privileges"
        exit 1
    fi
    
    log_success "Server connection validated"
}

# Setup remote server environment
setup_remote_server() {
    log_step "Setting up remote server environment..."
    
    ssh "$SSH_USER@$SERVER_IP" << 'EOF'
        set -euo pipefail
        
        echo "Updating system packages..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update -qq
            sudo apt-get install -y curl wget git htop net-tools ufw
        elif command -v yum &> /dev/null; then
            sudo yum update -y -q
            sudo yum install -y curl wget git htop net-tools firewalld
        fi
        
        # Install Docker if not present
        if ! command -v docker &> /dev/null; then
            echo "Installing Docker..."
            curl -fsSL https://get.docker.com -o get-docker.sh
            sudo sh get-docker.sh
            sudo usermod -aG docker $USER
            sudo systemctl enable docker
            sudo systemctl start docker
            rm get-docker.sh
        else
            echo "Docker is already installed"
        fi
        
        # Install Docker Compose if not present
        if ! command -v docker-compose &> /dev/null; then
            echo "Installing Docker Compose..."
            DOCKER_COMPOSE_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep 'tag_name' | cut -d\" -f4)
            sudo curl -L "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
            sudo chmod +x /usr/local/bin/docker-compose
        else
            echo "Docker Compose is already installed"
        fi
        
        # Create deployment directory structure
        sudo mkdir -p /opt/crypto-trading/{logs,historical-data,ml-models,binance-data,config,nginx,monitoring,backups}
        sudo chown -R $USER:$USER /opt/crypto-trading
        
        # Set up log rotation
        sudo tee /etc/logrotate.d/crypto-trading > /dev/null << 'LOGROTATE'
/opt/crypto-trading/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0644 trader trader
    postrotate
        /usr/bin/docker-compose -f /opt/crypto-trading/docker-compose.prod.yml restart crypto-trader 2>/dev/null || true
    endscript
}
LOGROTATE
        
        # Configure basic firewall (optional)
        if command -v ufw &> /dev/null; then
            echo "Configuring UFW firewall..."
            sudo ufw --force reset > /dev/null
            sudo ufw default deny incoming
            sudo ufw default allow outgoing
            sudo ufw allow ssh
            sudo ufw allow 80/tcp   # HTTP
            sudo ufw allow 443/tcp  # HTTPS
            sudo ufw allow 8080/tcp # Web Dashboard
            # sudo ufw --force enable  # Uncomment to enable firewall
        fi
        
        echo "Remote server setup completed successfully"
EOF
    
    log_success "Remote server environment setup completed"
}

# Deploy application files
deploy_application() {
    log_step "Deploying application files to remote server..."
    
    # Create temporary deployment package
    local temp_dir=$(mktemp -d)
    local deploy_package="$temp_dir/crypto-trading-deploy.tar.gz"
    
    log_info "Creating deployment package..."
    tar -czf "$deploy_package" \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='node_modules' \
        --exclude='.env' \
        --exclude='historical_exports' \
        --exclude='binance_exports' \
        --exclude='trading/logs' \
        --exclude='ml_results' \
        --exclude='*.tar' \
        --exclude='*.tar.gz' \
        "$COMPOSE_FILE" \
        .env.production \
        config/ \
        nginx/ \
        monitoring/ \
        scripts/ \
        README.md 2>/dev/null || true
    
    # Transfer deployment package
    log_info "Transferring deployment package ($(du -h "$deploy_package" | cut -f1))..."
    scp -q "$deploy_package" "$SSH_USER@$SERVER_IP:$DEPLOY_PATH/"
    
    # Extract and setup on remote server
    ssh "$SSH_USER@$SERVER_IP" << EOF
        set -euo pipefail
        cd $DEPLOY_PATH
        
        # Backup existing deployment if it exists
        if [[ -f docker-compose.prod.yml ]]; then
            echo "Backing up existing deployment..."
            BACKUP_DIR="backups/\$(date +%Y%m%d_%H%M%S)"
            mkdir -p "\$BACKUP_DIR"
            cp -r *.yml config/ nginx/ monitoring/ "\$BACKUP_DIR/" 2>/dev/null || true
            echo "Backup created in \$BACKUP_DIR"
        fi
        
        # Extract new deployment
        echo "Extracting deployment package..."
        tar -xzf crypto-trading-deploy.tar.gz
        rm crypto-trading-deploy.tar.gz
        
        # Rename .env.production to .env
        if [[ -f .env.production ]]; then
            cp .env.production .env
        fi
        
        # Set proper permissions
        find . -name "*.sh" -exec chmod +x {} \; 2>/dev/null || true
        
        echo "Application files deployed successfully to $DEPLOY_PATH"
EOF
    
    # Cleanup
    rm -rf "$temp_dir"
    
    log_success "✓ Application deployment completed"
}

# Deploy Docker image
deploy_docker_image() {
    log_step "Deploying Docker image to remote server..."
    
    if [[ "${PUSH_TO_REGISTRY}" == "true" ]]; then
        # Push to registry and pull on server
        log_info "Pushing image to Docker registry..."
        docker push "$DOCKER_IMAGE"
        
        ssh "$SSH_USER@$SERVER_IP" << EOF
            set -euo pipefail
            cd $DEPLOY_PATH
            
            echo "Pulling Docker image from registry..."
            docker pull $DOCKER_IMAGE
            
            echo "Docker image pulled successfully"
EOF
    else
        # Save image locally and transfer
        log_info "Saving Docker image locally..."
        local temp_dir=$(mktemp -d)
        local image_file="$temp_dir/crypto-trading-engine.tar"
        
        docker save "$DOCKER_IMAGE" -o "$image_file"
        local image_size=$(du -h "$image_file" | cut -f1)
        
        log_info "Transferring Docker image ($image_size)..."
        scp -q "$image_file" "$SSH_USER@$SERVER_IP:$DEPLOY_PATH/"
        
        ssh "$SSH_USER@$SERVER_IP" << EOF
            set -euo pipefail
            cd $DEPLOY_PATH
            
            echo "Loading Docker image..."
            docker load -i crypto-trading-engine.tar
            rm crypto-trading-engine.tar
            
            echo "Docker image loaded successfully"
EOF
        
        rm -rf "$temp_dir"
    fi
    
    log_success "✓ Docker image deployment completed"
}

# Start services
start_services() {
    log_step "Starting services on remote server..."
    
    ssh "$SSH_USER@$SERVER_IP" << EOF
        set -euo pipefail
        cd $DEPLOY_PATH
        
        # Stop existing services gracefully
        echo "Stopping existing services..."
        docker-compose -f $COMPOSE_FILE down --remove-orphans --timeout 30 2>/dev/null || true
        
        # Cleanup old containers and images if needed
        echo "Cleaning up old containers..."
        docker system prune -f > /dev/null 2>&1 || true
        
        # Start new services
        echo "Starting services in production mode..."
        docker-compose -f $COMPOSE_FILE up -d
        
        # Wait for services to initialize
        echo "Waiting for services to initialize (60 seconds)..."
        sleep 60
        
        # Check service status
        echo "=== Service Status ==="
        docker-compose -f $COMPOSE_FILE ps
        
        echo "Services started successfully"
EOF
    
    log_success "✓ Services deployment completed"
}

# Verify deployment
verify_deployment() {
    log_step "Verifying deployment health..."
    
    ssh "$SSH_USER@$SERVER_IP" "cd $DEPLOY_PATH && bash" << 'EOF'
        set -euo pipefail
        
        echo "=== Container Status ==="
        docker-compose -f docker-compose.prod.yml ps
        
        echo -e "\n=== Container Health ==="
        for container in $(docker-compose -f docker-compose.prod.yml ps -q); do
            name=$(docker inspect --format='{{.Name}}' $container | sed 's/^\///')
            status=$(docker inspect --format='{{.State.Health.Status}}' $container 2>/dev/null || echo "no-health-check")
            echo "$name: $status"
        done
        
        echo -e "\n=== Recent Container Logs ==="
        docker-compose -f docker-compose.prod.yml logs --tail=10 --no-color
        
        echo -e "\n=== System Resources ==="
        echo "Memory Usage:"
        free -h
        echo -e "\nDisk Usage:"
        df -h /opt/crypto-trading
        echo -e "\nDocker Disk Usage:"
        docker system df
        
        echo -e "\n=== Network Connectivity Tests ==="
        # Test internal service connectivity
        if docker-compose -f docker-compose.prod.yml exec -T crypto-trader curl -sf http://localhost:8080/health > /dev/null 2>&1; then
            echo "✓ Web dashboard health check passed"
        else
            echo "✗ Web dashboard health check failed"
        fi
        
        # Check if ports are accessible externally
        if nc -z localhost 8080 2>/dev/null; then
            echo "✓ Port 8080 is accessible"
        else
            echo "✗ Port 8080 is not accessible"
        fi
        
        echo -e "\n=== Deployment Verification Completed ==="
EOF
    
    log_success "✓ Deployment verification completed"
}

# Display post-deployment information
show_deployment_info() {
    log_step "Deployment Summary"
    
    cat << EOF

${GREEN}=== DEPLOYMENT COMPLETED SUCCESSFULLY ===${NC}

${BLUE}Server Information:${NC}
  Server: $SSH_USER@$SERVER_IP
  Deploy Path: $DEPLOY_PATH
  Docker Image: $DOCKER_IMAGE
  Compose File: $COMPOSE_FILE

${BLUE}Application URLs:${NC}
  🌐 Web Dashboard: http://$SERVER_IP:8080
  📊 Trading API: http://$SERVER_IP:8081
  🤖 ML Service: http://$SERVER_IP:8082
  📈 Grafana: http://$SERVER_IP:3000
  📊 Prometheus: http://$SERVER_IP:9090

${BLUE}Useful Commands:${NC}
  View logs: ssh $SSH_USER@$SERVER_IP "cd $DEPLOY_PATH && docker-compose -f $COMPOSE_FILE logs -f"
  Restart: ssh $SSH_USER@$SERVER_IP "cd $DEPLOY_PATH && docker-compose -f $COMPOSE_FILE restart"
  Stop: ssh $SSH_USER@$SERVER_IP "cd $DEPLOY_PATH && docker-compose -f $COMPOSE_FILE down"
  Status: ssh $SSH_USER@$SERVER_IP "cd $DEPLOY_PATH && docker-compose -f $COMPOSE_FILE ps"

${YELLOW}Security Reminders:${NC}
  ⚠️  Configure SSL/TLS certificates for production use
  ⚠️  Set up proper firewall rules and access controls
  ⚠️  Update default passwords in .env file
  ⚠️  Enable monitoring and alerting
  ⚠️  Regular backup strategy for persistent data

${YELLOW}Next Steps:${NC}
  1. Configure SSL certificates
  2. Set up monitoring alerts
  3. Test trading functionality with paper trading first
  4. Configure backup automation
  5. Set up log aggregation

EOF
}

# Main deployment function
main() {
    echo -e "${PURPLE}╔════════════════════════════════════════════╗${NC}"
    echo -e "${PURPLE}║       Crypto Trading Engine Deployment    ║${NC}"
    echo -e "${PURPLE}║            Remote Server Setup             ║${NC}"
    echo -e "${PURPLE}╚════════════════════════════════════════════╝${NC}"
    echo
    
    # Show usage if no arguments provided
    if [[ $# -eq 0 ]]; then
        show_usage
        exit 1
    fi
    
    log_info "Starting deployment to remote server..."
    log_info "Target: $SSH_USER@$SERVER_IP"
    
    # Execute deployment steps
    check_prerequisites
    validate_server
    setup_remote_server
    deploy_application
    deploy_docker_image
    start_services
    verify_deployment
    show_deployment_info
    
    log_success "Remote deployment completed successfully! 🚀"
}

# Handle command line arguments
case "${1:-}" in
    -h|--help)
        show_usage
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac
