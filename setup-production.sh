#!/bin/bash

# ================================================================
# Production Environment Setup Script
# Crypto Trading Engine - Complete Production Setup
# ================================================================
# This script sets up the complete production environment locally
# before deploying to remote servers

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="crypto-trading-engine"
DOCKER_IMAGE="coinstardon/crypto-trading-engine:1.0.0"

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

# Check if Docker is installed and running
check_docker() {
    log_step "Checking Docker installation..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    log_success "Docker and Docker Compose are available"
}

# Verify the Docker image exists
check_docker_image() {
    log_step "Checking Docker image..."
    
    if ! docker images --format "table {{.Repository}}:{{.Tag}}" | grep -q "$DOCKER_IMAGE"; then
        log_warning "Docker image $DOCKER_IMAGE not found locally"
        log_info "Building Docker image..."
        
        if docker build -t "$DOCKER_IMAGE" .; then
            log_success "Docker image built successfully"
        else
            log_error "Failed to build Docker image"
            exit 1
        fi
    else
        log_success "Docker image $DOCKER_IMAGE is available"
    fi
}

# Setup environment configuration
setup_environment() {
    log_step "Setting up environment configuration..."
    
    # Create .env.production if it doesn't exist
    if [[ ! -f ".env.production" ]]; then
        if [[ -f ".env.production.template" ]]; then
            log_info "Creating .env.production from template..."
            cp .env.production.template .env.production
            log_warning "Please edit .env.production with your actual configuration values"
        else
            log_error ".env.production.template not found"
            exit 1
        fi
    else
        log_success ".env.production file exists"
    fi
    
    # Check for required configuration files
    local config_files=(
        "config/trading_config.json"
        "config/ml_config.json"
    )
    
    for file in "${config_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_warning "Configuration file $file not found"
            log_info "Creating default configuration..."
            mkdir -p "$(dirname "$file")"
            
            if [[ "$file" == *"trading_config.json" ]]; then
                cat > "$file" << 'EOF'
{
    "trading": {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "max_trades_per_day": 20,
        "position_size_percent": 5,
        "stop_loss_percent": 2.0,
        "take_profit_percent": 4.0,
        "min_time_between_trades": 300
    },
    "risk_management": {
        "max_daily_loss_percent": 5.0,
        "max_drawdown_percent": 10.0,
        "emergency_stop_enabled": true
    },
    "indicators": {
        "rsi_period": 14,
        "ma_short": 10,
        "ma_long": 20,
        "bollinger_period": 20
    }
}
EOF
            elif [[ "$file" == *"ml_config.json" ]]; then
                cat > "$file" << 'EOF'
{
    "model": {
        "type": "random_forest",
        "features": ["rsi", "macd", "bb_position", "volume_sma"],
        "lookback_period": 50,
        "prediction_threshold": 0.6
    },
    "training": {
        "retrain_interval": 3600,
        "validation_split": 0.2,
        "test_split": 0.1
    },
    "data": {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "history_days": 365
    }
}
EOF
            fi
        else
            log_success "Configuration file $file exists"
        fi
    done
}

# Create required directories
setup_directories() {
    log_step "Setting up required directories..."
    
    local directories=(
        "monitoring/grafana/dashboards"
        "monitoring/grafana/datasources"
        "nginx/ssl"
        "nginx/logs"
        "config"
        "scripts"
    )
    
    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log_info "Created directory: $dir"
        fi
    done
    
    log_success "Directory structure created"
}

# Setup SSL certificates (self-signed for development)
setup_ssl_certificates() {
    log_step "Setting up SSL certificates..."
    
    local ssl_dir="nginx/ssl"
    
    if [[ ! -f "$ssl_dir/cert.pem" || ! -f "$ssl_dir/key.pem" ]]; then
        log_info "Creating self-signed SSL certificates for development..."
        
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout "$ssl_dir/key.pem" \
            -out "$ssl_dir/cert.pem" \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost" \
            &> /dev/null
        
        log_success "SSL certificates created"
        log_warning "These are self-signed certificates for development only"
        log_warning "Use proper certificates from a CA for production"
    else
        log_success "SSL certificates already exist"
    fi
}

# Test the production setup locally
test_production_setup() {
    log_step "Testing production setup locally..."
    
    # Stop any existing containers
    log_info "Stopping existing containers..."
    docker-compose -f docker-compose.prod.yml down --remove-orphans &>/dev/null || true
    
    # Start services
    log_info "Starting production services..."
    if docker-compose -f docker-compose.prod.yml up -d; then
        log_success "Services started successfully"
        
        # Wait for services to be ready
        log_info "Waiting for services to be ready (60 seconds)..."
        sleep 60
        
        # Check service health
        log_info "Checking service health..."
        docker-compose -f docker-compose.prod.yml ps
        
        # Test endpoints
        log_info "Testing service endpoints..."
        
        # Test main application
        if curl -sf http://localhost:8080/health > /dev/null 2>&1; then
            log_success "✓ Web dashboard is accessible"
        else
            log_warning "✗ Web dashboard is not accessible"
        fi
        
        # Test if containers are running
        local running_containers=$(docker-compose -f docker-compose.prod.yml ps -q | wc -l)
        log_info "Running containers: $running_containers"
        
        log_success "Production setup test completed"
        
        # Optionally stop the test environment
        read -p "Stop test environment? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker-compose -f docker-compose.prod.yml down
            log_info "Test environment stopped"
        fi
        
    else
        log_error "Failed to start production services"
        exit 1
    fi
}

# Generate deployment summary
generate_deployment_summary() {
    log_step "Generating deployment summary..."
    
    cat > DEPLOYMENT_CHECKLIST.md << 'EOF'
# Production Deployment Checklist

## Pre-Deployment Checklist

### Configuration
- [ ] Update `.env.production` with real API keys and secrets
- [ ] Configure `config/trading_config.json` with your trading parameters
- [ ] Configure `config/ml_config.json` with your ML model settings
- [ ] Update `nginx/nginx.conf` with your domain name (if using SSL)
- [ ] Generate proper SSL certificates for production

### Security
- [ ] Change default passwords in `.env.production`
- [ ] Configure firewall rules on the server
- [ ] Set up proper authentication for monitoring endpoints
- [ ] Review and update CORS settings
- [ ] Enable rate limiting in production

### Server Requirements
- [ ] Minimum 4GB RAM, 2 CPU cores
- [ ] 50GB+ storage space
- [ ] Docker and Docker Compose installed
- [ ] SSH access configured
- [ ] Proper user permissions set

## Deployment Commands

### Local Testing
```bash
# Test production setup locally
./setup-production.sh

# Deploy to remote server
./deploy-remote.sh [server_ip] [ssh_user]
```

### Remote Server Setup
```bash
# SSH into your server
ssh user@your-server-ip

# Clone the repository (if needed)
git clone https://github.com/your-username/crypto-trading-engine.git
cd crypto-trading-engine

# Run the deployment script
./deploy-remote.sh
```

## Post-Deployment Tasks

### Monitoring Setup
- [ ] Access Grafana at `http://your-server:3000`
- [ ] Configure Grafana dashboards
- [ ] Set up alerting rules
- [ ] Test notification channels

### Trading Configuration
- [ ] Start with paper trading mode
- [ ] Monitor logs for errors
- [ ] Verify API connections
- [ ] Test emergency stop functionality

### Maintenance
- [ ] Set up automated backups
- [ ] Configure log rotation
- [ ] Monitor resource usage
- [ ] Plan for updates and maintenance windows

## Important URLs (Replace with your server IP)

- Web Dashboard: http://your-server:8080
- Trading API: http://your-server:8081
- ML Service: http://your-server:8082
- Grafana: http://your-server:3000
- Prometheus: http://your-server:9090

## Security Notes

⚠️ **IMPORTANT**: 
- Never commit real API keys to version control
- Use environment-specific configuration files
- Regularly rotate secrets and passwords
- Monitor for unauthorized access attempts
- Keep the system updated with security patches

## Support

If you encounter issues:
1. Check container logs: `docker-compose -f docker-compose.prod.yml logs`
2. Verify service health: `docker-compose -f docker-compose.prod.yml ps`
3. Check system resources: `docker stats`
4. Review configuration files for errors

EOF

    log_success "Deployment checklist created: DEPLOYMENT_CHECKLIST.md"
}

# Create quick deployment scripts
create_quick_scripts() {
    log_step "Creating quick deployment scripts..."
    
    # Create quick start script
    cat > quick-start.sh << 'EOF'
#!/bin/bash
# Quick start script for production environment

set -e

echo "🚀 Starting Crypto Trading Engine (Production Mode)"

# Check if Docker image exists
if ! docker images --format "table {{.Repository}}:{{.Tag}}" | grep -q "coinstardon/crypto-trading-engine:1.0.0"; then
    echo "❌ Docker image not found. Please run: docker build -t coinstardon/crypto-trading-engine:1.0.0 ."
    exit 1
fi

# Start services
docker-compose -f docker-compose.prod.yml up -d

echo "✅ Services started successfully!"
echo "📊 Web Dashboard: http://localhost:8080"
echo "🔧 Trading API: http://localhost:8081"
echo "🤖 ML Service: http://localhost:8082"
echo "📈 Grafana: http://localhost:3000"

echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service status
docker-compose -f docker-compose.prod.yml ps

echo "🎉 Crypto Trading Engine is running!"
EOF

    # Create quick stop script
    cat > quick-stop.sh << 'EOF'
#!/bin/bash
# Quick stop script for production environment

echo "🛑 Stopping Crypto Trading Engine..."

docker-compose -f docker-compose.prod.yml down

echo "✅ All services stopped successfully!"
EOF

    # Create logs viewing script
    cat > view-logs.sh << 'EOF'
#!/bin/bash
# View logs script

SERVICE=${1:-}

if [[ -z "$SERVICE" ]]; then
    echo "📋 Available services:"
    docker-compose -f docker-compose.prod.yml config --services
    echo ""
    echo "Usage: $0 [service_name]"
    echo "Example: $0 crypto-trader"
    echo "Or use 'all' to view all logs"
    exit 1
fi

if [[ "$SERVICE" == "all" ]]; then
    docker-compose -f docker-compose.prod.yml logs -f
else
    docker-compose -f docker-compose.prod.yml logs -f "$SERVICE"
fi
EOF

    # Make scripts executable
    chmod +x quick-start.sh quick-stop.sh view-logs.sh deploy-remote.sh

    log_success "Quick deployment scripts created"
    log_info "  - quick-start.sh: Start all services"
    log_info "  - quick-stop.sh: Stop all services"
    log_info "  - view-logs.sh: View service logs"
    log_info "  - deploy-remote.sh: Deploy to remote server"
}

# Main setup function
main() {
    echo -e "${PURPLE}╔════════════════════════════════════════════╗${NC}"
    echo -e "${PURPLE}║     Crypto Trading Engine Production      ║${NC}"
    echo -e "${PURPLE}║           Environment Setup                ║${NC}"
    echo -e "${PURPLE}╚════════════════════════════════════════════╝${NC}"
    echo
    
    log_info "Setting up production environment..."
    
    check_docker
    check_docker_image
    setup_directories
    setup_environment
    setup_ssl_certificates
    generate_deployment_summary
    create_quick_scripts
    
    log_success "✅ Production environment setup completed!"
    
    echo
    echo -e "${GREEN}Next Steps:${NC}"
    echo "1. Edit .env.production with your actual API keys and configuration"
    echo "2. Review and customize configuration files in config/"
    echo "3. Test locally with: ./quick-start.sh"
    echo "4. Deploy to remote server with: ./deploy-remote.sh [server_ip] [user]"
    echo
    echo -e "${YELLOW}Important:${NC} Please review DEPLOYMENT_CHECKLIST.md before deploying to production!"
    echo
    
    # Ask if user wants to test locally
    read -p "Test the production setup locally now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        test_production_setup
    else
        log_info "You can test later with: ./quick-start.sh"
    fi
}

# Run the setup
main "$@"
