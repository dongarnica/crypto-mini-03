#!/bin/bash
set -e

echo "🐳 Enhanced Crypto Trading System - Docker Deployment"
echo "====================================================="

# Configuration
IMAGE_NAME="crypto-trading-enhanced"
CONTAINER_NAME="crypto-trading-system"
DOCKER_COMPOSE_FILE="docker-compose.enhanced.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if .env file exists
    if [ ! -f ".env" ]; then
        print_warning ".env file not found. Creating template..."
        cat > .env << EOF
# Binance API Configuration
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_secret_here

# Alpaca API Configuration (Optional)
ALPACA_API_KEY=your_alpaca_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Trading Configuration
PAPER_TRADING=true
LOG_LEVEL=INFO
ENHANCED_OUTPUT=true
DOCKER_MODE=true

# Process Timing Configuration
TRADING_CYCLE_MINUTES=5
EXPORT_CYCLE_HOURS=2
ML_RETRAIN_CYCLE_HOURS=6
EOF
        print_warning "Please edit .env file with your API credentials before running the container."
    fi
    
    # Create data directories
    print_status "Creating data directories..."
    mkdir -p data/{trading-logs,ml-results,binance-exports,historical-exports}
    mkdir -p backups
    
    print_success "Prerequisites check completed"
}

# Function to build the Docker image
build_image() {
    print_status "Building enhanced Docker image..."
    
    # Build the image with enhanced features
    docker build \
        --target production \
        --tag ${IMAGE_NAME}:latest \
        --file Dockerfile \
        . || {
        print_error "Docker build failed"
        exit 1
    }
    
    print_success "Docker image built successfully: ${IMAGE_NAME}:latest"
}

# Function to stop existing containers
stop_existing() {
    print_status "Stopping existing containers..."
    
    # Stop container if running
    if docker ps -q -f name=${CONTAINER_NAME} | grep -q .; then
        docker stop ${CONTAINER_NAME}
        print_status "Stopped existing container: ${CONTAINER_NAME}"
    fi
    
    # Remove container if exists
    if docker ps -aq -f name=${CONTAINER_NAME} | grep -q .; then
        docker rm ${CONTAINER_NAME}
        print_status "Removed existing container: ${CONTAINER_NAME}"
    fi
    
    # Stop Docker Compose services if running
    if [ -f "${DOCKER_COMPOSE_FILE}" ]; then
        docker-compose -f ${DOCKER_COMPOSE_FILE} down || true
    fi
}

# Function to run the enhanced container
run_container() {
    print_status "Starting enhanced crypto trading container..."
    
    # Check if we should use Docker Compose or direct docker run
    if [ "$1" = "compose" ]; then
        run_with_compose
    else
        run_with_docker
    fi
}

# Function to run with Docker Compose (recommended)
run_with_compose() {
    print_status "Using Docker Compose for enhanced deployment..."
    
    # Start with Docker Compose
    docker-compose -f ${DOCKER_COMPOSE_FILE} up -d crypto-trader-enhanced
    
    print_success "Enhanced crypto trading system started with Docker Compose"
    print_status "Container name: crypto-trading-enhanced"
    print_status "View logs with: docker-compose -f ${DOCKER_COMPOSE_FILE} logs -f crypto-trader-enhanced"
}

# Function to run with direct Docker command
run_with_docker() {
    print_status "Using direct Docker run..."
    
    # Run the container with enhanced features
    docker run -d \
        --name ${CONTAINER_NAME} \
        --restart unless-stopped \
        --env-file .env \
        -e ENHANCED_OUTPUT=true \
        -e DOCKER_MODE=true \
        -v "$(pwd)/data/trading-logs:/app/trading/logs" \
        -v "$(pwd)/data/ml-results:/app/ml_results" \
        -v "$(pwd)/data/binance-exports:/app/binance_exports" \
        -v "$(pwd)/.env:/app/.env:ro" \
        --memory=4g \
        --cpus=2.0 \
        ${IMAGE_NAME}:latest
    
    print_success "Enhanced crypto trading container started"
    print_status "Container name: ${CONTAINER_NAME}"
    print_status "View logs with: docker logs -f ${CONTAINER_NAME}"
}

# Function to show container status
show_status() {
    print_status "Container Status:"
    echo "=================="
    
    # Show running containers
    if docker ps -f name=${CONTAINER_NAME} --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -q ${CONTAINER_NAME}; then
        docker ps -f name=${CONTAINER_NAME} --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    elif docker ps -f name=crypto-trading-enhanced --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -q crypto-trading-enhanced; then
        docker ps -f name=crypto-trading-enhanced --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    else
        print_warning "No crypto trading containers are currently running"
    fi
    
    echo ""
    print_status "Data Directories:"
    echo "=================="
    ls -la data/ 2>/dev/null || print_warning "Data directory not found"
}

# Function to view logs
view_logs() {
    print_status "Viewing container logs..."
    
    if docker ps -q -f name=crypto-trading-enhanced | grep -q .; then
        docker logs -f crypto-trading-enhanced
    elif docker ps -q -f name=${CONTAINER_NAME} | grep -q .; then
        docker logs -f ${CONTAINER_NAME}
    else
        print_error "No running crypto trading container found"
        exit 1
    fi
}

# Function to enter container shell
enter_shell() {
    print_status "Entering container shell..."
    
    if docker ps -q -f name=crypto-trading-enhanced | grep -q .; then
        docker exec -it crypto-trading-enhanced /bin/bash
    elif docker ps -q -f name=${CONTAINER_NAME} | grep -q .; then
        docker exec -it ${CONTAINER_NAME} /bin/bash
    else
        print_error "No running crypto trading container found"
        exit 1
    fi
}

# Function to show help
show_help() {
    echo "Enhanced Crypto Trading System - Docker Deployment Script"
    echo "========================================================"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  build           Build the Docker image"
    echo "  run             Run the container (default method)"
    echo "  run-compose     Run with Docker Compose (recommended)"
    echo "  run-docker      Run with direct Docker command"
    echo "  stop            Stop running containers"
    echo "  restart         Restart the system (stop + build + run)"
    echo "  status          Show container status"
    echo "  logs            View container logs"
    echo "  shell           Enter container shell"
    echo "  cleanup         Clean up containers and images"
    echo "  dev             Start development container"
    echo "  help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build                    # Build the Docker image"
    echo "  $0 run-compose              # Start with Docker Compose"
    echo "  $0 logs                     # View live logs"
    echo "  $0 restart                  # Full restart"
    echo ""
    echo "Features:"
    echo "  ✅ Enhanced output with countdown timers"
    echo "  ✅ Integrated data export service"
    echo "  ✅ ML model retraining"
    echo "  ✅ Trading engine with enhanced display"
    echo "  ✅ Automatic process monitoring"
    echo "  ✅ Docker health checks"
}

# Function to cleanup
cleanup() {
    print_status "Cleaning up containers and images..."
    
    # Stop and remove containers
    docker-compose -f ${DOCKER_COMPOSE_FILE} down --volumes --remove-orphans 2>/dev/null || true
    docker stop ${CONTAINER_NAME} 2>/dev/null || true
    docker rm ${CONTAINER_NAME} 2>/dev/null || true
    
    # Remove images
    docker rmi ${IMAGE_NAME}:latest 2>/dev/null || true
    
    # Prune unused images
    docker image prune -f
    
    print_success "Cleanup completed"
}

# Function to start development container
start_dev() {
    print_status "Starting development container..."
    
    stop_existing
    build_image
    
    # Run development container with Docker Compose
    docker-compose -f ${DOCKER_COMPOSE_FILE} --profile development up -d crypto-trader-dev-enhanced
    
    print_success "Development container started"
    print_status "Access with: docker exec -it crypto-trading-dev-enhanced /bin/bash"
}

# Main script logic
case "$1" in
    "build")
        check_prerequisites
        build_image
        ;;
    "run")
        check_prerequisites
        stop_existing
        build_image
        run_container "compose"
        show_status
        ;;
    "run-compose")
        check_prerequisites
        stop_existing
        build_image
        run_container "compose"
        show_status
        ;;
    "run-docker")
        check_prerequisites
        stop_existing
        build_image
        run_container "docker"
        show_status
        ;;
    "stop")
        stop_existing
        print_success "Containers stopped"
        ;;
    "restart")
        print_status "Restarting enhanced crypto trading system..."
        stop_existing
        check_prerequisites
        build_image
        run_container "compose"
        show_status
        ;;
    "status")
        show_status
        ;;
    "logs")
        view_logs
        ;;
    "shell")
        enter_shell
        ;;
    "cleanup")
        cleanup
        ;;
    "dev")
        start_dev
        ;;
    "help"|"--help"|"-h")
        show_help
        ;;
    "")
        print_status "No command specified. Starting enhanced system with Docker Compose..."
        check_prerequisites
        stop_existing
        build_image
        run_container "compose"
        show_status
        echo ""
        print_success "Enhanced crypto trading system is now running!"
        print_status "Use '$0 logs' to view live output"
        print_status "Use '$0 help' for more commands"
        ;;
    *)
        print_error "Unknown command: $1"
        print_status "Use '$0 help' for available commands"
        exit 1
        ;;
esac
