#!/bin/bash
# ================================================================
# Crypto Trading App - Docker Deployment Script
# ================================================================
# This script helps deploy and manage the crypto trading application
# in Docker containers with proper setup and validation.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="crypto-trading-app"
DOCKER_IMAGE="crypto-trader:latest"
COMPOSE_FILE="docker-compose.yml"

# Function to print colored output
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

# Function to check if Docker is installed and running
check_docker() {
    print_status "Checking Docker installation..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    print_success "Docker is installed and running"
}

# Function to check if Docker Compose is available
check_docker_compose() {
    print_status "Checking Docker Compose..."
    
    if docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    elif docker-compose --version &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Docker Compose is available"
}

# Function to validate environment file
validate_env() {
    print_status "Validating environment configuration..."
    
    if [ ! -f .env ]; then
        print_warning ".env file not found. Creating from template..."
        cp .env.template .env 2>/dev/null || {
            print_error "No .env template found. Please create .env file with your API credentials."
            exit 1
        }
        print_warning "Please update .env file with your actual API credentials before proceeding."
        return 1
    fi
    
    # Check for required environment variables
    local missing_vars=()
    
    if ! grep -q "ALPACA_API_KEY=" .env || grep -q "ALPACA_API_KEY=$" .env; then
        missing_vars+=("ALPACA_API_KEY")
    fi
    
    if ! grep -q "BINANCE_API_KEY=" .env || grep -q "BINANCE_API_KEY=$" .env; then
        missing_vars+=("BINANCE_API_KEY")
    fi
    
    if [ ${#missing_vars[@]} -gt 0 ]; then
        print_error "Missing or empty environment variables: ${missing_vars[*]}"
        print_error "Please update .env file with your API credentials."
        return 1
    fi
    
    print_success "Environment configuration validated"
    return 0
}

# Function to create necessary directories
create_directories() {
    print_status "Creating data directories..."
    
    mkdir -p data/trading-logs
    mkdir -p data/historical-exports
    mkdir -p data/ml-results
    mkdir -p data/binance-exports
    mkdir -p backups
    
    print_success "Data directories created"
}

# Function to build Docker image
build_image() {
    print_status "Building Docker image..."
    
    docker build -t $DOCKER_IMAGE . || {
        print_error "Failed to build Docker image"
        exit 1
    }
    
    print_success "Docker image built successfully"
}

# Function to start the application
start_app() {
    print_status "Starting crypto trading application..."
    
    $COMPOSE_CMD up -d crypto-trader || {
        print_error "Failed to start application"
        exit 1
    }
    
    print_success "Application started successfully"
    print_status "Container status:"
    docker ps --filter "name=crypto-trading"
}

# Function to stop the application
stop_app() {
    print_status "Stopping crypto trading application..."
    
    $COMPOSE_CMD down
    
    print_success "Application stopped"
}

# Function to view logs
view_logs() {
    local service=${1:-crypto-trader}
    print_status "Viewing logs for $service..."
    $COMPOSE_CMD logs -f $service
}

# Function to show application status
show_status() {
    print_status "Application Status:"
    echo "===================="
    
    # Check if containers are running
    if docker ps --filter "name=crypto-trading-engine" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -q crypto-trading-engine; then
        print_success "Trading engine is running"
        docker ps --filter "name=crypto-trading-engine" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    else
        print_warning "Trading engine is not running"
    fi
    
    echo ""
    print_status "Recent logs:"
    $COMPOSE_CMD logs --tail=10 crypto-trader 2>/dev/null || echo "No logs available"
}

# Function to backup data
backup_data() {
    print_status "Creating data backup..."
    
    local backup_name="backup_$(date +%Y%m%d_%H%M%S)"
    
    $COMPOSE_CMD run --rm data-backup || {
        print_error "Backup failed"
        exit 1
    }
    
    print_success "Backup created successfully"
}

# Function to show help
show_help() {
    echo "Crypto Trading App - Docker Deployment Script"
    echo "=============================================="
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup      - Complete setup (build, create directories, validate env)"
    echo "  start      - Start the trading application"
    echo "  stop       - Stop the trading application"
    echo "  restart    - Restart the trading application"
    echo "  status     - Show application status"
    echo "  logs       - View application logs"
    echo "  build      - Build Docker image"
    echo "  backup     - Create data backup"
    echo "  clean      - Clean up containers and images"
    echo "  dev        - Start development environment"
    echo "  monitor    - Start log monitoring interface"
    echo "  help       - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup      # Initial setup"
    echo "  $0 start      # Start trading"
    echo "  $0 logs       # View logs"
    echo "  $0 status     # Check status"
}

# Function to start development environment
start_dev() {
    print_status "Starting development environment..."
    $COMPOSE_CMD --profile development up -d crypto-trader-dev
    print_success "Development environment started"
    print_status "Access with: docker exec -it crypto-trading-dev bash"
}

# Function to start monitoring
start_monitoring() {
    print_status "Starting log monitoring interface..."
    $COMPOSE_CMD --profile monitoring up -d log-viewer
    print_success "Log viewer started at http://localhost:9999"
}

# Function to clean up
clean_up() {
    print_status "Cleaning up Docker resources..."
    $COMPOSE_CMD down --volumes --remove-orphans
    docker image rm $DOCKER_IMAGE 2>/dev/null || true
    print_success "Cleanup completed"
}

# Main script logic
main() {
    case "${1:-help}" in
        setup)
            check_docker
            check_docker_compose
            create_directories
            build_image
            validate_env || {
                print_warning "Please update .env file and run 'start' command when ready"
                exit 0
            }
            print_success "Setup completed successfully!"
            print_status "Run '$0 start' to begin trading"
            ;;
        start)
            check_docker
            check_docker_compose
            validate_env || exit 1
            create_directories
            start_app
            ;;
        stop)
            stop_app
            ;;
        restart)
            stop_app
            sleep 2
            start_app
            ;;
        status)
            show_status
            ;;
        logs)
            view_logs
            ;;
        build)
            check_docker
            build_image
            ;;
        backup)
            backup_data
            ;;
        clean)
            clean_up
            ;;
        dev)
            check_docker
            check_docker_compose
            create_directories
            start_dev
            ;;
        monitor)
            start_monitoring
            ;;
        help)
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
