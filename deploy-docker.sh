#!/bin/bash

# ================================================================
# Crypto Trading Engine - Deployment Script
# ================================================================
# This script helps deploy the crypto trading engine using Docker
# Author: Crypto Trading Team
# Date: June 2, 2025

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOCKER_IMAGE="codespacesdev/crypto-trading-engine"
CONTAINER_NAME="crypto-trading-engine"
COMPOSE_FILE="docker-compose.deploy.yml"

# Functions
print_header() {
    echo -e "${BLUE}"
    echo "================================================================"
    echo "  Crypto Trading Engine - Deployment Script"
    echo "================================================================"
    echo -e "${NC}"
}

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    print_step "Checking dependencies..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    
    echo "✅ All dependencies are available"
}

check_environment() {
    print_step "Checking environment configuration..."
    
    if [ ! -f ".env" ]; then
        print_warning ".env file not found. Creating from template..."
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_warning "Please edit .env file with your actual configuration before continuing."
            print_warning "At minimum, set your ALPACA_API_KEY and ALPACA_SECRET_KEY"
            echo ""
            echo "Edit .env file now? (y/n)"
            read -r response
            if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
                ${EDITOR:-nano} .env
            else
                print_warning "Remember to configure .env before running the application!"
            fi
        else
            print_error ".env.example file not found. Cannot create .env file."
            exit 1
        fi
    else
        echo "✅ .env file found"
        
        # Check for required variables
        if ! grep -q "ALPACA_API_KEY=" .env || grep -q "your_api_key_here" .env; then
            print_warning "ALPACA_API_KEY not configured in .env file"
        fi
        
        if ! grep -q "ALPACA_SECRET_KEY=" .env || grep -q "your_secret_key_here" .env; then
            print_warning "ALPACA_SECRET_KEY not configured in .env file"
        fi
    fi
}

create_directories() {
    print_step "Creating required directories..."
    
    mkdir -p logs
    mkdir -p ml_results
    mkdir -p historical_exports
    mkdir -p config
    
    # Set proper permissions
    chmod 755 logs ml_results historical_exports config
    
    echo "✅ Directories created"
}

pull_image() {
    print_step "Pulling latest Docker image..."
    
    if [ "$1" = "--force" ] || [ "$1" = "-f" ]; then
        docker pull ${DOCKER_IMAGE}:latest
    else
        echo "Pull latest image? (y/n)"
        read -r response
        if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            docker pull ${DOCKER_IMAGE}:latest
        fi
    fi
    
    echo "✅ Image ready"
}

deploy_application() {
    print_step "Deploying application..."
    
    # Stop existing container if running
    if docker ps -q -f name=${CONTAINER_NAME} | grep -q .; then
        print_step "Stopping existing container..."
        docker-compose -f ${COMPOSE_FILE} down
    fi
    
    # Start the application
    print_step "Starting crypto trading engine..."
    docker-compose -f ${COMPOSE_FILE} up -d
    
    echo "✅ Application deployed"
}

check_health() {
    print_step "Checking application health..."
    
    echo "Waiting for application to start..."
    sleep 10
    
    # Check container status
    if docker ps -q -f name=${CONTAINER_NAME} | grep -q .; then
        echo "✅ Container is running"
        
        # Check logs for startup success
        echo ""
        echo "Recent logs:"
        docker-compose -f ${COMPOSE_FILE} logs --tail=10 crypto-trader
        
        echo ""
        echo "✅ Application appears to be healthy"
        echo "Monitor logs with: docker-compose -f ${COMPOSE_FILE} logs -f crypto-trader"
    else
        print_error "Container is not running"
        echo "Check logs with: docker-compose -f ${COMPOSE_FILE} logs crypto-trader"
        exit 1
    fi
}

show_status() {
    print_step "Application status:"
    
    echo ""
    echo "Container Status:"
    docker-compose -f ${COMPOSE_FILE} ps
    
    echo ""
    echo "Recent Logs:"
    docker-compose -f ${COMPOSE_FILE} logs --tail=20 crypto-trader
    
    echo ""
    echo "Useful Commands:"
    echo "  View logs:     docker-compose -f ${COMPOSE_FILE} logs -f crypto-trader"
    echo "  Stop app:      docker-compose -f ${COMPOSE_FILE} down"
    echo "  Restart app:   docker-compose -f ${COMPOSE_FILE} restart"
    echo "  Update app:    ./deploy.sh --update"
    echo ""
}

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help, -h      Show this help message"
    echo "  --status, -s    Show application status"
    echo "  --update, -u    Update to latest image and restart"
    echo "  --force, -f     Force pull latest image without prompting"
    echo "  --stop          Stop the application"
    echo "  --logs          Show application logs"
    echo ""
    echo "Examples:"
    echo "  $0              Deploy with default settings"
    echo "  $0 --update     Update to latest version"
    echo "  $0 --status     Check current status"
    echo ""
}

# Main execution
main() {
    case "$1" in
        --help|-h)
            show_help
            exit 0
            ;;
        --status|-s)
            print_header
            show_status
            exit 0
            ;;
        --stop)
            print_header
            print_step "Stopping application..."
            docker-compose -f ${COMPOSE_FILE} down
            echo "✅ Application stopped"
            exit 0
            ;;
        --logs)
            docker-compose -f ${COMPOSE_FILE} logs -f crypto-trader
            exit 0
            ;;
        --update|-u)
            print_header
            check_dependencies
            pull_image --force
            deploy_application
            check_health
            show_status
            exit 0
            ;;
        --force|-f)
            FORCE_PULL=true
            ;;
        "")
            # Default deployment
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
    
    # Default deployment flow
    print_header
    check_dependencies
    check_environment
    create_directories
    pull_image $1
    deploy_application
    check_health
    show_status
}

# Run main function with all arguments
main "$@"
