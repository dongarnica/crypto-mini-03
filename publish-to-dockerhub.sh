#!/bin/bash

# ================================================================
# Crypto Trading Engine - Docker Hub Build & Publish Script
# ================================================================
# This script builds and publishes the 3-class crypto trading engine
# to Docker Hub for easy deployment and distribution.
# 
# Author: Crypto Trading Strategy Engine
# Date: June 2, 2025
# ================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
DOCKER_HUB_USERNAME="${DOCKER_HUB_USERNAME:-}"
DOCKER_HUB_REPO="crypto-trading-3class"
IMAGE_NAME="crypto-trading-3class"
VERSION="1.0.0"
LATEST_TAG="latest"

# Auto-detect architecture
ARCH=$(uname -m)
case $ARCH in
    x86_64) PLATFORM="linux/amd64" ;;
    aarch64|arm64) PLATFORM="linux/arm64" ;;
    *) PLATFORM="linux/amd64" ;;
esac

print_header() {
    echo -e "${BLUE}"
    echo "================================================================"
    echo "  🚀 Crypto Trading Engine - Docker Hub Publisher"
    echo "  📦 3-Class Model Focus - Zero Execution Rate Fix"
    echo "================================================================"
    echo -e "${NC}"
}

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Function to prompt for Docker Hub username if not set
get_docker_hub_username() {
    if [ -z "$DOCKER_HUB_USERNAME" ]; then
        echo -e "${YELLOW}Enter your Docker Hub username:${NC}"
        read -r DOCKER_HUB_USERNAME
        
        if [ -z "$DOCKER_HUB_USERNAME" ]; then
            print_error "Docker Hub username is required"
            exit 1
        fi
    fi
    
    print_info "Using Docker Hub username: $DOCKER_HUB_USERNAME"
}

# Function to check dependencies
check_dependencies() {
    print_step "Checking dependencies..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
    
    print_success "All dependencies available"
}

# Function to verify 3-class system configuration
verify_3class_config() {
    print_step "Verifying 3-class system configuration..."
    
    # Check if verification script exists and run it
    if [ -f "verify_3class_system.py" ]; then
        if python3 verify_3class_system.py >/dev/null 2>&1; then
            print_success "3-class system verification passed"
        else
            print_error "3-class system verification failed"
            exit 1
        fi
    else
        print_warning "3-class verification script not found, proceeding..."
    fi
    
    # Check if 3-class models exist
    if [ -d "ml_results" ]; then
        model_count=$(find ml_results -name "*3class_enhanced.h5" | wc -l)
        binary_count=$(find ml_results -name "*binary*" | wc -l)
        
        print_info "Found $model_count 3-class models"
        print_info "Found $binary_count binary models (should be 0)"
        
        if [ "$model_count" -eq 0 ]; then
            print_error "No 3-class models found! Please train models first."
            exit 1
        fi
        
        if [ "$binary_count" -gt 0 ]; then
            print_warning "Binary models still exist. Consider running cleanup_binary_models.py"
        fi
    fi
}

# Function to build Docker image
build_image() {
    print_step "Building Docker image..."
    
    local full_image_name="$DOCKER_HUB_USERNAME/$DOCKER_HUB_REPO"
    
    print_info "Building for platform: $PLATFORM"
    print_info "Image name: $full_image_name:$VERSION"
    
    # Build the Docker image
    docker build \
        --platform "$PLATFORM" \
        --tag "$full_image_name:$VERSION" \
        --tag "$full_image_name:$LATEST_TAG" \
        --label "version=$VERSION" \
        --label "build-date=$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --label "description=3-Class Crypto Trading Engine - Zero Execution Rate Fix" \
        --label "architecture=$ARCH" \
        .
    
    if [ $? -eq 0 ]; then
        print_success "Docker image built successfully"
    else
        print_error "Docker image build failed"
        exit 1
    fi
}

# Function to test Docker image
test_image() {
    print_step "Testing Docker image..."
    
    local full_image_name="$DOCKER_HUB_USERNAME/$DOCKER_HUB_REPO:$VERSION"
    
    # Test if image can start and basic functionality works
    print_info "Running basic container test..."
    
    # Create a test container that exits after verification
    if docker run --rm "$full_image_name" python3 -c "
import sys
sys.path.insert(0, '/app')
from trading.models import TradingConfig
config = TradingConfig()
print(f'✅ Config loaded: binary={config.use_binary_classification}')
print('✅ Container test passed')
" 2>/dev/null; then
        print_success "Docker image test passed"
    else
        print_error "Docker image test failed"
        exit 1
    fi
}

# Function to login to Docker Hub
docker_hub_login() {
    print_step "Logging into Docker Hub..."
    
    # Check if already logged in
    if docker info | grep -q "Username: $DOCKER_HUB_USERNAME"; then
        print_info "Already logged into Docker Hub as $DOCKER_HUB_USERNAME"
        return
    fi
    
    print_info "Please enter your Docker Hub password:"
    docker login --username "$DOCKER_HUB_USERNAME"
    
    if [ $? -eq 0 ]; then
        print_success "Docker Hub login successful"
    else
        print_error "Docker Hub login failed"
        exit 1
    fi
}

# Function to push to Docker Hub
push_to_docker_hub() {
    print_step "Publishing to Docker Hub..."
    
    local full_image_name="$DOCKER_HUB_USERNAME/$DOCKER_HUB_REPO"
    
    # Push versioned image
    print_info "Pushing $full_image_name:$VERSION..."
    docker push "$full_image_name:$VERSION"
    
    # Push latest tag
    print_info "Pushing $full_image_name:$LATEST_TAG..."
    docker push "$full_image_name:$LATEST_TAG"
    
    if [ $? -eq 0 ]; then
        print_success "Successfully published to Docker Hub!"
        echo ""
        print_info "🎯 Your image is now available at:"
        echo -e "${PURPLE}   docker pull $full_image_name:$VERSION${NC}"
        echo -e "${PURPLE}   docker pull $full_image_name:$LATEST_TAG${NC}"
        echo ""
        print_info "📊 To run the 3-class trading engine:"
        echo -e "${PURPLE}   docker run -d --name crypto-trader $full_image_name:$LATEST_TAG${NC}"
    else
        print_error "Failed to publish to Docker Hub"
        exit 1
    fi
}

# Function to clean up local images (optional)
cleanup_local() {
    print_step "Cleaning up local images..."
    
    local full_image_name="$DOCKER_HUB_USERNAME/$DOCKER_HUB_REPO"
    
    echo -e "${YELLOW}Do you want to remove local images to save space? (y/N):${NC}"
    read -r cleanup_response
    
    if [[ "$cleanup_response" =~ ^[Yy]$ ]]; then
        docker rmi "$full_image_name:$VERSION" "$full_image_name:$LATEST_TAG" 2>/dev/null || true
        print_info "Local images removed"
    else
        print_info "Local images retained"
    fi
}

# Function to show usage instructions
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -u, --username USERNAME   Docker Hub username"
    echo "  -v, --version VERSION     Image version (default: $VERSION)"
    echo "  --no-test                Skip image testing"
    echo "  --no-cleanup             Skip cleanup prompt"
    echo "  -h, --help               Show this help"
    echo ""
    echo "Environment Variables:"
    echo "  DOCKER_HUB_USERNAME      Docker Hub username"
    echo ""
    echo "Examples:"
    echo "  $0 --username myusername"
    echo "  DOCKER_HUB_USERNAME=myuser $0"
}

# Main execution function
main() {
    print_header
    
    # Parse command line arguments
    SKIP_TEST=false
    SKIP_CLEANUP=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -u|--username)
                DOCKER_HUB_USERNAME="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            --no-test)
                SKIP_TEST=true
                shift
                ;;
            --no-cleanup)
                SKIP_CLEANUP=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Execute build and publish pipeline
    get_docker_hub_username
    check_dependencies
    verify_3class_config
    build_image
    
    if [ "$SKIP_TEST" = false ]; then
        test_image
    fi
    
    docker_hub_login
    push_to_docker_hub
    
    if [ "$SKIP_CLEANUP" = false ]; then
        cleanup_local
    fi
    
    echo ""
    print_success "🎉 Docker Hub publication complete!"
    print_info "🚀 Your 3-class crypto trading engine is now publicly available"
}

# Execute main function with all arguments
main "$@"
