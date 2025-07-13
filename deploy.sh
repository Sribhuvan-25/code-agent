#!/bin/bash

# Simple Production Deployment Script for Backspace AI Coding Agent
# This script will help you deploy your application step by step

set -e  # Exit on any error

echo "🚀 Starting Backspace AI Coding Agent Deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
check_docker() {
    print_status "Checking Docker installation..."
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running or not installed!"
        exit 1
    fi
    print_status "Docker is running ✓"
}

# Check if Docker Compose is available
check_docker_compose() {
    print_status "Checking Docker Compose..."
    if ! docker-compose --version > /dev/null 2>&1; then
        print_error "Docker Compose is not installed!"
        exit 1
    fi
    print_status "Docker Compose is available ✓"
}

# Check environment file
check_env_file() {
    print_status "Checking environment configuration..."
    if [ ! -f ".env.prod" ]; then
        print_warning "Production environment file (.env.prod) not found!"
        print_status "Creating from example..."
        if [ -f "env.prod.example" ]; then
            cp env.prod.example .env.prod
            print_warning "Please edit .env.prod with your actual API keys and tokens!"
            print_warning "Required: OPENAI_API_KEY or ANTHROPIC_API_KEY, GITHUB_TOKEN"
            read -p "Press Enter after you've configured .env.prod..."
        else
            print_error "No environment example file found!"
            exit 1
        fi
    fi
    print_status "Environment file found ✓"
}

# Build the application
build_app() {
    print_status "Building Backspace AI Coding Agent..."
    docker-compose -f docker-compose.prod.yml build
    print_status "Build completed ✓"
}

# Start the services
start_services() {
    print_status "Starting services..."
    docker-compose -f docker-compose.prod.yml up -d
    
    print_status "Waiting for services to be healthy..."
    sleep 30
    
    # Check if services are running
    if docker-compose -f docker-compose.prod.yml ps | grep -q "Up"; then
        print_status "Services are running ✓"
    else
        print_error "Some services failed to start!"
        docker-compose -f docker-compose.prod.yml logs
        exit 1
    fi
}

# Check service health
check_health() {
    print_status "Checking service health..."
    
    # Check main application
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_status "Main application is healthy ✓"
    else
        print_warning "Main application health check failed"
    fi
    
    # Check Jaeger
    if curl -f http://localhost:16686 > /dev/null 2>&1; then
        print_status "Jaeger telemetry is accessible ✓"
    else
        print_warning "Jaeger telemetry check failed"
    fi
    
    # Check Redis
    if docker-compose -f docker-compose.prod.yml exec redis redis-cli ping > /dev/null 2>&1; then
        print_status "Redis is healthy ✓"
    else
        print_warning "Redis health check failed"
    fi
}

# Show service information
show_info() {
    echo ""
    print_status "🎉 Deployment completed successfully!"
    echo ""
    echo "📊 Service Information:"
    echo "  • Main Application: http://localhost:8000"
    echo "  • Nginx Load Balancer: http://localhost:80"
    echo "  • Jaeger Telemetry: http://localhost:16686"
    echo "  • Redis: localhost:6379"
    echo ""
    echo "🔧 Useful Commands:"
    echo "  • View logs: docker-compose -f docker-compose.prod.yml logs -f"
    echo "  • Stop services: docker-compose -f docker-compose.prod.yml down"
    echo "  • Restart services: docker-compose -f docker-compose.prod.yml restart"
    echo "  • Check status: docker-compose -f docker-compose.prod.yml ps"
    echo ""
    print_status "Ready to use! 🚀"
}

# Main deployment function
deploy() {
    print_status "Starting deployment process..."
    
    check_docker
    check_docker_compose
    check_env_file
    build_app
    start_services
    check_health
    show_info
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        deploy
        ;;
    "stop")
        print_status "Stopping services..."
        docker-compose -f docker-compose.prod.yml down
        print_status "Services stopped ✓"
        ;;
    "restart")
        print_status "Restarting services..."
        docker-compose -f docker-compose.prod.yml restart
        print_status "Services restarted ✓"
        ;;
    "logs")
        print_status "Showing logs..."
        docker-compose -f docker-compose.prod.yml logs -f
        ;;
    "status")
        print_status "Service status:"
        docker-compose -f docker-compose.prod.yml ps
        ;;
    *)
        echo "Usage: $0 {deploy|stop|restart|logs|status}"
        echo "  deploy  - Deploy the application (default)"
        echo "  stop    - Stop all services"
        echo "  restart - Restart all services"
        echo "  logs    - Show service logs"
        echo "  status  - Show service status"
        exit 1
        ;;
esac 