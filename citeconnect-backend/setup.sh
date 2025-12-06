#!/bin/bash

# CiteConnect Backend Setup Script
# This script sets up the entire backend infrastructure

set -e  # Exit on error

echo "================================================"
echo "CiteConnect Backend Setup"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if .env file exists
if [ ! -f .env ]; then
    print_warning ".env file not found. Creating from .env.example..."
    cp .env.example .env
    print_info ".env file created. Please update with your credentials."
    print_warning "Setup cannot continue without proper .env configuration."
    exit 1
fi

# Load environment variables
print_info "Loading environment variables..."
source .env

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

print_info "Docker and Docker Compose found."

# Create necessary directories
print_info "Creating directories..."
mkdir -p models
mkdir -p logs
mkdir -p mlflow
print_info "Directories created."

# Build Docker images
print_info "Building Docker images..."
docker-compose build

# Start services
print_info "Starting services..."
docker-compose up -d

# Wait for services to be healthy
print_info "Waiting for services to be healthy..."
sleep 10

# Check if API is running
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_info "API is healthy!"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        print_warning "Waiting for API to be ready... ($RETRY_COUNT/$MAX_RETRIES)"
        sleep 2
    fi
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    print_error "API failed to start within timeout period."
    print_info "Checking logs..."
    docker-compose logs api
    exit 1
fi

# Display service status
print_info "Service Status:"
docker-compose ps

echo ""
print_info "================================================"
print_info "Setup Complete!"
print_info "================================================"
echo ""
print_info "Services available at:"
print_info "  - API: http://localhost:8000"
print_info "  - API Docs: http://localhost:8000/docs"
print_info "  - MLflow: http://localhost:5000"
print_info "  - Flower (Celery): http://localhost:5555"
print_info "  - Redis: localhost:6379"
echo ""
print_info "Useful commands:"
print_info "  - View logs: docker-compose logs -f"
print_info "  - Stop services: docker-compose down"
print_info "  - Restart services: docker-compose restart"
print_info "  - Check health: curl http://localhost:8000/health"
echo ""
print_info "================================================"