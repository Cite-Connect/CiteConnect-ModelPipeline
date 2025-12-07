#!/bin/bash

# Quick script to run hyperparameter tuning
# This script will:
# 1. Check Docker setup
# 2. Build/start services if needed
# 3. Run hyperparameter tuning
# 4. Display results

set -e

echo "================================================"
echo "CiteConnect Hyperparameter Tuning"
echo "================================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${RED}ERROR: .env file not found!${NC}"
    echo "Please create .env file with required environment variables."
    echo "See HYPERPARAMETER_TUNING_SETUP.md for details."
    exit 1
fi

echo -e "${GREEN}✓${NC} .env file found"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}ERROR: Docker not found!${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}ERROR: Docker Compose not found!${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Docker and Docker Compose found"

# Check if services are running
if ! docker-compose ps | grep -q "Up"; then
    echo -e "${YELLOW}Services not running. Starting services...${NC}"
    echo ""
    
    # Build if needed
    echo "Building Docker images..."
    docker-compose build
    
    # Start services
    echo "Starting services..."
    docker-compose up -d
    
    # Wait for services to be ready
    echo "Waiting for services to be ready..."
    sleep 10
    
    # Check API health
    MAX_RETRIES=30
    RETRY_COUNT=0
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if curl -f http://localhost:8000/health > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC} API is healthy!"
            break
        else
            RETRY_COUNT=$((RETRY_COUNT + 1))
            echo "Waiting for API... ($RETRY_COUNT/$MAX_RETRIES)"
            sleep 2
        fi
    done
    
    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        echo -e "${RED}ERROR: API failed to start${NC}"
        docker-compose logs api | tail -20
        exit 1
    fi
else
    echo -e "${GREEN}✓${NC} Services are running"
fi

echo ""
echo "================================================"
echo "Running Hyperparameter Tuning"
echo "================================================"
echo ""

# Run hyperparameter tuning
echo "Executing hyperparameter tuning script..."
docker-compose exec -T api python scripts/hyperparameter_tuning_cold_start.py

echo ""
echo "================================================"
echo "Results"
echo "================================================"
echo ""

# Check if results file exists
if [ -f "bias_config/best_hyperparameters_cold_start.json" ]; then
    echo -e "${GREEN}✓${NC} Results saved to: bias_config/best_hyperparameters_cold_start.json"
    echo ""
    echo "Best configuration summary:"
    python3 -c "
import json
try:
    with open('bias_config/best_hyperparameters_cold_start.json', 'r') as f:
        data = json.load(f)
    best = data.get('best_config', {})
    print(f\"  Weights: {best.get('weights', {})}\")
    print(f\"  Metrics: {best.get('metrics', {})}\")
except Exception as e:
    print(f\"  Error reading results: {e}\")
" 2>/dev/null || echo "  (Install python3 to view summary)"
else
    echo -e "${YELLOW}⚠${NC}  Results file not found. Check logs above for errors."
fi

echo ""
echo "================================================"
echo "Next Steps"
echo "================================================"
echo ""
echo "1. View detailed results:"
echo "   cat bias_config/best_hyperparameters_cold_start.json"
echo ""
echo "2. View MLflow experiments:"
echo "   http://localhost:5001"
echo ""
echo "3. Run sensitivity analysis (optional):"
echo "   docker-compose exec api python scripts/sensitivity_cold_start_weights.py"
echo ""
echo "================================================"

