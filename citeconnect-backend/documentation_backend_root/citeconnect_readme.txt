# CiteConnect Backend

AI-powered academic paper recommendation system with cold-start capabilities, multi-model comparison, and citation-based ground truth evaluation.

## 🎯 Project Status

**✅ Currently Working:**
- User registration & authentication (JWT)
- Profile creation with hierarchical interests
- State management (cold_start → expert journey)
- Database connectivity with Supabase
- ML model loading (MiniLM 384-dim)
- Health monitoring

**⏳ Ready After Data Population:**
- Paper recommendations
- Semantic search
- Interaction tracking
- Ground truth evaluation
- A/B testing

---

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Testing](#testing)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

---

## Overview

CiteConnect is a sophisticated recommendation system that:
- **Solves cold-start problem** using rich user profiles (research stage, goals, reading level)
- **Compares embedding models** (all-MiniLM-L6-v2 384-dim vs SPECTER2 768-dim)
- **Evaluates objectively** using citation networks as ground truth
- **Personalizes progressively** as users interact with the system
- **Detects bias** across user segments using slicing techniques

---

## Key Features

### For Users
- ✅ **Instant Recommendations**: Get relevant papers from day one without interaction history
- ✅ **Personalized Experience**: Recommendations improve as you interact
- ✅ **Multi-Level Discovery**: From foundational classics to cutting-edge research
- ✅ **Domain-Aware**: Supports healthcare, fintech, and quantum computing

### For ML/MLOps
- ✅ **A/B Testing**: Compare model performance with statistical significance
- ✅ **Experiment Tracking**: MLflow integration for model versioning
- ✅ **Bias Detection**: Continuous monitoring across user segments
- ✅ **Model Registry**: Automated validation before deployment
- ✅ **Comprehensive Evaluation**: Citation-based metrics + user engagement

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Application                   │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │  Bootstrap  │  │   Runtime    │  │   Background   │ │
│  │  Services   │  │   Services   │  │    Workers     │ │
│  └─────────────┘  └──────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                   ▼
  ┌──────────┐      ┌──────────┐      ┌──────────────┐
  │  Supabase│      │  Redis   │      │    MLflow    │
  │PostgreSQL│      │  Cache   │      │   Tracking   │
  │ +pgvector│      └──────────┘      └──────────────┘
  └──────────┘
```

### Tech Stack
- **Backend**: FastAPI, Python 3.11, AsyncPG
- **Database**: Supabase (PostgreSQL + pgvector)
- **Caching**: Redis
- **ML Models**: sentence-transformers (MiniLM, SPECTER2)
- **MLOps**: MLflow, Celery
- **Deployment**: Docker, Docker Compose

---

## Quick Start

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- Supabase account with project created
- 8GB RAM, 20GB disk space

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd citeconnect-backend
```

### 2. Configure Environment
```bash
cp .env.example .env
nano .env
```

**Required settings:**
```env
# From Supabase Dashboard → Settings → Database
DATABASE_URL=postgresql://postgres.xxxxx:[PASSWORD]@aws-x-xx.pooler.supabase.com:6543/postgres
SUPABASE_URL=https://xxxxx.supabase.co

# Generate random key
SECRET_KEY=$(openssl rand -hex 32)
```

**CRITICAL:** Use **port 6543** (transaction pooler) for Docker, not 5432!

### 3. Deploy
```bash
chmod +x setup.sh
./setup.sh
```

### 4. Verify
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "checks": {
    "database": "healthy",
    "models": {
      "all-MiniLM-L6-v2": "healthy"
    }
  }
}
```

### 5. Create First User
```bash
curl -X POST http://localhost:8000/api/v1/users/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "TestPass123!",
    "full_name": "Test User"
  }'
```

---

## Testing

### Run Automated Tests
```bash
# Make test script executable
chmod +x test_user_flow.sh

# Run complete user flow tests
./test_user_flow.sh
```

### Manual Testing

See [TESTING_GUIDE.md](TESTING_GUIDE.md) for comprehensive test scenarios.

Quick tests:
```bash
# Health check
curl http://localhost:8000/health

# Register user
curl -X POST http://localhost:8000/api/v1/users/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com","password":"Test123!","full_name":"Test"}'

# Create profile (use valid domain: healthcare, fintech, quantum_computing)
curl -X POST http://localhost:8000/api/v1/users/1/profile \
  -H "Content-Type: application/json" \
  -d '{
    "primary_domain": "healthcare",
    "reading_level": "advanced",
    "interests": ["ml", "ai", "cv"]
  }'
```

---

## API Documentation

### Base URL
```
http://localhost:8000/api/v1
```

### Authentication
All endpoints (except `/register` and `/login`) require JWT token:
```bash
# Get token from registration or login
export TOKEN="your-access-token"

# Use in requests
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/users/1/profile
```

### Key Endpoints

#### User Management
- `POST /users/register` - Register new user
- `POST /users/login` - Authenticate
- `POST /users/{id}/profile` - Create profile
- `GET /users/{id}/profile` - Get profile
- `PUT /users/{id}/profile` - Update profile
- `GET /users/{id}/state` - Get recommendation state
- `GET /users/{id}/interests` - Get interest hierarchy

#### Papers (Requires Data)
- `GET /papers/{id}` - Get paper details
- `POST /papers/search` - Search papers
- `GET /papers/trending` - Get trending papers
- `GET /papers/domain/{domain}` - Papers by domain

#### Recommendations (Requires Data)
- `POST /recommendations` - Get personalized recommendations
- `GET /recommendations/{user_id}/history` - Recommendation history

#### Interactions (Requires Data)
- `POST /interactions/{user_id}` - Track interaction
- `GET /interactions/{user_id}/statistics` - Get engagement metrics

**Full API documentation:** http://localhost:8000/docs

---

## Configuration

### Environment Variables

#### Critical Settings
```env
DATABASE_URL=postgresql://...  # Must use port 6543 (pooler)
SECRET_KEY=...  # Min 32 characters
```

#### Domain Configuration
Currently supports **3 domains only**:
- `healthcare` - Medical, clinical, health research
- `fintech` - Finance, trading, risk management  
- `quantum_computing` - Quantum algorithms, qubits, quantum ML

To add more domains, update the Supabase schema constraints.

#### Model Configuration
```env
EMBEDDING_MODEL_MINILM=sentence-transformers/all-MiniLM-L6-v2
DEFAULT_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

SPECTER2 model currently on hold due to configuration issues.

---

## Development

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (requires .env configuration)
uvicorn app.main:app --reload --port 8000
```

### Code Quality
```bash
# Format
black app/

# Lint
ruff check app/

# Type checking
mypy app/
```

### Database Migrations

Schema is managed in Supabase. For changes:
1. Update schema in Supabase SQL Editor
2. Update repository code to match
3. Test with validation scripts

---

## Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose logs api

# Common fixes:
# 1. Wrong DATABASE_URL format
echo $DATABASE_URL  # Should use port 6543

# 2. Database unreachable
ping aws-x-xx-xxxx.pooler.supabase.com

# 3. Check .env file exists
ls -la .env
```

### Database Connection Issues

```bash
# Test connection outside Docker
psql "$DATABASE_URL" -c "SELECT 1"

# Verify port (must be 6543 for Docker)
echo $DATABASE_URL | grep 6543

# Check Supabase project status
# Visit: https://status.supabase.com
```

### Models Not Loading

```bash
# Check disk space
df -h

# Check memory
free -h

# View model loading logs
docker-compose logs api | grep "Loading.*model"
```

### Common Error Solutions

| Error | Solution |
|-------|----------|
| `Network unreachable` | Check DATABASE_URL, verify Supabase is accessible |
| `Invalid API key` | Set `SUPABASE_KEY=` to empty (it's optional) |
| `Prepared statement already exists` | Already fixed with `statement_cache_size=0` |
| `password cannot be longer than 72 bytes` | Already fixed with `bcrypt==4.0.1` |
| `module has no attribute 'router'` | Verify all API files exist and are complete |

---

## Project Structure

```
citeconnect-backend/
├── app/
│   ├── main.py              # Application entry point
│   ├── config.py            # Configuration management
│   ├── api/v1/              # API endpoints
│   ├── db/repositories/     # Data access layer
│   ├── services/            # Business logic
│   ├── models/              # Pydantic models
│   └── utils/               # Utilities & logging
├── workers/                 # Celery background tasks
├── scripts/                 # Management scripts
├── docker-compose.yml       # Service orchestration
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

---

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes
4. Run tests: `pytest`
5. Commit: `git commit -m 'Add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Open Pull Request

---

## License

MIT License - See LICENSE file for details

---

## Acknowledgments

- SPECTER2 model by Allen Institute for AI
- sentence-transformers library
- Supabase for PostgreSQL + pgvector
- MLflow for experiment tracking

---

## Quick Reference

```bash
# Start services
docker-compose up -d

# Stop services  
docker-compose down

# View logs
docker-compose logs -f api

# Rebuild after code changes
docker-compose build api && docker-compose up -d

# Check health
curl http://localhost:8000/health

# Run tests
./test_user_flow.sh

# Validate data (after loading papers)
docker-compose exec api python scripts/validate_data.py

# Initialize ground truth (after loading papers)
docker-compose exec api python scripts/initialize_ground_truth.py
```# CiteConnect Backend

AI-powered academic paper recommendation system with cold-start capabilities, multi-model comparison, and citation-based ground truth evaluation.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)
- [API Documentation](#api-documentation)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## Overview

CiteConnect is a sophisticated recommendation system that:
- **Solves cold-start problem** using rich user profiles (research stage, goals, reading level)
- **Compares embedding models** (all-MiniLM-L6-v2 384-dim vs SPECTER2 768-dim)
- **Evaluates objectively** using citation networks as ground truth
- **Personalizes progressively** as users interact with the system
- **Detects bias** across user segments using slicing techniques

## Key Features

### For Users
- **Instant Quality Recommendations**: Get relevant papers from day one without interaction history
- **Personalized Experience**: Recommendations improve as you interact
- **Multi-Level Discovery**: From foundational classics to cutting-edge research
- **Domain-Aware**: Understands academic domains and research stages

### For ML/MLOps
- **A/B Testing**: Compare model performance with statistical significance
- **Experiment Tracking**: MLflow integration for model versioning
- **Bias Detection**: Continuous monitoring across user segments
- **Model Registry**: Automated validation before deployment
- **Comprehensive Evaluation**: Citation-based metrics + user engagement

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Application                   │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │  Bootstrap  │  │   Runtime    │  │   Background   │ │
│  │  Services   │  │   Services   │  │    Workers     │ │
│  └─────────────┘  └──────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                   ▼
  ┌──────────┐      ┌──────────┐      ┌──────────────┐
  │  Supabase│      │  Redis   │      │    MLflow    │
  │PostgreSQL│      │  Cache   │      │   Tracking   │
  │ +pgvector│      └──────────┘      └──────────────┘
  └──────────┘
```

### Tech Stack
- **Backend**: FastAPI, Python 3.11
- **Database**: Supabase (PostgreSQL + pgvector)
- **Caching**: Redis
- **ML Models**: sentence-transformers, SPECTER2
- **MLOps**: MLflow, Celery
- **Deployment**: Docker, Docker Compose

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- Supabase account (free tier sufficient for development)
- 4GB RAM minimum (8GB recommended)
- 10GB disk space for models

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/citeconnect-backend.git
cd citeconnect-backend
```

### 2. Configure Environment
```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your credentials
nano .env
```

**Required Configuration:**
```env
# Supabase (get from https://supabase.com/dashboard)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
DATABASE_URL=postgresql://user:pass@db.your-project.supabase.co:5432/postgres

# Security (generate strong key)
SECRET_KEY=your-secret-key-min-32-chars
```

### 3. Run Setup Script
```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Build Docker images
- Start all services
- Initialize databases
- Load ML models
- Perform health checks

### 4. Verify Installation
```bash
# Check health
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "version": "1.0.0",
  "checks": {
    "database": "healthy",
    "models": {
      "all-MiniLM-L6-v2": "healthy",
      "specter2": "healthy"
    }
  }
}
```

### 5. Access Services
- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **MLflow**: http://localhost:5000
- **Flower (Celery Monitoring)**: http://localhost:5555

## Configuration

### Environment Variables

#### Application Settings
```env
APP_NAME=CiteConnect
ENVIRONMENT=development  # development, staging, production
DEBUG=true
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

#### Database Settings
```env
DATABASE_URL=postgresql://...
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
DB_POOL_TIMEOUT=30
```

#### Model Settings
```env
EMBEDDING_MODEL_MINILM=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_MODEL_SPECTER=allenai/specter2
MODEL_CACHE_DIR=./models
EMBEDDING_BATCH_SIZE=32
```

#### Performance Thresholds
```env
COLD_START_PROFILE_ALIGNMENT_THRESHOLD=0.6
COLD_START_GROUND_TRUTH_THRESHOLD=0.5
MATURE_PRECISION_AT_10_THRESHOLD=0.3
BIAS_VARIANCE_THRESHOLD=0.2
```

#### Rate Limiting
```env
RATE_LIMIT_RECOMMENDATIONS_PER_HOUR=100
RATE_LIMIT_RECOMMENDATIONS_PER_MINUTE=20
RATE_LIMIT_PROFILE_UPDATE_PER_HOUR=10
```

## Development

### Local Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run database migrations (if applicable)
# alembic upgrade head

# Start development server
uvicorn app.main:app --reload --port 8000
```

### Code Style
```bash
# Format code
black app/

# Lint code
ruff check app/

# Type checking
mypy app/
```

### Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit

# Setup hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Testing

### Run Tests
```bash
# All tests
pytest

# With coverage
pytest --cov=app --cov-report=html

# Specific test file
pytest tests/unit/test_embedding_service.py

# Watch mode
pytest-watch
```

### Test Structure
```
tests/
├── unit/              # Unit tests for services
├── integration/       # Integration tests for APIs
├── evaluation/        # Model evaluation tests
└── fixtures/          # Test data
```

## Deployment

### Production Deployment

#### 1. Update Environment
```bash
# Set production values
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
```

#### 2. Build and Deploy
```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Deploy
docker-compose -f docker-compose.prod.yml up -d
```

#### 3. Initialize Database
```bash
# Run initialization scripts
docker-compose exec api python scripts/initialize_ground_truth.py
docker-compose exec api python scripts/seed_canonical_papers.py
```

### Scaling

#### Horizontal Scaling
```bash
# Scale API workers
docker-compose up -d --scale api=3

# Scale Celery workers
docker-compose up -d --scale worker=5
```

#### Resource Allocation
```yaml
# docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

## API Documentation

### Authentication
```bash
# Get access token
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password"}'
```

### Create User Profile
```bash
curl -X POST http://localhost:8000/api/v1/users/profile \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "research_stage": "phd",
    "primary_domain": "machine_learning",
    "interests": ["deep learning", "nlp", "computer vision"],
    "reading_level": "advanced"
  }'
```

### Get Recommendations
```bash
curl -X POST http://localhost:8000/api/v1/recommendations \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "count": 10,
    "model_preference": "all-MiniLM-L6-v2"
  }'
```

For complete API documentation, visit `/docs` endpoint when running.

## Monitoring

### Logs
```bash
# View all logs
docker-compose logs -f

# View specific service
docker-compose logs -f api

# Last 100 lines
docker-compose logs --tail=100 api
```

### Metrics

#### Application Metrics
- **Request Rate**: Requests per second
- **Latency**: P50, P95, P99 response times
- **Error Rate**: 4xx and 5xx responses
- **Cache Hit Rate**: Redis cache performance

#### ML Metrics
- **Model Latency**: Embedding generation time
- **Recommendation Quality**: Precision@K, Recall@K
- **User Engagement**: CTR, Save Rate
- **Bias Metrics**: Variance across user segments

#### System Metrics
- **CPU Usage**: Per service
- **Memory Usage**: Per service
- **Database Connections**: Active/idle
- **Queue Length**: Celery task backlog

### Health Monitoring
```bash
# Check service health
curl http://localhost:8000/health

# Check MLflow
curl http://localhost:5000/health

# Check Redis
redis-cli ping
```

## Troubleshooting

### Common Issues

#### Services Won't Start
```bash
# Check logs
docker-compose logs api

# Common causes:
# 1. Port conflicts - check if ports are in use
sudo lsof -i :8000

# 2. Database connection - verify Supabase credentials
docker-compose exec api python -c "from app.db.connection import db; import asyncio; asyncio.run(db.connect())"

# 3. Model loading - ensure enough disk space
df -h
```

#### Models Not Loading
```bash
# Check model cache directory
ls -lh models/

# Manually download models
docker-compose exec api python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Check available memory
free -h
```

#### Database Connection Issues
```bash
# Test connection
docker-compose exec api python scripts/test_db_connection.py

# Check Supabase status
# Visit: https://status.supabase.com

# Verify connection string
echo $DATABASE_URL
```

#### Slow Performance
```bash
# Check resource usage
docker stats

# Increase worker count
docker-compose up -d --scale worker=5

# Enable caching
# Verify Redis is running
docker-compose ps redis

# Check cache hit rate
redis-cli
> INFO stats
```

### Debug Mode

Enable detailed logging:
```env
DEBUG=true
LOG_LEVEL=DEBUG
```

Then restart services:
```bash
docker-compose restart api
docker-compose logs -f api
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests: `pytest`
5. Commit: `git commit -m 'Add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Open a Pull Request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Support

- **Issues**: https://github.com/yourusername/citeconnect-backend/issues
- **Email**: support@citeconnect.io
- **Documentation**: https://docs.citeconnect.io

## Acknowledgments

- SPECTER2 model by Allen Institute for AI
- sentence-transformers library
- Supabase for PostgreSQL + pgvector
- MLflow for experiment tracking