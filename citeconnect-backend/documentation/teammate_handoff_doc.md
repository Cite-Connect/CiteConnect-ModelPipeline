# CiteConnect Backend - Teammate Handoff Documentation

**Project:** CiteConnect Research Paper Recommendation System  
**Course:** IE7305 - Northeastern University  
**Team:** Dennis Jose, Abhinav Aditya, Anusha Srinivasan, Dhiksha Mathanagopal, Sahil Mohanty  
**Presentation:** December 2025 @ Google HQ  
**Document Date:** November 12, 2025  
**Status:** Phase 8 Complete - Auth & User Management Working

---

## Table of Contents

1. [Quick Start Guide](#1-quick-start-guide)
2. [What's Been Built](#2-whats-been-built)
3. [Project Structure](#3-project-structure)
4. [Database Setup](#4-database-setup)
5. [Testing Guide](#5-testing-guide)
6. [API Documentation](#6-api-documentation)
7. [Common Issues & Solutions](#7-common-issues--solutions)
8. [Next Steps](#8-next-steps)
9. [Important Notes](#9-important-notes)

---

## 1. Quick Start Guide

### Prerequisites
- Docker Desktop installed and running
- Python 3.11 installed
- Terminal/Command line access

### Setup Steps (15 minutes)

```bash
# 1. Clone/Navigate to project
cd ~/Documents/GitHub/ModelPipeline

# 2. Activate virtual environment
source venv/bin/activate

# 3. Verify Python version
python --version
# Should show: Python 3.11.x

# 4. Install dependencies (if not already done)
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 5. Start all databases
docker-compose up -d

# Wait for databases to start (30 seconds)
sleep 30

# 6. Check database status
docker-compose ps
# All should show "Up (healthy)"

# 7. Navigate to backend
cd citeconnect-backend

# 8. Run database migrations (creates PostgreSQL tables)
alembic upgrade head

# 9. Initialize Neo4j schema
./scripts/init_neo4j.sh

# 10. Seed sample data
./scripts/seed_neo4j.sh
python scripts/seed_users.py

# 11. Verify setup
python verify_setup.py
python test_db_connectivity.py

# 12. Start FastAPI application
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Success indicators:**
- All databases show "healthy" ✓
- Alembic creates 13 tables ✓
- Neo4j has 5 sample papers ✓
- 4 test users created ✓
- FastAPI starts without errors ✓
- http://localhost:8000/docs loads ✓

---

## 2. What's Been Built

### 2.1 Infrastructure (100% Complete)

**Core Modules:**
- ✅ `app/core/exceptions.py` - Custom exception hierarchy
- ✅ `app/core/logging.py` - Structured JSON logging
- ✅ `app/core/config.py` - Pydantic settings (loads from .env)
- ✅ `app/core/security.py` - JWT tokens & password hashing

**Database Clients:**
- ✅ `app/db/postgres.py` - PostgreSQL with asyncpg (connection pooling, queries)
- ✅ `app/db/redis_client.py` - Redis caching (get/set/delete operations)
- ✅ `app/db/weaviate_client.py` - Vector database (semantic search)
- ✅ `app/db/neo4j_client.py` - Graph database (citation networks)

### 2.2 Data Models (100% Complete)

All Pydantic models for data validation:
- ✅ `app/models/user.py` - User, UserInterest, UserDomain, UserProfileEmbedding
- ✅ `app/models/paper.py` - Paper, PaperMetadata, PaperWithScore
- ✅ `app/models/cluster.py` - Cluster, ClusterPaper
- ✅ `app/models/interaction.py` - Interaction, InteractionContext
- ✅ `app/models/graph.py` - GraphNode, GraphEdge, CitationNetwork

### 2.3 API Schemas (100% Complete)

Request/response schemas for API:
- ✅ `app/schemas/auth.py` - LoginRequest, RegisterRequest, TokenResponse
- ✅ `app/schemas/user.py` - UserResponse, UserUpdateRequest
- ✅ `app/schemas/paper.py` - PaperResponse, PaperDetailResponse
- ✅ `app/schemas/search.py` - SearchRequest, SearchResponse
- ✅ `app/schemas/cluster.py` - ClusterResponse

### 2.4 API Endpoints (30% Complete)

**Working:**
- ✅ POST `/api/v1/auth/register` - User registration
- ✅ POST `/api/v1/auth/login` - User login
- ✅ POST `/api/v1/auth/refresh` - Token refresh
- ✅ GET `/api/v1/users/me` - Get user profile (protected)
- ✅ PUT `/api/v1/users/me` - Update user profile (protected)
- ✅ GET `/api/v1/health` - Health check

**Not Yet Implemented:**
- ❌ Paper endpoints (GET /papers/{id}, POST /papers/{id}/save, etc.)
- ❌ Search endpoint (GET /search)
- ❌ Graph endpoint (GET /papers/{id}/graph)
- ❌ Cluster endpoints
- ❌ Interaction tracking

### 2.5 Services (20% Complete)

**Working:**
- ✅ `app/services/auth_service.py` - Registration, login, token refresh
- ✅ `app/services/user_service.py` - Get/update user profile
- ✅ `app/api/deps.py` - Authentication dependencies

**Not Yet Implemented:**
- ❌ `app/services/embedding_service.py` - SPECTER embedding generation
- ❌ `app/services/search_service.py` - Semantic search
- ❌ `app/services/graph_service.py` - Citation graph generation
- ❌ `app/services/clustering_service.py` - K-means clustering
- ❌ `app/services/recommendation_service.py` - Starter kit generation
- ❌ All other services

### 2.6 Databases (100% Setup)

**PostgreSQL:**
- ✅ 13 tables created via Alembic migration
- ✅ All indexes and constraints in place
- ✅ 4 test users seeded
- ✅ Viewable in DBeaver

**Neo4j:**
- ✅ Constraints and indexes created
- ✅ 5 sample papers with citation relationships
- ✅ Viewable in Neo4j Browser (http://localhost:7474)

**Weaviate:**
- ✅ Paper schema created
- ✅ Ready for vector storage
- ❌ No papers seeded yet

**Redis:**
- ✅ Connected and working
- ✅ Cache operations functional

---

## 3. Project Structure

```
ModelPipeline/                              # Root directory
├── .env                                    # Environment configuration
├── requirements.txt                        # Production dependencies
├── requirements-dev.txt                    # Dev dependencies
├── docker-compose.yml                      # Database services
├── venv/                                   # Virtual environment (Python 3.11)
│
└── citeconnect-backend/                    # Backend code
    ├── alembic/                            # Database migrations
    │   ├── versions/
    │   │   └── 001_initial_schema.py       # PostgreSQL tables
    │   ├── env.py                          # Alembic config
    │   └── alembic.ini                     # Alembic settings
    │
    ├── app/
    │   ├── main.py                         # ✅ FastAPI app with routes
    │   │
    │   ├── api/
    │   │   ├── deps.py                     # ✅ Auth dependencies
    │   │   └── v1/
    │   │       ├── auth.py                 # ✅ Auth endpoints
    │   │       └── users.py                # ✅ User endpoints
    │   │
    │   ├── core/                           # ✅ All complete
    │   │   ├── exceptions.py
    │   │   ├── logging.py
    │   │   ├── config.py
    │   │   └── security.py
    │   │
    │   ├── db/                             # ✅ All complete
    │   │   ├── postgres.py
    │   │   ├── redis_client.py
    │   │   ├── weaviate_client.py
    │   │   └── neo4j_client.py
    │   │
    │   ├── models/                         # ✅ All complete
    │   │   ├── user.py
    │   │   ├── paper.py
    │   │   ├── cluster.py
    │   │   ├── interaction.py
    │   │   └── graph.py
    │   │
    │   ├── schemas/                        # ✅ All complete
    │   │   ├── auth.py
    │   │   ├── user.py
    │   │   ├── paper.py
    │   │   ├── search.py
    │   │   └── cluster.py
    │   │
    │   ├── services/                       # ⚠️ Partial
    │   │   ├── auth_service.py             # ✅ Working
    │   │   ├── user_service.py             # ✅ Working
    │   │   └── [others not implemented]
    │   │
    │   ├── utils/                          # ❌ Empty
    │   ├── tasks/                          # ❌ Empty
    │   └── middleware/                     # ❌ Empty
    │
    ├── scripts/
    │   ├── init_neo4j.sh                   # ✅ Initialize Neo4j schema
    │   ├── seed_neo4j.sh                   # ✅ Seed Neo4j with papers
    │   └── seed_users.py                   # ✅ Seed PostgreSQL with users
    │
    ├── verify_setup.py                     # ✅ Setup verification
    └── test_db_connectivity.py             # ✅ Database connectivity test
```

---

## 4. Database Setup

### 4.1 PostgreSQL Tables

**13 tables created:**

| Table | Purpose | Sample Data |
|-------|---------|-------------|
| users | User accounts | 4 users |
| user_domains | User's research domain | 4 domains |
| user_interests | Research keywords | ~12 interests |
| user_profile_embeddings | User profile vectors | Empty (will be generated) |
| papers | Paper metadata | Empty (ready for ingestion) |
| user_interactions | Interaction tracking | Empty |
| user_saved_papers | Saved papers | Empty |
| user_liked_papers | Liked papers | Empty |
| paper_clusters | Thematic clusters | Empty |
| cluster_papers | Papers in clusters | Empty |
| rate_limits | API rate limiting | Empty |
| system_metrics | System monitoring | Empty |
| alembic_version | Migration tracking | 1 record |

**Connection Info:**
- Host: 127.0.0.1 (or localhost)
- Port: 5432
- Database: citeconnect
- User: citeconnect
- Password: password

### 4.2 Neo4j Graph

**Sample data loaded:**
- 5 Paper nodes (AlphaFold, RoseTTAFold, Drug Discovery, Genomics, Clinical AI)
- 4 CITES relationships
- 4 CITED_BY relationships

**Access:**
- Browser: http://localhost:7474
- User: neo4j
- Password: password

**Test query:**
```cypher
MATCH (p:Paper) RETURN p LIMIT 25
```

### 4.3 Weaviate

**Schema created:**
- Collection: Paper (768-dimensional vectors)
- Properties: paper_id, title, abstract, domain, year, etc.

**Access:**
- HTTP: http://localhost:8080

**Status:** Schema ready, no papers inserted yet

### 4.4 Redis

**Status:** Connected and working
**Port:** 6379

---

## 5. Testing Guide

### 5.1 Verify Environment Setup

```bash
cd ~/Documents/GitHub/ModelPipeline
source venv/bin/activate
cd citeconnect-backend

# Test 1: Verify imports
python verify_setup.py
# Expected: All tests pass

# Test 2: Verify databases
python test_db_connectivity.py
# Expected: 4/4 databases connected

# Test 3: Check database data
python << 'EOF'
import asyncio
from app.db.postgres import execute_query

async def check_data():
    # Count users
    users = await execute_query("SELECT COUNT(*) FROM users", fetch_one=True)
    print(f"Users in database: {users['count']}")
    
    # Show users
    user_list = await execute_query(
        "SELECT user_id, email, name FROM users", 
        fetch_all=True
    )
    for u in user_list:
        print(f"  - {u['email']}: {u['name']}")

asyncio.run(check_data())
EOF
```

---

### 5.2 Start the Application

```bash
# Terminal 1: Start FastAPI
cd ~/Documents/GitHub/ModelPipeline/citeconnect-backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# You should see:
# INFO: CiteConnect Backend Started Successfully
# INFO: Registered routes: /auth/register, /auth/login, /auth/refresh, /users/me
```

**Check these URLs:**
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/api/v1/health
- Neo4j Browser: http://localhost:7474

---

### 5.3 Test User Registration

**Method 1: Using Swagger UI (http://localhost:8000/docs)**

1. Click on `POST /api/v1/auth/register`
2. Click "Try it out"
3. Use this request body:
```json
{
  "email": "teammate@example.com",
  "password": "SecurePass123!",
  "name": "Teammate Name",
  "domain": "healthcare",
  "interests": ["machine learning", "NLP", "clinical AI"]
}
```
4. Click "Execute"
5. Should return 201 with user_id and access_token

**Method 2: Using curl**

```bash
curl -X POST "http://localhost:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "teammate@example.com",
    "password": "SecurePass123!",
    "name": "Teammate Name",
    "domain": "healthcare",
    "interests": ["machine learning", "NLP", "clinical AI"]
  }'
```

---

### 5.4 Test User Login

```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "sarah.chen@example.com",
    "password": "Password123!"
  }'
```

**Save the access_token from the response!**

---

### 5.5 Test Protected Endpoint

```bash
# Replace YOUR_TOKEN_HERE with actual token from login
curl -X GET "http://localhost:8000/api/v1/users/me" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

**Should return user profile with interests**

---

### 5.6 Verify Data in Database

**Using psql:**
```bash
# Connect to database
docker exec -it citeconnect-postgres psql -U citeconnect -d citeconnect

# In psql:
SELECT * FROM users;
SELECT * FROM user_domains;
SELECT * FROM user_interests;

\q
```

**Using DBeaver or other GUI:**
- Host: localhost (or 127.0.0.1)
- Port: 5432
- Database: citeconnect
- User: citeconnect
- Password: password

---

## 6. API Documentation

### 6.1 Authentication Endpoints

#### POST /api/v1/auth/register

**Purpose:** Create new user account

**Request:**
```json
{
  "email": "user@example.com",
  "password": "Password123!",
  "name": "Full Name",
  "domain": "healthcare",
  "interests": ["keyword1", "keyword2"],
  "google_scholar_url": null
}
```

**Response (201):**
```json
{
  "user_id": 5,
  "email": "user@example.com",
  "name": "Full Name",
  "domain": "healthcare",
  "access_token": "eyJhbGci...",
  "refresh_token": "eyJhbGci...",
  "token_type": "bearer",
  "expires_in": 86400,
  "starter_kit_status": "processing"
}
```

**What it does:**
1. Validates email format and password strength
2. Checks if email already exists
3. Hashes password with bcrypt
4. INSERT INTO users, user_domains, user_interests
5. Generates JWT tokens
6. Returns tokens for immediate login

---

#### POST /api/v1/auth/login

**Purpose:** Authenticate existing user

**Request:**
```json
{
  "email": "user@example.com",
  "password": "Password123!"
}
```

**Response (200):**
```json
{
  "user_id": 1,
  "email": "user@example.com",
  "name": "Full Name",
  "domain": "healthcare",
  "access_token": "eyJhbGci...",
  "refresh_token": "eyJhbGci...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

**What it does:**
1. Looks up user by email
2. Verifies password hash
3. Checks account is active
4. Generates new JWT tokens
5. Returns tokens

---

### 6.2 User Profile Endpoints

#### GET /api/v1/users/me

**Purpose:** Get current user's profile

**Headers Required:**
```
Authorization: Bearer {access_token}
```

**Response (200):**
```json
{
  "user_id": 1,
  "email": "user@example.com",
  "name": "Full Name",
  "domain": "healthcare",
  "interests": [
    {"keyword": "NLP", "source": "manual", "weight": 1.0},
    {"keyword": "clinical trials", "source": "manual", "weight": 1.0}
  ],
  "google_scholar_url": null,
  "created_at": "2025-11-12T01:30:00Z"
}
```

---

#### PUT /api/v1/users/me

**Purpose:** Update user profile

**Headers Required:**
```
Authorization: Bearer {access_token}
```

**Request:**
```json
{
  "name": "Updated Name",
  "interests": ["new keyword 1", "new keyword 2"]
}
```

**Response (200):**
```json
{
  "user_id": 1,
  "message": "Profile updated successfully",
  "regenerate_clusters": true
}
```

---

## 7. Common Issues & Solutions

### Issue 1: "role 'citeconnect' does not exist"

**Problem:** Connecting to wrong PostgreSQL instance

**Solution:**
```bash
# Stop local Postgres.app if running
killall postgres

# Check what's on port 5432
lsof -i :5432
# Should only show Docker

# Restart PostgreSQL container
docker-compose restart postgres
```

---

### Issue 2: Alembic can't find .env

**Problem:** Path issues with .env file

**Solution:**
```bash
# Verify .env exists in ModelPipeline root
ls -la ~/Documents/GitHub/ModelPipeline/.env

# Check DATABASE_URL is correct
cat ~/Documents/GitHub/ModelPipeline/.env | grep DATABASE_URL
# Should show: postgresql://citeconnect:password@127.0.0.1:5432/citeconnect
```

---

### Issue 3: Import errors

**Problem:** Virtual environment not activated

**Solution:**
```bash
# Check which python
which python
# Must show: .../ModelPipeline/venv/bin/python

# If not, reactivate
cd ~/Documents/GitHub/ModelPipeline
source venv/bin/activate
```

---

### Issue 4: Databases not connecting

**Problem:** Docker containers not running

**Solution:**
```bash
# Check container status
docker-compose ps

# Start all containers
docker-compose up -d

# Wait and check again
sleep 30
docker-compose ps
```

---

### Issue 5: Port already in use

**Problem:** Another service using database ports

**Solution:**
```bash
# Check what's using port 5432
lsof -i :5432

# Stop conflicting service or change docker-compose.yml to use different port
```

---

## 8. Next Steps

### 8.1 Immediate Tasks for Continuation

**Priority 1: Build Embedding Foundation (Required for recommendations)**
1. `app/utils/embedding.py` - SPECTER model wrapper
2. `app/utils/similarity.py` - Cosine similarity functions
3. `app/services/embedding_service.py` - Embedding generation service

**Priority 2: Add Test Papers**
1. `scripts/seed_papers.py` - Seed 50-100 papers in PostgreSQL + Weaviate
2. This enables testing semantic search and recommendations

**Priority 3: Build Search Service**
1. `app/services/search_service.py` - Semantic search logic
2. `app/api/v1/search.py` - Search endpoint
3. Test with seeded papers

**Priority 4: Build Recommendation Service**
1. `app/services/recommendation_service.py` - Starter kit generation
2. `app/services/clustering_service.py` - K-means clustering
3. Test recommendation metrics (Precision@10, Recall@10)

---

### 8.2 Testing Recommendation Strategy

**Objective:** Validate recommendation scoring weights

**Current weights (from LLD):**
```python
score = (
    0.35 * semantic_similarity +
    0.20 * citation_relevance +
    0.15 * keyword_match +
    0.15 * popularity +
    0.10 * recency +
    0.05 * diversity
)
```

**To test:**
1. Seed 100+ papers in database
2. Implement recommendation service
3. Generate recommendations for test users
4. Measure:
   - Precision@10 (≥0.80 target)
   - Recall@10 (≥0.75 target)
   - MRR (≥0.70 target)
5. Experiment with different weights
6. Track with MLflow

---

### 8.3 Required for Model Development Guidelines

Per the assignment requirements:

**Must Implement:**
1. ✅ Loading data from data pipeline (PostgreSQL tables ready)
2. ❌ Model validation code (Recall@10, Precision@10, MRR)
3. ❌ Bias detection using domain slicing
4. ❌ Experiment tracking with MLflow
5. ❌ CI/CD pipeline for model validation
6. ❌ Push model to artifact registry (Google Cloud Artifact Registry)

---

## 9. Important Notes

### 9.1 Test Credentials

**Pre-seeded users:**

| Email | Password | Domain | Interests |
|-------|----------|--------|-----------|
| sarah.chen@example.com | Password123! | healthcare | NLP, clinical trials, drug discovery, protein folding |
| john.smith@example.com | Password123! | fintech | fraud detection, algorithmic trading, risk management |
| maria.garcia@example.com | Password123! | quantum_computing | quantum algorithms, error correction, quantum ML |
| test@example.com | Password123! | healthcare | machine learning, deep learning |

---

### 9.2 Key Configuration

**From .env:**
```bash
# Databases
DATABASE_URL=postgresql://citeconnect:password@127.0.0.1:5432/citeconnect
NEO4J_URI=bolt://localhost:7687
WEAVIATE_URL=http://localhost:8080
REDIS_URL=redis://localhost:6379/0

# JWT
SECRET_KEY=your-secret-key-min-32-characters-long-change-in-production
ACCESS_TOKEN_EXPIRE_HOURS=24

# SPECTER
SPECTER_MODEL_NAME=allenai/specter
EMBEDDING_DIMENSION=768
```

---

### 9.3 Code Quality Standards

All implemented code follows these standards:
- ✅ Comprehensive logging (every function logs entry/exit/errors)
- ✅ Full documentation (Google-style docstrings)
- ✅ Type hints on all parameters and returns
- ✅ Error handling with try-except
- ✅ No circular imports
- ✅ No unused variables
- ✅ Proper async/await usage

**Continue following these standards for new code!**

---

### 9.4 Database Inspection Commands

**PostgreSQL:**
```bash
docker exec -it citeconnect-postgres psql -U citeconnect -d citeconnect
```

**Neo4j:**
```bash
docker exec -it citeconnect-neo4j cypher-shell -u neo4j -p password
```

**Redis:**
```bash
docker exec -it citeconnect-redis redis-cli
```

---

### 9.5 Useful Commands

**Start everything:**
```bash
cd ~/Documents/GitHub/ModelPipeline
docker-compose up -d
cd citeconnect-backend
uvicorn app.main:app --reload
```

**Stop everything:**
```bash
# Stop FastAPI: Ctrl+C
docker-compose down
```

**Reset databases:**
```bash
docker-compose down -v
docker-compose up -d
cd citeconnect-backend
alembic upgrade head
./scripts/init_neo4j.sh
./scripts/seed_neo4j.sh
python scripts/seed_users.py
```

**View logs:**
```bash
docker-compose logs -f postgres
docker-compose logs -f neo4j
```

---

## 10. File Locations Reference

### Configuration Files (ModelPipeline root)
- `.env` - Environment configuration (DO NOT COMMIT)
- `.env.example` - Example configuration
- `requirements.txt` - Python dependencies
- `docker-compose.yml` - Database services

### Source Code (citeconnect-backend/)
- `app/main.py` - FastAPI application
- `app/core/*` - Core infrastructure (complete)
- `app/db/*` - Database clients (complete)
- `app/models/*` - Data models (complete)
- `app/schemas/*` - API schemas (complete)
- `app/services/auth_service.py` - Auth logic (complete)
- `app/services/user_service.py` - User logic (complete)
- `app/api/v1/auth.py` - Auth endpoints (complete)
- `app/api/v1/users.py` - User endpoints (complete)

### Scripts (citeconnect-backend/scripts/)
- `init_neo4j.sh` - Initialize Neo4j schema
- `seed_neo4j.sh` - Seed Neo4j with papers
- `seed_users.py` - Seed PostgreSQL with users

### Tests
- `verify_setup.py` - Verify environment setup
- `test_db_connectivity.py` - Test all 4 databases

---

## 11. Development Workflow

### Typical Development Session

```bash
# 1. Start fresh
cd ~/Documents/GitHub/ModelPipeline
source venv/bin/activate

# 2. Ensure databases are running
docker-compose ps
# If not running: docker-compose up -d

# 3. Navigate to backend
cd citeconnect-backend

# 4. Make code changes in app/

# 5. Test immediately (FastAPI auto-reloads)
# Visit http://localhost:8000/docs

# 6. Check logs in terminal running uvicorn

# 7. Test with curl or Swagger UI

# 8. Verify data in database
docker exec -it citeconnect-postgres psql -U citeconnect -d citeconnect
```

---

### Adding New Features

**To add a new endpoint:**

1. Create schema in `app/schemas/`
2. Create service logic in `app/services/`
3. Create API endpoint in `app/api/v1/`
4. Register router in `app/main.py`
5. Test in Swagger UI
6. Verify database changes

**Example:** Adding paper search
1. Schema: Already exists (`app/schemas/search.py`)
2. Service: Need to create `app/services/search_service.py`
3. API: Need to create `app/api/v1/search.py`
4. Register: Add to `app/main.py`

---

## 12. What Works Right Now

### ✅ You Can Test These Flows:

**Flow 1: User Registration**
```
POST /auth/register
    ↓
Creates user in PostgreSQL
    ↓
Returns JWT token
    ↓
User can immediately login
```

**Flow 2: User Login**
```
POST /auth/login
    ↓
Validates credentials
    ↓
Returns JWT token
    ↓
Token valid for 24 hours
```

**Flow 3: Get User Profile**
```
GET /users/me (with token)
    ↓
Validates JWT token
    ↓
Fetches user from PostgreSQL
    ↓
Fetches domain and interests
    ↓
Returns complete profile
```

**Flow 4: Update Profile**
```
PUT /users/me (with token, new data)
    ↓
Validates JWT token
    ↓
Updates user in PostgreSQL
    ↓
Deletes old interests
    ↓
Inserts new interests
    ↓
Invalidates Redis cache
```

---

### ❌ What Doesn't Work Yet:

- Searching for papers
- Viewing paper details
- Citation graph visualization
- Cluster generation (starter kit)
- Saving/liking papers
- Recommendations
- Interaction tracking

**These require implementing more services (Phase 9-10 in original plan)**

---

## 13. Performance & Monitoring

### Current Metrics

**API Response Times (from logs):**
- Registration: ~50-100ms
- Login: ~30-50ms
- Get Profile: ~20-30ms

**Database Query Times:**
- User lookup: ~5-10ms
- Interest fetch: ~5-10ms

**Connection Pool Stats:**
- PostgreSQL: 10 connections max
- Redis: 50 connections max

---

### Monitoring Tools

**Application Logs:**
```bash
# View live logs
tail -f logs/citeconnect.log

# Search for specific user
grep "user_id=1" logs/citeconnect.log
```

**Database Monitoring:**
```bash
# PostgreSQL stats
docker exec -it citeconnect-postgres psql -U citeconnect -d citeconnect -c \
  "SELECT * FROM pg_stat_database WHERE datname='citeconnect';"

# Check active connections
docker exec -it citeconnect-postgres psql -U citeconnect -d citeconnect -c \
  "SELECT count(*) FROM pg_stat_activity;"
```

---

## 14. Security Notes

### Current Security Implementation

**Password Security:**
- Bcrypt hashing with automatic salt
- Minimum 8 characters
- Must include: digit, uppercase, lowercase

**JWT Tokens:**
- HS256 algorithm
- 24-hour expiration for access tokens
- 7-day expiration for refresh tokens
- Signed with SECRET_KEY from .env

**API Security:**
- CORS configured for localhost:3000, localhost:5173
- Bearer token authentication on protected endpoints
- Input validation with Pydantic

**Database Security:**
- Parameterized queries (prevent SQL injection)
- Password not stored in version control (.env in .gitignore)
- Docker network isolation

---

### Security TODOs

For production deployment:
- [ ] Change SECRET_KEY in .env (use 32+ random characters)
- [ ] Enable HTTPS
- [ ] Add rate limiting (partially implemented in schema)
- [ ] Add API key rotation
- [ ] Enable database encryption at rest
- [ ] Set up firewall rules

---

## 15. Troubleshooting Checklist

If something doesn't work, check:

- [ ] Virtual environment activated (`which python`)
- [ ] All databases running (`docker-compose ps`)
- [ ] .env file exists and has correct values
- [ ] FastAPI app is running (`curl http://localhost:8000`)
- [ ] No port conflicts (`lsof -i :5432 :6379 :7687 :8080`)
- [ ] Migrations applied (`alembic current`)
- [ ] Test users seeded (`docker exec ... psql SELECT * FROM users;`)

---

## 16. Contact & Resources

**Team Communication:**
- MLOps Lead: Dennis Jose
- Project GitHub: (shared repository)

**Key Documents:**
- Low-Level Design: `citeconnect_lld.md`
- Scoping Document: `CiteConnect - Scoping.docx`
- Model Guidelines: `Model Development Guidelines.pdf`

**External Resources:**
- FastAPI Docs: https://fastapi.tiangolo.com/
- Pydantic: https://docs.pydantic.dev/
- SPECTER Paper: https://arxiv.org/abs/2004.07180

---

## 17. Quick Reference

### Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| DATABASE_URL | postgresql://citeconnect:password@127.0.0.1:5432/citeconnect | PostgreSQL connection |
| NEO4J_URI | bolt://localhost:7687 | Neo4j connection |
| WEAVIATE_URL | http://localhost:8080 | Weaviate connection |
| REDIS_URL | redis://localhost:6379/0 | Redis connection |
| SECRET_KEY | [32+ chars] | JWT signing key |
| SPECTER_MODEL_NAME | allenai/specter | Embedding model |

---

### Database Ports

| Database | Port | GUI Access |
|----------|------|------------|
| PostgreSQL | 5432 | DBeaver, pgAdmin |
| Neo4j | 7474 (HTTP), 7687 (Bolt) | http://localhost:7474 |
| Weaviate | 8080 | http://localhost:8080 |
| Redis | 6379 | redis-cli |

---

### API Endpoints

| Method | Endpoint | Auth Required | Status |
|--------|----------|---------------|--------|
| POST | /api/v1/auth/register | No | ✅ Working |
| POST | /api/v1/auth/login | No | ✅ Working |
| POST | /api/v1/auth/refresh | No | ✅ Working |
| GET | /api/v1/users/me | Yes | ✅ Working |
| PUT | /api/v1/users/me | Yes | ✅ Working |
| GET | /api/v1/health | No | ✅ Working |

---

## 18. Success Criteria Checklist

Before considering this phase complete:

- [x] All databases running and connected
- [x] PostgreSQL tables created (13 tables)
- [x] Neo4j schema initialized
- [x] Sample data seeded
- [x] User can register via API
- [x] User can login via API
- [x] JWT authentication working
- [x] Protected endpoints require auth
- [x] Data persists in database
- [x] API documentation accessible

**All criteria met!** ✅ Ready for next phase.

---

## 19. Handoff Checklist

Before starting work, verify:

- [ ] Can activate venv (`source venv/bin/activate`)
- [ ] Can start databases (`docker-compose up -d`)
- [ ] All 4 databases healthy (`python test_db_connectivity.py`)
- [ ] Can start FastAPI (`uvicorn app.main:app --reload`)
- [ ] Can access Swagger UI (http://localhost:8000/docs)
- [ ] Can register new user via API
- [ ] Can login with test user (sarah.chen@example.com / Password123!)
- [ ] Can access protected endpoint with token
- [ ] Can view data in DBeaver/Neo4j Browser

**If all checked, you're ready to continue!**

---

**Document Version:** 1.0  
**For Questions:** Contact Dennis Jose (MLOps Lead)  
**Last Verified:** November 12, 2025
