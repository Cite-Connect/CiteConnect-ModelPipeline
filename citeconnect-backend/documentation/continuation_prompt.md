# CiteConnect Backend Development - Continuation Prompt

**USE THIS AS YOUR FIRST MESSAGE IN A NEW CLAUDE CHAT**

---

I'm continuing development on the CiteConnect backend, a research paper recommendation system for IE7305 (Northeastern University). The project combines semantic search, citation graph analysis, and personalized recommendations. We're presenting at Google HQ in December 2025.

## Current Progress: ~40% Complete

My teammate Dennis has built the foundation. I need to continue from where he left off, following the existing code patterns and architecture exactly.

---

## Project Context

**What CiteConnect Does:**
- Recommends academic papers using SPECTER embeddings (768-dim vectors)
- Semantic search via Weaviate vector database
- Citation network visualization via Neo4j graph database
- Personalized clusters using K-means (k=3)
- User profile learning from interactions

**Technical Stack:**
- Backend: FastAPI (Python 3.11) with asyncpg, async patterns
- Databases: PostgreSQL (metadata), Neo4j (citations), Weaviate (embeddings), Redis (cache)
- ML: SPECTER model (allenai/specter via sentence-transformers)
- Deployment Target: Google Cloud Platform

**Key Metrics to Achieve:**
- Precision@10 ≥ 0.80
- Recall@10 ≥ 0.75
- Query Latency (p95) < 2 seconds

---

## What's Already Implemented ✅

### Phase 6: Core Infrastructure (100% Complete)

**`app/core/exceptions.py`**
- Complete exception hierarchy: CiteConnectException (base), AuthenticationError, ValidationError, DatabaseError, etc.
- All exceptions log automatically with structured logging

**`app/core/logging.py`**
- JSONFormatter for production logs
- ColoredFormatter for development
- Structured logging with request IDs
- setup_logging() function configures everything

**`app/core/config.py`**
- Pydantic Settings-based configuration
- Loads from .env file in ModelPipeline root (not in citeconnect-backend)
- Validates all settings with @validator decorators
- Provides helper methods (is_development, get_cache_ttl, etc.)

**`app/core/security.py`**
- hash_password() - bcrypt hashing
- verify_password() - constant-time comparison
- create_access_token(), create_refresh_token() - JWT generation
- decode_token() - JWT validation
- create_token_pair() - convenience function

---

### Phase 7: Database Connections (100% Complete)

**`app/db/postgres.py`**
- Async connection pooling with asyncpg
- get_db_pool() - returns global pool
- execute_query() - convenience wrapper
- execute_transaction() - atomic operations
- Retry logic with exponential backoff (3 attempts)

**`app/db/redis_client.py`**
- Async Redis client with redis.asyncio
- cache_set(), cache_get(), cache_delete()
- cache_increment() for counters
- serialize_value() / deserialize_value() helpers
- TTL management

**`app/db/weaviate_client.py`**
- Weaviate client with Paper schema
- SPECTER embeddings (768-dim, cosine similarity)
- search_papers() - vector similarity search
- batch_insert_papers() - bulk operations
- HNSW config: ef=64, efConstruction=128

**`app/db/neo4j_client.py`**
- Async Neo4j driver
- execute_query(), execute_write_query()
- create_paper_node(), create_citation_relationship()
- get_cited_papers(), get_citing_papers(), get_co_cited_papers()
- get_citation_network() - builds graph structure

---

### Phase 8: Models & Schemas (100% Complete)

**All Pydantic models in `app/models/`:**
- user.py - User, UserInterest, UserDomain, UserProfileEmbedding
- paper.py - Paper, PaperMetadata, PaperWithScore
- cluster.py - Cluster, ClusterPaper
- interaction.py - Interaction, InteractionContext
- graph.py - GraphNode, GraphEdge, CitationNetwork

**All API schemas in `app/schemas/`:**
- auth.py - LoginRequest, RegisterRequest, TokenResponse
- user.py - UserResponse, UserUpdateRequest, HomeResponse, DashboardResponse
- paper.py - PaperResponse, PaperDetailResponse
- search.py - SearchRequest, SearchResponse
- cluster.py - ClusterResponse

---

### Phase Auth: Authentication & User Management (100% Complete)

**`app/api/deps.py`**
- get_current_user_id() - Extract user_id from JWT
- get_current_user() - Fetch full user profile
- get_optional_user_id() - For endpoints that work without auth
- Uses HTTPBearer for token extraction

**`app/services/auth_service.py`**
- register_user() - Complete registration flow
  - Validates email not duplicate
  - Hashes password
  - INSERT INTO users, user_domains, user_interests
  - Generates JWT tokens
  - Returns user data + tokens
- login_user() - Authentication flow
  - Validates credentials
  - Generates tokens
- refresh_access_token() - Token refresh logic

**`app/services/user_service.py`**
- get_user_profile() - Fetches user with domain and interests
- update_user_profile() - Updates name, interests, google_scholar_url
- Invalidates Redis cache when interests change

**`app/api/v1/auth.py`**
- POST /auth/register (calls register_user)
- POST /auth/login (calls login_user)
- POST /auth/refresh (calls refresh_access_token)

**`app/api/v1/users.py`**
- GET /users/me (requires auth, calls get_user_profile)
- PUT /users/me (requires auth, calls update_user_profile)

**`app/main.py`**
- FastAPI application with lifespan management
- Database initialization on startup
- Route registration for auth and users
- Exception handlers
- CORS middleware
- Health check endpoint

---

### Databases (100% Setup)

**PostgreSQL:**
- Migration: 001_initial_schema applied
- 13 tables created (users, user_domains, user_interests, papers, user_interactions, user_saved_papers, user_liked_papers, paper_clusters, cluster_papers, user_profile_embeddings, rate_limits, system_metrics, alembic_version)
- 4 test users seeded
- All indexes created (full-text search GIN indexes on papers.title and papers.abstract)

**Neo4j:**
- Constraints: paper_id_unique, user_id_unique
- Indexes: paper_id, paper_domain, paper_year, user_id, user_domain
- 5 sample papers seeded with 4 citation relationships
- Accessible at http://localhost:7474 (neo4j/password)

**Weaviate:**
- Paper collection schema created
- 768-dimensional vector index ready
- No papers inserted yet

**Redis:**
- Connected and functional
- TTL values defined in config

---

## Important Implementation Details

### Directory Structure

```
ModelPipeline/                  ← Root (config files here)
├── .env                        ← Environment variables
├── requirements.txt
├── docker-compose.yml
├── venv/                       ← Python 3.11 virtual environment
│
└── citeconnect-backend/        ← Source code here
    ├── app/
    │   ├── main.py
    │   ├── api/
    │   ├── core/               ← COMPLETE
    │   ├── db/                 ← COMPLETE
    │   ├── models/             ← COMPLETE
    │   ├── schemas/            ← COMPLETE
    │   ├── services/           ← auth_service, user_service COMPLETE
    │   ├── utils/              ← EMPTY - TODO
    │   ├── tasks/              ← EMPTY - TODO
    │   └── middleware/         ← EMPTY - TODO
    ├── alembic/
    ├── scripts/
    └── tests/
```

---

### Critical Code Quality Standards (MUST FOLLOW)

1. **No Circular Imports**
   - Import order: core → db → models → schemas → services → api
   - Use TYPE_CHECKING for forward references if needed

2. **Comprehensive Logging**
   - EVERY function must log entry: `logger.info(f"Starting {function_name} with param={value}")`
   - Log before/after database calls
   - Log errors with full context and exc_info=True
   - Use appropriate log levels (DEBUG, INFO, WARNING, ERROR)

3. **No Unused Variables**
   - Every variable declared must be used
   - Use underscore prefix for intentionally unused: `_unused`

4. **Proper Async/Await**
   - Only use async for I/O operations (database, API calls)
   - Don't mix sync and async incorrectly
   - All database operations in app/db/ are async
   - FastAPI endpoints use async def

5. **Full Documentation**
   - Module docstring at top of every file
   - Class docstrings for every class
   - Function docstrings (Google style) for every function
   - Type hints on ALL parameters and returns

6. **Error Handling**
   - Try-except blocks with logging
   - Raise custom exceptions from app.core.exceptions
   - Never silently fail

---

### Key Configuration Decisions

**SPECTER Model:**
- Using `allenai/specter` (NOT specter2)
- specter2 had loading issues, specter works directly
- 768-dimensional embeddings
- Model cached in ~/.cache/huggingface/

**Database Connections:**
- Use `127.0.0.1` NOT `localhost` (avoids IPv6 issues with psycopg2)
- All databases in Docker, exposed via port mapping
- Local Postgres.app must be stopped to avoid conflicts

**Pydantic Validators:**
- Currently using @validator (V1 style) - works but deprecated warnings
- Can be migrated to @field_validator (V2 style) later

**Environment File:**
- .env is in ModelPipeline root
- app/core/config.py looks for ../.env
- alembic/env.py loads from ../../.env

---

## Current Working Directory Setup

**Where to run commands:**

```bash
# From ModelPipeline root:
- docker-compose commands
- source venv/bin/activate

# From citeconnect-backend/:
- uvicorn app.main:app --reload
- alembic commands
- python verify_setup.py
- python test_db_connectivity.py
- python scripts/seed_users.py
- ./scripts/init_neo4j.sh
```

---

## Test Credentials

**Login credentials for testing:**
```
Email: sarah.chen@example.com
Password: Password123!
Domain: healthcare
Interests: NLP, clinical trials, drug discovery, protein folding

Email: john.smith@example.com
Password: Password123!
Domain: fintech

Email: maria.garcia@example.com
Password: Password123!
Domain: quantum_computing

Email: test@example.com
Password: Password123!
Domain: healthcare
```

---

## What to Build Next

### Immediate Priority: Embedding & Search Foundation

**You need to implement Phase 9 (Utilities) and Phase 10 (Services) to enable recommendations.**

**Phase 9: Utilities (~2-3 hours)**

1. **`app/utils/embedding.py`** - SPECTER model wrapper
   - load_specter_model() - Load and cache model
   - encode_text(text) → 768-dim vector
   - encode_paper(title, abstract, intro) → 768-dim vector
   - encode_batch(texts[]) → multiple vectors
   - Must use settings.SPECTER_MODEL_NAME from config

2. **`app/utils/similarity.py`** - Similarity calculations
   - cosine_similarity(vec1, vec2) → float (0-1)
   - euclidean_distance(vec1, vec2) → float
   - compute_similarity_matrix(vectors) → NxN matrix

3. **`app/utils/clustering.py`** - Clustering algorithms
   - kmeans_clustering(embeddings, k=3) → labels
   - force_directed_layout(graph) → positions
   - Uses scikit-learn for k-means
   - Uses NetworkX for graph layouts

4. **`app/utils/pdf_parser.py`** - PDF text extraction
   - extract_text_from_pdf(file_path) → text
   - Use PyMuPDF or pdfminer.six

5. **`app/utils/validators.py`** - Custom validators
   - validate_paper_id(paper_id) → bool
   - validate_domain(domain) → bool

**Phase 10: Core Services (~4-6 hours)**

1. **`app/services/embedding_service.py`** (PRIORITY 1)
   - Class: EmbeddingService
   - Methods:
     - __init__() - Load SPECTER model
     - encode_text(text) - Single text encoding
     - encode_paper(title, abstract, intro) - Paper encoding
     - encode_batch(texts, batch_size=32) - Batch encoding
     - compute_similarity(emb1, emb2) - Cosine similarity
   - Must load model ONCE and reuse (singleton pattern)
   - Must handle model loading errors gracefully

2. **`app/services/search_service.py`** (PRIORITY 2)
   - Class: SearchService
   - Dependencies: Weaviate, EmbeddingService, Redis, PostgreSQL
   - Methods:
     - search(query, domain, filters, limit) → List[papers]
     - Steps:
       1. Check Redis cache
       2. Generate query embedding
       3. Query Weaviate with vector
       4. Enrich with PostgreSQL metadata
       5. Calculate relevance scores
       6. Cache results
   - Relevance scoring: 0.85*semantic + 0.15*keyword_boost

3. **`app/services/recommendation_service.py`** (PRIORITY 3)
   - Class: RecommendationService
   - Methods:
     - generate_starter_kit(user_id) → 3 clusters
     - _build_query_embedding(user_profile) - Combine interests
     - _multi_strategy_retrieval() - Get candidate papers
     - _score_candidates() - Multi-factor scoring
       - 0.35 × semantic_similarity
       - 0.20 × citation_relevance
       - 0.15 × keyword_match
       - 0.15 × popularity
       - 0.10 × recency
       - 0.05 × diversity
     - _cluster_papers() - K-means clustering
   - Uses EmbeddingService, SearchService, ClusteringService

4. **`app/services/clustering_service.py`**
   - generate_home_clusters(user_id, n_clusters=3)
   - Uses k-means from sklearn
   - Generates theme names (can use simple rules or LLM)

---

## Reference Documents

**You have access to these documents:**
1. **Low-Level Design:** `citeconnect_lld.md` - Complete architecture
2. **Scoping Document:** Contains metrics, objectives, failure analysis
3. **Model Development Guidelines:** Requirements for model validation, bias detection, CI/CD
4. **Backend Generation Prompt:** `backend_generation_prompt.md` - Code standards

**Follow the LLD document exactly for:**
- Database schemas (Section 5)
- API specifications (Section 6)
- Service implementations (Section 7)
- Algorithms (Section 8)
- Data flows (Section 9)

---

## Environment Setup

**Location:** `~/Documents/GitHub/ModelPipeline`

**Virtual Environment:**
- Python 3.11
- Located at: `ModelPipeline/venv/`
- Activate: `source venv/bin/activate`

**Configuration File (.env):**
- Located at: `ModelPipeline/.env`
- Key settings:
  ```
  DATABASE_URL=postgresql://citeconnect:password@127.0.0.1:5432/citeconnect
  NEO4J_URI=bolt://localhost:7687
  WEAVIATE_URL=http://localhost:8080
  REDIS_URL=redis://localhost:6379/0
  SPECTER_MODEL_NAME=allenai/specter
  EMBEDDING_DIMENSION=768
  ```

**Docker Services:**
- All run via docker-compose.yml in ModelPipeline root
- PostgreSQL: port 5432
- Neo4j: ports 7474 (browser), 7687 (bolt)
- Weaviate: port 8080
- Redis: port 6379

---

## Database State

### PostgreSQL (127.0.0.1:5432, user: citeconnect, password: password)

**Tables (13 total):**
- `users` - 4 test users
- `user_domains` - 4 domains
- `user_interests` - ~12 interests total
- `papers` - EMPTY (ready for data)
- `user_interactions` - EMPTY
- `user_saved_papers` - EMPTY
- `user_liked_papers` - EMPTY
- `paper_clusters` - EMPTY
- `cluster_papers` - EMPTY
- `user_profile_embeddings` - EMPTY
- `rate_limits` - EMPTY
- `system_metrics` - EMPTY
- `alembic_version` - 1 record

**Migration Status:** 001_initial_schema applied

### Neo4j (localhost:7474, user: neo4j, password: password)

**Data:**
- 5 Paper nodes (arxiv:2401.001 through arxiv:2401.005)
- 4 CITES relationships
- 4 CITED_BY relationships

**Schema:**
- Constraints: paper_id_unique, user_id_unique
- Indexes: paper_id, paper_domain, paper_year

### Weaviate (localhost:8080)

**Schema:** Paper collection created (768-dim vectors, cosine similarity)
**Data:** EMPTY - no papers inserted yet

### Redis (localhost:6379)

**Status:** Connected, no data yet

---

## Current API Endpoints

**Working (test in Swagger UI: http://localhost:8000/docs):**

- `POST /api/v1/auth/register` - Create user account
- `POST /api/v1/auth/login` - Authenticate user
- `POST /api/v1/auth/refresh` - Refresh access token
- `GET /api/v1/users/me` - Get user profile (requires auth)
- `PUT /api/v1/users/me` - Update profile (requires auth)
- `GET /api/v1/health` - Health check

**Not Implemented Yet:**
- Paper endpoints
- Search endpoint
- Graph endpoint
- Cluster endpoints
- Interaction tracking

---

## Code Patterns to Follow

### Pattern 1: Service Class Structure

```python
# app/services/example_service.py

import logging
from typing import List, Dict, Any

from app.core.exceptions import DatabaseError
from app.db.postgres import execute_query

logger = logging.getLogger(__name__)


class ExampleService:
    """
    Service class description.
    
    Attributes:
        dependency1: Description
        dependency2: Description
    """
    
    def __init__(self, dependency1, dependency2):
        """Initialize service with dependencies."""
        logger.info("Initializing ExampleService")
        self.dependency1 = dependency1
        self.dependency2 = dependency2
    
    async def some_method(self, param: str) -> Dict[str, Any]:
        """
        Method description.
        
        Args:
            param: Parameter description
        
        Returns:
            Result description
        
        Raises:
            DatabaseError: When database operation fails
        """
        logger.info(f"Starting some_method with param={param}")
        
        try:
            # Implementation here
            result = await execute_query("SELECT ...", param, fetch_one=True)
            
            logger.info(f"some_method completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"some_method failed: {str(e)}", exc_info=True)
            raise DatabaseError(f"Operation failed: {str(e)}")
```

---

### Pattern 2: API Endpoint Structure

```python
# app/api/v1/example.py

import logging
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_current_user_id
from app.schemas.example import ExampleRequest, ExampleResponse
from app.services.example_service import ExampleService
from app.core.exceptions import DatabaseError

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/endpoint", response_model=ExampleResponse)
async def endpoint_name(
    request: ExampleRequest,
    user_id: int = Depends(get_current_user_id)
):
    """
    Endpoint description.
    
    Detailed explanation of what it does.
    """
    logger.info(f"Endpoint called", extra={"user_id": user_id})
    
    try:
        # Call service
        result = await service.method(request.param)
        
        logger.info("Endpoint completed successfully")
        return result
        
    except DatabaseError as e:
        logger.error(f"Endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.message
        )
```

---

### Pattern 3: Database Query Pattern

```python
# Always use parameterized queries
result = await execute_query(
    "SELECT * FROM users WHERE user_id = $1",  # $1, $2, etc. for params
    user_id,  # Parameters as separate args
    fetch_one=True  # or fetch_all=True
)

# Never concatenate SQL
# ❌ BAD: f"SELECT * FROM users WHERE email = '{email}'"
# ✅ GOOD: execute_query("SELECT * FROM users WHERE email = $1", email)
```

---

## Resolved Issues & Solutions

### Issue 1: SPECTER2 Model Not Loading
**Solution:** Use `allenai/specter` instead of `allenai/specter2`

### Issue 2: PostgreSQL Role Doesn't Exist
**Solution:** Stop local Postgres.app, use only Docker PostgreSQL

### Issue 3: Localhost IPv6 Connection Issues
**Solution:** Use `127.0.0.1` instead of `localhost` in DATABASE_URL

### Issue 4: Alembic Using System Python
**Solution:** Verify `which alembic` shows venv path, reinstall if needed

### Issue 5: Email Validation Missing
**Solution:** Install `pip install 'pydantic[email]'`

### Issue 6: Bcrypt Version Incompatibility
**Solution:** Install `bcrypt==4.0.1` explicitly

---

## How to Continue

### Verification Steps Before Starting

```bash
# 1. Navigate to project
cd ~/Documents/GitHub/ModelPipeline

# 2. Activate venv
source venv/bin/activate

# 3. Check Python
which python
# Must show: .../ModelPipeline/venv/bin/python

# 4. Start databases
docker-compose up -d
sleep 30

# 5. Verify all healthy
docker-compose ps

# 6. Navigate to backend
cd citeconnect-backend

# 7. Test setup
python verify_setup.py
# Should show: All tests passed

# 8. Test databases
python test_db_connectivity.py
# Should show: 4/4 connected

# 9. Start FastAPI
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 10. Verify endpoints
curl http://localhost:8000/api/v1/health
# Should show all services healthy
```

---

### Recommended Next Steps

**Step 1: Build Embedding Foundation (Required)**

Create `app/utils/embedding.py`:
```python
# This will be used by ALL services for generating embeddings
# Load SPECTER model once, reuse throughout application
# Pattern:
class SpecterModelWrapper:
    _instance = None  # Singleton
    
    def __init__(self):
        self.model = SentenceTransformer('allenai/specter')
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    def encode(self, text: str) -> np.ndarray:
        return self.model.encode(text)
```

**Step 2: Seed Papers for Testing**

Create `scripts/seed_papers.py`:
```python
# Add 50-100 papers to PostgreSQL + Weaviate
# This enables testing search and recommendations
# Papers should span all 3 domains
```

**Step 3: Build Search Service**

Implement `app/services/search_service.py`:
- Uses EmbeddingService to convert query to vector
- Queries Weaviate for similar papers
- Enriches with PostgreSQL metadata
- Caches in Redis

**Step 4: Create Search Endpoint**

Implement `app/api/v1/search.py`:
- GET /search?q={query}&domain={domain}&limit={limit}
- Calls SearchService
- Returns ranked papers

---

## Important Constraints

### Must Use These Exact Versions

```
Python: 3.11.x
fastapi==0.104.1
pydantic==2.4.2
sentence-transformers==2.5.1
torch==2.1.1
asyncpg==0.29.0
weaviate-client==3.24.2
neo4j==5.14.1
redis==5.0.1
```

### Must Follow LLD Specifications

**Embedding Dimension:** Exactly 768 (SPECTER output)

**Clustering:** K-means with k=3 for home page

**Scoring Weights (from LLD Section 8.1):**
```python
final_score = (
    0.35 * semantic_similarity +
    0.20 * citation_relevance +
    0.15 * keyword_match +
    0.15 * popularity +
    0.10 * recency +
    0.05 * diversity
)
```

**Cache TTLs (from LLD Section 11):**
- starter_kit: 86400s (24h)
- graph: 3600s (1h)
- search: 1800s (30min)

---

## Files NOT to Modify

These are complete and working - don't change:
- `app/core/*` - All core modules
- `app/db/*` - All database clients
- `app/models/*` - All data models
- `app/schemas/*` - All API schemas
- `app/services/auth_service.py` - Auth logic
- `app/services/user_service.py` - User logic
- `app/api/v1/auth.py` - Auth endpoints
- `app/api/v1/users.py` - User endpoints
- `app/api/deps.py` - Dependencies
- `alembic/versions/001_initial_schema.py` - Database migration

**Only modify `app/main.py` to add new route registrations**

---

## Success Criteria for Next Phase

**You'll know Phase 9-10 is complete when you can:**

1. Generate embedding for text: `embedding_service.encode_text("machine learning")` → 768-dim vector
2. Search for papers: `GET /search?q=protein folding` → ranked results
3. Get citation graph: `GET /papers/{id}/graph` → nodes and edges
4. Measure metrics: Precision@10, Recall@10, MRR
5. All operations log comprehensively
6. All operations cache appropriately
7. Test suite passes (>80% coverage target)

---

## Sample Workflow for Building Search Service

```python
# 1. Create app/utils/embedding.py
# 2. Test: python -c "from app.utils.embedding import encode_text; print(encode_text('test').shape)"
# 3. Create app/services/embedding_service.py
# 4. Create scripts/seed_papers.py and run it
# 5. Create app/services/search_service.py
# 6. Create app/api/v1/search.py
# 7. Register route in app/main.py
# 8. Test: GET /search?q=machine learning
# 9. Verify results in Swagger UI
# 10. Check logs for comprehensive logging
```

---

## Questions to Ask If Stuck

**If unsure about implementation:**
- "How should I implement {feature} following the LLD Section X?"
- "What's the correct async pattern for {database} operations?"
- "How do I integrate {service} with existing code without circular imports?"

**If errors occur:**
- "I'm getting error {error_message}, what's the solution?"
- "How do I debug {specific_issue}?"

**For architecture decisions:**
- "The LLD specifies {approach}, but I'm seeing {alternative}, which should I use?"
- "How does {component} integrate with {other_component}?"

---

## Contact Information

**Primary Developer (Current Phase):** Dennis Jose (MLOps Lead)
**Project Repository:** (shared repository)
**Presentation:** December 2025 @ Google HQ

---

## Final Checklist Before Starting

- [ ] Can activate venv and see correct Python version
- [ ] Docker containers all running and healthy
- [ ] Can access Swagger UI at localhost:8000/docs
- [ ] Can register new user via API
- [ ] Can login with sarah.chen@example.com / Password123!
- [ ] Can access /users/me with valid token
- [ ] Can see 4 users in PostgreSQL
- [ ] Can see 5 papers in Neo4j Browser
- [ ] Reference documents (LLD, scoping) accessible
- [ ] Understand the multi-database architecture
- [ ] Know where to run commands (root vs backend directory)

**If all checked, you're ready to build the next phase!**

---

## Your First Task

**I recommend starting with:**

"Help me implement Phase 9 (Utilities) following the CiteConnect Low-Level Design document. I need to create the embedding utility functions first, starting with `app/utils/embedding.py`. The file should load the SPECTER model (allenai/specter) and provide functions to encode text and papers into 768-dimensional vectors. Follow the existing code quality standards in the project (comprehensive logging, full documentation, type hints, error handling)."

Then continue with similarity.py, clustering.py, and the core services.

---

**Good luck with the implementation! Follow the established patterns and you'll build on a solid foundation.**
