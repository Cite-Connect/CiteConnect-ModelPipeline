# CiteConnect Database Schema & Troubleshooting Guide

**Project:** CiteConnect Research Paper Recommendation System  
**Document Type:** Complete Database Reference  
**Last Updated:** November 12, 2025

---

## Table of Contents

1. [Overview](#1-overview)
2. [PostgreSQL Schema](#2-postgresql-schema)
3. [Neo4j Schema](#3-neo4j-schema)
4. [Weaviate Schema](#4-weaviate-schema)
5. [Redis Schema](#5-redis-schema)
6. [Data Population Flows](#6-data-population-flows)
7. [Database Relationships](#7-database-relationships)
8. [Setup & Migration Guide](#8-setup--migration-guide)
9. [Troubleshooting Guide](#9-troubleshooting-guide)
10. [Inspection & Monitoring](#10-inspection--monitoring)

---

## 1. Overview

CiteConnect uses a **multi-database architecture** optimized for different data types:

| Database | Purpose | Data Type | Access Pattern |
|----------|---------|-----------|----------------|
| **PostgreSQL** | User profiles, paper metadata, interactions | Relational | CRUD, aggregations |
| **Weaviate** | Paper embeddings for semantic search | Vector | Similarity search |
| **Neo4j** | Citation relationships | Graph | Graph traversal |
| **Redis** | Caching, sessions, rate limiting | Key-value | Fast read/write |

---

## 2. PostgreSQL Schema

### 2.1 Database Information

- **Database Name:** #######
- **User:** #######
- **Password:** ##########
- **Port:** 5432 (mapped from Docker)
- **Total Tables:** 13
- **Migration Tool:** Alembic
- **Current Revision:** 001_initial_schema

---

### 2.2 Table Details

#### Table 1: `users`

**Purpose:** Store user account information

**Columns:**

| Column | Type | Constraints | Purpose |
|--------|------|-------------|---------|
| user_id | INTEGER | PRIMARY KEY, AUTO_INCREMENT | Unique user identifier |
| email | VARCHAR(255) | UNIQUE, NOT NULL | User email (login credential) |
| password_hash | VARCHAR(255) | NOT NULL | Bcrypt hashed password |
| name | VARCHAR(255) | NOT NULL | User's full name |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Account creation time |
| updated_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Last profile update |
| is_active | BOOLEAN | DEFAULT TRUE | Account active status |
| google_scholar_url | VARCHAR(500) | NULL | Google Scholar profile link |
| semantic_scholar_author_id | VARCHAR(100) | NULL | Semantic Scholar author ID |

**Indexes:**
- PRIMARY KEY on user_id
- UNIQUE INDEX on email
- INDEX on semantic_scholar_author_id

**Relationships:**
- Referenced by: user_domains, user_interests, user_profile_embeddings, user_interactions, user_saved_papers, user_liked_papers, paper_clusters, rate_limits

**Sample Data:**
```sql
INSERT INTO users (email, password_hash, name) 
VALUES ('sarah@example.com', '$2b$12$...', 'Sarah Chen');
```

---

#### Table 2: `user_domains`

**Purpose:** Store user's selected research domain (1:1 with users)

**Columns:**

| Column | Type | Constraints | Purpose |
|--------|------|-------------|---------|
| user_id | INTEGER | PRIMARY KEY, FK → users(user_id) | User identifier |
| domain | VARCHAR(50) | NOT NULL, CHECK (healthcare, fintech, quantum_computing) | Selected domain |
| selected_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Selection time |

**Why This Table Exists:**
- Enforces single domain per user
- Allows domain changes over time
- Tracks when domain was selected

**Sample Data:**
```sql
INSERT INTO user_domains (user_id, domain) 
VALUES (1, 'healthcare');
```

---

#### Table 3: `user_interests`

**Purpose:** Store user's research interest keywords

**Columns:**

| Column | Type | Constraints | Purpose |
|--------|------|-------------|---------|
| interest_id | INTEGER | PRIMARY KEY, AUTO_INCREMENT | Interest identifier |
| user_id | INTEGER | FK → users(user_id) CASCADE | User who has this interest |
| interest_keyword | VARCHAR(100) | NOT NULL | Research keyword (e.g., "NLP") |
| source | VARCHAR(50) | CHECK (manual, google_scholar, inferred) | How interest was added |
| weight | FLOAT | DEFAULT 1.0, CHECK (0.0-1.0) | Interest importance weight |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | When added |

**Indexes:**
- INDEX on user_id (for fast lookup)

**Why Weights Exist:**
- Manual interests: weight = 1.0 (highest)
- Google Scholar interests: weight = 0.8 (inferred from publications)
- Inferred interests: weight = 0.5-0.7 (learned from interactions)

**Sample Data:**
```sql
INSERT INTO user_interests (user_id, interest_keyword, source, weight) 
VALUES 
  (1, 'machine learning', 'manual', 1.0),
  (1, 'protein folding', 'google_scholar', 0.8),
  (1, 'clinical trials', 'inferred', 0.6);
```

---

#### Table 4: `user_profile_embeddings`

**Purpose:** Store user's profile as 768-dimensional vector for personalization

**Columns:**

| Column | Type | Constraints | Purpose |
|--------|------|-------------|---------|
| user_id | INTEGER | PRIMARY KEY, FK → users(user_id) | User identifier |
| embedding_vector | FLOAT8[] | NOT NULL | 768-dimensional vector |
| last_updated | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Last update time |
| based_on_papers | TEXT[] | NULL | Paper IDs used to build embedding |
| interaction_count | INTEGER | DEFAULT 0 | Number of interactions since update |

**How Embedding is Generated:**
```python
# Weighted average of saved/liked paper embeddings
saved_papers = get_user_saved_papers(user_id, last_30_days)
saved_embeddings = [get_embedding(p) for p in saved_papers]

liked_papers = get_user_liked_papers(user_id, last_30_days)
liked_embeddings = [get_embedding(p) for p in liked_papers]

user_embedding = (
    0.5 * average(saved_embeddings) +
    0.3 * average(liked_embeddings) +
    0.2 * average(long_viewed_embeddings)
)
```

**Update Trigger:** Every 10 interactions

**Sample Data:**
```sql
INSERT INTO user_profile_embeddings (user_id, embedding_vector, based_on_papers) 
VALUES (1, ARRAY[0.1, 0.2, ..., 0.05], ARRAY['arxiv:2401.001', 'arxiv:2401.002']);
```

---

#### Table 5: `papers`

**Purpose:** Store paper metadata and content

**Columns:**

| Column | Type | Constraints | Purpose |
|--------|------|-------------|---------|
| paper_id | VARCHAR(100) | PRIMARY KEY | Unique identifier (e.g., arxiv:2401.12345) |
| title | TEXT | NOT NULL | Paper title |
| authors | TEXT[] | NULL | Array of author names |
| year | INTEGER | NULL | Publication year |
| venue | VARCHAR(255) | NULL | Journal or conference name |
| citation_count | INTEGER | DEFAULT 0 | Number of citations |
| abstract | TEXT | NULL | Paper abstract |
| summary | TEXT | NULL | AI-generated summary |
| introduction | TEXT | NULL | Introduction section text |
| gcs_pdf_path | VARCHAR(500) | NULL | Google Cloud Storage PDF path |
| domain | VARCHAR(50) | CHECK (healthcare, fintech, quantum_computing) | Research domain |
| ingested_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Ingestion time |
| updated_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Last update time |

**Indexes:**
- PRIMARY KEY on paper_id
- INDEX on domain (for filtering by domain)
- INDEX on year (for filtering by year)
- INDEX on citation_count (for popularity sorting)
- **GIN INDEX** on title (full-text search)
- **GIN INDEX** on abstract (full-text search)

**Why Full-Text Indexes:**
```sql
-- Enable fast keyword search
SELECT * FROM papers 
WHERE to_tsvector('english', title) @@ to_tsquery('machine & learning');
```

**Sample Data:**
```sql
INSERT INTO papers (paper_id, title, authors, year, venue, citation_count, abstract, domain)
VALUES (
  'arxiv:2401.12345',
  'AlphaFold: Improved protein structure prediction',
  ARRAY['Jumper, J.', 'Evans, R.', 'Pritzel, A.'],
  2021,
  'Nature',
  9432,
  'Proteins are essential to life...',
  'healthcare'
);
```

---

#### Table 6: `user_interactions`

**Purpose:** Track all user interactions with papers for personalization

**Columns:**

| Column | Type | Constraints | Purpose |
|--------|------|-------------|---------|
| interaction_id | INTEGER | PRIMARY KEY, AUTO_INCREMENT | Unique interaction ID |
| user_id | INTEGER | FK → users CASCADE | User who interacted |
| paper_id | VARCHAR(100) | FK → papers CASCADE | Paper interacted with |
| interaction_type | VARCHAR(50) | CHECK (view, click, save, like, read_time, click_node, search) | Type of interaction |
| duration_seconds | INTEGER | NULL | Duration (for read_time) |
| context | JSONB | NULL | Additional context |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Interaction time |

**Indexes:**
- INDEX on user_id
- INDEX on paper_id
- INDEX on interaction_type
- INDEX on created_at
- **GIN INDEX** on context (JSON queries)

**Interaction Types:**
- **view:** User viewed paper details
- **click:** User clicked on paper card
- **save:** User saved paper to library
- **like:** User liked paper
- **read_time:** User spent time reading (duration_seconds set)
- **click_node:** User clicked paper in citation graph
- **search:** User performed search (paper appeared in results)

**Context Examples:**
```json
// Search interaction
{"source": "search_results", "query": "protein folding", "position": 3}

// Graph interaction
{"source": "citation_graph", "source_paper_id": "arxiv:2401.001"}

// Cluster interaction
{"source": "home_cluster", "cluster_id": 1}
```

**Sample Data:**
```sql
INSERT INTO user_interactions (user_id, paper_id, interaction_type, duration_seconds, context)
VALUES (
  1,
  'arxiv:2401.12345',
  'read_time',
  180,
  '{"source": "search_results", "query": "alphafold"}'::jsonb
);
```

---

#### Table 7: `user_saved_papers`

**Purpose:** Papers saved to user's library

**Columns:**

| Column | Type | Constraints | Purpose |
|--------|------|-------------|---------|
| user_id | INTEGER | PRIMARY KEY (composite), FK → users CASCADE | User identifier |
| paper_id | VARCHAR(100) | PRIMARY KEY (composite), FK → papers CASCADE | Paper identifier |
| saved_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Save timestamp |
| notes | TEXT | NULL | User's notes about paper |

**Indexes:**
- Composite PRIMARY KEY (user_id, paper_id)
- INDEX on user_id
- INDEX on saved_at (for recent saves)

**Sample Data:**
```sql
INSERT INTO user_saved_papers (user_id, paper_id, notes)
VALUES (1, 'arxiv:2401.12345', 'Important for chapter 3 of my thesis');
```

---

#### Table 8: `user_liked_papers`

**Purpose:** Papers liked by user (simpler than saved, no notes)

**Columns:**

| Column | Type | Constraints | Purpose |
|--------|------|-------------|---------|
| user_id | INTEGER | PRIMARY KEY (composite), FK → users CASCADE | User identifier |
| paper_id | VARCHAR(100) | PRIMARY KEY (composite), FK → papers CASCADE | Paper identifier |
| liked_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Like timestamp |

**Difference from Saved:**
- Liked = Quick bookmark/upvote
- Saved = Added to library with optional notes

**Sample Data:**
```sql
INSERT INTO user_liked_papers (user_id, paper_id)
VALUES (1, 'arxiv:2401.12345');
```

---

#### Table 9: `paper_clusters`

**Purpose:** Store thematic clusters for user home page

**Columns:**

| Column | Type | Constraints | Purpose |
|--------|------|-------------|---------|
| cluster_id | INTEGER | PRIMARY KEY, AUTO_INCREMENT | Cluster identifier |
| user_id | INTEGER | FK → users CASCADE | User who owns cluster |
| cluster_name | VARCHAR(255) | NOT NULL | Cluster theme name (LLM-generated) |
| theme_description | TEXT | NULL | Theme description |
| domain | VARCHAR(50) | CHECK (healthcare, fintech, quantum_computing) | Domain |
| paper_count | INTEGER | DEFAULT 0 | Number of papers |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Creation time |
| expires_at | TIMESTAMP | NULL | Cache expiration |

**Indexes:**
- INDEX on user_id
- INDEX on expires_at (for cache invalidation)

**How Clusters Are Generated:**
1. User registers → Celery task triggered
2. Retrieve 100 candidate papers (semantic search)
3. K-means clustering (k=3)
4. LLM generates theme names
5. Store in this table
6. Set expires_at = now() + 24 hours

**Sample Data:**
```sql
INSERT INTO paper_clusters (user_id, cluster_name, theme_description, domain, paper_count)
VALUES (
  1,
  'AI-Driven Protein Structure Prediction',
  'Machine learning approaches for predicting protein structures',
  'healthcare',
  12
);
```

---

#### Table 10: `cluster_papers`

**Purpose:** Many-to-many relationship between clusters and papers with positioning

**Columns:**

| Column | Type | Constraints | Purpose |
|--------|------|-------------|---------|
| cluster_id | INTEGER | PRIMARY KEY (composite), FK → paper_clusters CASCADE | Cluster ID |
| paper_id | VARCHAR(100) | PRIMARY KEY (composite), FK → papers CASCADE | Paper ID |
| centrality_score | FLOAT | NULL | How central paper is to cluster (0-1) |
| is_reference_paper | BOOLEAN | DEFAULT FALSE | Is this the cluster's reference paper |
| similarity_to_reference | FLOAT | NULL | Similarity to reference paper |
| position_x | FLOAT | NULL | X coordinate for graph layout |
| position_y | FLOAT | NULL | Y coordinate for graph layout |

**Indexes:**
- Composite PRIMARY KEY
- INDEX on cluster_id
- INDEX on paper_id
- INDEX on is_reference_paper (one per cluster)

**Position Coordinates:**
- Pre-computed using force-directed layout algorithm
- Stored to avoid recalculating on every page load
- Range: 0-500 for both x and y

**Sample Data:**
```sql
INSERT INTO cluster_papers (
  cluster_id, paper_id, centrality_score, is_reference_paper, 
  similarity_to_reference, position_x, position_y
)
VALUES (
  1, 'arxiv:2401.12345', 0.95, TRUE, 1.0, 250.0, 200.0
);
```

---

#### Table 11: `rate_limits`

**Purpose:** API rate limiting per user per endpoint

**Columns:**

| Column | Type | Constraints | Purpose |
|--------|------|-------------|---------|
| user_id | INTEGER | PRIMARY KEY (composite), FK → users CASCADE | User ID |
| endpoint | VARCHAR(255) | PRIMARY KEY (composite) | API endpoint |
| request_count | INTEGER | DEFAULT 0 | Requests in current window |
| window_start | TIMESTAMP | PRIMARY KEY (composite) | Window start time |

**Indexes:**
- Composite PRIMARY KEY
- INDEX on window_start

**How It Works:**
```python
# Check rate limit
key = (user_id=123, endpoint='/api/v1/search', window_start='2025-11-12 10:00:00')
current_count = SELECT request_count WHERE key
if current_count >= 100:
    raise RateLimitError()

# Increment
UPDATE rate_limits SET request_count = request_count + 1 WHERE key
```

---

#### Table 12: `system_metrics`

**Purpose:** Store application metrics for monitoring

**Columns:**

| Column | Type | Constraints | Purpose |
|--------|------|-------------|---------|
| metric_id | INTEGER | PRIMARY KEY, AUTO_INCREMENT | Metric ID |
| metric_name | VARCHAR(100) | NOT NULL | Metric name |
| metric_value | FLOAT | NOT NULL | Metric value |
| metadata | JSONB | NULL | Additional metadata |
| recorded_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Record time |

**Indexes:**
- INDEX on metric_name
- INDEX on recorded_at

**Metric Examples:**
```sql
-- API latency
INSERT INTO system_metrics (metric_name, metric_value, metadata)
VALUES ('api.search.latency_ms', 245, '{"endpoint": "/search"}'::jsonb);

-- Embedding generation time
INSERT INTO system_metrics (metric_name, metric_value)
VALUES ('embedding.generation_seconds', 1.2);
```

---

### 2.3 PostgreSQL Relationships Diagram

```
users (1) ←→ (1) user_domains
  ↓ (1:N)
  ├→ user_interests
  ├→ user_profile_embeddings (1:1)
  ├→ user_interactions
  ├→ user_saved_papers (N:M with papers)
  ├→ user_liked_papers (N:M with papers)
  ├→ paper_clusters
  └→ rate_limits

papers (1) ←→ (N) user_interactions
  ↓ (N:M)
  ├→ user_saved_papers
  ├→ user_liked_papers
  └→ cluster_papers

paper_clusters (1) ←→ (N) cluster_papers
```

---

## 3. Neo4j Schema

### 3.1 Database Information

- **Database Name:** neo4j (default)
- **User:** neo4j
- **Password:** password
- **Port (Browser):** 7474
- **Port (Bolt):** 7687
- **Access:** http://localhost:7474

---

### 3.2 Node Types

#### Node: `Paper`

**Properties:**

| Property | Type | Required | Purpose |
|----------|------|----------|---------|
| paper_id | STRING | YES (UNIQUE) | Primary identifier |
| title | STRING | YES | Paper title |
| year | INTEGER | YES | Publication year |
| domain | STRING | YES | Research domain |
| citation_count | INTEGER | NO | Number of citations |

**Constraint:**
```cypher
CREATE CONSTRAINT paper_id_unique IF NOT EXISTS
FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE;
```

**Indexes:**
```cypher
CREATE INDEX paper_id_index FOR (p:Paper) ON (p.paper_id);
CREATE INDEX paper_domain_index FOR (p:Paper) ON (p.domain);
CREATE INDEX paper_year_index FOR (p:Paper) ON (p.year);
```

**Sample Node:**
```cypher
CREATE (p:Paper {
  paper_id: 'arxiv:2401.12345',
  title: 'AlphaFold: Improved protein structure prediction',
  year: 2021,
  domain: 'healthcare',
  citation_count: 9432
})
```

---

#### Node: `User`

**Properties:**

| Property | Type | Required | Purpose |
|----------|------|----------|---------|
| user_id | INTEGER | YES (UNIQUE) | User identifier |
| domain | STRING | YES | User's domain |

**Why User Nodes Exist:**
- Track user interactions in graph (future feature)
- Recommendation graph: User → VIEWED → Paper
- Enables collaborative filtering

**Sample Node:**
```cypher
CREATE (u:User {
  user_id: 123,
  domain: 'healthcare'
})
```

---

### 3.3 Relationship Types

#### Relationship: `CITES`

**Direction:** (Paper)-[:CITES]->(Paper)

**Properties:**

| Property | Type | Purpose |
|----------|------|---------|
| citation_context | STRING | Optional: where/how paper is cited |

**Usage:**
```cypher
// Paper A cites Paper B
CREATE (a:Paper {paper_id: 'arxiv:001'})-[:CITES {citation_context: 'In introduction'}]->(b:Paper {paper_id: 'arxiv:002'})
```

---

#### Relationship: `CITED_BY`

**Direction:** (Paper)-[:CITED_BY]->(Paper)

**Purpose:** Reverse citation for faster queries

**Why It Exists:**
- Finding citing papers is common query
- Bidirectional relationships improve query performance

```cypher
// Automatically create reverse relationships
MATCH (p1:Paper)-[c:CITES]->(p2:Paper)
WHERE NOT exists((p2)-[:CITED_BY]->(p1))
CREATE (p2)-[:CITED_BY]->(p1)
```

---

#### Relationship: `SIMILAR_TO`

**Direction:** (Paper)-[:SIMILAR_TO]->(Paper)

**Properties:**

| Property | Type | Purpose |
|----------|------|---------|
| similarity_score | FLOAT | Cosine similarity (0.0-1.0) |

**Purpose:** Store semantic similarity for papers without direct citations

**Created When:** Embedding similarity > 0.80

---

### 3.4 Common Neo4j Queries

**Query 1: Find papers cited by a paper**
```cypher
MATCH (ref:Paper {paper_id: $paper_id})-[:CITES]->(cited:Paper)
RETURN cited
LIMIT 25
```

**Query 2: Find papers citing a paper**
```cypher
MATCH (citing:Paper)-[:CITES]->(ref:Paper {paper_id: $paper_id})
RETURN citing
LIMIT 25
```

**Query 3: Find co-cited papers**
```cypher
MATCH (ref:Paper {paper_id: $paper_id})<-[:CITES]-(citing)-[:CITES]->(co_cited:Paper)
WHERE co_cited.paper_id <> $paper_id
RETURN co_cited, COUNT(citing) as co_occurrence_count
ORDER BY co_occurrence_count DESC
LIMIT 25
```

**Query 4: Citation path between two papers**
```cypher
MATCH path = shortestPath(
  (p1:Paper {paper_id: $id1})-[:CITES*1..3]-(p2:Paper {paper_id: $id2})
)
RETURN path
```

---

## 4. Weaviate Schema

### 4.1 Database Information

- **Version:** 1.22.4
- **Port:** 8080
- **Access:** http://localhost:8080
- **Authentication:** Anonymous (development)

---

### 4.2 Collection: Paper

**Class Name:** Paper

**Vector Configuration:**
- **Dimensions:** 768 (SPECTER embeddings)
- **Distance Metric:** cosine
- **Algorithm:** HNSW (Hierarchical Navigable Small World)
- **Parameters:**
  - ef: 64 (query-time accuracy)
  - efConstruction: 128 (index-time accuracy)
  - maxConnections: 32 (graph connectivity)

**Properties:**

| Property | Data Type | Indexed | Searchable | Purpose |
|----------|-----------|---------|------------|---------|
| paper_id | text | YES | NO | Primary identifier |
| title | text | NO | YES | Paper title (for hybrid search) |
| abstract | text | NO | YES | Paper abstract |
| summary | text | NO | YES | AI-generated summary |
| domain | text | YES | NO | Filter by domain |
| year | int | YES | NO | Filter by year |
| citation_count | int | YES | NO | Filter by popularity |
| authors | text[] | NO | YES | Author names |

**Vector Storage:**
- Each paper has one 768-dimensional embedding vector
- Generated from: title + abstract + introduction
- Used for semantic similarity search

---

### 4.3 How Weaviate Search Works

**Query Flow:**
```python
# 1. User searches "antibody design machine learning"
query_text = "antibody design machine learning"

# 2. Generate query embedding
query_embedding = specter_model.encode(query_text)
# → [0.123, -0.456, ..., 0.789] (768 dims)

# 3. Search Weaviate
results = weaviate.query.get("Paper", [
    "paper_id", "title", "abstract", "year"
]).with_near_vector({
    "vector": query_embedding,
    "certainty": 0.7  # Min similarity threshold
}).with_where({
    "path": ["domain"],
    "operator": "Equal",
    "valueText": "healthcare"
}).with_limit(20).do()

# 4. Returns papers ranked by cosine similarity
```

**Result Structure:**
```json
{
  "data": {
    "Get": {
      "Paper": [
        {
          "paper_id": "nature:2023:antibody_dl",
          "title": "Deep learning for antibody structure prediction",
          "abstract": "...",
          "year": 2023,
          "_additional": {
            "certainty": 0.89,
            "distance": 0.11
          }
        }
      ]
    }
  }
}
```

---

## 5. Redis Schema

### 5.1 Key-Value Store

Redis doesn't have a fixed schema. We use naming conventions:

### 5.2 Key Naming Patterns

| Pattern | TTL | Purpose | Value Type |
|---------|-----|---------|------------|
| `user:session:{user_id}` | 24h | JWT session data | JSON |
| `starter_kit:{user_id}` | 24h | Cached clusters for home page | JSON |
| `cluster:{cluster_id}` | 1h | Cluster details | JSON |
| `graph:{paper_id}` | 1h | Citation graph structure | JSON |
| `search:{hash}:{domain}` | 30min | Search results | JSON |
| `paper:meta:{paper_id}` | 24h | Paper metadata | JSON |
| `user:embedding:{user_id}` | 6h | User profile embedding | Binary |
| `rate_limit:{user_id}:{endpoint}` | 1min | Rate limit counter | Integer |

### 5.3 Example Data Structures

**User Session:**
```json
{
  "user_id": 123,
  "email": "user@example.com",
  "domain": "healthcare",
  "expires_at": "2025-11-09T10:00:00Z"
}
```

**Starter Kit:**
```json
{
  "user_id": 123,
  "clusters": [
    {
      "cluster_id": 1,
      "name": "AI-Driven Protein Structure",
      "papers": [...],
      "reference_paper": {...}
    }
  ],
  "generated_at": "2025-11-08T10:00:00Z"
}
```

**Graph Cache:**
```json
{
  "reference_paper": {...},
  "related_papers": [...],
  "edges": [...],
  "generated_at": "2025-11-08T10:30:00Z"
}
```

---

## 6. Data Population Flows

### 6.1 User Registration Flow

```
User submits registration form
    ↓
Backend receives POST /auth/register
    ↓
TRANSACTION START
    ├─> INSERT INTO users (email, password_hash, name)
    │   Returns: user_id = 123
    │
    ├─> INSERT INTO user_domains (user_id, domain)
    │   Values: (123, 'healthcare')
    │
    ├─> INSERT INTO user_interests (user_id, interest_keyword, source)
    │   For each interest: (123, 'NLP', 'manual')
    │
    └─> IF google_scholar_url provided:
        ├─> Fetch author papers from Semantic Scholar API
        ├─> Generate author_profile_embedding
        └─> INSERT INTO user_profile_embeddings
TRANSACTION COMMIT
    ↓
Trigger Celery task: generate_starter_kit(user_id=123)
    ↓
ASYNC (Celery Worker):
    ├─> Query Weaviate: semantic search based on interests
    ├─> K-means clustering (k=3)
    ├─> Generate theme names with LLM
    ├─> INSERT INTO paper_clusters (3 clusters)
    ├─> INSERT INTO cluster_papers (18-21 papers total)
    └─> Cache in Redis: starter_kit:123
    ↓
User receives response with access_token
User can now view home page with 3 clusters
```

---

### 6.2 Paper Ingestion Flow (Airflow DAG)

```
Airflow DAG triggers (daily at 2 AM)
    ↓
Step 1: Fetch papers from Semantic Scholar API
    ├─> Query: papers published in last 24 hours
    ├─> Domains: healthcare, fintech, quantum_computing
    └─> Returns: ~2000 papers/day
    ↓
Step 2: For each paper:
    ├─> Download PDF (if open access)
    ├─> Parse PDF → Extract text (PyMuPDF)
    ├─> Generate SPECTER embedding
    │   embedding = specter.encode(title + abstract + intro)
    │   → 768-dimensional vector
    │
    ├─> PostgreSQL: INSERT INTO papers (metadata)
    │
    ├─> Weaviate: Insert paper with embedding
    │   weaviate.insert(paper_id, embedding, metadata)
    │
    └─> Neo4j: 
        ├─> CREATE (p:Paper {paper_id, title, year, domain})
        └─> For each citation:
            CREATE (p)-[:CITES]->(cited:Paper)
    ↓
Step 3: Post-processing
    ├─> Update citation counts in PostgreSQL
    ├─> Create CITED_BY reverse relationships in Neo4j
    └─> Log metrics to system_metrics table
```

---

### 6.3 Search Flow

```
User types "antibody design" in search bar
    ↓
Frontend: GET /api/v1/search?q=antibody+design&domain=healthcare
    ↓
Backend SearchService:
    ├─> Step 1: Check Redis cache
    │   cache_key = "search:{hash(query)}:healthcare"
    │   IF cached: return cached_results
    │
    ├─> Step 2: Generate query embedding
    │   query_embedding = specter.encode("antibody design")
    │
    ├─> Step 3: Query Weaviate
    │   results = weaviate.search(
    │       vector=query_embedding,
    │       filters={domain: 'healthcare'},
    │       limit=20
    │   )
    │   Returns: [{paper_id, similarity_score}, ...]
    │
    ├─> Step 4: Enrich with PostgreSQL metadata
    │   SELECT * FROM papers WHERE paper_id IN (...)
    │
    ├─> Step 5: Calculate relevance scores
    │   relevance = 0.85 * semantic_sim + 0.15 * keyword_boost
    │
    ├─> Step 6: Cache results
    │   cache_set(cache_key, results, ttl=1800)
    │
    └─> Step 7: Track interaction
        INSERT INTO user_interactions (
            user_id, paper_id=NULL, interaction_type='search',
            context={'query': 'antibody design'}
        )
    ↓
Return results to frontend
```

---

### 6.4 Citation Graph Generation Flow

```
User clicks paper node in graph
    ↓
Frontend: GET /papers/arxiv:2401.001/graph?limit=25
    ↓
Backend GraphService:
    ├─> Step 1: Check Redis cache
    │   cache_key = "graph:arxiv:2401.001"
    │   IF cached: return cached_graph
    │
    ├─> Step 2: Get reference paper embedding from Weaviate
    │   ref_embedding = weaviate.get_embedding(paper_id)
    │
    ├─> Step 3: Parallel queries
    │   ├─> Neo4j: Get cited papers (MATCH -[:CITES]->)
    │   ├─> Neo4j: Get citing papers (MATCH <-[:CITES]-)
    │   ├─> Neo4j: Get co-cited papers
    │   └─> Weaviate: Semantic search (similar papers)
    │
    ├─> Step 4: Merge and deduplicate results
    │   all_papers = cited + citing + co_cited + similar
    │   unique_papers = deduplicate_by_paper_id(all_papers)
    │
    ├─> Step 5: Score and rank
    │   For each paper:
    │       score = 0.7 * semantic_sim + 0.3 * citation_relevance
    │   Select top 25 papers
    │
    ├─> Step 6: Build graph structure
    │   nodes = [reference_paper] + related_papers
    │   edges = calculate_edges(nodes)  # From Neo4j + similarity
    │
    ├─> Step 7: Compute layout (NetworkX force-directed)
    │   positions = spring_layout(graph)
    │   Add position_x, position_y to each node
    │
    ├─> Step 8: Cache graph
    │   cache_set(cache_key, graph, ttl=3600)
    │
    └─> Step 9: Track interaction
        INSERT INTO user_interactions (
            interaction_type='click_node',
            context={'source_paper_id': 'previous_paper'}
        )
    ↓
Return graph JSON to frontend
    ↓
Cytoscape.js renders interactive visualization
```

---

### 6.5 User Profile Update Flow

```
User saves 10th paper
    ↓
INSERT INTO user_saved_papers (user_id, paper_id)
INSERT INTO user_interactions (interaction_type='save')
    ↓
Check interaction count since last profile update
SELECT COUNT(*) FROM user_interactions 
WHERE user_id=123 AND created_at > last_profile_update
    ↓
IF count >= 10:
    Trigger Celery task: update_user_profile_embedding(user_id=123)
    ↓
    ASYNC (Celery Worker):
        ├─> Fetch saved papers (last 30 days)
        │   SELECT paper_id FROM user_saved_papers 
        │   WHERE user_id=123 AND saved_at > NOW() - INTERVAL '30 days'
        │
        ├─> Fetch liked papers (last 30 days)
        │
        ├─> Fetch long-viewed papers (read_time > 3 min)
        │
        ├─> Get embeddings from Weaviate
        │   embeddings = [get_embedding(p) for p in papers]
        │
        ├─> Calculate weighted average
        │   new_profile_emb = (
        │       0.5 * avg(saved_embeddings) +
        │       0.3 * avg(liked_embeddings) +
        │       0.2 * avg(viewed_embeddings)
        │   )
        │
        ├─> UPDATE user_profile_embeddings
        │   SET embedding_vector = new_profile_emb,
        │       last_updated = NOW(),
        │       interaction_count = 0
        │
        └─> DELETE Redis cache: starter_kit:123
            (Forces regeneration on next home page visit)
```

---

## 7. Database Relationships

### 7.1 Cross-Database Relationships

**paper_id is the connecting key across all databases:**

```
PostgreSQL.papers.paper_id (VARCHAR)
    ↓
    ├─> Weaviate.Paper.paper_id (text property)
    │   - Same embedding vector
    │
    ├─> Neo4j.Paper.paper_id (STRING property)
    │   - Same paper in citation graph
    │
    └─> Redis: "graph:{paper_id}", "paper:meta:{paper_id}"
        - Cached data for this paper
```

**user_id connects user data:**

```
PostgreSQL.users.user_id (INTEGER)
    ↓
    ├─> user_domains, user_interests, user_profile_embeddings
    ├─> user_interactions, user_saved_papers, user_liked_papers
    ├─> paper_clusters
    │
    ├─> Neo4j.User.user_id (future)
    │
    └─> Redis: "user:session:{user_id}", "starter_kit:{user_id}"
```

---

### 7.2 Data Consistency Strategy

**Eventual Consistency:**
- Paper added to PostgreSQL first (source of truth)
- Asynchronously added to Weaviate and Neo4j
- If Weaviate/Neo4j fail, retry with exponential backoff
- Background job reconciles inconsistencies

**Transaction Boundaries:**
- PostgreSQL: ACID transactions for related tables
- Neo4j: Transactions for citation creation
- Weaviate: Individual inserts (batch mode for bulk)
- Redis: Fire-and-forget (cache can be regenerated)

---

## 8. Setup & Migration Guide

### 8.1 Initial Setup (Fresh Installation)

**Prerequisites:**
- Docker Desktop installed
- Python 3.11 installed

**Steps:**

```bash
# 1. Clone repository
git clone <repository-url>
cd ModelPipeline

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your settings

# 5. Start databases
docker-compose up -d

# Wait for databases to be ready
sleep 30

# 6. Run PostgreSQL migrations
cd citeconnect-backend
alembic upgrade head

# 7. Initialize Neo4j schema
./scripts/init_neo4j.sh

# 8. Seed sample data (optional)
./scripts/seed_neo4j.sh

# 9. Verify setup
python verify_setup.py
python test_db_connectivity.py

# 10. Start application
uvicorn app.main:app --reload
```

---

### 8.2 Database Migration Commands

**Check current migration:**
```bash
alembic current
```

**Create new migration:**
```bash
alembic revision -m "description of changes"
```

**Apply migrations:**
```bash
alembic upgrade head
```

**Rollback one migration:**
```bash
alembic downgrade -1
```

**Rollback all:**
```bash
alembic downgrade base
```

**View migration history:**
```bash
alembic history
```

---

## 9. Troubleshooting Guide

### 9.1 PostgreSQL Issues

#### Error: "role 'citeconnect' does not exist"

**Symptoms:**
```
psycopg2.OperationalError: FATAL: role "citeconnect" does not exist
```

**Causes:**
1. PostgreSQL container not initialized with correct user
2. Connecting to wrong PostgreSQL instance (local vs Docker)

**Solutions:**

**Solution 1: Check which PostgreSQL is responding**
```bash
# Check what's on port 5432
lsof -i :5432

# Should only show Docker, not local Postgres.app
```

**Solution 2: Recreate PostgreSQL container**
```bash
docker-compose down
docker volume rm modelpipeline_postgres_data
docker-compose up -d postgres
sleep 15
```

**Solution 3: Create user manually**
```bash
docker exec -it citeconnect-postgres psql -U postgres -c \
  "CREATE USER citeconnect WITH PASSWORD 'password' SUPERUSER;"
docker exec -it citeconnect-postgres psql -U postgres -c \
  "CREATE DATABASE citeconnect OWNER citeconnect;"
```

---

#### Error: "database 'citeconnect' does not exist"

**Solutions:**
```bash
# Create database
docker exec -it citeconnect-postgres psql -U citeconnect -c \
  "CREATE DATABASE citeconnect;"

# Verify
docker exec -it citeconnect-postgres psql -U citeconnect -c "\l"
```

---

#### Error: Alembic can't parse URL

**Symptoms:**
```
Could not parse SQLAlchemy URL from string 'jdbc:postgresql://...'
```

**Cause:** Wrong DATABASE_URL format in .env

**Solution:**
```bash
# Fix .env file
DATABASE_URL=postgresql://citeconnect:password@127.0.0.1:5432/citeconnect

# NOT:
# DATABASE_URL=jdbc:postgresql://... (Java format)
# DATABASE_URL=postgresql://localhost:5432/... (IPv6 issues)
```

---

#### Error: Connection timeout

**Solution:**
```bash
# Increase connection timeout in .env
DB_POOL_SIZE=5  # Reduce pool size
DB_MAX_OVERFLOW=10

# Or restart PostgreSQL
docker-compose restart postgres
```

---

### 9.2 Neo4j Issues

#### Error: "Connection refused" to Neo4j

**Causes:**
1. Neo4j still starting up (takes 20-30 seconds)
2. Wrong credentials

**Solutions:**

**Check status:**
```bash
docker-compose ps neo4j
# Should show "Up (healthy)"

# Check logs
docker-compose logs neo4j
```

**Wait longer:**
```bash
# Neo4j takes time to start
sleep 30
./scripts/init_neo4j.sh
```

**Verify credentials:**
```bash
# Test connection
docker exec -it citeconnect-neo4j cypher-shell -u neo4j -p password
```

---

#### Error: "the input device is not a TTY"

**Cause:** Using `-it` flags in scripts with heredoc

**Solution:** Remove `-it` or use separate command per statement

**Bad:**
```bash
docker exec -it container cypher-shell << 'EOF'
CREATE ...
EOF
```

**Good:**
```bash
docker exec container cypher-shell -u neo4j -p password "CREATE ..."
```

---

### 9.3 Weaviate Issues

#### Error: Weaviate not ready

**Solution:**
```bash
# Check status
docker-compose ps weaviate

# Check readiness
curl http://localhost:8080/v1/.well-known/ready

# Restart if needed
docker-compose restart weaviate
sleep 10
```

---

#### Error: Schema already exists

**Not an error!** Weaviate schema is idempotent. Running create_schema() multiple times is safe.

---

### 9.4 Redis Issues

#### Error: Connection refused

**Solution:**
```bash
# Start Redis
docker-compose up -d redis

# Test connection
docker exec -it citeconnect-redis redis-cli ping
# Should return: PONG
```

---

### 9.5 General Database Issues

#### Error: "Cannot connect to Docker daemon"

**Solution:**
```bash
# Start Docker Desktop
open -a Docker

# Wait for Docker to start
sleep 10

# Verify
docker ps
```

---

#### Error: Port already in use

**Find what's using the port:**
```bash
lsof -i :5432  # PostgreSQL
lsof -i :6379  # Redis
lsof -i :7687  # Neo4j
lsof -i :8080  # Weaviate
```

**Solution:** Stop the conflicting process or use different port in docker-compose.yml

---

## 10. Inspection & Monitoring

### 10.1 PostgreSQL Inspection

**Connect to database:**
```bash
docker exec -it citeconnect-postgres psql -U citeconnect -d citeconnect
```

**Useful commands:**
```sql
-- List tables
\dt

-- Describe table
\d users
\d papers

-- List indexes
\di

-- View table data
SELECT * FROM users LIMIT 10;

-- Count records
SELECT COUNT(*) FROM papers;

-- Complex query
SELECT 
    u.email,
    COUNT(DISTINCT sp.paper_id) as saved_papers,
    COUNT(DISTINCT lp.paper_id) as liked_papers
FROM users u
LEFT JOIN user_saved_papers sp ON u.user_id = sp.user_id
LEFT JOIN user_liked_papers lp ON u.user_id = lp.user_id
GROUP BY u.email;
```

---

### 10.2 Neo4j Inspection

**Neo4j Browser:** http://localhost:7474

**Useful queries:**
```cypher
// Count nodes
MATCH (n) RETURN count(n);

// Count papers
MATCH (p:Paper) RETURN count(p);

// Count citations
MATCH ()-[r:CITES]->() RETURN count(r);

// View all papers
MATCH (p:Paper) RETURN p LIMIT 25;

// Citation network around a paper
MATCH path = (ref:Paper {paper_id: 'arxiv:2401.001'})-[:CITES*1..2]-(related)
RETURN path;

// Most cited papers
MATCH (p:Paper)
RETURN p.paper_id, p.title, p.citation_count
ORDER BY p.citation_count DESC
LIMIT 10;

// Papers by domain
MATCH (p:Paper)
RETURN p.domain, count(p) as count
GROUP BY p.domain;
```

---

### 10.3 Weaviate Inspection

**API endpoint:** http://localhost:8080/v1

**Useful commands:**
```bash
# Get schema
curl http://localhost:8080/v1/schema

# Count objects
curl http://localhost:8080/v1/objects

# Query papers
curl -X POST http://localhost:8080/v1/graphql \
  -H "Content-Type: application/json" \
  -d '{
    "query": "{Get{Paper{paper_id title}}}"
  }'
```

---

### 10.4 Redis Inspection

**Connect to Redis:**
```bash
docker exec -it citeconnect-redis redis-cli
```

**Useful commands:**
```bash
# List all keys
KEYS *

# Get value
GET user:session:123

# Check TTL
TTL graph:arxiv:2401.001

# Count keys by pattern
KEYS starter_kit:* | wc -l

# Delete all cache (DANGEROUS!)
FLUSHDB
```

---

## 11. Data Verification Checklist

After setup, verify each database:

### PostgreSQL ✅
- [ ] 13 tables exist (`\dt`)
- [ ] Can insert user (`INSERT INTO users ...`)
- [ ] Can query papers (`SELECT * FROM papers LIMIT 1`)
- [ ] Indexes exist (`\di`)
- [ ] Foreign keys work (try inserting with invalid FK)

### Neo4j ✅
- [ ] Constraints exist (`SHOW CONSTRAINTS`)
- [ ] Indexes exist (`SHOW INDEXES`)
- [ ] Can create paper node
- [ ] Can create citation relationship
- [ ] Can query citation network

### Weaviate ✅
- [ ] Is ready (`curl http://localhost:8080/v1/.well-known/ready`)
- [ ] Schema exists (`curl http://localhost:8080/v1/schema`)
- [ ] Can insert object with vector
- [ ] Can search by vector

### Redis ✅
- [ ] Responds to PING
- [ ] Can SET and GET
- [ ] TTL works correctly

---

## 12. Database Backup & Recovery

### PostgreSQL Backup

```bash
# Backup database
docker exec citeconnect-postgres pg_dump -U citeconnect citeconnect > backup.sql

# Restore
docker exec -i citeconnect-postgres psql -U citeconnect -d citeconnect < backup.sql
```

### Neo4j Backup

```bash
# Backup (export to CSV)
docker exec citeconnect-neo4j cypher-shell -u neo4j -p password \
  "CALL apoc.export.csv.all('backup.csv', {})"

# Or use neo4j-admin dump (requires stop)
docker-compose stop neo4j
docker exec citeconnect-neo4j neo4j-admin dump --database=neo4j --to=/backups/neo4j.dump
docker-compose start neo4j
```

---

## 13. Performance Optimization

### PostgreSQL

**Slow queries:**
```sql
-- Enable query logging
ALTER SYSTEM SET log_min_duration_statement = 100; -- Log queries > 100ms

-- View slow queries
SELECT * FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;
```

**Missing index:**
```sql
-- Analyze query plan
EXPLAIN ANALYZE SELECT * FROM papers WHERE domain = 'healthcare';

-- Add index if needed
CREATE INDEX idx_papers_domain ON papers(domain);
```

---

### Neo4j

**Slow queries:**
```cypher
// Profile query
PROFILE MATCH (p:Paper {paper_id: $id})-[:CITES]->(cited) RETURN cited;

// Add index if needed
CREATE INDEX paper_id_lookup FOR (p:Paper) ON (p.paper_id);
```

---

### Weaviate

**Tune HNSW parameters:**
```python
# For higher recall (more accurate but slower)
"ef": 128  # Increase from 64

# For faster queries (less accurate)
"ef": 32  # Decrease from 64
```

---

## 14. Common Error Messages & Solutions

| Error | Database | Solution |
|-------|----------|----------|
| role "citeconnect" does not exist | PostgreSQL | Stop local Postgres.app, use only Docker |
| connection to server at "localhost" (::1) | PostgreSQL | Use 127.0.0.1 instead of localhost |
| jdbc:postgresql URL | Alembic | Fix .env: use postgresql:// not jdbc: |
| Module 'pydantic' not found | Alembic | Use venv's alembic, not system alembic |
| Connection refused | Neo4j | Wait 30 seconds for startup |
| the input device is not a TTY | Neo4j | Remove -it flags from scripts |
| Weaviate schema not found | Weaviate | Run create_schema() on app startup |
| Redis WRONGTYPE | Redis | Key has wrong type, delete and recreate |

---

## 15. Quick Reference Commands

### Start Everything
```bash
cd ~/Documents/GitHub/ModelPipeline
docker-compose up -d
sleep 30
cd citeconnect-backend
uvicorn app.main:app --reload
```

### Stop Everything
```bash
cd ~/Documents/GitHub/ModelPipeline
docker-compose down
# Add -v to also remove volumes
```

### Reset Databases
```bash
docker-compose down -v
docker-compose up -d
cd citeconnect-backend
alembic upgrade head
./scripts/init_neo4j.sh
./scripts/seed_neo4j.sh
```

### View Logs
```bash
docker-compose logs -f postgres
docker-compose logs -f neo4j
docker-compose logs -f weaviate
docker-compose logs -f redis
```

---

**Document Version:** 1.0  
**Completeness:** Covers all implemented database schemas  
**Next Update:** After Weaviate seeding and data pipeline implementation
