# CiteConnect: Low-Level Design Document

**Version:** 1.0  
**Date:** November 2025  
**Authors:** Dennis Jose, Abhinav Aditya, Anusha Srinivasan, Dhiksha Mathanagopal, Sahil Mohanty  
**Target Deployment:** Google Cloud Platform  
**Presentation Date:** December 2025 @ Google HQ

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Project Structure](#4-project-structure)
5. [Database Schema Design](#5-database-schema-design)
6. [API Specifications](#6-api-specifications)
7. [Component Design](#7-component-design)
8. [Algorithm Implementations](#8-algorithm-implementations)
9. [Data Flow Diagrams](#9-data-flow-diagrams)
10. [Security & Authentication](#10-security--authentication)
11. [Performance & Caching Strategy](#11-performance--caching-strategy)
12. [Error Handling & Monitoring](#12-error-handling--monitoring)
13. [Deployment Configuration](#13-deployment-configuration)
14. [Implementation Guidelines](#14-implementation-guidelines)

---

## 1. Executive Summary

### 1.1 Project Overview

CiteConnect is an AI-powered research paper recommendation system that combines semantic search, citation graph analysis, and personalized recommendations. The system processes academic papers from Healthcare, Fintech, and Quantum Computing domains.

### 1.2 Core Features

1. **User Profile Creation**: Domain selection, keyword input, Google Scholar import, paper upload
2. **Home Page Clustering**: 3 thematic clusters of papers displayed as cards or network graphs
3. **Dynamic Citation Network**: Interactive graph that updates based on selected paper
4. **Semantic Search**: SPECTER2-powered search with context-aware results
5. **Personalization**: Learning from user interactions (saves, likes, reading time)

### 1.3 Technical Goals

- **Precision@10**: ≥ 0.80 (80%+ relevance in top 10 results)
- **Recall@10**: ≥ 0.75
- **Query Latency (p95)**: < 2 seconds
- **System Availability**: 99.5%
- **Monthly Cost**: < $350

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                            │
│              Browser (React SPA + Cytoscape.js)                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓ HTTPS/REST
┌─────────────────────────────────────────────────────────────────┐
│                      API GATEWAY LAYER                          │
│         FastAPI (Python 3.11) - Auth + Rate Limiting            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     MICROSERVICES LAYER                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ User Service │  │Search Service│  │ Graph Service│          │
│  │   (FastAPI)  │  │   (FastAPI)  │  │   (FastAPI)  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │Cluster Svc   │  │Recommend Svc │  │Analytics Svc │          │
│  │   (FastAPI)  │  │   (FastAPI)  │  │   (FastAPI)  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      ASYNC TASK LAYER                           │
│              Celery Workers + Redis Queue                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ PostgreSQL   │  │  Weaviate    │  │    Neo4j     │          │
│  │  (Primary)   │  │ (Embeddings) │  │  (Citations) │          │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│  ┌──────────────┐  ┌──────────────┐                            │
│  │    Redis     │  │     GCS      │                            │
│  │   (Cache)    │  │ (PDF Store)  │                            │
│  └──────────────┘  └──────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **API Framework** | FastAPI | Async support, automatic docs, type hints, fast development |
| **Frontend** | React 18 + TypeScript | Industry standard, large ecosystem, graph library support |
| **Graph Viz** | Cytoscape.js | Better performance for network graphs, built-in layouts |
| **State Management** | Zustand | Simpler than Redux, sufficient for project scale |
| **Embedding Model** | SPECTER2 (allenai/specter2) | Fine-tuned on scientific papers with citation graphs |
| **Vector DB** | Weaviate | Open-source, good performance, easy integration |
| **Graph DB** | Neo4j | Industry standard for citation networks |
| **Task Queue** | Celery + Redis | Proven solution, good monitoring |
| **Deployment** | GCP Cloud Run + GKE | Auto-scaling, serverless for API, K8s for stateful services |

---

## 3. Technology Stack

### 3.1 Frontend Stack

```yaml
Core:
  - React: 18.2.0
  - TypeScript: 5.0+
  - Vite: 4.4+ (Build tool)

UI/Styling:
  - TailwindCSS: 3.3+
  - shadcn/ui: Latest (Component library)
  - Lucide React: Latest (Icons)

Graph Visualization:
  - Cytoscape.js: 3.26+
  - D3.js: 7.8+ (For custom visualizations)

State & Data:
  - Zustand: 4.4+ (State management)
  - React Query: 4.0+ (Server state, caching)
  - Axios: 1.5+ (HTTP client)

Forms & Validation:
  - React Hook Form: 7.45+
  - Zod: 3.22+ (Schema validation)

Charts:
  - Recharts: 2.8+ (Dashboard analytics)

Routing:
  - React Router: 6.16+
```

### 3.2 Backend Stack

```yaml
Core:
  - Python: 3.11+
  - FastAPI: 0.104+
  - Pydantic: 2.4+ (Data validation)
  - Uvicorn: 0.24+ (ASGI server)

ML/Embeddings:
  - sentence-transformers: 2.2+
  - transformers: 4.35+ (HuggingFace)
  - torch: 2.1+ (PyTorch)
  - scikit-learn: 1.3+ (Clustering, metrics)
  - numpy: 1.24+

Task Queue:
  - Celery: 5.3+
  - Redis: 4.6+ (Backend for Celery)

Database Clients:
  - psycopg2: 2.9+ (PostgreSQL)
  - weaviate-client: 3.24+
  - neo4j: 5.14+
  - redis: 5.0+

API Clients:
  - httpx: 0.25+ (Async HTTP)
  - requests: 2.31+

Authentication:
  - python-jose: 3.3+ (JWT)
  - passlib: 1.7+ (Password hashing)
  - bcrypt: 4.0+

Utilities:
  - python-dotenv: 1.0+ (Environment variables)
  - pydantic-settings: 2.0+
```

### 3.3 Database Versions

```yaml
PostgreSQL: 15.4
Neo4j: 5.13 Community Edition
Weaviate: 1.22+
Redis: 7.2+
```

### 3.4 DevOps & Infrastructure

```yaml
Containerization:
  - Docker: 24.0+
  - Docker Compose: 2.22+

CI/CD:
  - GitHub Actions

Cloud (GCP):
  - Cloud Run (API services)
  - GKE (Kubernetes for stateful services)
  - Cloud Storage (PDF storage)
  - Cloud SQL (PostgreSQL managed)
  - Cloud Memorystore (Redis managed)

Monitoring:
  - Prometheus: 2.47+
  - Grafana: 10.1+
  - Cloud Logging (GCP native)

Orchestration:
  - Apache Airflow: 2.7+ (Data pipeline)
```

---

## 4. Project Structure

### 4.1 Repository Structure

```
citeconnect/
├── README.md
├── docker-compose.yml
├── .env.example
├── .gitignore
│
├── backend/                          # Python FastAPI Backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                   # FastAPI application entry
│   │   ├── config.py                 # Configuration management
│   │   │
│   │   ├── api/                      # API routes
│   │   │   ├── __init__.py
│   │   │   ├── deps.py               # Dependencies (auth, db)
│   │   │   └── v1/                   # API version 1
│   │   │       ├── __init__.py
│   │   │       ├── auth.py           # /api/v1/auth/*
│   │   │       ├── users.py          # /api/v1/users/*
│   │   │       ├── papers.py         # /api/v1/papers/*
│   │   │       ├── search.py         # /api/v1/search/*
│   │   │       ├── clusters.py       # /api/v1/clusters/*
│   │   │       ├── graph.py          # /api/v1/graph/*
│   │   │       └── interactions.py   # /api/v1/interactions/*
│   │   │
│   │   ├── core/                     # Core functionality
│   │   │   ├── __init__.py
│   │   │   ├── security.py           # JWT, password hashing
│   │   │   ├── config.py             # Settings (Pydantic)
│   │   │   └── logging.py            # Logging setup
│   │   │
│   │   ├── models/                   # Pydantic models
│   │   │   ├── __init__.py
│   │   │   ├── user.py
│   │   │   ├── paper.py
│   │   │   ├── cluster.py
│   │   │   ├── interaction.py
│   │   │   └── graph.py
│   │   │
│   │   ├── schemas/                  # API request/response schemas
│   │   │   ├── __init__.py
│   │   │   ├── auth.py
│   │   │   ├── user.py
│   │   │   ├── paper.py
│   │   │   ├── search.py
│   │   │   └── cluster.py
│   │   │
│   │   ├── db/                       # Database connections
│   │   │   ├── __init__.py
│   │   │   ├── postgres.py           # PostgreSQL connection
│   │   │   ├── weaviate_client.py    # Weaviate connection
│   │   │   ├── neo4j_client.py       # Neo4j connection
│   │   │   └── redis_client.py       # Redis connection
│   │   │
│   │   ├── services/                 # Business logic
│   │   │   ├── __init__.py
│   │   │   ├── auth_service.py
│   │   │   ├── user_service.py
│   │   │   ├── paper_service.py
│   │   │   ├── search_service.py
│   │   │   ├── clustering_service.py
│   │   │   ├── graph_service.py
│   │   │   ├── recommendation_service.py
│   │   │   ├── embedding_service.py
│   │   │   ├── scholar_import_service.py
│   │   │   └── interaction_service.py
│   │   │
│   │   ├── tasks/                    # Celery tasks
│   │   │   ├── __init__.py
│   │   │   ├── celery_app.py         # Celery config
│   │   │   ├── starter_kit.py        # Generate starter kit
│   │   │   ├── clustering.py         # Cluster generation
│   │   │   ├── embeddings.py         # Batch embedding generation
│   │   │   └── user_profile.py       # Update user profiles
│   │   │
│   │   └── utils/                    # Utility functions
│   │       ├── __init__.py
│   │       ├── embedding.py          # SPECTER2 wrapper
│   │       ├── similarity.py         # Similarity calculations
│   │       ├── clustering.py         # K-means, layout algorithms
│   │       ├── pdf_parser.py         # PDF text extraction
│   │       └── validators.py         # Custom validators
│   │
│   ├── alembic/                      # Database migrations
│   │   ├── versions/
│   │   └── env.py
│   │
│   ├── tests/                        # Unit & integration tests
│   │   ├── __init__.py
│   │   ├── test_api/
│   │   ├── test_services/
│   │   └── test_utils/
│   │
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   ├── Dockerfile
│   ├── .env.example
│   └── pyproject.toml
│
├── frontend/                         # React Frontend
│   ├── public/
│   │   ├── index.html
│   │   └── favicon.ico
│   │
│   ├── src/
│   │   ├── main.tsx                  # Entry point
│   │   ├── App.tsx
│   │   ├── index.css
│   │   │
│   │   ├── components/               # Reusable components
│   │   │   ├── ui/                   # shadcn/ui components
│   │   │   │   ├── button.tsx
│   │   │   │   ├── card.tsx
│   │   │   │   ├── input.tsx
│   │   │   │   └── ...
│   │   │   │
│   │   │   ├── common/               # Common components
│   │   │   │   ├── Header.tsx
│   │   │   │   ├── Loading.tsx
│   │   │   │   ├── ErrorBoundary.tsx
│   │   │   │   └── ProtectedRoute.tsx
│   │   │   │
│   │   │   ├── auth/
│   │   │   │   ├── LoginForm.tsx
│   │   │   │   ├── SignupForm.tsx
│   │   │   │   └── DomainSelection.tsx
│   │   │   │
│   │   │   ├── home/
│   │   │   │   ├── ClusterCard.tsx
│   │   │   │   ├── ClusterContainer.tsx
│   │   │   │   ├── NetworkView.tsx
│   │   │   │   └── CardView.tsx
│   │   │   │
│   │   │   ├── graph/
│   │   │   │   ├── CytoscapeGraph.tsx
│   │   │   │   ├── GraphControls.tsx
│   │   │   │   └── NodeTooltip.tsx
│   │   │   │
│   │   │   ├── paper/
│   │   │   │   ├── PaperCard.tsx
│   │   │   │   ├── PaperDetail.tsx
│   │   │   │   ├── PaperMetadata.tsx
│   │   │   │   └── ActionButtons.tsx
│   │   │   │
│   │   │   └── search/
│   │   │       ├── SearchBar.tsx
│   │   │       └── SearchResults.tsx
│   │   │
│   │   ├── pages/                    # Page components
│   │   │   ├── LoginPage.tsx
│   │   │   ├── SignupPage.tsx
│   │   │   ├── HomePage.tsx
│   │   │   ├── PaperInteractionPage.tsx
│   │   │   ├── DashboardPage.tsx
│   │   │   └── NotFoundPage.tsx
│   │   │
│   │   ├── services/                 # API services
│   │   │   ├── api.ts                # Axios config
│   │   │   ├── authService.ts
│   │   │   ├── userService.ts
│   │   │   ├── paperService.ts
│   │   │   ├── searchService.ts
│   │   │   ├── clusterService.ts
│   │   │   └── graphService.ts
│   │   │
│   │   ├── store/                    # Zustand stores
│   │   │   ├── authStore.ts
│   │   │   ├── userStore.ts
│   │   │   ├── paperStore.ts
│   │   │   └── uiStore.ts
│   │   │
│   │   ├── hooks/                    # Custom React hooks
│   │   │   ├── useAuth.ts
│   │   │   ├── useDebounce.ts
│   │   │   ├── useGraph.ts
│   │   │   └── useIntersectionObserver.ts
│   │   │
│   │   ├── types/                    # TypeScript types
│   │   │   ├── user.ts
│   │   │   ├── paper.ts
│   │   │   ├── cluster.ts
│   │   │   ├── graph.ts
│   │   │   └── api.ts
│   │   │
│   │   ├── utils/                    # Utility functions
│   │   │   ├── formatters.ts
│   │   │   ├── validators.ts
│   │   │   └── constants.ts
│   │   │
│   │   └── styles/                   # Additional styles
│   │       └── globals.css
│   │
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── .env.example
│   └── Dockerfile
│
├── airflow/                          # Apache Airflow DAGs
│   ├── dags/
│   │   ├── data_ingestion_dag.py
│   │   ├── embedding_update_dag.py
│   │   └── user_profile_update_dag.py
│   └── config/
│       └── airflow.cfg
│
├── scripts/                          # Utility scripts
│   ├── init_databases.sh
│   ├── seed_data.py
│   └── load_specter_model.py
│
└── docs/                             # Documentation
    ├── API.md
    ├── DEPLOYMENT.md
    └── DEVELOPMENT.md
```

---

## 5. Database Schema Design

### 5.1 PostgreSQL Schema

#### 5.1.1 Users & Authentication

```sql
-- Users table
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    google_scholar_url VARCHAR(500),
    semantic_scholar_author_id VARCHAR(100)
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_semantic_scholar ON users(semantic_scholar_author_id);

-- User domains (single domain per user)
CREATE TABLE user_domains (
    user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    domain VARCHAR(50) NOT NULL CHECK (domain IN ('healthcare', 'fintech', 'quantum_computing')),
    selected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id)
);

-- User interests (keywords)
CREATE TABLE user_interests (
    interest_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    interest_keyword VARCHAR(100) NOT NULL,
    source VARCHAR(50) NOT NULL CHECK (source IN ('manual', 'google_scholar', 'inferred')),
    weight FLOAT DEFAULT 1.0 CHECK (weight >= 0.0 AND weight <= 1.0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_user_interests_user ON user_interests(user_id);

-- User profile embeddings (SPECTER2 768-dim)
CREATE TABLE user_profile_embeddings (
    user_id INTEGER PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
    embedding_vector FLOAT8[] NOT NULL, -- 768 dimensions
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    based_on_papers TEXT[], -- Array of paper_ids used to build embedding
    interaction_count INTEGER DEFAULT 0
);
```

#### 5.1.2 Papers & Metadata

```sql
-- Papers table (metadata only, embeddings in Weaviate)
CREATE TABLE papers (
    paper_id VARCHAR(100) PRIMARY KEY, -- e.g., "arxiv:2401.12345" or semantic_scholar_id
    title TEXT NOT NULL,
    authors TEXT[], -- Array of author names
    year INTEGER,
    venue VARCHAR(255),
    citation_count INTEGER DEFAULT 0,
    abstract TEXT,
    summary TEXT, -- AI-generated summary
    introduction TEXT, -- Extracted introduction section
    gcs_pdf_path VARCHAR(500), -- Google Cloud Storage path
    domain VARCHAR(50) CHECK (domain IN ('healthcare', 'fintech', 'quantum_computing')),
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_papers_domain ON papers(domain);
CREATE INDEX idx_papers_year ON papers(year);
CREATE INDEX idx_papers_citation_count ON papers(citation_count);
CREATE INDEX idx_papers_title_gin ON papers USING gin(to_tsvector('english', title));
CREATE INDEX idx_papers_abstract_gin ON papers USING gin(to_tsvector('english', abstract));
```

#### 5.1.3 User Interactions

```sql
-- User interactions (tracking)
CREATE TABLE user_interactions (
    interaction_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    paper_id VARCHAR(100) REFERENCES papers(paper_id) ON DELETE CASCADE,
    interaction_type VARCHAR(50) NOT NULL CHECK (
        interaction_type IN ('view', 'click', 'save', 'like', 'read_time', 'click_node', 'search')
    ),
    duration_seconds INTEGER, -- For read_time
    context JSONB, -- Flexible: {cluster_id, search_query, source_paper_id, etc.}
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_interactions_user ON user_interactions(user_id);
CREATE INDEX idx_interactions_paper ON user_interactions(paper_id);
CREATE INDEX idx_interactions_type ON user_interactions(interaction_type);
CREATE INDEX idx_interactions_created ON user_interactions(created_at);
CREATE INDEX idx_interactions_context_gin ON user_interactions USING gin(context);

-- User saved papers
CREATE TABLE user_saved_papers (
    user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    paper_id VARCHAR(100) REFERENCES papers(paper_id) ON DELETE CASCADE,
    saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,
    PRIMARY KEY (user_id, paper_id)
);

CREATE INDEX idx_saved_papers_user ON user_saved_papers(user_id);
CREATE INDEX idx_saved_papers_saved_at ON user_saved_papers(saved_at);

-- User liked papers
CREATE TABLE user_liked_papers (
    user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    paper_id VARCHAR(100) REFERENCES papers(paper_id) ON DELETE CASCADE,
    liked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, paper_id)
);

CREATE INDEX idx_liked_papers_user ON user_liked_papers(user_id);
```

#### 5.1.4 Clustering

```sql
-- Paper clusters (pre-computed for home page)
CREATE TABLE paper_clusters (
    cluster_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE, -- User-specific clusters
    cluster_name VARCHAR(255) NOT NULL,
    theme_description TEXT,
    domain VARCHAR(50) CHECK (domain IN ('healthcare', 'fintech', 'quantum_computing')),
    paper_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP -- For cache invalidation
);

CREATE INDEX idx_clusters_user ON paper_clusters(user_id);
CREATE INDEX idx_clusters_expires ON paper_clusters(expires_at);

-- Cluster-paper relationships
CREATE TABLE cluster_papers (
    cluster_id INTEGER REFERENCES paper_clusters(cluster_id) ON DELETE CASCADE,
    paper_id VARCHAR(100) REFERENCES papers(paper_id) ON DELETE CASCADE,
    centrality_score FLOAT, -- How central to cluster (0-1)
    is_reference_paper BOOLEAN DEFAULT FALSE, -- One per cluster
    similarity_to_reference FLOAT, -- For visualization (bubble size)
    position_x FLOAT, -- Pre-computed graph layout
    position_y FLOAT,
    PRIMARY KEY (cluster_id, paper_id)
);

CREATE INDEX idx_cluster_papers_cluster ON cluster_papers(cluster_id);
CREATE INDEX idx_cluster_papers_paper ON cluster_papers(paper_id);
CREATE INDEX idx_cluster_papers_reference ON cluster_papers(is_reference_paper);
```

#### 5.1.5 System Tables

```sql
-- API rate limiting
CREATE TABLE rate_limits (
    user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    endpoint VARCHAR(255) NOT NULL,
    request_count INTEGER DEFAULT 0,
    window_start TIMESTAMP NOT NULL,
    PRIMARY KEY (user_id, endpoint, window_start)
);

CREATE INDEX idx_rate_limits_window ON rate_limits(window_start);

-- System metrics
CREATE TABLE system_metrics (
    metric_id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    metadata JSONB,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_metrics_name ON system_metrics(metric_name);
CREATE INDEX idx_metrics_recorded ON system_metrics(recorded_at);
```

### 5.2 Weaviate Schema

```python
# Weaviate Collection Schema for Papers
{
    "class": "Paper",
    "description": "Research paper with SPECTER2 embeddings",
    "vectorizer": "none",  # We provide embeddings
    "properties": [
        {
            "name": "paper_id",
            "dataType": ["text"],
            "description": "Unique paper identifier",
            "indexFilterable": True,
            "indexSearchable": False
        },
        {
            "name": "title",
            "dataType": ["text"],
            "description": "Paper title",
            "indexFilterable": False,
            "indexSearchable": True
        },
        {
            "name": "abstract",
            "dataType": ["text"],
            "description": "Paper abstract",
            "indexFilterable": False,
            "indexSearchable": True
        },
        {
            "name": "summary",
            "dataType": ["text"],
            "description": "AI-generated summary",
            "indexFilterable": False,
            "indexSearchable": True
        },
        {
            "name": "domain",
            "dataType": ["text"],
            "description": "healthcare, fintech, or quantum_computing",
            "indexFilterable": True,
            "indexSearchable": False
        },
        {
            "name": "year",
            "dataType": ["int"],
            "description": "Publication year",
            "indexFilterable": True,
            "indexSearchable": False
        },
        {
            "name": "citation_count",
            "dataType": ["int"],
            "description": "Number of citations",
            "indexFilterable": True,
            "indexSearchable": False
        },
        {
            "name": "authors",
            "dataType": ["text[]"],
            "description": "List of authors",
            "indexFilterable": False,
            "indexSearchable": True
        }
    ],
    "vectorIndexConfig": {
        "distance": "cosine",  # Cosine similarity
        "ef": 64,              # HNSW parameter
        "efConstruction": 128,
        "maxConnections": 32
    }
}

# Note: Embeddings are 768-dimensional SPECTER2 vectors
# Stored implicitly with each object
```

### 5.3 Neo4j Schema

```cypher
-- Node definitions
// Paper node
CREATE CONSTRAINT paper_id_unique IF NOT EXISTS
FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE;

(:Paper {
    paper_id: string,        // Primary key
    title: string,
    year: int,
    domain: string,
    citation_count: int
})

// User node (for recommendation graph)
CREATE CONSTRAINT user_id_unique IF NOT EXISTS
FOR (u:User) REQUIRE u.user_id IS UNIQUE;

(:User {
    user_id: int,
    domain: string
})

-- Relationship definitions
// Citation relationships
(:Paper)-[:CITES {
    citation_context: string  // Optional: where/how cited
}]->(:Paper)

// Reverse citation (indexed for faster queries)
(:Paper)-[:CITED_BY]->(:Paper)

// Co-citation (papers cited together)
(:Paper)-[:CO_CITED_WITH {
    co_occurrence_count: int
}]->(:Paper)

// Semantic similarity (from SPECTER2)
(:Paper)-[:SIMILAR_TO {
    similarity_score: float  // 0.0 to 1.0
}]->(:Paper)

// User interactions (for recommendations)
(:User)-[:VIEWED {
    timestamp: datetime
}]->(:Paper)

(:User)-[:SAVED {
    timestamp: datetime
}]->(:Paper)

(:User)-[:LIKED {
    timestamp: datetime
}]->(:Paper)

-- Indexes for performance
CREATE INDEX paper_domain IF NOT EXISTS FOR (p:Paper) ON (p.domain);
CREATE INDEX paper_year IF NOT EXISTS FOR (p:Paper) ON (p.year);
CREATE INDEX user_domain IF NOT EXISTS FOR (u:User) ON (u.domain);
```

### 5.4 Redis Schema (Caching)

```
# Key naming conventions

# User sessions
user:session:{user_id} → JWT token data (TTL: 24 hours)

# Starter kit cache
starter_kit:{user_id} → JSON of clusters (TTL: 24 hours)

# Cluster cache
cluster:{cluster_id} → JSON of cluster data (TTL: 1 hour)

# Graph cache (dynamic)
graph:{paper_id} → JSON of graph structure (TTL: 1 hour)

# Search results cache
search:{query_hash}:{domain} → JSON of results (TTL: 30 minutes)

# API rate limiting
rate_limit:{user_id}:{endpoint} → request count (TTL: 1 minute/hour based on limit)

# User profile embedding cache
user:embedding:{user_id} → Binary embedding vector (TTL: 6 hours)

# Paper metadata cache (frequently accessed)
paper:meta:{paper_id} → JSON of paper metadata (TTL: 24 hours)

# Example structure
{
    "user:session:123": {
        "user_id": 123,
        "email": "user@example.com",
        "domain": "healthcare",
        "expires_at": "2025-11-09T10:00:00Z"
    },
    
    "starter_kit:123": {
        "user_id": 123,
        "clusters": [
            {
                "cluster_id": 1,
                "name": "AI-Driven Protein Structure",
                "papers": [...],
                "reference_paper": {...}
            },
            {...},
            {...}
        ],
        "generated_at": "2025-11-08T10:00:00Z"
    },
    
    "graph:arxiv:2401.12345": {
        "reference_paper": {...},
        "related_papers": [...],
        "edges": [...],
        "generated_at": "2025-11-08T10:30:00Z"
    }
}
```

---

## 6. API Specifications

### 6.1 Base Configuration

```
Base URL: https://api.citeconnect.com/api/v1
Content-Type: application/json
Authentication: Bearer token (JWT)
Rate Limit: 100 requests/minute per user
```

### 6.2 Authentication Endpoints

#### POST /auth/register

**Description**: Register new user with domain and interests

**Request Body**:
```json
{
    "email": "user@example.com",
    "password": "SecurePassword123!",
    "name": "Sarah Chen",
    "domain": "healthcare",
    "interests": ["NLP", "clinical trials", "drug discovery"],
    "google_scholar_url": "https://scholar.google.com/citations?user=ABC123", // Optional
    "uploaded_paper_file": "base64_encoded_pdf" // Optional
}
```

**Response** (201 Created):
```json
{
    "user_id": 12345,
    "email": "user@example.com",
    "name": "Sarah Chen",
    "domain": "healthcare",
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "bearer",
    "expires_in": 86400,
    "starter_kit_status": "processing" // "ready" or "processing"
}
```

**Errors**:
- 400: Invalid email/password format
- 409: Email already exists

#### POST /auth/login

**Request Body**:
```json
{
    "email": "user@example.com",
    "password": "SecurePassword123!"
}
```

**Response** (200 OK):
```json
{
    "user_id": 12345,
    "email": "user@example.com",
    "name": "Sarah Chen",
    "domain": "healthcare",
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "bearer",
    "expires_in": 86400
}
```

**Errors**:
- 401: Invalid credentials
- 404: User not found

#### POST /auth/refresh

**Request Headers**:
```
Authorization: Bearer {refresh_token}
```

**Response** (200 OK):
```json
{
    "access_token": "new_token...",
    "token_type": "bearer",
    "expires_in": 86400
}
```

### 6.3 User Endpoints

#### GET /users/me

**Description**: Get current user profile

**Response** (200 OK):
```json
{
    "user_id": 12345,
    "email": "user@example.com",
    "name": "Sarah Chen",
    "domain": "healthcare",
    "interests": [
        {
            "keyword": "NLP",
            "source": "manual",
            "weight": 1.0
        },
        {
            "keyword": "protein folding",
            "source": "google_scholar",
            "weight": 0.8
        }
    ],
    "google_scholar_url": "https://scholar.google.com/...",
    "created_at": "2025-11-01T10:00:00Z"
}
```

#### PUT /users/me

**Description**: Update user profile

**Request Body**:
```json
{
    "name": "Sarah Chen, PhD",
    "interests": ["NLP", "antibody design"],
    "google_scholar_url": "https://scholar.google.com/..."
}
```

**Response** (200 OK):
```json
{
    "user_id": 12345,
    "message": "Profile updated successfully",
    "regenerate_clusters": true // Indicates clusters will be regenerated
}
```

#### GET /users/me/home

**Description**: Get home page data with 3 clusters

**Query Parameters**:
- `view_type`: "network" or "card" (optional, default: "card")
- `force_refresh`: boolean (optional, default: false)

**Response** (200 OK):
```json
{
    "user": {
        "name": "Sarah Chen",
        "domain": "healthcare",
        "interests": ["NLP", "clinical trials"]
    },
    "clusters": [
        {
            "cluster_id": 1,
            "name": "AI-Driven Protein Structure Prediction",
            "theme": "Machine learning approaches for predicting protein structures",
            "paper_count": 12,
            "average_relevance": 0.91,
            "reference_paper": {
                "paper_id": "arxiv:2401.12345",
                "title": "AlphaFold: Improved protein structure prediction",
                "authors": ["Jumper, J.", "Evans, R."],
                "year": 2021,
                "citation_count": 9432,
                "similarity_to_user": 0.91
            },
            "papers": [
                {
                    "paper_id": "arxiv:2401.12345",
                    "title": "AlphaFold: Improved protein structure prediction",
                    "authors": ["Jumper, J.", "Evans, R."],
                    "year": 2021,
                    "citation_count": 9432,
                    "similarity_to_reference": 1.0,
                    "position_x": 250.0,
                    "position_y": 200.0,
                    "is_reference": true
                },
                {
                    "paper_id": "science:2021:rosettafold",
                    "title": "RoseTTAFold: Accurate prediction of protein structures",
                    "authors": ["Baek, M.", "DiMaio, F."],
                    "year": 2021,
                    "citation_count": 2156,
                    "similarity_to_reference": 0.94,
                    "position_x": 320.0,
                    "position_y": 210.0,
                    "is_reference": false
                }
                // ... more papers
            ]
        },
        {
            "cluster_id": 2,
            "name": "Computational Protein Folding Mechanisms",
            "paper_count": 11,
            "average_relevance": 0.85,
            "reference_paper": {...},
            "papers": [...]
        },
        {
            "cluster_id": 3,
            "name": "Machine Learning in Drug Design",
            "paper_count": 12,
            "average_relevance": 0.83,
            "reference_paper": {...},
            "papers": [...]
        }
    ],
    "generated_at": "2025-11-08T10:00:00Z",
    "expires_at": "2025-11-09T10:00:00Z"
}
```

#### GET /users/me/dashboard

**Description**: Get user dashboard analytics

**Response** (200 OK):
```json
{
    "profile": {
        "name": "Sarah Chen",
        "domain": "healthcare",
        "interests": ["NLP", "clinical trials"],
        "member_since": "2025-11-01"
    },
    "saved_papers": [
        {
            "paper_id": "...",
            "title": "...",
            "saved_at": "2025-11-05T14:20:00Z",
            "notes": "Important for chapter 3"
        }
    ],
    "liked_papers": [...],
    "analytics": {
        "total_papers_viewed": 156,
        "total_time_spent_hours": 12.5,
        "papers_saved_count": 25,
        "papers_liked_count": 18,
        "most_viewed_topic": "Clinical NLP",
        "topic_distribution": {
            "Clinical NLP": 45,
            "Drug Discovery": 30,
            "Genomics": 25
        },
        "reading_timeline": [
            {
                "date": "2025-11-01",
                "papers_read": 5
            },
            {
                "date": "2025-11-02",
                "papers_read": 8
            }
            // ... more dates
        ],
        "interaction_heatmap": {
            "monday": [0, 2, 5, 8, 12, 10, 8, 5, 3, 0],
            "tuesday": [...]
            // ... other days
        }
    }
}
```

### 6.4 Paper Endpoints

#### GET /papers/{paper_id}

**Description**: Get detailed paper information

**Response** (200 OK):
```json
{
    "paper_id": "arxiv:2401.12345",
    "title": "AlphaFold: Improved protein structure prediction",
    "authors": ["Jumper, J.", "Evans, R.", "Pritzel, A."],
    "year": 2021,
    "venue": "Nature",
    "citation_count": 9432,
    "abstract": "Proteins are essential to life...",
    "summary": "This paper presents AlphaFold 2, a deep learning system...",
    "introduction": "Recent advances in deep learning have...",
    "domain": "healthcare",
    "pdf_url": "https://storage.googleapis.com/.../alphafold.pdf",
    "external_links": {
        "arxiv": "https://arxiv.org/abs/2401.12345",
        "semantic_scholar": "https://www.semanticscholar.org/paper/..."
    }
}
```

#### GET /papers/{paper_id}/graph

**Description**: Get dynamic citation network for a paper

**Query Parameters**:
- `limit`: number of related papers (default: 25, max: 50)
- `similarity_threshold`: float 0-1 (default: 0.70)

**Response** (200 OK):
```json
{
    "reference_paper": {
        "paper_id": "arxiv:2401.12345",
        "title": "AlphaFold: Improved protein structure",
        "authors": ["Jumper, J."],
        "year": 2021,
        "citation_count": 9432,
        "abstract": "...",
        "summary": "..."
    },
    "related_papers": [
        {
            "paper_id": "science:2021:rosettafold",
            "title": "RoseTTAFold: Accurate prediction",
            "authors": ["Baek, M."],
            "year": 2021,
            "citation_count": 2156,
            "similarity_score": 0.94,
            "relationship_type": "semantic", // "semantic", "citation", "co-citation"
            "position_x": 320.0,
            "position_y": 210.0
        }
        // ... more papers
    ],
    "edges": [
        {
            "source": "arxiv:2401.12345",
            "target": "science:2021:rosettafold",
            "weight": 0.94,
            "type": "semantic" // "cites", "cited_by", "co-cited", "semantic"
        }
        // ... more edges
    ],
    "graph_metadata": {
        "total_nodes": 26,
        "total_edges": 45,
        "layout_algorithm": "force_directed",
        "generated_at": "2025-11-08T10:30:00Z"
    }
}
```

#### POST /papers/{paper_id}/save

**Description**: Save paper to user library

**Request Body**:
```json
{
    "notes": "Important for my research on antibody design" // Optional
}
```

**Response** (200 OK):
```json
{
    "message": "Paper saved successfully",
    "saved_at": "2025-11-08T10:30:00Z"
}
```

#### POST /papers/{paper_id}/like

**Description**: Like a paper

**Response** (200 OK):
```json
{
    "message": "Paper liked successfully",
    "liked_at": "2025-11-08T10:30:00Z"
}
```

#### DELETE /papers/{paper_id}/save

**Description**: Remove paper from saved list

**Response** (200 OK):
```json
{
    "message": "Paper removed from saved list"
}
```

#### DELETE /papers/{paper_id}/like

**Description**: Unlike a paper

**Response** (200 OK):
```json
{
    "message": "Paper unliked successfully"
}
```

### 6.5 Search Endpoints

#### GET /search

**Description**: Semantic search for papers

**Query Parameters**:
- `q`: query string (required)
- `domain`: filter by domain (optional, uses user's domain by default)
- `year_min`: minimum year (optional)
- `year_max`: maximum year (optional)
- `limit`: results to return (default: 20, max: 50)
- `offset`: pagination offset (default: 0)

**Response** (200 OK):
```json
{
    "query": "antibody design machine learning",
    "domain": "healthcare",
    "total_results": 847,
    "shown_results": 20,
    "results": [
        {
            "paper_id": "nature:2023:antibody_dl",
            "title": "Deep learning for antibody structure prediction",
            "authors": ["Smith, J.", "Johnson, A."],
            "year": 2023,
            "venue": "Nature",
            "citation_count": 234,
            "abstract": "We present a deep learning framework...",
            "summary": "This paper introduces...",
            "relevance_score": 0.89,
            "match_explanation": {
                "semantic_similarity": 0.89,
                "keyword_matches": ["antibody", "design", "machine learning"],
                "confidence": "high"
            }
        }
        // ... more results
    ],
    "filters_applied": {
        "domain": "healthcare",
        "year_min": null,
        "year_max": null
    },
    "search_time_ms": 245
}
```

### 6.6 Interaction Endpoints

#### POST /interactions

**Description**: Track user interaction with paper

**Request Body**:
```json
{
    "paper_id": "arxiv:2401.12345",
    "interaction_type": "view", // "view", "click", "click_node", "read_time"
    "duration_seconds": 120, // Optional, for "read_time"
    "context": { // Optional
        "source": "search_results",
        "query": "antibody design",
        "cluster_id": 1,
        "source_paper_id": "nature:2023:antibody_dl"
    }
}
```

**Response** (201 Created):
```json
{
    "interaction_id": 98765,
    "message": "Interaction recorded",
    "profile_updated": false // True if this triggers profile embedding update
}
```

### 6.7 Cluster Endpoints

#### GET /clusters/{cluster_id}

**Description**: Get detailed cluster information

**Response** (200 OK):
```json
{
    "cluster_id": 1,
    "name": "AI-Driven Protein Structure Prediction",
    "theme": "Machine learning approaches for predicting protein structures",
    "domain": "healthcare",
    "paper_count": 12,
    "reference_paper": {...},
    "papers": [...],
    "created_at": "2025-11-08T10:00:00Z"
}
```

---

## 7. Component Design

### 7.1 Backend Services

#### 7.1.1 EmbeddingService

**Purpose**: Generate SPECTER2 embeddings for text and papers

**Class Definition**:
```python
class EmbeddingService:
    """
    Service for generating SPECTER2 embeddings
    """
    
    def __init__(self, model_name: str = "allenai/specter2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model(self) -> None:
        """Load SPECTER2 model and tokenizer"""
        # Implementation details in code
        pass
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to 768-dim embedding
        
        Args:
            text: Input text (query, keywords, etc.)
        
        Returns:
            768-dimensional numpy array
        """
        pass
    
    def encode_paper(self, title: str, abstract: str, 
                     introduction: str = None) -> np.ndarray:
        """
        Encode paper to 768-dim embedding
        
        Args:
            title: Paper title
            abstract: Paper abstract
            introduction: Optional introduction section
        
        Returns:
            768-dimensional numpy array
        """
        pass
    
    def encode_batch(self, texts: List[str], 
                     batch_size: int = 32) -> np.ndarray:
        """
        Batch encode multiple texts
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
        
        Returns:
            Array of shape (n_texts, 768)
        """
        pass
    
    def compute_similarity(self, emb1: np.ndarray, 
                          emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between embeddings
        
        Args:
            emb1: First embedding (768-dim)
            emb2: Second embedding (768-dim)
        
        Returns:
            Similarity score (0-1)
        """
        pass
```

**Key Methods**:
- `load_model()`: Load SPECTER2 from HuggingFace
- `encode_text()`: Single text encoding
- `encode_paper()`: Paper encoding (title + abstract + intro)
- `encode_batch()`: Batch encoding for efficiency
- `compute_similarity()`: Cosine similarity calculation

#### 7.1.2 SearchService

**Purpose**: Handle semantic search queries

**Class Definition**:
```python
class SearchService:
    """
    Service for semantic search using Weaviate
    """
    
    def __init__(self, weaviate_client, embedding_service, redis_client):
        self.weaviate = weaviate_client
        self.embedding_service = embedding_service
        self.redis = redis_client
    
    def search(self, query: str, domain: str, 
               filters: Dict, limit: int = 20) -> List[Dict]:
        """
        Perform semantic search
        
        Args:
            query: Search query string
            domain: User's domain (healthcare/fintech/quantum)
            filters: Additional filters (year, citation_count, etc.)
            limit: Number of results
        
        Returns:
            List of papers with similarity scores
        """
        # 1. Check cache
        cache_key = self._generate_cache_key(query, domain, filters)
        cached = self.redis.get(cache_key)
        if cached:
            return cached
        
        # 2. Generate query embedding
        query_embedding = self.embedding_service.encode_text(query)
        
        # 3. Query Weaviate
        results = self._query_weaviate(
            query_vector=query_embedding,
            domain=domain,
            filters=filters,
            limit=limit
        )
        
        # 4. Fetch metadata from PostgreSQL
        enriched_results = self._enrich_with_metadata(results)
        
        # 5. Cache results
        self.redis.setex(cache_key, 1800, enriched_results)  # 30 min TTL
        
        return enriched_results
    
    def _query_weaviate(self, query_vector: np.ndarray, 
                       domain: str, filters: Dict, 
                       limit: int) -> List[Dict]:
        """Query Weaviate vector database"""
        pass
    
    def _enrich_with_metadata(self, results: List[Dict]) -> List[Dict]:
        """Fetch additional metadata from PostgreSQL"""
        pass
    
    def _generate_cache_key(self, query: str, domain: str, 
                           filters: Dict) -> str:
        """Generate cache key for query"""
        pass
```

**Workflow**:
1. Check Redis cache for query
2. Generate query embedding with SPECTER2
3. Query Weaviate with embedding + filters
4. Enrich results with PostgreSQL metadata
5. Cache results (30 min TTL)

#### 7.1.3 ClusteringService

**Purpose**: Generate thematic clusters for home page

**Class Definition**:
```python
class ClusteringService:
    """
    Service for generating paper clusters
    """
    
    def __init__(self, weaviate_client, neo4j_client, 
                 embedding_service, llm_client):
        self.weaviate = weaviate_client
        self.neo4j = neo4j_client
        self.embedding_service = embedding_service
        self.llm = llm_client  # For theme naming
    
    def generate_home_clusters(self, user_id: int, 
                               n_clusters: int = 3) -> List[Dict]:
        """
        Generate 3 thematic clusters for user home page
        
        Args:
            user_id: User ID
            n_clusters: Number of clusters (default: 3)
        
        Returns:
            List of cluster dictionaries
        """
        # 1. Load user profile embedding
        user_embedding = self._load_user_embedding(user_id)
        
        # 2. Retrieve candidate papers (semantic search)
        candidate_papers = self._retrieve_candidates(
            user_embedding=user_embedding,
            limit=100
        )
        
        # 3. Get embeddings for candidate papers
        paper_embeddings = self._get_paper_embeddings(candidate_papers)
        
        # 4. Perform k-means clustering (k=3)
        cluster_labels = self._kmeans_clustering(
            embeddings=paper_embeddings,
            n_clusters=n_clusters
        )
        
        # 5. Assign papers to clusters
        clusters = self._assign_papers_to_clusters(
            papers=candidate_papers,
            labels=cluster_labels
        )
        
        # 6. Generate theme names (LLM)
        for cluster in clusters:
            cluster['name'] = self._generate_theme_name(
                cluster['papers']
            )
        
        # 7. Select reference papers (most central)
        for cluster in clusters:
            cluster['reference_paper'] = self._select_reference_paper(
                cluster['papers']
            )
        
        # 8. Compute graph layouts
        for cluster in clusters:
            cluster['papers'] = self._compute_graph_layout(
                papers=cluster['papers'],
                reference_paper=cluster['reference_paper']
            )
        
        # 9. Store clusters in PostgreSQL
        self._store_clusters(user_id, clusters)
        
        return clusters
    
    def _kmeans_clustering(self, embeddings: np.ndarray, 
                          n_clusters: int) -> np.ndarray:
        """
        Perform k-means clustering on embeddings
        
        Args:
            embeddings: Array of shape (n_papers, 768)
            n_clusters: Number of clusters
        
        Returns:
            Array of cluster labels
        """
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        return labels
    
    def _generate_theme_name(self, papers: List[Dict]) -> str:
        """
        Generate cluster theme name using LLM
        
        Args:
            papers: List of papers in cluster
        
        Returns:
            Theme name (e.g., "AI-Driven Protein Structure")
        """
        # Extract paper titles
        titles = [p['title'] for p in papers[:10]]  # Top 10
        
        # LLM prompt
        prompt = f"""Generate a concise 4-6 word research theme name 
        that captures the main topic of these papers:
        
        {titles}
        
        Theme name:"""
        
        theme_name = self.llm.generate(prompt, max_tokens=20)
        return theme_name.strip()
    
    def _compute_graph_layout(self, papers: List[Dict], 
                             reference_paper: Dict) -> List[Dict]:
        """
        Compute force-directed graph layout
        
        Args:
            papers: List of papers in cluster
            reference_paper: Reference paper (center)
        
        Returns:
            Papers with position_x, position_y added
        """
        import networkx as nx
        
        # Build similarity graph
        G = nx.Graph()
        for paper in papers:
            G.add_node(paper['paper_id'])
        
        # Add edges (similarity > 0.8)
        for i, p1 in enumerate(papers):
            for p2 in papers[i+1:]:
                sim = self.embedding_service.compute_similarity(
                    p1['embedding'], p2['embedding']
                )
                if sim > 0.8:
                    G.add_edge(p1['paper_id'], p2['paper_id'], weight=sim)
        
        # Compute layout (force-directed)
        pos = nx.spring_layout(
            G, 
            k=0.5,  # Optimal distance between nodes
            iterations=100,
            seed=42
        )
        
        # Add positions to papers
        for paper in papers:
            x, y = pos[paper['paper_id']]
            paper['position_x'] = float(x * 500)  # Scale to pixel coords
            paper['position_y'] = float(y * 500)
        
        return papers
```

**Clustering Algorithm**:
1. Load user profile embedding
2. Retrieve 100 candidate papers (semantic search)
3. Extract paper embeddings
4. K-means clustering (k=3)
5. Generate theme names with LLM
6. Select reference papers (closest to centroid)
7. Compute graph layouts (force-directed)
8. Store in PostgreSQL + cache in Redis

#### 7.1.4 GraphService

**Purpose**: Generate dynamic citation networks

**Class Definition**:
```python
class GraphService:
    """
    Service for generating dynamic citation networks
    """
    
    def __init__(self, weaviate_client, neo4j_client, 
                 embedding_service, redis_client):
        self.weaviate = weaviate_client
        self.neo4j = neo4j_client
        self.embedding_service = embedding_service
        self.redis = redis_client
    
    def generate_paper_graph(self, paper_id: str, 
                            limit: int = 25,
                            similarity_threshold: float = 0.70) -> Dict:
        """
        Generate dynamic citation network for a paper
        
        Args:
            paper_id: Reference paper ID
            limit: Number of related papers
            similarity_threshold: Minimum similarity score
        
        Returns:
            Graph structure with nodes and edges
        """
        # 1. Check cache
        cache_key = f"graph:{paper_id}"
        cached = self.redis.get(cache_key)
        if cached:
            return cached
        
        # 2. Get reference paper embedding
        ref_embedding = self._get_paper_embedding(paper_id)
        
        # 3. Semantic search for similar papers
        semantic_papers = self._semantic_search(
            embedding=ref_embedding,
            limit=limit * 2,  # Get more to filter
            threshold=similarity_threshold
        )
        
        # 4. Citation-based retrieval (Neo4j)
        citation_papers = self._citation_search(
            paper_id=paper_id,
            limit=limit
        )
        
        # 5. Merge and deduplicate
        all_papers = self._merge_papers(semantic_papers, citation_papers)
        
        # 6. Build graph structure
        graph = self._build_graph_structure(
            reference_paper_id=paper_id,
            related_papers=all_papers[:limit],
            similarity_threshold=similarity_threshold
        )
        
        # 7. Compute layout
        graph = self._compute_layout(graph)
        
        # 8. Cache result
        self.redis.setex(cache_key, 3600, graph)  # 1 hour TTL
        
        return graph
    
    def _semantic_search(self, embedding: np.ndarray, 
                        limit: int, threshold: float) -> List[Dict]:
        """Query Weaviate for semantically similar papers"""
        pass
    
    def _citation_search(self, paper_id: str, limit: int) -> List[Dict]:
        """
        Query Neo4j for citation-related papers
        
        Cypher query:
        MATCH (ref:Paper {paper_id: $paper_id})
        MATCH (ref)-[:CITES]->(cited:Paper)
        RETURN cited
        UNION
        MATCH (ref:Paper {paper_id: $paper_id})
        MATCH (citing:Paper)-[:CITES]->(ref)
        RETURN citing
        LIMIT $limit
        """
        pass
    
    def _build_graph_structure(self, reference_paper_id: str,
                              related_papers: List[Dict],
                              similarity_threshold: float) -> Dict:
        """
        Build graph with nodes and edges
        
        Returns:
            {
                "reference_paper": {...},
                "related_papers": [...],
                "edges": [...]
            }
        """
        pass
    
    def _compute_layout(self, graph: Dict) -> Dict:
        """Compute force-directed layout for graph"""
        pass
```

**Graph Generation Algorithm**:
1. Check Redis cache
2. Get reference paper embedding
3. Semantic search (Weaviate) for similar papers
4. Citation search (Neo4j) for cited/citing papers
5. Merge and deduplicate results
6. Build graph structure (nodes + edges)
7. Compute force-directed layout
8. Cache result (1 hour TTL)

#### 7.1.5 RecommendationService

**Purpose**: Generate personalized recommendations

**Class Definition**:
```python
class RecommendationService:
    """
    Service for personalized recommendations
    """
    
    def __init__(self, postgres_db, weaviate_client, 
                 neo4j_client, embedding_service):
        self.db = postgres_db
        self.weaviate = weaviate_client
        self.neo4j = neo4j_client
        self.embedding_service = embedding_service
    
    def generate_starter_kit(self, user_id: int) -> List[Dict]:
        """
        Generate initial starter kit (called after signup)
        
        This is a Celery task triggered on registration
        
        Args:
            user_id: User ID
        
        Returns:
            List of 3 clusters (15-20 papers total)
        """
        # 1. Load user inputs
        user_profile = self._load_user_profile(user_id)
        
        # 2. Build composite query embedding
        query_embedding = self._build_query_embedding(user_profile)
        
        # 3. Multi-strategy retrieval
        candidates = self._multi_strategy_retrieval(
            query_embedding=query_embedding,
            user_profile=user_profile
        )
        
        # 4. Multi-factor scoring
        scored_papers = self._score_candidates(
            candidates=candidates,
            user_profile=user_profile
        )
        
        # 5. Select top papers
        top_papers = sorted(
            scored_papers, 
            key=lambda x: x['final_score'], 
            reverse=True
        )[:35]
        
        # 6. Quality validation
        top_papers = self._validate_quality(top_papers)
        
        # 7. Cluster into 3 themes
        clusters = self._cluster_papers(top_papers, n_clusters=3)
        
        return clusters
    
    def _build_query_embedding(self, user_profile: Dict) -> np.ndarray:
        """
        Build composite query embedding from user inputs
        
        Weights:
        - Keywords only: 100% keywords
        - Keywords + Scholar: 50% keywords + 50% author profile
        - All three: 40% keywords + 30% author + 30% uploaded paper
        """
        embeddings = []
        weights = []
        
        # Keyword embedding
        if user_profile.get('interests'):
            keyword_text = " ".join(user_profile['interests'])
            keyword_emb = self.embedding_service.encode_text(keyword_text)
            embeddings.append(keyword_emb)
            weights.append(0.4 if len(embeddings) > 1 else 1.0)
        
        # Author profile embedding (from Google Scholar)
        if user_profile.get('author_papers'):
            author_emb = self._compute_author_embedding(
                user_profile['author_papers']
            )
            embeddings.append(author_emb)
            weights.append(0.3)
        
        # Uploaded paper embedding
        if user_profile.get('uploaded_paper'):
            paper_emb = self.embedding_service.encode_paper(
                title=user_profile['uploaded_paper']['title'],
                abstract=user_profile['uploaded_paper']['abstract']
            )
            embeddings.append(paper_emb)
            weights.append(0.3)
        
        # Weighted average
        query_embedding = np.average(
            embeddings, 
            axis=0, 
            weights=weights
        )
        
        return query_embedding
    
    def _multi_strategy_retrieval(self, query_embedding: np.ndarray,
                                  user_profile: Dict) -> List[Dict]:
        """
        Multi-strategy paper retrieval
        
        Strategies:
        A. Semantic search (Weaviate) - 200 papers
        B. Citation-based (Neo4j) - 50 papers
        C. Keyword search (PostgreSQL) - 40 papers
        D. Popular in domain - 30 papers
        
        Returns deduplicated list of ~150-200 papers
        """
        all_candidates = []
        
        # Strategy A: Semantic search
        semantic_papers = self._semantic_retrieval(
            query_embedding, 
            user_profile['domain'],
            limit=200
        )
        all_candidates.extend(semantic_papers)
        
        # Strategy B: Citation-based (if uploaded paper)
        if user_profile.get('uploaded_paper_id'):
            citation_papers = self._citation_retrieval(
                user_profile['uploaded_paper_id'],
                limit=50
            )
            all_candidates.extend(citation_papers)
        
        # Strategy C: Keyword search
        keyword_papers = self._keyword_retrieval(
            user_profile['interests'],
            user_profile['domain'],
            limit=40
        )
        all_candidates.extend(keyword_papers)
        
        # Strategy D: Popular papers in domain
        popular_papers = self._popular_papers_retrieval(
            user_profile['domain'],
            limit=30
        )
        all_candidates.extend(popular_papers)
        
        # Deduplicate by paper_id
        seen = set()
        unique_candidates = []
        for paper in all_candidates:
            if paper['paper_id'] not in seen:
                seen.add(paper['paper_id'])
                unique_candidates.append(paper)
        
        return unique_candidates
    
    def _score_candidates(self, candidates: List[Dict],
                         user_profile: Dict) -> List[Dict]:
        """
        Multi-factor scoring for each candidate
        
        Score = 0.35 × semantic_similarity +
                0.20 × citation_relevance +
                0.15 × keyword_match +
                0.15 × popularity +
                0.10 × recency +
                0.05 × diversity
        """
        for paper in candidates:
            # Calculate each component
            semantic_sim = paper.get('semantic_similarity', 0.0)
            citation_rel = self._calculate_citation_relevance(
                paper, user_profile
            )
            keyword_match = self._calculate_keyword_match(
                paper, user_profile['interests']
            )
            popularity = self._calculate_popularity_score(paper)
            recency = self._calculate_recency_score(paper['year'])
            diversity = 0.0  # Calculated later after initial ranking
            
            # Final score
            paper['final_score'] = (
                0.35 * semantic_sim +
                0.20 * citation_rel +
                0.15 * keyword_match +
                0.15 * popularity +
                0.10 * recency +
                0.05 * diversity
            )
            
            # Store components for debugging
            paper['score_components'] = {
                'semantic': semantic_sim,
                'citation': citation_rel,
                'keyword': keyword_match,
                'popularity': popularity,
                'recency': recency
            }
        
        return candidates
    
    def _calculate_popularity_score(self, paper: Dict) -> float:
        """
        Calculate normalized popularity score
        
        Formula:
        citation_rate = citation_count / years_since_publication
        popularity_score = citation_rate / max_citation_rate
        """
        current_year = 2025
        years_since = max(current_year - paper['year'], 1)
        citation_rate = paper['citation_count'] / years_since
        
        # Normalize (assumes max_citation_rate computed from all candidates)
        # For now, use a reasonable max (e.g., 1000 citations/year)
        max_rate = 1000
        popularity_score = min(citation_rate / max_rate, 1.0)
        
        return popularity_score
    
    def _calculate_recency_score(self, year: int) -> float:
        """
        Calculate recency score
        
        2024+: 1.0
        2023: 0.9
        2022: 0.8
        2021: 0.7
        2020: 0.6
        <2020: 0.5
        """
        current_year = 2025
        age = current_year - year
        
        if age <= 0:
            return 1.0
        elif age == 1:
            return 0.9
        elif age == 2:
            return 0.8
        elif age == 3:
            return 0.7
        elif age == 4:
            return 0.6
        else:
            return 0.5
```

### 7.2 Frontend Components

#### 7.2.1 CytoscapeGraph Component

**Purpose**: Render interactive citation network

**Component Structure**:
```typescript
// src/components/graph/CytoscapeGraph.tsx

import { useEffect, useRef, useState } from 'react';
import cytoscape from 'cytoscape';

interface Node {
  paper_id: string;
  title: string;
  year: number;
  similarity_score: number;
  position_x: number;
  position_y: number;
  is_reference: boolean;
}

interface Edge {
  source: string;
  target: string;
  weight: number;
  type: 'semantic' | 'cites' | 'cited_by';
}

interface GraphData {
  reference_paper: Node;
  related_papers: Node[];
  edges: Edge[];
}

interface CytoscapeGraphProps {
  graphData: GraphData;
  onNodeClick: (paperId: string) => void;
  onNodeHover: (paperId: string | null) => void;
}

export const CytoscapeGraph: React.FC<CytoscapeGraphProps> = ({
  graphData,
  onNodeClick,
  onNodeHover
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<cytoscape.Core | null>(null);
  
  useEffect(() => {
    if (!containerRef.current || !graphData) return;
    
    // Initialize Cytoscape
    const cy = cytoscape({
      container: containerRef.current,
      
      // Elements
      elements: {
        nodes: [
          graphData.reference_paper,
          ...graphData.related_papers
        ].map(node => ({
          data: {
            id: node.paper_id,
            label: node.title,
            similarity: node.similarity_score,
            isReference: node.is_reference,
            year: node.year
          },
          position: {
            x: node.position_x,
            y: node.position_y
          }
        })),
        
        edges: graphData.edges.map(edge => ({
          data: {
            id: `${edge.source}-${edge.target}`,
            source: edge.source,
            target: edge.target,
            weight: edge.weight,
            type: edge.type
          }
        }))
      },
      
      // Style
      style: [
        {
          selector: 'node',
          style: {
            'background-color': (ele: any) => 
              ele.data('isReference') ? '#3b82f6' : '#8b5cf6',
            'width': (ele: any) => 
              ele.data('similarity') * 60 + 20, // 20-80px based on similarity
            'height': (ele: any) => 
              ele.data('similarity') * 60 + 20,
            'label': 'data(label)',
            'font-size': '10px',
            'text-wrap': 'wrap',
            'text-max-width': '100px',
            'text-valign': 'bottom',
            'text-halign': 'center',
            'color': '#fff'
          }
        },
        {
          selector: 'edge',
          style: {
            'width': (ele: any) => ele.data('weight') * 3,
            'line-color': '#cbd5e1',
            'target-arrow-color': '#cbd5e1',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier'
          }
        },
        {
          selector: 'node:selected',
          style: {
            'border-width': 3,
            'border-color': '#fbbf24'
          }
        }
      ],
      
      // Layout
      layout: {
        name: 'preset', // Use pre-computed positions
        fit: true,
        padding: 50
      },
      
      // Interaction
      userZoomingEnabled: true,
      userPanningEnabled: true,
      boxSelectionEnabled: false
    });
    
    // Event handlers
    cy.on('tap', 'node', (event) => {
      const node = event.target;
      onNodeClick(node.data('id'));
    });
    
    cy.on('mouseover', 'node', (event) => {
      const node = event.target;
      onNodeHover(node.data('id'));
    });
    
    cy.on('mouseout', 'node', () => {
      onNodeHover(null);
    });
    
    cyRef.current = cy;
    
    return () => {
      cy.destroy();
    };
  }, [graphData]);
  
  return (
    <div 
      ref={containerRef} 
      className="w-full h-full bg-slate-900 rounded-lg"
    />
  );
};
```

**Key Features**:
- Pre-computed node positions (from backend)
- Dynamic node sizing (based on similarity)
- Color coding (reference vs related papers)
- Interactive (click, hover, zoom, pan)
- Edge weight visualization

#### 7.2.2 ClusterCard Component

**Purpose**: Display cluster information in card view

**Component Structure**:
```typescript
// src/components/home/ClusterCard.tsx

interface Cluster {
  cluster_id: number;
  name: string;
  theme: string;
  paper_count: number;
  average_relevance: number;
  reference_paper: {
    paper_id: string;
    title: string;
    authors: string[];
    year: number;
    citation_count: number;
  };
}

interface ClusterCardProps {
  cluster: Cluster;
  onViewNetwork: (clusterId: number) => void;
}

export const ClusterCard: React.FC<ClusterCardProps> = ({
  cluster,
  onViewNetwork
}) => {
  return (
    <div className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition">
      {/* Header */}
      <div className="mb-4">
        <h3 className="text-xl font-bold text-gray-900">
          {cluster.name}
        </h3>
        <p className="text-sm text-gray-600 mt-2">
          {cluster.theme}
        </p>
      </div>
      
      {/* Stats */}
      <div className="flex items-center gap-4 mb-4">
        <div className="flex items-center gap-1">
          <FileText className="w-4 h-4 text-gray-500" />
          <span className="text-sm text-gray-700">
            {cluster.paper_count} papers
          </span>
        </div>
        <div className="flex items-center gap-1">
          <Target className="w-4 h-4 text-green-500" />
          <span className="text-sm text-gray-700">
            {Math.round(cluster.average_relevance * 100)}% relevance
          </span>
        </div>
      </div>
      
      {/* Reference Paper Preview */}
      <div className="bg-gray-50 rounded p-4 mb-4">
        <div className="flex items-start gap-2">
          <Star className="w-5 h-5 text-yellow-500 flex-shrink-0 mt-1" />
          <div>
            <p className="text-sm font-medium text-gray-900 line-clamp-2">
              {cluster.reference_paper.title}
            </p>
            <p className="text-xs text-gray-600 mt-1">
              {cluster.reference_paper.authors.slice(0, 3).join(', ')}
              {cluster.reference_paper.authors.length > 3 && ' et al.'}
            </p>
            <div className="flex items-center gap-3 mt-2">
              <span className="text-xs text-gray-500">
                {cluster.reference_paper.year}
              </span>
              <span className="text-xs text-gray-500">
                {cluster.reference_paper.citation_count.toLocaleString()} citations
              </span>
            </div>
          </div>
        </div>
      </div>
      
      {/* Actions */}
      <button
        onClick={() => onViewNetwork(cluster.cluster_id)}
        className="w-full bg-blue-600 text-white py-2 rounded-md hover:bg-blue-700 transition"
      >
        View Network Graph
      </button>
    </div>
  );
};
```

---

## 8. Algorithm Implementations

### 8.1 Starter Kit Generation Algorithm

**Input**:
- user_id
- domain (healthcare/fintech/quantum_computing)
- interests (keywords)
- google_scholar_url (optional)
- uploaded_paper (optional)

**Output**:
- 3 clusters
- 15-20 total papers
- Each cluster has reference paper + graph layout

**Algorithm**:
```
FUNCTION generate_starter_kit(user_id):
    
    // Step 1: Load user inputs
    user_profile = load_user_profile(user_id)
    
    // Step 2: Build composite query embedding
    embeddings = []
    weights = []
    
    IF user_profile.interests:
        keyword_text = join(user_profile.interests, " ")
        keyword_emb = SPECTER2.encode(keyword_text)
        embeddings.append(keyword_emb)
        weights.append(0.4)
    
    IF user_profile.google_scholar_url:
        author_papers = fetch_semantic_scholar_papers(user_profile.scholar_id)
        author_emb = weighted_average([
            SPECTER2.encode(paper.title + paper.abstract) 
            for paper in author_papers
        ], weights=[paper.citations for paper in author_papers])
        embeddings.append(author_emb)
        weights.append(0.3)
    
    IF user_profile.uploaded_paper:
        paper_emb = SPECTER2.encode(
            uploaded_paper.title + uploaded_paper.abstract
        )
        embeddings.append(paper_emb)
        weights.append(0.3)
    
    query_embedding = weighted_average(embeddings, weights)
    
    // Step 3: Multi-strategy retrieval
    candidates = []
    
    // Strategy A: Semantic search (Weaviate)
    semantic_papers = weaviate.search(
        vector=query_embedding,
        filters={domain: user_profile.domain},
        limit=200
    )
    candidates.extend(semantic_papers)
    
    // Strategy B: Citation-based (Neo4j) - if uploaded paper
    IF user_profile.uploaded_paper_id:
        citation_papers = neo4j.query("""
            MATCH (uploaded:Paper {paper_id: $paper_id})
            MATCH (uploaded)-[:CITES]->(cited:Paper)
            RETURN cited
            UNION
            MATCH (citing:Paper)-[:CITES]->(uploaded)
            RETURN citing
            LIMIT 50
        """, paper_id=user_profile.uploaded_paper_id)
        candidates.extend(citation_papers)
    
    // Strategy C: Keyword search (PostgreSQL)
    FOR EACH keyword IN user_profile.interests:
        keyword_papers = postgres.query("""
            SELECT * FROM papers
            WHERE domain = $domain
            AND (to_tsvector('english', title) @@ to_tsquery($keyword)
                 OR to_tsvector('english', abstract) @@ to_tsquery($keyword))
            ORDER BY citation_count DESC
            LIMIT 10
        """, domain=user_profile.domain, keyword=keyword)
        candidates.extend(keyword_papers)
    
    // Strategy D: Popular papers in domain
    popular_papers = postgres.query("""
        SELECT * FROM papers
        WHERE domain = $domain
        AND year >= 2020
        AND citation_count >= 200
        ORDER BY (0.7 * citation_count + 0.3 * (2025 - year)) DESC
        LIMIT 30
    """, domain=user_profile.domain)
    candidates.extend(popular_papers)
    
    // Step 4: Deduplicate
    candidates = deduplicate_by_paper_id(candidates)  // ~150-200 unique
    
    // Step 5: Multi-factor scoring
    FOR EACH paper IN candidates:
        semantic_sim = cosine_similarity(
            query_embedding, 
            paper.embedding
        )
        
        citation_rel = 0.0
        IF user_profile.uploaded_paper_id:
            IF paper IN citation_papers:
                citation_rel = 1.0
            ELSE IF paper.co_cited_with(uploaded_paper):
                citation_rel = 0.8
        
        keyword_match = count_keyword_matches(
            paper.title + paper.abstract,
            user_profile.interests
        ) / len(user_profile.interests)
        
        years_since_pub = max(2025 - paper.year, 1)
        citation_rate = paper.citation_count / years_since_pub
        popularity = min(citation_rate / 1000, 1.0)
        
        recency = CASE paper.year:
            >= 2024: 1.0
            == 2023: 0.9
            == 2022: 0.8
            == 2021: 0.7
            == 2020: 0.6
            ELSE: 0.5
        
        paper.final_score = (
            0.35 * semantic_sim +
            0.20 * citation_rel +
            0.15 * keyword_match +
            0.15 * popularity +
            0.10 * recency +
            0.05 * 0.0  // diversity calculated later
        )
    
    // Step 6: Select top 35 papers
    candidates.sort(key=lambda x: x.final_score, reverse=True)
    top_papers = candidates[:35]
    
    // Step 7: Quality validation
    IF any(paper.final_score < 0.60 for paper in top_papers):
        // Remove low-quality papers
        top_papers = [p for p in top_papers if p.final_score >= 0.60]
    
    // Step 8: K-means clustering (k=3)
    paper_embeddings = [paper.embedding for paper in top_papers]
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(paper_embeddings)
    
    // Step 9: Assign papers to clusters
    clusters = [[] for _ in range(3)]
    FOR i, paper IN enumerate(top_papers):
        clusters[cluster_labels[i]].append(paper)
    
    // Step 10: Generate theme names (LLM)
    FOR cluster IN clusters:
        paper_titles = [p.title for p in cluster[:10]]
        cluster.name = LLM.generate(f"""
            Generate a 4-6 word research theme for these papers:
            {paper_titles}
        """)
    
    // Step 11: Select reference papers (closest to centroid)
    FOR cluster IN clusters:
        cluster_embeddings = [p.embedding for p in cluster]
        centroid = mean(cluster_embeddings)
        
        closest_paper = min(cluster, key=lambda p: 
            euclidean_distance(p.embedding, centroid)
        )
        cluster.reference_paper = closest_paper
    
    // Step 12: Compute similarity to reference
    FOR cluster IN clusters:
        FOR paper IN cluster:
            paper.similarity_to_reference = cosine_similarity(
                paper.embedding,
                cluster.reference_paper.embedding
            )
    
    // Step 13: Select 5-7 papers per cluster
    FOR cluster IN clusters:
        cluster.papers.sort(
            key=lambda x: x.final_score, 
            reverse=True
        )
        cluster.papers = cluster.papers[:6]  // Keep top 6
    
    // Step 14: Compute graph layouts
    FOR cluster IN clusters:
        // Build similarity graph
        G = NetworkGraph()
        FOR paper IN cluster.papers:
            G.add_node(paper.paper_id)
        
        FOR p1 IN cluster.papers:
            FOR p2 IN cluster.papers:
                IF p1 != p2:
                    sim = cosine_similarity(p1.embedding, p2.embedding)
                    IF sim > 0.80:
                        G.add_edge(p1.paper_id, p2.paper_id, weight=sim)
        
        // Force-directed layout
        positions = spring_layout(
            G, 
            k=0.5, 
            iterations=100, 
            seed=42
        )
        
        FOR paper IN cluster.papers:
            x, y = positions[paper.paper_id]
            paper.position_x = x * 500  // Scale to pixels
            paper.position_y = y * 500
    
    // Step 15: Store in PostgreSQL
    FOR cluster IN clusters:
        INSERT INTO paper_clusters (
            user_id, cluster_name, theme_description, 
            domain, paper_count
        ) VALUES (
            user_id, cluster.name, cluster.theme,
            user_profile.domain, len(cluster.papers)
        )
        cluster_id = LASTVAL()
        
        FOR paper IN cluster.papers:
            INSERT INTO cluster_papers (
                cluster_id, paper_id, centrality_score,
                is_reference_paper, similarity_to_reference,
                position_x, position_y
            ) VALUES (
                cluster_id, paper.paper_id, paper.final_score,
                paper == cluster.reference_paper,
                paper.similarity_to_reference,
                paper.position_x, paper.position_y
            )
    
    // Step 16: Cache in Redis
    redis.setex(
        key=f"starter_kit:{user_id}",
        value=json.dumps(clusters),
        ttl=86400  // 24 hours
    )
    
    RETURN clusters
```

### 8.2 Dynamic Graph Generation Algorithm

**Input**:
- paper_id (reference paper)
- limit (number of related papers, default 25)
- similarity_threshold (default 0.70)

**Output**:
- Graph structure with nodes, edges, and positions

**Algorithm**:
```
FUNCTION generate_paper_graph(paper_id, limit=25, threshold=0.70):
    
    // Step 1: Check cache
    cache_key = f"graph:{paper_id}"
    cached = redis.get(cache_key)
    IF cached:
        RETURN json.loads(cached)
    
    // Step 2: Get reference paper
    ref_paper = postgres.query(
        "SELECT * FROM papers WHERE paper_id = $1",
        paper_id
    )
    ref_embedding = weaviate.get_embedding(paper_id)
    
    // Step 3: Semantic search (Weaviate)
    semantic_papers = weaviate.search(
        vector=ref_embedding,
        filters={
            domain: ref_paper.domain,
            similarity: {">": threshold}
        },
        limit=limit * 2  // Get more to filter
    )
    
    // Step 4: Citation search (Neo4j)
    citation_papers = neo4j.query("""
        MATCH (ref:Paper {paper_id: $paper_id})
        MATCH (ref)-[:CITES]->(cited:Paper)
        RETURN cited
        UNION
        MATCH (ref:Paper {paper_id: $paper_id})
        MATCH (citing:Paper)-[:CITES]->(ref)
        RETURN citing
        UNION
        MATCH (ref:Paper {paper_id: $paper_id})
        MATCH (ref)-[:CITES]->(p1:Paper)
        MATCH (other:Paper)-[:CITES]->(p1)
        WHERE other.paper_id != $paper_id
        RETURN other
        LIMIT $limit
    """, paper_id=paper_id, limit=limit)
    
    // Step 5: Merge and deduplicate
    all_papers = semantic_papers + citation_papers
    unique_papers = deduplicate_by_paper_id(all_papers)
    
    // Step 6: Score by relevance
    FOR paper IN unique_papers:
        semantic_score = cosine_similarity(
            ref_embedding, 
            paper.embedding
        )
        
        // Check if has citation relationship
        citation_score = 0.0
        IF paper IN citation_papers:
            IF paper.cites(ref_paper):
                citation_score = 1.0
            ELIF ref_paper.cites(paper):
                citation_score = 1.0
            ELSE:  // Co-cited
                citation_score = 0.8
        
        paper.relevance_score = (
            0.7 * semantic_score + 
            0.3 * citation_score
        )
    
    // Step 7: Select top papers
    unique_papers.sort(key=lambda x: x.relevance_score, reverse=True)
    related_papers = unique_papers[:limit]
    
    // Step 8: Build graph structure
    graph = {
        "reference_paper": ref_paper,
        "related_papers": [],
        "edges": []
    }
    
    // Add nodes
    FOR paper IN related_papers:
        graph["related_papers"].append({
            "paper_id": paper.paper_id,
            "title": paper.title,
            "authors": paper.authors,
            "year": paper.year,
            "citation_count": paper.citation_count,
            "similarity_score": paper.relevance_score
        })
    
    // Add edges
    all_nodes = [ref_paper] + related_papers
    FOR p1 IN all_nodes:
        FOR p2 IN all_nodes:
            IF p1 != p2:
                // Check citation relationship
                IF neo4j.has_edge(p1.paper_id, p2.paper_id):
                    edge_type = "cites"
                    weight = 1.0
                ELSE:
                    // Check semantic similarity
                    sim = cosine_similarity(p1.embedding, p2.embedding)
                    IF sim > threshold:
                        edge_type = "semantic"
                        weight = sim
                    ELSE:
                        CONTINUE
                
                graph["edges"].append({
                    "source": p1.paper_id,
                    "target": p2.paper_id,
                    "weight": weight,
                    "type": edge_type
                })
    
    // Step 9: Compute force-directed layout
    G = NetworkGraph()
    FOR node IN all_nodes:
        G.add_node(node.paper_id)
    
    FOR edge IN graph["edges"]:
        G.add_edge(
            edge["source"], 
            edge["target"], 
            weight=edge["weight"]
        )
    
    // Spring layout (force-directed)
    positions = spring_layout(
        G,
        k=0.5,         // Optimal distance
        iterations=100,
        scale=500,     // Scale to 500x500 space
        center=(250, 250),
        seed=42
    )
    
    // Add positions to nodes
    FOR paper IN graph["related_papers"]:
        x, y = positions[paper["paper_id"]]
        paper["position_x"] = x
        paper["position_y"] = y
    
    // Reference paper at center
    graph["reference_paper"]["position_x"] = 250
    graph["reference_paper"]["position_y"] = 250
    
    // Step 10: Cache result
    redis.setex(
        key=cache_key,
        value=json.dumps(graph),
        ttl=3600  // 1 hour
    )
    
    RETURN graph
```

### 8.3 Semantic Search Algorithm

**Input**:
- query (string)
- domain (healthcare/fintech/quantum)
- filters (year range, citation count, etc.)
- limit (default 20)

**Output**:
- Ranked list of papers with relevance scores

**Algorithm**:
```
FUNCTION semantic_search(query, domain, filters, limit=20):
    
    // Step 1: Check cache
    cache_key = generate_cache_key(query, domain, filters)
    cached = redis.get(cache_key)
    IF cached:
        RETURN json.loads(cached)
    
    // Step 2: Generate query embedding
    query_embedding = SPECTER2.encode(query)
    
    // Step 3: Build Weaviate filter
    weaviate_filter = {
        "operator": "And",
        "operands": [
            {
                "path": ["domain"],
                "operator": "Equal",
                "valueText": domain
            }
        ]
    }
    
    IF filters.year_min:
        weaviate_filter["operands"].append({
            "path": ["year"],
            "operator": "GreaterThanEqual",
            "valueInt": filters.year_min
        })
    
    IF filters.year_max:
        weaviate_filter["operands"].append({
            "path": ["year"],
            "operator": "LessThanEqual",
            "valueInt": filters.year_max
        })
    
    // Step 4: Query Weaviate
    results = weaviate.query(
        collection="Paper",
        vector=query_embedding,
        filters=weaviate_filter,
        limit=limit,
        with_distance=True
    )
    
    // Step 5: Enrich with PostgreSQL metadata
    paper_ids = [r["paper_id"] for r in results]
    metadata = postgres.query("""
        SELECT paper_id, title, authors, year, venue, 
               citation_count, abstract, summary
        FROM papers
        WHERE paper_id = ANY($1)
    """, paper_ids)
    
    // Merge Weaviate results with PostgreSQL metadata
    enriched_results = []
    FOR result IN results:
        paper_meta = find_metadata(metadata, result["paper_id"])
        
        // Calculate relevance score
        semantic_sim = 1 - result["distance"]  // Convert distance to similarity
        
        // Keyword matching boost
        keyword_boost = calculate_keyword_match(
            query.lower().split(),
            paper_meta.title.lower() + " " + paper_meta.abstract.lower()
        )
        
        relevance_score = 0.85 * semantic_sim + 0.15 * keyword_boost
        
        enriched_results.append({
            "paper_id": paper_meta.paper_id,
            "title": paper_meta.title,
            "authors": paper_meta.authors,
            "year": paper_meta.year,
            "venue": paper_meta.venue,
            "citation_count": paper_meta.citation_count,
            "abstract": paper_meta.abstract,
            "summary": paper_meta.summary,
            "relevance_score": relevance_score,
            "match_explanation": {
                "semantic_similarity": semantic_sim,
                "keyword_matches": extract_keyword_matches(query, paper_meta),
                "confidence": get_confidence_level(relevance_score)
            }
        })
    
    // Step 6: Sort by relevance
    enriched_results.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    // Step 7: Cache results
    redis.setex(
        key=cache_key,
        value=json.dumps(enriched_results),
        ttl=1800  // 30 minutes
    )
    
    RETURN enriched_results

FUNCTION calculate_keyword_match(query_words, text):
    matches = 0
    FOR word IN query_words:
        IF word IN text:
            matches += 1
    RETURN matches / len(query_words)

FUNCTION get_confidence_level(score):
    IF score >= 0.85:
        RETURN "high"
    ELIF score >= 0.70:
        RETURN "medium"
    ELSE:
        RETURN "low"
```

---

## 9. Data Flow Diagrams

### 9.1 User Registration Flow

```
[User] → [Frontend: SignupForm]
    ↓ POST /api/v1/auth/register
    {email, password, name, domain, interests, google_scholar_url, uploaded_paper}
    ↓
[API Gateway: FastAPI]
    ↓ Validate input (Pydantic)
    ↓ Hash password (bcrypt)
    ↓
[UserService]
    ↓ INSERT INTO users
    ↓ INSERT INTO user_domains
    ↓ INSERT INTO user_interests
[PostgreSQL]
    ↓ user_id = 12345
    ↓
[IF google_scholar_url provided]
    ↓ Fetch author papers
[Semantic Scholar API]
    ↓ author_papers = [...]
    ↓ Generate author_profile_embedding
[SPECTER2 EmbeddingService]
    ↓ author_emb = [768-dim vector]
    ↓ INSERT INTO user_profile_embeddings
[PostgreSQL]
    ↓
[IF uploaded_paper provided]
    ↓ Extract text (PyPDF)
    ↓ Generate paper_embedding
[SPECTER2 EmbeddingService]
    ↓ paper_emb = [768-dim vector]
    ↓
[Trigger Async Task]
    ↓ celery.send_task("generate_starter_kit")
[Celery Queue → Redis]
    ↓
[Celery Worker]
    ↓ generate_starter_kit(user_id)
    ↓ [See Starter Kit Algorithm Section 8.1]
    ↓ Query Weaviate (semantic search)
    ↓ Query Neo4j (citation search)
    ↓ Query PostgreSQL (keyword/popular papers)
    ↓ Score candidates
    ↓ Cluster papers (k-means)
    ↓ Generate theme names (LLM)
    ↓ Compute layouts (force-directed)
    ↓ Store clusters
[PostgreSQL: paper_clusters, cluster_papers]
    ↓ Cache clusters
[Redis: starter_kit:{user_id}]
    ↓
[Response to Frontend]
    ↓ {user_id, access_token, starter_kit_status: "ready"}
[Frontend: Navigate to HomePage]
```

### 9.2 Home Page Load Flow

```
[User] → [Frontend: HomePage]
    ↓ GET /api/v1/users/me/home
    ↓ Headers: {Authorization: Bearer <token>}
    ↓
[API Gateway: FastAPI]
    ↓ Verify JWT token
    ↓ Extract user_id
    ↓
[UserService.get_home_clusters]
    ↓ Check cache
[Redis: starter_kit:{user_id}]
    ↓ IF cache HIT → return clusters
    ↓ IF cache MISS:
        ↓ Query PostgreSQL
[PostgreSQL]
    SELECT * FROM paper_clusters 
    WHERE user_id = 12345 
    AND expires_at > NOW()
    ↓ IF found → return clusters
    ↓ IF not found OR expired:
        ↓ Trigger regeneration
[ClusteringService.generate_home_clusters]
    ↓ [See Algorithm Section 8.1]
    ↓ Store + cache
    ↓
[Response to Frontend]
{
    "clusters": [
        {cluster_1},
        {cluster_2},
        {cluster_3}
    ]
}
    ↓
[Frontend Renders]
    ↓ IF view_type == "card":
        [ClusterCard components]
    ↓ ELSE IF view_type == "network":
        [CytoscapeGraph components]
```

### 9.3 Dynamic Graph Update Flow (Node Click)

```
[User clicks node in network graph]
    ↓ paper_id = "science:2021:rosettafold"
[Frontend: CytoscapeGraph]
    ↓ onNodeClick(paper_id)
    ↓ setLoading(true)
    ↓ GET /api/v1/papers/{paper_id}/graph
    ↓
[API Gateway: FastAPI]
    ↓ Verify auth
    ↓
[GraphService.generate_paper_graph]
    ↓ Check cache
[Redis: graph:{paper_id}]
    ↓ IF cache HIT → return graph
    ↓ IF cache MISS:
        ↓
    [Get reference paper embedding]
[Weaviate]
    ↓ ref_embedding = [768-dim]
        ↓
    [Semantic search]
[Weaviate.search(vector=ref_embedding, limit=50)]
    ↓ semantic_papers = [...]
        ↓
    [Citation search]
[Neo4j]
    MATCH (ref:Paper {paper_id: $paper_id})-[:CITES|CITED_BY]-(related)
    RETURN related
    ↓ citation_papers = [...]
        ↓
    [Merge + deduplicate]
    ↓ unique_papers = [...]
        ↓
    [Score papers]
    ↓ FOR EACH paper:
        relevance = 0.7*semantic_sim + 0.3*citation_rel
    ↓ Sort by relevance
    ↓ Select top 25 papers
        ↓
    [Build graph structure]
    ↓ nodes = [ref_paper] + related_papers
    ↓ edges = calculate_edges(nodes)
        ↓
    [Compute layout]
    ↓ positions = spring_layout(graph)
    ↓ FOR EACH node:
        node.position_x = positions[node.id].x
        node.position_y = positions[node.id].y
        ↓
    [Cache graph]
[Redis: graph:{paper_id}, TTL=1h]
    ↓
[Response to Frontend]
{
    "reference_paper": {...},
    "related_papers": [{...}],
    "edges": [{...}]
}
    ↓
[Frontend: CytoscapeGraph]
    ↓ Animate transition (fade out old graph)
    ↓ Update reference paper panel
    ↓ Re-render Cytoscape with new data
    ↓ Animate transition (fade in new graph)
    ↓ setLoading(false)
    ↓
[Track Interaction]
    ↓ POST /api/v1/interactions
    {
        paper_id: "science:2021:rosettafold",
        interaction_type: "click_node",
        context: {
            source_paper_id: "arxiv:2401.12345"
        }
    }
        ↓
[InteractionService]
    ↓ INSERT INTO user_interactions
[PostgreSQL]
```

### 9.4 Search Flow

```
[User types in search bar]
    ↓ query = "antibody design machine learning"
[Frontend: SearchBar]
    ↓ useDebounce(query, 500ms)
    ↓ GET /api/v1/search?q={query}&domain={domain}
    ↓
[API Gateway: FastAPI]
    ↓ Rate limit check
[Redis: rate_limit:{user_id}:search]
    ↓ IF exceeded → 429 Too Many Requests
    ↓ ELSE → proceed
    ↓
[SearchService.search]
    ↓ Check cache
[Redis: search:{hash(query)}:{domain}]
    ↓ IF cache HIT → return results
    ↓ IF cache MISS:
        ↓
    [Generate query embedding]
[SPECTER2 EmbeddingService]
    ↓ query_emb = encode_text(query)
        ↓
    [Query Weaviate]
[Weaviate]
    search(
        vector=query_emb,
        filters={domain: domain},
        limit=20
    )
    ↓ results = [{paper_id, distance}, ...]
        ↓
    [Enrich with metadata]
[PostgreSQL]
    SELECT * FROM papers 
    WHERE paper_id IN (...)
    ↓ metadata = [...]
        ↓
    [Calculate relevance scores]
    ↓ FOR EACH result:
        semantic_sim = 1 - result.distance
        keyword_boost = keyword_match(query, paper.text)
        relevance = 0.85*semantic_sim + 0.15*keyword_boost
        ↓
    [Sort by relevance]
    ↓ results.sort(key=relevance, reverse=True)
        ↓
    [Cache results]
[Redis: search:..., TTL=30min]
    ↓
[Response to Frontend]
{
    "query": "...",
    "total_results": 847,
    "results": [{...}]
}
    ↓
[Frontend: SearchResults]
    ↓ Render paper cards
    ↓ Highlight matching keywords
    ↓
[Track Search]
    ↓ POST /api/v1/interactions
    {
        interaction_type: "search",
        context: {query: "..."}
    }
```

### 9.5 User Interaction Tracking & Profile Update Flow

```
[User saves paper]
    ↓ POST /api/v1/papers/{paper_id}/save
[Frontend]
    ↓
[API Gateway]
    ↓
[PaperService.save_paper]
    ↓ INSERT INTO user_saved_papers
[PostgreSQL]
    ↓
[InteractionService.track_interaction]
    ↓ INSERT INTO user_interactions
    {
        user_id: 12345,
        paper_id: "nature:2023:antibody",
        interaction_type: "save",
        timestamp: NOW()
    }
[PostgreSQL]
    ↓
[Check if profile update needed]
    ↓ Count recent interactions
    SELECT COUNT(*) FROM user_interactions
    WHERE user_id = 12345
    AND created_at > (last_profile_update + interval '6 hours')
    ↓ IF count >= 10:
        ↓ Trigger profile update
[Celery Task: update_user_profile_embedding]
    ↓
[Celery Worker]
    ↓ Fetch saved + liked papers (last 30 days)
[PostgreSQL]
    SELECT paper_id FROM user_saved_papers
    WHERE user_id = 12345
    AND saved_at > NOW() - interval '30 days'
    ↓ paper_ids = [...]
        ↓
    [Get embeddings for papers]
[Weaviate]
    ↓ embeddings = [...]
        ↓
    [Calculate weighted average]
    saved_weight = 0.5
    liked_weight = 0.3
    viewed_weight = 0.2
    
    new_profile_emb = (
        saved_weight * avg(saved_embeddings) +
        liked_weight * avg(liked_embeddings) +
        viewed_weight * avg(long_viewed_embeddings)
    )
        ↓
    [Update profile]
[PostgreSQL]
    UPDATE user_profile_embeddings
    SET embedding_vector = new_profile_emb,
        last_updated = NOW(),
        interaction_count = interaction_count + 10
    WHERE user_id = 12345
        ↓
    [Invalidate cluster cache]
[Redis]
    DELETE starter_kit:12345
        ↓
    [Next home page visit → regenerate clusters]
```

---

## 10. Security & Authentication

### 10.1 Authentication Flow

**JWT Token Structure**:
```json
{
    "header": {
        "alg": "HS256",
        "typ": "JWT"
    },
    "payload": {
        "sub": "12345",          // user_id
        "email": "user@example.com",
        "domain": "healthcare",
        "exp": 1699459200,       // Expiration timestamp
        "iat": 1699372800,       // Issued at timestamp
        "type": "access"         // "access" or "refresh"
    },
    "signature": "..."
}
```

**Token Generation**:
```python
from jose import jwt
from datetime import datetime, timedelta

def create_access_token(user_id: int, email: str, domain: str) -> str:
    """Generate JWT access token"""
    payload = {
        "sub": str(user_id),
        "email": email,
        "domain": domain,
        "exp": datetime.utcnow() + timedelta(hours=24),
        "iat": datetime.utcnow(),
        "type": "access"
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return token

def verify_token(token: str) -> dict:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

**Password Hashing**:
```python
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)
```

### 10.2 API Security

**Rate Limiting**:
```python
from fastapi import Request, HTTPException
import time

async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware"""
    user_id = get_user_id_from_token(request)
    endpoint = request.url.path
    
    # Check rate limit
    key = f"rate_limit:{user_id}:{endpoint}"
    current_count = redis.get(key) or 0
    
    if int(current_count) >= RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again in 60 seconds."
        )
    
    # Increment counter
    redis.incr(key)
    redis.expire(key, 60)  # 1 minute window
    
    response = await call_next(request)
    return response

# Rate limits per endpoint
RATE_LIMITS = {
    "/api/v1/search": 100,      # 100 requests/minute
    "/api/v1/papers/*": 200,    # 200 requests/minute
    "/api/v1/graph/*": 50,      # 50 requests/minute (expensive)
}
```

**Input Validation**:
```python
from pydantic import BaseModel, EmailStr, constr, validator

class UserRegistration(BaseModel):
    email: EmailStr
    password: constr(min_length=8, max_length=100)
    name: constr(min_length=2, max_length=255)
    domain: Literal["healthcare", "fintech", "quantum_computing"]
    interests: List[constr(min_length=2, max_length=100)]
    
    @validator('password')
    def password_strength(cls, v):
        """Validate password strength"""
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one digit')
        if not any(char.isupper() for char in v):
            raise ValueError('Password must contain at least one uppercase letter')
        return v
    
    @validator('interests')
    def validate_interests(cls, v):
        """Validate interests list"""
        if len(v) < 1:
            raise ValueError('At least one interest required')
        if len(v) > 10:
            raise ValueError('Maximum 10 interests allowed')
        return v
```

**SQL Injection Prevention**:
```python
# Always use parameterized queries
# NEVER concatenate user input into SQL

# ❌ UNSAFE (SQL injection vulnerability)
query = f"SELECT * FROM users WHERE email = '{email}'"

# ✅ SAFE (parameterized query)
query = "SELECT * FROM users WHERE email = %s"
result = cursor.execute(query, (email,))
```

**CORS Configuration**:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://citeconnect.com",
        "https://www.citeconnect.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
```

### 10.3 Environment Variables

```bash
# .env.example

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/citeconnect
WEAVIATE_URL=http://localhost:8080
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
REDIS_URL=redis://localhost:6379/0

# JWT
SECRET_KEY=your-secret-key-min-32-characters-long
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_HOURS=24

# GCP
GCP_PROJECT_ID=citeconnect-prod
GCS_BUCKET_NAME=citeconnect-papers
GCP_CREDENTIALS_PATH=/path/to/credentials.json

# External APIs
SEMANTIC_SCHOLAR_API_KEY=your-api-key
OPENAI_API_KEY=your-openai-key  # For LLM theme generation

# SPECTER2 Model
SPECTER_MODEL_NAME=allenai/specter2
SPECTER_CACHE_DIR=/models/specter2

# Rate Limiting
RATE_LIMIT_PER_MINUTE=100

# Celery
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2
```

---

## 11. Performance & Caching Strategy

### 11.1 Caching Layers

**Layer 1: Application Cache (Redis)**
```python
# Cache configuration
CACHE_TTL = {
    "user_session": 86400,        # 24 hours
    "starter_kit": 86400,         # 24 hours
    "cluster": 3600,              # 1 hour
    "graph": 3600,                # 1 hour
    "search_results": 1800,       # 30 minutes
    "paper_metadata": 86400,      # 24 hours
    "user_embedding": 21600,      # 6 hours
}

# Cache invalidation rules
def invalidate_user_cache(user_id: int):
    """Invalidate all user-related caches"""
    redis.delete(f"starter_kit:{user_id}")
    redis.delete(f"user:embedding:{user_id}")
    # Clusters will expire naturally or be regenerated

def invalidate_paper_cache(paper_id: str):
    """Invalidate paper-related caches"""
    redis.delete(f"graph:{paper_id}")
    redis.delete(f"paper:meta:{paper_id}")
```

**Layer 2: Database Query Cache**
```python
# PostgreSQL query result caching
@cache_result(ttl=3600)
def get_popular_papers(domain: str, limit: int) -> List[Dict]:
    """Cache popular papers query"""
    query = """
        SELECT * FROM papers
        WHERE domain = %s
        ORDER BY citation_count DESC
        LIMIT %s
    """
    return db.execute(query, (domain, limit))
```

**Layer 3: CDN Caching (for static assets)**
```
# Cloud CDN configuration
- Frontend assets (JS, CSS): Cache 1 year
- Images, icons: Cache 1 year
- API responses: No CDN caching (dynamic)
```

### 11.2 Database Optimization

**PostgreSQL Indexes**:
```sql
-- Primary indexes (already defined in schema)
CREATE INDEX idx_papers_domain ON papers(domain);
CREATE INDEX idx_papers_year ON papers(year);
CREATE INDEX idx_papers_citation_count ON papers(citation_count);

-- Composite indexes for common queries
CREATE INDEX idx_papers_domain_year ON papers(domain, year);
CREATE INDEX idx_user_interactions_user_type ON user_interactions(user_id, interaction_type);

-- GIN indexes for full-text search
CREATE INDEX idx_papers_title_gin ON papers USING gin(to_tsvector('english', title));
CREATE INDEX idx_papers_abstract_gin ON papers USING gin(to_tsvector('english', abstract));

-- Partial indexes for active users
CREATE INDEX idx_active_users ON users(user_id) WHERE is_active = TRUE;
```

**Neo4j Indexes**:
```cypher
-- Node property indexes
CREATE INDEX paper_id_index FOR (p:Paper) ON (p.paper_id);
CREATE INDEX paper_domain_index FOR (p:Paper) ON (p.domain);
CREATE INDEX user_id_index FOR (u:User) ON (u.user_id);

-- Composite indexes
CREATE INDEX paper_domain_year FOR (p:Paper) ON (p.domain, p.year);
```

**Weaviate Optimization**:
```python
# Weaviate HNSW configuration
{
    "vectorIndexConfig": {
        "distance": "cosine",
        "ef": 64,              # Query-time accuracy (higher = more accurate but slower)
        "efConstruction": 128, # Index-time accuracy
        "maxConnections": 32   # Graph connectivity (higher = better recall)
    }
}

# Batch import optimization
weaviate.batch.configure(
    batch_size=100,
    dynamic=True,
    timeout_retries=3,
    callback=None,
)
```

### 11.3 Query Optimization

**Pagination**:
```python
# Efficient pagination using keyset/cursor-based pagination
def get_papers_paginated(domain: str, cursor: str = None, limit: int = 20):
    """
    Cursor-based pagination (more efficient than OFFSET)
    """
    if cursor:
        query = """
            SELECT * FROM papers
            WHERE domain = %s
            AND paper_id > %s
            ORDER BY paper_id
            LIMIT %s
        """
        results = db.execute(query, (domain, cursor, limit))
    else:
        query = """
            SELECT * FROM papers
            WHERE domain = %s
            ORDER BY paper_id
            LIMIT %s
        """
        results = db.execute(query, (domain, limit))
    
    next_cursor = results[-1]['paper_id'] if results else None
    return results, next_cursor
```

**Batch Operations**:
```python
# Batch embedding generation
def generate_embeddings_batch(papers: List[Dict], batch_size: int = 32):
    """Generate embeddings in batches for efficiency"""
    embeddings = []
    
    for i in range(0, len(papers), batch_size):
        batch = papers[i:i+batch_size]
        texts = [f"{p['title']} {p['abstract']}" for p in batch]
        
        # Single model forward pass for entire batch
        batch_embeddings = embedding_service.encode_batch(texts)
        embeddings.extend(batch_embeddings)
    
    return embeddings
```

### 11.4 Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| **API Response Time (p95)** | < 2 seconds | Prometheus + Grafana |
| **Graph Generation** | < 3 seconds | Backend timing logs |
| **Search Query** | < 1 second | Weaviate metrics |
| **Database Queries** | < 500ms | PostgreSQL slow query log |
| **Page Load Time** | < 3 seconds | Lighthouse metrics |
| **Cache Hit Rate** | > 70% | Redis INFO stats |

---

## 12. Error Handling & Monitoring

### 12.1 Error Handling

**Exception Hierarchy**:
```python
# app/core/exceptions.py

class CiteConnectException(Exception):
    """Base exception for CiteConnect"""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class AuthenticationError(CiteConnectException):
    """Authentication failed"""
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, status_code=401)

class AuthorizationError(CiteConnectException):
    """User not authorized"""
    def __init__(self, message: str = "Not authorized"):
        super().__init__(message, status_code=403)

class ResourceNotFoundError(CiteConnectException):
    """Resource not found"""
    def __init__(self, resource: str, identifier: str):
        message = f"{resource} with id '{identifier}' not found"
        super().__init__(message, status_code=404)

class ValidationError(CiteConnectException):
    """Input validation failed"""
    def __init__(self, message: str):
        super().__init__(message, status_code=400)

class RateLimitError(CiteConnectException):
    """Rate limit exceeded"""
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, status_code=429)

class DatabaseError(CiteConnectException):
    """Database operation failed"""
    def __init__(self, message: str):
        super().__init__(message, status_code=500)

class ExternalServiceError(CiteConnectException):
    """External service call failed"""
    def __init__(self, service: str, message: str):
        msg = f"{service} service error: {message}"
        super().__init__(msg, status_code=503)
```

**Global Exception Handler**:
```python
# app/main.py

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.core.exceptions import CiteConnectException
import logging

app = FastAPI()
logger = logging.getLogger(__name__)

@app.exception_handler(CiteConnectException)
async def citeconnect_exception_handler(request: Request, exc: CiteConnectException):
    """Handle custom exceptions"""
    logger.error(f"CiteConnect Error: {exc.message}", extra={
        "status_code": exc.status_code,
        "path": request.url.path,
        "method": request.method
    })
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "type": exc.__class__.__name__,
                "status_code": exc.status_code
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.exception(f"Unexpected error: {str(exc)}", extra={
        "path": request.url.path,
        "method": request.method
    })
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "An unexpected error occurred",
                "type": "InternalServerError",
                "status_code": 500
            }
        }
    )
```

**Service-Level Error Handling**:
```python
# app/services/search_service.py

from app.core.exceptions import (
    ExternalServiceError, 
    DatabaseError,
    ValidationError
)
import logging

logger = logging.getLogger(__name__)

class SearchService:
    def search(self, query: str, domain: str, limit: int) -> List[Dict]:
        try:
            # Validate inputs
            if not query or len(query) < 2:
                raise ValidationError("Query must be at least 2 characters")
            
            if limit > 100:
                raise ValidationError("Limit cannot exceed 100")
            
            # Generate embedding
            try:
                query_embedding = self.embedding_service.encode_text(query)
            except Exception as e:
                logger.error(f"Embedding generation failed: {str(e)}")
                raise ExternalServiceError("SPECTER2", "Failed to generate embedding")
            
            # Query Weaviate
            try:
                results = self.weaviate.search(
                    vector=query_embedding,
                    filters={"domain": domain},
                    limit=limit
                )
            except Exception as e:
                logger.error(f"Weaviate query failed: {str(e)}")
                raise DatabaseError("Vector search failed")
            
            # Enrich with metadata
            try:
                enriched = self._enrich_with_metadata(results)
            except Exception as e:
                logger.error(f"Metadata enrichment failed: {str(e)}")
                # Don't fail completely, return partial results
                enriched = results
            
            return enriched
            
        except CiteConnectException:
            # Re-raise custom exceptions
            raise
        except Exception as e:
            # Catch any other exception
            logger.exception(f"Unexpected error in search: {str(e)}")
            raise DatabaseError("Search operation failed")
```

### 12.2 Logging Configuration

**Structured Logging**:
```python
# app/core/logging.py

import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """Format logs as JSON for better parsing"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id
        if hasattr(record, 'duration_ms'):
            log_data['duration_ms'] = record.duration_ms
        
        # Add exception info
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)

# Configure logging
def setup_logging():
    """Setup application logging"""
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(console_handler)
    
    # File handler (for local development)
    file_handler = logging.FileHandler('logs/citeconnect.log')
    file_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(file_handler)
    
    # Set levels for external libraries
    logging.getLogger('uvicorn').setLevel(logging.WARNING)
    logging.getLogger('fastapi').setLevel(logging.INFO)
```

**Request Logging Middleware**:
```python
# app/middleware/logging_middleware.py

import time
import uuid
from fastapi import Request
import logging

logger = logging.getLogger(__name__)

async def logging_middleware(request: Request, call_next):
    """Log all requests with timing"""
    
    # Generate request ID
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Start timer
    start_time = time.time()
    
    # Log request
    logger.info(f"Request started", extra={
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "client_ip": request.client.host,
        "user_agent": request.headers.get("user-agent")
    })
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration_ms = (time.time() - start_time) * 1000
    
    # Log response
    logger.info(f"Request completed", extra={
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "duration_ms": round(duration_ms, 2)
    })
    
    # Add request ID to response headers
    response.headers["X-Request-ID"] = request_id
    
    return response
```

### 12.3 Monitoring Setup

**Prometheus Metrics**:
```python
# app/monitoring/metrics.py

from prometheus_client import Counter, Histogram, Gauge
from functools import wraps
import time

# Define metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

ACTIVE_REQUESTS = Gauge(
    'http_requests_active',
    'Active HTTP requests',
    ['method', 'endpoint']
)

EMBEDDING_GENERATION_TIME = Histogram(
    'embedding_generation_seconds',
    'Time to generate embeddings',
    ['model']
)

WEAVIATE_QUERY_TIME = Histogram(
    'weaviate_query_duration_seconds',
    'Weaviate query duration'
)

CACHE_HITS = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_key']
)

CACHE_MISSES = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['cache_key']
)

# Decorator for timing functions
def track_duration(metric: Histogram, **labels):
    """Decorator to track function duration"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                metric.labels(**labels).observe(duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                metric.labels(**labels).observe(duration)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# Usage example
@track_duration(EMBEDDING_GENERATION_TIME, model="specter2")
def generate_embedding(text: str) -> np.ndarray:
    """Generate embedding with timing"""
    return specter_model.encode(text)
```

**Metrics Endpoint**:
```python
# app/api/v1/monitoring.py

from fastapi import APIRouter
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

router = APIRouter()

@router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
```

**Health Check Endpoint**:
```python
# app/api/v1/health.py

from fastapi import APIRouter, status
from pydantic import BaseModel
import asyncpg
import redis
import weaviate

router = APIRouter()

class HealthCheck(BaseModel):
    status: str
    postgres: str
    redis: str
    weaviate: str
    neo4j: str

@router.get("/health", response_model=HealthCheck)
async def health_check():
    """System health check"""
    health = {
        "status": "healthy",
        "postgres": "unknown",
        "redis": "unknown",
        "weaviate": "unknown",
        "neo4j": "unknown"
    }
    
    # Check PostgreSQL
    try:
        await db.execute("SELECT 1")
        health["postgres"] = "healthy"
    except Exception:
        health["postgres"] = "unhealthy"
        health["status"] = "degraded"
    
    # Check Redis
    try:
        redis_client.ping()
        health["redis"] = "healthy"
    except Exception:
        health["redis"] = "unhealthy"
        health["status"] = "degraded"
    
    # Check Weaviate
    try:
        weaviate_client.is_ready()
        health["weaviate"] = "healthy"
    except Exception:
        health["weaviate"] = "unhealthy"
        health["status"] = "degraded"
    
    # Check Neo4j
    try:
        with neo4j_driver.session() as session:
            session.run("RETURN 1")
        health["neo4j"] = "healthy"
    except Exception:
        health["neo4j"] = "unhealthy"
        health["status"] = "degraded"
    
    # Return appropriate status code
    status_code = status.HTTP_200_OK if health["status"] == "healthy" else status.HTTP_503_SERVICE_UNAVAILABLE
    
    return Response(content=health, status_code=status_code)
```

**Alerting Rules** (Prometheus):
```yaml
# alerting_rules.yml

groups:
  - name: citeconnect_alerts
    interval: 30s
    rules:
      # API latency alerts
      - alert: HighAPILatency
        expr: histogram_quantile(0.95, http_request_duration_seconds) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High API latency detected"
          description: "P95 latency is {{ $value }}s (threshold: 2s)"
      
      # Error rate alerts
      - alert: HighErrorRate
        expr: rate(http_requests_total{status_code=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} (threshold: 5%)"
      
      # Cache hit rate
      - alert: LowCacheHitRate
        expr: rate(cache_hits_total[5m]) / (rate(cache_hits_total[5m]) + rate(cache_misses_total[5m])) < 0.7
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low cache hit rate"
          description: "Cache hit rate is {{ $value }} (threshold: 70%)"
      
      # Database connection
      - alert: DatabaseDown
        expr: up{job="postgres"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "PostgreSQL database is down"
      
      # Weaviate availability
      - alert: WeaviateDown
        expr: up{job="weaviate"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Weaviate vector database is down"
```

---

## 13. Deployment Configuration

### 13.1 Docker Configuration

**Backend Dockerfile**:
```dockerfile
# backend/Dockerfile

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download SPECTER2 model (do this during build to cache)
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('allenai/specter2')"

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

**Frontend Dockerfile**:
```dockerfile
# frontend/Dockerfile

# Build stage
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package.json package-lock.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Build application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built assets
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD wget -q --spider http://localhost:80 || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
```

**Docker Compose** (Local Development):
```yaml
# docker-compose.yml

version: '3.8'

services:
  # PostgreSQL
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: citeconnect
      POSTGRES_PASSWORD: password
      POSTGRES_DB: citeconnect
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_db.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U citeconnect"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5

  # Neo4j
  neo4j:
    image: neo4j:5
    environment:
      NEO4J_AUTH: neo4j/password
      NEO4J_PLUGINS: '["apoc"]'
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    volumes:
      - neo4j_data:/data
    healthcheck:
      test: ["CMD", "cypher-shell", "-u", "neo4j", "-p", "password", "RETURN 1"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Weaviate
  weaviate:
    image: semitechnologies/weaviate:1.22.4
    environment:
      QUERY_DEFAULTS_LIMIT: 20
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      CLUSTER_HOSTNAME: 'node1'
    ports:
      - "8080:8080"
    volumes:
      - weaviate_data:/var/lib/weaviate
    healthcheck:
      test: ["CMD", "wget", "--spider", "http://localhost:8080/v1/.well-known/ready"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Backend API
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    environment:
      DATABASE_URL: postgresql://citeconnect:password@postgres:5432/citeconnect
      REDIS_URL: redis://redis:6379/0
      WEAVIATE_URL: http://weaviate:8080
      NEO4J_URI: bolt://neo4j:7687
      NEO4J_USER: neo4j
      NEO4J_PASSWORD: password
      SECRET_KEY: dev-secret-key-change-in-production
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app
      - model_cache:/root/.cache/huggingface
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      weaviate:
        condition: service_healthy
      neo4j:
        condition: service_healthy

  # Celery Worker
  celery_worker:
    build:
      context: ./backend
      dockerfile: Dockerfile
    command: celery -A app.tasks.celery_app worker --loglevel=info
    environment:
      DATABASE_URL: postgresql://citeconnect:password@postgres:5432/citeconnect
      REDIS_URL: redis://redis:6379/0
      CELERY_BROKER_URL: redis://redis:6379/1
      CELERY_RESULT_BACKEND: redis://redis:6379/2
      WEAVIATE_URL: http://weaviate:8080
      NEO4J_URI: bolt://neo4j:7687
      NEO4J_USER: neo4j
      NEO4J_PASSWORD: password
    volumes:
      - ./backend:/app
      - model_cache:/root/.cache/huggingface
    depends_on:
      - postgres
      - redis
      - weaviate
      - neo4j

  # Frontend
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:80"
    depends_on:
      - backend

  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  # Grafana
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus

volumes:
  postgres_data:
  redis_data:
  neo4j_data:
  weaviate_data:
  model_cache:
  prometheus_data:
  grafana_data:
```

### 13.2 Kubernetes Configuration

**Backend Deployment**:
```yaml
# k8s/backend-deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: citeconnect-backend
  namespace: citeconnect
spec:
  replicas: 3
  selector:
    matchLabels:
      app: citeconnect-backend
  template:
    metadata:
      labels:
        app: citeconnect-backend
    spec:
      containers:
      - name: backend
        image: gcr.io/citeconnect-prod/backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: citeconnect-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: citeconnect-secrets
              key: redis-url
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: citeconnect-secrets
              key: jwt-secret
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
      imagePullSecrets:
      - name: gcr-json-key

---
apiVersion: v1
kind: Service
metadata:
  name: citeconnect-backend-service
  namespace: citeconnect
spec:
  selector:
    app: citeconnect-backend
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

**Horizontal Pod Autoscaler**:
```yaml
# k8s/backend-hpa.yaml

apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: citeconnect-backend-hpa
  namespace: citeconnect
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: citeconnect-backend
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
      - type: Pods
        value: 2
        periodSeconds: 30
```

### 13.3 CI/CD Pipeline

**GitHub Actions Workflow**:
```yaml
# .github/workflows/deploy.yml

name: Build and Deploy

on:
  push:
    branches: [main, staging]
  pull_request:
    branches: [main]

env:
  GCP_PROJECT_ID: citeconnect-prod
  GKE_CLUSTER: citeconnect-cluster
  GKE_ZONE: us-central1-a

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        cd backend
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        cd backend
        pytest tests/ --cov=app --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./backend/coverage.xml

  build-backend:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    steps:
    - uses: actions/checkout@v3
    
    - name: Authenticate to Google Cloud
      uses: google-github-actions/auth@v1
      with:
        credentials_json: ${{ secrets.GCP_SA_KEY }}
    
    - name: Set up Cloud SDK
      uses: google-github-actions/setup-gcloud@v1
    
    - name: Configure Docker
      run: gcloud auth configure-docker
    
    - name: Build and push backend image
      run: |
        cd backend
        docker build -t gcr.io/$GCP_PROJECT_ID/backend:$GITHUB_SHA .
        docker push gcr.io/$GCP_PROJECT_ID/backend:$GITHUB_SHA
        docker tag gcr.io/$GCP_PROJECT_ID/backend:$GITHUB_SHA gcr.io/$GCP_PROJECT_ID/backend:latest
        docker push gcr.io/$GCP_PROJECT_ID/backend:latest

  build-frontend:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
    
    - name: Build frontend
      run: |
        cd frontend
        npm ci
        npm run build
    
    - name: Authenticate to Google Cloud
      uses: google-github-actions/auth@v1
      with:
        credentials_json: ${{ secrets.GCP_SA_KEY }}
    
    - name: Build and push frontend image
      run: |
        cd frontend
        docker build -t gcr.io/$GCP_PROJECT_ID/frontend:$GITHUB_SHA .
        docker push gcr.io/$GCP_PROJECT_ID/frontend:$GITHUB_SHA
        docker tag gcr.io/$GCP_PROJECT_ID/frontend:$GITHUB_SHA gcr.io/$GCP_PROJECT_ID/frontend:latest
        docker push gcr.io/$GCP_PROJECT_ID/frontend:latest

  deploy:
    needs: [build-backend, build-frontend]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3
    
    - name: Authenticate to Google Cloud
      uses: google-github-actions/auth@v1
      with:
        credentials_json: ${{ secrets.GCP_SA_KEY }}
    
    - name: Get GKE credentials
      run: |
        gcloud container clusters get-credentials $GKE_CLUSTER --zone $GKE_ZONE
    
    - name: Deploy to GKE
      run: |
        kubectl set image deployment/citeconnect-backend \
          backend=gcr.io/$GCP_PROJECT_ID/backend:$GITHUB_SHA \
          --namespace=citeconnect
        
        kubectl set image deployment/citeconnect-frontend \
          frontend=gcr.io/$GCP_PROJECT_ID/frontend:$GITHUB_SHA \
          --namespace=citeconnect
        
        kubectl rollout status deployment/citeconnect-backend --namespace=citeconnect
        kubectl rollout status deployment/citeconnect-frontend --namespace=citeconnect
```

---

## 14. Implementation Guidelines

### 14.1 Development Workflow

**Phase 1: Foundation (Week 1-2)**
1. Set up project structure (as defined in Section 4)
2. Configure databases (PostgreSQL, Neo4j, Weaviate, Redis)
3. Implement authentication (JWT, password hashing)
4. Create base API structure (FastAPI app, routes, middleware)
5. Set up SPECTER2 embedding service
6. Write database initialization scripts

**Phase 2: Core Services (Week 3-5)**
1. Implement EmbeddingService with SPECTER2
2. Implement SearchService (Weaviate integration)
3. Implement GraphService (Neo4j + layout computation)
4. Implement ClusteringService (k-means + theme generation)
5. Implement RecommendationService (starter kit generation)
6. Set up Celery for async tasks

**Phase 3: Frontend (Week 3-5, Parallel)**
1. Set up React project with TypeScript
2. Implement authentication pages (login, signup)
3. Implement home page with cluster cards
4. Implement CytoscapeGraph component
5. Implement search functionality
6. Implement paper detail page
7. Implement dashboard

**Phase 4: Integration (Week 6-7)**
1. Connect frontend to backend APIs
2. Implement real-time graph updates
3. Test end-to-end user flows
4. Implement interaction tracking
5. Test personalization loop
6. Performance optimization

**Phase 5: Testing & Polish (Week 8-9)**
1. Write comprehensive unit tests
2. Write integration tests
3. Performance testing & optimization
4. Security testing
5. UI/UX improvements
6. Documentation

**Phase 6: Deployment (Week 10-11)**
1. Set up GCP infrastructure
2. Configure CI/CD pipeline
3. Deploy to staging environment
4. Load testing
5. Deploy to production
6. Monitoring setup

**Phase 7: Presentation Prep (Week 12)**
1. Prepare demo scenarios
2. Create presentation slides
3. Record demo video
4. Practice presentation
5. Final bug fixes

### 14.2 Code Quality Standards

**Python (Backend)**:
```python
# Use type hints
def generate_embedding(text: str) -> np.ndarray:
    """Generate SPECTER2 embedding for text"""
    pass

# Use Pydantic for validation
from pydantic import BaseModel, validator

class PaperSchema(BaseModel):
    paper_id: str
    title: str
    year: int
    
    @validator('year')
    def validate_year(cls, v):
        if v < 1900 or v > 2030:
            raise ValueError('Invalid year')
        return v

# Use docstrings (Google style)
def search(query: str, domain: str, limit: int = 20) -> List[Dict]:
    """
    Perform semantic search for papers.
    
    Args:
        query: Search query string
        domain: Filter by domain (healthcare/fintech/quantum)
        limit: Maximum number of results (default: 20)
    
    Returns:
        List of paper dictionaries with similarity scores
    
    Raises:
        ValidationError: If query is invalid
        DatabaseError: If search fails
    """
    pass

# Use logging, not print
logger.info(f"Searching for: {query}")
logger.error(f"Search failed: {error}")

# Follow PEP 8
# - 4 spaces for indentation
# - Max line length: 100 characters
# - snake_case for functions/variables
# - PascalCase for classes
```

**TypeScript (Frontend)**:
```typescript
// Use explicit types
interface Paper {
  paper_id: string;
  title: string;
  authors: string[];
  year: number;
  citation_count: number;
}

// Use interfaces for props
interface PaperCardProps {
  paper: Paper;
  onSave: (paperId: string) => void;
  onLike: (paperId: string) => void;
}

// Use functional components with hooks
const PaperCard: React.FC<PaperCardProps> = ({ paper, onSave, onLike }) => {
  const [isSaved, setIsSaved] = useState(false);
  
  return (
    <div className="paper-card">
      {/* ... */}
    </div>
  );
};

// Use async/await for API calls
const searchPapers = async (query: string): Promise<Paper[]> => {
  try {
    const response = await api.get<SearchResponse>('/search', {
      params: { q: query }
    });
    return response.data.results;
  } catch (error) {
    console.error('Search failed:', error);
    throw error;
  }
};
```

### 14.3 Testing Strategy

**Backend Tests**:
```python
# Unit tests (pytest)
# tests/test_services/test_embedding_service.py

import pytest
from app.services.embedding_service import EmbeddingService

@pytest.fixture
def embedding_service():
    return EmbeddingService(model_name="allenai/specter2")

def test_encode_text(embedding_service):
    """Test text encoding"""
    text = "machine learning for healthcare"
    embedding = embedding_service.encode_text(text)
    
    assert embedding.shape == (768,)
    assert embedding.dtype == np.float32

def test_encode_paper(embedding_service):
    """Test paper encoding"""
    title = "AlphaFold: Protein structure prediction"
    abstract = "This paper presents AlphaFold..."
    
    embedding = embedding_service.encode_paper(title, abstract)
    
    assert embedding.shape == (768,)

def test_similarity_computation(embedding_service):
    """Test similarity calculation"""
    emb1 = np.random.rand(768)
    emb2 = np.random.rand(768)
    
    sim = embedding_service.compute_similarity(emb1, emb2)
    
    assert 0 <= sim <= 1
```

**Integration Tests**:
```python
# tests/test_integration/test_search_flow.py

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@pytest.fixture
def auth_headers(test_user):
    """Get authentication headers"""
    response = client.post("/api/v1/auth/login", json={
        "email": test_user.email,
        "password": "testpassword"
    })
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

def test_search_flow(auth_headers):
    """Test complete search flow"""
    # Perform search
    response = client.get(
        "/api/v1/search",
        params={"q": "machine learning", "limit": 10},
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) <= 10
    
    # Check result structure
    result = data["results"][0]
    assert "paper_id" in result
    assert "title" in result
    assert "relevance_score" in result
```

**Frontend Tests** (React Testing Library):
```typescript
// src/components/graph/__tests__/CytoscapeGraph.test.tsx

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { CytoscapeGraph } from '../CytoscapeGraph';

const mockGraphData = {
  reference_paper: {
    paper_id: 'test-1',
    title: 'Test Paper',
    // ...
  },
  related_papers: [
    // ...
  ],
  edges: [
    // ...
  ]
};

describe('CytoscapeGraph', () => {
  it('renders graph with nodes', async () => {
    const onNodeClick = jest.fn();
    
    render(
      <CytoscapeGraph 
        graphData={mockGraphData}
        onNodeClick={onNodeClick}
        onNodeHover={() => {}}
      />
    );
    
    await waitFor(() => {
      expect(screen.getByRole('presentation')).toBeInTheDocument();
    });
  });
  
  it('calls onNodeClick when node is clicked', async () => {
    const onNodeClick = jest.fn();
    
    render(
      <CytoscapeGraph 
        graphData={mockGraphData}
        onNodeClick={onNodeClick}
        onNodeHover={() => {}}
      />
    );
    
    // Simulate node click
    // (Note: Cytoscape testing requires special setup)
  });
});
```

### 14.4 Performance Optimization Checklist

- [ ] Enable Redis caching for all expensive operations
- [ ] Implement database connection pooling
- [ ] Use batch operations for embedding generation
- [ ] Pre-compute and cache cluster layouts
- [ ] Implement lazy loading for paper lists
- [ ] Use CDN for static assets
- [ ] Optimize database queries with proper indexes
- [ ] Implement pagination for large result sets
- [ ] Use React.memo for expensive components
- [ ] Implement virtual scrolling for long lists
- [ ] Compress API responses (gzip)
- [ ] Use HTTP/2 for multiplexing
- [ ] Implement request debouncing for search
- [ ] Use Web Workers for expensive frontend calculations

### 14.5 Security Checklist

- [ ] Use HTTPS only (no HTTP)
- [ ] Implement JWT token expiration and refresh
- [ ] Hash all passwords with bcrypt
- [ ] Validate all user inputs (Pydantic)
- [ ] Implement rate limiting per user/endpoint
- [ ] Use parameterized SQL queries (prevent injection)
- [ ] Sanitize all user-generated content
- [ ] Implement CORS properly
- [ ] Store secrets in environment variables
- [ ] Use GCP Secret Manager for production
- [ ] Implement API key rotation
- [ ] Enable database encryption at rest
- [ ] Use secure session management (Redis)
- [ ] Implement audit logging for sensitive operations
- [ ] Regular security scanning (Snyk, Dependabot)

---

## 15. Appendix

### 15.1 Glossary

| Term | Definition |
|------|------------|
| **SPECTER2** | Scientific Paper Embeddings using Citation-informed TransformER (version 2). A model fine-tuned for generating embeddings of research papers. |
| **Embedding** | A dense vector representation (768 dimensions) of text or papers, used for semantic similarity. |
| **Cosine Similarity** | A measure of similarity between two vectors, ranging from 0 (orthogonal) to 1 (identical). |
| **Force-Directed Layout** | A graph layout algorithm that positions nodes based on attractive/repulsive forces. |
| **K-Means Clustering** | An unsupervised learning algorithm that partitions data into k clusters. |
| **Citation Graph** | A directed graph where nodes are papers and edges represent citation relationships. |
| **Semantic Search** | Search based on meaning rather than exact keyword matching. |
| **Cold Start** | The problem of making recommendations for new users with no interaction history. |
| **TTL** | Time To Live - duration before cached data expires. |
| **p95 Latency** | The 95th percentile of response times (95% of requests complete faster). |

### 15.2 Environment Setup Guide

**Local Development Setup**:
```bash
# 1. Clone repository
git clone https://github.com/your-org/citeconnect.git
cd citeconnect

# 2. Set up Python environment (backend)
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 3. Download SPECTER2 model
python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('allenai/specter2')"

# 4. Set up Node.js environment (frontend)
cd ../frontend
npm install

# 5. Set up databases with Docker Compose
cd ..
docker-compose up -d postgres redis neo4j weaviate

# 6. Initialize databases
cd backend
alembic upgrade head
python scripts/seed_data.py

# 7. Start backend
uvicorn app.main:app --reload

# 8. Start frontend (new terminal)
cd frontend
npm run dev

# 9. Start Celery worker (new terminal)
cd backend
celery -A app.tasks.celery_app worker --loglevel=info
```

### 15.3 Useful Commands

**Database**:
```bash
# PostgreSQL
psql -U citeconnect -d citeconnect

# Create migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1

# Neo4j
cypher-shell -u neo4j -p password

# Redis
redis-cli
```

**Docker**:
```bash
# Build and start all services
docker-compose up --build

# Stop all services
docker-compose down

# View logs
docker-compose logs -f backend

# Execute command in container
docker-compose exec backend bash
```

**Testing**:
```bash
# Run all backend tests
cd backend
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test
pytest tests/test_services/test_search_service.py::test_semantic_search

# Run frontend tests
cd frontend
npm test

# Run e2e tests
npm run test:e2e
```

**Deployment**:
```bash
# Build Docker images
docker build -t backend:latest -f backend/Dockerfile backend/
docker build -t frontend:latest -f frontend/Dockerfile frontend/

# Push to GCR
docker tag backend:latest gcr.io/citeconnect-prod/backend:latest
docker push gcr.io/citeconnect-prod/backend:latest

# Deploy to GKE
kubectl apply -f k8s/
kubectl rollout status deployment/citeconnect-backend
```

### 15.4 Additional Resources

**Documentation**:
- FastAPI: https://fastapi.tiangolo.com/
- React: https://react.dev/
- SPECTER2: https://huggingface.co/allenai/specter2
- Weaviate: https://weaviate.io/developers/weaviate
- Neo4j: https://neo4j.com/docs/
- Cytoscape.js: https://js.cytoscape.org/

**Related Papers**:
- SPECTER: Document-level Representation Learning using Citation-informed Transformers (2020)
- SPECTER 2.0: Better Scientific Paper Embeddings with GPT-style Transformers (2023)

---

**Document Version**: 1.0  
**Last Updated**: November 2025  
**Maintained By**: CiteConnect Team

**For Code Generation**: This document provides complete specifications for implementing CiteConnect. Follow the project structure in Section 4, implement services as described in Section 7, use algorithms from Section 8, and follow the database schemas in Section 5. All API specifications are in Section 6.