# CiteConnect Backend

Research paper recommendation system using semantic search, citation networks, and personalized recommendations.

## Project Structure
```
citeconnect-backend/
├── app/                    # Main application code
│   ├── api/               # API endpoints
│   ├── core/              # Core functionality (security, config)
│   ├── db/                # Database connections
│   ├── models/            # Pydantic data models
│   ├── schemas/           # API request/response schemas
│   ├── services/          # Business logic
│   ├── tasks/             # Celery async tasks
│   ├── utils/             # Utility functions
│   └── middleware/        # Custom middleware
├── alembic/               # Database migrations
├── tests/                 # Test suite
├── scripts/               # Utility scripts
└── logs/                  # Application logs
```

## Setup Instructions

### 1. Create Virtual Environment
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements-dev.txt
```

### 3. Configure Environment
```bash
cp .env.example .env
# Edit .env with your configuration
```

### 4. Start Required Services

Using Docker Compose (recommended):
```bash
docker-compose up -d postgres redis neo4j weaviate
```

### 5. Run Database Migrations
```bash
alembic upgrade head
```

### 6. Start Application
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Development

### Run Tests
```bash
pytest
```

### Run with Coverage
```bash
pytest --cov=app --cov-report=html
```

### Code Formatting
```bash
black app/
isort app/
```

### Type Checking
```bash
mypy app/
```

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Technologies

- **FastAPI**: Web framework
- **PostgreSQL**: Primary database
- **Neo4j**: Citation graph database
- **Weaviate**: Vector database for embeddings
- **Redis**: Caching and session management
- **Celery**: Async task queue
- **SPECTER2**: Scientific paper embeddings

## License

MIT

