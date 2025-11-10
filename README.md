# ModelPipeline

MLOps project repository containing CiteConnect and other ML projects.

## Structure
```
ModelPipeline/
├── requirements.txt       # Shared dependencies
├── venv/                  # Shared virtual environment
├── .env                   # Environment configuration
└── citeconnect-backend/   # CiteConnect project
    ├── app/              # Application code
    ├── tests/            # Test suite
    └── scripts/          # Utility scripts
```

## Setup

### 1. Create Virtual Environment (from root)
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

### 4. Run Application
```bash
# From ModelPipeline root
cd citeconnect-backend
uvicorn app.main:app --reload
```

### 5. Run Tests
```bash
# From ModelPipeline root
pytest
```

## Projects

- **citeconnect-backend**: Research paper recommendation system

