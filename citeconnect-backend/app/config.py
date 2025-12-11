"""
Configuration management for CiteConnect backend.
Loads settings from environment variables with validation.
"""
import structlog
import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from functools import lru_cache
from pydantic import Field

logger = structlog.get_logger(__name__)

# Resolve .env file path relative to this config file
# This ensures it works regardless of where the server is run from
_config_dir = Path(__file__).parent.parent  # Go up from app/config.py to citeconnect-backend/
_env_file = _config_dir / ".env"

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
# --- IMPORTANT CHANGE HERE ---
    model_config = SettingsConfigDict(
        env_file=str(_env_file) if _env_file.exists() else ".env",  # Use absolute path if exists, else fallback
        env_file_encoding="utf-8",
        extra="allow"               # Kept your original 'extra: "allow"'
    )    
    # Application
    APP_NAME: str = "CiteConnect"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # API
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:8000", "https://pulp-three.vercel.app"]
    
    # Database - Supabase PostgreSQL
    # Webhook and Pub/Sub settings
    WEBHOOK_SECRET: Optional[str] = None
    SUPABASE_URL: str
    SUPABASE_KEY: Optional[str] = None  # Optional - only needed for Supabase auth/storage
    DATABASE_URL: str  # postgres:// connection string
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30
    
    # Redis Cache
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    CACHE_TTL_MULTIPLIER: float = 1.0
    
    # ML Models
    EMBEDDING_MODEL_MINILM: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_MODEL_SPECTER: str = "allenai/specter"
    MODEL_CACHE_DIR: str = "./models"
    DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_MAX_LENGTH: int = 512
    
    # MLflow
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "citeconnect-recommendations"
    
    # GCP (if using)
    GCP_PROJECT_ID: Optional[str] = None
    GCP_BUCKET_NAME: Optional[str] = None
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None
    
    # LLM Services (OpenAI for query refinement)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"  # Cheapest OpenAI model. Options: gpt-4o-mini, gpt-3.5-turbo
    
    # Performance Thresholds
    COLD_START_PROFILE_ALIGNMENT_THRESHOLD: float = 0.6
    COLD_START_GROUND_TRUTH_THRESHOLD: float = 0.5
    MATURE_PRECISION_AT_10_THRESHOLD: float = 0.3
    MATURE_CTR_THRESHOLD: float = 0.25
    BIAS_VARIANCE_THRESHOLD: float = 0.2
    
    # Cache TTLs (seconds)
    CACHE_TTL_USER_EMBEDDING_COLD_START: int = 86400  # 24 hours
    CACHE_TTL_USER_EMBEDDING_EARLY: int = 21600  # 6 hours
    CACHE_TTL_USER_EMBEDDING_MATURE: int = 3600  # 1 hour
    CACHE_TTL_USER_EMBEDDING_EXPERT: int = 1800  # 30 minutes
    CACHE_TTL_RECOMMENDATIONS_COLD_START: int = 3600
    CACHE_TTL_RECOMMENDATIONS_EARLY: int = 1800
    CACHE_TTL_RECOMMENDATIONS_MATURE: int = 900
    CACHE_TTL_RECOMMENDATIONS_EXPERT: int = 600
    CACHE_TTL_GROUND_TRUTH: int = 604800  # 1 week
    CACHE_TTL_CANONICAL_PAPERS: int = 86400
    
    # Rate Limiting
    RATE_LIMIT_RECOMMENDATIONS_PER_HOUR: int = 100
    RATE_LIMIT_RECOMMENDATIONS_PER_MINUTE: int = 20
    RATE_LIMIT_PROFILE_UPDATE_PER_HOUR: int = 10
    
    # Recommendation Settings
    DEFAULT_RECOMMENDATION_COUNT: int = 10
    MAX_RECOMMENDATION_COUNT: int = 50
    MIN_GROUND_TRUTH_CITATIONS: int = 10
    MAX_GROUND_TRUTH_CITATIONS: int = 100
    MIN_REFERENCE_COVERAGE: float = 0.3
    
    # User State Transitions
    STATE_COLD_START_TO_EARLY_MIN_SAVES: int = 2
    STATE_COLD_START_TO_EARLY_MIN_INTERACTIONS: int = 5
    STATE_EARLY_TO_MATURE_MIN_SAVES: int = 5
    STATE_EARLY_TO_MATURE_MIN_INTERACTIONS: int = 15
    STATE_EARLY_TO_MATURE_MIN_DOMAINS: int = 3
    STATE_EARLY_TO_MATURE_MIN_DAYS: int = 7
    STATE_MATURE_TO_EXPERT_MIN_INTERACTIONS: int = 50
    STATE_MATURE_TO_EXPERT_MIN_DAYS: int = 30
    
    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 120
    
    # Background Workers
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # Interaction thresholds for stage transitions
    EARLY_STAGE_THRESHOLD:int = 10      # cold_start → early
    MATURE_STAGE_THRESHOLD:int = 50     # early → mature
    EXPERT_STAGE_THRESHOLD:int = 200    # mature → expert
    
    # Update frequency
    UPDATE_EVERY_N_INTERACTIONS:int = 10

    # Allowed values from Supabase schema
    ALLOWED_DOMAINS: list[str] = ['healthcare', 'fintech', 'quantum_computing']
    ALLOWED_RESEARCH_STAGES: list[str] = [
        'undergraduate', 'masters', 'phd', 'postdoc', 
        'professor', 'industry', 'independent'
    ]
    ALLOWED_READING_LEVELS: list[str] = ['introductory', 'intermediate', 'advanced', 'expert']
    ALLOWED_TIME_AVAILABILITY: list[str] = ['casual_reader', 'part_time_researcher', 'full_time_researcher']

    # Graph Generation Settings
    GRAPH_SEMANTIC_MIN_SIMILARITY: float = Field(
        default=0.6,
        env="GRAPH_SEMANTIC_MIN_SIMILARITY",
        description="Minimum cosine similarity threshold for semantic matching (0-1)"
    )
    
    GRAPH_SEMANTIC_LIMIT: int = Field(
        default=20,
        env="GRAPH_SEMANTIC_LIMIT",
        description="Maximum number of semantic similar papers to retrieve"
    )
    
    GRAPH_HYBRID_TRIGGER_THRESHOLD: int = Field(
        default=5,
        env="GRAPH_HYBRID_TRIGGER_THRESHOLD",
        description="Minimum citation nodes before triggering semantic fallback"
    )
    
    GRAPH_DEFAULT_MODEL: str = Field(
        default="minilm",
        env="GRAPH_DEFAULT_MODEL",
        description="Default embedding model for semantic similarity (minilm or specter)"
    )
    
    GRAPH_DOMAIN_YEAR_RANGE: int = Field(
        default=3,
        env="GRAPH_DOMAIN_YEAR_RANGE",
        description="Year range (+/-) for domain filtering fallback"
    )

    # Embedding Model Configuration
    EMBEDDING_MODELS: dict = {
        "minilm": {
            "table": "paper_embeddings_minilm",
            "dimension": 384,
            "name": "all-MiniLM-L6-v2"
        },
        "specter": {
            "table": "paper_embeddings_specter",
            "dimension": 768,
            "name": "allenai/specter2"
        }
    }

@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Using lru_cache ensures settings are loaded once and reused.
    
    Returns:
        Settings: Application settings instance
    """
    logger.info("Loading application settings", environment=Settings().ENVIRONMENT)
    return Settings()


# Export for easy imports
settings = get_settings()