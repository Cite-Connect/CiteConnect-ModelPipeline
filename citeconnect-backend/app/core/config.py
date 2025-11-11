# app/core/config.py

"""
Configuration Management Module

This module manages application configuration using Pydantic Settings.
Configuration is loaded from environment variables and .env files.

Features:
- Type-safe configuration with Pydantic validation
- Environment variable loading with .env file support
- Separate settings for different environments (dev, staging, production)
- Automatic validation of required settings
- Default values for optional settings

Usage:
    from app.core.config import get_settings
    
    settings = get_settings()
    print(settings.DATABASE_URL)
    print(settings.SECRET_KEY)
"""

import logging
from functools import lru_cache
from typing import Optional, List
from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Initialize logger for this module
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All settings can be configured via environment variables or .env file.
    Pydantic automatically validates types and required fields.
    
    Attributes:
        Environment and application settings
        Database connection strings
        External API keys
        Model configuration
        Security settings
    """
    
    # ==================== Application Settings ====================
    
    APP_NAME: str = Field(
        default="CiteConnect",
        description="Application name"
    )
    
    ENVIRONMENT: str = Field(
        default="development",
        description="Environment: development, staging, or production"
    )
    
    DEBUG: bool = Field(
        default=True,
        description="Debug mode flag"
    )
    
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL"
    )
    
    API_V1_PREFIX: str = Field(
        default="/api/v1",
        description="API version 1 prefix"
    )
    
    # ==================== Database Settings ====================
    
    DATABASE_URL: str = Field(
        default="postgresql://citeconnect:password@localhost:5432/citeconnect",
        description="PostgreSQL connection URL"
    )
    
    POSTGRES_USER: str = Field(
        default="citeconnect",
        description="PostgreSQL username"
    )
    
    POSTGRES_PASSWORD: str = Field(
        default="password",
        description="PostgreSQL password"
    )
    
    POSTGRES_DB: str = Field(
        default="citeconnect",
        description="PostgreSQL database name"
    )
    
    POSTGRES_HOST: str = Field(
        default="localhost",
        description="PostgreSQL host"
    )
    
    POSTGRES_PORT: int = Field(
        default=5432,
        description="PostgreSQL port"
    )
    
    DB_POOL_SIZE: int = Field(
        default=10,
        description="Database connection pool size"
    )
    
    DB_MAX_OVERFLOW: int = Field(
        default=20,
        description="Maximum database connection overflow"
    )
    
    # ==================== Weaviate Settings ====================
    
    WEAVIATE_URL: str = Field(
        default="http://localhost:8080",
        description="Weaviate vector database URL"
    )
    
    WEAVIATE_API_KEY: Optional[str] = Field(
        default=None,
        description="Weaviate API key (optional for local development)"
    )
    
    WEAVIATE_TIMEOUT: int = Field(
        default=30,
        description="Weaviate request timeout in seconds"
    )
    
    # ==================== Neo4j Settings ====================
    
    NEO4J_URI: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j connection URI"
    )
    
    NEO4J_USER: str = Field(
        default="neo4j",
        description="Neo4j username"
    )
    
    NEO4J_PASSWORD: str = Field(
        default="password",
        description="Neo4j password"
    )
    
    NEO4J_MAX_CONNECTION_LIFETIME: int = Field(
        default=3600,
        description="Neo4j max connection lifetime in seconds"
    )
    
    NEO4J_MAX_CONNECTION_POOL_SIZE: int = Field(
        default=50,
        description="Neo4j connection pool size"
    )
    
    # ==================== Redis Settings ====================
    
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    
    REDIS_HOST: str = Field(
        default="localhost",
        description="Redis host"
    )
    
    REDIS_PORT: int = Field(
        default=6379,
        description="Redis port"
    )
    
    REDIS_DB: int = Field(
        default=0,
        description="Redis database number"
    )
    
    REDIS_PASSWORD: Optional[str] = Field(
        default=None,
        description="Redis password (optional)"
    )
    
    REDIS_MAX_CONNECTIONS: int = Field(
        default=50,
        description="Redis connection pool size"
    )
    
    # ==================== Celery Settings ====================
    
    CELERY_BROKER_URL: str = Field(
        default="redis://localhost:6379/1",
        description="Celery broker URL"
    )
    
    CELERY_RESULT_BACKEND: str = Field(
        default="redis://localhost:6379/2",
        description="Celery result backend URL"
    )
    
    CELERY_TASK_TRACK_STARTED: bool = Field(
        default=True,
        description="Track Celery task start time"
    )
    
    CELERY_TASK_TIME_LIMIT: int = Field(
        default=3600,
        description="Celery task time limit in seconds (1 hour)"
    )
    
    # ==================== JWT and Security Settings ====================
    
    SECRET_KEY: str = Field(
        default="your-secret-key-min-32-characters-long-change-in-production",
        description="Secret key for JWT encoding"
    )
    
    ALGORITHM: str = Field(
        default="HS256",
        description="JWT encoding algorithm"
    )
    
    ACCESS_TOKEN_EXPIRE_HOURS: int = Field(
        default=24,
        description="Access token expiration time in hours"
    )
    
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(
        default=7,
        description="Refresh token expiration time in days"
    )
    
    # ==================== CORS Settings ====================
    
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        description="Allowed CORS origins"
    )
    
    ALLOWED_METHODS: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "PATCH"],
        description="Allowed HTTP methods"
    )
    
    ALLOWED_HEADERS: List[str] = Field(
        default=["*"],
        description="Allowed HTTP headers"
    )
    
    # ==================== GCP Settings ====================
    
    GCP_PROJECT_ID: Optional[str] = Field(
        default=None,
        description="Google Cloud Platform project ID"
    )
    
    GCS_BUCKET_NAME: Optional[str] = Field(
        default=None,
        description="Google Cloud Storage bucket name for PDFs"
    )
    
    GCP_CREDENTIALS_PATH: Optional[str] = Field(
        default=None,
        description="Path to GCP service account credentials JSON"
    )
    
    # ==================== External API Settings ====================
    
    SEMANTIC_SCHOLAR_API_KEY: Optional[str] = Field(
        default=None,
        description="Semantic Scholar API key (optional, increases rate limits)"
    )
    
    OPENAI_API_KEY: Optional[str] = Field(
        default=None,
        description="OpenAI API key for LLM operations"
    )
    
    # ==================== SPECTER Model Settings ====================
    
    SPECTER_MODEL_NAME: str = Field(
        default="allenai/specter",
        description="SPECTER model name for embeddings"
    )
    
    SPECTER_CACHE_DIR: str = Field(
        default="./models/specter",
        description="Directory to cache SPECTER model"
    )
    
    EMBEDDING_DIMENSION: int = Field(
        default=768,
        description="Embedding vector dimension"
    )
    
    EMBEDDING_BATCH_SIZE: int = Field(
        default=32,
        description="Batch size for embedding generation"
    )
    
    # ==================== Rate Limiting Settings ====================
    
    RATE_LIMIT_PER_MINUTE: int = Field(
        default=100,
        description="API rate limit per minute per user"
    )
    
    RATE_LIMIT_SEARCH: int = Field(
        default=100,
        description="Search endpoint rate limit per minute"
    )
    
    RATE_LIMIT_GRAPH: int = Field(
        default=50,
        description="Graph endpoint rate limit per minute (expensive operation)"
    )
    
    # ==================== Cache TTL Settings ====================
    
    CACHE_TTL_USER_SESSION: int = Field(
        default=86400,
        description="User session cache TTL in seconds (24 hours)"
    )
    
    CACHE_TTL_STARTER_KIT: int = Field(
        default=86400,
        description="Starter kit cache TTL in seconds (24 hours)"
    )
    
    CACHE_TTL_CLUSTER: int = Field(
        default=3600,
        description="Cluster cache TTL in seconds (1 hour)"
    )
    
    CACHE_TTL_GRAPH: int = Field(
        default=3600,
        description="Graph cache TTL in seconds (1 hour)"
    )
    
    CACHE_TTL_SEARCH: int = Field(
        default=1800,
        description="Search results cache TTL in seconds (30 minutes)"
    )
    
    CACHE_TTL_PAPER_METADATA: int = Field(
        default=86400,
        description="Paper metadata cache TTL in seconds (24 hours)"
    )
    
    CACHE_TTL_USER_EMBEDDING: int = Field(
        default=21600,
        description="User embedding cache TTL in seconds (6 hours)"
    )
    
    # ==================== Clustering Settings ====================
    
    N_CLUSTERS_HOME: int = Field(
        default=3,
        description="Number of clusters for home page"
    )
    
    CLUSTER_MIN_PAPERS: int = Field(
        default=5,
        description="Minimum papers per cluster"
    )
    
    CLUSTER_MAX_PAPERS: int = Field(
        default=7,
        description="Maximum papers per cluster for display"
    )
    
    # ==================== Recommendation Settings ====================
    
    RECOMMENDATION_SEMANTIC_WEIGHT: float = Field(
        default=0.35,
        description="Weight for semantic similarity in recommendations"
    )
    
    RECOMMENDATION_CITATION_WEIGHT: float = Field(
        default=0.20,
        description="Weight for citation relevance in recommendations"
    )
    
    RECOMMENDATION_KEYWORD_WEIGHT: float = Field(
        default=0.15,
        description="Weight for keyword matching in recommendations"
    )
    
    RECOMMENDATION_POPULARITY_WEIGHT: float = Field(
        default=0.15,
        description="Weight for popularity in recommendations"
    )
    
    RECOMMENDATION_RECENCY_WEIGHT: float = Field(
        default=0.10,
        description="Weight for recency in recommendations"
    )
    
    RECOMMENDATION_DIVERSITY_WEIGHT: float = Field(
        default=0.05,
        description="Weight for diversity in recommendations"
    )
    
    # ==================== Search Settings ====================
    
    SEARCH_DEFAULT_LIMIT: int = Field(
        default=20,
        description="Default number of search results"
    )
    
    SEARCH_MAX_LIMIT: int = Field(
        default=100,
        description="Maximum number of search results"
    )
    
    SEARCH_SIMILARITY_THRESHOLD: float = Field(
        default=0.70,
        description="Minimum similarity score for search results"
    )
    
    # ==================== Graph Settings ====================
    
    GRAPH_DEFAULT_LIMIT: int = Field(
        default=25,
        description="Default number of related papers in graph"
    )
    
    GRAPH_MAX_LIMIT: int = Field(
        default=50,
        description="Maximum number of related papers in graph"
    )
    
    GRAPH_SIMILARITY_THRESHOLD: float = Field(
        default=0.70,
        description="Minimum similarity for graph edges"
    )
    
    # ==================== Validators ====================
    
    @validator("ENVIRONMENT")
    def validate_environment(cls, v: str) -> str:
        """
        Validate environment setting.
        
        Args:
            v: Environment value
        
        Returns:
            Validated environment value
        
        Raises:
            ValueError: If environment is not valid
        """
        allowed = ["development", "staging", "production"]
        if v.lower() not in allowed:
            logger.warning(
                f"Invalid environment '{v}', must be one of {allowed}. "
                f"Defaulting to 'development'"
            )
            return "development"
        return v.lower()
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v: str) -> str:
        """
        Validate log level setting.
        
        Args:
            v: Log level value
        
        Returns:
            Validated log level value
        
        Raises:
            ValueError: If log level is not valid
        """
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            logger.warning(
                f"Invalid log level '{v}', must be one of {allowed}. "
                f"Defaulting to 'INFO'"
            )
            return "INFO"
        return v.upper()
    
    @validator("SECRET_KEY")
    def validate_secret_key(cls, v: str, values: dict) -> str:
        """
        Validate secret key in production.
        
        Args:
            v: Secret key value
            values: Other validated values
        
        Returns:
            Validated secret key
        
        Raises:
            ValueError: If secret key is default in production
        """
        environment = values.get("ENVIRONMENT", "development")
        
        if environment == "production":
            if v == "your-secret-key-min-32-characters-long-change-in-production":
                raise ValueError(
                    "SECRET_KEY must be changed from default value in production!"
                )
            
            if len(v) < 32:
                raise ValueError(
                    "SECRET_KEY must be at least 32 characters long in production!"
                )
        
        return v
    
    @validator("ALLOWED_ORIGINS", pre=True)
    def parse_allowed_origins(cls, v) -> List[str]:
        """
        Parse ALLOWED_ORIGINS from comma-separated string or list.
        
        Args:
            v: ALLOWED_ORIGINS value (string or list)
        
        Returns:
            List of allowed origins
        """
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    # ==================== Pydantic Configuration ====================
    
    model_config = SettingsConfigDict(
        env_file="../.env",  # Look for .env in parent directory (ModelPipeline root)
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"  # Ignore extra fields in .env file
    )
    
    # ==================== Helper Methods ====================
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.ENVIRONMENT == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.ENVIRONMENT == "production"
    
    @property
    def database_url_async(self) -> str:
        """Get async PostgreSQL connection URL for asyncpg."""
        return self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
    
    def get_cache_ttl(self, cache_type: str) -> int:
        """
        Get cache TTL for a specific cache type.
        
        Args:
            cache_type: Type of cache (user_session, starter_kit, etc.)
        
        Returns:
            Cache TTL in seconds
        """
        cache_ttl_map = {
            "user_session": self.CACHE_TTL_USER_SESSION,
            "starter_kit": self.CACHE_TTL_STARTER_KIT,
            "cluster": self.CACHE_TTL_CLUSTER,
            "graph": self.CACHE_TTL_GRAPH,
            "search": self.CACHE_TTL_SEARCH,
            "paper_metadata": self.CACHE_TTL_PAPER_METADATA,
            "user_embedding": self.CACHE_TTL_USER_EMBEDDING,
        }
        return cache_ttl_map.get(cache_type, 3600)  # Default 1 hour


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache to ensure settings are loaded only once and reused
    across the application. This is the recommended way to access settings.
    
    Returns:
        Settings instance with all configuration
    
    Example:
        >>> from app.core.config import get_settings
        >>> settings = get_settings()
        >>> print(settings.DATABASE_URL)
    """
    logger.info("Loading application settings")
    
    try:
        settings = Settings()
        logger.info(
            "Settings loaded successfully",
            extra={
                "environment": settings.ENVIRONMENT,
                "debug": settings.DEBUG,
                "log_level": settings.LOG_LEVEL
            }
        )
        return settings
    except Exception as e:
        logger.error(f"Failed to load settings: {str(e)}", exc_info=True)
        raise


# Initialize module logger
logger = logging.getLogger(__name__)
logger.info("Configuration module loaded successfully")