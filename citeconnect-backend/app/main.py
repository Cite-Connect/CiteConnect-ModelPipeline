# app/main.py

"""
CiteConnect FastAPI Application

Main application entry point. Configures FastAPI app with:
- CORS middleware
- Logging middleware
- Exception handlers
- API routers
- Database connection lifecycle
- Health check endpoints

Usage:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.core.exceptions import CiteConnectException

# Import database clients
from app.db.postgres import get_db_pool, close_db_pool
from app.db.redis_client import get_redis_client, close_redis_client
from app.db.weaviate_client import get_weaviate_client, close_weaviate_client, create_schema
from app.db.neo4j_client import get_neo4j_driver, close_neo4j_driver

# Import API routers
from app.api.v1 import auth, users

# Get settings
settings = get_settings()

# Setup logging
setup_logging(
    log_level=settings.LOG_LEVEL,
    environment=settings.ENVIRONMENT,
    log_file="logs/citeconnect.log" if not settings.DEBUG else None
)

# Initialize logger for this module
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events:
    - Startup: Initialize database connections, create schemas
    - Shutdown: Close database connections gracefully
    
    Args:
        app: FastAPI application instance
    """
    # Startup
    logger.info("=" * 60)
    logger.info("CiteConnect Backend Starting Up")
    logger.info("=" * 60)
    
    logger.info(
        "Application configuration loaded",
        extra={
            "environment": settings.ENVIRONMENT,
            "debug": settings.DEBUG,
            "log_level": settings.LOG_LEVEL
        }
    )
    
    # Initialize database connections
    try:
        logger.info("Initializing database connections...")
        
        # PostgreSQL
        logger.info("Connecting to PostgreSQL...")
        try:
            pool = await get_db_pool()
            logger.info("PostgreSQL connection pool created successfully")
        except Exception as e:
            logger.warning(f"PostgreSQL connection failed: {str(e)}")
            logger.warning("Application will start but database operations will fail")
        
        # Redis
        logger.info("Connecting to Redis...")
        try:
            redis = await get_redis_client()
            logger.info("Redis client created successfully")
        except Exception as e:
            logger.warning(f"Redis connection failed: {str(e)}")
            logger.warning("Caching will be disabled")
        
        # Weaviate
        logger.info("Connecting to Weaviate...")
        try:
            weaviate = get_weaviate_client()
            logger.info("Weaviate client created successfully")
            
            # Create schema if doesn't exist
            logger.info("Ensuring Weaviate schema exists...")
            create_schema()
            logger.info("Weaviate schema ready")
        except Exception as e:
            logger.warning(f"Weaviate connection failed: {str(e)}")
            logger.warning("Vector search will be unavailable")
        
        # Neo4j
        logger.info("Connecting to Neo4j...")
        try:
            driver = await get_neo4j_driver()
            logger.info("Neo4j driver created successfully")
        except Exception as e:
            logger.warning(f"Neo4j connection failed: {str(e)}")
            logger.warning("Citation graph features will be unavailable")
        
        logger.info("=" * 60)
        logger.info("CiteConnect Backend Started Successfully")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("=" * 60)
    logger.info("CiteConnect Backend Shutting Down")
    logger.info("=" * 60)
    
    # Close database connections
    try:
        logger.info("Closing database connections...")
        
        await close_db_pool()
        logger.info("PostgreSQL connection pool closed")
        
        await close_redis_client()
        logger.info("Redis client closed")
        
        close_weaviate_client()
        logger.info("Weaviate client closed")
        
        await close_neo4j_driver()
        logger.info("Neo4j driver closed")
        
        logger.info("All database connections closed successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}", exc_info=True)
    
    logger.info("=" * 60)
    logger.info("CiteConnect Backend Shut Down")
    logger.info("=" * 60)


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="Research paper recommendation system with semantic search and citation networks",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=settings.ALLOWED_METHODS,
    allow_headers=settings.ALLOWED_HEADERS,
)


# Exception Handlers
@app.exception_handler(CiteConnectException)
async def citeconnect_exception_handler(request: Request, exc: CiteConnectException):
    """
    Handle custom CiteConnect exceptions.
    
    Args:
        request: FastAPI request object
        exc: CiteConnectException instance
    
    Returns:
        JSONResponse with error details
    """
    logger.error(
        f"CiteConnect error: {exc.message}",
        extra={
            "status_code": exc.status_code,
            "path": request.url.path,
            "method": request.method,
            "details": exc.details
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "type": exc.__class__.__name__,
                "status_code": exc.status_code,
                "details": exc.details
            }
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle Pydantic validation errors.
    
    Args:
        request: FastAPI request object
        exc: RequestValidationError instance
    
    Returns:
        JSONResponse with validation error details
    """
    logger.warning(
        f"Validation error: {str(exc)}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "errors": exc.errors()
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "message": "Validation error",
                "type": "ValidationError",
                "status_code": 422,
                "details": exc.errors()
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle unexpected exceptions.
    
    Args:
        request: FastAPI request object
        exc: Exception instance
    
    Returns:
        JSONResponse with generic error message
    """
    logger.exception(
        f"Unexpected error: {str(e)}",
        extra={
            "path": request.url.path,
            "method": request.method
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "message": "An unexpected error occurred",
                "type": "InternalServerError",
                "status_code": 500
            }
        }
    )


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint.
    
    Returns basic API information.
    """
    logger.info("Root endpoint accessed")
    
    return {
        "message": "Welcome to CiteConnect API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


# Health check endpoint
@app.get("/api/v1/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Checks status of all database connections.
    
    Returns:
        Health status of all services
    """
    logger.debug("Health check requested")
    
    from app.db.postgres import check_db_health
    from app.db.redis_client import check_redis_health
    from app.db.weaviate_client import check_weaviate_health
    from app.db.neo4j_client import check_neo4j_health
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "postgres": "unknown",
            "redis": "unknown",
            "weaviate": "unknown",
            "neo4j": "unknown"
        }
    }
    
    # Check PostgreSQL
    try:
        postgres_healthy = await check_db_health()
        health_status["services"]["postgres"] = "healthy" if postgres_healthy else "unhealthy"
        if not postgres_healthy:
            health_status["status"] = "degraded"
    except Exception as e:
        logger.error(f"PostgreSQL health check failed: {str(e)}")
        health_status["services"]["postgres"] = "unhealthy"
        health_status["status"] = "degraded"
    
    # Check Redis
    try:
        redis_healthy = await check_redis_health()
        health_status["services"]["redis"] = "healthy" if redis_healthy else "unhealthy"
        if not redis_healthy:
            health_status["status"] = "degraded"
    except Exception as e:
        logger.error(f"Redis health check failed: {str(e)}")
        health_status["services"]["redis"] = "unhealthy"
        health_status["status"] = "degraded"
    
    # Check Weaviate
    try:
        weaviate_healthy = check_weaviate_health()
        health_status["services"]["weaviate"] = "healthy" if weaviate_healthy else "unhealthy"
        if not weaviate_healthy:
            health_status["status"] = "degraded"
    except Exception as e:
        logger.error(f"Weaviate health check failed: {str(e)}")
        health_status["services"]["weaviate"] = "unhealthy"
        health_status["status"] = "degraded"
    
    # Check Neo4j
    try:
        neo4j_healthy = await check_neo4j_health()
        health_status["services"]["neo4j"] = "healthy" if neo4j_healthy else "unhealthy"
        if not neo4j_healthy:
            health_status["status"] = "degraded"
    except Exception as e:
        logger.error(f"Neo4j health check failed: {str(e)}")
        health_status["services"]["neo4j"] = "unhealthy"
        health_status["status"] = "degraded"
    
    # Determine status code
    status_code = status.HTTP_200_OK if health_status["status"] == "healthy" else status.HTTP_503_SERVICE_UNAVAILABLE
    
    logger.info(
        f"Health check completed: {health_status['status']}",
        extra=health_status["services"]
    )
    
    return JSONResponse(
        status_code=status_code,
        content=health_status
    )


# Register API routers
app.include_router(
    auth.router,
    prefix=f"{settings.API_V1_PREFIX}/auth",
    tags=["Authentication"]
)

app.include_router(
    users.router,
    prefix=f"{settings.API_V1_PREFIX}/users",
    tags=["Users"]
)


logger.info("FastAPI application initialized successfully")
logger.info(f"Registered routes: /auth/register, /auth/login, /auth/refresh, /users/me")
