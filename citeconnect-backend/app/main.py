"""
CiteConnect Backend Application Entry Point.
Initializes all services, loads models, and configures FastAPI.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import time

from app.config import settings
from app.utils.logger import setup_logging, get_logger
from app.db.connection import db
from app.db.repositories.embedding_repo import EmbeddingRepository
from app.services.bootstrap.embedding_service import EmbeddingService

# Setup logging first
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info(
        "Starting CiteConnect application",
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT
    )
    
    try:
        # Initialize database connection
        logger.info("Initializing database connection")
        await db.connect()
        logger.info("Database connected successfully")
        
        # Perform health check
        is_healthy = await db.health_check()
        if not is_healthy:
            logger.error("Database health check failed")
            raise RuntimeError("Database not accessible")
        
        # Initialize repositories
        logger.info("Initializing repositories")
        from app.db.repositories.embedding_repo import EmbeddingRepository
        from app.db.repositories.paper_repo import PaperRepository
        from app.db.repositories.ground_truth_repo import GroundTruthRepository
        from app.db.repositories.user_repo import UserRepository
        from app.db.repositories.interaction_repo import InteractionRepository
        
        embedding_repo = EmbeddingRepository(db)
        paper_repo = PaperRepository(db)
        ground_truth_repo = GroundTruthRepository(db)
        user_repo = UserRepository(db)
        interaction_repo = InteractionRepository(db)
        
        # Initialize bootstrap services
        logger.info("Initializing bootstrap services")
        from app.services.bootstrap.embedding_service import EmbeddingService
        from app.services.bootstrap.ground_truth_service import GroundTruthService
        from app.services.bootstrap.experiment_service import ExperimentService
        
        # 1. Embedding Service (Singleton)
        embedding_service = EmbeddingService() 
        
        # 2. Ground Truth Service
        ground_truth_service = GroundTruthService(ground_truth_repo, paper_repo)
        if hasattr(ground_truth_service, 'initialize'):
            await ground_truth_service.initialize()
        
        # 3. Experiment Service
        experiment_service = ExperimentService(db)
        
        # Initialize runtime services
        logger.info("Initializing runtime services")
        from app.services.runtime.user_state_service import UserStateService
        from app.services.evaluation_service import EvaluationService
        from app.services.recommendation_service import RecommendationService
        from app.services.runtime.recommendation_orchestrator import RecommendationOrchestrator
        
        user_state_service = UserStateService(user_repo, interaction_repo)
        
        # Initialize specialized workers
        rec_service = RecommendationService(db)
        evaluation_service = EvaluationService(db)
        
        # Initialize Orchestrator with dependencies
        recommendation_orchestrator = RecommendationOrchestrator(
            rec_service=rec_service,
            eval_service=evaluation_service,
            experiment_service=experiment_service,
            user_state_service=user_state_service
        )
        
        # Store in app state for access in endpoints
        app.state.db = db
        app.state.embedding_repo = embedding_repo
        app.state.paper_repo = paper_repo
        app.state.ground_truth_repo = ground_truth_repo
        app.state.user_repo = user_repo
        app.state.interaction_repo = interaction_repo
        app.state.embedding_service = embedding_service
        app.state.ground_truth_service = ground_truth_service
        app.state.user_state_service = user_state_service
        app.state.evaluation_service = evaluation_service
        app.state.experiment_service = experiment_service
        app.state.recommendation_orchestrator = recommendation_orchestrator
        
        # Check model health (CRITICAL FIX: Removed 'await')
        model_health = embedding_service.health_check()
        logger.info(
            "Model health check complete",
            results=model_health
        )
        
        if not all(model_health.values()):
            logger.warning(
                "Some models failed health check",
                failed=[ k for k, v in model_health.items() if not v]
            )
        
        logger.info(
            "Application startup complete",
            models_loaded=list(embedding_service.models.keys())
        )
        
        yield
        
    except Exception as e:
        logger.error(
            "Application startup failed",
            error=str(e),
            exc_info=True
        )
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down application")
        
        try:
            await db.disconnect()
            logger.info("Database disconnected")
        except Exception as e:
            logger.error(
                "Error during shutdown",
                error=str(e),
                exc_info=True
            )
        
        logger.info("Application shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered academic paper recommendation system",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log all incoming requests with timing.
    """
    request_id = str(time.time())
    start_time = time.time()
    
    logger.info(
        "Request received",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        client=request.client.host if request.client else "unknown"
    )
    
    try:
        response = await call_next(request)
        
        duration_ms = (time.time() - start_time) * 1000
        
        logger.info(
            "Request completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2)
        )
        
        return response
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        
        logger.error(
            "Request failed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            duration_ms=round(duration_ms, 2),
            error=str(e),
            exc_info=True
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An internal error occurred",
                    "request_id": request_id
                }
            }
        )


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
):
    """Handle validation errors with detailed messages."""
    logger.warning(
        "Validation error",
        path=request.url.path,
        errors=exc.errors()
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid request parameters",
                "details": exc.errors()
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors gracefully."""
    logger.error(
        "Unhandled exception",
        path=request.url.path,
        error=str(exc),
        exc_info=True
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred"
            }
        }
    )


# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring.
    Checks database and model availability.
    """
    logger.debug("Health check requested")
    
    health_status = {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "checks": {}
    }
    
    try:
        # Check database
        db_healthy = await db.health_check()
        health_status["checks"]["database"] = "healthy" if db_healthy else "unhealthy"
        
        # Check models if service is available
        if hasattr(app.state, 'embedding_service'):
            # Fix here too: Removed await
            model_health = app.state.embedding_service.health_check() 
            health_status["checks"]["models"] = {
                model: "healthy" if healthy else "unhealthy"
                for model, healthy in model_health.items()
            }
        else:
            health_status["checks"]["models"] = "not_initialized"
        
        # Determine overall status
        all_healthy = (
            db_healthy and
            (not hasattr(app.state, 'embedding_service') or 
             all(model_health.values()))
        )
        
        health_status["status"] = "healthy" if all_healthy else "degraded"
        
        logger.debug(
            "Health check complete",
            status=health_status["status"]
        )
        
        return health_status
        
    except Exception as e:
        logger.error(
            "Health check failed",
            error=str(e),
            exc_info=True
        )
        
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "docs": "/docs" if settings.DEBUG else "disabled",
        "health": "/health"
    }


# Import and include routers
from app.api.v1 import recommendations, users, papers, interactions

app.include_router(
    recommendations.router,
    prefix=f"{settings.API_V1_PREFIX}/recommendations",
    tags=["recommendations"]
)
app.include_router(
    users.router,
    prefix=f"{settings.API_V1_PREFIX}/users",
    tags=["users"]
)
app.include_router(
    papers.router,
    prefix=f"{settings.API_V1_PREFIX}/papers",
    tags=["papers"]
)
app.include_router(
    interactions.router,
    prefix=f"{settings.API_V1_PREFIX}/interactions",
    tags=["interactions"]
)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )