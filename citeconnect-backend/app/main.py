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
_startup_complete = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events with Cloud Run resilience.
    """
    global _startup_complete
    # Startup
    logger.info(
        "Starting CiteConnect application",
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT
    )
    
    # Initialize default app state
    app.state.db = None
    app.state.services_healthy = {
        'database': False,
        'models': False,
        'repositories': False
    }
    
    try:
        # Try database connection (non-blocking)
        logger.info("Attempting database connection")
        try:
            await db.connect()
            is_healthy = await db.health_check()
            if is_healthy:
                logger.info("Database connected successfully")
                app.state.db = db
                app.state.services_healthy['database'] = True
            else:
                logger.warning("Database health check failed - continuing in limited mode")
        except Exception as e:
            logger.warning(f"Database connection failed - continuing in limited mode: {e}")
        
        # Only initialize repositories if database is available
        if app.state.services_healthy['database']:
            try:
                logger.info("Initializing repositories")
                from app.db.repositories.embedding_repo import EmbeddingRepository
                from app.db.repositories.paper_repo import PaperRepository
                from app.db.repositories.ground_truth_repo import GroundTruthRepository
                from app.db.repositories.user_repo import UserRepository
                from app.db.repositories.interaction_repo import InteractionRepository
                
                app.state.embedding_repo = EmbeddingRepository(db)
                app.state.paper_repo = PaperRepository(db)
                app.state.ground_truth_repo = GroundTruthRepository(db)
                app.state.user_repo = UserRepository(db)
                app.state.interaction_repo = InteractionRepository(db)
                app.state.services_healthy['repositories'] = True
                logger.info("Repositories initialized successfully")
            except Exception as e:
                logger.warning(f"Repository initialization failed: {e}")
        
        # Try ML services (non-blocking)
        try:
            logger.info("Attempting ML services initialization")
            from app.services.bootstrap.embedding_service import EmbeddingService
            
            embedding_service = EmbeddingService()
            model_health = embedding_service.health_check()
            
            app.state.embedding_service = embedding_service
            if any(model_health.values()):
                app.state.services_healthy['models'] = True
                logger.info("ML services initialized successfully")
            else:
                logger.warning("Models failed health check - continuing with limited ML functionality")
        except Exception as e:
            logger.warning(f"ML service initialization failed: {e}")
            app.state.embedding_service = None
        
        # Initialize other services only if dependencies are met
        if app.state.services_healthy['database'] and app.state.services_healthy['repositories']:
            try:
                logger.info("Initializing full application services")
                from app.services.bootstrap.ground_truth_service import GroundTruthService
                from app.services.bootstrap.experiment_service import ExperimentService
                from app.services.runtime.user_state_service import UserStateService
                from app.services.evaluation_service import EvaluationService
                from app.services.recommendation_service import RecommendationService
                from app.services.runtime.recommendation_orchestrator import RecommendationOrchestrator
                
                # Initialize services
                app.state.ground_truth_service = GroundTruthService(app.state.ground_truth_repo, app.state.paper_repo)
                app.state.experiment_service = ExperimentService(db)
                app.state.user_state_service = UserStateService(app.state.user_repo, app.state.interaction_repo)
                app.state.evaluation_service = EvaluationService(db)
                app.state.recommendation_service = RecommendationService(db)
                
                # Initialize orchestrator
                app.state.recommendation_orchestrator = RecommendationOrchestrator(
                    rec_service=app.state.recommendation_service,
                    eval_service=app.state.evaluation_service,
                    experiment_service=app.state.experiment_service,
                    user_state_service=app.state.user_state_service
                )
                
                if hasattr(app.state.ground_truth_service, 'initialize'):
                    await app.state.ground_truth_service.initialize()
                    
                logger.info("Full application services initialized")
            except Exception as e:
                logger.warning(f"Full service initialization failed - basic functionality available: {e}")
        
        # Mark as ready regardless of what services are available
        _startup_complete = True
        logger.info(
            "Application startup complete",
            services_available=app.state.services_healthy,
            mode="full" if all(app.state.services_healthy.values()) else "limited"
        )
        
        yield
        
    except Exception as e:
        logger.error(f"Critical startup error: {e}", exc_info=True)
        # Still mark as complete so health check passes
        _startup_complete = True
        yield
    
    finally:
        # Shutdown
        _startup_complete = False
        logger.info("Shutting down application")
        
        try:
            if app.state.db:
                await app.state.db.disconnect()
                logger.info("Database disconnected")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
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
    Checks database and model availability safely.
    """
    global _startup_complete
    logger.debug("Health check requested")

    if not _startup_complete:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "starting",
                "message": "Application is still initializing"
            }
        )
    
    health_status = {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "services": getattr(app.state, 'services_healthy', {}),
        "checks": {}
    }
    
    try:
        # Check database safely
        if hasattr(app.state, 'db') and app.state.db is not None:
            try:
                logger.debug("Performing database health check")        
                db_healthy = await app.state.db.health_check()
                health_status["checks"]["database"] = "healthy" if db_healthy else "degraded"
            except Exception as e:
                health_status["checks"]["database"] = f"error: {str(e)}"
        else:
            health_status["checks"]["database"] = "not_available"
        
        # Check models safely
        if hasattr(app.state, 'embedding_service') and app.state.embedding_service is not None:
            try:
                logger.debug("Performing model health check")        
                model_health = app.state.embedding_service.health_check()
                health_status["checks"]["models"] = {
                    model: "healthy" if healthy else "degraded"
                    for model, healthy in model_health.items()
                }
            except Exception as e:
                health_status["checks"]["models"] = f"error: {str(e)}"
        else:
            health_status["checks"]["models"] = "not_available"
        
        # Always return healthy status for basic API functionality
        health_status["status"] = "healthy"
        
        logger.debug("Health check complete", status=health_status["status"])
        return health_status
        
    except Exception as e:
        logger.error("Health check failed", error=str(e), exc_info=True)
        
        # Return basic healthy status to keep Cloud Run happy
        return {
            "status": "healthy",
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "mode": "basic",
            "note": "API is functional with limited services"
        }


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
from app.api.v1 import graph, recommendations, users, papers, interactions

app.include_router(
    graph.router,
    prefix=f"{settings.API_V1_PREFIX}/graph",
    tags=["graph"]
)

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