"""
CiteConnect Backend Application Entry Point.
Initializes all services, loads models, and configures FastAPI.
"""

from contextlib import asynccontextmanager
import os
import json
import time
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, Request, status, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST

from app.config import settings
from app.utils.logger import setup_logging, get_logger
from app.db.connection import db
from app.services.bootstrap.embedding_service import EmbeddingService

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
setup_logging()
logger = get_logger(__name__)
_startup_complete = False

# -----------------------------------------------------------------------------
# Prometheus Metrics
# -----------------------------------------------------------------------------

# 1) Model performance / decay (offline evaluation)
MODEL_AVG_COMBINED_SCORE = Gauge(
    "model_avg_combined_score",
    "Average combined score from offline/cold-start evaluation (minilm)",
)
MODEL_AVG_PROFILE_ALIGNMENT = Gauge(
    "model_avg_profile_alignment",
    "Average profile alignment from offline evaluation (minilm)",
)
MODEL_AVG_GROUND_TRUTH_QUALITY = Gauge(
    "model_avg_ground_truth_quality",
    "Average ground-truth quality from offline evaluation (minilm)",
)
MODEL_PASS_RATE = Gauge(
    "model_pass_rate",
    "Pass rate from offline evaluation (minilm)",
)

# 2) Data drift (overall)
DATA_DRIFT_OVERALL_SCORE = Gauge(
    "data_drift_overall_score",
    "Overall data drift score (0 = none, 1 = high)",
)

# 3) Bias / fairness (overall disparities for user-profile bias)
BIAS_DISPARITY_RATIO = Gauge(
    "bias_disparity_ratio",
    "Overall disparity ratio between worst and best group (user-profile bias)",
)
BIAS_DISPARITY_DIFFERENCE = Gauge(
    "bias_disparity_difference",
    "Overall disparity difference between worst and best group (user-profile bias)",
)

# 4) Hyperparameter tuning (cold-start weights)
COLD_START_WEIGHT = Gauge(
    "cold_start_weight",
    "Cold-start scoring weight per component from best hyperparameters",
    ["component"],
)
COLD_START_BEST_AVG_COMBINED_SCORE = Gauge(
    "cold_start_best_avg_combined_score",
    "Best avg_combined_score found in cold-start hyperparameter tuning",
)

# 5) Sensitivity analysis
SENSITIVITY_MEAN_FINAL_SCORE = Gauge(
    "sensitivity_mean_final_score",
    "Mean final score per scenario in sensitivity analysis",
    ["scenario"],
)
SENSITIVITY_STD_FINAL_SCORE = Gauge(
    "sensitivity_std_final_score",
    "Std dev of final score per scenario in sensitivity analysis",
    ["scenario"],
)

# 6) Retrain trigger flag
RETRAIN_NEEDED = Gauge(
    "retrain_needed",
    "1 if retraining threshold is crossed based on performance and drift",
)

# -----------------------------------------------------------------------------
# Helper functions for metrics
# -----------------------------------------------------------------------------
def _safe_load_json(path: str) -> Any:
    """
    Helper to safely load a JSON file; returns None if missing/invalid.
    """
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load JSON for metrics", path=path, error=str(e))
        return None


def update_monitoring_metrics() -> None:
    """
    Read project JSON outputs (hyperparams, sensitivity, bias, drift, offline eval)
    and update Prometheus Gauges.

    This is called on each /metrics scrape so Grafana always sees the latest state.
    """
    # backend root: citeconnect-backend/
    backend_root = Path(__file__).resolve().parent.parent

    # Track values needed for retrain flag
    avg_combined: float | None = None
    drift_score: float | None = None

    # -------------------------------------------------------------------------
    # 1) Hyperparameter tuning (cold-start weights)
    # -------------------------------------------------------------------------
    hp_path = backend_root / "bias_config" / "best_hyperparameters_cold_start.json"
    hp_data = _safe_load_json(str(hp_path))

    if isinstance(hp_data, dict) and "best_config" in hp_data:
        best_cfg: Dict[str, Any] = hp_data["best_config"]
        weights: Dict[str, float] = best_cfg.get("weights", {}) or {}
        metrics: Dict[str, Any] = best_cfg.get("metrics", {}) or {}

        # Set a gauge per component weight
        for component, value in weights.items():
            try:
                COLD_START_WEIGHT.labels(component=component).set(float(value))
            except Exception as e:
                logger.warning(
                    "Failed to set cold_start_weight",
                    component=component,
                    value=value,
                    error=str(e),
                )

        # Best combined score
        if "avg_combined_score" in metrics:
            try:
                COLD_START_BEST_AVG_COMBINED_SCORE.set(
                    float(metrics["avg_combined_score"])
                )
            except Exception as e:
                logger.warning(
                    "Failed to set COLD_START_BEST_AVG_COMBINED_SCORE",
                    value=metrics.get("avg_combined_score"),
                    error=str(e),
                )

    # -------------------------------------------------------------------------
    # 2) Sensitivity analysis
    # -------------------------------------------------------------------------
    sens_path = backend_root / "bias_config" / "sensitivity_cold_start_report.json"
    sens_data = _safe_load_json(str(sens_path))

    if isinstance(sens_data, dict) and "scenarios" in sens_data:
        for scen in sens_data["scenarios"]:
            name = scen.get("name", "unknown")
            agg = scen.get("aggregate_metrics", {}) or {}
            mean_score = agg.get("mean_final_score")
            std_score = agg.get("std_final_score")

            if mean_score is not None:
                try:
                    SENSITIVITY_MEAN_FINAL_SCORE.labels(scenario=name).set(
                        float(mean_score)
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to set SENSITIVITY_MEAN_FINAL_SCORE",
                        scenario=name,
                        value=mean_score,
                        error=str(e),
                    )

            if std_score is not None:
                try:
                    SENSITIVITY_STD_FINAL_SCORE.labels(scenario=name).set(
                        float(std_score)
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to set SENSITIVITY_STD_FINAL_SCORE",
                        scenario=name,
                        value=std_score,
                        error=str(e),
                    )

    # -------------------------------------------------------------------------
    # 3) Bias / fairness metrics (overall, from bias_reports.json)
    # -------------------------------------------------------------------------
    bias_path = backend_root / "bias_reports.json"
    bias_data = _safe_load_json(str(bias_path))

    if isinstance(bias_data, dict):
        cf = bias_data.get("citation_fairness", {}) or {}
        disparity_ratio = cf.get("disparity_ratio")
        disparity_difference = cf.get("disparity_difference")

        if disparity_ratio is not None:
            try:
                BIAS_DISPARITY_RATIO.set(float(disparity_ratio))
            except Exception as e:
                logger.warning(
                    "Failed to set BIAS_DISPARITY_RATIO",
                    value=disparity_ratio,
                    error=str(e),
                )

        if disparity_difference is not None:
            try:
                BIAS_DISPARITY_DIFFERENCE.set(float(disparity_difference))
            except Exception as e:
                logger.warning(
                    "Failed to set BIAS_DISPARITY_DIFFERENCE",
                    value=disparity_difference,
                    error=str(e),
                )

    # -------------------------------------------------------------------------
    # 4) Offline evaluation from evaluation_results_minilm_*.json
    #    (model decay tracking)
    # -------------------------------------------------------------------------
    try:
        eval_files = sorted(backend_root.glob("evaluation_results_minilm_*.json"))
    except Exception as e:
        logger.warning(
            "Failed to glob evaluation_results_minilm_*.json", error=str(e)
        )
        eval_files = []

    if eval_files:
        # Use the newest by modification time
        latest_eval_path = max(eval_files, key=lambda p: p.stat().st_mtime)
        eval_data = _safe_load_json(str(latest_eval_path))

        if isinstance(eval_data, dict):
            agg = eval_data.get("aggregate_metrics", {}) or {}

            avg_combined = agg.get("avg_combined_score")
            avg_profile_alignment = agg.get("avg_profile_alignment")
            avg_gt_quality = agg.get("avg_ground_truth_quality")
            pass_rate = agg.get("pass_rate")

            if avg_combined is not None:
                try:
                    MODEL_AVG_COMBINED_SCORE.set(float(avg_combined))
                except Exception as e:
                    logger.warning(
                        "Failed to set MODEL_AVG_COMBINED_SCORE",
                        value=avg_combined,
                        error=str(e),
                    )

            if avg_profile_alignment is not None:
                try:
                    MODEL_AVG_PROFILE_ALIGNMENT.set(float(avg_profile_alignment))
                except Exception as e:
                    logger.warning(
                        "Failed to set MODEL_AVG_PROFILE_ALIGNMENT",
                        value=avg_profile_alignment,
                        error=str(e),
                    )

            if avg_gt_quality is not None:
                try:
                    MODEL_AVG_GROUND_TRUTH_QUALITY.set(float(avg_gt_quality))
                except Exception as e:
                    logger.warning(
                        "Failed to set MODEL_AVG_GROUND_TRUTH_QUALITY",
                        value=avg_gt_quality,
                        error=str(e),
                    )

            if pass_rate is not None:
                try:
                    MODEL_PASS_RATE.set(float(pass_rate))
                except Exception as e:
                    logger.warning(
                        "Failed to set MODEL_PASS_RATE",
                        value=pass_rate,
                        error=str(e),
                    )

    # -------------------------------------------------------------------------
    # 5) Data drift (production, from data_drift_report.json)
    # -------------------------------------------------------------------------
    drift_path = backend_root / "bias_config" / "data_drift_report.json"
    drift_data = _safe_load_json(str(drift_path))

    if isinstance(drift_data, dict):
        drift_score_val = drift_data.get("overall_drift_score")
        if drift_score_val is not None:
            try:
                drift_score = float(drift_score_val)
                DATA_DRIFT_OVERALL_SCORE.set(drift_score)
            except Exception as e:
                logger.warning(
                    "Failed to set DATA_DRIFT_OVERALL_SCORE",
                    value=drift_score_val,
                    error=str(e),
                )

    # -------------------------------------------------------------------------
    # 6) Retrain trigger flag based on performance + drift
    # -------------------------------------------------------------------------
    try:
        if avg_combined is not None and drift_score is not None:
            retrain_flag = 1 if (float(avg_combined) < 0.20 or float(drift_score) > 0.6) else 0
            RETRAIN_NEEDED.set(retrain_flag)
        else:
            # If we don't have both signals yet, default to 0
            RETRAIN_NEEDED.set(0)
    except Exception as e:
        logger.warning(
            "Failed to compute/set RETRAIN_NEEDED",
            avg_combined=avg_combined,
            drift_score=drift_score,
            error=str(e),
        )


# -----------------------------------------------------------------------------
# Application lifespan (startup / shutdown)
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events with Cloud Run resilience.
    """
    global _startup_complete

    logger.info(
        "Starting CiteConnect application",
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
    )

    # Initialize default app state
    app.state.db = None
    app.state.services_healthy = {
        "database": False,
        "models": False,
        "repositories": False,
    }

    try:
        # ---------------------------------------------------------------------
        # Database connection
        # ---------------------------------------------------------------------
        logger.info("Attempting database connection")
        try:
            await db.connect()
            is_healthy = await db.health_check()
            if is_healthy:
                logger.info("Database connected successfully")
                app.state.db = db
                app.state.services_healthy["database"] = True
            else:
                logger.warning(
                    "Database health check failed - continuing in limited mode"
                )
        except Exception as e:
            logger.warning(
                f"Database connection failed - continuing in limited mode: {e}"
            )

        # ---------------------------------------------------------------------
        # Repositories (only if DB is available)
        # ---------------------------------------------------------------------
        if app.state.services_healthy["database"]:
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

                app.state.services_healthy["repositories"] = True
                logger.info("Repositories initialized successfully")
            except Exception as e:
                logger.warning(f"Repository initialization failed: {e}")

        # ---------------------------------------------------------------------
        # ML services (embedding models)
        # ---------------------------------------------------------------------
        try:
            logger.info("Attempting ML services initialization")
            embedding_service = EmbeddingService()
            model_health = embedding_service.health_check()

            app.state.embedding_service = embedding_service
            if any(model_health.values()):
                app.state.services_healthy["models"] = True
                logger.info("ML services initialized successfully")
            else:
                logger.warning(
                    "Models failed health check - continuing with limited ML functionality"
                )
        except Exception as e:
            logger.warning(f"ML service initialization failed: {e}")
            app.state.embedding_service = None

        # ---------------------------------------------------------------------
        # Full application services (only if dependencies are met)
        # ---------------------------------------------------------------------
        if app.state.services_healthy["database"] and app.state.services_healthy[
            "repositories"
        ]:
            try:
                logger.info("Initializing full application services")
                from app.services.bootstrap.ground_truth_service import GroundTruthService
                from app.services.bootstrap.experiment_service import ExperimentService
                from app.services.runtime.user_state_service import UserStateService
                from app.services.evaluation_service import EvaluationService
                from app.services.recommendation_service import RecommendationService
                from app.services.runtime.recommendation_orchestrator import (
                    RecommendationOrchestrator,
                )

                app.state.ground_truth_service = GroundTruthService(
                    app.state.ground_truth_repo, app.state.paper_repo
                )
                app.state.experiment_service = ExperimentService(db)
                app.state.user_state_service = UserStateService(
                    app.state.user_repo, app.state.interaction_repo
                )
                app.state.evaluation_service = EvaluationService(db)
                app.state.recommendation_service = RecommendationService(db)

                app.state.recommendation_orchestrator = RecommendationOrchestrator(
                    rec_service=app.state.recommendation_service,
                    eval_service=app.state.evaluation_service,
                    experiment_service=app.state.experiment_service,
                    user_state_service=app.state.user_state_service,
                )

                if hasattr(app.state.ground_truth_service, "initialize"):
                    await app.state.ground_truth_service.initialize()

                logger.info("Full application services initialized")
            except Exception as e:
                logger.warning(
                    f"Full service initialization failed - basic functionality available: {e}"
                )

        _startup_complete = True
        logger.info(
            "Application startup complete",
            services_available=app.state.services_healthy,
            mode="full"
            if all(app.state.services_healthy.values())
            else "limited",
        )

        # Hand control back to FastAPI
        yield

    except Exception as e:
        logger.error(f"Critical startup error: {e}", exc_info=True)
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


# -----------------------------------------------------------------------------
# FastAPI application
# -----------------------------------------------------------------------------
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered academic paper recommendation system",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Middleware: request logging
# -----------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests with timing."""
    request_id = str(time.time())
    start_time = time.time()

    logger.info(
        "Request received",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        client=request.client.host if request.client else "unknown",
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
            duration_ms=round(duration_ms, 2),
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
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An internal error occurred",
                    "request_id": request_id,
                }
            },
        )


# -----------------------------------------------------------------------------
# Exception handlers
# -----------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed messages."""
    logger.warning("Validation error", path=request.url.path, errors=exc.errors())
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid request parameters",
                "details": exc.errors(),
            }
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors gracefully."""
    logger.error(
        "Unhandled exception", path=request.url.path, error=str(exc), exc_info=True
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
            }
        },
    )


# -----------------------------------------------------------------------------
# Health & root endpoints
# -----------------------------------------------------------------------------
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
                "message": "Application is still initializing",
            },
        )

    health_status: Dict[str, Any] = {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "services": getattr(app.state, "services_healthy", {}),
        "checks": {},
    }

    try:
        # Database
        if hasattr(app.state, "db") and app.state.db is not None:
            try:
                logger.debug("Performing database health check")
                db_healthy = await app.state.db.health_check()
                health_status["checks"]["database"] = (
                    "healthy" if db_healthy else "degraded"
                )
            except Exception as e:
                health_status["checks"]["database"] = f"error: {str(e)}"
        else:
            health_status["checks"]["database"] = "not_available"

        # Models
        if (
            hasattr(app.state, "embedding_service")
            and app.state.embedding_service is not None
        ):
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

        health_status["status"] = "healthy"
        logger.debug("Health check complete", status=health_status["status"])
        return health_status

    except Exception as e:
        logger.error("Health check failed", error=str(e), exc_info=True)
        return {
            "status": "healthy",
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "mode": "basic",
            "note": "API is functional with limited services",
        }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "docs": "/docs" if settings.DEBUG else "disabled",
        "health": "/health",
    }

@app.get("/debug/embedding-service")
async def debug_embedding_service():
    import traceback
    try:
        from app.services.bootstrap.embedding_service import EmbeddingService
        service = EmbeddingService()
        
        return {
            "service_initialized": True,
            "models_loaded": list(service.models.keys()),
            "health_check": service.health_check()
        }
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

# -----------------------------------------------------------------------------
# Routers
# -----------------------------------------------------------------------------
from app.api.v1 import graph, recommendations, users, papers, interactions, pubsub

app.include_router(
    graph.router, prefix=f"{settings.API_V1_PREFIX}/graph", tags=["graph"]
)
app.include_router(
    recommendations.router,
    prefix=f"{settings.API_V1_PREFIX}/recommendations",
    tags=["recommendations"],
)
app.include_router(
    users.router, prefix=f"{settings.API_V1_PREFIX}/users", tags=["users"]
)
app.include_router(
    papers.router, prefix=f"{settings.API_V1_PREFIX}/papers", tags=["papers"]
)
app.include_router(
    interactions.router,
    prefix=f"{settings.API_V1_PREFIX}/interactions",
    tags=["interactions"],
)

app.include_router(pubsub.router, prefix="/pubsub")


# -----------------------------------------------------------------------------
# Prometheus /metrics endpoint
# -----------------------------------------------------------------------------
@app.get("/metrics")
async def metrics_endpoint():
    """
    Prometheus metrics endpoint.
    Reads latest JSON artifacts and exposes them as metrics.
    """
    update_monitoring_metrics()
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


# -----------------------------------------------------------------------------
# Local dev entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
)
