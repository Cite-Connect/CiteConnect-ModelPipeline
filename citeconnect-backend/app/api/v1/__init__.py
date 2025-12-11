# =============================================================================
# app/api/v1/__init__.py
# =============================================================================
"""API version 1 endpoints."""
from fastapi import APIRouter
from app.api.v1 import recommendations, users, papers, interactions, graph

router = APIRouter()

# Include all v1 routers
router.include_router(
    graph.router,           # ✅ Add this
    prefix="/graph",        # ✅ Add this
    tags=["graph"]          # ✅ Add this
)

# Include all v1 routers
router.include_router(
    recommendations.router,
    prefix="/recommendations",
    tags=["recommendations"]
)
router.include_router(
    users.router,
    prefix="/users",
    tags=["users"]
)
router.include_router(
    papers.router,
    prefix="/papers",
    tags=["papers"]
)
router.include_router(
    interactions.router,
    prefix="/interactions",
    tags=["interactions"]
)

