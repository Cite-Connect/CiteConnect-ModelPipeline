# =============================================================================
# app/db/repositories/__init__.py
# =============================================================================
"""Data access repositories."""
from app.db.repositories.base import BaseRepository
from app.db.repositories.user_repo import UserRepository
from app.db.repositories.paper_repo import PaperRepository
from app.db.repositories.embedding_repo import EmbeddingRepository
from app.db.repositories.ground_truth_repo import GroundTruthRepository
from app.db.repositories.interaction_repo import InteractionRepository

__all__ = [
    "BaseRepository",
    "UserRepository",
    "PaperRepository",
    "EmbeddingRepository",
    "GroundTruthRepository",
    "InteractionRepository"
]