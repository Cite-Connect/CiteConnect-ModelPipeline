"""
Pytest configuration and fixtures for all tests.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Generator
import numpy as np

# Mock database connections
@pytest.fixture
def mock_db():
    """Mock database connection."""
    db = AsyncMock()
    db.execute = AsyncMock()
    db.fetch_one = AsyncMock()
    db.fetch_all = AsyncMock()
    return db

@pytest.fixture
def mock_user_repo():
    """Mock user repository."""
    repo = AsyncMock()
    repo.find_by_id = AsyncMock(return_value={
        'user_id': 1,
        'email': 'test@example.com',
        'name': 'Test User',
        'is_active': True
    })
    repo.find_by_email = AsyncMock(return_value=None)
    return repo

@pytest.fixture
def sample_user():
    """Sample user data."""
    return {
        'user_id': 1,
        'email': 'test@example.com',
        'name': 'Test User',
        'is_active': True,
        'primary_domain': 'healthcare',
        'research_stage': 'phd',
        'reading_level': 'intermediate'
    }

@pytest.fixture
def sample_embedding():
    """Sample embedding vector (768 dimensions for SPECTER2)."""
    return np.random.rand(768).astype(np.float32)

@pytest.fixture
def mock_embedding_service():
    """Mock embedding service."""
    service = MagicMock()
    service.encode_text = MagicMock(return_value=np.random.rand(768).astype(np.float32))
    service.encode_batch = MagicMock(return_value=np.random.rand(5, 768).astype(np.float32))
    return service

@pytest.fixture
def mock_llm_service():
    """Mock LLM service."""
    service = MagicMock()
    service.enabled = True
    service.refine_search_query = AsyncMock(return_value="refined query")
    return service

