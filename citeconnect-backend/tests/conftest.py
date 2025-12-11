"""
Pytest configuration and fixtures for all tests.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Generator
import numpy as np
import logging
import structlog

# Configure logging for tests at module level (runs once)
def pytest_configure(config):
    """Configure logging for test environment."""
    # Simple structlog configuration for tests without filter_by_level
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(),  # Simple console output for tests
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=None,  # Suppress output during tests
        level=logging.WARNING,  # Only show warnings and errors
    )

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

