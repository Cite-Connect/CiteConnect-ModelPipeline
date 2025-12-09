# tests/conftest.py
"""
Integration test fixtures - use REAL databases with manual cleanup.
Only cleans up USER data - uses existing papers from database.
"""
import pytest
import asyncio
import os
from app.db.connection import DatabaseConnection
from app.core.security import hash_password

# Test data identifiers
TEST_EMAIL_SUFFIX = '@test.integration.com'


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
async def test_db():
    """
    Real test database connection with cleanup.
    
    Uses EXISTING papers from database.
    Only creates and cleans up USER data.
    """
    # Connect to database
    db = DatabaseConnection()
    await db.connect()
    
    try:
        yield db  # Test runs here
    finally:
        # Cleanup - only delete USER data (not papers)
        try:
            # Delete in reverse dependency order
            await db.execute(
                """DELETE FROM user_interactions 
                   WHERE user_id IN (SELECT user_id FROM users WHERE email LIKE $1)""",
                f'%{TEST_EMAIL_SUFFIX}'
            )
            await db.execute(
                """DELETE FROM user_interest_hierarchy 
                   WHERE user_id IN (SELECT user_id FROM users WHERE email LIKE $1)""",
                f'%{TEST_EMAIL_SUFFIX}'
            )
            await db.execute(
                """DELETE FROM user_profiles_extended 
                   WHERE user_id IN (SELECT user_id FROM users WHERE email LIKE $1)""",
                f'%{TEST_EMAIL_SUFFIX}'
            )
            await db.execute(
                """DELETE FROM user_recommendation_state 
                   WHERE user_id IN (SELECT user_id FROM users WHERE email LIKE $1)""",
                f'%{TEST_EMAIL_SUFFIX}'
            )
            await db.execute(
                """DELETE FROM cold_start_evaluations 
                   WHERE user_id IN (SELECT user_id FROM users WHERE email LIKE $1)""",
                f'%{TEST_EMAIL_SUFFIX}'
            )
            await db.execute(
                """DELETE FROM user_embeddings_minilm 
                   WHERE user_id IN (SELECT user_id FROM users WHERE email LIKE $1)""",
                f'%{TEST_EMAIL_SUFFIX}'
            )
            await db.execute(
                """DELETE FROM user_embeddings_specter 
                   WHERE user_id IN (SELECT user_id FROM users WHERE email LIKE $1)""",
                f'%{TEST_EMAIL_SUFFIX}'
            )
            await db.execute(
                "DELETE FROM users WHERE email LIKE $1",
                f'%{TEST_EMAIL_SUFFIX}'
            )
            
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")
        
        # Disconnect
        await db.disconnect()


@pytest.fixture
def test_user_data():
    """Sample user data for integration tests."""
    return {
        'email': f'test_user_{os.getpid()}@test.integration.com',
        'password': 'TestPassword123!',
        'name': 'Integration Test User',
        'primary_domain': 'healthcare',
        'research_stage': 'phd',
        'reading_level': 'intermediate',
        'interests': ['machine learning', 'nlp', 'healthcare ai', 'clinical trials']
    }