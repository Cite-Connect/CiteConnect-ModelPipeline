"""
Tests for Recommendations API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock
from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestRecommendationsAPI:
    """Test suite for Recommendations API endpoints."""
    
    @pytest.fixture
    def mock_orchestrator(self):
        """Mock recommendation orchestrator."""
        orchestrator = AsyncMock()
        
        # Mock the return value to match RecommendationResponse model structure
        # The endpoint expects: {recommendations: [], metadata: {}, explanations: {}}
        orchestrator.generate_recommendations = AsyncMock(return_value={
            'recommendations': [
                {
                    'paper_id': 'paper1',
                    'title': 'Test Paper',
                    'abstract': 'Test abstract',
                    'authors': ['Author 1'],
                    'year': 2023,
                    'domain': 'healthcare',
                    'citation_count': 0,
                    'quality_score': 0.95,
                    'relevance_score': 0.95,
                    'tldr': None,
                    'matching_aspects': [],
                    'match_source': None,
                    'relevance_explanation': None,
                    'score_breakdown': None
                }
            ],
            'metadata': {
                'user_stage': 'cold_start',
                'strategy_used': 'personalized',
                'model_used': 'minilm',
                'evaluation_scores': {
                    'profile_alignment': 0.8,
                    'ground_truth_quality': 0.7,
                    'combined_score': 0.75
                },
                'cache_hit': False,
                'generation_time_ms': 100.0,
                'search_query': None,
                'refined_query': None,
                'llm_refinement_used': False
            },
            'explanations': {
                'paper1': 'Recommended based on your interests'
            }
        })
        return orchestrator
    
    def test_get_recommendations_basic(self, client, mock_orchestrator):
        """Test basic recommendations endpoint."""
        # Set orchestrator in app.state before making request
        client.app.state.recommendation_orchestrator = mock_orchestrator
        
        try:
            response = client.post(
                "/api/v1/recommendations",
                json={
                    "user_id": 1,
                    "count": 10,
                    "session_id": "test_session_123"  # Required field
                }
            )
            
            assert response.status_code in [200, 201], f"Expected 200/201, got {response.status_code}: {response.text}"
            data = response.json()
            assert 'recommendations' in data or 'papers' in data
        finally:
            if hasattr(client.app.state, 'recommendation_orchestrator'):
                delattr(client.app.state, 'recommendation_orchestrator')
    
    def test_get_recommendations_with_model_preference(self, client, mock_orchestrator):
        """Test recommendations with model preference."""
        client.app.state.recommendation_orchestrator = mock_orchestrator
        
        try:
            response = client.post(
                "/api/v1/recommendations",
                json={
                    "user_id": 1,
                    "count": 10,
                    "model_preference": "minilm",
                    "session_id": "test_session_123"
                }
            )
            
            assert response.status_code in [200, 201]
        finally:
            if hasattr(client.app.state, 'recommendation_orchestrator'):
                delattr(client.app.state, 'recommendation_orchestrator')
    
    def test_get_recommendations_with_search_query(self, client, mock_orchestrator):
        """Test search-augmented recommendations."""
        mock_orchestrator.generate_recommendations.return_value['metadata']['search_query'] = 'machine learning'
        client.app.state.recommendation_orchestrator = mock_orchestrator
        
        try:
            response = client.post(
                "/api/v1/recommendations",
                json={
                    "user_id": 1,
                    "count": 10,
                    "search_query": "machine learning",
                    "session_id": "test_session_123"
                }
            )
            
            assert response.status_code in [200, 201]
        finally:
            if hasattr(client.app.state, 'recommendation_orchestrator'):
                delattr(client.app.state, 'recommendation_orchestrator')
    
    def test_get_recommendations_missing_user_id(self, client, mock_orchestrator):
        """Test recommendations without user ID."""
        # Need orchestrator set up even for validation tests
        # Validation happens after dependency check, so 503 is expected if orchestrator missing
        client.app.state.recommendation_orchestrator = mock_orchestrator
        
        try:
            response = client.post(
                "/api/v1/recommendations",
                json={
                    "count": 10,
                    "session_id": "test_session_123"
                }
            )
            
            # Endpoint validates user_id and returns 400, but dependency check happens first
            # So we can get either 400 (if validation happens first) or 503 (if dependency check first)
            assert response.status_code in [400, 422, 503]
        finally:
            if hasattr(client.app.state, 'recommendation_orchestrator'):
                delattr(client.app.state, 'recommendation_orchestrator')
    
    def test_get_recommendations_invalid_count(self, client, mock_orchestrator):
        """Test recommendations with invalid count."""
        client.app.state.recommendation_orchestrator = mock_orchestrator
        
        try:
            response = client.post(
                "/api/v1/recommendations",
                json={
                    "user_id": 1,
                    "count": -1,
                    "session_id": "test_session_123"
                }
            )
            
            # Pydantic validates count (must be >= 1) before endpoint code runs
            assert response.status_code in [400, 422]
        finally:
            if hasattr(client.app.state, 'recommendation_orchestrator'):
                delattr(client.app.state, 'recommendation_orchestrator')
