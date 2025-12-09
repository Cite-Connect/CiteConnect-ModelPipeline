"""
Tests for RecommendationService.
"""
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.recommendation_service import RecommendationService
from app.db.connection import DatabaseConnection


@pytest.mark.asyncio
class TestRecommendationService:
    """Test suite for RecommendationService."""
    
    @pytest.fixture
    def mock_db(self):
        """Mock database connection."""
        db = AsyncMock(spec=DatabaseConnection)
        return db
    
    @pytest.fixture
    def recommendation_service(self, mock_db):
        """Create RecommendationService instance."""
        service = RecommendationService(mock_db)
        return service
    
    async def test_generate_cold_start_recommendations(self, recommendation_service, mock_db):
        """Test generating cold-start recommendations."""
        # Mock all internal methods to avoid complex dependencies
        with patch.object(recommendation_service.user_repo, 'get_profile', new_callable=AsyncMock) as mock_profile:
            mock_profile.return_value = {
                'user_id': 1,
                'primary_domain': 'healthcare',
                'name': 'Test User',
                'reading_level': 'intermediate',
                'research_stage': 'phd'
            }
            with patch.object(recommendation_service.user_repo, 'get_user_interests', new_callable=AsyncMock) as mock_interests:
                mock_interests.return_value = [
                    {'interest_term': 'machine learning', 'interest_level': 1}
                ]
                with patch.object(recommendation_service.user_embedding_service, 'get_or_generate_user_embeddings', new_callable=AsyncMock) as mock_emb:
                    mock_emb.return_value = {
                        'minilm': np.random.rand(384).astype(np.float32)
                    }
                    with patch.object(recommendation_service, '_retrieve_semantic_candidates', new_callable=AsyncMock) as mock_semantic:
                        mock_semantic.return_value = []
                        with patch.object(recommendation_service, '_retrieve_canonical_candidates', new_callable=AsyncMock) as mock_canonical:
                            mock_canonical.return_value = []
                            with patch.object(recommendation_service, '_retrieve_ground_truth_candidates', new_callable=AsyncMock) as mock_gt:
                                mock_gt.return_value = []
                                with patch.object(recommendation_service, '_apply_multi_factor_scoring', new_callable=AsyncMock) as mock_scoring:
                                    mock_scoring.return_value = []
                                    with patch.object(recommendation_service, '_apply_diversity_filtering', new_callable=AsyncMock) as mock_diversity:
                                        mock_diversity.return_value = []
                                        
                                        result = await recommendation_service.generate_cold_start_recommendations(
                                            user_id=1,
                                            count=10,
                                            model='minilm'
                                        )
        
        assert isinstance(result, dict)
        assert 'papers' in result
    
    def test_default_weights(self, recommendation_service):
        """Test default scoring weights."""
        assert 'semantic' in recommendation_service.DEFAULT_COLD_START_WEIGHTS
        assert 'citation' in recommendation_service.DEFAULT_COLD_START_WEIGHTS
        assert 'recency' in recommendation_service.DEFAULT_COLD_START_WEIGHTS
        
        # Weights should sum approximately to 1.0
        total = sum(recommendation_service.DEFAULT_COLD_START_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01
    
    def test_load_bias_config(self, recommendation_service):
        """Test loading bias mitigation configuration."""
        # Should handle missing config gracefully
        config = recommendation_service._load_bias_config()
        assert isinstance(config, dict)
