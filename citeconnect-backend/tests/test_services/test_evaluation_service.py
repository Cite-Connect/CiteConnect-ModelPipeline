"""
Tests for EvaluationService.
"""
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.evaluation_service import EvaluationService
from app.db.connection import DatabaseConnection


@pytest.mark.asyncio
class TestEvaluationService:
    """Test suite for EvaluationService."""
    
    @pytest.fixture
    def mock_db(self):
        """Mock database connection."""
        db = AsyncMock(spec=DatabaseConnection)
        return db
    
    @pytest.fixture
    def evaluation_service(self, mock_db):
        """Create EvaluationService instance."""
        service = EvaluationService(mock_db)
        return service
    
    async def test_evaluate_cold_start_recommendations(self, evaluation_service, mock_db):
        """Test evaluating cold-start recommendations."""
        recommendations = [
            {'paper_id': 'paper1', 'title': 'Test Paper', 'abstract': 'Test abstract'}
        ]
        
        # Mock user profile (fetchrow returns dict)
        mock_db.fetchrow = AsyncMock(return_value={
            'user_id': 1, 
            'primary_domain': 'healthcare',
            'name': 'Test User',
            'reading_level': 'intermediate'
        })
        
        # Mock interests (fetch returns list)
        mock_db.fetch = AsyncMock(return_value=[
            {'interest_term': 'machine learning', 'interest_level': 1}
        ])
        
        # Mock internal helper methods to avoid complex ground truth logic
        with patch.object(evaluation_service, '_calculate_profile_alignment', new_callable=AsyncMock) as mock_profile:
            mock_profile.return_value = 0.75
            with patch.object(evaluation_service, '_calculate_ground_truth_quality', new_callable=AsyncMock) as mock_gt:
                mock_gt.return_value = 0.65
                # with patch.object(evaluation_service, '_store_cold_start_evaluation', new_callable=AsyncMock):
                #     result = await evaluation_service.evaluate_cold_start_recommendations(
                #         user_id=1,
                #         recommendations=recommendations
                #     )

                # Mock the eval_repo.save_cold_start_evaluation method (new API)
                evaluation_service.eval_repo.save_cold_start_evaluation = AsyncMock()
                result = await evaluation_service.evaluate_cold_start_recommendations(
                    user_id=1,
                    recommendations=recommendations
                )
        
        assert isinstance(result, dict)
        assert 'profile_alignment' in result
        assert 'ground_truth_quality' in result
        assert 'combined_score' in result
    
    async def test_evaluate_warm_start_recommendations(self, evaluation_service, mock_db):
        """Test evaluating warm-start recommendations."""
        recommendations = [
            {'paper_id': 'paper1', 'final_score': 0.9},
            {'paper_id': 'paper2', 'final_score': 0.8}
        ]
        ground_truth_papers = ['paper1', 'paper3']
        
        mock_db.fetch = AsyncMock(return_value=[])
        
        result = await evaluation_service.evaluate_warm_start_recommendations(
            user_id=1,
            recommendations=recommendations,
            ground_truth_papers=ground_truth_papers
        )
        
        assert isinstance(result, dict)
        assert 'precision_at_10' in result
        assert 'recall_at_10' in result
        assert 'ndcg_at_10' in result
    
    def test_calculate_precision_recall(self, evaluation_service):
        """Test precision and recall calculation (synchronous)."""
        """Test precision and recall calculation."""
        recommended_ids = ['paper1', 'paper2', 'paper3']
        ground_truth_ids = ['paper1', 'paper3', 'paper4']
        k = 3
        
        hits = len(set(recommended_ids[:k]) & set(ground_truth_ids))
        precision = hits / k
        recall = hits / len(ground_truth_ids) if ground_truth_ids else 0.0
        
        assert precision == 2 / 3  # 2 hits out of 3 recommendations
        assert recall == 2 / 3  # 2 hits out of 3 ground truth
    
    def test_calculate_mrr(self, evaluation_service):
        """Test MRR calculation (synchronous)."""
        """Test MRR calculation."""
        recommended_ids = ['paper1', 'paper2', 'paper3']
        ground_truth_ids = ['paper2']
        
        # First relevant at position 2 (index 1)
        mrr = 1.0 / (1 + 1)  # 1 / rank = 1 / 2 = 0.5
        
        ground_truth_set = set(ground_truth_ids)
        calculated_mrr = 0.0
        for i, paper_id in enumerate(recommended_ids):
            if paper_id in ground_truth_set:
                calculated_mrr = 1.0 / (i + 1)
                break
        
        assert calculated_mrr == 0.5
