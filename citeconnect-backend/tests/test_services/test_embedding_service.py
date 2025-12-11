"""
Tests for EmbeddingService.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from app.services.bootstrap.embedding_service import EmbeddingService


class TestEmbeddingService:
    """Test suite for EmbeddingService."""
    
    @pytest.fixture
    def mock_sentence_transformer(self):
        """Mock SentenceTransformer model."""
        model = MagicMock()
        # SentenceTransformer.encode() returns 1D array for single string: (384,)
        model.encode = MagicMock(return_value=np.random.rand(384).astype(np.float32))
        model.max_seq_length = 512
        return model
    
    @pytest.fixture
    def embedding_service(self, mock_sentence_transformer):
        """Create EmbeddingService instance with mocked model."""
        # Reset singleton before creating new instance
        EmbeddingService._instance = None
        EmbeddingService._initialized = False
        
        with patch('app.services.bootstrap.embedding_service.SentenceTransformer') as mock_st:
            mock_st.return_value = mock_sentence_transformer
            service = EmbeddingService()
            # Make sure the mocked model is used
            service.models = {'minilm': mock_sentence_transformer, 'specter': mock_sentence_transformer}
            return service
    
    def test_singleton_pattern(self, embedding_service):
        """Test that EmbeddingService is a singleton."""
        service1 = EmbeddingService()
        service2 = EmbeddingService()
        assert service1 is service2
    
    def test_encode_text_minilm(self, embedding_service, mock_sentence_transformer):
        """Test encoding text with MiniLM model."""
        text = "machine learning"
        result = embedding_service.encode_text(text, model='minilm')
        
        assert isinstance(result, np.ndarray)
        # SentenceTransformer returns 1D array for single string
        assert len(result) == 384  # MiniLM dimension
        mock_sentence_transformer.encode.assert_called()
    
    def test_encode_text_specter(self, embedding_service, mock_sentence_transformer):
        """Test encoding text with SPECTER model."""
        # Mock SPECTER to return 768-dim vector
        mock_sentence_transformer.encode.return_value = np.random.rand(768).astype(np.float32)
        
        text = "deep learning"
        result = embedding_service.encode_text(text, model='specter')
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 768  # SPECTER dimension
    
    def test_encode_batch(self, embedding_service, mock_sentence_transformer):
        """Test batch encoding."""
        texts = ["text1", "text2", "text3"]
        # Batch encoding returns 2D array: (num_texts, embedding_dim)
        mock_sentence_transformer.encode.return_value = np.random.rand(3, 384).astype(np.float32)
        
        results = embedding_service.encode_batch(texts, model='minilm')
        
        assert isinstance(results, np.ndarray)
        assert results.shape == (3, 384)
        mock_sentence_transformer.encode.assert_called()
    
    def test_get_model_info(self, embedding_service):
        """Test getting model information."""
        info = embedding_service.get_model_info('minilm')
        
        assert isinstance(info, dict)
        assert info['loaded'] is True
        assert info['name'] == 'minilm'
        assert info['dimensions'] == 384
    
    def test_get_model_info_invalid_model(self, embedding_service):
        """Test getting info for non-existent model."""
        info = embedding_service.get_model_info('invalid_model')
        
        assert isinstance(info, dict)
        assert info['loaded'] is False
    
    def test_invalid_model_raises_error(self, embedding_service):
        """Test that invalid model name raises error."""
        with pytest.raises(ValueError):
            embedding_service.encode_text("test", model='invalid_model')
    
    def test_health_check(self, embedding_service, mock_sentence_transformer):
        """Test health check functionality."""
        health = embedding_service.health_check()
        
        assert isinstance(health, dict)
        # Should check both models
        assert 'minilm' in health or 'specter' in health
