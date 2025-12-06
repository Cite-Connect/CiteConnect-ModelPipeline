"""
Embedding service for text-to-vector conversion.
Loads ML models at startup and provides simple encoding interface.
"""
from typing import Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    Singleton service for managing embedding models.
    Loads models once at startup and reuses them for all requests.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern - only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize embedding service (only once)."""
        if not self._initialized:
            self.models: Dict[str, SentenceTransformer] = {}
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(
                "EmbeddingService created",
                device=self.device
            )
            
            # Load models immediately
            self._load_models()
            
            EmbeddingService._initialized = True
    
    def _load_models(self):
        """
        Load both embedding models into memory.
        This is called once when the service is first created.
        """
        logger.info("Loading embedding models")
        
        try:
            # Load MiniLM (384 dimensions)
            logger.info(
                "Loading MiniLM model",
                model_name=settings.EMBEDDING_MODEL_MINILM
            )
            
            self.models['minilm'] = SentenceTransformer(
                settings.EMBEDDING_MODEL_MINILM,
                device=self.device
            )
            
            logger.info(
                "MiniLM model loaded",
                dimensions=384
            )
            
            # Load SPECTER (768 dimensions)
            logger.info(
                "Loading SPECTER model",
                model_name=settings.EMBEDDING_MODEL_SPECTER
            )
            
            self.models['specter'] = SentenceTransformer(
                settings.EMBEDDING_MODEL_SPECTER,
                device=self.device
            )
            
            logger.info(
                "SPECTER model loaded",
                dimensions=768
            )
            
            logger.info(
                "All embedding models loaded successfully",
                models=list(self.models.keys()),
                device=self.device
            )
            
        except Exception as e:
            logger.error(
                "Failed to load embedding models",
                error=str(e),
                exc_info=True
            )
            raise RuntimeError(f"Could not load embedding models: {e}")
    
    def encode_text(
        self,
        text: str,
        model: str = 'minilm',
        normalize: bool = True
    ) -> np.ndarray:
        """
        Convert text to embedding vector.
        
        Args:
            text: Text to encode (e.g., "machine learning medical imaging")
            model: Model to use ('minilm' or 'specter')
            normalize: Whether to normalize the embedding vector
            
        Returns:
            numpy array of embedding (384-dim for minilm, 768-dim for specter)
            
        Example:
            >>> service = EmbeddingService()
            >>> embedding = service.encode_text("machine learning", model='minilm')
            >>> embedding.shape
            (384,)
            >>> embedding[:3]
            array([0.234, -0.567, 0.891])
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for encoding")
            # Return zero vector
            dim = 384 if model == 'minilm' else 768
            return np.zeros(dim)
        
        if model not in self.models:
            logger.error(
                "Model not found",
                model=model,
                available_models=list(self.models.keys())
            )
            raise ValueError(
                f"Model '{model}' not available. "
                f"Available: {list(self.models.keys())}"
            )
        
        try:
            # Get the model
            encoder = self.models[model]
            
            # Truncate if too long (models have max length ~512 tokens)
            max_length = 512
            if len(text.split()) > max_length:
                text = ' '.join(text.split()[:max_length])
                logger.debug(
                    "Text truncated",
                    original_words=len(text.split()),
                    truncated_to=max_length
                )
            
            # Encode text to vector
            embedding = encoder.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=False
            )
            
            logger.debug(
                "Text encoded successfully",
                text_preview=text[:50],
                model=model,
                embedding_shape=embedding.shape
            )
            
            return embedding
            
        except Exception as e:
            logger.error(
                "Text encoding failed",
                text_preview=text[:100],
                model=model,
                error=str(e),
                exc_info=True
            )
            raise
    
    def encode_batch(
        self,
        texts: list[str],
        model: str = 'minilm',
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode multiple texts efficiently in batches.
        
        Args:
            texts: List of texts to encode
            model: Model to use ('minilm' or 'specter')
            batch_size: Number of texts to process at once
            normalize: Whether to normalize embeddings
            
        Returns:
            numpy array of shape (num_texts, embedding_dim)
            
        Example:
            >>> texts = ["text 1", "text 2", "text 3"]
            >>> embeddings = service.encode_batch(texts, model='minilm')
            >>> embeddings.shape
            (3, 384)
        """
        if not texts:
            logger.warning("Empty text list provided for batch encoding")
            return np.array([])
        
        if model not in self.models:
            raise ValueError(
                f"Model '{model}' not available. "
                f"Available: {list(self.models.keys())}"
            )
        
        try:
            encoder = self.models[model]
            
            logger.info(
                "Encoding text batch",
                num_texts=len(texts),
                model=model,
                batch_size=batch_size
            )
            
            # Encode all texts
            embeddings = encoder.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=len(texts) > 100  # Show progress for large batches
            )
            
            logger.info(
                "Batch encoding complete",
                num_texts=len(texts),
                model=model,
                embeddings_shape=embeddings.shape
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(
                "Batch encoding failed",
                num_texts=len(texts),
                model=model,
                error=str(e),
                exc_info=True
            )
            raise
    
    def get_model_info(self, model: str) -> dict:
        """
        Get information about a loaded model.
        
        Args:
            model: Model name ('minilm' or 'specter')
            
        Returns:
            Dict with model information
        """
        if model not in self.models:
            return {'loaded': False}
        
        encoder = self.models[model]
        
        return {
            'loaded': True,
            'name': model,
            'model_path': settings.EMBEDDING_MODEL_MINILM if model == 'minilm' else settings.EMBEDDING_MODEL_SPECTER,
            'dimensions': 384 if model == 'minilm' else 768,
            'max_seq_length': encoder.max_seq_length,
            'device': str(self.device)
        }
    
    def health_check(self) -> dict:
        """
        Check if models are loaded and functional.
        
        Returns:
            Dict with health status for each model
        """
        logger.debug("Performing embedding service health check")
        
        health = {}
        
        for model_name in ['minilm', 'specter']:
            try:
                # Test encoding
                test_embedding = self.encode_text(
                    "health check test",
                    model=model_name
                )
                
                # Verify embedding is correct shape
                expected_dim = 384 if model_name == 'minilm' else 768
                is_healthy = (
                    test_embedding is not None and
                    len(test_embedding) == expected_dim
                )
                
                health[model_name] = 'healthy' if is_healthy else 'unhealthy'
                
            except Exception as e:
                logger.error(
                    "Health check failed for model",
                    model=model_name,
                    error=str(e)
                )
                health[model_name] = 'unhealthy'
        
        logger.info(
            "Health check complete",
            results=health
        )
        
        return health


# Create singleton instance that will be imported by other services
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """
    Get the singleton embedding service instance.
    
    Returns:
        EmbeddingService: The singleton instance
    """
    global _embedding_service
    
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    
    return _embedding_service