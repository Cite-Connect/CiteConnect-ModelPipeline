"""
Embedding service for generating and managing text embeddings.
Loads models at startup and provides batch processing capabilities.
"""
from typing import List, Optional, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from app.config import settings
from app.utils.logger import get_logger
from app.db.repositories.embedding_repo import EmbeddingRepository

logger = get_logger(__name__)


class EmbeddingService:
    """
    Manages multiple embedding models with caching and batch processing.
    Models are loaded once at startup for efficiency.
    """
    
    def __init__(self, embedding_repo: EmbeddingRepository):
        """
        Initialize embedding service.
        
        Args:
            embedding_repo: Repository for embedding storage
        """
        self.embedding_repo = embedding_repo
        self.models: Dict[str, SentenceTransformer] = {}
        self.model_dimensions: Dict[str, int] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(
            "EmbeddingService initialized",
            device=self.device
        )
    
    async def initialize(self) -> None:
        """
        Load embedding models into memory.
        Called during application startup.
        """
        logger.info("Loading embedding models")
        
        try:
            # Load MiniLM model (384 dimensions)
            logger.info(
                "Loading MiniLM model",
                model=settings.EMBEDDING_MODEL_MINILM
            )
            
            minilm = SentenceTransformer(
                settings.EMBEDDING_MODEL_MINILM,
                cache_folder=settings.MODEL_CACHE_DIR,
                device=self.device
            )
            self.models['all-MiniLM-L6-v2'] = minilm
            self.model_dimensions['all-MiniLM-L6-v2'] = 384
            
            logger.info("MiniLM model loaded successfully")
            
            # Load SPECTER model (768 dimensions)
            logger.info(
                "Loading SPECTER2 model",
                model=settings.EMBEDDING_MODEL_SPECTER
            )
            
            specter = SentenceTransformer(
                settings.EMBEDDING_MODEL_SPECTER,
                cache_folder=settings.MODEL_CACHE_DIR,
                device=self.device
            )
            self.models['specter2'] = specter
            self.model_dimensions['specter2'] = 768
            
            logger.info("SPECTER2 model loaded successfully")
            
            logger.info(
                "All embedding models loaded",
                models=list(self.models.keys()),
                device=self.device
            )
            
        except Exception as e:
            logger.error(
                "Model loading failed",
                error=str(e),
                exc_info=True
            )
            raise
    
    def _get_model(self, model_name: str) -> SentenceTransformer:
        """
        Get loaded model by name.
        
        Args:
            model_name: Model identifier
            
        Returns:
            SentenceTransformer: Loaded model
            
        Raises:
            ValueError: If model not loaded
        """
        if model_name not in self.models:
            logger.error(
                "Model not found",
                model=model_name,
                available=list(self.models.keys())
            )
            raise ValueError(f"Model {model_name} not loaded")
        
        return self.models[model_name]
    
    async def embed_text(
        self,
        text: str,
        model_name: str,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embedding for single text.
        
        Args:
            text: Text to embed
            model_name: Model to use
            normalize: Whether to normalize vector
            
        Returns:
            ndarray: Embedding vector
        """
        logger.debug(
            "Generating text embedding",
            text_length=len(text),
            model=model_name
        )
        
        try:
            model = self._get_model(model_name)
            
            # Truncate if necessary
            if len(text) > settings.EMBEDDING_MAX_LENGTH:
                text = text[:settings.EMBEDDING_MAX_LENGTH]
                logger.debug(
                    "Text truncated",
                    max_length=settings.EMBEDDING_MAX_LENGTH
                )
            
            # Generate embedding
            embedding = model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=False
            )
            
            logger.debug(
                "Embedding generated",
                dimension=len(embedding),
                model=model_name
            )
            
            return embedding
            
        except Exception as e:
            logger.error(
                "Embedding generation failed",
                model=model_name,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def embed_batch(
        self,
        texts: List[str],
        model_name: str,
        batch_size: Optional[int] = None,
        normalize: bool = True
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            model_name: Model to use
            batch_size: Batch size for processing
            normalize: Whether to normalize vectors
            
        Returns:
            List[ndarray]: Embedding vectors
        """
        if not texts:
            logger.debug("No texts provided for batch embedding")
            return []
        
        logger.info(
            "Generating batch embeddings",
            count=len(texts),
            model=model_name
        )
        
        try:
            model = self._get_model(model_name)
            
            # Truncate texts if necessary
            processed_texts = [
                t[:settings.EMBEDDING_MAX_LENGTH] for t in texts
            ]
            
            # Use configured batch size if not provided
            if batch_size is None:
                batch_size = settings.EMBEDDING_BATCH_SIZE
            
            # Generate embeddings in batches
            embeddings = model.encode(
                processed_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=len(texts) > 100
            )
            
            logger.info(
                "Batch embeddings generated",
                count=len(embeddings),
                model=model_name
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(
                "Batch embedding generation failed",
                count=len(texts),
                model=model_name,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def embed_paper(
        self,
        paper_id: str,
        title: str,
        abstract: Optional[str],
        model_name: str,
        save_to_db: bool = True
    ) -> np.ndarray:
        """
        Generate and optionally save paper embedding.
        
        Args:
            paper_id: Paper identifier
            title: Paper title
            abstract: Paper abstract
            model_name: Model to use
            save_to_db: Whether to save to database
            
        Returns:
            ndarray: Paper embedding
        """
        logger.debug(
            "Generating paper embedding",
            paper_id=paper_id,
            model=model_name,
            has_abstract=abstract is not None
        )
        
        # Combine title and abstract with weighting
        if abstract:
            # Title is repeated 3x for emphasis
            text = f"{title} {title} {title} {abstract}"
        else:
            text = f"{title} {title} {title}"
        
        try:
            embedding = await self.embed_text(text, model_name)
            
            if save_to_db:
                await self.embedding_repo.save_paper_embedding(
                    paper_id=paper_id,
                    embedding=embedding,
                    model_name=model_name,
                    embedding_source="title_abstract"
                )
                logger.debug(
                    "Paper embedding saved to database",
                    paper_id=paper_id
                )
            
            logger.info(
                "Paper embedding generated",
                paper_id=paper_id,
                model=model_name
            )
            
            return embedding
            
        except Exception as e:
            logger.error(
                "Paper embedding generation failed",
                paper_id=paper_id,
                model=model_name,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def embed_user_profile(
        self,
        user_id: int,
        research_stage: str,
        primary_domain: str,
        interests: List[str],
        research_goals: Optional[List[str]],
        model_name: str,
        save_to_db: bool = True
    ) -> np.ndarray:
        """
        Generate user profile embedding from attributes.
        
        Args:
            user_id: User identifier
            research_stage: User's research stage
            primary_domain: Primary research domain
            interests: List of interests
            research_goals: Research goals
            model_name: Model to use
            save_to_db: Whether to save to database
            
        Returns:
            ndarray: User profile embedding
        """
        logger.debug(
            "Generating user profile embedding",
            user_id=user_id,
            model=model_name,
            interests_count=len(interests)
        )
        
        # Build weighted text representation
        text_components = []
        
        # Research stage (weight: 2)
        text_components.extend([f"{research_stage} researcher"] * 2)
        
        # Primary domain (weight: 3)
        text_components.extend([primary_domain] * 3)
        
        # Interests (weight: 5 each - highest importance)
        for interest in interests:
            text_components.extend([interest] * 5)
        
        # Research goals (weight: 3)
        if research_goals:
            for goal in research_goals:
                text_components.extend([goal] * 3)
        
        # Combine into single text
        profile_text = " ".join(text_components)
        
        try:
            embedding = await self.embed_text(profile_text, model_name)
            
            if save_to_db:
                await self.embedding_repo.save_user_embedding(
                    user_id=user_id,
                    embedding=embedding,
                    model_name=model_name,
                    embedding_source="profile_based"
                )
                logger.debug(
                    "User embedding saved to database",
                    user_id=user_id
                )
            
            logger.info(
                "User profile embedding generated",
                user_id=user_id,
                model=model_name
            )
            
            return embedding
            
        except Exception as e:
            logger.error(
                "User profile embedding generation failed",
                user_id=user_id,
                model=model_name,
                error=str(e),
                exc_info=True
            )
            raise
    
    def get_dimension(self, model_name: str) -> int:
        """
        Get embedding dimension for model.
        
        Args:
            model_name: Model identifier
            
        Returns:
            int: Embedding dimension
        """
        return self.model_dimensions.get(model_name, 384)
    
    def get_version(self, model_name: str) -> str:
        """
        Get model version string.
        
        Args:
            model_name: Model identifier
            
        Returns:
            str: Model version
        """
        if model_name not in self.models:
            return "unknown"
        
        # Return model name as version for now
        # In production, track actual model versions
        return model_name
    
    async def health_check(self) -> Dict[str, bool]:
        """
        Check if all models are loaded and functional.
        
        Returns:
            Dict mapping model names to health status
        """
        logger.debug("Performing embedding service health check")
        
        health = {}
        
        for model_name in ['all-MiniLM-L6-v2', 'specter2']:
            try:
                # Try to encode a test string
                test_embedding = await self.embed_text(
                    "test",
                    model_name,
                    normalize=True
                )
                health[model_name] = len(test_embedding) > 0
            except Exception as e:
                logger.error(
                    "Model health check failed",
                    model=model_name,
                    error=str(e)
                )
                health[model_name] = False
        
        logger.info(
            "Health check complete",
            results=health
        )
        
        return health