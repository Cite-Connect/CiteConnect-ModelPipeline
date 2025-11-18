"""
Embedding Generation Service

Handles generation and management of embeddings for:
- User profiles (interests → embedding)
- Search queries
- Individual papers

NOTE: Currently using all-MiniLM-L6-v2 (384-dim) to match DataPipeline.
TODO: Migrate to allenai/specter2_base (768-dim) in Week 7-8.
"""

import numpy as np
from typing import List
import logging
from app.utils.embedding import get_embedder
from app.db.postgres import execute_query

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating and managing embeddings"""
    
    def __init__(self):
        """Initialize embedding service with SPECTER2 model"""
        self.embedder = get_embedder()
        logger.info("Embedding service initialized")
    
    async def generate_user_profile_embedding(self, user_id: int) -> np.ndarray:
        """
        Generate embedding vector for user's research profile
        
        Process:
        1. Fetch user's interests and domain from PostgreSQL
        2. Create weighted keyword representation
        3. Generate embedding using same model as DataPipeline
        4. Store in user_profile_embeddings table
        
        Args:
            user_id: User ID
        
        Returns:
            384-dimensional user profile embedding (matches paper embeddings)
        """
        logger.info(f"Generating profile embedding for user {user_id}")
        
        # Fetch user interests
        interests = await execute_query(
            """
            SELECT interest_keyword, weight 
            FROM user_interests 
            WHERE user_id = $1
            ORDER BY weight DESC
            """,
            user_id,
            fetch_all=True
        )
        
        if not interests:
            raise ValueError(f"No interests found for user {user_id}")
        
        # Fetch user domain
        domain_row = await execute_query(
            "SELECT domain FROM user_domains WHERE user_id = $1",
            user_id,
            fetch_one=True
        )
        
        if not domain_row:
            raise ValueError(f"No domain found for user {user_id}")
        
        domain = domain_row['domain']
        
        # Extract keywords and weights
        keywords = [i['interest_keyword'] for i in interests]
        weights = [float(i['weight']) for i in interests]
        
        logger.info(f"  Domain: {domain}")
        logger.info(f"  Interests: {keywords}")
        
        # Generate embedding
        embedding = self.embedder.embed_user_interests(
            interests=keywords,
            domain=domain,
            weights=weights
        )
        
        # Store in database
        await self._store_user_embedding(user_id, embedding)
        
        logger.info(f"✓ Generated and stored embedding for user {user_id}")
        
        return embedding
    
    async def get_user_profile_embedding(self, user_id: int) -> np.ndarray:
        """
        Get user's profile embedding from database
        
        Generates new embedding if not exists or outdated
        
        Args:
            user_id: User ID
        
        Returns:
            User profile embedding vector
        """
        # Check if embedding exists
        result = await execute_query(
            """
            SELECT embedding_vector, last_updated 
            FROM user_profile_embeddings 
            WHERE user_id = $1
            """,
            user_id,
            fetch_one=True
        )
        
        if result:
            # Embedding exists - check if needs refresh
            # For now, use existing (TODO: add staleness check)
            logger.info(f"Using existing embedding for user {user_id}")
            return np.array(result['embedding_vector'])
        
        # Generate new embedding
        logger.info(f"No embedding found, generating for user {user_id}")
        return await self.generate_user_profile_embedding(user_id)
    
    async def _store_user_embedding(
        self,
        user_id: int,
        embedding: np.ndarray
    ) -> None:
        """
        Store user profile embedding in PostgreSQL
        
        Args:
            user_id: User ID
            embedding: Embedding vector to store
        """
        await execute_query(
            """
            INSERT INTO user_profile_embeddings 
                (user_id, embedding_vector, last_updated, interaction_count)
            VALUES ($1, $2, CURRENT_TIMESTAMP, 0)
            ON CONFLICT (user_id) 
            DO UPDATE SET 
                embedding_vector = $2,
                last_updated = CURRENT_TIMESTAMP
            """,
            user_id,
            embedding.tolist()  # Convert to list for PostgreSQL array
        )
    
    async def refresh_user_embedding(self, user_id: int) -> np.ndarray:
        """
        Force refresh of user embedding
        
        Call this after user updates interests
        
        Args:
            user_id: User ID
        
        Returns:
            New embedding vector
        """
        logger.info(f"Refreshing embedding for user {user_id}")
        return await self.generate_user_profile_embedding(user_id)
    
    def embed_query(self, query_text: str) -> np.ndarray:
        """
        Generate embedding for search query
        
        Args:
            query_text: User's search query
        
        Returns:
            Query embedding
        """
        # Format as SPECTER2 expects
        formatted = f"[TITLE] {query_text} [ABSTRACT] {query_text}"
        return self.embedder.embed_text(formatted)


# Create singleton instance
embedding_service = EmbeddingService()