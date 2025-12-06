"""
Embedding repository for vector similarity search operations.
Handles both paper and user embeddings with pgvector.
"""
from typing import List, Optional, Tuple
import asyncpg
import numpy as np
from app.db.repositories.base import BaseRepository
from app.db.connection import DatabaseConnection
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingRepository(BaseRepository):
    """Repository for embedding-related operations."""
    
    @property
    def table_name(self) -> str:
        # This repository handles multiple tables
        return "embeddings"
    
    def __init__(self, db: DatabaseConnection):
        super().__init__(db)
        logger.info("EmbeddingRepository initialized")
    
    async def save_paper_embedding(
        self,
        paper_id: str,
        embedding: np.ndarray,
        model_name: str,
        generation_method: str = "abstract"
    ) -> None:
        """
        Save paper embedding to appropriate table based on model.
        
        Args:
            paper_id: Paper identifier
            embedding: Embedding vector
            model_name: 'all-MiniLM-L6-v2' or 'specter2'
            generation_method: How embedding was generated
        """
        logger.debug(
            "Saving paper embedding",
            paper_id=paper_id,
            model=model_name,
            dim=len(embedding),
            method=generation_method
        )
        
        # Determine table based on model
        table = (
            "paper_embeddings_minilm" 
            if "minilm" in model_name.lower()
            else "paper_embeddings_specter"
        )
        
        query = f"""
            INSERT INTO {table} (
                paper_id, embedding, model_name, model_version, embedding_source
            )
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (paper_id) 
            DO UPDATE SET
                embedding = EXCLUDED.embedding,
                model_name = EXCLUDED.model_name,
                model_version = EXCLUDED.model_version,
                embedding_source = EXCLUDED.embedding_source,
                updated_at = NOW()
        """
        
        try:
            # Convert numpy array to list for PostgreSQL
            embedding_list = embedding.tolist()
            
            await self.db.execute(
                query,
                paper_id,
                embedding_list,
                model_name,
                'v1.0',  # model_version
                generation_method  # embedding_source (changed from generation_method)
            )
            
            logger.info(
                "Paper embedding saved",
                paper_id=paper_id,
                model=model_name,
                table=table
            )
        except Exception as e:
            logger.error(
                "Paper embedding save failed",
                paper_id=paper_id,
                model=model_name,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def get_paper_embedding(
        self,
        paper_id: str,
        model_name: str
    ) -> Optional[np.ndarray]:
        """
        Retrieve paper embedding.
        
        Args:
            paper_id: Paper identifier
            model_name: Model name
            
        Returns:
            Optional[ndarray]: Embedding vector or None
        """
        logger.debug(
            "Retrieving paper embedding",
            paper_id=paper_id,
            model=model_name
        )
        
        table = (
            "paper_embeddings_minilm"
            if "minilm" in model_name.lower()
            else "paper_embeddings_specter"
        )
        
        query = f"""
            SELECT embedding
            FROM {table}
            WHERE paper_id = $1
        """
        
        try:
            result = await self.db.fetchval(query, paper_id)
            
            if result:
                embedding = np.array(result, dtype=np.float32)
                logger.debug(
                    "Paper embedding retrieved",
                    paper_id=paper_id,
                    dim=len(embedding)
                )
                return embedding
            
            logger.debug(
                "Paper embedding not found",
                paper_id=paper_id,
                model=model_name
            )
            return None
            
        except Exception as e:
            logger.error(
                "Paper embedding retrieval failed",
                paper_id=paper_id,
                model=model_name,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def find_similar_papers(
        self,
        query_embedding: np.ndarray,
        model_name: str,
        limit: int = 10,
        excluded_paper_ids: Optional[List[str]] = None,
        domain_filter: Optional[str] = None,
        min_year: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Find papers similar to query embedding using cosine similarity.
        
        Args:
            query_embedding: Query vector
            model_name: Model name
            limit: Maximum results
            excluded_paper_ids: Papers to exclude
            domain_filter: Optional domain constraint
            min_year: Optional minimum year
            
        Returns:
            List of (paper_id, similarity_score) tuples
        """
        logger.debug(
            "Finding similar papers",
            model=model_name,
            limit=limit,
            excluded_count=len(excluded_paper_ids) if excluded_paper_ids else 0,
            domain=domain_filter,
            min_year=min_year
        )
        
        table = (
            "paper_embeddings_minilm"
            if "minilm" in model_name.lower()
            else "paper_embeddings_specter"
        )
        
        # Build query with filters
        query = f"""
            WITH filtered_papers AS (
                SELECT p.paper_id
                FROM papers p
                WHERE 1=1
        """
        
        params = []
        param_count = 1
        
        # Add filters
        if excluded_paper_ids:
            query += f" AND p.paper_id != ALL(${param_count}::text[])"
            params.append(excluded_paper_ids)
            param_count += 1
        
        if domain_filter:
            query += f" AND p.domain = ${param_count}"
            params.append(domain_filter)
            param_count += 1
        
        if min_year:
            query += f" AND p.year >= ${param_count}"
            params.append(min_year)
            param_count += 1
        
        query += f"""
            ),
            similarity_scores AS (
                SELECT 
                    fp.paper_id,
                    1 - (pe.embedding <=> ${param_count}::vector) as similarity
                FROM filtered_papers fp
                JOIN {table} pe ON fp.paper_id = pe.paper_id
            )
            SELECT 
                ss.paper_id,
                ss.similarity,
                p.citation_count,
                pqs.composite_score
            FROM similarity_scores ss
            JOIN papers p ON ss.paper_id = p.paper_id
            LEFT JOIN paper_quality_scores pqs ON p.paper_id = pqs.paper_id
            ORDER BY 
                (ss.similarity * 0.7 + COALESCE(pqs.composite_score, 0.5) * 0.3) DESC
            LIMIT ${param_count + 1}
        """
        
        params.extend([query_embedding.tolist(), limit])
        
        try:
            results = await self.db.fetch(query, *params)
            
            paper_scores = [
                (row['paper_id'], float(row['similarity']))
                for row in results
            ]
            
            logger.info(
                "Similar papers found",
                model=model_name,
                count=len(paper_scores),
                top_score=paper_scores[0][1] if paper_scores else None
            )
            
            return paper_scores
            
        except Exception as e:
            logger.error(
                "Similar papers search failed",
                model=model_name,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def save_user_embedding(
        self,
        user_id: int,
        embedding: np.ndarray,
        model_name: str,
        generation_method: str = "profile_based",
        based_on_papers: Optional[List[str]] = None
    ) -> None:
        """
        Save user embedding.
        
        Args:
            user_id: User identifier
            embedding: Embedding vector
            model_name: Model name
            generation_method: 'profile_based' or 'interaction_based'
            based_on_papers: Paper IDs used to generate embedding
        """
        logger.debug(
            "Saving user embedding",
            user_id=user_id,
            model=model_name,
            dim=len(embedding),
            method=generation_method,
            paper_count=len(based_on_papers) if based_on_papers else 0
        )
        
        table = (
            "user_embeddings_minilm"
            if "minilm" in model_name.lower()
            else "user_embeddings_specter"
        )
        
        query = f"""
            INSERT INTO {table} (
                user_id, embedding, model_version, 
                generation_method, based_on_papers
            )
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (user_id)
            DO UPDATE SET
                embedding = EXCLUDED.embedding,
                model_version = EXCLUDED.model_version,
                generation_method = EXCLUDED.generation_method,
                based_on_papers = EXCLUDED.based_on_papers,
                updated_at = NOW()
        """
        
        try:
            await self.db.execute(
                query,
                user_id,
                embedding.tolist(),
                model_name,
                generation_method,
                based_on_papers or []
            )
            
            logger.info(
                "User embedding saved",
                user_id=user_id,
                model=model_name,
                table=table
            )
        except Exception as e:
            logger.error(
                "User embedding save failed",
                user_id=user_id,
                model=model_name,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def get_user_embedding(
        self,
        user_id: int,
        model_name: str
    ) -> Optional[Tuple[np.ndarray, str, str]]:
        """
        Retrieve user embedding with metadata.
        
        Args:
            user_id: User identifier
            model_name: Model name
            
        Returns:
            Optional tuple of (embedding, generation_method, updated_at)
        """
        logger.debug(
            "Retrieving user embedding",
            user_id=user_id,
            model=model_name
        )
        
        table = (
            "user_embeddings_minilm"
            if "minilm" in model_name.lower()
            else "user_embeddings_specter"
        )
        
        query = f"""
            SELECT 
                embedding,
                generation_method,
                updated_at
            FROM {table}
            WHERE user_id = $1
        """
        
        try:
            result = await self.db.fetchrow(query, user_id)
            
            if result:
                embedding = np.array(result['embedding'], dtype=np.float32)
                logger.debug(
                    "User embedding retrieved",
                    user_id=user_id,
                    dim=len(embedding),
                    method=result['generation_method']
                )
                return (
                    embedding,
                    result['generation_method'],
                    str(result['updated_at'])
                )
            
            logger.debug(
                "User embedding not found",
                user_id=user_id,
                model=model_name
            )
            return None
            
        except Exception as e:
            logger.error(
                "User embedding retrieval failed",
                user_id=user_id,
                model=model_name,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def bulk_save_paper_embeddings(
        self,
        embeddings: List[Tuple[str, np.ndarray]],
        model_name: str,
        generation_method: str = "abstract"
    ) -> int:
        """
        Bulk save paper embeddings for efficiency.
        
        Args:
            embeddings: List of (paper_id, embedding) tuples
            model_name: Model name
            generation_method: How embeddings were generated
            
        Returns:
            int: Number of embeddings saved
        """
        if not embeddings:
            return 0
        
        logger.info(
            "Bulk saving paper embeddings",
            count=len(embeddings),
            model=model_name
        )
        
        table = (
            "paper_embeddings_minilm"
            if "minilm" in model_name.lower()
            else "paper_embeddings_specter"
        )
        
        # Prepare batch insert
        values = [
            (pid, emb.tolist(), model_name, 'v1.0', generation_method)
            for pid, emb in embeddings
        ]
        
        query = f"""
            INSERT INTO {table} (
                paper_id, embedding, model_name, model_version, embedding_source
            )
            SELECT * FROM unnest($1::text[], $2::vector[], $3::text[], $4::text[], $5::text[])
            ON CONFLICT (paper_id)
            DO UPDATE SET
                embedding = EXCLUDED.embedding,
                updated_at = NOW()
        """
        
        try:
            # Split into batches if too large
            batch_size = 100
            total_saved = 0
            
            for i in range(0, len(values), batch_size):
                batch = values[i:i + batch_size]
                
                paper_ids = [v[0] for v in batch]
                embeddings_list = [v[1] for v in batch]
                models = [v[2] for v in batch]
                versions = [v[3] for v in batch]
                sources = [v[4] for v in batch]
                
                await self.db.execute(
                    query,
                    paper_ids,
                    embeddings_list,
                    models,
                    versions,
                    sources
                )
                
                total_saved += len(batch)
                
                logger.debug(
                    "Batch saved",
                    batch=i // batch_size + 1,
                    count=len(batch)
                )
            
            logger.info(
                "Bulk save complete",
                total=total_saved,
                model=model_name
            )
            
            return total_saved
            
        except Exception as e:
            logger.error(
                "Bulk save failed",
                count=len(embeddings),
                model=model_name,
                error=str(e),
                exc_info=True
            )
            raise