"""
Paper repository for managing academic paper data.
Handles paper metadata, citations, and search operations.
"""
from typing import List, Optional, Dict, Any
import asyncpg
from app.db.repositories.base import BaseRepository
from app.db.connection import DatabaseConnection
from app.utils.logger import get_logger
import numpy as np

logger = get_logger(__name__)


class PaperRepository(BaseRepository):
    """Repository for paper-related database operations."""
    
    @property
    def table_name(self) -> str:
        return "papers"
    
    def __init__(self, db: DatabaseConnection):
        super().__init__(db)
        logger.info("PaperRepository initialized")
    
    async def find_by_paper_id(self, paper_id: str) -> Optional[asyncpg.Record]:
        """
        Find paper by paper_id (SHA hash).
        
        Args:
            paper_id: Paper identifier (SHA-1 hash)
            
        Returns:
            Optional[Record]: Paper record or None
        """
        logger.debug("Finding paper by paper_id", paper_id=paper_id)
        
        query = """
            SELECT *
            FROM papers
            WHERE paper_id = $1
        """
        
        try:
            result = await self.db.fetchrow(query, paper_id)
            logger.debug(
                "Paper lookup complete",
                paper_id=paper_id,
                found=result is not None
            )
            return result
        except Exception as e:
            logger.error(
                "Paper lookup failed",
                paper_id=paper_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def find_by_domain(
        self,
        domain: str,
        limit: int = 100,
        min_year: Optional[int] = None
    ) -> List[asyncpg.Record]:
        """
        Find papers by domain with optional year filter.
        
        Args:
            domain: Research domain
            limit: Maximum number of papers
            min_year: Minimum publication year
            
        Returns:
            List[Record]: Matching papers
        """
        logger.debug(
            "Finding papers by domain",
            domain=domain,
            limit=limit,
            min_year=min_year
        )
        
        query = """
            SELECT *
            FROM papers
            WHERE domain = $1
        """
        
        params = [domain]
        
        if min_year:
            query += " AND year >= $2"
            params.append(min_year)
        
        query += f" ORDER BY citation_count DESC LIMIT ${len(params) + 1}"
        params.append(limit)
        
        try:
            results = await self.db.fetch(query, *params)
            logger.info(
                "Domain papers retrieved",
                domain=domain,
                count=len(results),
                min_year=min_year
            )
            return results
        except Exception as e:
            logger.error(
                "Domain paper lookup failed",
                domain=domain,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def find_by_ids(
        self,
        paper_ids: List[str]
    ) -> List[asyncpg.Record]:
        """
        Find multiple papers by IDs.
        
        Args:
            paper_ids: List of paper identifiers
            
        Returns:
            List[Record]: Found papers
        """
        if not paper_ids:
            logger.debug("No paper IDs provided")
            return []
        
        logger.debug(
            "Finding papers by IDs",
            count=len(paper_ids)
        )
        
        query = """
            SELECT *
            FROM papers
            WHERE paper_id = ANY($1::text[])
        """
        
        try:
            results = await self.db.fetch(query, paper_ids)
            logger.debug(
                "Batch paper lookup complete",
                requested=len(paper_ids),
                found=len(results)
            )
            return results
        except Exception as e:
            logger.error(
                "Batch paper lookup failed",
                count=len(paper_ids),
                error=str(e),
                exc_info=True
            )
            raise
    
    async def search_by_text(
        self,
        search_text: str,
        limit: int = 20
    ) -> List[asyncpg.Record]:
        """
        Full-text search on paper titles and abstracts.
        
        Args:
            search_text: Search query
            limit: Maximum results
            
        Returns:
            List[Record]: Matching papers ranked by relevance
        """
        logger.debug(
            "Performing text search",
            query=search_text,
            limit=limit
        )
        
        query = """
            SELECT 
                *,
                ts_rank(
                    to_tsvector('english', title || ' ' || COALESCE(abstract, '')),
                    plainto_tsquery('english', $1)
                ) as relevance
            FROM papers
            WHERE 
                to_tsvector('english', title || ' ' || COALESCE(abstract, '')) 
                @@ plainto_tsquery('english', $1)
            ORDER BY relevance DESC, citation_count DESC
            LIMIT $2
        """
        
        try:
            results = await self.db.fetch(query, search_text, limit)
            logger.info(
                "Text search complete",
                query=search_text,
                results=len(results)
            )
            return results
        except Exception as e:
            logger.error(
                "Text search failed",
                query=search_text,
                error=str(e),
                exc_info=True
            )
            raise
    async def semantic_search(
        self,
        embedding: np.ndarray,
        model: str = 'minilm',
        domain: Optional[str] = None,
        limit: int = 50,
        min_similarity: float = 0.3
    ) -> List[Dict]:
        """
        Semantic search using vector similarity.
        
        Args:
            embedding: Query embedding vector
            model: Model name ('minilm' or 'specter')
            domain: Optional domain filter
            limit: Maximum results
            min_similarity: Minimum similarity threshold
            
        Returns:
            List[Dict]: Papers ranked by semantic similarity
        """
        logger.debug(
            "Performing semantic search",
            model=model,
            domain=domain,
            limit=limit
        )
        
        # Determine embedding table
        embedding_table = f'paper_embeddings_{model}'
        
        # Convert embedding to PostgreSQL vector format
        embedding_str = '[' + ','.join(map(str, embedding.tolist())) + ']'
        
        # Build query
        query = f"""
            SELECT 
                p.paper_id,
                p.title,
                p.abstract,
                p.authors,
                p.year,
                p.citation_count,
                p.domain,
                p.sub_domains,
                p.venue,
                p.reference_ids,
                1 - (pe.embedding <=> $1::vector) as similarity
            FROM papers p
            JOIN {embedding_table} pe ON p.paper_id = pe.paper_id
            WHERE 1 - (pe.embedding <=> $1::vector) >= $2
        """
        
        params = [embedding_str, min_similarity]
        
        if domain:
            query += f" AND p.domain = ${len(params) + 1}"
            params.append(domain)
        
        query += f" ORDER BY pe.embedding <=> $1::vector LIMIT ${len(params) + 1}"
        params.append(limit)
        
        try:
            results = await self.db.fetch(query, *params)
            papers = [dict(row) for row in results]
            
            logger.info(
                "Semantic search complete",
                model=model,
                domain=domain,
                results=len(papers),
                avg_similarity=np.mean([p['similarity'] for p in papers]) if papers else 0
            )
            
            return papers
            
        except Exception as e:
            logger.error(
                "Semantic search failed",
                model=model,
                error=str(e),
                exc_info=True
            )
            raise

    async def get_paper_citations(
        self,
        paper_id: str
    ) -> List[str]:
        """
        Get list of papers that cite this paper.
        
        Args:
            paper_id: Paper identifier
            
        Returns:
            List[str]: Paper IDs of citing papers
        """
        logger.debug("Getting paper citations", paper_id=paper_id)
        
        query = """
            SELECT paper_id
            FROM papers
            WHERE $1 = ANY(reference_ids)
        """
        
        try:
            results = await self.db.fetch(query, paper_id)
            citations = [row['paper_id'] for row in results]
            logger.debug(
                "Citations retrieved",
                paper_id=paper_id,
                count=len(citations)
            )
            return citations
        except Exception as e:
            logger.error(
                "Citations retrieval failed",
                paper_id=paper_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def get_paper_references(
        self,
        paper_id: str
    ) -> List[str]:
        """
        Get list of papers referenced by this paper.
        
        Args:
            paper_id: Paper identifier
            
        Returns:
            List[str]: Paper IDs in bibliography
        """
        logger.debug("Getting paper references", paper_id=paper_id)
        
        query = """
            SELECT reference_ids
            FROM papers
            WHERE paper_id = $1
        """
        
        try:
            result = await self.db.fetchval(query, paper_id)
            references = result or []
            logger.debug(
                "References retrieved",
                paper_id=paper_id,
                count=len(references)
            )
            return references
        except Exception as e:
            logger.error(
                "References retrieval failed",
                paper_id=paper_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def get_trending_papers(
        self,
        domain: Optional[str] = None,
        days: int = 30,
        limit: int = 20
    ) -> List[Dict]:
        """
        Get trending papers based on recent years and high citations.
        
        Args:
            domain: Optional domain filter
            days: Not used (kept for compatibility)
            limit: Maximum results
            
        Returns:
            List[Dict]: Trending papers
        """
        logger.debug(
            "Getting trending papers",
            domain=domain,
            limit=limit
        )
        
        try:
            query = """
                SELECT 
                    paper_id,
                    title,
                    abstract,
                    authors,
                    year,
                    citation_count,
                    domain,
                    sub_domains,
                    venue,
                    reference_ids
                FROM papers
                WHERE year >= EXTRACT(YEAR FROM CURRENT_DATE) - 2
                  AND citation_count >= 10
            """
            
            params = []
            
            if domain:
                query += " AND domain = $1"
                params.append(domain)
            
            query += f" ORDER BY citation_count DESC, year DESC LIMIT ${len(params) + 1}"
            params.append(limit)
            
            logger.info(
                "Fetching trending papers",
                domain=domain,
                limit=limit
            )
            
            results = await self.db.fetch(query, *params)
            
            papers = [dict(row) for row in results]
            
            logger.info(
                "Trending papers retrieved",
                domain=domain,
                count=len(papers)
            )
            
            return papers
            
        except Exception as e:
            logger.error(
                "Trending papers retrieval failed",
                domain=domain,
                error=str(e),
                exc_info=True
            )
            raise
    async def _get_paper_embeddings(
        self,
        paper_ids: List[str],
        model: str
    ) -> Dict[str, np.ndarray]:
        """
        Batch fetch embeddings for multiple papers.
        
        Args:
            paper_ids: List of paper IDs
            model: Model name ('minilm' or 'specter')
            
        Returns:
            Dict mapping paper_id to embedding vector
        """
        if not paper_ids:
            return {}
        
        # Map model to table
        table_map = {
            'minilm': 'paper_embeddings_minilm',
            'specter': 'paper_embeddings_specter',
            'all-MiniLM-L6-v2': 'paper_embeddings_minilm',
            'specter2': 'paper_embeddings_specter'
        }
        
        table = table_map.get(model, 'paper_embeddings_minilm')
        
        query = f"""
            SELECT paper_id, embedding
            FROM {table}
            WHERE paper_id = ANY($1::text[])
        """
        
        results = await self.db.fetch(query, paper_ids)
        
        # Convert to dict
        embeddings = {}
        for row in results:
            # Convert PostgreSQL vector to numpy array
            embedding_list = row['embedding']
            embeddings[row['paper_id']] = np.array(embedding_list)
        
        logger.debug(
            "Paper embeddings fetched",
            model=model,
            requested=len(paper_ids),
            found=len(embeddings)
        )
        
        return embeddings
    async def update_quality_score(
        self,
        paper_id: str,
        quality_score: float
    ) -> None:
        """
        Update paper quality score.
        Note: This method is a placeholder - quality scores are not currently stored.
        
        Args:
            paper_id: Paper identifier
            quality_score: Composite quality score
        """
        logger.warning(
            "Quality score update requested but not implemented",
            paper_id=paper_id,
            score=quality_score,
            reason="paper_quality_scores table does not exist"
        )
        # No-op - table doesn't exist
        pass
