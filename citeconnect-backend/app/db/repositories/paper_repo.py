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
            SELECT 
                p.*,
                pqs.composite_score as quality_score
            FROM papers p
            LEFT JOIN paper_quality_scores pqs ON p.paper_id = pqs.paper_id
            WHERE p.paper_id = $1
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
            SELECT 
                p.*,
                pqs.composite_score as quality_score
            FROM papers p
            LEFT JOIN paper_quality_scores pqs ON p.paper_id = pqs.paper_id
            WHERE p.domain = $1
        """
        
        params = [domain]
        
        if min_year:
            query += " AND p.year >= $2"
            params.append(min_year)
            query += f" ORDER BY p.citation_count DESC LIMIT ${len(params) + 1}"
        else:
            query += " ORDER BY p.citation_count DESC LIMIT $2"
        
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
            SELECT 
                p.*,
                pqs.composite_score as quality_score
            FROM papers p
            LEFT JOIN paper_quality_scores pqs ON p.paper_id = pqs.paper_id
            WHERE p.paper_id = ANY($1::text[])
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
                p.*,
                pqs.composite_score as quality_score,
                ts_rank(
                    to_tsvector('english', p.title || ' ' || COALESCE(p.abstract, '')),
                    plainto_tsquery('english', $1)
                ) as relevance
            FROM papers p
            LEFT JOIN paper_quality_scores pqs ON p.paper_id = pqs.paper_id
            WHERE 
                to_tsvector('english', p.title || ' ' || COALESCE(p.abstract, '')) 
                @@ plainto_tsquery('english', $1)
            ORDER BY relevance DESC, p.citation_count DESC
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
            SELECT citation_ids
            FROM papers
            WHERE paper_id = $1
        """
        
        try:
            result = await self.db.fetchval(query, paper_id)
            citations = result or []
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
    ) -> List[asyncpg.Record]:
        """
        Get trending papers (recent with high citations).
        
        Args:
            domain: Optional domain filter
            days: Consider papers from last N days
            limit: Maximum results
            
        Returns:
            List[Record]: Trending papers
        """
        logger.debug(
            "Getting trending papers",
            domain=domain,
            days=days,
            limit=limit
        )
        
        query = """
            SELECT 
                p.*,
                pqs.composite_score as quality_score,
                (p.citation_count::float / GREATEST(
                    EXTRACT(DAY FROM NOW() - p.published_date), 1
                )) as trend_score
            FROM papers p
            LEFT JOIN paper_quality_scores pqs ON p.paper_id = pqs.paper_id
            WHERE p.published_date >= NOW() - INTERVAL '1 day' * $1
        """
        
        params = [days]
        
        if domain:
            query += " AND p.domain = $2"
            params.append(domain)
            query += f" ORDER BY trend_score DESC LIMIT ${len(params) + 1}"
        else:
            query += " ORDER BY trend_score DESC LIMIT $2"
        
        params.append(limit)
        
        try:
            results = await self.db.fetch(query, *params)
            logger.info(
                "Trending papers retrieved",
                domain=domain,
                count=len(results)
            )
            return results
        except Exception as e:
            logger.error(
                "Trending papers retrieval failed",
                domain=domain,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def update_quality_score(
        self,
        paper_id: str,
        quality_score: float
    ) -> None:
        """
        Update paper quality score.
        
        Args:
            paper_id: Paper identifier
            quality_score: Composite quality score
        """
        logger.debug(
            "Updating quality score",
            paper_id=paper_id,
            score=quality_score
        )
        
        query = """
            INSERT INTO paper_quality_scores (paper_id, composite_score)
            VALUES ($1, $2)
            ON CONFLICT (paper_id) 
            DO UPDATE SET 
                composite_score = EXCLUDED.composite_score,
                updated_at = NOW()
        """
        
        try:
            await self.db.execute(query, paper_id, quality_score)
            logger.info(
                "Quality score updated",
                paper_id=paper_id,
                score=quality_score
            )
        except Exception as e:
            logger.error(
                "Quality score update failed",
                paper_id=paper_id,
                error=str(e),
                exc_info=True
            )
            raise