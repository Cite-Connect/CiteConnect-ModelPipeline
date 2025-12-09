"""
Ground truth repository for managing evaluation papers and relationships.
Handles canonical papers, ground truth relationships, and evaluation data.
UPDATED: Fixed for Supabase schema with generated columns and array-based canonical papers.
"""
from typing import List, Optional, Dict, Any, Tuple
import asyncpg
from app.db.repositories.base import BaseRepository
from app.db.connection import DatabaseConnection
from app.utils.logger import get_logger

logger = get_logger(__name__)


class GroundTruthRepository(BaseRepository):
    """Repository for ground truth and canonical paper operations."""
    
    @property
    def table_name(self) -> str:
        return "ground_truth_papers"
    
    def __init__(self, db: DatabaseConnection):
        super().__init__(db)
        logger.info("GroundTruthRepository initialized")
    
    async def get_ground_truth_papers(
        self,
        domain: Optional[str] = None,
        min_quality_score: float = 0.5
    ) -> List[asyncpg.Record]:
        """
        Get high-quality papers suitable for ground truth evaluation.
        
        Args:
            domain: Optional domain filter
            min_quality_score: Minimum quality threshold
            
        Returns:
            List[Record]: Ground truth papers
        """
        logger.debug(
            "Getting ground truth papers",
            domain=domain,
            min_quality_score=min_quality_score
        )
        
        query = """
            SELECT gtp.*, p.title, p.abstract, p.domain
            FROM ground_truth_papers gtp
            JOIN papers p ON gtp.paper_id = p.paper_id
            WHERE gtp.quality_score >= $1
        """
        
        params = [min_quality_score]
        
        if domain:
            query += " AND p.domain = $2"
            params.append(domain)
        
        query += " ORDER BY gtp.quality_score DESC"
        
        try:
            results = await self.db.fetch(query, *params)
            logger.info(
                "Ground truth papers retrieved",
                count=len(results),
                domain=domain
            )
            return results
        except Exception as e:
            logger.error(
                "Ground truth papers retrieval failed",
                domain=domain,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def get_canonical_papers(
        self,
        domain: str,
        tier: str,
        user_stage: str,
        limit: int = 10
    ) -> List[asyncpg.Record]:
        """
        Get canonical papers for cold-start recommendations.
        Uses array-based structure from Supabase schema.
        
        Args:
            domain: Research domain ('healthcare', 'fintech', 'quantum_computing')
            tier: 'foundational', 'trending', or 'recent'
            user_stage: User's research stage (kept for API compatibility)
            limit: Number of papers to return
            
        Returns:
            List[Record]: Canonical papers with full details
        """
        logger.debug(
            "Getting canonical papers",
            domain=domain,
            tier=tier,
            user_stage=user_stage,
            limit=limit
        )
        
        # Get paper_ids array from domain_canonical_papers
        array_query = """
            SELECT paper_ids
            FROM domain_canonical_papers
            WHERE domain = $1
              AND recommendation_tier = $2
        """
        
        try:
            result = await self.db.fetchrow(array_query, domain, tier)
            
            if not result or not result['paper_ids']:
                logger.warning(
                    "No canonical papers found",
                    domain=domain,
                    tier=tier
                )
                return []
            
            # Get the array and limit it
            paper_ids = result['paper_ids'][:limit]
            
            # Fetch full paper details
            papers_query = """
                SELECT 
                    p.*
                FROM papers p
                WHERE p.paper_id = ANY($1::text[])
                ORDER BY p.citation_count DESC
            """
            
            papers = await self.db.fetch(papers_query, paper_ids)
            
            logger.info(
                "Canonical papers retrieved",
                domain=domain,
                tier=tier,
                count=len(papers)
            )
            
            return papers
            
        except Exception as e:
            logger.error(
                "Canonical papers retrieval failed",
                domain=domain,
                tier=tier,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def get_canonical_papers_sampled(
        self,
        domain: str,
        tier_distribution: Dict[str, int]
    ) -> List[Dict]:
        """
        Get canonical papers with sampling by tier.
        
        Args:
            domain: Research domain
            tier_distribution: Dict of {tier: count} to sample
            
        Returns:
            List of canonical papers with tier labels
        """
        canonical_papers = []
        
        for tier, tier_count in tier_distribution.items():
            # Get paper IDs for this tier
            query = """
                SELECT paper_ids
                FROM domain_canonical_papers
                WHERE domain = $1
                AND recommendation_tier = $2
            """
            
            result = await self.db.fetchrow(query, domain, tier)
            
            if result and result['paper_ids']:
                paper_ids = result['paper_ids']
                
                # Sample requested count
                import random
                sample_size = min(tier_count, len(paper_ids))
                sampled_ids = random.sample(paper_ids, sample_size)
                
                # Fetch full paper details
                papers_query = """
                    SELECT 
                        paper_id, title, abstract, authors, year,
                        citation_count, domain, sub_domains, venue
                    FROM papers
                    WHERE paper_id = ANY($1::text[])
                """
                
                papers = await self.db.fetch(papers_query, sampled_ids)
                
                # Add tier label
                for paper in papers:
                    paper_dict = dict(paper)
                    paper_dict['canonical_tier'] = tier
                    canonical_papers.append(paper_dict)
        
            return canonical_papers


    async def find_relevant_ground_truth_papers(
        self,
        interest_terms: List[str],
        domain: str,
        limit: int = 10
    ) -> List[str]:
        """
        Find ground truth papers matching user interests.
        
        Args:
            interest_terms: User's interest terms
            domain: User's domain
            limit: Max papers to return
            
        Returns:
            List of GT paper IDs
        """
        # Build ILIKE patterns
        conditions = ' OR '.join([
            f"p.title ILIKE '%{term}%' OR p.abstract ILIKE '%{term}%'"
            for term in interest_terms
        ])
        
        query = f"""
            SELECT p.paper_id
            FROM papers p
            INNER JOIN ground_truth_papers gtp ON p.paper_id = gtp.paper_id
            WHERE p.domain = $1
            AND ({conditions})
            LIMIT $2
        """
        
        results = await self.db.fetch(query, domain, limit)
        return [r['paper_id'] for r in results]

    async def get_ground_truth_relationships(
            self,
            paper_id: str
        ) -> Optional[asyncpg.Record]:
            """
            Get pre-computed relationships for a ground truth paper.
            
            Args:
                paper_id: Paper identifier
                
            Returns:
                Optional[Record]: Relationship data including citation network
            """
            logger.debug(
                "Getting ground truth relationships",
                paper_id=paper_id
            )
            
            query = """
                SELECT *
                FROM ground_truth_relationships
                WHERE paper_id = $1
            """
            
            try:
                result = await self.db.fetchrow(query, paper_id)
                
                if result:
                    logger.debug(
                        "Relationships found",
                        paper_id=paper_id,
                        citation_count=len(result.get('citation_network', []))
                    )
                else:
                    logger.debug(
                        "No relationships found",
                        paper_id=paper_id
                    )
                
                return result
                
            except Exception as e:
                logger.error(
                    "Relationships retrieval failed",
                    paper_id=paper_id,
                    error=str(e),
                    exc_info=True
                )
                raise
    
    async def create_ground_truth_paper(
        self,
        paper_id: str,
        num_references: int,
        reference_coverage: float,
        quality_score: float,
        is_canonical: bool = False,
        canonical_tier: Optional[str] = None
    ) -> asyncpg.Record:
        """
        Register a paper as ground truth.
        NOTE: reference_coverage is GENERATED by database - we don't insert it.
        Database auto-calculates as: references_in_corpus / reference_count
        
        Args:
            paper_id: Paper identifier
            num_references: Number of references (stored as reference_count)
            reference_coverage: Coverage ratio (used to calculate references_in_corpus, NOT inserted)
            quality_score: Composite quality score
            is_canonical: Whether paper is canonical for domain
            canonical_tier: Tier if canonical
            
        Returns:
            Record: Created ground truth paper with auto-generated reference_coverage
        """
        logger.info(
            "Creating ground truth paper",
            paper_id=paper_id,
            quality_score=quality_score,
            is_canonical=is_canonical,
            ref_count=num_references
        )
        
        # Calculate references_in_corpus from coverage
        # This is what database will use to generate reference_coverage
        references_in_corpus = int(num_references * reference_coverage)
        
        # Get paper domain - required field in schema
        domain_query = "SELECT domain FROM papers WHERE paper_id = $1"
        paper_domain = await self.db.fetchval(domain_query, paper_id)
        
        if not paper_domain:
            logger.error(
                "Paper not found or has no domain",
                paper_id=paper_id
            )
            raise ValueError(f"Paper {paper_id} not found or has no domain")
        
        # DON'T insert reference_coverage - it's GENERATED ALWAYS AS
        # Database will calculate it from: references_in_corpus / reference_count
        query = """
            INSERT INTO ground_truth_papers (
                paper_id, reference_count, references_in_corpus,
                quality_score, domain,
                is_canonical, canonical_tier
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (paper_id)
            DO UPDATE SET
                reference_count = EXCLUDED.reference_count,
                references_in_corpus = EXCLUDED.references_in_corpus,
                quality_score = EXCLUDED.quality_score,
                is_canonical = EXCLUDED.is_canonical,
                canonical_tier = EXCLUDED.canonical_tier
            RETURNING *
        """
        
        try:
            result = await self.db.fetchrow(
                query,
                paper_id,
                num_references,
                references_in_corpus,
                quality_score,
                paper_domain,
                is_canonical,
                canonical_tier
            )
            
            logger.info(
                "Ground truth paper created",
                paper_id=paper_id,
                domain=paper_domain,
                ref_coverage=result.get('reference_coverage')  # Read auto-generated value
            )
            return result
            
        except Exception as e:
            logger.error(
                "Ground truth paper creation failed",
                paper_id=paper_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def save_ground_truth_relationships(
        self,
        paper_id: str,
        citation_network: List[str],
        co_cited_papers: List[str],
        bibliographic_couples: List[str],
        network_centrality: float
    ) -> asyncpg.Record:
        """
        Save pre-computed relationships for ground truth paper.
        NOTE: citation_network_size is GENERATED by database.
        
        Args:
            paper_id: Paper identifier
            citation_network: Direct citations
            co_cited_papers: Frequently co-cited papers
            bibliographic_couples: Papers citing same sources
            network_centrality: PageRank-style score
            
        Returns:
            Record: Created relationship record
        """
        logger.debug(
            "Saving ground truth relationships",
            paper_id=paper_id,
            citation_count=len(citation_network),
            co_cited_count=len(co_cited_papers)
        )
        
        # Don't insert citation_network_size - it's GENERATED
        query = """
            INSERT INTO ground_truth_relationships (
                paper_id, citation_network, co_cited_papers,
                bibliographic_couples, network_centrality
            )
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (paper_id)
            DO UPDATE SET
                citation_network = EXCLUDED.citation_network,
                co_cited_papers = EXCLUDED.co_cited_papers,
                bibliographic_couples = EXCLUDED.bibliographic_couples,
                network_centrality = EXCLUDED.network_centrality,
                last_updated = NOW()
            RETURNING *
        """
        
        try:
            result = await self.db.fetchrow(
                query,
                paper_id,
                citation_network,
                co_cited_papers,
                bibliographic_couples,
                network_centrality
            )
            
            logger.info(
                "Relationships saved",
                paper_id=paper_id,
                network_size=result.get('citation_network_size')  # Read generated value
            )
            return result
            
        except Exception as e:
            logger.error(
                "Relationships save failed",
                paper_id=paper_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def evaluate_against_ground_truth(
        self,
        recommended_paper_ids: List[str],
        ground_truth_paper_id: str
    ) -> Dict[str, Any]:
        """
        Evaluate recommendations against ground truth citations.
        
        Args:
            recommended_paper_ids: Papers that were recommended
            ground_truth_paper_id: Ground truth paper to compare against
            
        Returns:
            Dict with evaluation metrics
        """
        logger.debug(
            "Evaluating against ground truth",
            ground_truth_id=ground_truth_paper_id,
            rec_count=len(recommended_paper_ids)
        )
        
        # Get ground truth relationships
        relationships = await self.get_ground_truth_relationships(
            ground_truth_paper_id
        )
        
        if not relationships:
            logger.warning(
                "No ground truth relationships found",
                paper_id=ground_truth_paper_id
            )
            return {
                "ground_truth_quality": 0.0,
                "hits": 0,
                "total_relevant": 0
            }
        
        # Extract relevant papers (citations + co-cited)
        relevant_papers = set(relationships.get('citation_network', []))
        relevant_papers.update(relationships.get('co_cited_papers', []))
        
        # Calculate hits
        recommended_set = set(recommended_paper_ids)
        hits = len(recommended_set & relevant_papers)
        
        # Calculate quality score
        quality = hits / len(recommended_paper_ids) if recommended_paper_ids else 0.0
        
        result = {
            "ground_truth_quality": quality,
            "hits": hits,
            "total_relevant": len(relevant_papers),
            "total_recommended": len(recommended_paper_ids)
        }
        
        logger.info(
            "Ground truth evaluation complete",
            quality=quality,
            hits=hits
        )
        
        return result
    
    async def get_canonical_papers_by_domain(
        self,
        domain: str
    ) -> Dict[str, List[str]]:
        """
        Get all canonical paper IDs grouped by tier for a domain.
        Uses array-based structure from Supabase schema.
        
        Args:
            domain: Research domain
            
        Returns:
            Dict mapping tier to list of paper IDs
        """
        logger.debug("Getting canonical papers by tier", domain=domain)
        
        query = """
            SELECT recommendation_tier, paper_ids
            FROM domain_canonical_papers
            WHERE domain = $1
            ORDER BY recommendation_tier
        """
        
        try:
            results = await self.db.fetch(query, domain)
            
            # Build dict from results (already grouped in table)
            by_tier = {
                row['recommendation_tier']: row['paper_ids']
                for row in results
            }
            
            logger.info(
                "Canonical papers grouped",
                domain=domain,
                tiers=list(by_tier.keys()),
                total_papers=sum(len(papers) for papers in by_tier.values())
            )
            
            return by_tier
            
        except Exception as e:
            logger.error(
                "Canonical papers grouping failed",
                domain=domain,
                error=str(e),
                exc_info=True
            )
            raise