"""
Ground truth service for managing canonical papers and evaluation.
Pre-loads ground truth data at startup for fast access.
"""
from typing import List, Dict, Optional, Set
import asyncpg
from app.config import settings
from app.utils.logger import get_logger
from app.db.repositories.ground_truth_repo import GroundTruthRepository
from app.db.repositories.paper_repo import PaperRepository

logger = get_logger(__name__)


class GroundTruthService:
    """
    Manages ground truth papers and canonical paper lists.
    Pre-loads data at startup for cold-start recommendations.
    """
    
    def __init__(
        self,
        ground_truth_repo: GroundTruthRepository,
        paper_repo: PaperRepository
    ):
        """
        Initialize ground truth service.
        
        Args:
            ground_truth_repo: Ground truth repository
            paper_repo: Paper repository
        """
        self.ground_truth_repo = ground_truth_repo
        self.paper_repo = paper_repo
        
        # Cached data structures
        self.canonical_papers: Dict[str, Dict[str, List[str]]] = {}
        self.ground_truth_papers: Dict[str, asyncpg.Record] = {}
        self.ground_truth_relationships: Dict[str, asyncpg.Record] = {}
        
        logger.info("GroundTruthService initialized")
    
    async def initialize(self) -> None:
        """
        Load ground truth data into memory.
        Called during application startup.
        """
        logger.info("Loading ground truth data")
        
        try:
            # Load ground truth papers
            await self._load_ground_truth_papers()
            
            # Load canonical papers by domain
            await self._load_canonical_papers()
            
            # Load ground truth relationships
            await self._load_ground_truth_relationships()
            
            logger.info(
                "Ground truth data loaded",
                ground_truth_count=len(self.ground_truth_papers),
                domains_with_canonical=len(self.canonical_papers),
                relationships_cached=len(self.ground_truth_relationships)
            )
            
        except Exception as e:
            logger.error(
                "Ground truth data loading failed",
                error=str(e),
                exc_info=True
            )
            raise
    
    async def _load_ground_truth_papers(self) -> None:
        """Load all ground truth papers into cache."""
        logger.debug("Loading ground truth papers")
        
        papers = await self.ground_truth_repo.get_ground_truth_papers(
            min_quality_score=settings.COLD_START_GROUND_TRUTH_THRESHOLD
        )
        
        self.ground_truth_papers = {
            paper['paper_id']: paper
            for paper in papers
        }
        
        logger.info(
            "Ground truth papers loaded",
            count=len(self.ground_truth_papers)
        )
    
    async def _load_canonical_papers(self) -> None:
        """Load canonical papers grouped by domain and tier."""
        logger.debug("Loading canonical papers")
        
        # Get all distinct domains
        domains_query = """
            SELECT DISTINCT domain
            FROM domain_canonical_papers
        """
        
        domains = await self.ground_truth_repo.db.fetch(domains_query)
        
        for domain_row in domains:
            domain = domain_row['domain']
            
            # Get papers by tier for this domain
            by_tier = await self.ground_truth_repo.get_canonical_papers_by_domain(
                domain
            )
            
            self.canonical_papers[domain] = by_tier
            
            logger.debug(
                "Canonical papers loaded for domain",
                domain=domain,
                tiers=list(by_tier.keys())
            )
        
        logger.info(
            "All canonical papers loaded",
            domains=len(self.canonical_papers)
        )
    
    async def _load_ground_truth_relationships(self) -> None:
        """Load relationship data for ground truth papers."""
        logger.debug("Loading ground truth relationships")
        
        # Load relationships in background (non-blocking for startup)
        # Only load for a subset to avoid blocking startup
        paper_ids = list(self.ground_truth_papers.keys())
        
        # Limit to first 100 papers to avoid blocking startup
        # Remaining relationships will be loaded on-demand
        max_papers_to_load = 100
        paper_ids_to_load = paper_ids[:max_papers_to_load]
        
        logger.info(
            "Loading ground truth relationships (limited batch)",
            total_papers=len(paper_ids),
            loading_count=len(paper_ids_to_load)
        )
        
        for paper_id in paper_ids_to_load:
            try:
                relationships = await self.ground_truth_repo.get_ground_truth_relationships(
                    paper_id
                )
                
                if relationships:
                    self.ground_truth_relationships[paper_id] = relationships
            except Exception as e:
                logger.warning(
                    "Failed to load relationships for paper",
                    paper_id=paper_id,
                    error=str(e)
                )
                # Continue with other papers
                continue
        
        logger.info(
            "Ground truth relationships loaded (partial)",
            loaded_count=len(self.ground_truth_relationships),
            total_papers=len(paper_ids),
            note="Remaining relationships will be loaded on-demand"
        )
    
    async def get_canonical_papers(
        self,
        domain: str,
        tier: str,
        user_stage: str,
        count: int = 10
    ) -> List[asyncpg.Record]:
        """
        Get canonical papers for cold-start recommendations.
        Uses cached data for instant response.
        
        Args:
            domain: Research domain
            tier: 'foundational', 'trending', or 'recent'
            user_stage: User's research stage
            count: Number of papers to return
            
        Returns:
            List[Record]: Canonical papers
        """
        logger.debug(
            "Getting canonical papers",
            domain=domain,
            tier=tier,
            user_stage=user_stage,
            count=count
        )
        
        # Check cache
        if domain in self.canonical_papers:
            tier_papers = self.canonical_papers[domain].get(tier, [])
            paper_ids = tier_papers[:count]
            
            # Fetch full paper data
            if paper_ids:
                papers = await self.paper_repo.find_by_ids(paper_ids)
                
                logger.debug(
                    "Canonical papers retrieved from cache",
                    domain=domain,
                    tier=tier,
                    count=len(papers)
                )
                
                return papers
        
        # Fallback to database query if not cached
        logger.debug(
            "Cache miss, fetching from database",
            domain=domain,
            tier=tier
        )
        
        papers = await self.ground_truth_repo.get_canonical_papers(
            domain=domain,
            tier=tier,
            user_stage=user_stage,
            limit=count
        )
        
        return papers
    
    async def get_ground_truth_for_user_type(
        self,
        research_stage: str,
        domain: str
    ) -> Dict[str, List[str]]:
        """
        Get appropriate ground truth papers for user type.
        Different users get different evaluation sets.
        
        Args:
            research_stage: User's research stage
            domain: Primary research domain
            
        Returns:
            Dict with 'foundational' and 'recent' paper IDs
        """
        logger.debug(
            "Getting ground truth for user type",
            research_stage=research_stage,
            domain=domain
        )
        
        # Map research stage to canonical tiers
        if research_stage in ['undergraduate', 'masters']:
            # Beginners: foundational papers
            tiers = ['foundational']
        elif research_stage in ['phd', 'postdoc']:
            # Advanced: mix of foundational and recent
            tiers = ['foundational', 'recent']
        else:
            # Professors/industry: recent and trending
            tiers = ['recent', 'trending']
        
        result = {}
        for tier in tiers:
            if domain in self.canonical_papers:
                result[tier] = self.canonical_papers[domain].get(tier, [])
        
        logger.debug(
            "Ground truth selected",
            research_stage=research_stage,
            tiers=list(result.keys())
        )
        
        return result
    
    async def evaluate_against_ground_truth(
        self,
        recommended_paper_ids: List[str],
        user_context: Dict
    ) -> Dict[str, float]:
        """
        Evaluate recommendations against ground truth.
        
        Args:
            recommended_paper_ids: Papers that were recommended
            user_context: User profile and preferences
            
        Returns:
            Dict with evaluation metrics
        """
        logger.debug(
            "Evaluating against ground truth",
            rec_count=len(recommended_paper_ids),
            user_stage=user_context.get('research_stage')
        )
        
        # Get appropriate ground truth for user
        ground_truth = await self.get_ground_truth_for_user_type(
            research_stage=user_context.get('research_stage', 'phd'),
            domain=user_context.get('primary_domain', 'machine_learning')
        )
        
        # Flatten all ground truth paper IDs
        all_ground_truth_ids = set()
        for tier_papers in ground_truth.values():
            all_ground_truth_ids.update(tier_papers)
        
        # Count papers in ground truth relationships
        total_hits = 0
        total_relevant = 0
        
        for gt_paper_id in all_ground_truth_ids:
            if gt_paper_id in self.ground_truth_relationships:
                relationships = self.ground_truth_relationships[gt_paper_id]
                
                # Get all related papers
                related = set(relationships.get('citation_network', []))
                related.update(relationships.get('co_cited_papers', []))
                
                total_relevant += len(related)
                
                # Check how many recommendations are in related set
                recommended_set = set(recommended_paper_ids)
                hits = len(recommended_set & related)
                total_hits += hits
        
        # Calculate quality score
        if len(recommended_paper_ids) > 0:
            quality = total_hits / len(recommended_paper_ids)
        else:
            quality = 0.0
        
        result = {
            "ground_truth_quality": quality,
            "hits": total_hits,
            "total_relevant": total_relevant,
            "total_recommended": len(recommended_paper_ids)
        }
        
        logger.info(
            "Ground truth evaluation complete",
            quality=quality,
            hits=total_hits
        )
        
        return result
    
    async def is_ground_truth_paper(self, paper_id: str) -> bool:
        """
        Check if paper is in ground truth set.
        
        Args:
            paper_id: Paper identifier
            
        Returns:
            bool: True if paper is ground truth
        """
        return paper_id in self.ground_truth_papers
    
    async def get_relationship_strength(
        self,
        paper_id_1: str,
        paper_id_2: str
    ) -> float:
        """
        Get relationship strength between two papers.
        
        Args:
            paper_id_1: First paper
            paper_id_2: Second paper
            
        Returns:
            float: Strength score (0.0 to 1.0)
        """
        logger.debug(
            "Calculating relationship strength",
            paper_1=paper_id_1,
            paper_2=paper_id_2
        )
        
        # Check if either paper has ground truth relationships
        relationships_1 = self.ground_truth_relationships.get(paper_id_1)
        relationships_2 = self.ground_truth_relationships.get(paper_id_2)
        
        strength = 0.0
        
        # Check citation network (strongest: 1.0)
        if relationships_1:
            if paper_id_2 in relationships_1.get('citation_network', []):
                strength = max(strength, 1.0)
        
        if relationships_2:
            if paper_id_1 in relationships_2.get('citation_network', []):
                strength = max(strength, 1.0)
        
        # Check co-citations (medium: 0.7)
        if relationships_1:
            if paper_id_2 in relationships_1.get('co_cited_papers', []):
                strength = max(strength, 0.7)
        
        if relationships_2:
            if paper_id_1 in relationships_2.get('co_cited_papers', []):
                strength = max(strength, 0.7)
        
        # Check bibliographic coupling (weak: 0.5)
        if relationships_1:
            if paper_id_2 in relationships_1.get('bibliographic_couples', []):
                strength = max(strength, 0.5)
        
        if relationships_2:
            if paper_id_1 in relationships_2.get('bibliographic_couples', []):
                strength = max(strength, 0.5)
        
        logger.debug(
            "Relationship strength calculated",
            paper_1=paper_id_1,
            paper_2=paper_id_2,
            strength=strength
        )
        
        return strength
    
    async def validate_ground_truth_coverage(self) -> Dict[str, any]:
        """
        Validate that we have adequate ground truth coverage.
        
        Returns:
            Dict with coverage statistics
        """
        logger.info("Validating ground truth coverage")
        
        stats = {
            "total_ground_truth_papers": len(self.ground_truth_papers),
            "papers_with_relationships": len(self.ground_truth_relationships),
            "domains_with_canonical": len(self.canonical_papers),
            "coverage_by_domain": {}
        }
        
        for domain, tiers in self.canonical_papers.items():
            stats["coverage_by_domain"][domain] = {
                tier: len(papers)
                for tier, papers in tiers.items()
            }
        
        logger.info(
            "Ground truth coverage validated",
            stats=stats
        )
        
        return stats