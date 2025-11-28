"""
Initialize ground truth papers and relationships.
Identifies high-quality papers and pre-computes citation networks.
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.utils.logger import setup_logging, get_logger
from app.db.connection import db
from app.db.repositories.paper_repo import PaperRepository
from app.db.repositories.ground_truth_repo import GroundTruthRepository

setup_logging()
logger = get_logger(__name__)


async def identify_ground_truth_papers():
    """
    Identify papers suitable for ground truth evaluation.
    Uses correct field names: reference_ids, citation_ids.
    
    Criteria:
    - Has 10-100 references (enough for eval, not a review)
    - At least 30% of references in our corpus
    - Quality score > threshold
    """
    logger.info("Identifying ground truth papers")
    
    await db.connect()
    
    try:
        paper_repo = PaperRepository(db)
        gt_repo = GroundTruthRepository(db)
        
        # Query papers with reference data (using reference_ids)
        query = """
            SELECT 
                p.paper_id,
                p.reference_ids,
                array_length(p.reference_ids, 1) as ref_count,
                p.citation_count,
                p.year,
                p.domain
            FROM papers p
            WHERE 
                array_length(p.reference_ids, 1) BETWEEN $1 AND $2
                AND p.citation_count >= 10
            ORDER BY p.citation_count DESC
            LIMIT 1000
        """
        
        candidates = await db.fetch(
            query,
            settings.MIN_GROUND_TRUTH_CITATIONS,
            settings.MAX_GROUND_TRUTH_CITATIONS
        )
        
        logger.info(
            "Ground truth candidates found",
            count=len(candidates)
        )
        
        # Process each candidate
        ground_truth_count = 0
        
        for candidate in candidates:
            paper_id = candidate['paper_id']
            references = candidate['reference_ids'] or []  # Changed from 'references'
            
            if not references:
                continue
            
            # Check how many references are in our corpus
            ref_check_query = """
                SELECT COUNT(*) 
                FROM papers 
                WHERE paper_id = ANY($1::text[])
            """
            
            refs_in_corpus = await db.fetchval(ref_check_query, references)
            
            reference_coverage = refs_in_corpus / len(references)
            
            if reference_coverage >= settings.MIN_REFERENCE_COVERAGE:
                # Calculate quality score
                # Based on: citations, recency, coverage
                # Note: reference_coverage will be auto-calculated by database
                citation_score = min(candidate['citation_count'] / 1000, 1.0)
                recency_score = max(0, (candidate['year'] - 2000) / 24) if candidate['year'] else 0.5
                coverage_score = reference_coverage  # Used for quality calc only
                
                quality_score = (
                    citation_score * 0.4 +
                    recency_score * 0.2 +
                    coverage_score * 0.4
                )
                
                # Create ground truth paper
                # Pass reference_coverage for quality calculation, but it won't be inserted
                # Database will auto-calculate it from reference_count and references_in_corpus
                await gt_repo.create_ground_truth_paper(
                    paper_id=paper_id,
                    num_references=len(references),
                    reference_coverage=reference_coverage,  # Used internally, not inserted
                    quality_score=quality_score
                )
                
                ground_truth_count += 1
                
                logger.debug(
                    "Ground truth paper created",
                    paper_id=paper_id,
                    quality_score=quality_score,
                    coverage=reference_coverage
                )
        
        logger.info(
            "Ground truth papers identified",
            count=ground_truth_count
        )
        
        return ground_truth_count
        
    finally:
        await db.disconnect()


async def compute_ground_truth_relationships():
    """
    Pre-compute citation relationships for ground truth papers.
    Uses correct field names: citation_ids and reference_ids.
    """
    logger.info("Computing ground truth relationships")
    
    await db.connect()
    
    try:
        gt_repo = GroundTruthRepository(db)
        paper_repo = PaperRepository(db)
        
        # Get all ground truth papers
        gt_papers = await gt_repo.get_ground_truth_papers()
        
        logger.info(
            "Processing relationships for papers",
            count=len(gt_papers)
        )
        
        for gt_paper in gt_papers:
            paper_id = gt_paper['paper_id']
            
            # Get direct citations and references (using correct field names)
            citations = await paper_repo.get_paper_citations(paper_id)  # Uses citation_ids
            references = await paper_repo.get_paper_references(paper_id)  # Uses reference_ids
            
            # Find co-cited papers
            # Papers that are frequently cited together with this paper
            co_cited_query = """
                WITH this_paper_citers AS (
                    SELECT unnest(citation_ids) as citing_paper
                    FROM papers
                    WHERE paper_id = $1
                ),
                co_cited_counts AS (
                    SELECT 
                        unnest(reference_ids) as co_cited_paper,
                        COUNT(*) as co_citation_count
                    FROM papers p
                    WHERE p.paper_id IN (SELECT citing_paper FROM this_paper_citers)
                    GROUP BY co_cited_paper
                )
                SELECT co_cited_paper
                FROM co_cited_counts
                WHERE co_citation_count >= 3
                  AND co_cited_paper != $1
                ORDER BY co_citation_count DESC
                LIMIT 50
            """
            
            co_cited_results = await db.fetch(co_cited_query, paper_id)
            co_cited_papers = [r['co_cited_paper'] for r in co_cited_results]
            
            # Find bibliographic couples
            # Papers that cite the same sources
            bib_couple_query = """
                WITH this_paper_refs AS (
                    SELECT unnest(reference_ids) as ref_paper
                    FROM papers
                    WHERE paper_id = $1
                ),
                couple_counts AS (
                    SELECT 
                        p.paper_id as couple_paper,
                        COUNT(*) as shared_refs
                    FROM papers p,
                         unnest(p.reference_ids) as ref
                    WHERE ref IN (SELECT ref_paper FROM this_paper_refs)
                      AND p.paper_id != $1
                    GROUP BY p.paper_id
                )
                SELECT couple_paper
                FROM couple_counts
                WHERE shared_refs >= 3
                ORDER BY shared_refs DESC
                LIMIT 50
            """
            
            bib_couple_results = await db.fetch(bib_couple_query, paper_id)
            bib_couples = [r['couple_paper'] for r in bib_couple_results]
            
            # Calculate network centrality (simplified PageRank proxy)
            # Based on citation count and reference quality
            centrality = min(len(citations) / 100, 1.0)
            
            # Save relationships
            await gt_repo.save_ground_truth_relationships(
                paper_id=paper_id,
                citation_network=citations[:100],  # Limit size
                co_cited_papers=co_cited_papers,
                bibliographic_couples=bib_couples,
                network_centrality=centrality
            )
            
            logger.debug(
                "Relationships computed",
                paper_id=paper_id,
                citations=len(citations),
                co_cited=len(co_cited_papers),
                bib_couples=len(bib_couples)
            )
        
        logger.info(
            "Ground truth relationships computed",
            count=len(gt_papers)
        )
        
    finally:
        await db.disconnect()


async def identify_canonical_papers():
    """
    Identify canonical papers for each domain.
    Uses array-based structure in Supabase schema.
    """
    logger.info("Identifying canonical papers")
    
    await db.connect()
    
    try:
        # Only process allowed domains from Supabase schema
        domains = ['healthcare', 'fintech', 'quantum_computing']
        
        for domain in domains:
            logger.info(f"Processing domain: {domain}")
            
            # === FOUNDATIONAL PAPERS (highly cited classics) ===
            foundational_query = """
                SELECT array_agg(paper_id ORDER BY citation_count DESC) as paper_ids,
                       AVG(citation_count::float / 1000) as avg_score,
                       COUNT(*) as count
                FROM (
                    SELECT paper_id, citation_count
                    FROM papers
                    WHERE domain = $1
                      AND citation_count >= 100
                      AND year BETWEEN 2010 AND 2020
                    ORDER BY citation_count DESC
                    LIMIT 20
                ) subq
            """
            
            result = await db.fetchrow(foundational_query, domain)
            
            if result and result['paper_ids']:
                insert_query = """
                    INSERT INTO domain_canonical_papers (
                        domain, recommendation_tier, paper_ids, 
                        avg_quality_score, paper_count
                    )
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (domain, recommendation_tier)
                    DO UPDATE SET
                        paper_ids = EXCLUDED.paper_ids,
                        avg_quality_score = EXCLUDED.avg_quality_score,
                        paper_count = EXCLUDED.paper_count,
                        updated_at = NOW()
                """
                
                await db.execute(
                    insert_query,
                    domain,
                    'foundational',
                    result['paper_ids'],
                    float(result['avg_score']) if result['avg_score'] else 0.0,
                    result['count']
                )
                
                logger.debug(
                    "Foundational papers added",
                    domain=domain,
                    count=result['count']
                )
            
            # === RECENT PAPERS (last 2 years, high quality) ===
            recent_query = """
                SELECT array_agg(paper_id ORDER BY citation_count DESC) as paper_ids,
                       AVG(citation_count::float / 100) as avg_score,
                       COUNT(*) as count
                FROM (
                    SELECT paper_id, citation_count
                    FROM papers
                    WHERE domain = $1
                      AND year >= EXTRACT(YEAR FROM CURRENT_DATE) - 2
                      AND citation_count >= 10
                    ORDER BY citation_count DESC
                    LIMIT 20
                ) subq
            """
            
            result = await db.fetchrow(recent_query, domain)
            
            if result and result['paper_ids']:
                await db.execute(
                    insert_query,
                    domain,
                    'recent',
                    result['paper_ids'],
                    float(result['avg_score']) if result['avg_score'] else 0.0,
                    result['count']
                )
                
                logger.debug(
                    "Recent papers added",
                    domain=domain,
                    count=result['count']
                )
            
            # === TRENDING PAPERS (high citation velocity) ===
            trending_query = """
                SELECT array_agg(paper_id ORDER BY trend_score DESC) as paper_ids,
                       AVG(trend_score) as avg_score,
                       COUNT(*) as count
                FROM (
                    SELECT 
                        paper_id,
                        (citation_count::float / GREATEST(
                            EXTRACT(DAY FROM NOW() - ingested_at), 1
                        )) as trend_score
                    FROM papers
                    WHERE domain = $1
                      AND ingested_at >= NOW() - INTERVAL '90 days'
                      AND citation_count >= 5
                    ORDER BY trend_score DESC
                    LIMIT 15
                ) subq
            """
            
            result = await db.fetchrow(trending_query, domain)
            
            if result and result['paper_ids']:
                await db.execute(
                    insert_query,
                    domain,
                    'trending',
                    result['paper_ids'],
                    float(result['avg_score']) if result['avg_score'] else 0.0,
                    result['count']
                )
                
                logger.debug(
                    "Trending papers added",
                    domain=domain,
                    count=result['count']
                )
        
        logger.info(
            "Canonical papers identified for all domains",
            domain_count=len(domains)
        )
        
    finally:
        await db.disconnect()


async def main():
    """Main initialization function."""
    logger.info("=" * 60)
    logger.info("Starting ground truth initialization")
    logger.info("=" * 60)
    
    try:
        # Step 1: Identify ground truth papers
        gt_count = await identify_ground_truth_papers()
        
        # Step 2: Compute relationships
        await compute_ground_truth_relationships()
        
        # Step 3: Identify canonical papers
        await identify_canonical_papers()
        
        logger.info("=" * 60)
        logger.info("Ground truth initialization complete")
        logger.info(f"Ground truth papers: {gt_count}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(
            "Ground truth initialization failed",
            error=str(e),
            exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())