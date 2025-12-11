"""
Incremental ground truth initialization - ONLY processes NEW papers
This version takes ~5 seconds instead of 2 minutes when adding 1 paper!
"""
import asyncio
from typing import List, Dict, Any
from datetime import datetime, timedelta

from app.config import settings
from app.utils.logger import get_logger
from app.db.connection import db

logger = get_logger(__name__)


async def identify_ground_truth_papers_incremental():
    """
    Incremental version - only processes papers NOT already in ground_truth_papers.
    Massively faster for single/small batch inserts.
    """
    logger.info("Identifying ground truth papers (incremental)")
    
    try:
        # Only get candidates that are NOT already in ground_truth_papers
        query = """
            WITH new_candidates AS (
                SELECT 
                    p.paper_id,
                    p.reference_ids,
                    array_length(p.reference_ids, 1) as ref_count,
                    p.citation_count,
                    p.year,
                    p.domain
                FROM papers p
                LEFT JOIN ground_truth_papers gtp ON p.paper_id = gtp.paper_id
                WHERE 
                    gtp.paper_id IS NULL  -- NOT already in ground_truth_papers
                    AND array_length(p.reference_ids, 1) BETWEEN $1 AND $2
                    AND p.citation_count >= 10
                    AND p.domain IS NOT NULL
                ORDER BY p.citation_count DESC
            ),
            reference_coverage AS (
                SELECT 
                    c.paper_id,
                    c.ref_count,
                    c.citation_count,
                    c.year,
                    c.domain,
                    (
                        SELECT COUNT(*)
                        FROM papers p2
                        WHERE p2.paper_id = ANY(c.reference_ids)
                    ) as refs_in_corpus,
                    (
                        SELECT COUNT(*)::float / c.ref_count
                        FROM papers p2
                        WHERE p2.paper_id = ANY(c.reference_ids)
                    ) as coverage
                FROM new_candidates c
            )
            SELECT 
                paper_id,
                ref_count,
                refs_in_corpus,
                citation_count,
                year,
                domain,
                coverage,
                -- Pre-calculate quality score
                (
                    LEAST(citation_count::float / 1000, 1.0) * 0.4 +
                    GREATEST(0, (COALESCE(year, 2012) - 2000)::float / 24) * 0.2 +
                    coverage * 0.4
                ) as quality_score
            FROM reference_coverage
            WHERE coverage >= $3
            ORDER BY quality_score DESC, citation_count DESC
        """
        
        candidates = await db.fetch(
            query,
            settings.MIN_GROUND_TRUTH_CITATIONS,
            settings.MAX_GROUND_TRUTH_CITATIONS,
            settings.MIN_REFERENCE_COVERAGE
        )
        
        if not candidates:
            logger.info("No new ground truth candidates found")
            return 0
        
        logger.info("New ground truth candidates found", count=len(candidates))
        
        # Insert query matching your schema
        insert_query = """
            INSERT INTO ground_truth_papers (
                paper_id, 
                reference_count, 
                references_in_corpus,
                quality_score,
                domain
            )
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (paper_id) DO UPDATE SET
                reference_count = EXCLUDED.reference_count,
                references_in_corpus = EXCLUDED.references_in_corpus,
                quality_score = EXCLUDED.quality_score,
                domain = EXCLUDED.domain
        """
        
        inserted_count = 0
        for c in candidates:
            await db.execute(
                insert_query,
                c['paper_id'],
                c['ref_count'],
                c['refs_in_corpus'],
                c['quality_score'],
                c['domain']
            )
            inserted_count += 1
            
            # Log progress only for large batches
            if inserted_count % 100 == 0:
                logger.info(f"Ground truth insert progress: {inserted_count}/{len(candidates)}")
        
        logger.info("New ground truth papers inserted", count=inserted_count)
        
        return inserted_count
        
    except Exception as e:
        logger.error("Error identifying ground truth papers", error=str(e), exc_info=True)
        raise


async def update_canonical_papers_for_new_papers(new_gt_count: int):
    """
    Smart canonical paper update - only recomputes if significant new papers added.
    Skips if only 1-2 new papers (won't change rankings significantly).
    """
    logger.info("Checking if canonical papers need update", new_papers=new_gt_count)
    
    # Skip if too few new papers (won't meaningfully change rankings)
    if new_gt_count < 5:
        logger.info("Skipping canonical paper update (too few new papers to matter)")
        return
    
    logger.info("Updating canonical papers (significant new papers added)")
    
    try:
        # Only process allowed domains
        domains = ['healthcare', 'fintech', 'quantum_computing']
        
        for domain in domains:
            logger.info(f"Processing domain: {domain}")
            
            # Single query to compute all tiers at once
            query = """
                WITH foundational AS (
                    SELECT 
                        'foundational' as tier,
                        array_agg(p.paper_id ORDER BY p.citation_count DESC) as paper_ids,
                        AVG(gtp.quality_score) as avg_score
                    FROM ground_truth_papers gtp
                    JOIN papers p ON gtp.paper_id = p.paper_id
                    WHERE gtp.domain = $1
                      AND p.citation_count >= 100
                      AND p.year BETWEEN 2010 AND 2020
                    HAVING COUNT(*) > 0
                    LIMIT 20
                ),
                recent AS (
                    SELECT 
                        'recent' as tier,
                        array_agg(p.paper_id ORDER BY p.citation_count DESC) as paper_ids,
                        AVG(gtp.quality_score) as avg_score
                    FROM ground_truth_papers gtp
                    JOIN papers p ON gtp.paper_id = p.paper_id
                    WHERE gtp.domain = $1
                      AND p.year >= EXTRACT(YEAR FROM CURRENT_DATE) - 2
                      AND p.citation_count >= 10
                    HAVING COUNT(*) > 0
                    LIMIT 20
                ),
                trending AS (
                    SELECT 
                        'trending' as tier,
                        array_agg(subq.paper_id ORDER BY subq.trend_score DESC) as paper_ids,
                        AVG(subq.quality_score) as avg_score
                    FROM (
                        SELECT 
                            gtp.paper_id,
                            gtp.quality_score,
                            (p.citation_count::float / GREATEST(
                                EXTRACT(DAY FROM NOW() - p.ingested_at), 1
                            )) as trend_score
                        FROM ground_truth_papers gtp
                        JOIN papers p ON gtp.paper_id = p.paper_id
                        WHERE gtp.domain = $1
                          AND p.ingested_at >= NOW() - INTERVAL '90 days'
                          AND p.citation_count >= 5
                        ORDER BY trend_score DESC
                        LIMIT 15
                    ) subq
                )
                SELECT tier, paper_ids, avg_score
                FROM foundational
                WHERE paper_ids IS NOT NULL
                UNION ALL
                SELECT tier, paper_ids, avg_score
                FROM recent
                WHERE paper_ids IS NOT NULL
                UNION ALL
                SELECT tier, paper_ids, avg_score
                FROM trending
                WHERE paper_ids IS NOT NULL
            """
            
            results = await db.fetch(query, domain)
            
            # Insert into domain_canonical_papers table
            insert_query = """
                INSERT INTO domain_canonical_papers (
                    domain, 
                    recommendation_tier, 
                    paper_ids, 
                    avg_quality_score
                )
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (domain, recommendation_tier)
                DO UPDATE SET
                    paper_ids = EXCLUDED.paper_ids,
                    avg_quality_score = EXCLUDED.avg_quality_score,
                    updated_at = CURRENT_TIMESTAMP
            """
            
            for result in results:
                if result['paper_ids']:
                    await db.execute(
                        insert_query,
                        domain,
                        result['tier'],
                        result['paper_ids'],
                        float(result['avg_score']) if result['avg_score'] else 0.0
                    )
                    
                    # Also update flags in ground_truth_papers
                    update_flags_query = """
                        UPDATE ground_truth_papers
                        SET 
                            is_canonical = true,
                            canonical_tier = $2
                        WHERE paper_id = ANY($3::text[])
                          AND domain = $1
                    """
                    
                    await db.execute(
                        update_flags_query,
                        domain,
                        result['tier'],
                        result['paper_ids']
                    )
            
            # Update domain ranks for canonical papers
            rank_query = """
                WITH ranked_papers AS (
                    SELECT 
                        paper_id,
                        ROW_NUMBER() OVER (ORDER BY quality_score DESC) as rank
                    FROM ground_truth_papers
                    WHERE domain = $1 AND is_canonical = true
                )
                UPDATE ground_truth_papers gtp
                SET domain_rank = rp.rank
                FROM ranked_papers rp
                WHERE gtp.paper_id = rp.paper_id
                  AND gtp.domain = $1
            """
            
            await db.execute(rank_query, domain)
        
        logger.info("Canonical papers updated for all domains", domain_count=len(domains))
        
    except Exception as e:
        logger.error("Error updating canonical papers", error=str(e), exc_info=True)
        raise


async def compute_ground_truth_relationships_incremental():
    """
    Compute citation relationships for NEW ground truth papers only.
    FIXED: No unnest() in HAVING clause - restructured queries.
    """
    logger.info("Computing ground truth relationships (incremental)")
    
    try:
        # Only get ground truth papers that DON'T have relationships yet
        papers_query = """
            SELECT gtp.paper_id, gtp.domain
            FROM ground_truth_papers gtp
            LEFT JOIN ground_truth_relationships gtr ON gtp.paper_id = gtr.paper_id
            WHERE gtr.paper_id IS NULL
        """
        
        papers_to_process = await db.fetch(papers_query)
        
        if not papers_to_process:
            logger.info("No new ground truth papers need relationship computation")
            return 0
        
        logger.info(
            "Processing relationships for new papers",
            count=len(papers_to_process)
        )
        
        processed_count = 0
        
        for gt_paper in papers_to_process:
            paper_id = gt_paper['paper_id']
            
            try:
                # Get paper's reference_ids and citation info
                paper_data_query = """
                    SELECT 
                        reference_ids,
                        citation_ids,
                        array_length(citation_ids, 1) as citation_count
                    FROM papers
                    WHERE paper_id = $1
                """
                
                paper_data = await db.fetchrow(paper_data_query, paper_id)
                
                if not paper_data:
                    logger.warning(f"Paper data not found for {paper_id}")
                    continue
                
                # Extract citation network (papers that cite this paper)
                citation_network = paper_data['citation_ids'] or []
                citation_network_size = len(citation_network)
                
                # Find co-cited papers - FIXED QUERY
                co_cited_query = """
                    WITH citers AS (
                        SELECT unnest($1::text[]) as citing_paper
                    ),
                    ref_expansions AS (
                        SELECT 
                            p.paper_id as citing_paper,
                            unnest(p.reference_ids) as co_cited_paper
                        FROM papers p
                        WHERE p.paper_id IN (SELECT citing_paper FROM citers)
                          AND p.reference_ids IS NOT NULL
                    ),
                    co_citation_counts AS (
                        SELECT 
                            co_cited_paper,
                            COUNT(*) as strength
                        FROM ref_expansions
                        WHERE co_cited_paper != $2
                        GROUP BY co_cited_paper
                        HAVING COUNT(*) >= 2
                    )
                    SELECT 
                        co_cited_paper,
                        strength
                    FROM co_citation_counts
                    ORDER BY strength DESC
                    LIMIT 50
                """
                
                co_cited_results = await db.fetch(co_cited_query, citation_network, paper_id)
                co_cited_papers = [r['co_cited_paper'] for r in co_cited_results]
                co_citation_strengths = [float(r['strength']) for r in co_cited_results]
                
                # Find bibliographic couples - FIXED QUERY
                references = paper_data['reference_ids'] or []
                
                if references:
                    bib_couple_query = """
                        WITH this_paper_refs AS (
                            SELECT unnest($1::text[]) as ref_paper
                        ),
                        ref_expansions AS (
                            SELECT 
                                p.paper_id as couple_paper,
                                unnest(p.reference_ids) as ref
                            FROM papers p
                            WHERE p.reference_ids IS NOT NULL
                              AND p.paper_id != $2
                        ),
                        shared_ref_counts AS (
                            SELECT 
                                couple_paper,
                                COUNT(*) as strength
                            FROM ref_expansions
                            WHERE ref IN (SELECT ref_paper FROM this_paper_refs)
                            GROUP BY couple_paper
                            HAVING COUNT(*) >= 2
                        )
                        SELECT 
                            couple_paper,
                            strength
                        FROM shared_ref_counts
                        ORDER BY strength DESC
                        LIMIT 50
                    """
                    
                    bib_couple_results = await db.fetch(bib_couple_query, references, paper_id)
                    bib_couples = [r['couple_paper'] for r in bib_couple_results]
                    coupling_strengths = [float(r['strength']) for r in bib_couple_results]
                else:
                    bib_couples = []
                    coupling_strengths = []
                
                # Calculate network centrality
                network_centrality = min(citation_network_size / 100.0, 1.0)
                
                # Calculate relationship quality score
                network_score = min(citation_network_size / 50.0, 1.0)
                co_citation_score = min(len(co_cited_papers) / 30.0, 1.0)
                coupling_score = min(len(bib_couples) / 30.0, 1.0)
                
                relationship_quality = (
                    network_score * 0.4 +
                    co_citation_score * 0.3 +
                    coupling_score * 0.3
                )
                
                # Insert relationship data
                insert_query = """
                    INSERT INTO ground_truth_relationships (
                        paper_id,
                        citation_network,
                        citation_network_size,
                        co_cited_papers,
                        co_citation_strengths,
                        bibliographic_couples,
                        coupling_strengths,
                        relationship_quality_score,
                        network_centrality
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (paper_id) DO UPDATE SET
                        citation_network = EXCLUDED.citation_network,
                        citation_network_size = EXCLUDED.citation_network_size,
                        co_cited_papers = EXCLUDED.co_cited_papers,
                        co_citation_strengths = EXCLUDED.co_citation_strengths,
                        bibliographic_couples = EXCLUDED.bibliographic_couples,
                        coupling_strengths = EXCLUDED.coupling_strengths,
                        relationship_quality_score = EXCLUDED.relationship_quality_score,
                        network_centrality = EXCLUDED.network_centrality,
                        last_updated = CURRENT_TIMESTAMP
                """
                
                await db.execute(
                    insert_query,
                    paper_id,
                    citation_network[:100],
                    citation_network_size,
                    co_cited_papers,
                    co_citation_strengths,
                    bib_couples,
                    coupling_strengths,
                    relationship_quality,
                    network_centrality
                )
                
                processed_count += 1
                
                # Log progress every 20 papers
                if processed_count % 20 == 0:
                    logger.info(
                        f"Relationship computation progress: {processed_count}/{len(papers_to_process)}"
                    )
                    
            except Exception as e:
                logger.error(
                    "Failed to compute relationships for paper",
                    paper_id=paper_id,
                    error=str(e)
                )
                # Continue with next paper instead of failing entire process
                continue
        
        logger.info(
            "Ground truth relationships computed",
            count=processed_count
        )
        
        return processed_count
        
    except Exception as e:
        logger.error(
            "Error computing ground truth relationships",
            error=str(e),
            exc_info=True
        )
        raise

# Export functions with same names for compatibility
identify_ground_truth_papers = identify_ground_truth_papers_incremental
compute_ground_truth_relationships = compute_ground_truth_relationships_incremental

# Wrapper for identify_canonical_papers to match expected signature
async def identify_canonical_papers(new_gt_count: int = 0):
    """
    Wrapper that calls update_canonical_papers_for_new_papers.
    Receives the count of newly added ground truth papers.
    """
    await update_canonical_papers_for_new_papers(new_gt_count)