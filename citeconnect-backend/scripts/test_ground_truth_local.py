"""
Test script to generate ground truth candidates locally without writing to database.
Outputs results to a JSON file for review.
"""
import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.utils.logger import setup_logging, get_logger
from app.db.connection import db
from app.db.repositories.paper_repo import PaperRepository

setup_logging()
logger = get_logger(__name__)


async def identify_ground_truth_candidates_local():
    """
    Identify papers suitable for ground truth evaluation (read-only).
    Outputs results to a JSON file instead of writing to database.
    """
    logger.info("Identifying ground truth candidates (LOCAL TEST - no database writes)")
    
    await db.connect()
    
    try:
        # Query papers with reference data (using reference_ids)
        query = """
            SELECT 
                p.paper_id,
                p.reference_ids,
                array_length(p.reference_ids, 1) as ref_count,
                p.citation_count,
                p.year,
                p.domain,
                p.title
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
        ground_truth_papers = []
        stats = {
            "total_candidates": len(candidates),
            "qualified_count": 0,
            "disqualified_count": 0,
            "disqualification_reasons": {
                "no_references": 0,
                "low_coverage": 0
            }
        }
        
        for candidate in candidates:
            paper_id = candidate['paper_id']
            references = candidate['reference_ids'] or []
            
            if not references:
                stats["disqualification_reasons"]["no_references"] += 1
                continue
            
            # Check how many references are in our corpus
            ref_check_query = """
                SELECT COUNT(*) 
                FROM papers 
                WHERE paper_id = ANY($1::text[])
            """
            
            refs_in_corpus = await db.fetchval(ref_check_query, references)
            reference_coverage = refs_in_corpus / len(references) if references else 0
            
            if reference_coverage >= settings.MIN_REFERENCE_COVERAGE:
                # Calculate quality score
                citation_score = min(candidate['citation_count'] / 1000, 1.0)
                recency_score = max(0, (candidate['year'] - 2000) / 24) if candidate['year'] else 0.5
                coverage_score = reference_coverage
                
                quality_score = (
                    citation_score * 0.4 +
                    recency_score * 0.2 +
                    coverage_score * 0.4
                )
                
                # Add to results (without writing to database)
                ground_truth_papers.append({
                    "paper_id": paper_id,
                    "title": candidate.get('title', 'N/A'),
                    "domain": candidate['domain'],
                    "year": candidate['year'],
                    "citation_count": candidate['citation_count'],
                    "num_references": len(references),
                    "references_in_corpus": refs_in_corpus,
                    "reference_coverage": round(reference_coverage, 4),
                    "quality_score": round(quality_score, 4),
                    "citation_score": round(citation_score, 4),
                    "recency_score": round(recency_score, 4),
                    "coverage_score": round(coverage_score, 4)
                })
                
                stats["qualified_count"] += 1
                
                logger.debug(
                    "Ground truth candidate qualified",
                    paper_id=paper_id,
                    quality_score=quality_score,
                    coverage=reference_coverage
                )
            else:
                stats["disqualification_reasons"]["low_coverage"] += 1
                stats["disqualified_count"] += 1
        
        # Sort by quality score
        ground_truth_papers.sort(key=lambda x: x['quality_score'], reverse=True)
        
        # Prepare output
        output = {
            "generated_at": datetime.now().isoformat(),
            "settings": {
                "min_ground_truth_citations": settings.MIN_GROUND_TRUTH_CITATIONS,
                "max_ground_truth_citations": settings.MAX_GROUND_TRUTH_CITATIONS,
                "min_reference_coverage": settings.MIN_REFERENCE_COVERAGE
            },
            "statistics": stats,
            "ground_truth_papers": ground_truth_papers,
            "summary": {
                "total_qualified": len(ground_truth_papers),
                "top_10_quality_scores": [p["quality_score"] for p in ground_truth_papers[:10]],
                "average_quality_score": round(
                    sum(p["quality_score"] for p in ground_truth_papers) / len(ground_truth_papers) 
                    if ground_truth_papers else 0, 
                    4
                ),
                "by_domain": {}
            }
        }
        
        # Group by domain
        for paper in ground_truth_papers:
            domain = paper["domain"]
            if domain not in output["summary"]["by_domain"]:
                output["summary"]["by_domain"][domain] = 0
            output["summary"]["by_domain"][domain] += 1
        
        logger.info(
            "Ground truth candidates identified (LOCAL)",
            qualified=stats["qualified_count"],
            disqualified=stats["disqualified_count"]
        )
        
        return output
        
    finally:
        await db.disconnect()


async def compute_ground_truth_relationships_local(ground_truth_paper_ids: list):
    """
    Compute relationships for ground truth papers (read-only).
    Returns relationship data without writing to database.
    """
    logger.info("Computing ground truth relationships (LOCAL TEST - no database writes)")
    
    await db.connect()
    
    try:
        paper_repo = PaperRepository(db)
        relationships_data = []
        
        for paper_id in ground_truth_paper_ids:
            try:
                # Get direct citations and references
                citations = await paper_repo.get_paper_citations(paper_id)
                references = await paper_repo.get_paper_references(paper_id)
                
                # Find co-cited papers
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
                    SELECT co_cited_paper, co_citation_count
                    FROM co_cited_counts
                    WHERE co_citation_count >= 3
                      AND co_cited_paper != $1
                    ORDER BY co_citation_count DESC
                    LIMIT 50
                """
                
                co_cited_results = await db.fetch(co_cited_query, paper_id)
                co_cited_papers = [r['co_cited_paper'] for r in co_cited_results]
                
                # Find bibliographic couples
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
                    SELECT couple_paper, shared_refs
                    FROM couple_counts
                    WHERE shared_refs >= 3
                    ORDER BY shared_refs DESC
                    LIMIT 50
                """
                
                bib_couple_results = await db.fetch(bib_couple_query, paper_id)
                bib_couples = [r['couple_paper'] for r in bib_couple_results]
                
                # Calculate network centrality
                centrality = min(len(citations) / 100, 1.0)
                
                relationships_data.append({
                    "paper_id": paper_id,
                    "citation_network_count": len(citations),
                    "citation_network_sample": citations[:10],  # First 10 for preview
                    "co_cited_papers_count": len(co_cited_papers),
                    "co_cited_papers_sample": co_cited_papers[:10],
                    "bibliographic_couples_count": len(bib_couples),
                    "bibliographic_couples_sample": bib_couples[:10],
                    "network_centrality": round(centrality, 4)
                })
                
            except Exception as e:
                logger.warning(
                    "Failed to compute relationships for paper",
                    paper_id=paper_id,
                    error=str(e)
                )
                relationships_data.append({
                    "paper_id": paper_id,
                    "error": str(e)
                })
        
        logger.info(
            "Relationships computed (LOCAL)",
            papers_processed=len(relationships_data)
        )
        
        return relationships_data
        
    finally:
        await db.disconnect()


async def main():
    """Main function to generate local ground truth test."""
    logger.info("=" * 60)
    logger.info("Starting LOCAL ground truth candidate generation")
    logger.info("(No database writes - output to JSON file)")
    logger.info("=" * 60)
    
    try:
        # Generate candidates
        output = await identify_ground_truth_candidates_local()
        
        # Compute relationships for top 20 papers (to keep output manageable)
        top_paper_ids = [p["paper_id"] for p in output["ground_truth_papers"][:20]]
        logger.info(f"Computing relationships for top {len(top_paper_ids)} papers")
        relationships = await compute_ground_truth_relationships_local(top_paper_ids)
        
        # Add relationships to output
        output["relationships"] = relationships
        output["relationships_summary"] = {
            "papers_with_relationships": len(relationships),
            "total_citations": sum(r.get("citation_network_count", 0) for r in relationships),
            "total_co_cited": sum(r.get("co_cited_papers_count", 0) for r in relationships),
            "total_bib_couples": sum(r.get("bibliographic_couples_count", 0) for r in relationships),
            "average_centrality": round(
                sum(r.get("network_centrality", 0) for r in relationships) / len(relationships)
                if relationships else 0,
                4
            )
        }
        
        # Write to JSON file
        output_file = Path(__file__).parent / "ground_truth_candidates_local.json"
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info("=" * 60)
        logger.info("Local ground truth generation complete")
        logger.info(f"Qualified papers: {output['statistics']['qualified_count']}")
        logger.info(f"Output file: {output_file}")
        logger.info("=" * 60)
        
        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total candidates analyzed: {output['statistics']['total_candidates']}")
        print(f"Qualified: {output['statistics']['qualified_count']}")
        print(f"Disqualified: {output['statistics']['disqualified_count']}")
        print(f"\nDisqualification reasons:")
        for reason, count in output['statistics']['disqualification_reasons'].items():
            print(f"  - {reason}: {count}")
        print(f"\nBy domain:")
        for domain, count in output['summary']['by_domain'].items():
            print(f"  - {domain}: {count}")
        print(f"\nAverage quality score: {output['summary']['average_quality_score']}")
        if 'relationships_summary' in output:
            print(f"\nRelationships computed for {output['relationships_summary']['papers_with_relationships']} papers:")
            print(f"  - Total citations: {output['relationships_summary']['total_citations']}")
            print(f"  - Total co-cited papers: {output['relationships_summary']['total_co_cited']}")
            print(f"  - Total bibliographic couples: {output['relationships_summary']['total_bib_couples']}")
            print(f"  - Average centrality: {output['relationships_summary']['average_centrality']}")
        print(f"\nResults saved to: {output_file}")
        print("=" * 60)
        
    except Exception as e:
        logger.error(
            "Local ground truth generation failed",
            error=str(e),
            exc_info=True
        )
        raise


if __name__ == "__main__":
    asyncio.run(main())

