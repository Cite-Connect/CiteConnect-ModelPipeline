"""
Data quality validation script.
Validates that paper data is suitable for recommendations and ground truth.
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.utils.logger import setup_logging, get_logger
from app.db.connection import db

setup_logging()
logger = get_logger(__name__)


async def validate_papers():
    """Validate paper data quality."""
    print("\n" + "=" * 70)
    print("PAPER DATA VALIDATION")
    print("=" * 70)
    
    # Check total papers
    total = await db.fetchval("SELECT COUNT(*) FROM papers")
    print(f"\n📊 Total Papers: {total}")
    
    if total == 0:
        print("❌ CRITICAL: No papers found in database!")
        return False
    
    # Check distribution by domain
    print("\n📈 Distribution by Domain:")
    by_domain = await db.fetch("""
        SELECT domain, COUNT(*) as count
        FROM papers
        GROUP BY domain
        ORDER BY count DESC
    """)
    
    all_domains_ok = True
    for row in by_domain:
        domain = row['domain']
        count = row['count']
        
        if count < 100:
            print(f"   ⚠️  {domain}: {count} papers (need 100+)")
            all_domains_ok = False
        else:
            print(f"   ✅ {domain}: {count} papers")
    
    # Check required fields
    print("\n📝 Data Completeness:")
    
    with_abstract = await db.fetchval("""
        SELECT COUNT(*) FROM papers WHERE abstract IS NOT NULL
    """)
    abstract_pct = (with_abstract / total * 100) if total > 0 else 0
    print(f"   Abstract: {with_abstract}/{total} ({abstract_pct:.1f}%)")
    
    with_year = await db.fetchval("""
        SELECT COUNT(*) FROM papers WHERE year IS NOT NULL
    """)
    year_pct = (with_year / total * 100) if total > 0 else 0
    print(f"   Year: {with_year}/{total} ({year_pct:.1f}%)")
    
    with_authors = await db.fetchval("""
        SELECT COUNT(*) FROM papers 
        WHERE authors IS NOT NULL AND array_length(authors, 1) > 0
    """)
    authors_pct = (with_authors / total * 100) if total > 0 else 0
    print(f"   Authors: {with_authors}/{total} ({authors_pct:.1f}%)")
    
    # Check citation network (using correct field names)
    print("\n🔗 Citation Network:")
    
    with_refs = await db.fetchval("""
        SELECT COUNT(*) FROM papers 
        WHERE array_length(reference_ids, 1) > 0
    """)
    refs_pct = (with_refs / total * 100) if total > 0 else 0
    print(f"   Papers with references: {with_refs}/{total} ({refs_pct:.1f}%)")
    
    with_cites = await db.fetchval("""
        SELECT COUNT(*) FROM papers 
        WHERE array_length(citation_ids, 1) > 0
    """)
    cites_pct = (with_cites / total * 100) if total > 0 else 0
    print(f"   Papers with citations: {with_cites}/{total} ({cites_pct:.1f}%)")
    
    # Ground truth candidates (using correct field names)
    print("\n🎯 Ground Truth Candidates:")
    
    candidates = await db.fetchval("""
        SELECT COUNT(*) FROM papers
        WHERE array_length(reference_ids, 1) BETWEEN 10 AND 100
          AND citation_count >= 10
    """)
    
    if candidates < 50:
        print(f"   ⚠️  Only {candidates} candidates (need 50+ for robust evaluation)")
    else:
        print(f"   ✅ {candidates} candidates (good for ground truth)")
    
    # Check reference coverage (using correct field names)
    avg_coverage = await db.fetchval("""
        WITH paper_coverage AS (
            SELECT 
                paper_id,
                array_length(reference_ids, 1) as total_refs,
                (
                    SELECT COUNT(*)
                    FROM unnest(reference_ids) as ref_id
                    WHERE EXISTS (
                        SELECT 1 FROM papers WHERE paper_id = ref_id
                    )
                ) as refs_in_corpus
            FROM papers
            WHERE array_length(reference_ids, 1) > 0
        )
        SELECT AVG(refs_in_corpus::float / NULLIF(total_refs, 0))
        FROM paper_coverage
    """)
    
    coverage_pct = (avg_coverage * 100) if avg_coverage else 0
    print(f"   Average reference coverage: {coverage_pct:.1f}%")
    
    if coverage_pct < 30:
        print("   ⚠️  Low coverage - may affect ground truth quality")
    
    return all_domains_ok and total >= 300 and candidates >= 50


async def validate_embeddings():
    """Validate embedding data quality."""
    print("\n" + "=" * 70)
    print("EMBEDDING DATA VALIDATION")
    print("=" * 70)
    
    # Check MiniLM embeddings
    print("\n🔢 MiniLM Embeddings (384-dim):")
    
    total_papers = await db.fetchval("SELECT COUNT(*) FROM papers")
    total_embeddings = await db.fetchval("SELECT COUNT(*) FROM paper_embeddings_minilm")
    
    coverage = (total_embeddings / total_papers * 100) if total_papers > 0 else 0
    print(f"   Papers: {total_papers}")
    print(f"   Embeddings: {total_embeddings}")
    print(f"   Coverage: {coverage:.1f}%")
    
    if coverage < 95:
        print(f"   ❌ CRITICAL: Only {coverage:.1f}% coverage (need 95%+)")
        return False
    else:
        print(f"   ✅ Coverage is good ({coverage:.1f}%)")
    
    # Check embedding dimensions
    print("\n📏 Embedding Dimension Check:")
    
    sample_embedding = await db.fetchval("""
        SELECT embedding 
        FROM paper_embeddings_minilm 
        LIMIT 1
    """)
    
    if sample_embedding:
        dim = len(sample_embedding)
        if dim != 384:
            print(f"   ❌ CRITICAL: Wrong dimension ({dim}, expected 384)")
            return False
        else:
            print(f"   ✅ Correct dimension (384)")
    
    # Check for null embeddings
    null_count = await db.fetchval("""
        SELECT COUNT(*) FROM paper_embeddings_minilm
        WHERE embedding IS NULL
    """)
    
    if null_count > 0:
        print(f"   ⚠️  WARNING: {null_count} null embeddings found")
    
    return coverage >= 95


async def validate_ground_truth_readiness():
    """Check if data is ready for ground truth initialization."""
    print("\n" + "=" * 70)
    print("GROUND TRUTH READINESS CHECK")
    print("=" * 70)
    
    # Check if ground truth already exists
    existing_gt = await db.fetchval("SELECT COUNT(*) FROM ground_truth_papers")
    
    if existing_gt > 0:
        print(f"\n⚠️  Ground truth already initialized ({existing_gt} papers)")
        print("   Run initialization script to refresh if needed")
        return True
    
    # Check candidates
    print("\n🔍 Analyzing Candidate Papers:")
    
    candidates_by_domain = await db.fetch("""
        SELECT 
            domain,
            COUNT(*) as candidate_count
        FROM papers
        WHERE array_length(reference_ids, 1) BETWEEN 10 AND 100
          AND citation_count >= 10
        GROUP BY domain
        ORDER BY domain
    """)
    
    total_candidates = 0
    for row in candidates_by_domain:
        count = row['candidate_count']
        total_candidates += count
        
        if count < 15:
            print(f"   ⚠️  {row['domain']}: {count} candidates (need 15+ per domain)")
        else:
            print(f"   ✅ {row['domain']}: {count} candidates")
    
    print(f"\n   Total candidates: {total_candidates}")
    
    if total_candidates < 50:
        print("   ❌ Not enough candidates for ground truth")
        return False
    
    # Check reference coverage distribution
    print("\n📊 Reference Coverage Distribution:")
    
    coverage_stats = await db.fetchrow("""
        WITH coverage AS (
            SELECT 
                (
                    SELECT COUNT(*)
                    FROM unnest(reference_ids) as ref_id
                    WHERE EXISTS (
                        SELECT 1 FROM papers WHERE paper_id = ref_id
                    )
                )::float / NULLIF(array_length(reference_ids, 1), 0) as coverage_ratio
            FROM papers
            WHERE array_length(reference_ids, 1) BETWEEN 10 AND 100
        )
        SELECT 
            AVG(coverage_ratio) as avg_coverage,
            MIN(coverage_ratio) as min_coverage,
            MAX(coverage_ratio) as max_coverage
        FROM coverage
    """)
    
    if coverage_stats:
        avg = coverage_stats['avg_coverage'] or 0
        min_cov = coverage_stats['min_coverage'] or 0
        max_cov = coverage_stats['max_coverage'] or 0
        
        print(f"   Average: {avg*100:.1f}%")
        print(f"   Min: {min_cov*100:.1f}%")
        print(f"   Max: {max_cov*100:.1f}%")
        
        if avg < 0.3:
            print("   ⚠️  Low average coverage - ground truth quality may be limited")
    
    return total_candidates >= 50


async def main():
    """Run all validation checks."""
    logger.info("Starting data validation")
    
    await db.connect()
    
    try:
        # Run all validations
        papers_ok = await validate_papers()
        embeddings_ok = await validate_embeddings()
        ground_truth_ready = await validate_ground_truth_readiness()
        
        # Summary
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        
        print(f"\n✓ Paper Data: {'PASS ✅' if papers_ok else 'FAIL ❌'}")
        print(f"✓ Embeddings: {'PASS ✅' if embeddings_ok else 'FAIL ❌'}")
        print(f"✓ Ground Truth Ready: {'YES ✅' if ground_truth_ready else 'NO ❌'}")
        
        if papers_ok and embeddings_ok:
            print("\n🎉 Data quality is good!")
            
            if ground_truth_ready:
                print("\n📌 NEXT STEP: Run ground truth initialization")
                print("   docker-compose exec api python scripts/initialize_ground_truth.py")
            else:
                print("\n⚠️  Need more papers with citation data for ground truth")
                print("   Current candidates may be sufficient for testing")
        else:
            print("\n❌ Data quality issues found - see details above")
            print("   Fix issues before proceeding")
        
        print("\n" + "=" * 70)
        
    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())