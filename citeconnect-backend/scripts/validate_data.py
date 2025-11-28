"""
Data quality validation script.
Validates that paper data is suitable for recommendations and ground truth.
Updated to match Supabase schema field names.
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


async def validate_users_and_profiles():
    """Validate that user data is set up correctly."""
    print("\n" + "=" * 70)
    print("USER & PROFILE VALIDATION")
    print("=" * 70)
    
    # Check users
    user_count = await db.fetchval("SELECT COUNT(*) FROM users WHERE is_active = true")
    print(f"\n👥 Active Users: {user_count}")
    
    if user_count == 0:
        print("   ⚠️  No users found. Create test users first.")
        return False
    
    # Check profiles
    profile_count = await db.fetchval("SELECT COUNT(*) FROM user_profiles_extended")
    print(f"👤 User Profiles: {profile_count}/{user_count}")
    
    if profile_count < user_count:
        print(f"   ⚠️  {user_count - profile_count} users without profiles")
    
    # Check profile completeness distribution
    completeness_stats = await db.fetchrow("""
        SELECT 
            AVG(profile_completeness) as avg,
            MIN(profile_completeness) as min,
            MAX(profile_completeness) as max
        FROM user_profiles_extended
    """)
    
    if completeness_stats and completeness_stats['avg']:
        print(f"\n📊 Profile Completeness:")
        print(f"   Average: {completeness_stats['avg']:.2f}")
        print(f"   Range: {completeness_stats['min']:.2f} - {completeness_stats['max']:.2f}")
        
        if completeness_stats['avg'] < 0.5:
            print("   ⚠️  Low average completeness - recommendations may be less accurate")
    
    # Check interests in hierarchy table
    interest_count = await db.fetchval("SELECT COUNT(*) FROM user_interest_hierarchy")
    avg_interests = interest_count / profile_count if profile_count > 0 else 0
    
    print(f"\n🎯 User Interests:")
    print(f"   Total interests: {interest_count}")
    print(f"   Average per user: {avg_interests:.1f}")
    
    if avg_interests < 3:
        print("   ⚠️  Users should have at least 3 interests each")
    else:
        print("   ✅ Good interest coverage")
    
    # Check by interest level
    by_level = await db.fetch("""
        SELECT interest_level, COUNT(*) as count
        FROM user_interest_hierarchy
        GROUP BY interest_level
        ORDER BY interest_level
    """)
    
    if len(by_level) > 0:
        print("\n   By Interest Level:")
        for row in by_level:
            print(f"     Level {row['interest_level']}: {row['count']} interests")
    
    # Check recommendation states
    state_count = await db.fetchval("SELECT COUNT(*) FROM user_recommendation_state")
    print(f"\n📈 Recommendation States: {state_count}/{user_count}")
    
    if state_count < user_count:
        print(f"   ⚠️  {user_count - state_count} users without recommendation states")
    
    return profile_count >= user_count and interest_count >= (profile_count * 3)


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
        print("   Backend is working, but needs paper data to generate recommendations.")
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
    required_domains = ['healthcare', 'fintech', 'quantum_computing']
    
    for row in by_domain:
        domain = row['domain']
        count = row['count']
        
        if count < 100:
            print(f"   ⚠️  {domain}: {count} papers (need 100+ for quality recommendations)")
            all_domains_ok = False
        else:
            print(f"   ✅ {domain}: {count} papers")
    
    # Check if all required domains are present
    found_domains = [row['domain'] for row in by_domain]
    missing_domains = set(required_domains) - set(found_domains)
    
    if missing_domains:
        print(f"   ⚠️  Missing domains: {', '.join(missing_domains)}")
        all_domains_ok = False
    
    # Check required fields
    print("\n📝 Data Completeness:")
    
    with_abstract = await db.fetchval("""
        SELECT COUNT(*) FROM papers WHERE abstract IS NOT NULL
    """)
    abstract_pct = (with_abstract / total * 100) if total > 0 else 0
    status = "✅" if abstract_pct > 90 else "⚠️"
    print(f"   {status} Abstract: {with_abstract}/{total} ({abstract_pct:.1f}%)")
    
    with_year = await db.fetchval("""
        SELECT COUNT(*) FROM papers WHERE year IS NOT NULL
    """)
    year_pct = (with_year / total * 100) if total > 0 else 0
    status = "✅" if year_pct > 95 else "⚠️"
    print(f"   {status} Year: {with_year}/{total} ({year_pct:.1f}%)")
    
    with_authors = await db.fetchval("""
        SELECT COUNT(*) FROM papers 
        WHERE authors IS NOT NULL AND array_length(authors, 1) > 0
    """)
    authors_pct = (with_authors / total * 100) if total > 0 else 0
    status = "✅" if authors_pct > 90 else "⚠️"
    print(f"   {status} Authors: {with_authors}/{total} ({authors_pct:.1f}%)")
    
    # Check citation network
    print("\n🔗 Citation Network:")
    
    with_refs = await db.fetchval("""
        SELECT COUNT(*) FROM papers 
        WHERE array_length(reference_ids, 1) > 0
    """)
    refs_pct = (with_refs / total * 100) if total > 0 else 0
    status = "✅" if refs_pct > 50 else "⚠️"
    print(f"   {status} Papers with references: {with_refs}/{total} ({refs_pct:.1f}%)")
    
    with_cites = await db.fetchval("""
        SELECT COUNT(*) FROM papers 
        WHERE array_length(citation_ids, 1) > 0
    """)
    cites_pct = (with_cites / total * 100) if total > 0 else 0
    status = "✅" if cites_pct > 50 else "⚠️"
    print(f"   {status} Papers with citations: {with_cites}/{total} ({cites_pct:.1f}%)")
    
    # Ground truth candidates
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
    
    # Check reference coverage
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
    status = "✅" if coverage_pct > 30 else "⚠️"
    print(f"   {status} Average reference coverage: {coverage_pct:.1f}%")
    
    if coverage_pct < 30:
        print("      Low coverage may affect ground truth quality")
    
    return all_domains_ok and total >= 300 and candidates >= 50


async def validate_embeddings():
    """Validate embedding data quality for BOTH models."""
    print("\n" + "=" * 70)
    print("EMBEDDING DATA VALIDATION")
    print("=" * 70)
    
    total_papers = await db.fetchval("SELECT COUNT(*) FROM papers")
    
    if total_papers == 0:
        print("   ⚠️  No papers in database - skipping embedding validation")
        return False
    
    # Check MiniLM embeddings (384-dim)
    print("\n🔢 MiniLM Embeddings (384-dim):")
    
    minilm_count = await db.fetchval("SELECT COUNT(*) FROM paper_embeddings_minilm")
    minilm_coverage = (minilm_count / total_papers * 100) if total_papers > 0 else 0
    
    print(f"   Papers: {total_papers}")
    print(f"   Embeddings: {minilm_count}")
    print(f"   Coverage: {minilm_coverage:.1f}%")
    
    minilm_ok = False
    if minilm_coverage < 95:
        print(f"   ⚠️  Coverage below 95% (missing {total_papers - minilm_count} embeddings)")
    else:
        print(f"   ✅ Excellent coverage!")
        minilm_ok = True
    
    # Check MiniLM dimension
    if minilm_count > 0:
        sample_minilm = await db.fetchval("""
            SELECT embedding 
            FROM paper_embeddings_minilm 
            LIMIT 1
        """)
        
        if sample_minilm:
            minilm_dim = len(sample_minilm)
            if minilm_dim != 384:
                print(f"   ❌ Wrong dimension ({minilm_dim}, expected 384)")
                minilm_ok = False
            else:
                print(f"   ✅ Correct dimension (384)")
    
    # Check SPECTER embeddings (768-dim)
    print("\n🔢 SPECTER Embeddings (768-dim):")
    
    specter_count = await db.fetchval("SELECT COUNT(*) FROM paper_embeddings_specter")
    specter_coverage = (specter_count / total_papers * 100) if total_papers > 0 else 0
    
    print(f"   Papers: {total_papers}")
    print(f"   Embeddings: {specter_count}")
    print(f"   Coverage: {specter_coverage:.1f}%")
    
    specter_ok = False
    if specter_count == 0:
        print(f"   ⚠️  No SPECTER embeddings found (optional)")
    elif specter_coverage < 95:
        print(f"   ⚠️  Coverage below 95% (missing {total_papers - specter_count} embeddings)")
    else:
        print(f"   ✅ Excellent coverage!")
        specter_ok = True
    
    # Check SPECTER dimension
    if specter_count > 0:
        sample_specter = await db.fetchval("""
            SELECT embedding 
            FROM paper_embeddings_specter 
            LIMIT 1
        """)
        
        if sample_specter:
            specter_dim = len(sample_specter)
            if specter_dim != 768:
                print(f"   ⚠️  Wrong dimension ({specter_dim}, expected 768)")
            else:
                print(f"   ✅ Correct dimension (768)")
    
    # Check for null embeddings in both tables
    print("\n🔍 Data Quality Checks:")
    
    minilm_nulls = await db.fetchval("""
        SELECT COUNT(*) FROM paper_embeddings_minilm
        WHERE embedding IS NULL
    """)
    
    if minilm_nulls > 0:
        print(f"   ⚠️  MiniLM: {minilm_nulls} null embeddings found")
    else:
        print(f"   ✅ MiniLM: No null embeddings")
    
    if specter_count > 0:
        specter_nulls = await db.fetchval("""
            SELECT COUNT(*) FROM paper_embeddings_specter
            WHERE embedding IS NULL
        """)
        
        if specter_nulls > 0:
            print(f"   ⚠️  SPECTER: {specter_nulls} null embeddings found")
        else:
            print(f"   ✅ SPECTER: No null embeddings")
    
    # Summary
    print("\n📊 Embedding Summary:")
    print(f"   MiniLM ready: {'YES ✅' if minilm_ok else 'NO ⚠️'}")
    print(f"   SPECTER ready: {'YES ✅' if specter_ok else 'NO ⚠️' if specter_count > 0 else 'NOT POPULATED ⚠️'}")
    
    # At least one model should have good coverage
    return minilm_ok or specter_ok


async def validate_ground_truth_readiness():
    """Check if data is ready for ground truth initialization."""
    print("\n" + "=" * 70)
    print("GROUND TRUTH READINESS CHECK")
    print("=" * 70)
    
    # Check if ground truth already exists
    existing_gt = await db.fetchval("SELECT COUNT(*) FROM ground_truth_papers")
    
    if existing_gt > 0:
        print(f"\n⚠️  Ground truth already initialized ({existing_gt} papers)")
        print("   To refresh, run initialization script again")
        
        # Show existing ground truth stats
        by_domain = await db.fetch("""
            SELECT domain, COUNT(*) as count
            FROM ground_truth_papers
            GROUP BY domain
            ORDER BY domain
        """)
        
        if len(by_domain) > 0:
            print("\n   Current Ground Truth Distribution:")
            for row in by_domain:
                print(f"     - {row['domain']}: {row['count']} papers")
        
        return True
    
    # Check if we have any papers to analyze
    total_papers = await db.fetchval("SELECT COUNT(*) FROM papers")
    
    if total_papers == 0:
        print("\n❌ No papers in database - cannot initialize ground truth")
        return False
    
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
            print(f"   ⚠️  {row['domain']}: {count} candidates (ideally 15+ per domain)")
        else:
            print(f"   ✅ {row['domain']}: {count} candidates")
    
    print(f"\n   Total candidates: {total_candidates}")
    
    if total_candidates < 50:
        print("   ❌ Not enough candidates for ground truth")
        print("   Need papers with:")
        print("     - 10-100 references (reference_ids)")
        print("     - At least 10 citations (citation_count)")
        return False
    elif total_candidates < 100:
        print("   ⚠️  Minimum candidates met, but 100+ recommended")
    else:
        print("   ✅ Excellent candidate pool!")
    
    # Check reference coverage distribution
    print("\n📊 Reference Coverage Analysis:")
    
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
            MAX(coverage_ratio) as max_coverage,
            COUNT(*) as paper_count
        FROM coverage
        WHERE coverage_ratio IS NOT NULL
    """)
    
    if coverage_stats and coverage_stats['avg_coverage']:
        avg = coverage_stats['avg_coverage']
        min_cov = coverage_stats['min_coverage']
        max_cov = coverage_stats['max_coverage']
        
        print(f"   Average: {avg*100:.1f}%")
        print(f"   Range: {min_cov*100:.1f}% - {max_cov*100:.1f}%")
        print(f"   Papers analyzed: {coverage_stats['paper_count']}")
        
        if avg < 0.3:
            print("   ⚠️  Low average coverage (<30%)")
            print("      Many referenced papers not in corpus")
            print("      Ground truth evaluation will be limited")
        elif avg < 0.5:
            print("   ⚠️  Medium coverage (30-50%)")
            print("      Acceptable but could be better")
        else:
            print("   ✅ Good coverage (>50%)")
    else:
        print("   ⚠️  Could not calculate coverage (no papers with references)")
    
    return total_candidates >= 50


async def main():
    """Run all validation checks."""
    logger.info("Starting data validation")
    
    await db.connect()
    
    try:
        # Run all validations
        users_ok = await validate_users_and_profiles()
        papers_ok = await validate_papers()
        embeddings_ok = await validate_embeddings()
        ground_truth_ready = await validate_ground_truth_readiness()
        
        # Summary
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        
        print(f"\n✓ User Data: {'PASS ✅' if users_ok else 'NEEDS ATTENTION ⚠️'}")
        print(f"✓ Paper Data: {'PASS ✅' if papers_ok else 'FAIL ❌'}")
        print(f"✓ Embeddings: {'PASS ✅' if embeddings_ok else 'FAIL ❌'}")
        print(f"✓ Ground Truth Ready: {'YES ✅' if ground_truth_ready else 'NO ❌'}")
        
        if users_ok:
            print("\n✅ User management is working!")
        
        if papers_ok and embeddings_ok:
            print("\n🎉 Paper data quality is good!")
            
            if ground_truth_ready:
                print("\n📌 NEXT STEP: Run ground truth initialization")
                print("   Command: docker-compose exec api python scripts/initialize_ground_truth.py")
            else:
                print("\n⚠️  Need more papers with citation data for ground truth")
                print("   Current candidates may be sufficient for basic testing")
                print("   For production, aim for 100+ candidates")
        else:
            print("\n❌ Data quality issues found - see details above")
            print("\n📋 Action Items:")
            if not papers_ok:
                print("   1. Load paper data (minimum 100 per domain)")
            if not embeddings_ok:
                print("   2. Generate embeddings for all papers (384-dim)")
            print("   3. Ensure citation networks are complete (reference_ids, citation_ids)")
        
        print("\n" + "=" * 70)
        
    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())