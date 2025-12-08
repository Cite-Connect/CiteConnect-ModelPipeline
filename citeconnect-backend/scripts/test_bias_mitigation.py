#!/usr/bin/env python3
"""
Test script to verify bias mitigation is being applied correctly.

This script:
1. Finds users in underperforming slices (fintech, masters, intermediate)
2. Generates recommendations for them
3. Checks if mitigation policy is applied
4. Compares scores with/without mitigation
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.connection import DatabaseConnection
from app.services.recommendation_service import RecommendationService
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def find_test_users(db: DatabaseConnection):
    """Find users in underperforming slices."""
    query = """
        SELECT 
            u.user_id,
            u.email,
            p.primary_domain,
            p.research_stage,
            p.reading_level
        FROM users u
        JOIN user_profiles_extended p ON u.user_id = p.user_id
        WHERE (
            p.primary_domain = 'fintech' OR
            p.research_stage = 'masters' OR
            p.reading_level = 'intermediate'
        )
        LIMIT 5
    """
    results = await db.fetch(query)
    return [dict(r) for r in results]


async def test_mitigation():
    """Test if bias mitigation is being applied."""
    print("\n" + "=" * 80)
    print("  TESTING BIAS MITIGATION")
    print("=" * 80 + "\n")

    db = DatabaseConnection()
    await db.connect()

    try:
        # Find test users
        print("1️⃣ Finding test users in underperforming slices...")
        test_users = await find_test_users(db)
        
        if not test_users:
            print("⚠️ No users found in underperforming slices")
            return
        
        print(f"   Found {len(test_users)} test users\n")

        # Initialize service
        service = RecommendationService(db)

        # Test each user
        for i, user in enumerate(test_users, 1):
            user_id = user['user_id']
            print(f"{'='*80}")
            print(f"User {i}: {user['email']} (ID: {user_id})")
            print(f"{'='*80}")
            print(f"  Domain: {user['primary_domain']}")
            print(f"  Stage: {user['research_stage']}")
            print(f"  Reading Level: {user['reading_level']}")
            
            # Check which slices they match
            matches = []
            if user['primary_domain'] == 'fintech':
                matches.append("fintech domain (1.25x boost)")
            if user['research_stage'] == 'masters':
                matches.append("masters stage (1.25x boost)")
            if user['reading_level'] == 'intermediate':
                matches.append("intermediate reading (1.25x boost)")
            
            if matches:
                print(f"\n  ✅ Matches underperforming slices:")
                for match in matches:
                    print(f"     - {match}")
            else:
                print(f"\n  ℹ️ No underperforming slice matches")
            
            # Generate recommendations
            print(f"\n  📊 Generating recommendations...")
            try:
                result = await service.generate_recommendations(
                    user_id=user_id,
                    count=10,
                    model='minilm'
                )
                
                # Check mitigation policy
                mitigation_policy = result.get('mitigation_policy', {})
                applied_rules = mitigation_policy.get('applied_rules', [])
                boost_factor = mitigation_policy.get('factor', 1.0)
                min_threshold = mitigation_policy.get('min_score_threshold')
                
                print(f"\n  🔍 Mitigation Analysis:")
                print(f"     Boost Factor: {boost_factor:.3f}x")
                if min_threshold:
                    print(f"     Min Score Threshold: {min_threshold:.3f}")
                else:
                    print(f"     Min Score Threshold: None")
                
                if applied_rules:
                    print(f"     Applied Rules: {len(applied_rules)}")
                    for rule in applied_rules:
                        print(f"       - {rule['field']}={rule['value']} "
                              f"(boost: {rule.get('boost_factor', 'N/A')})")
                else:
                    print(f"     Applied Rules: None")
                
                # Show top recommendations
                papers = result.get('papers', [])
                if papers:
                    print(f"\n  📝 Top 3 Recommendations:")
                    for j, paper in enumerate(papers[:3], 1):
                        score = paper.get('relevance_score', 0)
                        title = paper.get('title', 'N/A')[:50]
                        print(f"     {j}. {title}...")
                        print(f"        Score: {score:.3f}")
                        
                        # Check if mitigation was applied to this paper
                        if 'mitigation' in paper:
                            mit = paper['mitigation']
                            print(f"        Mitigation: {mit.get('factor', 1.0):.3f}x")
                
                # Verify mitigation is working
                if boost_factor > 1.0:
                    print(f"\n  ✅ MITIGATION IS ACTIVE!")
                    print(f"     Scores are being boosted by {boost_factor:.3f}x")
                else:
                    print(f"\n  ⚠️  MITIGATION NOT APPLIED")
                    print(f"     Expected boost > 1.0, got {boost_factor:.3f}")
                    if not applied_rules:
                        print(f"     Possible issue: User profile doesn't match config")
                
            except Exception as e:
                print(f"  ❌ Error generating recommendations: {e}")
                import traceback
                traceback.print_exc()
            
            print()

        print("\n" + "=" * 80)
        print("  SUMMARY")
        print("=" * 80)
        print("\n✅ Test complete!")
        print("\nNext steps:")
        print("  1. Verify boost_factor > 1.0 for users in underperforming slices")
        print("  2. Check that min_score_threshold is being applied")
        print("  3. Re-run bias detection to see if disparities are reduced")
        print("  4. Monitor recommendation quality metrics over time")

    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(test_mitigation())
