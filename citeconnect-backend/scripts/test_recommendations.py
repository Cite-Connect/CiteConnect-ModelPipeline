#!/usr/bin/env python3

"""
Test Recommendation System

Tests the complete recommendation pipeline:
1. Generate user profile embeddings
2. Generate recommendations from pickle file
3. Evaluate metrics
4. Detect bias

Run: python scripts/test_recommendations.py
"""

import asyncio
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.embedding_service import embedding_service
from app.services.recommendation_service import recommendation_service
from app.services.evaluation_service import evaluation_service
from app.db.postgres import execute_query


async def test_full_pipeline():
    """Test complete recommendation pipeline"""
    
    print("\n" + "="*80)
    print("  CiteConnect Recommendation System Test")
    print("="*80 + "\n")
    
    # Get test users
    users = await execute_query(
        """
        SELECT u.user_id, u.email, u.name, ud.domain
        FROM users u
        JOIN user_domains ud ON u.user_id = ud.user_id
        WHERE u.email LIKE '%@example.com'
        ORDER BY u.user_id
        """,
        fetch_all=True
    )
    
    print(f"Found {len(users)} test users:\n")
    for user in users:
        print(f"  {user['user_id']}. {user['name']} ({user['email']})")
        print(f"      Domain: {user['domain']}")
    
    print("\n" + "-"*80 + "\n")
    
    # Test each user
    all_results = []
    
    for user in users:
        user_id = user['user_id']
        email = user['email']
        
        print(f"Testing User: {user['name']} (ID: {user_id})")
        print("-" * 40)
        
        try:
            # Step 1: Generate/Get user embedding
            print("  [1/4] Generating user profile embedding...")
            user_embedding = await embedding_service.get_user_profile_embedding(user_id)
            print(f"        ✓ Embedding shape: {user_embedding.shape}")
            
            # Step 2: Generate recommendations
            print("  [2/4] Generating recommendations...")
            recommendations = await recommendation_service.generate_recommendations(
                user_id=user_id,
                top_k=10
            )
            print(f"        ✓ Generated {len(recommendations)} recommendations")
            
            # Show top 3
            print("\n        Top 3 Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"          {i}. {rec['title'][:60]}...")
                print(f"             Year: {rec['year']} | Citations: {rec['citation_count']}")
                print(f"             Score: {rec['composite_score']:.3f}")
                print(f"               - Semantic: {rec['score_components']['semantic_similarity']:.3f}")
                print(f"               - Citations: {rec['score_components']['normalized_citations']:.3f}")
                print(f"               - Recency: {rec['score_components']['recency_score']:.3f}")
            
            # Step 3: Evaluate metrics
            print(f"\n  [3/4] Evaluating metrics...")
            rec_ids = [r['paper_id'] for r in recommendations]
            metrics = await evaluation_service.evaluate_recommendations(
                user_id=user_id,
                recommended_paper_ids=rec_ids,
                k=10
            )
            
            print(f"        Precision@10: {metrics['precision_at_k']:.3f} (target: ≥0.60)")
            print(f"        Recall@10: {metrics['recall_at_k']:.3f} (target: ≥0.75)")
            print(f"        MRR: {metrics['mrr']:.3f} (target: ≥0.70)")
            print(f"        NDCG@10: {metrics['ndcg_at_k']:.3f}")
            
            if metrics['ground_truth_size'] == 0:
                print(f"        ⚠ No ground truth available (expected for new users)")
            
            # Step 4: Bias detection
            print(f"\n  [4/4] Running bias detection...")
            bias_report = await evaluation_service.detect_domain_bias(
                user_id=user_id,
                recommended_papers=recommendations,
                threshold=0.50
            )
            
            if bias_report['is_biased']:
                print(f"        ⚠ BIAS DETECTED:")
                for domain, pct in bias_report['biased_domains'].items():
                    print(f"          {domain}: {pct:.1%}")
            else:
                print(f"        ✓ No significant bias detected")
            
            print(f"        Domain distribution:")
            for domain, pct in bias_report['domain_distribution'].items():
                print(f"          {domain}: {pct:.1%}")
            
            all_results.append({
                'user_id': user_id,
                'email': email,
                'metrics': metrics,
                'bias_report': bias_report
            })
            
            print()
            
        except Exception as e:
            print(f"        ✗ Error: {str(e)}\n")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*80)
    print("  Test Summary")
    print("="*80 + "\n")
    
    if all_results:
        # Calculate averages
        avg_precision = np.mean([r['metrics']['precision_at_k'] for r in all_results])
        avg_recall = np.mean([r['metrics']['recall_at_k'] for r in all_results])
        avg_mrr = np.mean([r['metrics']['mrr'] for r in all_results])
        avg_ndcg = np.mean([r['metrics']['ndcg_at_k'] for r in all_results])
        
        print(f"Average Metrics Across {len(all_results)} Users:")
        print(f"  Precision@10: {avg_precision:.3f} {'✓' if avg_precision >= 0.60 else '✗'} (target: ≥0.60)")
        print(f"  Recall@10: {avg_recall:.3f} {'✓' if avg_recall >= 0.75 else '✗'} (target: ≥0.75)")
        print(f"  MRR: {avg_mrr:.3f} {'✓' if avg_mrr >= 0.70 else '✗'} (target: ≥0.70)")
        print(f"  NDCG@10: {avg_ndcg:.3f}")
        
        # Check if targets met
        targets_met = (
            avg_precision >= 0.60 and
            avg_recall >= 0.75 and
            avg_mrr >= 0.70
        )
        
        if targets_met:
            print(f"\n  ✓ ALL TARGETS MET")
        else:
            print(f"\n  ⚠ Some targets not met (expected without ground truth)")
            print(f"    Action: Seed user interactions for proper evaluation")
    else:
        print("  No results (all tests failed)")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(test_full_pipeline())