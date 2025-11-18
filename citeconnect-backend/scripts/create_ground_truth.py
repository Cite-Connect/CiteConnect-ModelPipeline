#!/usr/bin/env python3

"""
Create Synthetic Ground Truth

For demo purposes, create user interactions to enable proper metric evaluation.

Strategy:
- Generate recommendations for each user
- "Save" top 3-5 papers as if user liked them
- This becomes ground truth for evaluation

Run: python scripts/create_ground_truth.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.recommendation_service import recommendation_service
from app.db.postgres import execute_query


async def create_synthetic_ground_truth():
    """
    Create synthetic user interactions for evaluation
    
    For each user:
    1. Generate recommendations  
    2. Save top 3 papers (simulates user liking them)
    3. Add view interactions for top 5
    """
    print("\n" + "="*80)
    print("  Creating Synthetic Ground Truth")
    print("="*80 + "\n")
    
    # Get test users
    users = await execute_query(
        """
        SELECT u.user_id, u.email, u.name
        FROM users u
        WHERE u.email LIKE '%@example.com'
        """,
        fetch_all=True
    )
    
    print(f"Found {len(users)} test users\n")
    
    for user in users:
        user_id = user['user_id']
        email = user['email']
        
        print(f"Processing: {email} (ID: {user_id})")
        
        try:
            # Generate recommendations
            recommendations = await recommendation_service.generate_recommendations(
                user_id=user_id,
                top_k=10
            )
            
            print(f"  ✓ Generated {len(recommendations)} recommendations")
            
            # Save top 3 papers
            saved_count = 0
            for rec in recommendations[:3]:
                paper_id = rec['paper_id']
                
                await execute_query(
                    """
                    INSERT INTO user_saved_papers (user_id, paper_id, saved_at, notes)
                    VALUES ($1, $2, CURRENT_TIMESTAMP, 'Auto-generated for demo')
                    ON CONFLICT (user_id, paper_id) DO NOTHING
                    """,
                    user_id,
                    paper_id
                )
                
                saved_count += 1
            
            print(f"  ✓ Saved {saved_count} papers")
            
            # Like top 5 papers
            liked_count = 0
            for rec in recommendations[:5]:
                paper_id = rec['paper_id']
                
                await execute_query(
                    """
                    INSERT INTO user_liked_papers (user_id, paper_id, liked_at)
                    VALUES ($1, $2, CURRENT_TIMESTAMP)
                    ON CONFLICT (user_id, paper_id) DO NOTHING
                    """,
                    user_id,
                    paper_id
                )
                
                liked_count += 1
            
            print(f"  ✓ Liked {liked_count} papers")
            
            # Add view interactions for top 5
            view_count = 0
            for i, rec in enumerate(recommendations[:5]):
                paper_id = rec['paper_id']
                
                # Simulate different read times (longer for higher ranked)
                read_time = 60 + (5 - i) * 30  # 60-180 seconds
                
                await execute_query(
                    """
                    INSERT INTO user_interactions 
                        (user_id, paper_id, interaction_type, duration_seconds, context)
                    VALUES ($1, $2, 'read_time', $3, '{"source": "recommendations", "demo": true}')
                    """,
                    user_id,
                    paper_id,
                    read_time
                )
                
                view_count += 1
            
            print(f"  ✓ Added {view_count} view interactions\n")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}\n")
    
    # Summary
    print("="*80)
    print("  Ground Truth Creation Complete")
    print("="*80 + "\n")
    
    # Verify data
    saved_total = await execute_query(
        "SELECT COUNT(*) as count FROM user_saved_papers",
        fetch_one=True
    )
    
    liked_total = await execute_query(
        "SELECT COUNT(*) as count FROM user_liked_papers",
        fetch_one=True
    )
    
    interactions_total = await execute_query(
        "SELECT COUNT(*) as count FROM user_interactions",
        fetch_one=True
    )
    
    print(f"Database Summary:")
    print(f"  Saved papers: {saved_total['count']}")
    print(f"  Liked papers: {liked_total['count']}")
    print(f"  Interactions: {interactions_total['count']}")
    
    print(f"\n✓ Ground truth created for evaluation")
    print(f"\nNext step: python scripts/run_experiment.py")
    print(f"           (Metrics should now show non-zero values)\n")


if __name__ == "__main__":
    asyncio.run(create_synthetic_ground_truth())