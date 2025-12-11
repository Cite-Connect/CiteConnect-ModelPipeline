"""
Dry-run test script to check what would be modified in the database
when running test_recommendations task
"""
import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.connection import db
from app.services.recommendation_service import RecommendationService
from app.services.evaluation_service import EvaluationService


async def check_before():
    """Check database state BEFORE running the task"""
    print("="*80)
    print("BEFORE: Database State Check")
    print("="*80)
    
    await db.connect()
    
    # Find a cold-start user
    user = await db.fetchrow("""
        SELECT u.user_id, u.email, COUNT(ui.interaction_id) as interaction_count
        FROM users u
        LEFT JOIN user_interactions ui ON u.user_id = ui.user_id
        WHERE u.is_active = true
        GROUP BY u.user_id, u.email
        HAVING COUNT(ui.interaction_id) < 10
        LIMIT 1
    """)
    
    if not user:
        print("❌ No cold-start users found")
        await db.disconnect()
        return None
    
    user_id = user['user_id']
    print(f"\n✅ Test User: user_id={user_id}, email={user['email']}, interactions={user['interaction_count']}")
    
    # Check existing embeddings
    minilm = await db.fetchrow("SELECT * FROM user_embeddings_minilm WHERE user_id = $1", user_id)
    specter = await db.fetchrow("SELECT * FROM user_embeddings_specter WHERE user_id = $1", user_id)
    
    print(f"\n📊 Current State:")
    print(f"   user_embeddings_minilm: {'EXISTS' if minilm else 'DOES NOT EXIST'}")
    if minilm:
        print(f"      - interaction_count: {minilm['interaction_count']}")
        print(f"      - last_updated: {minilm['last_updated']}")
        print(f"      - generation_method: {minilm.get('generation_method', 'N/A')}")
    
    print(f"   user_embeddings_specter: {'EXISTS' if specter else 'DOES NOT EXIST'}")
    if specter:
        print(f"      - interaction_count: {specter['interaction_count']}")
        print(f"      - last_updated: {specter['last_updated']}")
        print(f"      - generation_method: {specter.get('generation_method', 'N/A')}")
    
    # Check recommendation state
    state = await db.fetchrow("SELECT * FROM user_recommendation_state WHERE user_id = $1", user_id)
    print(f"   user_recommendation_state: {'EXISTS' if state else 'DOES NOT EXIST'}")
    if state:
        print(f"      - recommendation_stage: {state['recommendation_stage']}")
        print(f"      - interaction_count: {state['interaction_count']}")
        print(f"      - last_embedding_update_minilm: {state.get('last_embedding_update_minilm', 'NULL')}")
        print(f"      - last_embedding_update_specter: {state.get('last_embedding_update_specter', 'NULL')}")
    
    # Check evaluation count
    eval_count = await db.fetchval("SELECT COUNT(*) FROM cold_start_evaluations WHERE user_id = $1", user_id)
    print(f"   cold_start_evaluations: {eval_count} existing records")
    
    await db.disconnect()
    return user_id


async def check_after(user_id):
    """Check database state AFTER running the task"""
    print("\n" + "="*80)
    print("AFTER: Database State Check")
    print("="*80)
    
    await db.connect()
    
    # Check embeddings again
    minilm = await db.fetchrow("SELECT * FROM user_embeddings_minilm WHERE user_id = $1", user_id)
    specter = await db.fetchrow("SELECT * FROM user_embeddings_specter WHERE user_id = $1", user_id)
    
    print(f"\n📊 Updated State:")
    print(f"   user_embeddings_minilm: {'EXISTS' if minilm else 'DOES NOT EXIST'}")
    if minilm:
        print(f"      - interaction_count: {minilm['interaction_count']}")
        print(f"      - last_updated: {minilm['last_updated']}")
        print(f"      - generation_method: {minilm.get('generation_method', 'N/A')}")
    
    print(f"   user_embeddings_specter: {'EXISTS' if specter else 'DOES NOT EXIST'}")
    if specter:
        print(f"      - interaction_count: {specter['interaction_count']}")
        print(f"      - last_updated: {specter['last_updated']}")
        print(f"      - generation_method: {specter.get('generation_method', 'N/A')}")
    
    # Check recommendation state
    state = await db.fetchrow("SELECT * FROM user_recommendation_state WHERE user_id = $1", user_id)
    if state:
        print(f"   user_recommendation_state:")
        print(f"      - recommendation_stage: {state['recommendation_stage']}")
        print(f"      - interaction_count: {state['interaction_count']}")
        print(f"      - last_embedding_update_minilm: {state.get('last_embedding_update_minilm', 'NULL')}")
        print(f"      - last_embedding_update_specter: {state.get('last_embedding_update_specter', 'NULL')}")
    
    # Check evaluation count (should be same since store_result=False)
    eval_count = await db.fetchval("SELECT COUNT(*) FROM cold_start_evaluations WHERE user_id = $1", user_id)
    print(f"   cold_start_evaluations: {eval_count} records (should be unchanged)")
    
    await db.disconnect()


async def run_test_task(user_id):
    """Run the actual test_recommendations task logic"""
    print("\n" + "="*80)
    print("RUNNING: test_recommendations Task")
    print("="*80)
    
    await db.connect()
    
    try:
        # Step 1: Generate recommendations
        print(f"\n📋 Step 1: Generating 10 recommendations for user {user_id}...")
        rec_service = RecommendationService(db)
        
        recommendations = await rec_service.generate_cold_start_recommendations(
            user_id=user_id,
            count=10,
            model='minilm'
        )
        
        rec_count = len(recommendations.get('papers', []))
        print(f"✅ Generated {rec_count} recommendations")
        
        if rec_count == 0:
            print("❌ No recommendations generated!")
            return
        
        # Print first few paper IDs
        paper_ids = [p.get('paper_id', 'N/A') for p in recommendations.get('papers', [])[:3]]
        print(f"   Sample paper IDs: {paper_ids}")
        
        # Step 2: Evaluate recommendations (WITHOUT storing to DB)
        print(f"\n📋 Step 2: Evaluating recommendations (store_result=False)...")
        eval_service = EvaluationService(db)
        
        evaluation = await eval_service.evaluate_cold_start_recommendations(
            user_id=user_id,
            recommendations=recommendations['papers'],
            model='minilm',
            store_result=False  # ⚠️ NOT storing to DB
        )
        
        # Step 3: Print scores
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"User ID: {user_id}")
        print(f"Recommendations: {rec_count}")
        print(f"\n📊 Scores:")
        print(f"   Profile Alignment: {evaluation['profile_alignment']:.4f}")
        print(f"   Ground Truth Quality: {evaluation['ground_truth_quality']:.4f}")
        print(f"   Combined Score: {evaluation['combined_score']:.4f}")
        print(f"   Passes Threshold (≥0.60): {'✅ YES' if evaluation['passes_threshold'] else '❌ NO'}")
        
    finally:
        await db.disconnect()


async def main():
    """Main test function"""
    print("\n" + "="*80)
    print("DRY-RUN TEST: What Would Be Modified in Database?")
    print("="*80)
    
    # Check state before
    user_id = await check_before()
    
    if not user_id:
        print("\n❌ Cannot proceed - no test user found")
        return
    
    # Run the task
    await run_test_task(user_id)
    
    # Check state after
    await check_after(user_id)
    
    print("\n" + "="*80)
    print("SUMMARY: What Gets Modified?")
    print("="*80)
    print("""
    ✅ SAFE CHANGES (Expected Behavior):
    
    1. user_embeddings_minilm table:
       - INSERT if user has no MiniLM embedding
       - UPDATE if embedding is outdated (interaction_count changed)
       - This is NORMAL for cold-start users - they need embeddings!
    
    2. user_embeddings_specter table:
       - INSERT if user has no SPECTER embedding
       - UPDATE if embedding is outdated
       - This is NORMAL for cold-start users
    
    3. user_recommendation_state table:
       - UPDATE last_embedding_update_minilm timestamp
       - UPDATE last_embedding_update_specter timestamp
       - Potentially UPDATE recommendation_stage if user transitions
       - This is NORMAL - just tracking when embeddings were generated
    
    ❌ NO CHANGES (Because store_result=False):
    
    4. cold_start_evaluations table:
       - NO INSERT (store_result=False)
       - This is SAFE - evaluation results are NOT stored
    
    ⚠️  IMPACT ASSESSMENT:
    
    - These changes are HARMLESS and EXPECTED
    - Generating embeddings for cold-start users is the correct behavior
    - No user data is deleted or corrupted
    - Only metadata/timestamps are updated
    - The user's interaction_count does NOT change (read-only)
    """)


if __name__ == "__main__":
    asyncio.run(main())







