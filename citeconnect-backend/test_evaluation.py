"""Test the evaluation service."""
import asyncio
from app.db.connection import db
from app.services.recommendation_service import RecommendationService
from app.services.evaluation_service import EvaluationService
from app.utils.logger import setup_logging

async def test():
    setup_logging()
    await db.connect()
    
    print("\n" + "="*70)
    print("TESTING EVALUATION SERVICE")
    print("="*70)
    
    # Initialize services
    rec_service = RecommendationService(db)
    eval_service = EvaluationService(db)
    
    user_id = 7  # Your test user
    
    # Step 1: Generate recommendations
    print(f"\n1️⃣ Generating recommendations for user {user_id}...")
    recommendations = await rec_service.generate_recommendations(
        user_id=user_id,
        count=10,
        model='minilm'
    )
    
    print(f"✅ Generated {len(recommendations['papers'])} recommendations")
    
    # Step 2: Evaluate recommendations
    print(f"\n2️⃣ Evaluating recommendations...")
    evaluation = await eval_service.evaluate_cold_start_recommendations(
            user_id=user_id,
            recommendations=recommendations['papers'],
            model='minilm',  # ADD THIS
            store_result=True
        )
    
    # Step 3: Display results
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")
    
    print(f"\n📊 Metrics:")
    print(f"   Profile Alignment:    {evaluation['profile_alignment']:.4f} (target: ≥0.60)")
    print(f"   Ground Truth Quality: {evaluation['ground_truth_quality']:.4f} (target: ≥0.50)")
    print(f"   Combined Score:       {evaluation['combined_score']:.4f} (target: ≥0.60)")
    
    print(f"\n🎯 Result: {'✅ PASS' if evaluation['passes_threshold'] else '❌ FAIL'}")
    
    # Step 4: Show top 3 papers with details
    print(f"\n📝 Top 3 Recommended Papers:")
    for i, paper in enumerate(recommendations['papers'][:3], 1):
        print(f"\n{i}. {paper['title'][:60]}...")
        print(f"   Final Score: {paper['final_score']:.3f}")
        
        if 'score_breakdown' in paper:
            breakdown = paper['score_breakdown']
            print(f"   Breakdown:")
            print(f"      Semantic:      {breakdown.get('semantic', 0):.3f}")
            print(f"      Citation:      {breakdown.get('citation', 0):.3f}")
            print(f"      Recency:       {breakdown.get('recency', 0):.3f}")
            print(f"      Ground Truth:  {breakdown.get('ground_truth', 0):.3f}")
            print(f"      Reading Level: {breakdown.get('reading_level', 0):.3f}")
    
    print(f"\n{'='*70}")
    print("✅ Evaluation test complete!")
    print(f"{'='*70}\n")
    
    await db.disconnect()

asyncio.run(test())