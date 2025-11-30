import asyncio
from app.db.connection import db
from app.services.recommendation_service import RecommendationService

async def test():
    await db.connect()
    
    service = RecommendationService(db)
    
    # Test cold-start
    result = await service.generate_recommendations(
        user_id=7,
        count=10,
        model='minilm'
    )
    
    print(f"✅ Generated {len(result['papers'])} recommendations")
    print(f"Method: {result['method']}")
    print(f"Total candidates: {result['total_candidates']}")
    
    print("\nTop 3 papers:")
    for i, paper in enumerate(result['papers'][:10], 1):
        print(f"{i}. {paper['title'][:60]}...")
        print(f"   Score: {paper['relevance_score']:.3f}")
        print(f"   Explanation: {paper['relevance_explanation']}")
    
    await db.disconnect()

asyncio.run(test())
