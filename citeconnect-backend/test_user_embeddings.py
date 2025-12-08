from app.db.connection import db
from app.services.user_embedding_service import UserEmbeddingService
import asyncio

async def test():
    await db.connect()
    
    user_id = 2  # Your test user
    service = UserEmbeddingService(db)
    
    print(f"\n{'='*60}")
    print(f"Testing User Embedding Generation for user_id={user_id}")
    print(f"{'='*60}\n")
    
    # Generate embeddings for both models
    embeddings = await service.get_or_generate_user_embeddings(user_id)
    
    print(f"✅ Embeddings generated!")
    print(f"\n   MiniLM:")
    print(f"      Shape: {embeddings['minilm'].shape}")
    print(f"      Range: [{embeddings['minilm'].min():.3f}, {embeddings['minilm'].max():.3f}]")
    print(f"      Sample: [{embeddings['minilm'][0]:.3f}, {embeddings['minilm'][1]:.3f}, {embeddings['minilm'][2]:.3f}, ...]")
    
    print(f"\n   SPECTER:")
    print(f"      Shape: {embeddings['specter'].shape}")
    print(f"      Range: [{embeddings['specter'].min():.3f}, {embeddings['specter'].max():.3f}]")
    print(f"      Sample: [{embeddings['specter'][0]:.3f}, {embeddings['specter'][1]:.3f}, {embeddings['specter'][2]:.3f}, ...]")
    
    # Check database
    print(f"\n{'='*60}")
    print("Checking Database Storage")
    print(f"{'='*60}\n")
    
    minilm_db = await db.fetchrow(
        "SELECT * FROM user_embeddings_minilm WHERE user_id=$1", user_id
    )
    specter_db = await db.fetchrow(
        "SELECT * FROM user_embeddings_specter WHERE user_id=$1", user_id
    )
    
    if minilm_db:
        print(f"✅ MiniLM embedding stored")
        print(f"   Method: {minilm_db['generation_method']}")
        print(f"   Interaction count: {minilm_db['interaction_count']}")
    
    if specter_db:
        print(f"✅ SPECTER embedding stored")
        print(f"   Method: {specter_db['generation_method']}")
        print(f"   Interaction count: {specter_db['interaction_count']}")
    
    # Check state
    state = await db.fetchrow(
        "SELECT * FROM user_recommendation_state WHERE user_id=$1", user_id
    )
    
    if state:
        print(f"\n✅ Recommendation state updated")
        print(f"   Stage: {state['recommendation_stage']}")
        print(f"   Last MiniLM update: {state['last_embedding_update_minilm']}")
        print(f"   Last SPECTER update: {state['last_embedding_update_specter']}")
    
    print(f"\n{'='*60}")
    print("✅ Test Complete!")
    print(f"{'='*60}\n")
    
    await db.disconnect()

asyncio.run(test())
