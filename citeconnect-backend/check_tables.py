import asyncio
from app.db.connection import db

async def check():
    await db.connect()
    
    user_id = 1
    
    print(f"\nData for user_id={user_id}")
    print("="*60)
    
    # Table 1: users
    user = await db.fetchrow('SELECT * FROM users WHERE user_id=$1', user_id)
    print(f"\n1. users: {user['email'] if user else 'Not found'}")
    
    # Table 2: user_recommendation_state
    state = await db.fetchrow('SELECT * FROM user_recommendation_state WHERE user_id=$1', user_id)
    print(f"2. state: {state['recommendation_stage'] if state else 'Not found'}")
    
    # Table 3: user_profiles_extended
    profile = await db.fetchrow('SELECT * FROM user_profiles_extended WHERE user_id=$1', user_id)
    print(f"3. profile: {profile['primary_domain'] if profile else 'Not found'}")
    
    # Table 4: user_interest_hierarchy
    interests = await db.fetch('SELECT * FROM user_interest_hierarchy WHERE user_id=$1', user_id)
    print(f"4. interests: {len(interests)} rows")
    
    # Table 5: user_interactions
    interactions = await db.fetch('SELECT * FROM user_interactions WHERE user_id=$1', user_id)
    print(f"5. interactions: {len(interactions)} rows")
    
    # Table 6: user_saved_papers
    saved = await db.fetch('SELECT * FROM user_saved_papers WHERE user_id=$1', user_id)
    print(f"6. saved_papers: {len(saved)} rows")
    
    # Table 7: user_embeddings
    emb = await db.fetchrow('SELECT * FROM user_embeddings_minilm WHERE user_id=$1', user_id)
    print(f"7. user_embeddings: {'Yes' if emb else 'Not yet generated'}")
    
    print("\n" + "="*60)
    
    await db.disconnect()

asyncio.run(check())
