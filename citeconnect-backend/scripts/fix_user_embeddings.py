#!/usr/bin/env python3

"""
Fix User Embeddings - Regenerate with Correct Model

Problem: User embeddings are 768-dim (SPECTER2) but paper embeddings are 384-dim
Solution: Regenerate user embeddings with all-MiniLM-L6-v2 (384-dim)

Run: python scripts/fix_user_embeddings.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from app.db.postgres import execute_query


async def regenerate_user_embeddings_384dim():
    """
    Regenerate user embeddings using all-MiniLM-L6-v2 (384-dim)
    
    This matches the dimension of paper embeddings in pickle file.
    """
    print("\n" + "="*80)
    print("  Regenerating User Embeddings with all-MiniLM-L6-v2 (384-dim)")
    print("="*80 + "\n")
    
    # Load correct model (384-dim)
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    print(f"✓ Model loaded\n")
    
    # Get all users
    users = await execute_query(
        """
        SELECT u.user_id, u.email, ud.domain
        FROM users u
        JOIN user_domains ud ON u.user_id = ud.user_id
        """,
        fetch_all=True
    )
    
    print(f"Found {len(users)} users\n")
    
    for user in users:
        user_id = user['user_id']
        email = user['email']
        domain = user['domain']
        
        print(f"Processing: {email}")
        
        # Get interests
        interests = await execute_query(
            """
            SELECT interest_keyword, weight
            FROM user_interests
            WHERE user_id = $1
            """,
            user_id,
            fetch_all=True
        )
        
        if not interests:
            print(f"  ⊘ No interests found, skipping")
            continue
        
        keywords = [i['interest_keyword'] for i in interests]
        weights = [float(i['weight']) for i in interests]
        
        print(f"  Domain: {domain}")
        print(f"  Interests: {', '.join(keywords)}")
        
        # Create weighted text
        weighted_keywords = []
        for keyword, weight in zip(keywords, weights):
            repetitions = int(weight * 3)
            weighted_keywords.extend([keyword] * repetitions)
        
        # Create text representation
        text = f"{domain} research: " + " ".join(weighted_keywords)
        
        # Generate embedding
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Mean pooling (correct for sentence-transformers)
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = (sum_embeddings / sum_mask).squeeze()
        
        embedding_array = embedding.numpy()
        
        print(f"  ✓ Generated embedding: shape {embedding_array.shape}")
        
        # Store in database
        await execute_query(
            """
            INSERT INTO user_profile_embeddings 
                (user_id, embedding_vector, last_updated, interaction_count)
            VALUES ($1, $2, CURRENT_TIMESTAMP, 0)
            ON CONFLICT (user_id) 
            DO UPDATE SET 
                embedding_vector = $2,
                last_updated = CURRENT_TIMESTAMP
            """,
            user_id,
            embedding_array.tolist()
        )
        
        print(f"  ✓ Stored in database\n")
    
    # Verify all embeddings
    print("="*80)
    print("  Verification")
    print("="*80 + "\n")
    
    embeddings = await execute_query(
        """
        SELECT user_id, array_length(embedding_vector, 1) as dim
        FROM user_profile_embeddings
        """,
        fetch_all=True
    )
    
    print("User embeddings in database:")
    for emb in embeddings:
        print(f"  User {emb['user_id']}: {emb['dim']}-dimensional")
    
    # Check if all are 384
    all_384 = all(emb['dim'] == 384 for emb in embeddings)
    
    if all_384:
        print(f"\n✓ ALL embeddings are 384-dimensional (matches paper embeddings)")
    else:
        print(f"\n✗ Dimension mismatch detected!")
    
    print("\n" + "="*80)
    print("  Regeneration Complete")
    print("="*80 + "\n")
    
    print("Next step: python scripts/test_recommendations.py")
    print()


if __name__ == "__main__":
    asyncio.run(regenerate_user_embeddings_384dim())