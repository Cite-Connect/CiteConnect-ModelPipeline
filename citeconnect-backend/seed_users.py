#!/usr/bin/env python3

"""
Seed Users Script

Creates test users in the database for development and testing.
Populates:
- users table
- user_domains table
- user_interests table
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.security import hash_password
from app.db.postgres import execute_query

async def seed_users():
    """Create test users in database."""
    
    print("\n" + "="*60)
    print("  Seeding CiteConnect with Test Users")
    print("="*60 + "\n")
    
    # Define test users
    test_users = [
        {
            "email": "sarah.chen@example.com",
            "password": "Password123!",
            "name": "Sarah Chen",
            "domain": "healthcare",
            "interests": ["NLP", "clinical trials", "drug discovery", "protein folding"],
            "google_scholar_url": None
        },
        {
            "email": "john.smith@example.com",
            "password": "Password123!",
            "name": "John Smith",
            "domain": "fintech",
            "interests": ["fraud detection", "algorithmic trading", "risk management"],
            "google_scholar_url": None
        },
        {
            "email": "maria.garcia@example.com",
            "password": "Password123!",
            "name": "Maria Garcia",
            "domain": "quantum_computing",
            "interests": ["quantum algorithms", "error correction", "quantum machine learning"],
            "google_scholar_url": None
        },
        {
            "email": "test@example.com",
            "password": "Password123!",
            "name": "Test User",
            "domain": "healthcare",
            "interests": ["machine learning", "deep learning"],
            "google_scholar_url": None
        }
    ]
    
    created_count = 0
    skipped_count = 0
    
    for user_data in test_users:
        try:
            email = user_data['email']
            
            # Check if user already exists
            existing = await execute_query(
                "SELECT user_id FROM users WHERE email = $1",
                email,
                fetch_one=True
            )
            
            if existing:
                print(f"⊘ Skipped: {email} (already exists)")
                skipped_count += 1
                continue
            
            # Hash password
            password_hash = hash_password(user_data['password'])
            
            # Insert user
            user_result = await execute_query(
                """
                INSERT INTO users (email, password_hash, name, google_scholar_url, created_at)
                VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
                RETURNING user_id
                """,
                email,
                password_hash,
                user_data['name'],
                user_data['google_scholar_url'],
                fetch_one=True
            )
            
            user_id = user_result['user_id']
            
            # Insert domain
            await execute_query(
                """
                INSERT INTO user_domains (user_id, domain, selected_at)
                VALUES ($1, $2, CURRENT_TIMESTAMP)
                """,
                user_id,
                user_data['domain']
            )
            
            # Insert interests
            for interest in user_data['interests']:
                await execute_query(
                    """
                    INSERT INTO user_interests (user_id, interest_keyword, source, weight)
                    VALUES ($1, $2, 'manual', 1.0)
                    """,
                    user_id,
                    interest
                )
            
            print(f"✓ Created: {email}")
            print(f"   - User ID: {user_id}")
            print(f"   - Domain: {user_data['domain']}")
            print(f"   - Interests: {len(user_data['interests'])}")
            
            created_count += 1
            
        except Exception as e:
            print(f"✗ Failed: {email} - {str(e)}")
    
    print("\n" + "="*60)
    print(f"  Seeding Complete")
    print("="*60)
    print(f"\nCreated: {created_count} users")
    print(f"Skipped: {skipped_count} users (already exist)")
    print(f"\nTest Credentials:")
    print(f"  Email: sarah.chen@example.com")
    print(f"  Password: Password123!")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(seed_users())
