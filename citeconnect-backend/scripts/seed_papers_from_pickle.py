#!/usr/bin/env python3

"""
Seed Papers Table from Pickle

Simple script to populate PostgreSQL papers table from embeddings_db.pkl.

This is required before creating user interactions (foreign key constraint).

Run: python scripts/seed_papers_from_pickle.py
"""

import asyncio
import sys
from pathlib import Path
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.postgres import execute_query


async def seed_papers():
    """Load papers from pickle and insert into PostgreSQL"""
    
    print("\n" + "="*80)
    print("  Seeding Papers Table from Pickle")
    print("="*80 + "\n")
    
    # Load pickle
    pickle_path = Path("working_data/embeddings_db.pkl")
    
    print(f"1. Loading pickle: {pickle_path}")
    
    if not pickle_path.exists():
        print(f"   ✗ File not found: {pickle_path.absolute()}")
        return
    
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        print(f"   ✓ Loaded successfully")
    except Exception as e:
        print(f"   ✗ Failed to load: {e}")
        return
    
    chunks = data.get('chunks', [])
    print(f"   ✓ Found {len(chunks)} chunks\n")
    
    # Extract unique papers
    print(f"2. Extracting unique papers...")
    
    unique_papers = {}
    for chunk in chunks:
        paper_id = chunk.get('paper_id')
        
        if not paper_id:
            continue
        
        if paper_id not in unique_papers:
            unique_papers[paper_id] = {
                'paper_id': paper_id,
                'title': chunk.get('paper_title', 'Unknown Title'),
                'year': chunk.get('paper_year', 0),
                'citation_count': chunk.get('citation_count', 0),
                'abstract': chunk.get('text', ''),
                'domain': 'healthcare'  # Default for demo
            }
    
    print(f"   ✓ Found {len(unique_papers)} unique papers\n")
    
    # Show papers that will be inserted
    print(f"3. Papers to insert:")
    for i, (paper_id, data) in enumerate(list(unique_papers.items())[:5], 1):
        title = data['title']
        if len(title) > 60:
            title = title[:60] + "..."
        print(f"   {i}. {paper_id[:20]}... | {title}")
    
    if len(unique_papers) > 5:
        print(f"   ... and {len(unique_papers) - 5} more\n")
    else:
        print()
    
    # Insert into database
    print(f"4. Inserting into PostgreSQL papers table...\n")
    
    inserted = 0
    skipped = 0
    failed = 0
    
    for paper_id, paper_data in unique_papers.items():
        try:
            # Check if exists
            existing = await execute_query(
                "SELECT paper_id FROM papers WHERE paper_id = $1",
                paper_id,
                fetch_one=True
            )
            
            if existing:
                skipped += 1
                continue
            
            # Insert
            await execute_query(
                """
                INSERT INTO papers (
                    paper_id, 
                    title, 
                    year, 
                    citation_count, 
                    abstract, 
                    domain, 
                    ingested_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, CURRENT_TIMESTAMP)
                """,
                paper_data['paper_id'],
                paper_data['title'],
                paper_data['year'],
                paper_data['citation_count'],
                paper_data['abstract'],
                paper_data['domain']
            )
            
            title = paper_data['title']
            if len(title) > 60:
                title = title[:60] + "..."
            print(f"   ✓ {inserted + 1}. {title}")
            
            inserted += 1
            
        except Exception as e:
            print(f"   ✗ Failed {paper_id}: {str(e)[:100]}")
            failed += 1
    
    # Summary
    print(f"\n{'='*80}")
    print(f"  Results")
    print(f"{'='*80}")
    print(f"\n  Inserted: {inserted} papers")
    print(f"  Skipped: {skipped} papers (already existed)")
    print(f"  Failed: {failed} papers")
    print(f"  Total in database: {inserted + skipped}")
    
    # Verify
    print(f"\n5. Verifying papers table...")
    
    try:
        count_result = await execute_query(
            "SELECT COUNT(*) as count FROM papers",
            fetch_one=True
        )
        
        total_count = count_result['count']
        print(f"   ✓ Papers in database: {total_count}")
        
        # Show sample
        sample = await execute_query(
            "SELECT paper_id, title, year FROM papers LIMIT 3",
            fetch_all=True
        )
        
        print(f"\n   Sample papers:")
        for paper in sample:
            title = paper['title']
            if len(title) > 60:
                title = title[:60] + "..."
            print(f"     - {paper['paper_id'][:20]}... | {title} ({paper['year']})")
        
        print(f"\n{'='*80}")
        
        if total_count > 0:
            print(f"✓ SUCCESS: {total_count} papers ready in database")
            print(f"\nNext step: python scripts/create_ground_truth.py")
        else:
            print(f"✗ WARNING: Papers table is still empty!")
        
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"   ✗ Failed to verify: {e}")
        return False
    
    return total_count > 0


if __name__ == "__main__":
    asyncio.run(seed_papers())