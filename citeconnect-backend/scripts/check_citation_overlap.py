#!/usr/bin/env python3

"""
Check Citation Overlap

Diagnostic script to see if cited papers are in our database.

Run: python scripts/check_citation_overlap.py
"""

import pickle
from pathlib import Path


def check_overlap():
    """Check if bibliography papers exist in database"""
    
    pickle_path = Path("working_data/embeddings_db.pkl")
    
    print("\n" + "="*80)
    print("  Citation Overlap Analysis")
    print("="*80 + "\n")
    
    # Load pickle
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    chunks = data['chunks']
    
    # Get all paper IDs in database
    db_paper_ids = set()
    paper_references = {}
    
    for chunk in chunks:
        paper_id = chunk['paper_id']
        db_paper_ids.add(paper_id)
        
        # Store references
        if paper_id not in paper_references:
            refs = chunk.get('references', [])
            if isinstance(refs, list):
                paper_references[paper_id] = set(refs)
            else:
                paper_references[paper_id] = set()
    
    print(f"Papers in database: {len(db_paper_ids)}")
    print(f"Papers with bibliographies: {sum(1 for refs in paper_references.values() if refs)}\n")
    
    # Check overlap for each paper
    print("Overlap Analysis:\n")
    
    total_refs = 0
    total_in_db = 0
    
    for paper_id, refs in paper_references.items():
        if not refs:
            continue
        
        # Check how many cited papers are in our database
        cited_in_db = refs.intersection(db_paper_ids)
        
        # Get paper title
        paper_title = None
        for chunk in chunks:
            if chunk['paper_id'] == paper_id:
                paper_title = chunk.get('paper_title', 'Unknown')
                break
        
        if len(paper_title) > 60:
            paper_title = paper_title[:60] + "..."
        
        overlap_pct = len(cited_in_db) / len(refs) * 100 if refs else 0
        
        print(f"Paper: {paper_title}")
        print(f"  Bibliography size: {len(refs)}")
        print(f"  Cited papers in DB: {len(cited_in_db)} ({overlap_pct:.1f}%)")
        
        if cited_in_db:
            print(f"  Found papers: {list(cited_in_db)[:3]}...")
        else:
            print(f"  ✗ NO OVERLAP - can't evaluate this paper")
        
        print()
        
        total_refs += len(refs)
        total_in_db += len(cited_in_db)
    
    # Overall statistics
    print("="*80)
    print("  Overall Statistics")
    print("="*80 + "\n")
    
    overall_overlap = total_in_db / total_refs * 100 if total_refs > 0 else 0
    
    print(f"Total bibliography references: {total_refs}")
    print(f"References found in database: {total_in_db} ({overall_overlap:.1f}%)")
    
    print(f"\n{'='*80}\n")
    
    if overall_overlap < 5:
        print("⚠️  PROBLEM IDENTIFIED:")
        print(f"   Only {overall_overlap:.1f}% of cited papers are in your database!")
        print()
        print("   Why metrics are 0.000:")
        print("   - Papers cite external work not in your 10-paper database")
        print("   - Can't find cited papers in recommendations")
        print("   - Results in 0 hits → 0 precision/recall/MRR")
        print()
        print("   Solutions:")
        print("   1. Ingest MORE papers (100+ papers from same domain)")
        print("   2. Use synthetic ground truth (create_ground_truth.py)")
        print("   3. Explain at submission: 'Limited by demo dataset size'")
    else:
        print(f"✓ Good overlap ({overall_overlap:.1f}%)")
        print(f"  Evaluation should work!")
    
    print()


if __name__ == "__main__":
    check_overlap()