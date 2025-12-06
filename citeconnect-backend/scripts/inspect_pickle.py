#!/usr/bin/env python3

"""
Inspect Pickle File Structure

Comprehensive inspection of embeddings_db.pkl to verify:
- Chunk structure
- Embedding dimensions
- Citation/reference data
- Data completeness

Run: python scripts/inspect_pickle.py
"""

import pickle
import numpy as np
from pathlib import Path


def inspect_pickle():
    """Comprehensive inspection of pickle file"""
    
    pickle_path = Path("working_data/embeddings_db.pkl")
    
    if not pickle_path.exists():
        print(f"✗ Pickle file not found: {pickle_path}")
        return
    
    print(f"\n{'='*80}")
    print(f"  Inspecting: {pickle_path}")
    print(f"  File size: {pickle_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"{'='*80}\n")
    
    # Load pickle
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    # ==================== Top-Level Structure ====================
    print("1. TOP-LEVEL STRUCTURE")
    print("-" * 40)
    print(f"   Type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"   Keys: {list(data.keys())}")
        
        for key in data.keys():
            value = data[key]
            print(f"\n   '{key}':")
            print(f"     Type: {type(value)}")
            
            if isinstance(value, list):
                print(f"     Length: {len(value)}")
            elif isinstance(value, np.ndarray):
                print(f"     Shape: {value.shape}")
                print(f"     Dtype: {value.dtype}")
    
    # ==================== Chunks Analysis ====================
    if 'chunks' in data:
        chunks = data['chunks']
        
        print(f"\n\n2. CHUNKS ANALYSIS")
        print("-" * 40)
        print(f"   Total chunks: {len(chunks)}")
        
        if len(chunks) > 0:
            first_chunk = chunks[0]
            
            print(f"\n   First chunk structure:")
            print(f"   Type: {type(first_chunk)}")
            
            if isinstance(first_chunk, dict):
                print(f"\n   Available fields ({len(first_chunk.keys())} total):")
                
                for key in sorted(first_chunk.keys()):
                    value = first_chunk[key]
                    
                    # Format value for display
                    if isinstance(value, str):
                        if len(value) > 60:
                            display = f"'{value[:60]}...'"
                        else:
                            display = f"'{value}'"
                    elif isinstance(value, list):
                        display = f"list[{len(value)} items]"
                    elif isinstance(value, np.ndarray):
                        display = f"ndarray{value.shape}"
                    else:
                        display = str(value)
                    
                    print(f"     - {key:20s}: {type(value).__name__:10s} = {display}")
                
                # Check for citation fields specifically
                print(f"\n   Citation Data Check:")
                
                if 'references' in first_chunk:
                    refs = first_chunk['references']
                    print(f"     ✓ 'references' field exists")
                    print(f"       Type: {type(refs)}")
                    if isinstance(refs, list):
                        print(f"       Count: {len(refs)}")
                        if len(refs) > 0:
                            print(f"       Sample: {refs[:3]}")
                    else:
                        print(f"       Count: N/A")
                else:
                    print(f"     ✗ 'references' field MISSING")
                
                if 'citations' in first_chunk:
                    cites = first_chunk['citations']
                    print(f"     ✓ 'citations' field exists")
                    print(f"       Type: {type(cites)}")
                    if isinstance(cites, list):
                        print(f"       Count: {len(cites)}")
                        if len(cites) > 0:
                            print(f"       Sample: {cites[:3]}")
                    else:
                        print(f"       Count: N/A")
                else:
                    print(f"     ✗ 'citations' field MISSING")
        
        # Papers with citations analysis
        print(f"\n   Citation Coverage:")
        
        papers_with_refs = 0
        papers_with_cites = 0
        total_refs = 0
        total_cites = 0
        
        for chunk in chunks:
            if chunk.get('references'):
                papers_with_refs += 1
                if isinstance(chunk['references'], list):
                    total_refs += len(chunk['references'])
            
            if chunk.get('citations'):
                papers_with_cites += 1
                if isinstance(chunk['citations'], list):
                    total_cites += len(chunk['citations'])
        
        print(f"     Papers with 'references': {papers_with_refs}/{len(chunks)}")
        print(f"     Papers with 'citations': {papers_with_cites}/{len(chunks)}")
        
        if papers_with_refs > 0:
            print(f"     Avg references per paper: {total_refs / papers_with_refs:.1f}")
        
        if papers_with_cites > 0:
            print(f"     Avg citations per paper: {total_cites / papers_with_cites:.1f}")
    
    # ==================== Embeddings Analysis ====================
    if 'embeddings' in data:
        embeddings = data['embeddings']
        
        print(f"\n\n3. EMBEDDINGS ANALYSIS")
        print("-" * 40)
        print(f"   Type: {type(embeddings)}")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Dtype: {embeddings.dtype}")
        print(f"   Dimension: {embeddings.shape[1]}")
        
        # Check if matches chunks
        if 'chunks' in data:
            if len(data['chunks']) == embeddings.shape[0]:
                print(f"   ✓ Embeddings count matches chunks count")
            else:
                print(f"   ✗ MISMATCH: {len(data['chunks'])} chunks but {embeddings.shape[0]} embeddings")
        
        # Statistics
        print(f"\n   Embedding Statistics:")
        print(f"     Min value: {embeddings.min():.4f}")
        print(f"     Max value: {embeddings.max():.4f}")
        print(f"     Mean: {embeddings.mean():.4f}")
        print(f"     Std dev: {embeddings.std():.4f}")
    
    # ==================== Sample Papers ====================
    if 'chunks' in data and len(data['chunks']) > 0:
        print(f"\n\n4. SAMPLE PAPERS")
        print("-" * 40)
        
        # Get unique papers
        unique_papers = {}
        for chunk in data['chunks']:
            paper_id = chunk['paper_id']
            if paper_id not in unique_papers:
                refs = chunk.get('references', [])
                cites = chunk.get('citations', [])
                
                unique_papers[paper_id] = {
                    'title': chunk.get('paper_title', 'Unknown'),
                    'year': chunk.get('paper_year', 0),
                    'citations': chunk.get('citation_count', 0),
                    'has_references': bool(refs),
                    'ref_count': len(refs) if isinstance(refs, list) else 0,
                    'has_citations': bool(cites),
                    'cite_count': len(cites) if isinstance(cites, list) else 0
                }
        
        print(f"   Unique papers: {len(unique_papers)}\n")
        
        # Show first 5 papers
        for i, (paper_id, info) in enumerate(list(unique_papers.items())[:5], 1):
            title = info['title']
            if len(title) > 60:
                title = title[:60] + "..."
            
            print(f"   {i}. {title}")
            print(f"      Year: {info['year']} | Citation count: {info['citations']}")
            
            ref_status = '✓' if info['has_references'] else '✗'
            cite_status = '✓' if info['has_citations'] else '✗'
            
            print(f"      References: {ref_status} ({info['ref_count']} papers)")
            print(f"      Citations: {cite_status} ({info['cite_count']} papers)")
            print()
    
    # ==================== Readiness Check ====================
    print(f"\n5. EVALUATION READINESS CHECK")
    print("-" * 40)
    
    ready_for_eval = False
    
    if 'chunks' in data and len(data['chunks']) > 0:
        sample_chunk = data['chunks'][0]
        
        has_refs = 'references' in sample_chunk and sample_chunk.get('references')
        has_embeddings = 'embeddings' in data and len(data['embeddings']) > 0
        
        if has_refs and has_embeddings:
            print(f"   ✓ Ready for offline evaluation!")
            print(f"   ✓ Has references (ground truth)")
            print(f"   ✓ Has embeddings (for similarity)")
            ready_for_eval = True
        else:
            print(f"   ✗ NOT ready for offline evaluation")
            
            if not has_refs:
                print(f"   ✗ Missing 'references' field")
                print(f"      Action: Add 'references' to chunk_dict in DataPipeline")
            
            if not has_embeddings:
                print(f"   ✗ Missing embeddings")
    
    print(f"\n{'='*80}\n")
    
    if ready_for_eval:
        print("✓ PICKLE IS READY FOR EVALUATION")
        print("\nNext step: python scripts/offline_evaluation.py")
    else:
        print("⚠ PICKLE NEEDS UPDATES")
        print("\nNext step: Modify DataPipeline to add 'references' field")
    
    print()


if __name__ == "__main__":
    inspect_pickle()