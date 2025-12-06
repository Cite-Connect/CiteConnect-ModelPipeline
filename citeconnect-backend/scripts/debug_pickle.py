#!/usr/bin/env python3

"""
Debug Pickle File Structure

Inspects the embeddings_db.pkl file to understand its structure.
"""

import pickle
import numpy as np
from pathlib import Path

def debug_pickle():
    """Inspect pickle file structure"""
    
    pickle_path = Path("working_data/embeddings_db.pkl")
    
    if not pickle_path.exists():
        print(f"✗ Pickle file not found: {pickle_path}")
        return
    
    print(f"\n{'='*80}")
    print(f"  Debugging: {pickle_path}")
    print(f"{'='*80}\n")
    
    # Load pickle
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    # Inspect structure
    print(f"Type of data: {type(data)}")
    print(f"Length: {len(data) if hasattr(data, '__len__') else 'N/A'}")
    
    if isinstance(data, dict):
        print(f"\nDictionary keys: {list(data.keys())[:10]}")
        
        # Inspect 'chunks' key
        if 'chunks' in data:
            chunks = data['chunks']
            print(f"\n'chunks' type: {type(chunks)}")
            print(f"'chunks' length: {len(chunks)}")
            
            if len(chunks) > 0:
                first_chunk = chunks[0]
                print(f"\nFirst chunk type: {type(first_chunk)}")
                
                if isinstance(first_chunk, dict):
                    print(f"First chunk keys: {list(first_chunk.keys())}")
                    print(f"\nFirst chunk structure:")
                    for k, v in list(first_chunk.items())[:10]:
                        v_str = str(v)[:100] if not isinstance(v, np.ndarray) else f"ndarray{v.shape}"
                        print(f"  {k}: {type(v).__name__} - {v_str}")
        
        # Inspect 'embeddings' key
        if 'embeddings' in data:
            embeddings = data['embeddings']
            print(f"\n'embeddings' type: {type(embeddings)}")
            
            if isinstance(embeddings, np.ndarray):
                print(f"'embeddings' shape: {embeddings.shape}")
                print(f"'embeddings' dtype: {embeddings.dtype}")
            elif isinstance(embeddings, list):
                print(f"'embeddings' length: {len(embeddings)}")
                if len(embeddings) > 0:
                    print(f"First embedding type: {type(embeddings[0])}")
                    if isinstance(embeddings[0], np.ndarray):
                        print(f"First embedding shape: {embeddings[0].shape}")
        
    elif isinstance(data, list):
        print(f"\nList length: {len(data)}")
        
        if len(data) > 0:
            first_item = data[0]
            print(f"First item type: {type(first_item)}")
            
            if isinstance(first_item, dict):
                print(f"First item keys: {list(first_item.keys())}")
                print(f"\nFirst item structure:")
                for k, v in list(first_item.items())[:10]:
                    v_str = str(v)[:100] if not isinstance(v, np.ndarray) else f"ndarray{v.shape}"
                    print(f"  {k}: {type(v).__name__} - {v_str}")
                    
                # Check if embedding exists
                if 'embedding' in first_item:
                    emb = first_item['embedding']
                    print(f"\nEmbedding details:")
                    print(f"  Type: {type(emb)}")
                    if isinstance(emb, np.ndarray):
                        print(f"  Shape: {emb.shape}")
                        print(f"  Dtype: {emb.dtype}")
                    else:
                        print(f"  Value: {str(emb)[:200]}")
            
            elif isinstance(first_item, str):
                print(f"First item (string): {first_item[:200]}")
    
    else:
        print(f"\nUnexpected data type: {type(data)}")
        print(f"Data preview: {str(data)[:500]}")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    debug_pickle()