"""
Test script for search-augmented recommendations.
"""
import requests
import json
from pprint import pprint

API_BASE = "http://localhost:8000/api/v1"


def test_search_augmented():
    """Test search-augmented recommendations."""
    print("=" * 70)
    print("TEST 1: Search-Augmented Recommendations")
    print("=" * 70)
    
    payload = {
        "user_id": 2,
        "count": 5,
        "model_preference": "minilm",
        "search_query": "neural networks for brain segmentation",
        "session_id": "test-search-001"
    }
    
    response = requests.post(f"{API_BASE}/recommendations", json=payload)
    
    print(f"\nStatus: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        
        print(f"✅ Papers returned: {len(result['recommendations'])}")
        print(f"✅ Strategy: {result['metadata']['strategy_used']}")
        print(f"✅ Search query: {result['metadata'].get('search_query')}")
        print(f"✅ Time: {result['metadata']['generation_time_ms']:.0f}ms")
        
        print("\nTop 3 Papers:")
        for i, paper in enumerate(result['recommendations'][:3], 1):
            print(f"\n  {i}. {paper['title'][:70]}...")
            print(f"     Score: {paper.get('relevance_score', 'N/A')}")
            print(f"     Match: {paper.get('match_source', 'N/A')}")
            print(f"     Why: {paper.get('relevance_explanation', 'N/A')[:80]}...")
            
            # Show score breakdown if available
            if 'score_breakdown' in paper:
                breakdown = paper['score_breakdown']
                print(f"     Breakdown: K={breakdown.get('keyword', 0):.2f}, "
                      f"S={breakdown.get('semantic', 0):.2f}, "
                      f"P={breakdown.get('profile', 0):.2f}")
    else:
        print(f"❌ Error: {response.text}")


def test_without_search():
    """Test regular recommendations (no search)."""
    print("\n" + "=" * 70)
    print("TEST 2: Regular Recommendations (No Search)")
    print("=" * 70)
    
    payload = {
        "user_id": 2,
        "count": 5,
        "model_preference": "minilm",
        "session_id": "test-regular-001"
    }
    
    response = requests.post(f"{API_BASE}/recommendations", json=payload)
    
    print(f"\nStatus: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Papers returned: {len(result['recommendations'])}")
        print(f"✅ Strategy: {result['metadata']['strategy_used']}")
        print(f"✅ Has search query: {result['metadata'].get('search_query') is not None}")
    else:
        print(f"❌ Error: {response.text}")


def test_different_queries():
    """Test various search queries."""
    print("\n" + "=" * 70)
    print("TEST 3: Different Search Queries")
    print("=" * 70)
    
    test_queries = [
        "deep learning for medical diagnosis",
        "transformer models in NLP",
        "reinforcement learning robotics",
        "quantum computing algorithms"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        
        payload = {
            "user_id": 2,
            "count": 3,
            "model_preference": "minilm",
            "search_query": query,
            "session_id": f"test-{query[:10]}"
        }
        
        response = requests.post(f"{API_BASE}/recommendations", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"  ✅ Found {len(result['recommendations'])} papers")
            print(f"  Top: {result['recommendations'][0]['title'][:60]}...")
        else:
            print(f"  ❌ Failed: {response.status_code}")


if __name__ == "__main__":
    test_search_augmented()
    test_without_search()
    test_different_queries()