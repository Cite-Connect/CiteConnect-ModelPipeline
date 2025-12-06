"""
Integration test for Recommendation API.
"""
import requests
import json
import time
import uuid # Added to generate session IDs

# Configuration
BASE_URL = "http://localhost:8000" 
API_PREFIX = "/api/v1/recommendations" 
USER_ID = 2

def print_section(title):
    print(f"\n{'='*60}")
    print(f"TESTING: {title}")
    print(f"{'='*60}")

def test_generate_recommendations():
    print_section("Generate Recommendations (POST /)")
    
    url = f"{BASE_URL}{API_PREFIX}"
    payload = {
        "user_id": USER_ID,
        "count": 5,
        # CRITICAL FIX: Use the exact string expected by the API validation
        "model_preference": "minilm", 
        # CRITICAL FIX: Add session_id (required by API)
        "session_id": str(uuid.uuid4()),
        "filters": {
            "min_year": 2020
        }
    }
    
    try:
        print(f"Requesting: {url}")
        response = requests.post(url, json=payload)
        
        if response.status_code != 200:
            print(f"❌ Error Response: {response.text}")
            
        response.raise_for_status()
        
        data = response.json()
        recommendations = data.get('recommendations', [])
        metadata = data.get('metadata', {})
        
        print(f"✅ Status: {response.status_code}")
        print(f"✅ Strategy Used: {metadata.get('strategy_used')}")
        print(f"✅ Generation Time: {metadata.get('generation_time_ms')}ms")
        print(f"✅ Papers Returned: {len(recommendations)}")
        
        if recommendations:
            print("\nSample Paper:")
            print(f"   - Title: {recommendations[0].get('title')}")
            print(f"   - Explanation: {recommendations[0].get('relevance_explanation')}")
            
        return [p['paper_id'] for p in recommendations]
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return []

def test_evaluate_endpoint(paper_ids):
    print_section("Evaluate Recommendations (POST /evaluate)")
    
    if not paper_ids:
        print("⚠️ Skipping evaluation (no papers)")
        return

    url = f"{BASE_URL}{API_PREFIX}/evaluate"
    params = {"user_id": USER_ID}
    
    try:
        print(f"Requesting: {url} for {len(paper_ids)} papers")
        response = requests.post(url, params=params, json=paper_ids)
        response.raise_for_status()
        
        data = response.json()
        metrics = data.get('metrics', {})
        print(f"✅ Combined Score: {metrics.get('combined_score')}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        if 'response' in locals(): print(f"Response: {response.text}")

def test_history_endpoint():
    print_section("Recommendation History (GET /{id}/history)")
    
    url = f"{BASE_URL}{API_PREFIX}/{USER_ID}/history"
    
    try:
        print(f"Requesting: {url}")
        response = requests.get(url, params={"limit": 5})
        
        if response.status_code != 200:
            print(f"❌ Error Response: {response.text}")

        response.raise_for_status()
        data = response.json()
        history = data.get('history', [])
        
        print(f"✅ Status: {response.status_code}")
        print(f"✅ History Items: {len(history)}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    print(f"Targeting API at: {BASE_URL}{API_PREFIX}")
    
    # 1. Generate
    paper_ids = test_generate_recommendations()
    
    # 2. Evaluate
    if paper_ids:
        test_evaluate_endpoint(paper_ids)
        
    # 3. History
    time.sleep(1) 
    test_history_endpoint()