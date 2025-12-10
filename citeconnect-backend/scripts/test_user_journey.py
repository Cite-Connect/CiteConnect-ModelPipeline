#!/usr/bin/env python3
"""
User Journey Simulation: Batch Processing
Simulates the Cold -> Warm -> Personalized journey for 30 users.
"""
import asyncio
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import requests
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.connection import DatabaseConnection
# Import the learning script
from scripts.batch_update_weights import update_user_weights

API_BASE = "http://localhost:8000/api/v1"
console = Console()

INTERACTION_PROFILES = {
    'engaged_user': [('click', 0.3, 3), ('save', 0.8, 4), ('like', 0.6, 2), ('download', 0.7, 1)]
}

def json_serial(obj):
    if isinstance(obj, datetime): return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

class UserJourneySimulator:
    def __init__(self):
        self.db: Optional[DatabaseConnection] = None
        self.session_id = f"simulation-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    async def initialize(self):
        self.db = DatabaseConnection()
        await self.db.connect()
        console.print("✅ Database connected", style="green")
    
    async def cleanup(self):
        if self.db: await self.db.disconnect()
        console.print("✅ Database disconnected", style="green")
    
    async def find_cold_start_users(self, limit: int = 30) -> List[int]:
        console.print(f"\n🔍 Finding up to {limit} cold-start users...", style="cyan")
        query = """
            SELECT u.user_id, u.email
            FROM users u
            JOIN user_recommendation_state s ON u.user_id = s.user_id
            WHERE s.recommendation_stage = 'cold_start' 
              AND s.interaction_count < 2
            LIMIT $1
        """
        results = await self.db.fetch(query, limit)
        if not results: 
            raise ValueError("No cold-start users available")
        
        user_ids = [r['user_id'] for r in results]
        console.print(f"✅ Found {len(user_ids)} users: {user_ids}", style="green")
        return user_ids
    
    async def get_user_snapshot(self, user_id: int) -> Dict:
        query = "SELECT * FROM user_recommendation_state WHERE user_id = $1"
        result = await self.db.fetchrow(query, user_id)
        return dict(result) if result else {}
    
    def get_recommendations(self, user_id: int) -> Dict:
        payload = {"user_id": user_id, "count": 10, "model_preference": "minilm", "session_id": self.session_id}
        # Increased timeout for local model loading
        response = requests.post(f"{API_BASE}/recommendations/test", json=payload, timeout=1200)
        if response.status_code != 200:
            raise Exception(f"API Error {response.status_code}: {response.text}")
        return response.json()
    
    async def track_interaction_direct(self, user_id: int, paper: Dict, interaction_type: str, position: int):
        """
        Insert interaction DIRECTLY to DB.
        """
        existing_breakdown = paper.get('score_breakdown') or {}
        
        def _get_val(source_dict, key, fallback=None):
            val = source_dict.get(key)
            if val is not None: return float(val)
            if fallback is not None: return float(fallback)
            return 0.5

        semantic_score = _get_val(existing_breakdown, "semantic")
        if semantic_score == 0.5:
             semantic_score = _get_val(paper, "relevance_score", 0.5)

        score_breakdown = {
            "semantic": semantic_score,
            "citation": _get_val(existing_breakdown, "citation"),
            "recency": _get_val(existing_breakdown, "recency"),
            "ground_truth": _get_val(existing_breakdown, "ground_truth"),
            "reading_level": _get_val(existing_breakdown, "reading_level"),
            "citation_network": _get_val(existing_breakdown, "citation_network", 0.0)
        }

        context = {
            "source": "recommendation",
            "position": position,
            "session_id": self.session_id,
            "score_breakdown": score_breakdown
        }
        
        query = """
            INSERT INTO user_interactions 
            (user_id, paper_id, interaction_type, context, created_at)
            VALUES ($1, $2, $3, $4, NOW())
        """
        
        await self.db.execute(
            query, 
            user_id, 
            paper['paper_id'], 
            interaction_type, 
            json.dumps(context)
        )
    
    async def simulate_interactions(self, user_id: int, papers: List[Dict], profile: str = 'engaged_user'):
        interaction_plan = INTERACTION_PROFILES.get(profile, INTERACTION_PROFILES['engaged_user'])
        
        interaction_num = 0
        # Simulating without progress bar for batch speed, or use simple print
        for interaction_type, strength, count in interaction_plan:
            for _ in range(count):
                if interaction_num >= len(papers): break
                paper = papers[interaction_num]
                
                await self.track_interaction_direct(user_id, paper, interaction_type, interaction_num + 1)
                
                interaction_num += 1
                await asyncio.sleep(0.05) # Tiny sleep to prevent DB lock contention
        
        console.print(f"   Generated {interaction_num} interactions for User {user_id}", style="dim")

    async def trigger_learning_loop(self):
        # Run batch update
        await update_user_weights(domain=None)

    async def run_simulation_for_user(self, user_id: int):
        console.print(f"\n🚀 Processing User ID: {user_id}", style="cyan bold")
        
        # 1. Initial State
        initial = await self.get_user_snapshot(user_id)
        if initial.get('recommendation_stage') != 'cold_start':
            console.print(f"   Skipping: User is already {initial.get('recommendation_stage')}", style="yellow")
            return

        # 2. Get Cold Start Recs
        try:
            cold_recs = self.get_recommendations(user_id)
        except Exception as e:
            console.print(f"   [red]Failed to get initial recommendations: {e}")
            return

        # 3. Simulate Interactions
        await self.simulate_interactions(user_id, cold_recs['recommendations'])
        
        # 4. Explicitly update interaction count
        await self.db.execute("UPDATE user_recommendation_state SET interaction_count = interaction_count + 10 WHERE user_id = $1", user_id)
        
        # 5. Trigger Learning (We do this per user loop to ensure sequential validity, 
        #    though in prod you'd do it once at the end. Here we want to verify individual updates.)
        await self.trigger_learning_loop()
        
        # 6. Verify Update
        final_state = await self.get_user_snapshot(user_id)
        weights = final_state.get('scoring_weights')
        
        if weights:
            if isinstance(weights, str): weights = json.loads(weights)
            # Just print the Semantic weight to keep output clean
            sem_weight = weights.get('semantic', 0)
            console.print(f"   ✅ Learned Weights! (Semantic: {sem_weight:.2f})", style="green")
        else:
            console.print("   ❌ Weights still NULL.", style="red")

async def main():
    simulator = UserJourneySimulator()
    try:
        await simulator.initialize()
        
        # 1. Find 30 Users
        user_ids = await simulator.find_cold_start_users(limit=30)
        
        # 2. Process each user
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("Simulating Users...", total=len(user_ids))
            
            for user_id in user_ids:
                await simulator.run_simulation_for_user(user_id)
                progress.update(task, advance=1)
                
        console.print("\n🎉 BATCH SIMULATION COMPLETE!", style="green bold")
        
    finally:
        await simulator.cleanup()

if __name__ == "__main__":
    asyncio.run(main())