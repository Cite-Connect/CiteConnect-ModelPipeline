#!/usr/bin/env python3
"""
User Journey Simulation: Cold-Start → Warm-Start → Personalized Weights
Simulates a complete user lifecycle including the learning loop.

This script:
1. Finds a cold-start user (0 interactions)
2. Gets initial profile-based recommendations (Default Weights)
3. Simulates realistic interactions (clicks, saves, likes)
   -> CRITICAL: Injects 'score_breakdown' into logs so learning works immediately.
4. Waits for stage transition (cold → warm)
5. TRIGGERS LEARNING LOOP (Batch Weight Update)
6. Gets new personalized recommendations (Personalized Weights)
7. Compares before/after to show system evolution

Usage:
    python scripts/simulate_user_journey.py --auto
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
from rich import print as rprint

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.connection import DatabaseConnection
# Import the learning script directly
from scripts.batch_update_weights import update_user_weights

# Configuration
API_BASE = "http://localhost:8000/api/v1"
console = Console()

# Interaction types
INTERACTION_PROFILES = {
    'engaged_user': [
        ('click', 0.3, 3),   # (type, strength, count)
        ('save', 0.8, 4),
        ('like', 0.6, 2),
        ('download', 0.7, 1)
    ]
}

def json_serial(obj):
    """JSON serializer for objects not serializable by default."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


class UserJourneySimulator:
    """Simulates complete user journey from cold-start to personalized warm-start."""
    
    def __init__(self, user_id: Optional[int] = None, api_base: str = API_BASE):
        self.user_id = user_id
        self.api_base = api_base
        self.db: Optional[DatabaseConnection] = None
        self.session_id = f"simulation-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Results storage
        self.cold_start_recs = None
        self.warm_start_recs = None
        self.interactions = []
        self.database_snapshots = []
    
    async def initialize(self):
        """Initialize database connection."""
        self.db = DatabaseConnection()
        await self.db.connect()
        console.print("✅ Database connected", style="green")
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.db:
            await self.db.disconnect()
            console.print("✅ Database disconnected", style="green")
    
    async def find_cold_start_user(self) -> int:
        """Find a user in cold-start stage."""
        console.print("\n🔍 Finding cold-start user...", style="cyan")
        
        query = """
            SELECT u.user_id, u.email, s.interaction_count, 
                   p.primary_domain, p.research_stage
            FROM users u
            JOIN user_recommendation_state s ON u.user_id = s.user_id
            JOIN user_profiles_extended p ON u.user_id = p.user_id
            WHERE s.recommendation_stage = 'cold_start'
              AND s.interaction_count < 2
            ORDER BY u.user_id
            LIMIT 5
        """
        
        results = await self.db.fetch(query)
        
        if not results:
            console.print("❌ No cold-start users found", style="red")
            raise ValueError("No cold-start users available")
        
        # Pick first user
        selected_user = results[0]['user_id']
        console.print(f"✅ Selected User ID: {selected_user} ({results[0]['email']})", style="green bold")
        
        return selected_user
    
    async def get_user_snapshot(self) -> Dict:
        """Get current user state from database."""
        query = """
            SELECT 
                u.user_id,
                u.email as username,
                s.interaction_count,
                s.recommendation_stage,
                s.scoring_weights, -- Capture weights to show evolution
                s.last_retrained_at,
                p.primary_domain
            FROM users u
            JOIN user_recommendation_state s ON u.user_id = s.user_id
            JOIN user_profiles_extended p ON u.user_id = p.user_id
            WHERE u.user_id = $1
        """
        
        result = await self.db.fetchrow(query, self.user_id)
        if result:
            res = dict(result)
            # Parse weights if they are a string
            if res.get('scoring_weights') and isinstance(res['scoring_weights'], str):
                res['scoring_weights'] = json.loads(res['scoring_weights'])
            return res
        return {}
    
    def get_recommendations(self) -> Dict:
        """Get recommendations from API."""
        payload = {
            "user_id": self.user_id,
            "count": 10,
            "model_preference": "minilm",
            "session_id": self.session_id
        }
        
        response = requests.post(
            f"{self.api_base}/recommendations/test",
            json=payload,
            timeout=300
        )
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} - {response.text}")
        
        return response.json()
    
    def track_interaction(self, paper: Dict, interaction_type: str, position: int) -> Dict:
        """
        Track interaction with context.
        CRITICAL: Manually inject score_breakdown if missing, so Batch Update script works.
        """
        
        # 1. Extract existing breakdown or create a synthetic one
        existing_breakdown = paper.get('score_breakdown') or {}
        
        # Ensure we have valid float values for the Batch Script to average
        score_breakdown = {
            "semantic": float(existing_breakdown.get("semantic", paper.get("relevance_score", 0.5))),
            "citation": float(existing_breakdown.get("citation", 0.5)),
            "recency": float(existing_breakdown.get("recency", 0.5)),
            "ground_truth": float(existing_breakdown.get("ground_truth", 0.5)),
            "reading_level": float(existing_breakdown.get("reading_level", 0.5)),
            "citation_network": float(existing_breakdown.get("citation_network", 0.0))
        }

        payload = {
            "paper_id": paper['paper_id'],
            "interaction_type": interaction_type,
            "context": {
                "source": "recommendation",
                "position": position,
                "session_id": self.session_id,
                # MANUAL INJECTION: This allows the learning script to read the data
                "score_breakdown": score_breakdown
            }
        }
        
        response = requests.post(
            f"{self.api_base}/interactions",
            params={"user_id": self.user_id},
            json=payload,
            timeout=30
        )
        
        return response.json()
    
    async def simulate_interactions(self, papers: List[Dict], profile: str = 'engaged_user'):
        """Simulate interactions."""
        interaction_plan = INTERACTION_PROFILES.get(profile, INTERACTION_PROFILES['engaged_user'])
        console.print(f"\n📝 Simulating interactions...", style="cyan")
        
        interaction_num = 0
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("Tracking interactions...", total=10)
            
            for interaction_type, strength, count in interaction_plan:
                for _ in range(count):
                    if interaction_num >= len(papers): break
                    
                    paper = papers[interaction_num]
                    # Pass the whole paper object so we can extract scores
                    self.track_interaction(paper, interaction_type, interaction_num + 1)
                    
                    self.interactions.append({
                        'paper_id': paper['paper_id'],
                        'type': interaction_type,
                        'strength': strength
                    })
                    
                    progress.update(task, advance=1, description=f"[green]Interaction: {interaction_type.upper()}")
                    interaction_num += 1
                    await asyncio.sleep(0.2) 
        
        console.print(f"✅ Generated {len(self.interactions)} interactions", style="green")

    async def trigger_learning_loop(self):
        """
        NEW STEP: Run the batch weight update script to learn from interactions.
        This populates the scoring_weights column in the DB.
        """
        console.print("\n🧠 STEP 3.5: Triggering Learning Loop (Batch Weight Update)...", style="magenta bold")
        
        with console.status("[bold magenta]Calculating personalized weights..."):
            # Run the imported update function
            # We pass None for domain to process all users (specifically our test user)
            await update_user_weights(domain=None)
            
        console.print("✅ Learning Loop Complete. User profile updated in DB.", style="magenta")

    async def run_simulation(self, interaction_profile: str = 'engaged_user'):
        console.print("\n" + "="*70, style="cyan bold")
        console.print("  USER JOURNEY SIMULATION: Cold → Warm → Personalized", style="cyan bold")
        console.print("="*70 + "\n", style="cyan bold")
        
        # 1. Setup
        if self.user_id is None:
            self.user_id = await self.find_cold_start_user()
        
        # 2. Initial State
        initial = await self.get_user_snapshot()
        console.print(f"\n📸 Initial State: {initial.get('recommendation_stage')} | Weights: {initial.get('scoring_weights') or 'Default (NULL)'}")
        
        # 3. Cold Start Recs
        console.print("\n🎯 STEP 1: Getting Cold-Start Recommendations", style="cyan bold")
        self.cold_start_recs = self.get_recommendations()
        console.print(f"✅ Received {len(self.cold_start_recs['recommendations'])} papers")
        
        # 4. Interact
        console.print(f"\n🎬 STEP 2: Simulating Interactions", style="cyan bold")
        await self.simulate_interactions(self.cold_start_recs['recommendations'], interaction_profile)
        
        # 5. Transition Check
        await asyncio.sleep(1)
        mid_state = await self.get_user_snapshot()
        console.print(f"\n📊 Post-Interaction State: {mid_state.get('recommendation_stage')} (Count: {mid_state.get('interaction_count')})")
        
        # 6. TRIGGER LEARNING (The New Part)
        # This replaces NULL weights with calculated JSON
        await self.trigger_learning_loop()
        
        # 7. Check New Weights
        final_state = await self.get_user_snapshot()
        weights = final_state.get('scoring_weights')
        if weights:
            console.print(f"\n✨ PERSONALIZED WEIGHTS LEARNED:", style="green bold")
            console.print(json.dumps(weights, indent=2), style="green")
        else:
            console.print("\n❌ Weights not updated. Check if interactions were logged correctly.", style="red")

        # 8. Warm Start Recs (Personalized)
        console.print("\n🎯 STEP 4: Getting Personalized Warm-Start Recommendations", style="cyan bold")
        self.warm_start_recs = self.get_recommendations()
        
        # 9. Comparison
        self.compare_recommendations()
        self.generate_report()

    def compare_recommendations(self):
        """Compare cold vs personalized."""
        console.print("\n📊 EVOLUTION ANALYSIS", style="cyan bold")
        
        cold_ids = [p['paper_id'] for p in self.cold_start_recs['recommendations'][:10]]
        warm_ids = [p['paper_id'] for p in self.warm_start_recs['recommendations'][:10]]
        
        overlap = set(cold_ids) & set(warm_ids)
        if len(set(cold_ids) | set(warm_ids)) > 0:
            jaccard = len(overlap) / len(set(cold_ids) | set(warm_ids))
        else:
            jaccard = 0.0
        
        console.print(f"  • Overlap Count: {len(overlap)}/10")
        console.print(f"  • Jaccard Similarity: {jaccard:.2f}")
        
        if jaccard < 0.5:
            console.print("✅ System successfully pivoted based on user interests!", style="green")
        else:
            console.print("⚠️  Recommendations didn't change much. User might have confirmed initial bias.", style="yellow")

    def generate_report(self):
        # Save a simple JSON report
        report = {
            'user_id': self.user_id,
            'timestamp': datetime.now().isoformat(),
            'weights': self.warm_start_recs.get('metadata', {}).get('scoring_weights')
        }
        Path(f"simulation_report_{self.user_id}.json").write_text(json.dumps(report, indent=2))
        console.print(f"\n💾 Report saved.", style="green")

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user-id', type=int)
    parser.add_argument('--auto', action='store_true')
    args = parser.parse_args()
    
    sim = UserJourneySimulator(user_id=args.user_id)
    try:
        await sim.initialize()
        await sim.run_simulation()
    finally:
        await sim.cleanup()

if __name__ == "__main__":
    asyncio.run(main())