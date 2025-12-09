#!/usr/bin/env python3
"""
User Journey Simulation: Cold-Start → Warm-Start
Simulates a complete user lifecycle with interactions and tracks evolution.

This script:
1. Finds a cold-start user (0-9 interactions)
2. Gets initial profile-based recommendations
3. Simulates realistic interactions (clicks, saves, likes)
4. Tracks database changes in real-time
5. Waits for stage transition (cold → early)
6. Gets new behavior-based recommendations
7. Compares before/after to show system evolution

Usage:
    python scripts/simulate_user_journey.py --user-id 3
    python scripts/simulate_user_journey.py --auto  # Picks user automatically
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
from datetime import datetime
import json
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.connection import DatabaseConnection
from app.config import settings

# Configuration
API_BASE = "http://localhost:8000/api/v1"
console = Console()

# Interaction types with weights and frequencies
INTERACTION_PROFILES = {
    'engaged_user': [
        ('click', 0.3, 3),   # (type, strength, count)
        ('save', 0.8, 4),
        ('like', 0.6, 2),
        ('download', 0.7, 1)
    ],
    'casual_user': [
        ('click', 0.3, 6),
        ('view', 0.2, 3),
        ('save', 0.8, 1)
    ],
    'researcher': [
        ('save', 0.8, 5),
        ('download', 0.7, 3),
        ('cite', 1.0, 2)
    ]
}
def json_serial(obj):
    """JSON serializer for objects not serializable by default."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


class UserJourneySimulator:
    """Simulates complete user journey from cold-start to warm-start."""
    
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
              AND s.interaction_count < 5
            ORDER BY u.user_id
            LIMIT 5
        """
        
        results = await self.db.fetch(query)
        
        if not results:
            console.print("❌ No cold-start users found", style="red")
            raise ValueError("No cold-start users available")
        
        # Display available users
        table = Table(title="Available Cold-Start Users")
        table.add_column("User ID", style="cyan")
        table.add_column("email", style="green")
        table.add_column("Interactions", style="yellow")
        table.add_column("Domain", style="magenta")
        table.add_column("Stage", style="blue")
        
        for user in results:
            table.add_row(
                str(user['user_id']),
                user['email'],
                str(user['interaction_count']),
                user['primary_domain'],
                user['research_stage']
            )
        
        console.print(table)
        
        # Pick first user
        selected_user = results[0]['user_id']
        console.print(f"\n✅ Selected User ID: {selected_user}", style="green bold")
        
        return selected_user
    
    async def get_user_snapshot(self) -> Dict:
        """Get current user state from database."""
        query = """
            SELECT 
                u.user_id,
                u.email as username,
                s.interaction_count,
                s.recommendation_stage,
                s.last_embedding_update_minilm as last_embedding_update,
                p.primary_domain,
                array_agg(ui.interest_term) as interests
            FROM users u
            JOIN user_recommendation_state s ON u.user_id = s.user_id
            JOIN user_profiles_extended p ON u.user_id = p.user_id
            LEFT JOIN user_interest_hierarchy ui ON u.user_id = ui.user_id
            WHERE u.user_id = $1
            GROUP BY u.user_id, username, s.interaction_count, 
                     s.recommendation_stage, last_embedding_update, p.primary_domain
        """
        
        result = await self.db.fetchrow(query, self.user_id)
        return dict(result) if result else {}
    
    def get_recommendations(self, search_query: Optional[str] = None) -> Dict:
        """Get recommendations from API."""
        payload = {
            "user_id": self.user_id,
            "count": 10,
            "model_preference": "minilm",
            "session_id": self.session_id
        }
        
        if search_query:
            payload["search_query"] = search_query
        
        response = requests.post(
            f"{self.api_base}/recommendations",
            json=payload,
            timeout=300  # 5 minute timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} - {response.text}")
        
        return response.json()
    
    def track_interaction(
        self,
        paper_id: str,
        interaction_type: str,
        position: int,
        duration_seconds: Optional[int] = None
    ) -> Dict:
        """Track a single interaction."""
        payload = {
            "paper_id": paper_id,
            "interaction_type": interaction_type,
            "duration_seconds": duration_seconds,
            "context": {
                "source": "recommendation",
                "position": position,
                "session_id": self.session_id
            }
        }
        
        response = requests.post(
            f"{self.api_base}/interactions",
            params={"user_id": self.user_id},
            json=payload,
            timeout=30
        )
        
        if response.status_code not in [200, 201]:
            raise Exception(f"Interaction failed: {response.status_code} - {response.text}")
        
        return response.json()
    
    async def simulate_interactions(
        self,
        papers: List[Dict],
        profile: str = 'engaged_user'
    ):
        """Simulate realistic user interactions."""
        interaction_plan = INTERACTION_PROFILES.get(profile, INTERACTION_PROFILES['engaged_user'])
        
        console.print(f"\n📝 Simulating {profile} behavior pattern...", style="cyan")
        
        interaction_num = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Tracking interactions...", total=10)
            
            for interaction_type, strength, count in interaction_plan:
                for _ in range(count):
                    if interaction_num >= 10:
                        break
                    
                    if interaction_num >= len(papers):
                        break
                    
                    paper = papers[interaction_num]
                    
                    # Simulate realistic duration
                    duration = None
                    if interaction_type in ['click', 'view']:
                        duration = 30 + (interaction_num * 10)
                    elif interaction_type in ['save', 'download']:
                        duration = 120
                    
                    # Track interaction
                    result = self.track_interaction(
                        paper_id=paper['paper_id'],
                        interaction_type=interaction_type,
                        position=interaction_num + 1,
                        duration_seconds=duration
                    )
                    
                    self.interactions.append({
                        'number': interaction_num + 1,
                        'paper_id': paper['paper_id'],
                        'paper_title': paper['title'][:60] + '...',
                        'type': interaction_type,
                        'strength': strength,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    progress.update(
                        task,
                        advance=1,
                        description=f"[green]✅ Interaction {interaction_num + 1}/10: {interaction_type.upper()} - {paper['title'][:40]}..."
                    )
                    
                    interaction_num += 1
                    await asyncio.sleep(0.5)  # Realistic delay
                    
                    if interaction_num >= 10:
                        break
        
        console.print(f"\n✅ Completed {len(self.interactions)} interactions", style="green bold")
    
    async def check_stage_transition(self) -> Dict:
        """Check if user transitioned to new stage."""
        snapshot = await self.get_user_snapshot()
        
        console.print("\n📊 User State After Interactions:", style="cyan")
        
        table = Table(show_header=True)
        table.add_column("Attribute", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("User ID", str(snapshot.get('user_id')))
        table.add_row("Username", snapshot.get('username', 'N/A'))
        table.add_row("Interaction Count", str(snapshot.get('interaction_count')))
        table.add_row("Stage", snapshot.get('recommendation_stage', 'N/A'))
        table.add_row("Domain", snapshot.get('primary_domain', 'N/A'))
        table.add_row("Last Embedding Update", str(snapshot.get('last_embedding_update', 'Never'))[:19])
        
        console.print(table)
        
        return snapshot
    
    def compare_recommendations(self):
        """Compare cold-start vs warm-start recommendations."""
        console.print("\n📊 RECOMMENDATION COMPARISON", style="cyan bold")
        
        # Create comparison table
        table = Table(title="Cold-Start vs Warm-Start Recommendations")
        table.add_column("Rank", style="cyan", width=5)
        table.add_column("Cold-Start (Profile-Based)", style="yellow", width=50)
        table.add_column("Warm-Start (Behavior-Based)", style="green", width=50)
        
        cold_papers = self.cold_start_recs['recommendations'][:5]
        warm_papers = self.warm_start_recs['recommendations'][:5]
        
        for i in range(5):
            cold_title = cold_papers[i]['title'][:47] + "..." if len(cold_papers[i]['title']) > 50 else cold_papers[i]['title']
            warm_title = warm_papers[i]['title'][:47] + "..." if len(warm_papers[i]['title']) > 50 else warm_papers[i]['title']
            
            table.add_row(
                f"#{i+1}",
                cold_title,
                warm_title
            )
        
        console.print(table)
        
        # Show metadata comparison
        console.print("\n📈 Metadata Comparison:", style="cyan")
        console.print(f"  Cold-Start Strategy: {self.cold_start_recs['metadata']['strategy_used']}")
        console.print(f"  Warm-Start Strategy: {self.warm_start_recs['metadata']['strategy_used']}")
        console.print(f"  Generation Time Change: {self.cold_start_recs['metadata']['generation_time_ms']:.0f}ms → {self.warm_start_recs['metadata']['generation_time_ms']:.0f}ms")
        
        # Check overlap
        cold_ids = set(p['paper_id'] for p in cold_papers)
        warm_ids = set(p['paper_id'] for p in warm_papers)
        overlap = len(cold_ids & warm_ids)
        
        console.print(f"\n  Recommendation Overlap: {overlap}/5 papers same")
        console.print(f"  New Papers: {5 - overlap}/5 different papers recommended")
        
        if overlap < 3:
            console.print("  ✅ Good personalization - recommendations evolved significantly!", style="green")
        else:
            console.print("  ⚠️  Limited change - may need more diverse interactions", style="yellow")
    
    async def show_database_changes(self):
        """Show what changed in database tables."""
        console.print("\n💾 DATABASE CHANGES", style="cyan bold")
        
        # Check user_interactions table
        interactions_query = """
            SELECT COUNT(*) as count, 
                   SUM(CASE WHEN interaction_strength > 0 THEN 1 ELSE 0 END) as positive,
                   AVG(interaction_strength) as avg_strength
            FROM user_interactions
            WHERE user_id = $1
        """
        stats = await self.db.fetchrow(interactions_query, self.user_id)
        
        console.print(f"\n📊 user_interactions table:")
        console.print(f"  • Total interactions: {stats['count']}")
        console.print(f"  • Positive interactions: {stats['positive']}")
        console.print(f"  • Average strength: {float(stats['avg_strength']):.2f}")
        
        # Check user_embeddings update
        embedding_query = """
             SELECT generation_method, interaction_count, created_at,
                CASE WHEN based_on_papers IS NOT NULL 
                THEN array_length(based_on_papers, 1) 
                ELSE 0 
            END as papers_used
            FROM user_embeddings_minilm
            WHERE user_id = $1
        """
        emb = await self.db.fetchrow(embedding_query, self.user_id)
        
        console.print(f"\n🧠 user_embeddings_minilm table:")
        console.print(f"  • Generation method: {emb['generation_method']}")
        console.print(f"  • Based on {emb['papers_used']} papers")
        console.print(f"  • Interaction count: {emb['interaction_count']}")
        console.print(f"  • Last updated: {emb['created_at']}")
        
        # Check recommendation_state
        state_query = """
            SELECT recommendation_stage, interaction_count, 
            last_embedding_update_minilm, 
            last_embedding_update_specter,
            last_recommendation_generated
            FROM user_recommendation_state
            WHERE user_id = $1
        """
        state = await self.db.fetchrow(state_query, self.user_id)
        
        console.print(f"\n📈 user_recommendation_state table:")
        console.print(f"  • Stage: {state['recommendation_stage']}")
        console.print(f"  • Interaction count: {state['interaction_count']}")
        console.print(f"  • Last embedding update: {state['last_embedding_update_minilm']}")
        console.print(f"  • Last recommendation generated: {state['last_embedding_update_specter']}")
        
        # Check recommendation_events
        events_query = """
            SELECT COUNT(*) as count,
                   array_agg(DISTINCT recommendation_strategy) as strategies
            FROM recommendation_events
            WHERE user_id = $1
        """
        events = await self.db.fetchrow(events_query, self.user_id)
        
        console.print(f"\n📝 recommendation_events table:")
        console.print(f"  • Total events: {events['count']}")
        console.print(f"  • Strategies used: {', '.join(events['strategies']) if events['strategies'] else 'None'}")
    
    async def run_simulation(self, interaction_profile: str = 'engaged_user'):
        """Run complete simulation."""
        console.print("\n" + "="*70, style="cyan bold")
        console.print("  USER JOURNEY SIMULATION: Cold-Start → Warm-Start", style="cyan bold")
        console.print("="*70 + "\n", style="cyan bold")
        
        # Step 1: Find user if not provided
        if self.user_id is None:
            self.user_id = await self.find_cold_start_user()
        else:
            console.print(f"✅ Using User ID: {self.user_id}", style="green")
        
        # Step 2: Get initial snapshot
        console.print("\n📸 INITIAL STATE", style="cyan bold")
        initial_snapshot = await self.get_user_snapshot()
        self.database_snapshots.append(('initial', initial_snapshot))
        
        console.print(f"  • Stage: {initial_snapshot.get('recommendation_stage')}")
        console.print(f"  • Interaction count: {initial_snapshot.get('interaction_count')}")
        console.print(f"  • Domain: {initial_snapshot.get('primary_domain')}")
        console.print(f"  • Interests: {', '.join(initial_snapshot.get('interests', [])[:3])}")
        
        # Step 3: Get cold-start recommendations
        console.print("\n🎯 STEP 1: Getting Cold-Start Recommendations", style="cyan bold")
        
        with console.status("[bold cyan]Generating profile-based recommendations..."):
            self.cold_start_recs = self.get_recommendations()
        
        console.print(f"✅ Received {len(self.cold_start_recs['recommendations'])} papers")
        console.print(f"  • Strategy: {self.cold_start_recs['metadata']['strategy_used']}")
        console.print(f"  • Generation time: {self.cold_start_recs['metadata']['generation_time_ms']:.0f}ms")
        
        # Show sample papers
        console.print("\n  Top 3 Papers:")
        for i, paper in enumerate(self.cold_start_recs['recommendations'][:3], 1):
            console.print(f"    {i}. {paper['title'][:60]}...")
        
        # Step 4: Simulate interactions
        console.print(f"\n🎬 STEP 2: Simulating User Interactions ({interaction_profile})", style="cyan bold")
        
        await self.simulate_interactions(
            papers=self.cold_start_recs['recommendations'],
            profile=interaction_profile
        )
        
        # Step 5: Check transition
        console.print("\n🔄 STEP 3: Checking Stage Transition", style="cyan bold")
        
        await asyncio.sleep(2)  # Brief pause
        
        transition_snapshot = await self.check_stage_transition()
        self.database_snapshots.append(('after_interactions', transition_snapshot))
        
        # Check if transitioned
        if transition_snapshot.get('recommendation_stage') != initial_snapshot.get('recommendation_stage'):
            console.print(f"\n🎉 STAGE TRANSITION DETECTED!", style="green bold")
            console.print(f"  {initial_snapshot.get('recommendation_stage')} → {transition_snapshot.get('recommendation_stage')}", style="green")
        else:
            console.print(f"\n⚠️  Still in {transition_snapshot.get('recommendation_stage')} stage", style="yellow")
            console.print(f"  Need {10 - transition_snapshot.get('interaction_count', 0)} more interactions for transition")
        
        # Step 6: Get warm-start recommendations
        console.print("\n🎯 STEP 4: Getting Warm-Start Recommendations", style="cyan bold")
        
        with console.status("[bold cyan]Generating behavior-based recommendations..."):
            self.warm_start_recs = self.get_recommendations()
        
        console.print(f"✅ Received {len(self.warm_start_recs['recommendations'])} papers")
        console.print(f"  • Strategy: {self.warm_start_recs['metadata']['strategy_used']}")
        console.print(f"  • Generation time: {self.warm_start_recs['metadata']['generation_time_ms']:.0f}ms")
        
        # Show sample papers
        console.print("\n  Top 3 Papers:")
        for i, paper in enumerate(self.warm_start_recs['recommendations'][:3], 1):
            console.print(f"    {i}. {paper['title'][:60]}...")
        
        # Step 7: Compare
        self.compare_recommendations()
        
        # Step 8: Show database changes
        await self.show_database_changes()
        
        # Step 9: Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate summary report."""
        console.print("\n" + "="*70, style="cyan bold")
        console.print("  SIMULATION SUMMARY", style="cyan bold")
        console.print("="*70 + "\n", style="cyan bold")
        
        # Interaction summary table
        table = Table(title="Interaction Log")
        table.add_column("#", style="cyan", width=6)
        table.add_column("Type", style="yellow", width=14)
        table.add_column("Paper", style="green", width=52)
        
        for interaction in self.interactions[:10]:
            table.add_row(
                str(interaction['number']),
                interaction['type'],
                interaction['paper_title']
            )
        
        console.print(table)
        
        # Build report with datetime serialization
        report = {
            'simulation_id': self.session_id,
            'user_id': self.user_id,
            'timestamp': datetime.now().isoformat(),  # ✅ Already serialized
            'interaction_profile': 'engaged_user',
            'total_interactions': len(self.interactions),
            'cold_start_recommendations': [
                {
                    'paper_id': p['paper_id'],
                    'title': p['title'],
                    'relevance_score': p.get('relevance_score')
                }
                for p in self.cold_start_recs['recommendations'][:5]
            ],
            'warm_start_recommendations': [
                {
                    'paper_id': p['paper_id'],
                    'title': p['title'],
                    'relevance_score': p.get('relevance_score')
                }
                for p in self.warm_start_recs['recommendations'][:5]
            ],
            'interactions': self.interactions,
            # ✅ Convert database snapshots to JSON-serializable format
            'database_snapshots': [
                {
                    'snapshot_type': snap_type,
                    'data': {
                        k: (v.isoformat() if isinstance(v, datetime) else v)
                        for k, v in snap_data.items()
                    }
                }
                for snap_type, snap_data in self.database_snapshots
            ]
        }
        
        # Save to file with datetime serialization
        report_path = Path(f"simulation_report_{self.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # Use custom JSON serializer
        report_path.write_text(json.dumps(report, indent=2, default=json_serial))
        
        console.print(f"\n💾 Report saved: {report_path}", style="green")
        
        # Final summary
        console.print("\n🎉 SIMULATION COMPLETE!", style="green bold")
        console.print(f"  • User evolved through recommendation stages")
        console.print(f"  • Recommendations adapted to user behavior")
        console.print(f"  • System learned user's true preferences")


async def main():
    parser = argparse.ArgumentParser(description='Simulate user journey from cold-start to warm-start')
    parser.add_argument('--user-id', type=int, help='Specific user ID to simulate')
    parser.add_argument('--auto', action='store_true', help='Auto-select cold-start user')
    parser.add_argument('--profile', type=str, default='engaged_user', 
                       choices=['engaged_user', 'casual_user', 'researcher'],
                       help='Interaction behavior profile')
    args = parser.parse_args()
    
    simulator = UserJourneySimulator(user_id=args.user_id)
    
    try:
        await simulator.initialize()
        await simulator.run_simulation(interaction_profile=args.profile)
        
    except KeyboardInterrupt:
        console.print("\n⚠️  Simulation interrupted by user", style="yellow")
    
    except Exception as e:
        console.print(f"\n❌ Simulation failed: {e}", style="red bold")
        raise
    
    finally:
        await simulator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())