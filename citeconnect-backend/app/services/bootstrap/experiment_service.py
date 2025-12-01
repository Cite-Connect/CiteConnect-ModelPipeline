from typing import Dict, List, Optional
from datetime import datetime
import json
from app.db.connection import DatabaseConnection
from app.utils.logger import get_logger

logger = get_logger(__name__)

class ExperimentService:
    def __init__(self, db: DatabaseConnection):
        self.db = db

    async def get_active_variant(self, user_id: int) -> str:
        """
        Determines if user is in Group A (MiniLM) or Group B (SPECTER).
        Reads from cache or DB, DOES NOT write to 'ab_test_comparisons'.
        """
        # Logic: Hash user_id to deterministically assign group
        # This allows consistent A/B testing without constant DB lookups
        if user_id % 2 == 0:
            return 'minilm'
        else:
            return 'specter'

    async def log_recommendation_event(
        self,
        user_id: int,
        model_name: str,
        recommendations: List[Dict],
        evaluation_scores: Dict,
        user_context: Dict,
        generation_time_ms: float
    ) -> int:
        """
        Logs a single EVENT.
        This is called by Orchestrator. 
        It writes to 'recommendation_events', NOT 'experiment_runs'.
        """
        try:
            query = """
                INSERT INTO recommendation_events (
                    user_id,
                    model_used,
                    recommended_paper_ids,
                    scores,
                    context_snapshot,
                    generation_time_ms,
                    created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, NOW())
                RETURNING event_id
            """
            
            paper_ids = [p['paper_id'] for p in recommendations]
            
            event_id = await self.db.fetchval(
                query,
                user_id,
                model_name,
                paper_ids,
                json.dumps(evaluation_scores),
                json.dumps(user_context.get('profile', {})), # Snapshot profile
                generation_time_ms
            )
            
            return event_id
            
        except Exception as e:
            logger.error("Failed to log recommendation event", error=str(e))
            return None