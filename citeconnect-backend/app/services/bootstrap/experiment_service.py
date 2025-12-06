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

    def _sanitize_for_json(self, data):
        """
        Recursively convert datetime objects and other non-JSON-serializable 
        types to JSON-compatible formats.
        """
        if isinstance(data, dict):
            return {k: self._sanitize_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_for_json(item) for item in data]
        elif isinstance(data, datetime):
            return data.isoformat()
        elif isinstance(data, (set, tuple)):
            return [self._sanitize_for_json(item) for item in data]
        elif hasattr(data, '__dict__'):
            # Handle objects with __dict__ (like asyncpg Records)
            return self._sanitize_for_json(dict(data))
        else:
            return data

    async def log_recommendation_event(
        self,
        user_id: int,
        model_name: str,
        recommendations: List[Dict],
        evaluation_scores: Dict,
        user_context: Dict,
        generation_time_ms: float
    ) -> Optional[int]:
        """
        Logs a single recommendation event.
        This is called by Orchestrator after generating recommendations.
        Writes to 'recommendation_events' table.
        
        Args:
            user_id: User identifier
            model_name: Embedding model used (minilm/specter)
            recommendations: List of recommended papers with scores
            evaluation_scores: Evaluation metrics (profile_alignment, etc.)
            user_context: User context including stage and profile
            generation_time_ms: Time taken to generate recommendations
            
        Returns:
            Optional[int]: Event ID if successful, None if failed
        """
        try:
            # Map internal strategy names to database-allowed values
            strategy_map = {
                'hybrid_service': 'hybrid',
                'fallback_trending': 'cold_start_canonical',
                'cold_start': 'cold_start_profile',
                'warm_start': 'interaction_based',
                'citation': 'citation_network'
            }
            query = """
                INSERT INTO recommendation_events (
                    user_id,
                    embedding_model,
                    recommended_paper_ids,
                    recommendation_scores,
                    user_stage,
                    recommendation_strategy,
                    dwell_times,
                    event_timestamp
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                RETURNING event_id
            """
            
            # Extract paper IDs and scores
            paper_ids = [p.get('paper_id') for p in recommendations]
            paper_scores = [float(p.get('final_score', 0.0)) for p in recommendations]
            
            # Get user stage and strategy
            user_stage = user_context.get('stage', 'cold_start')
            raw_strategy = evaluation_scores.get('strategy_used', 'hybrid_service')

            strategy = strategy_map.get(raw_strategy, 'hybrid')
            
            # Sanitize evaluation scores and user context for JSON storage
            sanitized_scores = self._sanitize_for_json(evaluation_scores)
            sanitized_context = self._sanitize_for_json(user_context)
            
            # Build dwell_times metadata (JSONB column)
            dwell_times_metadata = {
                'evaluation_scores': sanitized_scores,
                'generation_time_ms': generation_time_ms,
                'user_context_snapshot': sanitized_context,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(
                "Logging recommendation event",
                user_id=user_id,
                model=model_name,
                paper_count=len(paper_ids),
                user_stage=user_stage
            )
            
            event_id = await self.db.fetchval(
                query,
                user_id,
                model_name,
                paper_ids,
                paper_scores,
                user_stage,
                strategy,
                json.dumps(dwell_times_metadata)
            )
            
            logger.info(
                "Recommendation event logged successfully",
                event_id=event_id,
                user_id=user_id,
                model=model_name
            )
            
            return event_id
            
        except Exception as e:
            logger.error(
                "Failed to log recommendation event",
                error=str(e),
                error_type=type(e).__name__,
                user_id=user_id,
                exc_info=True
            )
            # Don't raise - logging failures shouldn't break recommendations
            return None