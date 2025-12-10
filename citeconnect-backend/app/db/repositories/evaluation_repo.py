"""
Evaluation repository for managing evaluation results.
Handles storage and retrieval of cold-start and warm-start evaluations.
"""
from typing import List, Optional, Dict, Any
import asyncpg
from datetime import datetime, timedelta
import json
from app.db.repositories.base import BaseRepository
from app.db.connection import DatabaseConnection
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EvaluationRepository(BaseRepository):
    """Repository for evaluation operations."""
    
    @property
    def table_name(self) -> str:
        return "cold_start_evaluations"
    
    def __init__(self, db: DatabaseConnection):
        super().__init__(db)
        logger.info("EvaluationRepository initialized")
    
    # ---------------------------------------------------------
    # Cold Start Methods
    # ---------------------------------------------------------

    async def save_cold_start_evaluation(
        self,
        user_id: int,
        embedding_model: str,
        profile_alignment: float,
        ground_truth_quality: float,
        recommendation_count: int,
        metadata: Optional[Dict] = None
    ) -> asyncpg.Record:
        """
        Save cold-start evaluation result.
        
        Note: 'combined_score' is omitted from the INSERT because it is 
        defined as GENERATED ALWAYS in the schema.
        """
        logger.debug(
            "Saving cold-start evaluation",
            user_id=user_id,
            model=embedding_model,
            profile_alignment=profile_alignment
        )
        
        # We explicitly handle metadata if provided, otherwise default to NULL (or empty dict depending on DB default)
        # Using specific query structure to handle the optional metadata
        if metadata:
            query = """
                INSERT INTO cold_start_evaluations (
                    user_id,
                    embedding_model,
                    profile_alignment,
                    ground_truth_quality,
                    recommendation_count,
                    evaluation_timestamp,
                    evaluation_metadata
                )
                VALUES ($1, $2, $3, $4, $5, NOW(), $6)
                RETURNING evaluation_id, combined_score
            """
            params = (
                user_id, embedding_model, profile_alignment, 
                ground_truth_quality, recommendation_count, metadata
            )
        else:
            query = """
                INSERT INTO cold_start_evaluations (
                    user_id,
                    embedding_model,
                    profile_alignment,
                    ground_truth_quality,
                    recommendation_count,
                    evaluation_timestamp
                )
                VALUES ($1, $2, $3, $4, $5, NOW())
                RETURNING evaluation_id, combined_score
            """
            params = (
                user_id, embedding_model, profile_alignment, 
                ground_truth_quality, recommendation_count
            )
        
        try:
            result = await self.db.fetchrow(query, *params)
            
            logger.info(
                "Cold-start evaluation saved",
                evaluation_id=result['evaluation_id'],
                user_id=user_id,
                # combined_score is returned from DB generation
                combined_score=result['combined_score']
            )
            return result
            
        except Exception as e:
            logger.error(
                "Failed to save cold-start evaluation",
                user_id=user_id,
                error=str(e),
                exc_info=True
            )
            raise

    async def get_cold_start_users_for_evaluation(self) -> List[int]:
        """Fetch all users ready for cold-start evaluation."""
        query = """
            SELECT user_id
            FROM user_recommendation_state
            WHERE recommendation_stage = 'cold_start'
              AND user_id IN (SELECT user_id FROM user_profiles_extended)
            ORDER BY user_id
        """
        results = await self.db.fetch(query)
        return [r['user_id'] for r in results]

    # ---------------------------------------------------------
    # Warm Start Methods
    # ---------------------------------------------------------

    async def save_warm_start_evaluation(
        self,
        user_id: int,
        precision_at_10: float,
        recall_at_10: float,
        ndcg_at_10: float,
        estimated_ctr: float,
        recommended_paper_ids: List[str],
        metadata: Dict[str, Any]
    ) -> asyncpg.Record:
        """Save warm-start evaluation result."""
        query = """
            INSERT INTO warm_start_evaluation (
                user_id,
                precision_at_10,
                recall_at_10,
                ndcg_at_10,
                estimated_ctr,
                recommended_paper_ids,
                evaluation_metadata,
                evaluated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
            RETURNING evaluation_id
        """
        try:
            metadata_json = json.dumps(metadata)
            result = await self.db.fetchrow(
                query,
                user_id, precision_at_10, recall_at_10, ndcg_at_10,
                estimated_ctr, recommended_paper_ids, metadata_json
            )
            logger.debug("Warm-start evaluation saved", evaluation_id=result['evaluation_id'])
            return result
        except Exception as e:
            logger.error("Failed to save warm-start evaluation", error=str(e))
            raise

    async def get_user_saved_paper_ids(self, user_id: int) -> List[str]:
        """Get IDs of papers saved by the user (proxy for ground truth)."""
        query = "SELECT paper_id FROM user_saved_papers WHERE user_id = $1"
        results = await self.db.fetch(query, user_id)
        return [r['paper_id'] for r in results]

    # ---------------------------------------------------------
    # Ground Truth & Helper Methods
    # ---------------------------------------------------------

    async def find_relevant_ground_truth_papers(
        self,
        user_interests: List[str],
        domain: str,
        limit: int = 10
    ) -> List[str]:
        """Find ground truth papers matching interests and domain."""
        if not user_interests:
            return []

        # Build search conditions dynamically
        # Note: We manually build the OR clause here. 
        # In a production environment with untrusted input, verify `interest` is safe 
        # or use specialized text search features (tsvector) instead of ILIKE.
        conditions = []
        for interest in user_interests:
            # Basic sanitization for SQL string literal
            safe_interest = interest.replace("'", "''")
            conditions.append(f"title ILIKE '%{safe_interest}%'")
            conditions.append(f"abstract ILIKE '%{safe_interest}%'")
        
        where_clause = ' OR '.join(conditions)
        
        query = f"""
            SELECT gtp.paper_id
            FROM ground_truth_papers gtp
            JOIN papers p ON gtp.paper_id = p.paper_id
            WHERE gtp.domain = $1
              AND ({where_clause})
            ORDER BY gtp.quality_score DESC
            LIMIT $2
        """
        results = await self.db.fetch(query, domain, limit)
        return [r['paper_id'] for r in results]

    # ---------------------------------------------------------
    # Reporting & Analysis Methods
    # ---------------------------------------------------------

    async def get_recent_evaluations(
        self, 
        evaluation_type: str, 
        limit: int = 100
    ) -> List[asyncpg.Record]:
        """Get most recent evaluations globally."""
        if evaluation_type == 'cold_start':
            query = """
                SELECT 
                    profile_alignment,
                    ground_truth_quality,
                    combined_score,
                    evaluation_timestamp as evaluated_at,
                    evaluation_metadata
                FROM cold_start_evaluations
                ORDER BY evaluation_timestamp DESC
                LIMIT $1
            """
        else:
            query = """
                SELECT 
                    precision_at_10,
                    recall_at_10,
                    ndcg_at_10,
                    estimated_ctr,
                    evaluated_at
                FROM warm_start_evaluation
                ORDER BY evaluated_at DESC
                LIMIT $1
            """
        return await self.db.fetch(query, limit)

    async def get_evaluations_by_user(
        self,
        user_id: int,
        evaluation_type: str = 'cold_start',
        limit: int = 10
    ) -> List[asyncpg.Record]:
        """Get evaluation history for a user."""
        if evaluation_type == 'cold_start':
            query = """
                SELECT *
                FROM cold_start_evaluations
                WHERE user_id = $1
                ORDER BY evaluation_timestamp DESC
                LIMIT $2
            """
        else:
            query = """
                SELECT *
                FROM warm_start_evaluation
                WHERE user_id = $1
                ORDER BY evaluated_at DESC
                LIMIT $2
            """
        return await self.db.fetch(query, user_id, limit)

    async def get_aggregate_statistics(
        self,
        evaluation_type: str = 'cold_start',
        time_window_days: Optional[int] = None,
        embedding_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get aggregate statistics for evaluations."""
        if evaluation_type == 'cold_start':
            query = """
                SELECT 
                    COUNT(*) as total_evaluations,
                    AVG(profile_alignment) as avg_profile_alignment,
                    AVG(ground_truth_quality) as avg_ground_truth_quality,
                    AVG(combined_score) as avg_combined_score,
                    STDDEV(combined_score) as std_combined_score,
                    MIN(combined_score) as min_combined_score,
                    MAX(combined_score) as max_combined_score,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY combined_score) as median_score,
                    COUNT(*) FILTER (WHERE combined_score >= 0.60) as passed_count
                FROM cold_start_evaluations
                WHERE 1=1
            """
            
            params = []
            param_num = 1
            
            if time_window_days:
                query += f" AND evaluation_timestamp >= NOW() - INTERVAL '{time_window_days} days'"
            
            if embedding_model:
                query += f" AND embedding_model = ${param_num}"
                params.append(embedding_model)
            
            result = await self.db.fetchrow(query, *params)
            
            if result and result['total_evaluations'] > 0:
                # Helper to safely float conversion
                def to_float(val): return float(val) if val is not None else 0.0
                
                return {
                    'total_evaluations': result['total_evaluations'],
                    'avg_profile_alignment': to_float(result['avg_profile_alignment']),
                    'avg_ground_truth_quality': to_float(result['avg_ground_truth_quality']),
                    'avg_combined_score': to_float(result['avg_combined_score']),
                    'std_combined_score': to_float(result['std_combined_score']),
                    'min_combined_score': to_float(result['min_combined_score']),
                    'max_combined_score': to_float(result['max_combined_score']),
                    'median_score': to_float(result['median_score']),
                    'pass_rate': to_float(result['passed_count']) / float(result['total_evaluations'])
                }
            return {'total_evaluations': 0, 'message': 'No evaluations found'}
        
        else:  # warm_start
            query = """
                SELECT 
                    COUNT(*) as total_evaluations,
                    AVG(precision_at_10) as avg_precision,
                    AVG(recall_at_10) as avg_recall,
                    AVG(ndcg_at_10) as avg_ndcg,
                    AVG(estimated_ctr) as avg_ctr
                FROM warm_start_evaluation
                WHERE 1=1
            """
            
            if time_window_days:
                query += f" AND evaluated_at >= NOW() - INTERVAL '{time_window_days} days'"
            
            result = await self.db.fetchrow(query)
            
            if result and result['total_evaluations'] > 0:
                def to_float(val): return float(val) if val is not None else 0.0
                
                return {
                    'total_evaluations': result['total_evaluations'],
                    'avg_precision': to_float(result['avg_precision']),
                    'avg_recall': to_float(result['avg_recall']),
                    'avg_ndcg': to_float(result['avg_ndcg']),
                    'avg_ctr': to_float(result['avg_ctr'])
                }
            return {'total_evaluations': 0, 'message': 'No evaluations found'}

    async def get_bias_analysis(self, segment_by: str = 'research_stage') -> List[Dict[str, Any]]:
        """Get evaluation scores grouped by user segment."""
        valid_segments = ['research_stage', 'reading_level', 'primary_domain']
        if segment_by not in valid_segments:
            raise ValueError(f"Invalid segment: {segment_by}. Must be one of {valid_segments}")
        
        query = f"""
            SELECT 
                p.{segment_by} as segment_value,
                COUNT(DISTINCT e.user_id) as user_count,
                ROUND(AVG(e.profile_alignment)::numeric, 4) as avg_profile_alignment,
                ROUND(AVG(e.ground_truth_quality)::numeric, 4) as avg_ground_truth_quality,
                ROUND(AVG(e.combined_score)::numeric, 4) as avg_combined_score,
                ROUND(STDDEV(e.combined_score)::numeric, 4) as std_combined_score,
                ROUND(MIN(e.combined_score)::numeric, 4) as min_combined_score,
                ROUND(MAX(e.combined_score)::numeric, 4) as max_combined_score,
                COUNT(*) FILTER (WHERE e.combined_score >= 0.60) as passed_count,
                COUNT(*) as total_count
            FROM cold_start_evaluations e
            JOIN user_profiles_extended p ON e.user_id = p.user_id
            WHERE p.{segment_by} IS NOT NULL
            GROUP BY p.{segment_by}
            ORDER BY avg_combined_score DESC
        """
        
        results = await self.db.fetch(query)
        
        segments = []
        for row in results:
            def to_float(val): return float(val) if val is not None else 0.0
            
            segments.append({
                'segment_name': segment_by,
                'segment_value': row['segment_value'],
                'user_count': row['user_count'],
                'avg_profile_alignment': to_float(row['avg_profile_alignment']),
                'avg_ground_truth_quality': to_float(row['avg_ground_truth_quality']),
                'avg_combined_score': to_float(row['avg_combined_score']),
                'std_combined_score': to_float(row['std_combined_score']),
                'min_combined_score': to_float(row['min_combined_score']),
                'max_combined_score': to_float(row['max_combined_score']),
                'pass_rate': to_float(row['passed_count']) / float(row['total_count']) if row['total_count'] > 0 else 0.0
            })
        return segments

    async def get_model_comparison(self, model_a: str, model_b: str) -> Dict[str, Any]:
        """Compare evaluation scores between two models."""
        query = """
            SELECT 
                embedding_model,
                COUNT(*) as evaluation_count,
                AVG(profile_alignment) as avg_profile_alignment,
                AVG(ground_truth_quality) as avg_ground_truth_quality,
                AVG(combined_score) as avg_combined_score,
                STDDEV(combined_score) as std_combined_score
            FROM cold_start_evaluations
            WHERE embedding_model IN ($1, $2)
            GROUP BY embedding_model
        """
        
        results = await self.db.fetch(query, model_a, model_b)
        
        comparison = {'model_a': model_a, 'model_b': model_b, 'results': {}}
        
        for row in results:
            def to_float(val): return float(val) if val is not None else 0.0
            
            comparison['results'][row['embedding_model']] = {
                'evaluation_count': row['evaluation_count'],
                'avg_profile_alignment': to_float(row['avg_profile_alignment']),
                'avg_ground_truth_quality': to_float(row['avg_ground_truth_quality']),
                'avg_combined_score': to_float(row['avg_combined_score']),
                'std_combined_score': to_float(row['std_combined_score'])
            }
        
        # Determine winner
        if model_a in comparison['results'] and model_b in comparison['results']:
            score_a = comparison['results'][model_a]['avg_combined_score']
            score_b = comparison['results'][model_b]['avg_combined_score']
            
            diff = abs(score_a - score_b)
            
            if diff > 0.05:
                comparison['winner'] = model_a if score_a > score_b else model_b
                comparison['confidence'] = 'high'
            elif diff > 0.02:
                comparison['winner'] = model_a if score_a > score_b else model_b
                comparison['confidence'] = 'medium'
            else:
                comparison['winner'] = 'tie'
                comparison['confidence'] = 'low'
            
            comparison['score_difference'] = round(diff, 4)
        else:
            comparison['winner'] = 'insufficient_data'
            comparison['confidence'] = 'none'
        
        return comparison

    async def delete_evaluations_for_user(self, user_id: int) -> int:
        """Delete all evaluations for a user."""
        query = "DELETE FROM cold_start_evaluations WHERE user_id = $1"
        result = await self.db.execute(query, user_id)
        # Result string is typically "DELETE N"
        return int(result.split()[-1]) if result else 0