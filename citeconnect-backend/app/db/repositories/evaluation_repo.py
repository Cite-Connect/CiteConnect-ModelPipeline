"""
Evaluation repository for managing evaluation results.
Handles storage and retrieval of cold-start and warm-start evaluations.
"""
from typing import List, Optional, Dict, Any
import asyncpg
from datetime import datetime, timedelta

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
        
        Args:
            user_id: User identifier
            embedding_model: Model used ('all-MiniLM-L6-v2' or 'specter2')
            profile_alignment: Profile alignment score (0-1)
            ground_truth_quality: Ground truth quality score (0-1)
            recommendation_count: Number of recommendations evaluated
            metadata: Optional additional metadata
            
        Returns:
            Record: Created evaluation record
        """
        logger.debug(
            "Saving cold-start evaluation",
            user_id=user_id,
            model=embedding_model,
            profile_alignment=profile_alignment,
            ground_truth_quality=ground_truth_quality
        )
        
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
        
        try:
            result = await self.db.fetchrow(
                query,
                user_id,
                embedding_model,
                profile_alignment,
                ground_truth_quality,
                recommendation_count
            )
            
            logger.info(
                "Cold-start evaluation saved",
                evaluation_id=result['evaluation_id'],
                user_id=user_id,
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
    
    async def get_evaluations_by_user(
        self,
        user_id: int,
        evaluation_type: str = 'cold_start',
        limit: int = 10
    ) -> List[asyncpg.Record]:
        """
        Get evaluation history for a user.
        
        Args:
            user_id: User identifier
            evaluation_type: 'cold_start' or 'warm_start'
            limit: Maximum records to return
            
        Returns:
            List of evaluation records
        """
        logger.debug(
            "Getting user evaluations",
            user_id=user_id,
            type=evaluation_type
        )
        
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
                FROM warm_start_evaluations
                WHERE user_id = $1
                ORDER BY evaluation_timestamp DESC
                LIMIT $2
            """
        
        results = await self.db.fetch(query, user_id, limit)
        
        logger.debug(
            "User evaluations retrieved",
            user_id=user_id,
            count=len(results)
        )
        
        return results
    
    async def get_aggregate_statistics(
        self,
        evaluation_type: str = 'cold_start',
        time_window_days: Optional[int] = None,
        embedding_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get aggregate statistics for evaluations.
        
        Args:
            evaluation_type: 'cold_start' or 'warm_start'
            time_window_days: Optional time window filter
            embedding_model: Optional model filter
            
        Returns:
            Dict with aggregate statistics
        """
        logger.debug(
            "Getting aggregate statistics",
            type=evaluation_type,
            time_window=time_window_days,
            model=embedding_model
        )
        
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
                stats = {
                    'total_evaluations': result['total_evaluations'],
                    'avg_profile_alignment': float(result['avg_profile_alignment']) if result['avg_profile_alignment'] else 0.0,
                    'avg_ground_truth_quality': float(result['avg_ground_truth_quality']) if result['avg_ground_truth_quality'] else 0.0,
                    'avg_combined_score': float(result['avg_combined_score']) if result['avg_combined_score'] else 0.0,
                    'std_combined_score': float(result['std_combined_score']) if result['std_combined_score'] else 0.0,
                    'min_combined_score': float(result['min_combined_score']) if result['min_combined_score'] else 0.0,
                    'max_combined_score': float(result['max_combined_score']) if result['max_combined_score'] else 0.0,
                    'median_score': float(result['median_score']) if result['median_score'] else 0.0,
                    'pass_rate': float(result['passed_count']) / float(result['total_evaluations']) if result['total_evaluations'] > 0 else 0.0
                }
            else:
                stats = {
                    'total_evaluations': 0,
                    'message': 'No evaluations found'
                }
        
        else:  # warm_start
            query = """
                SELECT 
                    COUNT(*) as total_evaluations,
                    AVG(precision_at_10) as avg_precision,
                    AVG(recall_at_10) as avg_recall,
                    AVG(ndcg_at_10) as avg_ndcg,
                    AVG(estimated_ctr) as avg_ctr
                FROM warm_start_evaluations
                WHERE 1=1
            """
            
            if time_window_days:
                query += f" AND evaluation_timestamp >= NOW() - INTERVAL '{time_window_days} days'"
            
            result = await self.db.fetchrow(query)
            
            if result and result['total_evaluations'] > 0:
                stats = {
                    'total_evaluations': result['total_evaluations'],
                    'avg_precision': float(result['avg_precision']) if result['avg_precision'] else 0.0,
                    'avg_recall': float(result['avg_recall']) if result['avg_recall'] else 0.0,
                    'avg_ndcg': float(result['avg_ndcg']) if result['avg_ndcg'] else 0.0,
                    'avg_ctr': float(result['avg_ctr']) if result['avg_ctr'] else 0.0
                }
            else:
                stats = {
                    'total_evaluations': 0,
                    'message': 'No evaluations found'
                }
        
        logger.info(
            "Aggregate statistics retrieved",
            type=evaluation_type,
            total=stats.get('total_evaluations', 0)
        )
        
        return stats
    
    async def get_bias_analysis(
        self,
        segment_by: str = 'research_stage'
    ) -> List[Dict[str, Any]]:
        """
        Get evaluation scores grouped by user segment for bias analysis.
        
        Args:
            segment_by: Column to segment by ('research_stage', 'reading_level', 'primary_domain')
            
        Returns:
            List of segment statistics
        """
        logger.info(
            "Running bias analysis",
            segment_by=segment_by
        )
        
        # Validate segment_by to prevent SQL injection
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
            segment = {
                'segment_name': segment_by,
                'segment_value': row['segment_value'],
                'user_count': row['user_count'],
                'avg_profile_alignment': float(row['avg_profile_alignment']) if row['avg_profile_alignment'] else 0.0,
                'avg_ground_truth_quality': float(row['avg_ground_truth_quality']) if row['avg_ground_truth_quality'] else 0.0,
                'avg_combined_score': float(row['avg_combined_score']) if row['avg_combined_score'] else 0.0,
                'std_combined_score': float(row['std_combined_score']) if row['std_combined_score'] else 0.0,
                'min_combined_score': float(row['min_combined_score']) if row['min_combined_score'] else 0.0,
                'max_combined_score': float(row['max_combined_score']) if row['max_combined_score'] else 0.0,
                'pass_rate': float(row['passed_count']) / float(row['total_count']) if row['total_count'] > 0 else 0.0
            }
            segments.append(segment)
        
        # Calculate bias metric (max - min scores)
        if len(segments) > 1:
            scores = [s['avg_combined_score'] for s in segments]
            bias_magnitude = max(scores) - min(scores)
            
            logger.info(
                "Bias analysis complete",
                segment_by=segment_by,
                bias_magnitude=bias_magnitude,
                segment_count=len(segments)
            )
        else:
            bias_magnitude = 0.0
        
        return {
            'segment_by': segment_by,
            'segments': segments,
            'bias_magnitude': round(bias_magnitude, 4),
            'bias_detected': bias_magnitude > 0.20
        }
    
    async def get_model_comparison(
        self,
        model_a: str = 'all-MiniLM-L6-v2',
        model_b: str = 'specter2'
    ) -> Dict[str, Any]:
        """
        Compare evaluation scores between two models.
        
        Args:
            model_a: First model name
            model_b: Second model name
            
        Returns:
            Comparison statistics
        """
        logger.info(
            "Comparing models",
            model_a=model_a,
            model_b=model_b
        )
        
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
        
        comparison = {
            'model_a': model_a,
            'model_b': model_b,
            'results': {}
        }
        
        for row in results:
            model_name = row['embedding_model']
            comparison['results'][model_name] = {
                'evaluation_count': row['evaluation_count'],
                'avg_profile_alignment': float(row['avg_profile_alignment']) if row['avg_profile_alignment'] else 0.0,
                'avg_ground_truth_quality': float(row['avg_ground_truth_quality']) if row['avg_ground_truth_quality'] else 0.0,
                'avg_combined_score': float(row['avg_combined_score']) if row['avg_combined_score'] else 0.0,
                'std_combined_score': float(row['std_combined_score']) if row['std_combined_score'] else 0.0
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
        
        logger.info(
            "Model comparison complete",
            winner=comparison.get('winner'),
            score_diff=comparison.get('score_difference')
        )
        
        return comparison
    
    async def get_evaluations_by_segment(
        self,
        segment_field: str,
        segment_value: str,
        limit: int = 100
    ) -> List[asyncpg.Record]:
        """
        Get evaluations for a specific user segment.
        
        Args:
            segment_field: Field to filter by ('research_stage', 'reading_level', 'primary_domain')
            segment_value: Value to filter for
            limit: Maximum records
            
        Returns:
            List of evaluation records
        """
        valid_fields = ['research_stage', 'reading_level', 'primary_domain']
        if segment_field not in valid_fields:
            raise ValueError(f"Invalid segment field: {segment_field}")
        
        query = f"""
            SELECT e.*
            FROM cold_start_evaluations e
            JOIN user_profiles_extended p ON e.user_id = p.user_id
            WHERE p.{segment_field} = $1
            ORDER BY e.evaluation_timestamp DESC
            LIMIT $2
        """
        
        results = await self.db.fetch(query, segment_value, limit)
        
        logger.debug(
            "Segment evaluations retrieved",
            segment=f"{segment_field}={segment_value}",
            count=len(results)
        )
        
        return results
    
    async def get_time_series_data(
        self,
        days: int = 30,
        model: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get evaluation scores over time for trend analysis.
        
        Args:
            days: Number of days to look back
            model: Optional model filter
            
        Returns:
            List of daily aggregated scores
        """
        logger.debug(
            "Getting time series data",
            days=days,
            model=model
        )
        
        query = """
            SELECT 
                DATE(evaluation_timestamp) as eval_date,
                COUNT(*) as evaluation_count,
                AVG(combined_score) as avg_score,
                STDDEV(combined_score) as std_score
            FROM cold_start_evaluations
            WHERE evaluation_timestamp >= NOW() - INTERVAL '%s days'
        """ % days
        
        if model:
            query += " AND embedding_model = $1"
            results = await self.db.fetch(query, model)
        else:
            results = await self.db.fetch(query)
        
        time_series = []
        for row in results:
            time_series.append({
                'date': row['eval_date'].isoformat(),
                'evaluation_count': row['evaluation_count'],
                'avg_score': float(row['avg_score']) if row['avg_score'] else 0.0,
                'std_score': float(row['std_score']) if row['std_score'] else 0.0
            })
        
        logger.debug(
            "Time series data retrieved",
            data_points=len(time_series)
        )
        
        return time_series
    
    async def get_top_performing_users(
        self,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get users with highest evaluation scores.
        
        Args:
            limit: Number of users to return
            
        Returns:
            List of top-performing users
        """
        query = """
            SELECT 
                e.user_id,
                u.email,
                p.research_stage,
                p.primary_domain,
                e.combined_score,
                e.profile_alignment,
                e.ground_truth_quality,
                e.embedding_model
            FROM cold_start_evaluations e
            JOIN users u ON e.user_id = u.user_id
            JOIN user_profiles_extended p ON e.user_id = p.user_id
            ORDER BY e.combined_score DESC
            LIMIT $1
        """
        
        results = await self.db.fetch(query, limit)
        
        top_users = []
        for row in results:
            top_users.append({
                'user_id': row['user_id'],
                'email': row['email'],
                'research_stage': row['research_stage'],
                'primary_domain': row['primary_domain'],
                'combined_score': float(row['combined_score']),
                'profile_alignment': float(row['profile_alignment']),
                'ground_truth_quality': float(row['ground_truth_quality']),
                'model': row['embedding_model']
            })
        
        logger.debug(
            "Top performing users retrieved",
            count=len(top_users)
        )
        
        return top_users
    
    async def get_bottom_performing_users(
        self,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get users with lowest evaluation scores (need improvement).
        
        Args:
            limit: Number of users to return
            
        Returns:
            List of bottom-performing users
        """
        query = """
            SELECT 
                e.user_id,
                u.email,
                p.research_stage,
                p.primary_domain,
                p.reading_level,
                e.combined_score,
                e.profile_alignment,
                e.ground_truth_quality
            FROM cold_start_evaluations e
            JOIN users u ON e.user_id = u.user_id
            JOIN user_profiles_extended p ON e.user_id = p.user_id
            ORDER BY e.combined_score ASC
            LIMIT $1
        """
        
        results = await self.db.fetch(query, limit)
        
        bottom_users = []
        for row in results:
            bottom_users.append({
                'user_id': row['user_id'],
                'email': row['email'],
                'research_stage': row['research_stage'],
                'primary_domain': row['primary_domain'],
                'reading_level': row['reading_level'],
                'combined_score': float(row['combined_score']),
                'profile_alignment': float(row['profile_alignment']),
                'ground_truth_quality': float(row['ground_truth_quality'])
            })
        
        logger.debug(
            "Bottom performing users retrieved",
            count=len(bottom_users)
        )
        
        return bottom_users
    
    async def delete_evaluations_for_user(
        self,
        user_id: int
    ) -> int:
        """
        Delete all evaluations for a user (for re-testing).
        
        Args:
            user_id: User identifier
            
        Returns:
            Number of deleted records
        """
        logger.info(
            "Deleting evaluations for user",
            user_id=user_id
        )
        
        query = """
            DELETE FROM cold_start_evaluations
            WHERE user_id = $1
        """
        
        result = await self.db.execute(query, user_id)
        
        # Extract count from result string "DELETE N"
        count = int(result.split()[-1]) if result else 0
        
        logger.info(
            "Evaluations deleted",
            user_id=user_id,
            count=count
        )
        
        return count