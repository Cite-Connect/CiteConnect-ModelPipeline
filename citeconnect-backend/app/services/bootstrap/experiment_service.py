"""
Experiment tracking service using MLflow.
Manages A/B tests, logs metrics, and tracks model performance.
"""
import mlflow
from typing import Dict, Optional, Any, List
from datetime import datetime
import uuid
from app.config import settings
from app.utils.logger import get_logger
from app.db.connection import DatabaseConnection

logger = get_logger(__name__)


class ExperimentService:
    """
    Manages experiments, A/B tests, and MLflow tracking.
    Tracks recommendation quality and model performance.
    """
    
    def __init__(self, db: DatabaseConnection):
        """
        Initialize experiment service.
        
        Args:
            db: Database connection
        """
        self.db = db
        self.active_experiments = {}
        self.assignment_cache = {}  # Ensures consistent user assignments
        
        logger.info("ExperimentService initialized")
    
    async def initialize(self) -> None:
        """
        Initialize MLflow and load active experiments.
        Called during application startup.
        """
        logger.info("Initializing experiment tracking")
        
        try:
            # Set MLflow tracking URI
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            
            # Create or get experiment
            try:
                experiment = mlflow.get_experiment_by_name(
                    settings.MLFLOW_EXPERIMENT_NAME
                )
                if experiment is None:
                    experiment_id = mlflow.create_experiment(
                        settings.MLFLOW_EXPERIMENT_NAME,
                        tags={
                            "project": "citeconnect",
                            "version": settings.APP_VERSION
                        }
                    )
                    logger.info(
                        "MLflow experiment created",
                        experiment_id=experiment_id
                    )
                else:
                    logger.info(
                        "MLflow experiment found",
                        experiment_id=experiment.experiment_id
                    )
            except Exception as e:
                logger.warning(
                    "MLflow experiment setup failed, continuing without tracking",
                    error=str(e)
                )
            
            # Load active experiments from database
            await self._load_active_experiments()
            
            logger.info(
                "Experiment service initialized",
                active_experiments=len(self.active_experiments)
            )
            
        except Exception as e:
            logger.error(
                "Experiment service initialization failed",
                error=str(e),
                exc_info=True
            )
            # Don't fail startup if MLflow is unavailable
            logger.warning("Continuing without experiment tracking")
    
    async def _load_active_experiments(self) -> None:
        """Load active experiments from database."""
        query = """
            SELECT run_id, embedding_model, experiment_type, user_segment
            FROM experiment_runs
            WHERE status = 'running'
        """
        
        try:
            results = await self.db.fetch(query)
            
            for row in results:
                self.active_experiments[row['run_id']] = {
                    'model': row['embedding_model'],
                    'type': row['experiment_type'],
                    'segment': row['user_segment']
                }
            
            logger.info(
                "Active experiments loaded",
                count=len(results)
            )
        except Exception as e:
            logger.warning(
                "Could not load active experiments",
                error=str(e)
            )
    
    async def log_recommendation_event(
        self,
        user_id: int,
        model_name: str,
        recommendations: List[Dict],
        evaluation_scores: Dict,
        user_context: Dict,
        generation_time_ms: float
    ) -> str:
        """
        Log recommendation event to MLflow and database.
        
        Args:
            user_id: User identifier
            model_name: Model used
            recommendations: List of recommended papers
            evaluation_scores: Evaluation metrics
            user_context: User context (stage, profile, etc.)
            generation_time_ms: Time taken to generate
            
        Returns:
            str: Event ID
        """
        logger.debug(
            "Logging recommendation event",
            user_id=user_id,
            model=model_name,
            rec_count=len(recommendations)
        )
        
        # Start MLflow run
        run_id = None
        try:
            with mlflow.start_run(run_name=f"rec_{user_id}_{datetime.now().isoformat()}"):
                # Log parameters
                mlflow.log_param("user_id", user_id)
                mlflow.log_param("model", model_name)
                mlflow.log_param("user_stage", user_context.get('stage'))
                mlflow.log_param("recommendation_count", len(recommendations))
                
                # Log metrics
                mlflow.log_metric("generation_time_ms", generation_time_ms)
                
                if evaluation_scores.get('profile_alignment'):
                    mlflow.log_metric(
                        "profile_alignment", 
                        evaluation_scores['profile_alignment']
                    )
                
                if evaluation_scores.get('ground_truth_quality'):
                    mlflow.log_metric(
                        "ground_truth_quality",
                        evaluation_scores['ground_truth_quality']
                    )
                
                if evaluation_scores.get('combined_score'):
                    mlflow.log_metric(
                        "combined_score",
                        evaluation_scores['combined_score']
                    )
                
                # Log tags
                mlflow.set_tag("user_domain", user_context.get('profile', {}).get('primary_domain'))
                mlflow.set_tag("user_stage", user_context.get('stage'))
                mlflow.set_tag("timestamp", datetime.now().isoformat())
                
                run_id = mlflow.active_run().info.run_id
                
                logger.debug(
                    "MLflow run logged",
                    run_id=run_id
                )
                
        except Exception as e:
            logger.warning(
                "MLflow logging failed, continuing",
                error=str(e)
            )
        
        # Save to database
        try:
            paper_ids = [p['paper_id'] for p in recommendations]
            scores = [p.get('relevance_score', 0.0) for p in recommendations]
            
            query = """
                INSERT INTO recommendation_events (
                    user_id, embedding_model, run_id,
                    recommended_paper_ids, recommendation_scores,
                    recommendation_strategy, user_stage
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING event_id
            """
            
            event_id = await self.db.fetchval(
                query,
                user_id,
                model_name,
                run_id,
                paper_ids,
                scores,
                user_context.get('strategy', 'unknown'),
                user_context.get('stage')
            )
            
            logger.info(
                "Recommendation event logged",
                event_id=event_id,
                mlflow_run_id=run_id
            )
            
            return str(event_id)
            
        except Exception as e:
            logger.error(
                "Failed to log recommendation event to database",
                error=str(e),
                exc_info=True
            )
            raise
    
    async def log_interaction_evaluation(
        self,
        user_id: int,
        model_name: str,
        metrics: Dict[str, float],
        interaction_count: int
    ) -> None:
        """
        Log interaction-based evaluation metrics.
        Called after user interactions to measure model performance.
        
        Args:
            user_id: User identifier
            model_name: Model used
            metrics: Evaluation metrics (precision, CTR, etc.)
            interaction_count: Number of interactions in evaluation window
        """
        logger.debug(
            "Logging interaction evaluation",
            user_id=user_id,
            model=model_name,
            interaction_count=interaction_count
        )
        
        # Log to MLflow
        try:
            with mlflow.start_run(run_name=f"eval_{user_id}_{datetime.now().isoformat()}"):
                mlflow.log_param("user_id", user_id)
                mlflow.log_param("model", model_name)
                mlflow.log_param("evaluation_type", "interaction_based")
                
                # Log all metrics
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                mlflow.log_metric("interaction_count", interaction_count)
                
                mlflow.set_tag("timestamp", datetime.now().isoformat())
                
                logger.debug("Interaction evaluation logged to MLflow")
                
        except Exception as e:
            logger.warning(
                "MLflow logging failed",
                error=str(e)
            )
        
        # Save to database
        try:
            query = """
                INSERT INTO interaction_evaluations (
                    user_id, embedding_model, 
                    profile_alignment, ground_truth_quality,
                    click_through_rate, save_rate,
                    precision_at_10, recall_at_10,
                    interaction_count
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """
            
            await self.db.execute(
                query,
                user_id,
                model_name,
                metrics.get('profile_alignment'),
                metrics.get('ground_truth_quality'),
                metrics.get('click_through_rate'),
                metrics.get('save_rate'),
                metrics.get('precision_at_10'),
                metrics.get('recall_at_10'),
                interaction_count
            )
            
            logger.info(
                "Interaction evaluation saved to database",
                user_id=user_id
            )
            
        except Exception as e:
            logger.error(
                "Failed to save interaction evaluation",
                error=str(e),
                exc_info=True
            )
    
    async def assign_user_to_experiment(
        self,
        user_id: int,
        experiment_type: str = "a_b_test"
    ) -> str:
        """
        Assign user to A/B test variant consistently.
        
        Args:
            user_id: User identifier
            experiment_type: Type of experiment
            
        Returns:
            str: Variant assignment ('A' or 'B')
        """
        # Check cache for consistent assignment
        cache_key = f"{experiment_type}_{user_id}"
        
        if cache_key in self.assignment_cache:
            variant = self.assignment_cache[cache_key]
            logger.debug(
                "User assignment from cache",
                user_id=user_id,
                variant=variant
            )
            return variant
        
        # Deterministic assignment based on user_id
        # Ensures same user always gets same variant
        variant = 'A' if user_id % 2 == 0 else 'B'
        
        self.assignment_cache[cache_key] = variant
        
        logger.info(
            "User assigned to variant",
            user_id=user_id,
            experiment=experiment_type,
            variant=variant
        )
        
        return variant
    
    async def compare_models(
        self,
        model_a: str,
        model_b: str,
        metric_name: str = "precision_at_10"
    ) -> Dict:
        """
        Compare performance of two models using A/B test data.
        
        Args:
            model_a: First model name
            model_b: Second model name
            metric_name: Metric to compare
            
        Returns:
            Dict with comparison results including statistical significance
        """
        logger.info(
            "Comparing models",
            model_a=model_a,
            model_b=model_b,
            metric=metric_name
        )
        
        # Get evaluation data for both models
        query = """
            SELECT 
                embedding_model,
                AVG(CASE 
                    WHEN $3 = 'precision_at_10' THEN precision_at_10
                    WHEN $3 = 'click_through_rate' THEN click_through_rate
                    WHEN $3 = 'save_rate' THEN save_rate
                    ELSE NULL
                END) as avg_metric,
                COUNT(*) as sample_size
            FROM interaction_evaluations
            WHERE embedding_model IN ($1, $2)
              AND evaluation_timestamp >= NOW() - INTERVAL '7 days'
            GROUP BY embedding_model
        """
        
        try:
            results = await self.db.fetch(query, model_a, model_b, metric_name)
            
            metrics = {row['embedding_model']: row for row in results}
            
            if model_a not in metrics or model_b not in metrics:
                logger.warning(
                    "Insufficient data for comparison",
                    model_a=model_a,
                    model_b=model_b
                )
                return {
                    "status": "insufficient_data",
                    "message": "Need more interactions for comparison"
                }
            
            # Calculate difference
            metric_a = metrics[model_a]['avg_metric']
            metric_b = metrics[model_b]['avg_metric']
            
            lift = ((metric_b - metric_a) / metric_a * 100) if metric_a > 0 else 0
            
            # Determine winner (simple comparison for now)
            # TODO: Add statistical significance test (t-test)
            winner = model_b if metric_b > metric_a else model_a
            
            comparison = {
                "model_a": model_a,
                "model_b": model_b,
                "metric": metric_name,
                "model_a_value": float(metric_a) if metric_a else 0,
                "model_b_value": float(metric_b) if metric_b else 0,
                "lift_percentage": round(lift, 2),
                "winner": winner,
                "sample_size_a": metrics[model_a]['sample_size'],
                "sample_size_b": metrics[model_b]['sample_size'],
                "status": "complete"
            }
            
            # Log comparison to MLflow
            try:
                with mlflow.start_run(run_name=f"comparison_{model_a}_vs_{model_b}"):
                    mlflow.log_param("model_a", model_a)
                    mlflow.log_param("model_b", model_b)
                    mlflow.log_param("metric", metric_name)
                    mlflow.log_metric(f"{model_a}_{metric_name}", metric_a or 0)
                    mlflow.log_metric(f"{model_b}_{metric_name}", metric_b or 0)
                    mlflow.log_metric("lift_percentage", lift)
                    mlflow.set_tag("winner", winner)
                    
                    logger.debug("Model comparison logged to MLflow")
                    
            except Exception as e:
                logger.warning(
                    "MLflow comparison logging failed",
                    error=str(e)
                )
            
            logger.info(
                "Model comparison complete",
                winner=winner,
                lift=lift
            )
            
            return comparison
            
        except Exception as e:
            logger.error(
                "Model comparison failed",
                error=str(e),
                exc_info=True
            )
            raise
    
    async def create_experiment_run(
        self,
        model_name: str,
        experiment_type: str,
        hyperparameters: Dict,
        user_segment: str = "all"
    ) -> str:
        """
        Create new experiment run in database and MLflow.
        
        Args:
            model_name: Embedding model name
            experiment_type: Type of experiment
            hyperparameters: Model/recommendation parameters
            user_segment: Target user segment
            
        Returns:
            str: Run ID
        """
        logger.info(
            "Creating experiment run",
            model=model_name,
            type=experiment_type,
            segment=user_segment
        )
        
        run_id = str(uuid.uuid4())
        mlflow_run_id = None
        
        # Start MLflow run
        try:
            with mlflow.start_run(run_name=f"{experiment_type}_{model_name}") as run:
                # Log parameters
                mlflow.log_param("embedding_model", model_name)
                mlflow.log_param("experiment_type", experiment_type)
                mlflow.log_param("user_segment", user_segment)
                
                for key, value in hyperparameters.items():
                    mlflow.log_param(key, value)
                
                mlflow_run_id = run.info.run_id
                
                logger.debug(
                    "MLflow run created",
                    mlflow_run_id=mlflow_run_id
                )
                
        except Exception as e:
            logger.warning(
                "MLflow run creation failed",
                error=str(e)
            )
        
        # Save to database
        try:
            # Get embedding dimension
            dim = 384 if 'minilm' in model_name.lower() else 768
            
            query = """
                INSERT INTO experiment_runs (
                    run_id, embedding_model, embedding_dimension,
                    hyperparameters, experiment_type, user_segment,
                    status, mlflow_run_id
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING run_id
            """
            
            await self.db.execute(
                query,
                run_id,
                model_name,
                dim,
                hyperparameters,
                experiment_type,
                user_segment,
                'running',
                mlflow_run_id
            )
            
            logger.info(
                "Experiment run created",
                run_id=run_id,
                mlflow_run_id=mlflow_run_id
            )
            
            return run_id
            
        except Exception as e:
            logger.error(
                "Failed to create experiment run",
                error=str(e),
                exc_info=True
            )
            raise
    
    async def log_cold_start_evaluation(
        self,
        user_id: int,
        model_name: str,
        profile_alignment: float,
        ground_truth_quality: float,
        recommendation_count: int
    ) -> None:
        """
        Log cold-start evaluation to database and MLflow.
        
        Args:
            user_id: User identifier
            model_name: Model used
            profile_alignment: Profile alignment score
            ground_truth_quality: Ground truth quality score
            recommendation_count: Number of recommendations
        """
        logger.debug(
            "Logging cold-start evaluation",
            user_id=user_id,
            model=model_name
        )
        
        combined_score = (
            profile_alignment * 0.6 +
            ground_truth_quality * 0.4
        )
        
        # Log to MLflow
        try:
            with mlflow.start_run(run_name=f"coldstart_eval_{user_id}"):
                mlflow.log_param("user_id", user_id)
                mlflow.log_param("model", model_name)
                mlflow.log_param("evaluation_type", "cold_start")
                
                mlflow.log_metric("profile_alignment", profile_alignment)
                mlflow.log_metric("ground_truth_quality", ground_truth_quality)
                mlflow.log_metric("combined_score", combined_score)
                mlflow.log_metric("recommendation_count", recommendation_count)
                
                # Check if passes threshold
                passes = (
                    profile_alignment >= settings.COLD_START_PROFILE_ALIGNMENT_THRESHOLD and
                    ground_truth_quality >= settings.COLD_START_GROUND_TRUTH_THRESHOLD
                )
                mlflow.log_metric("passes_threshold", 1.0 if passes else 0.0)
                
                logger.debug("Cold-start evaluation logged to MLflow")
                
        except Exception as e:
            logger.warning(
                "MLflow logging failed",
                error=str(e)
            )
        
        # Save to database
        try:
            query = """
                INSERT INTO cold_start_evaluations (
                    user_id, embedding_model,
                    profile_alignment, ground_truth_quality,
                    combined_score, recommendation_count
                )
                VALUES ($1, $2, $3, $4, $5, $6)
            """
            
            await self.db.execute(
                query,
                user_id,
                model_name,
                profile_alignment,
                ground_truth_quality,
                combined_score,
                recommendation_count
            )
            
            logger.info(
                "Cold-start evaluation saved",
                user_id=user_id,
                combined_score=combined_score
            )
            
        except Exception as e:
            logger.error(
                "Failed to save cold-start evaluation",
                error=str(e),
                exc_info=True
            )