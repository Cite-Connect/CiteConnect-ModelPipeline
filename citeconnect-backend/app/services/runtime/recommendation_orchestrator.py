"""
Recommendation orchestrator - central coordination of recommendation generation.
Coordinates the Service (Chef), Evaluation (Inspector), and Experiment (Logger).
"""
from typing import List, Dict, Optional
import time
from app.utils.logger import get_logger
from app.services.recommendation_service import RecommendationService
from app.services.evaluation_service import EvaluationService
from app.services.bootstrap.experiment_service import ExperimentService
from app.services.runtime.user_state_service import UserStateService

logger = get_logger(__name__)

class RecommendationOrchestrator:
    """
    Orchestrates recommendation generation.
    Refactored to delegate logic to RecommendationService and handle A/B testing/Logging.
    """
    
    def __init__(
        self,
        rec_service: RecommendationService,
        eval_service: EvaluationService,
        experiment_service: ExperimentService,
        user_state_service: UserStateService
    ):
        """
        Initialize orchestrator with required services.
        
        Args:
            rec_service: Core engine for retrieving and scoring papers.
            eval_service: Service for quality checks and metrics.
            experiment_service: Service for logging events and A/B routing.
            user_state_service: Service for retrieving user context/profile.
        """
        self.rec_service = rec_service
        self.eval_service = eval_service
        self.experiment_service = experiment_service
        self.user_state_service = user_state_service
        
        logger.info("RecommendationOrchestrator initialized (Delegation Mode)")
    
    async def generate_recommendations(
        self,
        user_id: int,
        model_name: Optional[str] = None, 
        count: int = 10,
        filters: Optional[Dict] = None
    ) -> Dict:
        """
        Generate recommendations by coordinating services.
        
        Flow:
        1. Context: Get user profile and A/B test variant.
        2. Generate: Delegate to RecommendationService (The "Chef").
        3. Fallback: If Chef fails, fetch trending papers.
        4. Evaluate: Run online quality check (The "Inspector").
        5. Log: Record event for offline analysis (The "Scribe").
        """
        start_time = time.time()
        
        # ---------------------------------------------------------
        # 1. Context & Strategy Assignment
        # ---------------------------------------------------------
        logger.info(
            "🎯 ORCHESTRATOR START",
            user_id=user_id,
            model_name=model_name,
            count=count
        )
        
        start_time = time.time()
        try:
            logger.info("Step 1: Getting user context")
            user_context = await self.user_state_service.get_user_context(user_id)
        except Exception as e:
            logger.warning(f"Failed to get user context: {e}. Using defaults.")
            user_context = {'stage': 'cold_start', 'profile': {}}

        # If model not forced by API, ask Experiment Service (A/B Testing)
        if not model_name:
            if self.experiment_service:
                try:
                    model_name = await self.experiment_service.get_active_variant(user_id)
                except Exception as e:
                    logger.error(f"Experiment service failed: {e}")
                    model_name = 'minilm'
            else:
                model_name = 'minilm' # Default fallback

        logger.info(
            "Generating recommendations",
            user_id=user_id,
            model=model_name,
            stage=user_context.get('stage')
        )

        recommendations = []
        strategy_used = "hybrid_service"
        
        # ---------------------------------------------------------
        # 2. Generation (The "Chef")
        # ---------------------------------------------------------
        try:
            # The RecommendationService handles the complex Cold/Warm logic internally.
            result = await self.rec_service.generate_recommendations(
                user_id=user_id,
                count=count,
                model=model_name,
                scoring_weights=None # Use Service defaults
            )
            recommendations = result.get('papers', [])
            
        except Exception as e:
            logger.error(f"Primary recommendation strategy failed: {e}", exc_info=True)
            
            # FALLBACK: If the main service crashes (e.g., Vector DB down), 
            # fall back to a safe "Trending" query that relies only on SQL.
            logger.warning("Falling back to Trending Papers")
            try:
                # Access trending papers via the repo attached to the service
                domain = user_context.get('profile', {}).get('primary_domain', 'machine_learning')
                recommendations = await self.rec_service.paper_repo.get_trending_papers(
                    domain=domain,
                    limit=count
                )
                
                # Convert raw rows to dicts and add dummy metadata
                recommendations = [dict(p) for p in recommendations]
                for p in recommendations:
                    p['relevance_explanation'] = "Popular in your field (Fallback)"
                    p['final_score'] = 0.5 # Default score
                
                strategy_used = "fallback_trending"
                
            except Exception as fallback_error:
                logger.error(f"Fallback strategy also failed: {fallback_error}")
                recommendations = []

        if not recommendations:
            # If everything fails, return empty list (let frontend handle empty state)
            logger.error(f"Failed to generate recommendations for user {user_id}")
            return {
                "recommendations": [],
                "metadata": {"error": "Generation failed"}
            }

        # ---------------------------------------------------------
        # 3. Evaluation (The "Inspector")
        # ---------------------------------------------------------
        # Run a quick online check. Set store_result=False to keep it fast.
        eval_report = {}
        try:
            if user_context.get('stage') == 'cold_start':
                eval_report = await self.eval_service.evaluate_cold_start_recommendations(
                    user_id=user_id,
                    recommendations=recommendations,
                    model=model_name,
                    store_result=False 
                )
            else:
                eval_report = await self.eval_service.evaluate_warm_start_recommendations(
                    user_id=user_id,
                    recommendations=recommendations,
                    store_result=False
                )
                
            # Optional Guardrail: Log warning if quality is very low
            score = eval_report.get('combined_score') or eval_report.get('precision_at_10', 0)
            if score < 0.2:
                logger.warning(
                    "Low quality recommendations generated", 
                    score=score,
                    user_id=user_id
                )
        except Exception as e:
            logger.warning(f"Online evaluation failed (non-blocking): {e}")

        # ---------------------------------------------------------
        # 4. Logging (The "Scribe")
        # ---------------------------------------------------------
        generation_time = (time.time() - start_time) * 1000
        
        if self.experiment_service:
            try:
                # Log event for MLflow/Offline analysis
                # Ideally, this should be non-blocking (fire and forget)
                eval_report_with_strategy = eval_report.copy()
                eval_report_with_strategy['strategy_used'] = strategy_used
                await self.experiment_service.log_recommendation_event(
                    user_id=user_id,
                    model_name=model_name,
                    recommendations=recommendations,
                    evaluation_scores=eval_report_with_strategy,
                    user_context=user_context,
                    generation_time_ms=generation_time
                )
            except Exception as e:
                logger.error(f"Failed to log experiment event: {e}")

        # ---------------------------------------------------------
        # 5. Response Construction
        # ---------------------------------------------------------
        response = {
        "recommendations": recommendations,
        "metadata": {
            "user_stage": user_context.get('stage'),
            "strategy_used": strategy_used,
            "model_used": model_name,
            "evaluation_score": eval_report.get('combined_score') or eval_report.get('precision_at_10',0.0),
            "evaluation_scores": eval_report,  # ADD THIS - full eval report
            "cache_hit": False,  # ADD THIS - set to False for now
            "generation_time_ms": round(generation_time, 2)
            }
        }
        
        return response