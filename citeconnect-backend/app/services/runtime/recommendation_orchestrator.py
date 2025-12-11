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
        search_query: Optional[str] = None,  # ← NEW PARAMETER
        filters: Optional[Dict] = None
    ) -> Dict:
        """
        Generate recommendations by coordinating services.
        Now supports search-augmented mode.
        """
        # CRITICAL: Store original search_query BEFORE any modifications
        original_search_query = search_query
        logger.info(
            "Orchestrator.generate_recommendations called",
            user_id=user_id,
            search_query=search_query,
            search_query_type=type(search_query).__name__,
            search_query_len=len(search_query) if search_query else 0,
            original_search_query=original_search_query,
            original_search_query_type=type(original_search_query).__name__
        )
        start_time = time.time()
        
        # Get user context
        try:
            user_context = await self.user_state_service.get_user_context(user_id)
        except Exception as e:
            logger.warning(f"Failed to get user context: {e}. Using defaults.")
            user_context = {'stage': 'cold_start', 'profile': {}}

        # Determine model
        if not model_name:
            if self.experiment_service:
                try:
                    model_name = await self.experiment_service.get_active_variant(user_id)
                except Exception as e:
                    logger.error(f"Experiment service failed: {e}")
                    model_name = 'minilm'
            else:
                model_name = 'minilm'

        logger.info(
            "Generating recommendations",
            user_id=user_id,
            model=model_name,
            stage=user_context.get('stage'),
            has_search_query=bool(search_query)  # ← NEW
        )

        recommendations = []
        search_result_metadata = {}  # Initialize to avoid scoping issues
        
        # ────────────────────────────────────────────────────────
        # ROUTE BASED ON SEARCH QUERY
        # ────────────────────────────────────────────────────────
        if search_query and search_query.strip():
            # SEARCH-AUGMENTED MODE
            strategy_used = "search_augmented"
            
            try:
                logger.info(
                    "Using search-augmented strategy",
                    search_query=search_query[:100],
                    original_search_query=original_search_query
                )
                
                result = await self.rec_service.generate_search_augmented_recommendations(
                    user_id=user_id,
                    search_query=search_query,
                    count=count,
                    model=model_name
                )
                recommendations = result.get('papers', [])
                # Store LLM refinement info from result
                search_result_metadata = {
                    'refined_query': result.get('refined_query'),
                    'llm_refinement_used': result.get('llm_refinement_used', False)
                }
                logger.info(
                    "Search-augmented result received",
                    papers_count=len(recommendations),
                    refined_query=result.get('refined_query'),
                    llm_refinement_used=result.get('llm_refinement_used', False),
                    search_query_from_result=result.get('search_query')
                )
                # Ensure original_search_query is preserved
                # The result should contain the original search_query, but if it doesn't, keep what we have
                result_search_query = result.get('search_query')
                if result_search_query and isinstance(result_search_query, str) and result_search_query.strip():
                    # Use the search_query from result (should be the original query)
                    original_search_query = result_search_query
                # If result doesn't have search_query but we have the input, preserve it
                elif not original_search_query or (isinstance(original_search_query, str) and not original_search_query.strip()):
                    if search_query and isinstance(search_query, str) and search_query.strip():
                        original_search_query = search_query
                
            except Exception as e:
                logger.error(f"Search-augmented strategy failed: {e}", exc_info=True)
                # Fallback to regular recommendations
                logger.warning("Falling back to profile-based recommendations")
                search_query = None  # Clear search to trigger fallback
                search_result_metadata = {}  # Reset metadata on fallback
        
        # ────────────────────────────────────────────────────────
        # FALLBACK: PROFILE-BASED MODE (Original Logic)
        # ────────────────────────────────────────────────────────
        if not recommendations:
            try:
                result = await self.rec_service.generate_recommendations(
                    user_id=user_id,
                    count=count,
                    model=model_name,
                    scoring_weights=None
                )
                recommendations = result.get('papers', [])
                strategy_used = result.get('method', 'hybrid_service')
                
            except Exception as e:
                logger.error(f"Primary recommendation strategy failed: {e}", exc_info=True)
                
                # Final fallback: Trending papers
                logger.warning("Falling back to Trending Papers")
                try:
                    domain = user_context.get('profile', {}).get('primary_domain', 'machine_learning')
                    recommendations = await self.rec_service.paper_repo.get_trending_papers(
                        domain=domain,
                        limit=count
                    )
                    
                    recommendations = [dict(p) for p in recommendations]
                    for p in recommendations:
                        p['relevance_explanation'] = "Popular in your field (Fallback)"
                        p['final_score'] = 0.5
                    
                    strategy_used = "fallback_trending"
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback strategy also failed: {fallback_error}")
                    recommendations = []

        if not recommendations:
            logger.error(f"Failed to generate recommendations for user {user_id}")
            return {
                "recommendations": [],
                "metadata": {"error": "Generation failed"}
            }
        # ────────────────────────────────────────────────────────
        # SANITIZATION: Cap Scores at 1.0 (Fix for ResponseValidationError)
        # ────────────────────────────────────────────────────────
        if recommendations:
            for paper in recommendations:
                # Check known score keys and clamp to max 1.0
                # 'relevance_score' is the key seen in validation errors
                for key in ['relevance_score', 'final_score', 'score']:
                    if key in paper and isinstance(paper[key], (int, float)):
                        original_val = paper[key]
                        if original_val > 1.0:
                            paper[key] = 1.0
                            # Optional: could log debug here if needed

        if not recommendations:
            logger.error(f"Failed to generate recommendations for user {user_id}")
            return {
                "recommendations": [],
                "metadata": {"error": "Generation failed"}
            }
        # ────────────────────────────────────────────────────────
        # EVALUATION
        # ────────────────────────────────────────────────────────
        eval_report = {}
        try:
            if user_context.get('stage') == 'cold_start':
                eval_report = await self.eval_service.evaluate_cold_start_recommendations(
                    user_id=user_id,
                    recommendations=recommendations,
                    model=model_name,
                    store_result=True 
                )
            else:
                eval_report = await self.eval_service.evaluate_warm_start_recommendations(
                    user_id=user_id,
                    recommendations=recommendations,
                    store_result=True
                )
                
            score = eval_report.get('combined_score') or eval_report.get('precision_at_10', 0)
            if score < 0.2:
                logger.warning(
                    "Low quality recommendations generated", 
                    score=score,
                    user_id=user_id
                )
        except Exception as e:
            logger.warning(f"Online evaluation failed (non-blocking): {e}")

        # ────────────────────────────────────────────────────────
        # LOGGING
        # ────────────────────────────────────────────────────────
        generation_time = (time.time() - start_time) * 1000
        
        if self.experiment_service:
            try:
                await self.experiment_service.log_recommendation_event(
                    user_id=user_id,
                    model_name=model_name,
                    recommendations=recommendations,
                    evaluation_scores=eval_report,
                    user_context=user_context,
                    generation_time_ms=generation_time
                )
            except Exception as e:
                logger.error(f"Failed to log experiment event: {e}")

        # ────────────────────────────────────────────────────────
        # RESPONSE
        # ────────────────────────────────────────────────────────
        logger.info(
            "Building response metadata",
            original_search_query=original_search_query,
            original_search_query_type=type(original_search_query).__name__,
            strategy_used=strategy_used
        )
        metadata = {
            "user_stage": user_context.get('stage'),
            "strategy_used": strategy_used,
            "model_used": model_name,
            "search_query": original_search_query,
            "evaluation_score": eval_report.get('combined_score') or eval_report.get('precision_at_10', 0.0),
            "evaluation_scores": eval_report if eval_report else {},
            "cache_hit": False,
            "generation_time_ms": round(generation_time, 2),
        }
        
        logger.info(
            "Metadata built",
            search_query_in_metadata=metadata.get('search_query'),
            strategy=metadata.get('strategy_used')
        )
        
        # Add LLM refinement info if available (from search-augmented mode)
        logger.info(
            "Adding LLM metadata to response",
            search_result_metadata_exists=bool(search_result_metadata),
            search_result_metadata_keys=list(search_result_metadata.keys()) if search_result_metadata else [],
            refined_query=search_result_metadata.get('refined_query') if search_result_metadata else None,
            llm_refinement_used=search_result_metadata.get('llm_refinement_used', False) if search_result_metadata else False
        )
        if search_result_metadata:
            metadata['refined_query'] = search_result_metadata.get('refined_query')
            metadata['llm_refinement_used'] = search_result_metadata.get('llm_refinement_used', False)
            logger.info(
                "LLM metadata added to response",
                refined_query_in_metadata=metadata.get('refined_query'),
                llm_refinement_used_in_metadata=metadata.get('llm_refinement_used')
            )
        else:
            logger.warning("No search_result_metadata - LLM info not added")
        
        response = {
            "recommendations": recommendations,
            "metadata": metadata
        }
        
        return response