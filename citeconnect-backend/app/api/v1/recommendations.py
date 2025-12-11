"""
Recommendation API endpoints.
Provides paper recommendations with personalization and evaluation.
UPDATED: Added JWT authentication to all endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Request
from typing import Optional, List
from app.models.paper import (
    RecommendationRequest,
    RecommendationResponse,
    PaperResponse
)
from app.utils.logger import get_logger
from app.services.runtime.recommendation_orchestrator import RecommendationOrchestrator
from app.services.evaluation_service import EvaluationService
from app.db.repositories.paper_repo import PaperRepository
from app.db.repositories.user_repo import UserRepository
from app.db.connection import get_db, DatabaseConnection
from app.api.v1.auth import get_current_user  # NEW IMPORT

logger = get_logger(__name__)

router = APIRouter()


def get_recommendation_orchestrator(request: Request) -> RecommendationOrchestrator:
    """
    Dependency to get recommendation orchestrator from app state.
    """
    logger.info("Getting recommendation orchestrator from app state")
    
    if not hasattr(request.app.state, 'recommendation_orchestrator'):
        logger.error(
            "Recommendation orchestrator not initialized",
            available_attrs=dir(request.app.state)
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation service not available - service still starting up"
        )
    
    orchestrator = request.app.state.recommendation_orchestrator
    logger.info("Successfully retrieved recommendation orchestrator")
    return orchestrator


@router.post(
    "",
    response_model=RecommendationResponse,
    summary="Get personalized paper recommendations",
    description="Generate personalized paper recommendations. **Requires authentication.**"
)
async def get_recommendations(
    request_data: RecommendationRequest,
    current_user: dict = Depends(get_current_user),  # AUTHENTICATION REQUIRED
    orchestrator: RecommendationOrchestrator = Depends(get_recommendation_orchestrator)
):
    """
    Generate paper recommendations.
    Supports optional search query for search-augmented mode.
    User can only request recommendations for themselves.
    """
    
    # Verify user is requesting their own recommendations
    if request_data.user_id != current_user['user_id']:
        logger.warning(
            "Unauthorized recommendation request",
            requesting_user=current_user['user_id'],
            target_user=request_data.user_id
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only request recommendations for yourself"
        )
    
    logger.info(
        "Recommendation request received",
        user_id=request_data.user_id,
        count=request_data.count,
        model=request_data.model_preference,
        session_id=request_data.session_id,
        has_search_query=bool(request_data.search_query)
    )
    
    try:
        # Map model names
        model_map = {
            'minilm': 'all-MiniLM-L6-v2',
            'specter2': 'specter',
            'specter': 'specter'
        }
        model_name = model_map.get(request_data.model_preference, request_data.model_preference)

        # Generate recommendations (with optional search)
        logger.info(
            "Calling orchestrator",
            user_id=request_data.user_id,
            search_query=request_data.search_query,
            search_query_type=type(request_data.search_query).__name__
        )
        result = await orchestrator.generate_recommendations(
            user_id=request_data.user_id,
            model_name=model_name,
            count=request_data.count,
            search_query=request_data.search_query,
            filters=request_data.filters.dict() if request_data.filters else None
        )
        logger.info(
            "Orchestrator returned",
            search_query_in_metadata=result.get('metadata', {}).get('search_query'),
            strategy=result.get('metadata', {}).get('strategy_used')
        )
        
        logger.info(
            "Recommendations generated successfully",
            user_id=request_data.user_id,
            count=len(result.get('recommendations', [])),
            strategy=result.get('metadata', {}).get('strategy_used'),
            search_query=request_data.search_query[:50] if request_data.search_query else None,
            time_ms=result.get('metadata', {}).get('generation_time_ms')
        )
        
        return result
        
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(
            "Recommendation generation failed",
            user_id=request_data.user_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "RECOMMENDATION_FAILED",
                    "message": "Failed to generate recommendations",
                    "details": str(e)
                }
            }
        )
@router.post(
    "/test",
    response_model=RecommendationResponse,
    summary="Test recommendations (no auth)",
    description="For testing/simulations only - bypasses authentication"
)
async def get_recommendations_test(
    request_data: RecommendationRequest,
    orchestrator: RecommendationOrchestrator = Depends(get_recommendation_orchestrator)
):
    """Generate recommendations without authentication (testing only)."""
    model_map = {'minilm': 'all-MiniLM-L6-v2', 'specter': 'specter', 'specter2': 'specter'}
    model_name = model_map.get(request_data.model_preference, request_data.model_preference)

    result = await orchestrator.generate_recommendations(
        user_id=request_data.user_id,
        model_name=model_name,
        count=request_data.count,
        search_query=request_data.search_query,
        filters=request_data.filters.dict() if request_data.filters else None
    )
    
    return result

@router.get(
    "/{user_id}/history",
    summary="Get recommendation history",
    description="Retrieve past recommendations. **Requires authentication.**"
)
async def get_recommendation_history(
    user_id: int,
    limit: int = 10,
    current_user: dict = Depends(get_current_user),  # AUTHENTICATION REQUIRED
    db: DatabaseConnection = Depends(get_db)
):
    """
    Get user's recommendation history.
    Uses InteractionRepository - NO SQL HERE.
    """
    
    # Verify user is accessing their own history
    if user_id != current_user['user_id']:
        logger.warning(
            "Unauthorized history access attempt",
            requesting_user=current_user['user_id'],
            target_user=user_id
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only view your own recommendation history"
        )
    
    logger.info(
        "Recommendation history requested", 
        user_id=user_id, 
        limit=limit
    )
    
    try:
        from app.db.repositories.interaction_repo import InteractionRepository
        interaction_repo = InteractionRepository(db)
        
        # Use repository method instead of direct SQL
        history_results = await interaction_repo.get_recommendation_history(
            user_id=user_id,
            limit=limit
        )
        
        history = [
            {
                "event_id": row['event_id'],
                "paper_ids": row['recommended_paper_ids'],
                "model": row['embedding_model'],
                "timestamp": row['event_timestamp'].isoformat()
            }
            for row in history_results
        ]
        
        return {
            "user_id": user_id,
            "history": history
        }
        
    except Exception as e:
        logger.error(
            "History retrieval failed", 
            user_id=user_id, 
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve recommendation history"
        )

@router.post(
    "/evaluate",
    summary="Evaluate recommendations",
    description="Evaluate recommendations against ground truth. **Requires authentication.**"
)
async def evaluate_recommendations(
    user_id: int,
    paper_ids: List[str],
    current_user: dict = Depends(get_current_user),  # AUTHENTICATION REQUIRED
    db: DatabaseConnection = Depends(get_db)
):
    """
    Evaluate specific papers for a user using the EvaluationService.
    User can only evaluate recommendations for themselves.
    """
    
    # Verify user is evaluating their own recommendations
    if user_id != current_user['user_id']:
        logger.warning(
            "Unauthorized evaluation attempt",
            requesting_user=current_user['user_id'],
            target_user=user_id
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only evaluate recommendations for yourself"
        )
    
    logger.info(
        "Evaluation requested",
        user_id=user_id,
        paper_count=len(paper_ids)
    )
    
    try:
        eval_service = EvaluationService(db)
        paper_repo = PaperRepository(db)
        
        papers = await paper_repo.find_by_ids(paper_ids)
        if not papers:
            raise HTTPException(
                status_code=404, 
                detail="No papers found for provided IDs"
            )
            
        recommendations = [dict(p) for p in papers]
        
        evaluation_result = await eval_service.evaluate_cold_start_recommendations(
            user_id=user_id,
            recommendations=recommendations,
            store_result=True 
        )
        
        return {
            "user_id": user_id,
            "paper_count": len(recommendations),
            "metrics": {
                "combined_score": evaluation_result.get('combined_score'),
                "profile_alignment": evaluation_result.get('profile_alignment'),
                "ground_truth_quality": evaluation_result.get('ground_truth_quality'),
                "diversity_score": evaluation_result.get('diversity_score')
            },
            "passed_threshold": evaluation_result.get('passes_threshold')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Evaluation failed",
            user_id=user_id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Evaluation failed"
        )