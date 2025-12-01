"""
Recommendation API endpoints.
Provides paper recommendations with personalization and evaluation.
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

logger = get_logger(__name__)

router = APIRouter()


def get_recommendation_orchestrator(request: Request) -> RecommendationOrchestrator:
    """
    Dependency to get recommendation orchestrator from app state.
    """
    if not hasattr(request.app.state, 'recommendation_orchestrator'):
        logger.error("Recommendation orchestrator not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation service not available"
        )
    
    return request.app.state.recommendation_orchestrator


@router.post(
    "",
    response_model=RecommendationResponse,
    summary="Get personalized paper recommendations",
    description="Generate personalized paper recommendations for a user."
)
async def get_recommendations(
    request_data: RecommendationRequest,
    orchestrator: RecommendationOrchestrator = Depends(get_recommendation_orchestrator)
):
    """
    Generate paper recommendations.
    """
    logger.info(
        "Recommendation request received",
        user_id=request_data.user_id,
        count=request_data.count,
        model=request_data.model_preference,
        session_id=request_data.session_id
    )
    
    try:
        # Validate user
        if not request_data.user_id:
            logger.warning(
                "Anonymous recommendation request",
                session_id=request_data.session_id
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User ID is required for recommendations"
            )
        
        # Map 'minilm' shortcode to full model name if necessary
        # This handles the case where frontend/scripts send short names
        model_map = {
            'minilm': 'all-MiniLM-L6-v2',
            'specter': 'specter2'
        }
        model_name = model_map.get(request_data.model_preference, request_data.model_preference)

        # Generate recommendations
        result = await orchestrator.generate_recommendations(
            user_id=request_data.user_id,
            model_name=model_name,
            count=request_data.count,
            filters=request_data.filters.dict() if request_data.filters else None
        )
        
        logger.info(
            "Recommendations generated successfully",
            user_id=request_data.user_id,
            count=len(result.get('recommendations', [])),
            strategy=result.get('metadata', {}).get('strategy_used'),
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
                    "details": str(e) if logger.level == "DEBUG" else None
                }
            }
        )


@router.get(
    "/{user_id}/history",
    summary="Get recommendation history",
    description="Retrieve past recommendations for a user"
)
async def get_recommendation_history(
    user_id: int,
    limit: int = 10,
    db: DatabaseConnection = Depends(get_db)
):
    """
    Get user's recommendation history.
    """
    logger.info("Recommendation history requested", user_id=user_id, limit=limit)
    
    try:
        # This matches the column name usually created by the schema migration
        query = """
           SELECT 
                re.event_id,
                re.recommended_paper_ids,
                re.embedding_model,
                re.event_timestamp
            FROM recommendation_events re
            WHERE re.user_id = $1
            ORDER BY re.event_timestamp DESC
            LIMIT $2
        """
        
        results = await db.fetch(query, user_id, limit)
        
        history = [
            {
                "event_id": row['event_id'],
                "paper_ids": row['recommended_paper_ids'],
                "model": row['embedding_model'],
                "timestamp": row['event_timestamp'].isoformat()
            }
            for row in results
        ]
        
        return {
            "user_id": user_id,
            "history": history
        }
        
    except Exception as e:
        logger.error("History retrieval failed", user_id=user_id, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve recommendation history"
        )


@router.post(
    "/evaluate",
    summary="Evaluate recommendations",
    description="Evaluate a set of recommendations against ground truth and user profile"
)
async def evaluate_recommendations(
    user_id: int,
    paper_ids: List[str],
    db: DatabaseConnection = Depends(get_db)
):
    """
    Evaluate specific papers for a user using the EvaluationService.
    """
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
            raise HTTPException(status_code=404, detail="No papers found for provided IDs")
            
        recommendations = [dict(p) for p in papers]
        
        evaluation_result = await eval_service.evaluate_cold_start_recommendations(
            user_id=user_id,
            recommendations=recommendations,
            store_result=False 
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
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Evaluation failed"
        )