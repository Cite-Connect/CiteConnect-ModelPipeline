"""
Recommendation API endpoints.
Provides paper recommendations with personalization and evaluation.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Request
from typing import Optional
from app.models.paper import (
    RecommendationRequest,
    RecommendationResponse,
    PaperResponse
)
from app.utils.logger import get_logger
from app.services.runtime.recommendation_orchestrator import RecommendationOrchestrator
from app.db.connection import get_db, DatabaseConnection

logger = get_logger(__name__)

router = APIRouter()


def get_recommendation_orchestrator(request: Request) -> RecommendationOrchestrator:
    """
    Dependency to get recommendation orchestrator from app state.
    
    Args:
        request: FastAPI request
        
    Returns:
        RecommendationOrchestrator: Orchestrator instance
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
    description="""
    Generate personalized paper recommendations for a user.
    
    Features:
    - **Cold-start support**: Works from day one using user profile
    - **Multi-model**: Compare all-MiniLM-L6-v2 vs SPECTER
    - **Fallback strategies**: Guaranteed recommendations even with failures
    - **Quality evaluation**: Real-time metrics on recommendation quality
    - **Diversity**: Ensures variety in authors, venues, and topics
    """
)
async def get_recommendations(
    request_data: RecommendationRequest,
    orchestrator: RecommendationOrchestrator = Depends(get_recommendation_orchestrator)
):
    """
    Generate paper recommendations.
    
    Args:
        request_data: Recommendation request parameters
        orchestrator: Recommendation orchestrator service
        
    Returns:
        RecommendationResponse: Recommendations with metadata
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
        
        # Generate recommendations
        result = await orchestrator.generate_recommendations(
            user_id=request_data.user_id,
            model_name=request_data.model_preference,
            count=request_data.count,
            filters=request_data.filters.dict() if request_data.filters else None
        )
        
        logger.info(
            "Recommendations generated successfully",
            user_id=request_data.user_id,
            count=len(result['recommendations']),
            strategy=result['metadata']['strategy_used'],
            time_ms=result['metadata']['generation_time_ms']
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
    
    Args:
        user_id: User identifier
        limit: Maximum records to return
        db: Database connection
        
    Returns:
        List of past recommendations
    """
    logger.info(
        "Recommendation history requested",
        user_id=user_id,
        limit=limit
    )
    
    try:
        query = """
            SELECT 
                re.event_id,
                re.recommended_paper_ids,
                re.recommendation_strategy,
                re.model_used,
                re.created_at
            FROM recommendation_events re
            WHERE re.user_id = $1
            ORDER BY re.created_at DESC
            LIMIT $2
        """
        
        results = await db.fetch(query, user_id, limit)
        
        history = [
            {
                "event_id": row['event_id'],
                "paper_ids": row['recommended_paper_ids'],
                "strategy": row['recommendation_strategy'],
                "model": row['embedding_model'],
                "timestamp": row['event_timestamp'].isoformat()
            }
            for row in results
        ]
        
        logger.info(
            "Recommendation history retrieved",
            user_id=user_id,
            count=len(history)
        )
        
        return {
            "user_id": user_id,
            "history": history
        }
        
    except Exception as e:
        logger.error(
            "History retrieval failed",
            user_id=user_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve recommendation history"
        )


@router.post(
    "/evaluate",
    summary="Evaluate recommendations",
    description="Evaluate a set of recommendations against ground truth"
)
async def evaluate_recommendations(
    user_id: int,
    paper_ids: list[str],
    db: DatabaseConnection = Depends(get_db)
):
    """
    Evaluate recommendations for quality metrics.
    
    Args:
        user_id: User identifier
        paper_ids: Paper IDs to evaluate
        db: Database connection
        
    Returns:
        Evaluation metrics
    """
    logger.info(
        "Evaluation requested",
        user_id=user_id,
        paper_count=len(paper_ids)
    )
    
    try:
        # This would use the evaluation service
        # For now, return placeholder
        
        evaluation = {
            "user_id": user_id,
            "paper_count": len(paper_ids),
            "metrics": {
                "ground_truth_quality": 0.0,
                "profile_alignment": 0.0,
                "diversity_score": 0.0
            },
            "message": "Evaluation service integration pending"
        }
        
        logger.info(
            "Evaluation complete",
            user_id=user_id
        )
        
        return evaluation
        
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