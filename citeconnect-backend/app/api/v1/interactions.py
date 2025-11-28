"""
Interaction tracking API endpoints.
Handles user-paper interactions for personalization and evaluation.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Optional
from app.models.paper import PaperInteractionRequest
from app.utils.logger import get_logger
from app.db.connection import get_db, DatabaseConnection
from app.db.repositories.interaction_repo import InteractionRepository
from app.db.repositories.user_repo import UserRepository

logger = get_logger(__name__)

router = APIRouter()


def get_interaction_repo(
    db: DatabaseConnection = Depends(get_db)
) -> InteractionRepository:
    """
    Dependency to get interaction repository.
    
    Args:
        db: Database connection
        
    Returns:
        InteractionRepository: Interaction repository instance
    """
    return InteractionRepository(db)


def get_user_repo(db: DatabaseConnection = Depends(get_db)) -> UserRepository:
    """Dependency to get user repository."""
    return UserRepository(db)


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Track user-paper interaction",
    description="""
    Record a user interaction with a paper.
    
    Interaction types and their weights:
    - **cite** (1.0): Strongest positive signal
    - **save** (0.8): Very strong interest
    - **download** (0.7): Strong interest
    - **like** (0.6): Positive signal
    - **click** (0.3): Mild interest
    - **view** (0.2): Weak signal
    - **dismiss** (-0.2): Mild negative
    - **not_interested** (-0.5): Strong negative
    
    These signals are used to update user embeddings and improve recommendations.
    """
)
async def track_interaction(
    user_id: int,
    interaction_data: PaperInteractionRequest,
    interaction_repo: InteractionRepository = Depends(get_interaction_repo),
    user_repo: UserRepository = Depends(get_user_repo)
):
    """
    Track user-paper interaction.
    
    Args:
        user_id: User identifier
        interaction_data: Interaction details
        interaction_repo: Interaction repository
        user_repo: User repository
        
    Returns:
        Interaction confirmation with embedding update status
    """
    logger.info(
        "Interaction tracking request",
        user_id=user_id,
        paper_id=interaction_data.paper_id,
        interaction_type=interaction_data.interaction_type
    )
    
    try:
        # Create interaction record
        interaction = await interaction_repo.create_interaction(
            user_id=user_id,
            paper_id=interaction_data.paper_id,
            interaction_type=interaction_data.interaction_type,
            duration_seconds=interaction_data.duration_seconds,
            source=interaction_data.context.source if interaction_data.context else None,
            position=interaction_data.context.position if interaction_data.context else None,
            session_id=interaction_data.context.session_id if interaction_data.context else None
        )
        
        # Increment user interaction count
        state = await user_repo.get_recommendation_state(user_id)
        if state:
            new_count = state.get('interaction_count', 0) + 1
            await user_repo.update_recommendation_state(
                user_id,
                {'interaction_count': new_count}
            )
        
        # Check if negative signal - add filter if needed
        if interaction_data.interaction_type in ['dismiss', 'not_interested']:
            await interaction_repo.add_paper_filter(
                user_id=user_id,
                paper_id=interaction_data.paper_id,
                filter_type='not_interested',
                reason=f"User {interaction_data.interaction_type}"
            )
            
            logger.info(
                "Paper filtered due to negative interaction",
                user_id=user_id,
                paper_id=interaction_data.paper_id
            )
        
        # Check if embedding update should be triggered
        should_update = await interaction_repo.check_embedding_update_trigger(
            user_id
        )
        
        if should_update:
            logger.info(
                "Embedding update triggered",
                user_id=user_id
            )
            # In production, this would queue a background task
            # celery_app.send_task('update_user_embedding', args=[user_id])
        
        logger.info(
            "Interaction tracked successfully",
            user_id=user_id,
            paper_id=interaction_data.paper_id,
            strength=interaction['interaction_strength']
        )
        
        return {
            "interaction_id": interaction['interaction_id'],
            "user_id": user_id,
            "paper_id": interaction_data.paper_id,
            "interaction_type": interaction_data.interaction_type,
            "strength": float(interaction['interaction_strength']),
            "embedding_update_triggered": should_update,
            "message": "Interaction tracked successfully"
        }
        
    except Exception as e:
        logger.error(
            "Interaction tracking failed",
            user_id=user_id,
            paper_id=interaction_data.paper_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Interaction tracking failed"
        )


@router.get(
    "/{user_id}/history",
    summary="Get user interaction history",
    description="Retrieve user's past interactions with papers"
)
async def get_interaction_history(
    user_id: int,
    limit: Optional[int] = 50,
    min_strength: Optional[float] = None,
    interaction_repo: InteractionRepository = Depends(get_interaction_repo)
):
    """
    Get user's interaction history.
    
    Args:
        user_id: User identifier
        limit: Maximum interactions to return
        min_strength: Optional minimum interaction strength
        interaction_repo: Interaction repository
        
    Returns:
        List of user interactions
    """
    logger.info(
        "Interaction history request",
        user_id=user_id,
        limit=limit
    )
    
    try:
        interactions = await interaction_repo.get_user_interactions(
            user_id=user_id,
            limit=limit,
            min_strength=min_strength
        )
        
        logger.info(
            "Interaction history retrieved",
            user_id=user_id,
            count=len(interactions)
        )
        
        return {
            "user_id": user_id,
            "interaction_count": len(interactions),
            "interactions": [dict(i) for i in interactions]
        }
        
    except Exception as e:
        logger.error(
            "Interaction history retrieval failed",
            user_id=user_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Interaction history retrieval failed"
        )


@router.get(
    "/{user_id}/statistics",
    summary="Get interaction statistics",
    description="Get aggregated statistics about user interactions"
)
async def get_interaction_statistics(
    user_id: int,
    days: int = 30,
    interaction_repo: InteractionRepository = Depends(get_interaction_repo)
):
    """
    Get user interaction statistics.
    
    Args:
        user_id: User identifier
        days: Look back period in days
        interaction_repo: Interaction repository
        
    Returns:
        Interaction statistics
    """
    logger.info(
        "Interaction statistics request",
        user_id=user_id,
        days=days
    )
    
    try:
        # Get interaction counts by type
        counts = await interaction_repo.get_interaction_counts(
            user_id=user_id,
            days=days
        )
        
        # Get meaningful interactions count
        meaningful_count = await interaction_repo.get_meaningful_interactions_count(
            user_id
        )
        
        # Get domains explored
        domains_count = await interaction_repo.get_domains_explored(
            user_id
        )
        
        # Calculate totals
        total_interactions = sum(counts.values())
        positive_interactions = sum(
            count for interaction_type, count in counts.items()
            if interaction_type in ['cite', 'save', 'download', 'like', 'click']
        )
        negative_interactions = sum(
            count for interaction_type, count in counts.items()
            if interaction_type in ['dismiss', 'not_interested']
        )
        
        statistics = {
            "user_id": user_id,
            "period_days": days,
            "total_interactions": total_interactions,
            "positive_interactions": positive_interactions,
            "negative_interactions": negative_interactions,
            "meaningful_interactions": meaningful_count,
            "domains_explored": domains_count,
            "interaction_breakdown": counts,
            "engagement_rate": (
                positive_interactions / total_interactions
                if total_interactions > 0 else 0.0
            )
        }
        
        logger.info(
            "Interaction statistics calculated",
            user_id=user_id,
            total=total_interactions
        )
        
        return statistics
        
    except Exception as e:
        logger.error(
            "Interaction statistics calculation failed",
            user_id=user_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Statistics calculation failed"
        )


@router.get(
    "/{user_id}/saved",
    summary="Get saved papers",
    description="Retrieve papers user has saved"
)
async def get_saved_papers(
    user_id: int,
    interaction_repo: InteractionRepository = Depends(get_interaction_repo)
):
    """
    Get user's saved papers.
    
    Args:
        user_id: User identifier
        interaction_repo: Interaction repository
        
    Returns:
        List of saved papers
    """
    logger.info("Saved papers request", user_id=user_id)
    
    try:
        saved_papers = await interaction_repo.get_saved_papers(user_id)
        
        logger.info(
            "Saved papers retrieved",
            user_id=user_id,
            count=len(saved_papers)
        )
        
        return {
            "user_id": user_id,
            "saved_count": len(saved_papers),
            "saved_papers": [dict(p) for p in saved_papers]
        }
        
    except Exception as e:
        logger.error(
            "Saved papers retrieval failed",
            user_id=user_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Saved papers retrieval failed"
        )


@router.delete(
    "/{user_id}/filters/{paper_id}",
    summary="Remove paper filter",
    description="Remove a paper from user's filter list"
)
async def remove_paper_filter(
    user_id: int,
    paper_id: str,
    db: DatabaseConnection = Depends(get_db)
):
    """
    Remove paper filter.
    
    Args:
        user_id: User identifier
        paper_id: Paper identifier
        db: Database connection
        
    Returns:
        Confirmation message
    """
    logger.info(
        "Filter removal request",
        user_id=user_id,
        paper_id=paper_id
    )
    
    try:
        query = """
            DELETE FROM user_paper_filters
            WHERE user_id = $1 AND paper_id = $2
        """
        
        result = await db.execute(query, user_id, paper_id)
        
        if result == "DELETE 0":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Filter not found"
            )
        
        logger.info(
            "Filter removed",
            user_id=user_id,
            paper_id=paper_id
        )
        
        return {
            "user_id": user_id,
            "paper_id": paper_id,
            "message": "Filter removed successfully"
        }
        
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(
            "Filter removal failed",
            user_id=user_id,
            paper_id=paper_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Filter removal failed"
        )