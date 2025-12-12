"""
Complete interaction tracking endpoint with all table updates.
Updates: user_interactions, user_recommendation_state (with stage transitions),
user_saved_papers, user_liked_papers, user_paper_filters, and triggers embedding regeneration.
UPDATED: Added JWT authentication to all endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from typing import Optional
from app.models.paper import PaperInteractionRequest
from app.utils.logger import get_logger
from app.db.connection import get_db, DatabaseConnection
from app.db.repositories.interaction_repo import InteractionRepository
from app.db.repositories.user_repo import UserRepository
from app.api.v1.auth import get_current_user  # FIXED IMPORT PATH
from app.services.user_embedding_service import UserEmbeddingService

logger = get_logger(__name__)

router = APIRouter()


def get_interaction_repo(
    db: DatabaseConnection = Depends(get_db)
) -> InteractionRepository:
    """Dependency to get interaction repository."""
    return InteractionRepository(db)


def get_user_repo(db: DatabaseConnection = Depends(get_db)) -> UserRepository:
    """Dependency to get user repository."""
    return UserRepository(db)


async def regenerate_embeddings_background(user_id: int, db: DatabaseConnection):
    """Background task to regenerate user embeddings."""
    try:
        embedding_service = UserEmbeddingService(db)
        await embedding_service.generate_user_embeddings(user_id)
        logger.info("Background embedding regeneration complete", user_id=user_id)
    except Exception as e:
        logger.error(
            "Background embedding regeneration failed",
            user_id=user_id,
            error=str(e),
            exc_info=True
        )


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Track user-paper interaction with full table updates",
    description="""
    Record a user interaction and update all related tables:
    
    **Tables Updated:**
    1. user_interactions - records the interaction
    2. user_recommendation_state - increments count, transitions stage
    3. user_saved_papers - adds paper if interaction is 'save'
    4. user_liked_papers - adds paper if interaction is 'like'
    5. user_paper_filters - filters paper if negative interaction
    6. user_embeddings_* - triggers regeneration every 5+ meaningful interactions
    
    **Interaction Types & Weights:**
    - cite (1.0), save (0.8), download (0.7), like (0.6)
    - click (0.3), view (0.2)
    - dismiss (-0.2), not_interested (-0.5)
    
    **Stage Transitions:**
    - cold_start (0-9) → early (10-49) → mature (50-199) → expert (200+)
    """
)
async def track_interaction(
    user_id: int,
    interaction_data: PaperInteractionRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),  # AUTHENTICATION REQUIRED
    db: DatabaseConnection = Depends(get_db),
    interaction_repo: InteractionRepository = Depends(get_interaction_repo),
    user_repo: UserRepository = Depends(get_user_repo)
):
    """
    Track user-paper interaction with comprehensive table updates.
    
    This is the COMPLETE interaction handler that ensures all tables
    stay synchronized with user behavior.
    """
    # Verify user is tracking their own interaction
    if current_user['user_id'] != user_id:
        logger.warning(
            "Unauthorized interaction tracking attempt",
            requesting_user=current_user['user_id'],
            target_user=user_id
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only track your own interactions"
        )    

    logger.info(
        "Interaction tracking request",
        user_id=user_id,
        paper_id=interaction_data.paper_id,
        interaction_type=interaction_data.interaction_type
    )
    
    try:
        # =====================================================================
        # STEP 1: Create interaction record
        # =====================================================================
        interaction = await interaction_repo.create_interaction(
            user_id=user_id,
            paper_id=interaction_data.paper_id,
            interaction_type=interaction_data.interaction_type,
            duration_seconds=interaction_data.duration_seconds,
            source=interaction_data.context.source if interaction_data.context else None,
            position=interaction_data.context.position if interaction_data.context else None,
            session_id=interaction_data.context.session_id if interaction_data.context else None,
            score_breakdown=interaction_data.context.score_breakdown if interaction_data.context else None
        )
        
        logger.info(
            "Interaction created",
            user_id=user_id,
            interaction_id=interaction['interaction_id'],
            strength=interaction['interaction_strength']
        )
        
        # =====================================================================
        # STEP 2: Update user_recommendation_state
        # (includes automatic stage transitions)
        # =====================================================================
        state = await user_repo.get_recommendation_state(user_id)
        
        if state:
            new_count = state.get('interaction_count', 0) + 1
            
            # Determine new stage based on interaction count
            if new_count >= 200:
                new_stage = 'expert'
            elif new_count >= 50:
                new_stage = 'mature'
            elif new_count >= 10:
                new_stage = 'early'
            else:
                new_stage = 'cold_start'
            
            old_stage = state.get('recommendation_stage', 'cold_start')
            
            # Update state with new count and stage
            await user_repo.update_recommendation_state(
                user_id,
                {
                    'interaction_count': new_count,
                    'recommendation_stage': new_stage
                }
            )
            
            # Log stage transition if it occurred
            if new_stage != old_stage:
                logger.info(
                    "Stage transition occurred",
                    user_id=user_id,
                    old_stage=old_stage,
                    new_stage=new_stage,
                    interaction_count=new_count
                )
        
        # =====================================================================
        # STEP 3: Update specialized tables based on interaction type
        # =====================================================================
        
        # 3A: Save to user_saved_papers - USE REPOSITORY
        if interaction_data.interaction_type == 'save':
            await user_repo.save_paper(
                user_id=user_id,
                paper_id=interaction_data.paper_id
            )
            logger.info(
                "Paper saved",
                user_id=user_id,
                paper_id=interaction_data.paper_id
            )

        # 3B: Add to user_liked_papers - USE REPOSITORY
        if interaction_data.interaction_type == 'like':
            await user_repo.like_paper(
                user_id=user_id,
                paper_id=interaction_data.paper_id
            )
            logger.info(
                "Paper liked",
                user_id=user_id,
                paper_id=interaction_data.paper_id
            )

        # 3C: Add negative filters - USE REPOSITORY (already correct!)
        if interaction_data.interaction_type in ['dismiss', 'not_interested']:
            filter_type = 'not_interested' if interaction_data.interaction_type == 'not_interested' else 'dismissed'
            
            await interaction_repo.add_paper_filter(
                user_id=user_id,
                paper_id=interaction_data.paper_id,
                filter_type=filter_type,
                reason=f"User {interaction_data.interaction_type}"
            )
            logger.info(
                "Paper filtered",
                user_id=user_id,
                paper_id=interaction_data.paper_id,
                filter_type=filter_type
            )

        
        # =====================================================================
        # STEP 4: Check if embedding regeneration needed
        # =====================================================================
        should_update = await interaction_repo.check_embedding_update_trigger(user_id)
        
        if should_update:
            logger.info(
                "Embedding regeneration triggered",
                user_id=user_id,
                reason="5+ meaningful interactions since last update"
            )
            
            # Trigger background embedding regeneration
            background_tasks.add_task(
                regenerate_embeddings_background,
                user_id,
                db
            )
        
        # =====================================================================
        # STEP 5: Build response
        # =====================================================================
        response = {
            "interaction_id": interaction['interaction_id'],
            "user_id": user_id,
            "paper_id": interaction_data.paper_id,
            "interaction_type": interaction_data.interaction_type,
            "strength": float(interaction['interaction_strength']),
            "message": "Interaction tracked successfully"
        }
        
        # Add stage transition info if it occurred
        if 'new_stage' in locals() and new_stage != old_stage:
            response['stage_transition'] = {
                'old_stage': old_stage,
                'new_stage': new_stage,
                'interaction_count': new_count
            }
        
        logger.info(
            "Interaction tracking complete",
            user_id=user_id,
            paper_id=interaction_data.paper_id,
            updates_made={
                'interaction_created': True,
                'state_updated': True,
                'paper_saved': interaction_data.interaction_type == 'save',
                'paper_liked': interaction_data.interaction_type == 'like',
                'paper_filtered': interaction_data.interaction_type in ['dismiss', 'not_interested'],
                'embedding_queued': should_update
            }
        )
        
        return response
        
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
    description="Retrieve user's past interactions with papers. **Requires authentication.**"
)
async def get_interaction_history(
    user_id: int,
    limit: Optional[int] = 50,
    min_strength: Optional[float] = None,
    current_user: dict = Depends(get_current_user),  # AUTHENTICATION REQUIRED
    interaction_repo: InteractionRepository = Depends(get_interaction_repo)
):
    """
    Get user's interaction history.
    User can only view their own history.
    """
    
    # Verify user is accessing their own history
    if current_user['user_id'] != user_id:
        logger.warning(
            "Unauthorized history access attempt",
            requesting_user=current_user['user_id'],
            target_user=user_id
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only view your own interaction history"
        )
    
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
    description="Get aggregated statistics about user interactions. **Requires authentication.**"
)
async def get_interaction_statistics(
    user_id: int,
    days: int = 30,
    current_user: dict = Depends(get_current_user),  # AUTHENTICATION REQUIRED
    interaction_repo: InteractionRepository = Depends(get_interaction_repo)
):
    """
    Get user interaction statistics.
    User can only view their own statistics.
    """
    
    # Verify user is accessing their own statistics
    if current_user['user_id'] != user_id:
        logger.warning(
            "Unauthorized statistics access attempt",
            requesting_user=current_user['user_id'],
            target_user=user_id
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only view your own interaction statistics"
        )
    
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
    "/saved",
    summary="Get saved papers",
    description="Retrieve papers user has saved. **Requires authentication.**"
)
async def get_saved_papers(
    user_id: int,
    current_user: dict = Depends(get_current_user),  # AUTHENTICATION REQUIRED
    interaction_repo: InteractionRepository = Depends(get_interaction_repo)
):
    """
    Get user's saved papers.
    User can only view their own saved papers.
    """
    
    # Verify user is accessing their own saved papers
    if current_user['user_id'] != user_id:
        logger.warning(
            "Unauthorized saved papers access attempt",
            requesting_user=current_user['user_id'],
            target_user=user_id
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only view your own saved papers"
        )
    
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
    description="Remove a paper from user's filter list. **Requires authentication.**"
)
async def remove_paper_filter(
    user_id: int,
    paper_id: str,
    current_user: dict = Depends(get_current_user),  # AUTHENTICATION REQUIRED
    db: DatabaseConnection = Depends(get_db)
):
    """
    Remove paper filter.
    User can only manage their own filters.
    """
    
    # Verify user is managing their own filters
    if current_user['user_id'] != user_id:
        logger.warning(
            "Unauthorized filter removal attempt",
            requesting_user=current_user['user_id'],
            target_user=user_id
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only manage your own paper filters"
        )
    
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