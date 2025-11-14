# app/api/v1/users.py

"""
User API Endpoints

This module provides API endpoints for user profile management:
- GET /users/me - Get current user profile
- PUT /users/me - Update user profile

All endpoints require authentication.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_current_user_id, get_current_user
from app.schemas.user import UserResponse, UserUpdateRequest, UserUpdateResponse
from app.services.user_service import get_user_profile, update_user_profile
from app.core.exceptions import ResourceNotFoundError, ValidationError, DatabaseError

# Initialize logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


@router.get("/me", response_model=UserResponse)
async def get_my_profile(current_user: dict = Depends(get_current_user)):
    """
    Get current user's profile.
    
    Returns complete user profile including domain and interests.
    
    Returns:
        User profile data
    
    Raises:
        401: Not authenticated
        404: User not found
        500: Server error
    """
    logger.info(
        f"Get profile request",
        extra={"user_id": current_user['user_id']}
    )
    
    try:
        # Get full profile with interests
        profile = await get_user_profile(current_user['user_id'])
        
        logger.debug(f"Profile retrieved for user_id={current_user['user_id']}")
        
        return profile
        
    except ResourceNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message
        )
        
    except Exception as e:
        logger.error(f"Get profile error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get profile"
        )


@router.put("/me", response_model=UserUpdateResponse)
async def update_my_profile(
    request: UserUpdateRequest,
    user_id: int = Depends(get_current_user_id)
):
    """
    Update current user's profile.
    
    Updates user profile fields. If interests are updated,
    clusters will be regenerated.
    
    Request Body:
        - name: Updated full name (optional)
        - interests: Updated research interests (optional)
        - google_scholar_url: Updated Google Scholar URL (optional)
    
    Returns:
        Update confirmation with regeneration status
    
    Raises:
        401: Not authenticated
        400: Validation error
        500: Server error
    """
    logger.info(
        f"Update profile request",
        extra={"user_id": user_id}
    )
    
    try:
        result = await update_user_profile(
            user_id=user_id,
            name=request.name,
            interests=request.interests,
            google_scholar_url=request.google_scholar_url
        )
        
        logger.info(
            f"Profile updated",
            extra={
                "user_id": user_id,
                "regenerate_clusters": result['regenerate_clusters']
            }
        )
        
        return result
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message
        )
        
    except Exception as e:
        logger.error(f"Update profile error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile"
        )


# Initialize module logger
logger.info("User API endpoints loaded successfully")
