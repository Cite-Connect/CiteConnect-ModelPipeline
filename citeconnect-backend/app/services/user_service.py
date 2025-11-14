# app/services/user_service.py

"""
User Service Module

This module handles user profile management including:
- Get user profile
- Update user profile
- Get user interests
- Update user interests

Dependencies:
- PostgreSQL for user data storage
"""

import logging
from typing import Dict, Any, List, Optional

from app.core.exceptions import ResourceNotFoundError, ValidationError, DatabaseError
from app.db.postgres import execute_query

# Initialize logger
logger = logging.getLogger(__name__)


async def get_user_profile(user_id: int) -> Dict[str, Any]:
    """
    Get complete user profile with domain and interests.
    
    Args:
        user_id: User's unique identifier
    
    Returns:
        Dictionary with user profile data
    
    Raises:
        ResourceNotFoundError: If user not found
        DatabaseError: If database operation fails
    
    Example:
        >>> profile = await get_user_profile(123)
        >>> print(profile['email'])
    """
    logger.info(f"Getting user profile for user_id={user_id}")
    
    try:
        # Get user basic info
        user = await execute_query(
            """
            SELECT user_id, email, name, created_at, updated_at, is_active,
                   google_scholar_url, semantic_scholar_author_id
            FROM users
            WHERE user_id = $1
            """,
            user_id,
            fetch_one=True
        )
        
        if not user:
            logger.warning(f"User not found: user_id={user_id}")
            raise ResourceNotFoundError("User", str(user_id))
        
        # Get user domain
        domain_result = await execute_query(
            "SELECT domain, selected_at FROM user_domains WHERE user_id = $1",
            user_id,
            fetch_one=True
        )
        
        # Get user interests
        interests = await execute_query(
            """
            SELECT interest_keyword, source, weight, created_at
            FROM user_interests
            WHERE user_id = $1
            ORDER BY weight DESC, created_at ASC
            """,
            user_id,
            fetch_all=True
        )
        
        # Build response
        profile = dict(user)
        profile['domain'] = domain_result['domain'] if domain_result else None
        profile['interests'] = [
            {
                "keyword": i['interest_keyword'],
                "source": i['source'],
                "weight": float(i['weight'])
            }
            for i in interests
        ]
        
        logger.debug(
            f"User profile retrieved",
            extra={"user_id": user_id, "interests_count": len(interests)}
        )
        
        return profile
        
    except ResourceNotFoundError:
        raise
        
    except Exception as e:
        logger.error(f"Failed to get user profile: {str(e)}", exc_info=True)
        raise DatabaseError(
            message=f"Failed to get user profile: {str(e)}",
            operation="get_user_profile"
        )


async def update_user_profile(
    user_id: int,
    name: Optional[str] = None,
    interests: Optional[List[str]] = None,
    google_scholar_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Update user profile.
    
    Args:
        user_id: User's unique identifier
        name: Updated name (optional)
        interests: Updated interests list (optional)
        google_scholar_url: Updated Google Scholar URL (optional)
    
    Returns:
        Dictionary with update status
    
    Raises:
        ResourceNotFoundError: If user not found
        DatabaseError: If database operation fails
    
    Example:
        >>> result = await update_user_profile(
        ...     user_id=123,
        ...     name="Sarah Chen, PhD",
        ...     interests=["NLP", "antibody design"]
        ... )
    """
    logger.info(f"Updating user profile for user_id={user_id}")
    
    try:
        # Verify user exists
        user = await execute_query(
            "SELECT user_id FROM users WHERE user_id = $1",
            user_id,
            fetch_one=True
        )
        
        if not user:
            logger.warning(f"Update attempt for non-existent user: {user_id}")
            raise ResourceNotFoundError("User", str(user_id))
        
        regenerate_clusters = False
        
        # Update name if provided
        if name is not None:
            logger.debug(f"Updating name for user_id={user_id}")
            
            await execute_query(
                """
                UPDATE users
                SET name = $1, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = $2
                """,
                name,
                user_id
            )
        
        # Update interests if provided
        if interests is not None:
            logger.debug(f"Updating interests for user_id={user_id}")
            
            # Delete existing interests
            await execute_query(
                "DELETE FROM user_interests WHERE user_id = $1",
                user_id
            )
            
            # Insert new interests
            for interest in interests:
                await execute_query(
                    """
                    INSERT INTO user_interests (user_id, interest_keyword, source, weight)
                    VALUES ($1, $2, 'manual', 1.0)
                    """,
                    user_id,
                    interest.strip()
                )
            
            # Interests changed - need to regenerate clusters
            regenerate_clusters = True
            
            logger.info(f"Updated interests for user_id={user_id}")
        
        # Update Google Scholar URL if provided
        if google_scholar_url is not None:
            logger.debug(f"Updating Google Scholar URL for user_id={user_id}")
            
            await execute_query(
                """
                UPDATE users
                SET google_scholar_url = $1, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = $2
                """,
                google_scholar_url,
                user_id
            )
        
        # Invalidate cached clusters if needed
        if regenerate_clusters:
            logger.info("Invalidating user clusters cache")
            from app.db.redis_client import cache_delete
            await cache_delete(f"starter_kit:{user_id}")
            
            # TODO: Trigger cluster regeneration
            # from app.tasks.clustering import regenerate_user_clusters
            # regenerate_user_clusters.delay(user_id)
        
        logger.info(
            f"User profile updated successfully",
            extra={"user_id": user_id, "regenerate_clusters": regenerate_clusters}
        )
        
        return {
            "user_id": user_id,
            "message": "Profile updated successfully",
            "regenerate_clusters": regenerate_clusters
        }
        
    except ResourceNotFoundError:
        raise
        
    except Exception as e:
        logger.error(f"Failed to update user profile: {str(e)}", exc_info=True)
        raise DatabaseError(
            message=f"Failed to update profile: {str(e)}",
            operation="update_user_profile"
        )


async def get_user_interests(user_id: int) -> List[Dict[str, Any]]:
    """
    Get user's research interests.
    
    Args:
        user_id: User's unique identifier
    
    Returns:
        List of interest dictionaries
    
    Example:
        >>> interests = await get_user_interests(123)
        >>> print(interests[0]['keyword'])
    """
    logger.info(f"Getting interests for user_id={user_id}")
    
    try:
        interests = await execute_query(
            """
            SELECT interest_keyword, source, weight, created_at
            FROM user_interests
            WHERE user_id = $1
            ORDER BY weight DESC, created_at ASC
            """,
            user_id,
            fetch_all=True
        )
        
        logger.debug(f"Retrieved {len(interests)} interests for user_id={user_id}")
        
        return [
            {
                "keyword": i['interest_keyword'],
                "source": i['source'],
                "weight": float(i['weight']),
                "created_at": i['created_at'].isoformat()
            }
            for i in interests
        ]
        
    except Exception as e:
        logger.error(f"Failed to get user interests: {str(e)}", exc_info=True)
        return []


# Initialize module logger
logger.info("User service module loaded successfully")
