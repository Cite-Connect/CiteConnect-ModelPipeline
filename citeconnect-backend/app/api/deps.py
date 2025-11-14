# app/api/deps.py

"""
API Dependencies Module

This module provides dependency injection functions for FastAPI endpoints.
Used for authentication, database connections, and other shared resources.

Dependencies:
- get_current_user: Extract and verify user from JWT token
- get_db: Get database connection
"""

import logging
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.security import decode_token, extract_user_id_from_token
from app.core.exceptions import AuthenticationError
from app.db.postgres import get_db_connection

# Initialize logger
logger = logging.getLogger(__name__)

# HTTP Bearer token scheme
security = HTTPBearer()


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> int:
    """
    Extract and verify user ID from JWT token.
    
    This dependency is used in protected endpoints to authenticate users.
    
    Args:
        credentials: HTTP Bearer token from Authorization header
    
    Returns:
        User ID from token
    
    Raises:
        HTTPException: If token is invalid or expired (401)
    
    Example:
        @app.get("/protected")
        async def protected_route(user_id: int = Depends(get_current_user_id)):
            return {"user_id": user_id}
    """
    logger.info("Authenticating user from JWT token")
    
    try:
        # Extract token from credentials
        token = credentials.credentials
        
        logger.debug("Decoding JWT token")
        
        # Extract user ID from token
        user_id = extract_user_id_from_token(token)
        
        logger.info(
            "User authenticated successfully",
            extra={"user_id": user_id}
        )
        
        return user_id
        
    except AuthenticationError as e:
        logger.warning(f"Authentication failed: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=e.message,
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    except Exception as e:
        logger.error(f"Unexpected authentication error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    user_id: int = Depends(get_current_user_id)
) -> dict:
    """
    Get full user profile from database.
    
    This dependency fetches the complete user profile after authentication.
    
    Args:
        user_id: User ID from JWT token
    
    Returns:
        User profile dictionary
    
    Raises:
        HTTPException: If user not found (404)
    
    Example:
        @app.get("/me")
        async def get_me(user: dict = Depends(get_current_user)):
            return user
    """
    logger.info(f"Fetching user profile for user_id={user_id}")
    
    from app.db.postgres import execute_query
    
    try:
        # Get user from database
        user = await execute_query(
            """
            SELECT user_id, email, name, created_at, updated_at, is_active,
                   google_scholar_url, semantic_scholar_author_id
            FROM users
            WHERE user_id = $1 AND is_active = TRUE
            """,
            user_id,
            fetch_one=True
        )
        
        if not user:
            logger.warning(f"User not found or inactive: user_id={user_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        logger.debug(f"User profile fetched for {user['email']}")
        
        return dict(user)
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Error fetching user profile: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch user profile"
        )


async def get_optional_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> Optional[int]:
    """
    Extract user ID from token if provided (optional authentication).
    
    Used for endpoints that work for both authenticated and anonymous users.
    
    Args:
        credentials: Optional HTTP Bearer token
    
    Returns:
        User ID if authenticated, None otherwise
    
    Example:
        @app.get("/papers")
        async def list_papers(user_id: Optional[int] = Depends(get_optional_user_id)):
            # Works for both logged-in and anonymous users
            pass
    """
    if not credentials:
        logger.debug("No authentication token provided (optional auth)")
        return None
    
    try:
        token = credentials.credentials
        user_id = extract_user_id_from_token(token)
        logger.debug(f"Optional auth: user_id={user_id}")
        return user_id
        
    except Exception as e:
        logger.debug(f"Optional auth failed: {str(e)}")
        return None


# Initialize module logger
logger.info("API dependencies module loaded successfully")
