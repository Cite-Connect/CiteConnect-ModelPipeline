"""
Authentication dependencies for protected endpoints.
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from typing import Optional

from app.config import settings
from app.utils.logger import get_logger
from app.db.connection import get_db, DatabaseConnection
from app.db.repositories.user_repo import UserRepository

logger = get_logger(__name__)

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: DatabaseConnection = Depends(get_db)
) -> dict:
    """
    Validate JWT token and return current user.
    
    Args:
        credentials: HTTP Bearer token
        db: Database connection
        
    Returns:
        User data dictionary
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        token = credentials.credentials
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        
        user_id: int = payload.get("user_id")
        email: str = payload.get("email")
        
        if user_id is None or email is None:
            logger.warning("Invalid token payload")
            raise credentials_exception
            
    except JWTError as e:
        logger.warning(f"JWT validation failed: {str(e)}")
        raise credentials_exception
    
    # Verify user exists and is active
    user_repo = UserRepository(db)
    user = await user_repo.find_by_id(user_id)
    
    if user is None:
        logger.warning(f"User not found: {user_id}")
        raise credentials_exception
    
    if not user.get('is_active', False):
        logger.warning(f"Inactive user attempted access: {user_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive"
        )
    
    return user


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    db: DatabaseConnection = Depends(get_db)
) -> Optional[dict]:
    """
    Optional authentication - returns user if token provided, None otherwise.
    Useful for endpoints that work differently for authenticated vs anonymous users.
    
    Args:
        credentials: Optional HTTP Bearer token
        db: Database connection
        
    Returns:
        User data dictionary or None
    """
    if credentials is None:
        return None
    
    try:
        return await get_current_user(credentials, db)
    except HTTPException:
        return None