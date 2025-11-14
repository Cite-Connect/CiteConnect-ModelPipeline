# app/api/v1/auth.py

"""
Authentication API Endpoints

This module provides API endpoints for authentication:
- POST /auth/register - User registration
- POST /auth/login - User login
- POST /auth/refresh - Token refresh

All endpoints are public (no authentication required).
"""

import logging
from fastapi import APIRouter, HTTPException, status

from app.schemas.auth import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    RefreshTokenRequest,
    TokenRefreshResponse
)
from app.services.auth_service import register_user, login_user, refresh_access_token
from app.core.exceptions import AuthenticationError, ValidationError, DatabaseError

# Initialize logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(request: RegisterRequest):
    """
    Register a new user.
    
    Creates a new user account with domain selection and research interests.
    Triggers async generation of personalized starter kit (3 paper clusters).
    
    Request Body:
        - email: User's email address
        - password: Password (min 8 chars, must have digit, uppercase, lowercase)
        - name: Full name
        - domain: Research domain (healthcare, fintech, quantum_computing)
        - interests: 1-10 research keywords
        - google_scholar_url: Optional Google Scholar profile
        - uploaded_paper_file: Optional base64 encoded PDF
    
    Returns:
        User profile with authentication tokens
    
    Raises:
        400: Validation error (invalid input)
        409: Email already registered
        500: Server error
    """
    logger.info(
        f"Registration request received",
        extra={"email": request.email, "domain": request.domain}
    )
    
    try:
        user_data = await register_user(
            email=request.email,
            password=request.password,
            name=request.name,
            domain=request.domain,
            interests=request.interests,
            google_scholar_url=request.google_scholar_url,
            uploaded_paper_file=request.uploaded_paper_file
        )
        
        logger.info(
            f"User registered successfully",
            extra={"user_id": user_data['user_id'], "email": request.email}
        )
        
        return user_data
        
    except ValidationError as e:
        logger.warning(f"Registration validation error: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message
        )
        
    except DatabaseError as e:
        if "already registered" in e.message.lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=e.message
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=e.message
            )


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """
    Login user and generate tokens.
    
    Authenticates user credentials and returns JWT tokens.
    
    Request Body:
        - email: User's email address
        - password: User's password
    
    Returns:
        User profile with authentication tokens
    
    Raises:
        401: Invalid credentials
        500: Server error
    """
    logger.info(f"Login request received for email: {request.email}")
    
    try:
        user_data = await login_user(
            email=request.email,
            password=request.password
        )
        
        logger.info(
            f"User logged in successfully",
            extra={"user_id": user_data['user_id'], "email": request.email}
        )
        
        return user_data
        
    except AuthenticationError as e:
        logger.warning(f"Login failed: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=e.message,
            headers={"WWW-Authenticate": "Bearer"}
        )
        
    except Exception as e:
        logger.error(f"Login error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/refresh", response_model=TokenRefreshResponse)
async def refresh_token(request: RefreshTokenRequest):
    """
    Refresh access token.
    
    Exchanges a valid refresh token for a new access token.
    
    Request Body:
        - refresh_token: JWT refresh token
    
    Returns:
        New access token
    
    Raises:
        401: Invalid or expired refresh token
        500: Server error
    """
    logger.info("Token refresh request received")
    
    try:
        token_data = await refresh_access_token(request.refresh_token)
        
        logger.info("Token refreshed successfully")
        
        return token_data
        
    except AuthenticationError as e:
        logger.warning(f"Token refresh failed: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=e.message,
            headers={"WWW-Authenticate": "Bearer"}
        )
        
    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )


# Initialize module logger
logger.info("Auth API endpoints loaded successfully")
