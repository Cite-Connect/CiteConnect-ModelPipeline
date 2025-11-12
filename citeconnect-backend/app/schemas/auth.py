# app/schemas/auth.py

"""
Authentication API Schemas Module

This module defines Pydantic schemas for authentication-related
API requests and responses.

Schemas:
- LoginRequest: User login request
- RegisterRequest: User registration request
- TokenResponse: Authentication token response
- RefreshTokenRequest: Token refresh request
"""

import logging
from typing import Optional, List
from pydantic import BaseModel, Field, EmailStr, validator

# Initialize logger for this module
logger = logging.getLogger(__name__)


class LoginRequest(BaseModel):
    """
    Login request schema.
    
    Request body for POST /auth/login endpoint.
    
    Attributes:
        email: User's email address
        password: User's password (plain text, will be hashed)
    """
    
    email: EmailStr = Field(
        ...,
        description="User email address",
        example="user@example.com"
    )
    
    password: str = Field(
        ...,
        min_length=8,
        max_length=100,
        description="User password",
        example="SecurePassword123!"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "SecurePassword123!"
            }
        }


class RegisterRequest(BaseModel):
    """
    User registration request schema.
    
    Request body for POST /auth/register endpoint.
    
    Attributes:
        email: User's email address
        password: User's password
        name: User's full name
        domain: Research domain selection
        interests: List of research interest keywords
        google_scholar_url: Optional Google Scholar profile URL
        uploaded_paper_file: Optional base64 encoded PDF
    """
    
    email: EmailStr = Field(
        ...,
        description="User email address",
        example="user@example.com"
    )
    
    password: str = Field(
        ...,
        min_length=8,
        max_length=100,
        description="User password"
    )
    
    name: str = Field(
        ...,
        min_length=2,
        max_length=255,
        description="User's full name",
        example="Sarah Chen"
    )
    
    domain: str = Field(
        ...,
        description="Research domain",
        example="healthcare"
    )
    
    interests: List[str] = Field(
        ...,
        min_items=1,
        max_items=10,
        description="Research interest keywords",
        example=["machine learning", "clinical trials", "drug discovery"]
    )
    
    google_scholar_url: Optional[str] = Field(
        None,
        max_length=500,
        description="Google Scholar profile URL",
        example="https://scholar.google.com/citations?user=ABC123"
    )
    
    uploaded_paper_file: Optional[str] = Field(
        None,
        description="Base64 encoded PDF file"
    )
    
    @validator('domain')
    def validate_domain(cls, v: str) -> str:
        """Validate domain is one of allowed values."""
        allowed_domains = ['healthcare', 'fintech', 'quantum_computing']
        if v not in allowed_domains:
            logger.error(f"Invalid domain in registration: {v}")
            raise ValueError(f"Domain must be one of {allowed_domains}")
        return v
    
    @validator('password')
    def validate_password_strength(cls, v: str) -> str:
        """
        Validate password strength.
        
        Requirements:
        - At least 8 characters
        - At least one digit
        - At least one uppercase letter
        - At least one lowercase letter
        """
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        
        if not any(char.isdigit() for char in v):
            raise ValueError("Password must contain at least one digit")
        
        if not any(char.isupper() for char in v):
            raise ValueError("Password must contain at least one uppercase letter")
        
        if not any(char.islower() for char in v):
            raise ValueError("Password must contain at least one lowercase letter")
        
        return v
    
    @validator('interests')
    def validate_interests(cls, v: List[str]) -> List[str]:
        """Validate interests list."""
        if not v:
            raise ValueError("At least one interest is required")
        
        if len(v) > 10:
            raise ValueError("Maximum 10 interests allowed")
        
        # Clean and validate each interest
        cleaned = []
        for interest in v:
            cleaned_interest = interest.strip()
            if len(cleaned_interest) < 2:
                logger.warning(f"Skipping too-short interest: {interest}")
                continue
            if len(cleaned_interest) > 100:
                logger.warning(f"Truncating long interest: {interest}")
                cleaned_interest = cleaned_interest[:100]
            cleaned.append(cleaned_interest)
        
        if not cleaned:
            raise ValueError("At least one valid interest is required")
        
        return cleaned
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "SecurePassword123!",
                "name": "Sarah Chen",
                "domain": "healthcare",
                "interests": ["NLP", "clinical trials", "drug discovery"],
                "google_scholar_url": "https://scholar.google.com/citations?user=ABC123",
                "uploaded_paper_file": None
            }
        }


class TokenResponse(BaseModel):
    """
    Authentication token response schema.
    
    Response for successful login or registration.
    
    Attributes:
        user_id: User's unique identifier
        email: User's email address
        name: User's full name
        domain: User's research domain
        access_token: JWT access token
        refresh_token: JWT refresh token (optional)
        token_type: Token type (always "bearer")
        expires_in: Token expiration time in seconds
        starter_kit_status: Status of starter kit generation
    """
    
    user_id: int = Field(
        ...,
        description="User ID",
        example=12345
    )
    
    email: str = Field(
        ...,
        description="User email",
        example="user@example.com"
    )
    
    name: str = Field(
        ...,
        description="User name",
        example="Sarah Chen"
    )
    
    domain: str = Field(
        ...,
        description="Research domain",
        example="healthcare"
    )
    
    access_token: str = Field(
        ...,
        description="JWT access token",
        example="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
    )
    
    refresh_token: Optional[str] = Field(
        None,
        description="JWT refresh token",
        example="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
    )
    
    token_type: str = Field(
        default="bearer",
        description="Token type"
    )
    
    expires_in: int = Field(
        ...,
        description="Token expiration time in seconds",
        example=86400
    )
    
    starter_kit_status: Optional[str] = Field(
        None,
        description="Starter kit generation status",
        example="processing"
    )
    
    @validator('starter_kit_status')
    def validate_starter_kit_status(cls, v: Optional[str]) -> Optional[str]:
        """Validate starter kit status."""
        if v is not None:
            allowed_statuses = ['processing', 'ready', 'failed']
            if v not in allowed_statuses:
                logger.warning(f"Unknown starter kit status: {v}")
                return 'processing'
        return v
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "user_id": 12345,
                "email": "user@example.com",
                "name": "Sarah Chen",
                "domain": "healthcare",
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 86400,
                "starter_kit_status": "processing"
            }
        }


class RefreshTokenRequest(BaseModel):
    """
    Token refresh request schema.
    
    Request body for POST /auth/refresh endpoint.
    
    Attributes:
        refresh_token: JWT refresh token to exchange for new access token
    """
    
    refresh_token: str = Field(
        ...,
        description="JWT refresh token",
        example="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
            }
        }


class TokenRefreshResponse(BaseModel):
    """
    Token refresh response schema.
    
    Response for successful token refresh.
    
    Attributes:
        access_token: New JWT access token
        token_type: Token type (always "bearer")
        expires_in: Token expiration time in seconds
    """
    
    access_token: str = Field(
        ...,
        description="New JWT access token",
        example="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
    )
    
    token_type: str = Field(
        default="bearer",
        description="Token type"
    )
    
    expires_in: int = Field(
        ...,
        description="Token expiration time in seconds",
        example=86400
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 86400
            }
        }


# Initialize module logger
logger.info("Auth schemas module loaded successfully")
