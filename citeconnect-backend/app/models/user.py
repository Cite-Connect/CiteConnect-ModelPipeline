# app/models/user.py

"""
User Data Models Module

This module defines internal Pydantic models for user-related data.
These models represent the structure of user data as stored in the database
and used throughout the application.

Models:
- User: Complete user profile information
- UserDomain: User's selected research domain
- UserInterest: User's research interests/keywords
- UserProfileEmbedding: User's profile embedding vector
"""

import logging
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, EmailStr, validator

# Initialize logger for this module
logger = logging.getLogger(__name__)


class UserDomain(BaseModel):
    """
    User's selected research domain.
    
    Attributes:
        user_id: User's unique identifier
        domain: Selected domain (healthcare, fintech, quantum_computing)
        selected_at: When domain was selected
    """
    
    user_id: int = Field(
        ...,
        description="User ID"
    )
    
    domain: str = Field(
        ...,
        description="Research domain"
    )
    
    selected_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Domain selection timestamp"
    )
    
    @validator('domain')
    def validate_domain(cls, v: str) -> str:
        """Validate domain is one of allowed values."""
        allowed_domains = ['healthcare', 'fintech', 'quantum_computing']
        if v not in allowed_domains:
            logger.error(f"Invalid domain: {v}")
            raise ValueError(f"Domain must be one of {allowed_domains}")
        return v
    
    class Config:
        """Pydantic configuration."""
        from_attributes = True
        json_schema_extra = {
            "example": {
                "user_id": 123,
                "domain": "healthcare",
                "selected_at": "2025-11-08T10:00:00"
            }
        }


class UserInterest(BaseModel):
    """
    User's research interest/keyword.
    
    Attributes:
        interest_id: Unique interest identifier
        user_id: User's unique identifier
        interest_keyword: Research interest keyword
        source: Source of interest (manual, google_scholar, inferred)
        weight: Interest weight/importance (0.0-1.0)
        created_at: When interest was added
    """
    
    interest_id: Optional[int] = Field(
        None,
        description="Interest ID (auto-generated)"
    )
    
    user_id: int = Field(
        ...,
        description="User ID"
    )
    
    interest_keyword: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Research interest keyword"
    )
    
    source: str = Field(
        ...,
        description="Source of interest"
    )
    
    weight: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Interest weight (0.0-1.0)"
    )
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp"
    )
    
    @validator('source')
    def validate_source(cls, v: str) -> str:
        """Validate source is one of allowed values."""
        allowed_sources = ['manual', 'google_scholar', 'inferred']
        if v not in allowed_sources:
            logger.error(f"Invalid source: {v}")
            raise ValueError(f"Source must be one of {allowed_sources}")
        return v
    
    class Config:
        """Pydantic configuration."""
        from_attributes = True
        json_schema_extra = {
            "example": {
                "interest_id": 1,
                "user_id": 123,
                "interest_keyword": "machine learning",
                "source": "manual",
                "weight": 1.0,
                "created_at": "2025-11-08T10:00:00"
            }
        }


class UserProfileEmbedding(BaseModel):
    """
    User's profile embedding vector.
    
    Generated from user's interactions and interests using SPECTER model.
    
    Attributes:
        user_id: User's unique identifier
        embedding_vector: 768-dimensional embedding vector
        last_updated: When embedding was last updated
        based_on_papers: List of paper IDs used to build embedding
        interaction_count: Number of interactions used
    """
    
    user_id: int = Field(
        ...,
        description="User ID"
    )
    
    embedding_vector: List[float] = Field(
        ...,
        description="768-dimensional embedding vector"
    )
    
    last_updated: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last update timestamp"
    )
    
    based_on_papers: List[str] = Field(
        default_factory=list,
        description="Paper IDs used to build embedding"
    )
    
    interaction_count: int = Field(
        default=0,
        ge=0,
        description="Number of interactions"
    )
    
    @validator('embedding_vector')
    def validate_embedding_dimension(cls, v: List[float]) -> List[float]:
        """Validate embedding has correct dimension."""
        if len(v) != 768:
            logger.error(f"Invalid embedding dimension: {len(v)}")
            raise ValueError("Embedding must be 768-dimensional")
        return v
    
    class Config:
        """Pydantic configuration."""
        from_attributes = True
        json_schema_extra = {
            "example": {
                "user_id": 123,
                "embedding_vector": [0.1] * 768,
                "last_updated": "2025-11-08T10:00:00",
                "based_on_papers": ["arxiv:2401.001", "arxiv:2401.002"],
                "interaction_count": 25
            }
        }


class User(BaseModel):
    """
    Complete user profile.
    
    Attributes:
        user_id: User's unique identifier
        email: User's email address
        password_hash: Hashed password
        name: User's full name
        created_at: Account creation timestamp
        updated_at: Last profile update timestamp
        is_active: Whether account is active
        google_scholar_url: Optional Google Scholar profile URL
        semantic_scholar_author_id: Optional Semantic Scholar author ID
        domain: User's research domain
        interests: List of user's research interests
    """
    
    user_id: Optional[int] = Field(
        None,
        description="User ID (auto-generated)"
    )
    
    email: EmailStr = Field(
        ...,
        description="User email address"
    )
    
    password_hash: str = Field(
        ...,
        description="Hashed password"
    )
    
    name: str = Field(
        ...,
        min_length=2,
        max_length=255,
        description="User's full name"
    )
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Account creation timestamp"
    )
    
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last update timestamp"
    )
    
    is_active: bool = Field(
        default=True,
        description="Account active status"
    )
    
    google_scholar_url: Optional[str] = Field(
        None,
        max_length=500,
        description="Google Scholar profile URL"
    )
    
    semantic_scholar_author_id: Optional[str] = Field(
        None,
        max_length=100,
        description="Semantic Scholar author ID"
    )
    
    # Related data (optional, loaded separately)
    domain: Optional[str] = Field(
        None,
        description="User's research domain"
    )
    
    interests: List[UserInterest] = Field(
        default_factory=list,
        description="User's research interests"
    )
    
    class Config:
        """Pydantic configuration."""
        from_attributes = True
        json_schema_extra = {
            "example": {
                "user_id": 123,
                "email": "user@example.com",
                "password_hash": "$2b$12$...",
                "name": "John Doe",
                "created_at": "2025-11-01T10:00:00",
                "updated_at": "2025-11-08T10:00:00",
                "is_active": True,
                "google_scholar_url": "https://scholar.google.com/citations?user=ABC123",
                "semantic_scholar_author_id": "123456",
                "domain": "healthcare",
                "interests": []
            }
        }


class UserSavedPaper(BaseModel):
    """
    User's saved paper.
    
    Attributes:
        user_id: User's unique identifier
        paper_id: Paper identifier
        saved_at: When paper was saved
        notes: Optional user notes about the paper
    """
    
    user_id: int = Field(
        ...,
        description="User ID"
    )
    
    paper_id: str = Field(
        ...,
        description="Paper ID"
    )
    
    saved_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Save timestamp"
    )
    
    notes: Optional[str] = Field(
        None,
        description="User notes about paper"
    )
    
    class Config:
        """Pydantic configuration."""
        from_attributes = True
        json_schema_extra = {
            "example": {
                "user_id": 123,
                "paper_id": "arxiv:2401.12345",
                "saved_at": "2025-11-08T10:00:00",
                "notes": "Important for chapter 3"
            }
        }


class UserLikedPaper(BaseModel):
    """
    User's liked paper.
    
    Attributes:
        user_id: User's unique identifier
        paper_id: Paper identifier
        liked_at: When paper was liked
    """
    
    user_id: int = Field(
        ...,
        description="User ID"
    )
    
    paper_id: str = Field(
        ...,
        description="Paper ID"
    )
    
    liked_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Like timestamp"
    )
    
    class Config:
        """Pydantic configuration."""
        from_attributes = True
        json_schema_extra = {
            "example": {
                "user_id": 123,
                "paper_id": "arxiv:2401.12345",
                "liked_at": "2025-11-08T10:00:00"
            }
        }


# Initialize module logger
logger.info("User models module loaded successfully")