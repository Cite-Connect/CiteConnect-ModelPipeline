# app/schemas/user.py

"""
User API Schemas Module

This module defines Pydantic schemas for user-related
API requests and responses.

Schemas:
- UserResponse: User profile response
- UserUpdateRequest: Update user profile request
- UserInterestResponse: User interest data
- HomeResponse: Home page with clusters
- DashboardResponse: User dashboard analytics
"""

import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, EmailStr, validator

# Initialize logger for this module
logger = logging.getLogger(__name__)


class UserInterestResponse(BaseModel):
    """
    User interest response schema.
    
    Attributes:
        keyword: Interest keyword
        source: Source of interest (manual, google_scholar, inferred)
        weight: Interest weight (0.0-1.0)
    """
    
    keyword: str = Field(
        ...,
        description="Interest keyword",
        example="machine learning"
    )
    
    source: str = Field(
        ...,
        description="Interest source",
        example="manual"
    )
    
    weight: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Interest weight",
        example=1.0
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "keyword": "NLP",
                "source": "manual",
                "weight": 1.0
            }
        }


class UserResponse(BaseModel):
    """
    User profile response schema.
    
    Response for GET /users/me endpoint.
    
    Attributes:
        user_id: User's unique identifier
        email: User's email address
        name: User's full name
        domain: Research domain
        interests: List of research interests
        google_scholar_url: Google Scholar profile URL
        created_at: Account creation timestamp
    """
    
    user_id: int = Field(
        ...,
        description="User ID",
        example=12345
    )
    
    email: str = Field(
        ...,
        description="Email address",
        example="user@example.com"
    )
    
    name: str = Field(
        ...,
        description="Full name",
        example="Sarah Chen"
    )
    
    domain: str = Field(
        ...,
        description="Research domain",
        example="healthcare"
    )
    
    interests: List[UserInterestResponse] = Field(
        default_factory=list,
        description="Research interests"
    )
    
    google_scholar_url: Optional[str] = Field(
        None,
        description="Google Scholar URL",
        example="https://scholar.google.com/citations?user=ABC123"
    )
    
    created_at: datetime = Field(
        ...,
        description="Account creation timestamp",
        example="2025-11-01T10:00:00Z"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "user_id": 12345,
                "email": "user@example.com",
                "name": "Sarah Chen",
                "domain": "healthcare",
                "interests": [
                    {
                        "keyword": "NLP",
                        "source": "manual",
                        "weight": 1.0
                    },
                    {
                        "keyword": "protein folding",
                        "source": "google_scholar",
                        "weight": 0.8
                    }
                ],
                "google_scholar_url": "https://scholar.google.com/citations?user=ABC123",
                "created_at": "2025-11-01T10:00:00Z"
            }
        }


class UserUpdateRequest(BaseModel):
    """
    User profile update request schema.
    
    Request body for PUT /users/me endpoint.
    All fields are optional (partial update).
    
    Attributes:
        name: Updated full name
        interests: Updated research interests
        google_scholar_url: Updated Google Scholar URL
    """
    
    name: Optional[str] = Field(
        None,
        min_length=2,
        max_length=255,
        description="Updated full name",
        example="Sarah Chen, PhD"
    )
    
    interests: Optional[List[str]] = Field(
        None,
        min_items=1,
        max_items=10,
        description="Updated research interests",
        example=["NLP", "antibody design"]
    )
    
    google_scholar_url: Optional[str] = Field(
        None,
        max_length=500,
        description="Updated Google Scholar URL",
        example="https://scholar.google.com/citations?user=ABC123"
    )
    
    @validator('interests')
    def validate_interests(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate interests if provided."""
        if v is None:
            return v
        
        if len(v) > 10:
            raise ValueError("Maximum 10 interests allowed")
        
        # Clean interests
        cleaned = [interest.strip() for interest in v if interest.strip()]
        
        if not cleaned:
            raise ValueError("At least one valid interest is required")
        
        return cleaned
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "name": "Sarah Chen, PhD",
                "interests": ["NLP", "antibody design"],
                "google_scholar_url": "https://scholar.google.com/citations?user=ABC123"
            }
        }


class UserUpdateResponse(BaseModel):
    """
    User profile update response schema.
    
    Response for PUT /users/me endpoint.
    
    Attributes:
        user_id: User's unique identifier
        message: Success message
        regenerate_clusters: Whether clusters will be regenerated
    """
    
    user_id: int = Field(
        ...,
        description="User ID",
        example=12345
    )
    
    message: str = Field(
        ...,
        description="Success message",
        example="Profile updated successfully"
    )
    
    regenerate_clusters: bool = Field(
        default=False,
        description="Whether clusters will be regenerated",
        example=True
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "user_id": 12345,
                "message": "Profile updated successfully",
                "regenerate_clusters": True
            }
        }


class ClusterSummaryResponse(BaseModel):
    """
    Summary of a cluster for home page.
    
    Attributes:
        cluster_id: Cluster identifier
        name: Cluster theme name
        theme: Theme description
        paper_count: Number of papers
        average_relevance: Average relevance score
        reference_paper: Reference paper summary
        papers: List of papers in cluster (for network view)
    """
    
    cluster_id: int = Field(
        ...,
        description="Cluster ID",
        example=1
    )
    
    name: str = Field(
        ...,
        description="Cluster theme name",
        example="AI-Driven Protein Structure Prediction"
    )
    
    theme: Optional[str] = Field(
        None,
        description="Theme description",
        example="Machine learning approaches for predicting protein structures"
    )
    
    paper_count: int = Field(
        ...,
        description="Number of papers",
        example=12
    )
    
    average_relevance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Average relevance score",
        example=0.91
    )
    
    reference_paper: Dict[str, Any] = Field(
        ...,
        description="Reference paper data"
    )
    
    papers: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Papers in cluster (for network view)"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "cluster_id": 1,
                "name": "AI-Driven Protein Structure Prediction",
                "theme": "Machine learning approaches for predicting protein structures",
                "paper_count": 12,
                "average_relevance": 0.91,
                "reference_paper": {
                    "paper_id": "arxiv:2401.12345",
                    "title": "AlphaFold: Improved protein structure prediction",
                    "authors": ["Jumper, J.", "Evans, R."],
                    "year": 2021,
                    "citation_count": 9432,
                    "similarity_to_user": 0.91
                },
                "papers": []
            }
        }


class HomeResponse(BaseModel):
    """
    Home page response schema.
    
    Response for GET /users/me/home endpoint.
    
    Attributes:
        user: User profile summary
        clusters: List of 3 thematic clusters
        generated_at: When clusters were generated
        expires_at: When clusters cache expires
    """
    
    user: Dict[str, Any] = Field(
        ...,
        description="User profile summary"
    )
    
    clusters: List[ClusterSummaryResponse] = Field(
        ...,
        min_items=3,
        max_items=3,
        description="Three thematic clusters"
    )
    
    generated_at: datetime = Field(
        ...,
        description="Generation timestamp",
        example="2025-11-08T10:00:00Z"
    )
    
    expires_at: datetime = Field(
        ...,
        description="Cache expiration timestamp",
        example="2025-11-09T10:00:00Z"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "user": {
                    "name": "Sarah Chen",
                    "domain": "healthcare",
                    "interests": ["NLP", "clinical trials"]
                },
                "clusters": [],
                "generated_at": "2025-11-08T10:00:00Z",
                "expires_at": "2025-11-09T10:00:00Z"
            }
        }


class DashboardAnalytics(BaseModel):
    """
    User dashboard analytics data.
    
    Attributes:
        total_papers_viewed: Total unique papers viewed
        total_time_spent_hours: Total time spent reading
        papers_saved_count: Number of saved papers
        papers_liked_count: Number of liked papers
        most_viewed_topic: Most frequently viewed topic
        topic_distribution: Distribution of views by topic
    """
    
    total_papers_viewed: int = Field(
        default=0,
        ge=0,
        description="Total papers viewed",
        example=156
    )
    
    total_time_spent_hours: float = Field(
        default=0.0,
        ge=0.0,
        description="Total time in hours",
        example=12.5
    )
    
    papers_saved_count: int = Field(
        default=0,
        ge=0,
        description="Saved papers count",
        example=25
    )
    
    papers_liked_count: int = Field(
        default=0,
        ge=0,
        description="Liked papers count",
        example=18
    )
    
    most_viewed_topic: Optional[str] = Field(
        None,
        description="Most viewed topic",
        example="Clinical NLP"
    )
    
    topic_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Topic view distribution",
        example={
            "Clinical NLP": 45,
            "Drug Discovery": 30,
            "Genomics": 25
        }
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "total_papers_viewed": 156,
                "total_time_spent_hours": 12.5,
                "papers_saved_count": 25,
                "papers_liked_count": 18,
                "most_viewed_topic": "Clinical NLP",
                "topic_distribution": {
                    "Clinical NLP": 45,
                    "Drug Discovery": 30,
                    "Genomics": 25
                }
            }
        }

class DashboardResponse(BaseModel):
    """
    User dashboard response schema.
    
    Response for GET /users/me/dashboard endpoint.
    
    Attributes:
        profile: User profile information
        saved_papers: List of saved papers
        liked_papers: List of liked papers
        analytics: User analytics data
    """
    
    profile: Dict[str, Any] = Field(
        ...,
        description="User profile"
    )
    
    saved_papers: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Saved papers"
    )
    
    liked_papers: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Liked papers"
    )
    
    analytics: DashboardAnalytics = Field(
        ...,
        description="User analytics"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "profile": {
                    "name": "Sarah Chen",
                    "domain": "healthcare",
                    "interests": ["NLP", "clinical trials"],
                    "member_since": "2025-11-01"
                },
                "saved_papers": [],
                "liked_papers": [],
                "analytics": {
                    "total_papers_viewed": 156,
                    "total_time_spent_hours": 12.5,
                    "papers_saved_count": 25,
                    "papers_liked_count": 18,
                    "most_viewed_topic": "Clinical NLP",
                    "topic_distribution": {
                        "Clinical NLP": 45,
                        "Drug Discovery": 30
                    }
                }
            }
        }
# Initialize module logger
logger.info("User schemas module loaded successfully")