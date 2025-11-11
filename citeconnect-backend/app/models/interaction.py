# app/models/interaction.py

"""
Interaction Data Models Module

This module defines internal Pydantic models for user interaction tracking.
Interactions are used to personalize recommendations and understand user behavior.

Models:
- Interaction: User interaction with a paper
- InteractionContext: Additional context for an interaction
- InteractionSummary: Aggregated interaction statistics
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator

# Initialize logger for this module
logger = logging.getLogger(__name__)


class InteractionContext(BaseModel):
    """
    Additional context for an interaction.
    
    Flexible model to store various contextual information about
    how a user interacted with a paper.
    
    Attributes:
        source: Where interaction occurred (search_results, cluster, graph)
        query: Search query if from search
        cluster_id: Cluster ID if from cluster view
        source_paper_id: Source paper if clicked from graph
        additional_data: Any other relevant context
    """
    
    source: Optional[str] = Field(
        None,
        description="Interaction source"
    )
    
    query: Optional[str] = Field(
        None,
        description="Search query if applicable"
    )
    
    cluster_id: Optional[int] = Field(
        None,
        description="Cluster ID if from cluster view"
    )
    
    source_paper_id: Optional[str] = Field(
        None,
        description="Source paper if clicked from graph"
    )
    
    additional_data: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional context data"
    )
    
    class Config:
        """Pydantic configuration."""
        from_attributes = True
        json_schema_extra = {
            "example": {
                "source": "search_results",
                "query": "machine learning healthcare",
                "cluster_id": None,
                "source_paper_id": None,
                "additional_data": {"page": 1, "position": 3}
            }
        }


class Interaction(BaseModel):
    """
    User interaction with a paper.
    
    Tracks various types of user interactions for personalization
    and analytics.
    
    Attributes:
        interaction_id: Unique interaction identifier
        user_id: User's unique identifier
        paper_id: Paper identifier
        interaction_type: Type of interaction
        duration_seconds: Duration for read_time interactions
        context: Additional context information
        created_at: When interaction occurred
    """
    
    interaction_id: Optional[int] = Field(
        None,
        description="Interaction ID (auto-generated)"
    )
    
    user_id: int = Field(
        ...,
        description="User ID"
    )
    
    paper_id: str = Field(
        ...,
        description="Paper ID"
    )
    
    interaction_type: str = Field(
        ...,
        description="Type of interaction"
    )
    
    duration_seconds: Optional[int] = Field(
        None,
        ge=0,
        description="Duration in seconds (for read_time)"
    )
    
    context: Optional[InteractionContext] = Field(
        None,
        description="Interaction context"
    )
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Interaction timestamp"
    )
    
    @validator('interaction_type')
    def validate_interaction_type(cls, v: str) -> str:
        """Validate interaction type is one of allowed values."""
        allowed_types = [
            'view',
            'click',
            'save',
            'like',
            'read_time',
            'click_node',
            'search'
        ]
        
        if v not in allowed_types:
            logger.error(f"Invalid interaction type: {v}")
            raise ValueError(f"Interaction type must be one of {allowed_types}")
        
        return v
    
    @validator('duration_seconds')
    def validate_duration(cls, v: Optional[int], values: dict) -> Optional[int]:
        """Validate duration is provided for read_time interactions."""
        interaction_type = values.get('interaction_type')
        
        if interaction_type == 'read_time' and v is None:
            logger.warning("read_time interaction without duration")
        
        return v
    
    class Config:
        """Pydantic configuration."""
        from_attributes = True
        json_schema_extra = {
            "example": {
                "interaction_id": 1,
                "user_id": 123,
                "paper_id": "arxiv:2401.12345",
                "interaction_type": "view",
                "duration_seconds": None,
                "context": {
                    "source": "search_results",
                    "query": "protein folding"
                },
                "created_at": "2025-11-08T10:00:00"
            }
        }


class InteractionSummary(BaseModel):
    """
    Summary of user interactions.
    
    Aggregated statistics about user's interaction patterns.
    
    Attributes:
        user_id: User's unique identifier
        total_interactions: Total number of interactions
        views_count: Number of paper views
        clicks_count: Number of paper clicks
        saves_count: Number of papers saved
        likes_count: Number of papers liked
        total_read_time_seconds: Total reading time
        avg_read_time_seconds: Average reading time per paper
        most_interacted_domain: Most frequently interacted domain
        interaction_rate_per_day: Average interactions per day
    """
    
    user_id: int = Field(
        ...,
        description="User ID"
    )
    
    total_interactions: int = Field(
        default=0,
        ge=0,
        description="Total interaction count"
    )
    
    views_count: int = Field(
        default=0,
        ge=0,
        description="View count"
    )
    
    clicks_count: int = Field(
        default=0,
        ge=0,
        description="Click count"
    )
    
    saves_count: int = Field(
        default=0,
        ge=0,
        description="Save count"
    )
    
    likes_count: int = Field(
        default=0,
        ge=0,
        description="Like count"
    )
    
    total_read_time_seconds: int = Field(
        default=0,
        ge=0,
        description="Total reading time"
    )
    
    avg_read_time_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Average reading time"
    )
    
    most_interacted_domain: Optional[str] = Field(
        None,
        description="Most interacted domain"
    )
    
    interaction_rate_per_day: float = Field(
        default=0.0,
        ge=0.0,
        description="Average interactions per day"
    )
    
    class Config:
        """Pydantic configuration."""
        from_attributes = True
        json_schema_extra = {
            "example": {
                "user_id": 123,
                "total_interactions": 250,
                "views_count": 150,
                "clicks_count": 80,
                "saves_count": 25,
                "likes_count": 18,
                "total_read_time_seconds": 45000,
                "avg_read_time_seconds": 300.0,
                "most_interacted_domain": "healthcare",
                "interaction_rate_per_day": 12.5
            }
        }


class InteractionAnalytics(BaseModel):
    """
    Detailed interaction analytics for dashboard.
    
    Attributes:
        user_id: User's unique identifier
        total_papers_viewed: Total unique papers viewed
        total_time_spent_hours: Total time spent in hours
        papers_saved_count: Papers saved count
        papers_liked_count: Papers liked count
        most_viewed_topic: Most viewed research topic
        topic_distribution: Distribution of views by topic
        recent_activity: Recent interaction timestamps
    """
    
    user_id: int = Field(
        ...,
        description="User ID"
    )
    
    total_papers_viewed: int = Field(
        default=0,
        ge=0,
        description="Total unique papers viewed"
    )
    
    total_time_spent_hours: float = Field(
        default=0.0,
        ge=0.0,
        description="Total time spent in hours"
    )
    
    papers_saved_count: int = Field(
        default=0,
        ge=0,
        description="Saved papers count"
    )
    
    papers_liked_count: int = Field(
        default=0,
        ge=0,
        description="Liked papers count"
    )
    
    most_viewed_topic: Optional[str] = Field(
        None,
        description="Most viewed topic"
    )
    
    topic_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Views by topic"
    )
    
    recent_activity: list[datetime] = Field(
        default_factory=list,
        description="Recent interaction timestamps"
    )
    
    class Config:
        """Pydantic configuration."""
        from_attributes = True
        json_schema_extra = {
            "example": {
                "user_id": 123,
                "total_papers_viewed": 156,
                "total_time_spent_hours": 12.5,
                "papers_saved_count": 25,
                "papers_liked_count": 18,
                "most_viewed_topic": "Clinical NLP",
                "topic_distribution": {
                    "Clinical NLP": 45,
                    "Drug Discovery": 30,
                    "Genomics": 25
                },
                "recent_activity": [
                    "2025-11-08T10:00:00",
                    "2025-11-08T09:30:00"
                ]
            }
        }


# Initialize module logger
logger.info("Interaction models module loaded successfully")