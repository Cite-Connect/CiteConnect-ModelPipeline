"""
Pydantic models for user-related data structures.
Provides validation and serialization for user data.
"""
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field, validator


class UserBase(BaseModel):
    """Base user model."""
    email: EmailStr
    name: Optional[str] = None


class UserCreate(UserBase):
    """User creation model."""
    password: str = Field(..., min_length=8)


class UserResponse(UserBase):
    """User response model."""
    user_id: int
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class UserProfile(BaseModel):
    """Extended user profile model."""
    user_id: int
    research_stage: str
    primary_domain: str
    interests: List[str]
    sub_domains: Optional[List[str]] = None
    research_goals: Optional[List[str]] = None
    reading_level: Optional[str] = "intermediate"
    preferred_venues: Optional[List[str]] = None
    years_experience: Optional[int] = None
    prefers_recent_papers: bool = True
    prefers_high_impact: bool = True
    profile_completeness: float
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class UserState(BaseModel):
    """User recommendation state model."""
    user_id: int
    recommendation_stage: str
    interaction_count: int = 0
    preferred_model: Optional[str] = None
    last_embedding_update_minilm: Optional[datetime] = None
    last_embedding_update_specter: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True