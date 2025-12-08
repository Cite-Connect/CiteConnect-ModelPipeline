"""
Pydantic models for paper-related data structures.
Provides validation and serialization for API requests/responses.
"""
from typing import Optional, List, Dict
from datetime import datetime
from pydantic import BaseModel, Field, validator


class PaperBase(BaseModel):
    """Base paper model with common fields."""
    paper_id: str = Field(..., description="Unique paper identifier (SHA-1 hash)")
    title: str = Field(..., min_length=1, max_length=500)
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = Field(None, ge=1900, le=2030)
    abstract: Optional[str] = Field(None, max_length=5000)
    
    @validator('authors')
    def validate_authors(cls, v):
        """Ensure authors list is not empty if provided."""
        if v is not None and len(v) == 0:
            raise ValueError("Authors list cannot be empty if provided")
        return v


class PaperCreate(PaperBase):
    """Model for creating a new paper."""
    domain: str = Field(..., description="Research domain")
    references: List[str] = Field(default_factory=list, description="Cited papers")
    published_date: Optional[datetime] = None
    venue: Optional[str] = None
    doi: Optional[str] = None


class PaperResponse(PaperBase):
    """Model for paper API responses."""
    domain: str
    citation_count: int = Field(default=0, ge=0)
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    tldr: Optional[str] = None
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    matching_aspects: List[str] = Field(default_factory=list)
    match_source: Optional[str] = None  # ← NEW
    relevance_explanation: Optional[str] = None  # ← NEW
    score_breakdown: Optional[Dict[str, float]] = None  # ← NEW
    
    class Config:
        from_attributes = True
        extra = "ignore"


class PaperWithEmbedding(PaperResponse):
    """Paper with embedding vector (for internal use)."""
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None


class PaperSearchRequest(BaseModel):
    """Request model for paper search."""
    query: str = Field(..., min_length=1, max_length=200)
    limit: int = Field(default=20, ge=1, le=100)
    domain_filter: Optional[str] = None
    min_year: Optional[int] = Field(None, ge=1900)
    
    
class RecommendationRequest(BaseModel):
    """Request model for paper recommendations."""
    user_id: Optional[int] = None
    count: int = Field(default=10, ge=1, le=50)
    model_preference: Optional[str] = Field(
        default="all-MiniLM-L6-v2",
        description="Embedding model to use"
    )
    strategy: Optional[str] = Field(
        default="personalized",
        description="Recommendation strategy"
    )
    search_query: Optional[str] = Field(  # ← NEW
        default=None,
        max_length=500,
        description="Optional search query to augment recommendations"
    )
    filters: Optional['RecommendationFilters'] = None
    session_id: str = Field(..., description="Session tracking ID")
    
    model_config = {"protected_namespaces": ()}
    
    @validator('model_preference')
    def validate_model(cls, v):
        """Validate model name."""
        allowed = ['all-MiniLM-L6-v2', 'specter2', 'auto','minilm','specter']
        if v not in allowed:
            raise ValueError(f"Model must be one of {allowed}")
        return v
    
    @validator('strategy')
    def validate_strategy(cls, v):
        """Validate strategy name."""
        allowed = ['personalized', 'canonical', 'trending', 'search']  # ← Added 'search'
        if v not in allowed:
            raise ValueError(f"Strategy must be one of {allowed}")
        return v
    
    @validator('search_query')  # ← NEW
    def validate_search_query(cls, v):
        """Validate and clean search query."""
        if v is not None:
            v = v.strip()
            if len(v) < 3:
                raise ValueError("Search query must be at least 3 characters")
        return v


class RecommendationFilters(BaseModel):
    """Filters for recommendations."""
    min_year: Optional[int] = Field(None, ge=1900)
    max_year: Optional[int] = Field(None, le=2030)
    domains: Optional[List[str]] = None
    exclude_paper_ids: Optional[List[str]] = None
    
    @validator('max_year')
    def validate_year_range(cls, v, values):
        """Ensure max_year >= min_year."""
        if 'min_year' in values and values['min_year'] and v:
            if v < values['min_year']:
                raise ValueError("max_year must be >= min_year")
        return v


class RecommendationMetadata(BaseModel):
    """Metadata about recommendation generation."""
    user_stage: str
    strategy_used: str
    model_used: str
    evaluation_scores: 'EvaluationScores'
    cache_hit: bool
    generation_time_ms: float
    
    model_config = {"protected_namespaces": ()}  # Disable Pydantic model_ namespace protection


class EvaluationScores(BaseModel):
    """Evaluation metrics for recommendations."""
    profile_alignment: Optional[float] = Field(None, ge=0.0, le=1.0)
    ground_truth_quality: Optional[float] = Field(None, ge=0.0, le=1.0)
    combined_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class RecommendationResponse(BaseModel):
    """Response model for recommendations."""
    recommendations: List[PaperResponse]
    metadata: RecommendationMetadata
    explanations: dict[str, str] = Field(
        default_factory=dict,
        description="Why each paper was recommended"
    )


class PaperInteractionRequest(BaseModel):
    """Request model for tracking paper interactions."""
    paper_id: str
    interaction_type: str = Field(
        ...,
        description="Type of interaction"
    )
    duration_seconds: Optional[int] = Field(None, ge=0)
    context: Optional['InteractionContext'] = None
    
    @validator('interaction_type')
    def validate_interaction_type(cls, v):
        """Validate interaction type."""
        allowed = [
            'view', 'click', 'save', 'like', 
            'download', 'cite', 'dismiss', 'not_interested'
        ]
        if v not in allowed:
            raise ValueError(f"Interaction type must be one of {allowed}")
        return v


class InteractionContext(BaseModel):
    """Context about where interaction occurred."""
    source: str = Field(..., description="Where interaction happened")
    position: Optional[int] = Field(None, ge=0, description="Position in list")
    session_id: str
    
    @validator('source')
    def validate_source(cls, v):
        """Validate source."""
        allowed = ['recommendation', 'search', 'citation_graph']
        if v not in allowed:
            raise ValueError(f"Source must be one of {allowed}")
        return v


# Update forward references
RecommendationRequest.update_forward_refs()
PaperInteractionRequest.update_forward_refs()