# app/schemas/search.py

"""
Search API Schemas Module

This module defines Pydantic schemas for search-related
API requests and responses.

Schemas:
- SearchRequest: Search query request
- SearchResponse: Search results response
- SearchFilters: Search filter options
"""

import logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator

# Initialize logger for this module
logger = logging.getLogger(__name__)


class SearchFilters(BaseModel):
    """
    Search filter options.
    
    Attributes:
        domain: Filter by domain
        year_min: Minimum publication year
        year_max: Maximum publication year
        citation_min: Minimum citation count
    """
    
    domain: Optional[str] = Field(
        None,
        description="Filter by domain",
        example="healthcare"
    )
    
    year_min: Optional[int] = Field(
        None,
        ge=1900,
        le=2030,
        description="Minimum year",
        example=2020
    )
    
    year_max: Optional[int] = Field(
        None,
        ge=1900,
        le=2030,
        description="Maximum year",
        example=2024
    )
    
    citation_min: Optional[int] = Field(
        None,
        ge=0,
        description="Minimum citations",
        example=10
    )
    
    @validator('year_max')
    def validate_year_range(cls, v: Optional[int], values: dict) -> Optional[int]:
        """Validate year_max is greater than year_min."""
        if v is not None and 'year_min' in values:
            year_min = values['year_min']
            if year_min is not None and v < year_min:
                raise ValueError("year_max must be greater than or equal to year_min")
        return v


class MatchExplanation(BaseModel):
    """
    Explanation of why a paper matches the query.
    
    Attributes:
        semantic_similarity: Semantic similarity score
        keyword_matches: List of matched keywords
        confidence: Confidence level (high, medium, low)
    """
    
    semantic_similarity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Semantic similarity score",
        example=0.89
    )
    
    keyword_matches: List[str] = Field(
        default_factory=list,
        description="Matched keywords",
        example=["antibody", "design", "machine learning"]
    )
    
    confidence: str = Field(
        ...,
        description="Confidence level",
        example="high"
    )
    
    @validator('confidence')
    def validate_confidence(cls, v: str) -> str:
        """Validate confidence level."""
        allowed = ['high', 'medium', 'low']
        if v not in allowed:
            return 'medium'
        return v


class SearchResultPaper(BaseModel):
    """
    Single search result paper.
    
    Attributes:
        paper_id: Paper identifier
        title: Paper title
        authors: List of authors
        year: Publication year
        venue: Publication venue
        citation_count: Citation count
        abstract: Paper abstract
        summary: AI-generated summary
        relevance_score: Relevance score (0.0-1.0)
        match_explanation: Why this paper matches
    """
    
    paper_id: str = Field(..., description="Paper ID")
    title: str = Field(..., description="Title")
    authors: List[str] = Field(default_factory=list, description="Authors")
    year: int = Field(..., description="Year")
    venue: Optional[str] = Field(None, description="Venue")
    citation_count: int = Field(default=0, description="Citations")
    abstract: str = Field(..., description="Abstract")
    summary: Optional[str] = Field(None, description="Summary")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance")
    match_explanation: MatchExplanation = Field(..., description="Match explanation")


class SearchResponse(BaseModel):
    """
    Search results response schema.
    
    Response for GET /search endpoint.
    
    Attributes:
        query: Original search query
        domain: Domain filter applied
        total_results: Total number of results found
        shown_results: Number of results returned
        results: List of search result papers
        filters_applied: Filters that were applied
        search_time_ms: Search execution time
    """
    
    query: str = Field(
        ...,
        description="Search query",
        example="antibody design machine learning"
    )
    
    domain: str = Field(
        ...,
        description="Domain filter",
        example="healthcare"
    )
    
    total_results: int = Field(
        ...,
        ge=0,
        description="Total results found",
        example=847
    )
    
    shown_results: int = Field(
        ...,
        ge=0,
        description="Results returned",
        example=20
    )
    
    results: List[SearchResultPaper] = Field(
        default_factory=list,
        description="Search results"
    )
    
    filters_applied: Dict[str, Any] = Field(
        default_factory=dict,
        description="Applied filters"
    )
    
    search_time_ms: int = Field(
        ...,
        ge=0,
        description="Search time in milliseconds",
        example=245
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "query": "antibody design machine learning",
                "domain": "healthcare",
                "total_results": 847,
                "shown_results": 20,
                "results": [],
                "filters_applied": {
                    "domain": "healthcare",
                    "year_min": None,
                    "year_max": None
                },
                "search_time_ms": 245
            }
        }


# Initialize module logger
logger.info("Search schemas module loaded successfully")
