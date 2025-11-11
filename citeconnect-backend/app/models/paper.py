# app/models/paper.py

"""
Paper Data Models Module

This module defines internal Pydantic models for paper-related data.
These models represent the structure of academic papers as stored in
the database and used throughout the application.

Models:
- Paper: Complete paper metadata and content
- PaperMetadata: Basic paper metadata
- PaperEmbedding: Paper embedding vector
"""

import logging
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, validator

# Initialize logger for this module
logger = logging.getLogger(__name__)


class PaperMetadata(BaseModel):
    """
    Basic paper metadata.
    
    Lightweight model containing essential paper information
    without full content.
    
    Attributes:
        paper_id: Unique paper identifier
        title: Paper title
        authors: List of author names
        year: Publication year
        venue: Publication venue (journal/conference)
        citation_count: Number of citations
        domain: Research domain
    """
    
    paper_id: str = Field(
        ...,
        description="Unique paper identifier (e.g., arxiv:2401.12345)"
    )
    
    title: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Paper title"
    )
    
    authors: List[str] = Field(
        default_factory=list,
        description="List of author names"
    )
    
    year: int = Field(
        ...,
        ge=1900,
        le=2030,
        description="Publication year"
    )
    
    venue: Optional[str] = Field(
        None,
        max_length=255,
        description="Publication venue"
    )
    
    citation_count: int = Field(
        default=0,
        ge=0,
        description="Number of citations"
    )
    
    domain: str = Field(
        ...,
        description="Research domain"
    )
    
    @validator('domain')
    def validate_domain(cls, v: str) -> str:
        """Validate domain is one of allowed values."""
        allowed_domains = ['healthcare', 'fintech', 'quantum_computing']
        if v not in allowed_domains:
            logger.error(f"Invalid domain: {v}")
            raise ValueError(f"Domain must be one of {allowed_domains}")
        return v
    
    @validator('authors')
    def validate_authors(cls, v: List[str]) -> List[str]:
        """Validate authors list is not empty."""
        if not v:
            logger.warning("Paper has no authors listed")
        return v
    
    class Config:
        """Pydantic configuration."""
        from_attributes = True
        json_schema_extra = {
            "example": {
                "paper_id": "arxiv:2401.12345",
                "title": "Deep Learning for Protein Structure Prediction",
                "authors": ["Smith, J.", "Johnson, A.", "Williams, B."],
                "year": 2024,
                "venue": "Nature",
                "citation_count": 156,
                "domain": "healthcare"
            }
        }


class Paper(BaseModel):
    """
    Complete paper with full content.
    
    Attributes:
        paper_id: Unique paper identifier
        title: Paper title
        authors: List of author names
        year: Publication year
        venue: Publication venue
        citation_count: Number of citations
        abstract: Paper abstract
        summary: AI-generated summary
        introduction: Introduction section text
        gcs_pdf_path: Google Cloud Storage path to PDF
        domain: Research domain
        ingested_at: When paper was ingested
        updated_at: Last update timestamp
    """
    
    paper_id: str = Field(
        ...,
        description="Unique paper identifier"
    )
    
    title: str = Field(
        ...,
        min_length=1,
        description="Paper title"
    )
    
    authors: List[str] = Field(
        default_factory=list,
        description="List of author names"
    )
    
    year: int = Field(
        ...,
        ge=1900,
        le=2030,
        description="Publication year"
    )
    
    venue: Optional[str] = Field(
        None,
        description="Publication venue"
    )
    
    citation_count: int = Field(
        default=0,
        ge=0,
        description="Number of citations"
    )
    
    abstract: str = Field(
        ...,
        min_length=1,
        description="Paper abstract"
    )
    
    summary: Optional[str] = Field(
        None,
        description="AI-generated summary"
    )
    
    introduction: Optional[str] = Field(
        None,
        description="Introduction section"
    )
    
    gcs_pdf_path: Optional[str] = Field(
        None,
        max_length=500,
        description="Google Cloud Storage PDF path"
    )
    
    domain: str = Field(
        ...,
        description="Research domain"
    )
    
    ingested_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Ingestion timestamp"
    )
    
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last update timestamp"
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
                "paper_id": "arxiv:2401.12345",
                "title": "AlphaFold: Improved protein structure prediction",
                "authors": ["Jumper, J.", "Evans, R.", "Pritzel, A."],
                "year": 2021,
                "venue": "Nature",
                "citation_count": 9432,
                "abstract": "Proteins are essential to life...",
                "summary": "This paper presents AlphaFold 2...",
                "introduction": "Recent advances in deep learning...",
                "gcs_pdf_path": "papers/arxiv/2401.12345.pdf",
                "domain": "healthcare",
                "ingested_at": "2025-11-01T10:00:00",
                "updated_at": "2025-11-08T10:00:00"
            }
        }


class PaperEmbedding(BaseModel):
    """
    Paper embedding vector.
    
    SPECTER embedding for semantic similarity search.
    
    Attributes:
        paper_id: Unique paper identifier
        embedding_vector: 768-dimensional embedding vector
        model_name: Name of embedding model used
        created_at: When embedding was generated
    """
    
    paper_id: str = Field(
        ...,
        description="Paper identifier"
    )
    
    embedding_vector: List[float] = Field(
        ...,
        description="768-dimensional embedding vector"
    )
    
    model_name: str = Field(
        default="allenai/specter",
        description="Embedding model name"
    )
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp"
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
                "paper_id": "arxiv:2401.12345",
                "embedding_vector": [0.1] * 768,
                "model_name": "allenai/specter",
                "created_at": "2025-11-08T10:00:00"
            }
        }


class PaperWithScore(PaperMetadata):
    """
    Paper metadata with relevance/similarity score.
    
    Used for search results and recommendations.
    
    Attributes:
        All attributes from PaperMetadata plus:
        relevance_score: Relevance/similarity score (0.0-1.0)
        match_explanation: Explanation of why paper matches
    """
    
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score (0.0-1.0)"
    )
    
    match_explanation: Optional[dict] = Field(
        None,
        description="Explanation of match"
    )
    
    class Config:
        """Pydantic configuration."""
        from_attributes = True
        json_schema_extra = {
            "example": {
                "paper_id": "arxiv:2401.12345",
                "title": "Deep Learning for Healthcare",
                "authors": ["Smith, J."],
                "year": 2024,
                "venue": "Nature",
                "citation_count": 100,
                "domain": "healthcare",
                "relevance_score": 0.89,
                "match_explanation": {
                    "semantic_similarity": 0.89,
                    "keyword_matches": ["deep learning", "healthcare"],
                    "confidence": "high"
                }
            }
        }


# Initialize module logger
logger.info("Paper models module loaded successfully")