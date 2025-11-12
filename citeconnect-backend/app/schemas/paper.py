# app/schemas/paper.py

"""
Paper API Schemas Module

This module defines Pydantic schemas for paper-related
API requests and responses.

Schemas:
- PaperResponse: Basic paper information
- PaperDetailResponse: Detailed paper with full content
- PaperSaveRequest: Save paper request
- PaperActionResponse: Response for save/like actions
"""

import logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

# Initialize logger for this module
logger = logging.getLogger(__name__)


class PaperResponse(BaseModel):
    """
    Basic paper information response.
    
    Attributes:
        paper_id: Paper identifier
        title: Paper title
        authors: List of authors
        year: Publication year
        venue: Publication venue
        citation_count: Number of citations
        domain: Research domain
        abstract: Paper abstract (truncated for list views)
    """
    
    paper_id: str = Field(
        ...,
        description="Paper identifier",
        example="arxiv:2401.12345"
    )
    
    title: str = Field(
        ...,
        description="Paper title",
        example="AlphaFold: Improved protein structure prediction"
    )
    
    authors: List[str] = Field(
        default_factory=list,
        description="Author names",
        example=["Jumper, J.", "Evans, R.", "Pritzel, A."]
    )
    
    year: int = Field(
        ...,
        description="Publication year",
        example=2021
    )
    
    venue: Optional[str] = Field(
        None,
        description="Publication venue",
        example="Nature"
    )
    
    citation_count: int = Field(
        default=0,
        description="Citation count",
        example=9432
    )
    
    domain: str = Field(
        ...,
        description="Research domain",
        example="healthcare"
    )
    
    abstract: Optional[str] = Field(
        None,
        description="Paper abstract",
        example="Proteins are essential to life..."
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "paper_id": "arxiv:2401.12345",
                "title": "AlphaFold: Improved protein structure prediction",
                "authors": ["Jumper, J.", "Evans, R."],
                "year": 2021,
                "venue": "Nature",
                "citation_count": 9432,
                "domain": "healthcare",
                "abstract": "Proteins are essential to life..."
            }
        }


class PaperDetailResponse(BaseModel):
    """
    Detailed paper information response.
    
    Response for GET /papers/{paper_id} endpoint.
    
    Attributes:
        paper_id: Paper identifier
        title: Paper title
        authors: List of authors
        year: Publication year
        venue: Publication venue
        citation_count: Number of citations
        abstract: Full abstract
        summary: AI-generated summary
        introduction: Introduction section
        domain: Research domain
        pdf_url: URL to PDF file
        external_links: Links to external sources
    """
    
    paper_id: str = Field(..., description="Paper identifier")
    title: str = Field(..., description="Paper title")
    authors: List[str] = Field(default_factory=list, description="Authors")
    year: int = Field(..., description="Publication year")
    venue: Optional[str] = Field(None, description="Venue")
    citation_count: int = Field(default=0, description="Citation count")
    abstract: str = Field(..., description="Abstract")
    summary: Optional[str] = Field(None, description="AI summary")
    introduction: Optional[str] = Field(None, description="Introduction")
    domain: str = Field(..., description="Domain")
    pdf_url: Optional[str] = Field(None, description="PDF URL")
    external_links: Dict[str, str] = Field(
        default_factory=dict,
        description="External links"
    )
    
    class Config:
        """Pydantic configuration."""
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
                "domain": "healthcare",
                "pdf_url": "https://storage.googleapis.com/.../alphafold.pdf",
                "external_links": {
                    "arxiv": "https://arxiv.org/abs/2401.12345",
                    "semantic_scholar": "https://www.semanticscholar.org/paper/..."
                }
            }
        }


class PaperSaveRequest(BaseModel):
    """
    Save paper request schema.
    
    Request body for POST /papers/{paper_id}/save endpoint.
    
    Attributes:
        notes: Optional notes about the paper
    """
    
    notes: Optional[str] = Field(
        None,
        max_length=1000,
        description="User notes about paper",
        example="Important for my research on antibody design"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "notes": "Important for chapter 3"
            }
        }


class PaperActionResponse(BaseModel):
    """
    Response for paper actions (save, like, etc).
    
    Attributes:
        message: Success message
        action: Action performed
        timestamp: When action occurred
    """
    
    message: str = Field(
        ...,
        description="Success message",
        example="Paper saved successfully"
    )
    
    action: str = Field(
        ...,
        description="Action performed",
        example="save"
    )
    
    timestamp: str = Field(
        ...,
        description="Action timestamp",
        example="2025-11-08T10:30:00Z"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "message": "Paper saved successfully",
                "action": "save",
                "timestamp": "2025-11-08T10:30:00Z"
            }
        }


# Initialize module logger
logger.info("Paper schemas module loaded successfully")
