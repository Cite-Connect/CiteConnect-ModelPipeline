# app/models/cluster.py

"""
Cluster Data Models Module

This module defines internal Pydantic models for cluster-related data.
Clusters are thematic groups of papers displayed on the user's home page.

Models:
- Cluster: Complete cluster information
- ClusterPaper: Paper within a cluster with position data
- ClusterMetadata: Basic cluster metadata
"""

import logging
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, validator

from app.models.paper import PaperMetadata

# Initialize logger for this module
logger = logging.getLogger(__name__)


class ClusterPaper(BaseModel):
    """
    Paper within a cluster with positioning information.
    
    Contains paper metadata plus cluster-specific data like
    position coordinates for graph visualization.
    
    Attributes:
        paper_id: Unique paper identifier
        title: Paper title
        authors: List of author names
        year: Publication year
        citation_count: Number of citations
        centrality_score: How central paper is to cluster (0.0-1.0)
        is_reference_paper: Whether this is the cluster's reference paper
        similarity_to_reference: Similarity to reference paper (0.0-1.0)
        position_x: X coordinate for graph layout
        position_y: Y coordinate for graph layout
    """
    
    paper_id: str = Field(
        ...,
        description="Paper identifier"
    )
    
    title: str = Field(
        ...,
        description="Paper title"
    )
    
    authors: List[str] = Field(
        default_factory=list,
        description="Author names"
    )
    
    year: int = Field(
        ...,
        description="Publication year"
    )
    
    citation_count: int = Field(
        default=0,
        ge=0,
        description="Citation count"
    )
    
    centrality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Centrality score within cluster"
    )
    
    is_reference_paper: bool = Field(
        default=False,
        description="Is this the cluster reference paper"
    )
    
    similarity_to_reference: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Similarity to reference paper"
    )
    
    position_x: float = Field(
        default=0.0,
        description="X coordinate for graph layout"
    )
    
    position_y: float = Field(
        default=0.0,
        description="Y coordinate for graph layout"
    )
    
    class Config:
        """Pydantic configuration."""
        from_attributes = True
        json_schema_extra = {
            "example": {
                "paper_id": "arxiv:2401.12345",
                "title": "AlphaFold: Improved protein structure prediction",
                "authors": ["Jumper, J.", "Evans, R."],
                "year": 2021,
                "citation_count": 9432,
                "centrality_score": 0.95,
                "is_reference_paper": True,
                "similarity_to_reference": 1.0,
                "position_x": 250.0,
                "position_y": 200.0
            }
        }


class ClusterMetadata(BaseModel):
    """
    Basic cluster metadata.
    
    Lightweight model containing essential cluster information
    without paper details.
    
    Attributes:
        cluster_id: Unique cluster identifier
        user_id: User who owns this cluster
        cluster_name: Cluster theme name
        theme_description: Description of cluster theme
        domain: Research domain
        paper_count: Number of papers in cluster
        average_relevance: Average relevance score
        created_at: When cluster was created
        expires_at: When cluster cache expires
    """
    
    cluster_id: Optional[int] = Field(
        None,
        description="Cluster ID (auto-generated)"
    )
    
    user_id: int = Field(
        ...,
        description="User ID"
    )
    
    cluster_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Cluster theme name"
    )
    
    theme_description: Optional[str] = Field(
        None,
        description="Description of cluster theme"
    )
    
    domain: str = Field(
        ...,
        description="Research domain"
    )
    
    paper_count: int = Field(
        default=0,
        ge=0,
        description="Number of papers in cluster"
    )
    
    average_relevance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average relevance score"
    )
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp"
    )
    
    expires_at: Optional[datetime] = Field(
        None,
        description="Cache expiration timestamp"
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
                "cluster_id": 1,
                "user_id": 123,
                "cluster_name": "AI-Driven Protein Structure Prediction",
                "theme_description": "Machine learning approaches for predicting protein structures",
                "domain": "healthcare",
                "paper_count": 12,
                "average_relevance": 0.87,
                "created_at": "2025-11-08T10:00:00",
                "expires_at": "2025-11-09T10:00:00"
            }
        }


class Cluster(ClusterMetadata):
    """
    Complete cluster with papers and reference paper.
    
    Extends ClusterMetadata with full paper details.
    
    Attributes:
        All attributes from ClusterMetadata plus:
        reference_paper: The reference paper for this cluster
        papers: List of papers in the cluster
    """
    
    reference_paper: Optional[ClusterPaper] = Field(
        None,
        description="Reference paper for this cluster"
    )
    
    papers: List[ClusterPaper] = Field(
        default_factory=list,
        description="Papers in this cluster"
    )
    
    @validator('papers')
    def validate_papers(cls, v: List[ClusterPaper], values: dict) -> List[ClusterPaper]:
        """
        Validate papers list.
        
        Ensures:
        - At least one paper is marked as reference
        - Paper count matches metadata
        """
        if not v:
            logger.warning("Cluster has no papers")
            return v
        
        # Check for reference paper
        reference_papers = [p for p in v if p.is_reference_paper]
        
        if len(reference_papers) == 0:
            logger.warning("Cluster has no reference paper")
        elif len(reference_papers) > 1:
            logger.warning(f"Cluster has multiple reference papers: {len(reference_papers)}")
        
        # Validate paper count
        paper_count = values.get('paper_count', 0)
        if paper_count != len(v):
            logger.warning(
                f"Paper count mismatch: metadata says {paper_count}, "
                f"but got {len(v)} papers"
            )
        
        return v
    
    class Config:
        """Pydantic configuration."""
        from_attributes = True
        json_schema_extra = {
            "example": {
                "cluster_id": 1,
                "user_id": 123,
                "cluster_name": "AI-Driven Protein Structure Prediction",
                "theme_description": "Machine learning approaches for predicting protein structures",
                "domain": "healthcare",
                "paper_count": 6,
                "average_relevance": 0.87,
                "created_at": "2025-11-08T10:00:00",
                "expires_at": "2025-11-09T10:00:00",
                "reference_paper": {
                    "paper_id": "arxiv:2401.12345",
                    "title": "AlphaFold 2",
                    "authors": ["Jumper, J."],
                    "year": 2021,
                    "citation_count": 9432,
                    "is_reference_paper": True,
                    "similarity_to_reference": 1.0,
                    "position_x": 250.0,
                    "position_y": 200.0
                },
                "papers": []
            }
        }


class ClusterSummary(BaseModel):
    """
    Summary of cluster for list views.
    
    Lightweight model for displaying clusters in lists without
    loading all paper details.
    
    Attributes:
        cluster_id: Cluster identifier
        cluster_name: Cluster theme name
        paper_count: Number of papers
        average_relevance: Average relevance score
        reference_paper_title: Title of reference paper
    """
    
    cluster_id: int = Field(
        ...,
        description="Cluster ID"
    )
    
    cluster_name: str = Field(
        ...,
        description="Cluster theme name"
    )
    
    paper_count: int = Field(
        ...,
        description="Number of papers"
    )
    
    average_relevance: float = Field(
        ...,
        description="Average relevance score"
    )
    
    reference_paper_title: Optional[str] = Field(
        None,
        description="Reference paper title"
    )
    
    class Config:
        """Pydantic configuration."""
        from_attributes = True
        json_schema_extra = {
            "example": {
                "cluster_id": 1,
                "cluster_name": "AI-Driven Protein Structure Prediction",
                "paper_count": 6,
                "average_relevance": 0.87,
                "reference_paper_title": "AlphaFold: Improved protein structure prediction"
            }
        }


# Initialize module logger
logger.info("Cluster models module loaded successfully")