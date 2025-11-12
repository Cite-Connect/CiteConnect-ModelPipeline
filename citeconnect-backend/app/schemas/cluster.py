# app/schemas/cluster.py

"""
Cluster API Schemas Module

This module defines Pydantic schemas for cluster-related
API requests and responses.

Schemas:
- ClusterResponse: Cluster details response
- ClusterPaperResponse: Paper within cluster
"""

import logging
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

# Initialize logger for this module
logger = logging.getLogger(__name__)


class ClusterPaperResponse(BaseModel):
    """
    Paper within a cluster response.
    
    Attributes:
        paper_id: Paper identifier
        title: Paper title
        authors: List of authors
        year: Publication year
        citation_count: Citation count
        similarity_to_reference: Similarity to reference paper
        position_x: X coordinate for graph
        position_y: Y coordinate for graph
        is_reference: Whether this is reference paper
    """
    
    paper_id: str = Field(..., description="Paper ID")
    title: str = Field(..., description="Title")
    authors: List[str] = Field(default_factory=list, description="Authors")
    year: int = Field(..., description="Year")
    citation_count: int = Field(default=0, description="Citations")
    similarity_to_reference: float = Field(..., description="Similarity to reference")
    position_x: float = Field(..., description="X coordinate")
    position_y: float = Field(..., description="Y coordinate")
    is_reference: bool = Field(default=False, description="Is reference paper")
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "paper_id": "arxiv:2401.12345",
                "title": "AlphaFold 2",
                "authors": ["Jumper, J."],
                "year": 2021,
                "citation_count": 9432,
                "similarity_to_reference": 1.0,
                "position_x": 250.0,
                "position_y": 200.0,
                "is_reference": True
            }
        }


class ClusterResponse(BaseModel):
    """
    Cluster details response.
    
    Response for GET /clusters/{cluster_id} endpoint.
    
    Attributes:
        cluster_id: Cluster identifier
        name: Cluster theme name
        theme: Theme description
        domain: Research domain
        paper_count: Number of papers
        reference_paper: Reference paper
        papers: List of papers in cluster
        created_at: Creation timestamp
    """
    
    cluster_id: int = Field(..., description="Cluster ID")
    name: str = Field(..., description="Cluster name")
    theme: str = Field(..., description="Theme description")
    domain: str = Field(..., description="Domain")
    paper_count: int = Field(..., description="Paper count")
    reference_paper: ClusterPaperResponse = Field(..., description="Reference paper")
    papers: List[ClusterPaperResponse] = Field(default_factory=list, description="Papers")
    created_at: datetime = Field(..., description="Creation timestamp")
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "cluster_id": 1,
                "name": "AI-Driven Protein Structure Prediction",
                "theme": "Machine learning approaches for predicting protein structures",
                "domain": "healthcare",
                "paper_count": 12,
                "reference_paper": {},
                "papers": [],
                "created_at": "2025-11-08T10:00:00Z"
            }
        }


# Initialize module logger
logger.info("Cluster schemas module loaded successfully")
