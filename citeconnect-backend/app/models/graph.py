# app/models/graph.py

"""
Graph Data Models Module

This module defines internal Pydantic models for citation graph data.
These models represent nodes and edges in the citation network visualization.

Models:
- GraphNode: A node in the citation graph (paper)
- GraphEdge: An edge in the citation graph (citation relationship)
- CitationNetwork: Complete citation network structure
"""

import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator

# Initialize logger for this module
logger = logging.getLogger(__name__)


class GraphNode(BaseModel):
    """
    Node in the citation graph.
    
    Represents a paper in the graph visualization with its metadata
    and positioning information.
    
    Attributes:
        paper_id: Unique paper identifier
        title: Paper title
        authors: List of author names
        year: Publication year
        citation_count: Number of citations
        domain: Research domain
        similarity_score: Similarity to reference paper (0.0-1.0)
        relationship_type: Type of relationship to reference paper
        position_x: X coordinate for graph layout
        position_y: Y coordinate for graph layout
        is_reference: Whether this is the reference (central) paper
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
        ge=1900,
        le=2030,
        description="Publication year"
    )
    
    citation_count: int = Field(
        default=0,
        ge=0,
        description="Citation count"
    )
    
    domain: str = Field(
        ...,
        description="Research domain"
    )
    
    similarity_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Similarity to reference paper"
    )
    
    relationship_type: str = Field(
        default="semantic",
        description="Relationship type to reference"
    )
    
    position_x: float = Field(
        default=0.0,
        description="X coordinate for visualization"
    )
    
    position_y: float = Field(
        default=0.0,
        description="Y coordinate for visualization"
    )
    
    is_reference: bool = Field(
        default=False,
        description="Is this the reference paper"
    )
    
    @validator('domain')
    def validate_domain(cls, v: str) -> str:
        """Validate domain is one of allowed values."""
        allowed_domains = ['healthcare', 'fintech', 'quantum_computing']
        if v not in allowed_domains:
            logger.error(f"Invalid domain: {v}")
            raise ValueError(f"Domain must be one of {allowed_domains}")
        return v
    
    @validator('relationship_type')
    def validate_relationship_type(cls, v: str) -> str:
        """Validate relationship type is one of allowed values."""
        allowed_types = ['semantic', 'citation', 'co-citation', 'cites', 'cited_by']
        
        if v not in allowed_types:
            logger.warning(f"Unknown relationship type: {v}, using 'semantic'")
            return 'semantic'
        
        return v
    
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
                "domain": "healthcare",
                "similarity_score": 0.94,
                "relationship_type": "semantic",
                "position_x": 320.0,
                "position_y": 210.0,
                "is_reference": False
            }
        }


class GraphEdge(BaseModel):
    """
    Edge in the citation graph.
    
    Represents a relationship between two papers (citation or semantic similarity).
    
    Attributes:
        source: Source paper ID (from)
        target: Target paper ID (to)
        weight: Edge weight (similarity score or citation strength)
        edge_type: Type of edge (cites, cited_by, semantic, co-citation)
        citation_context: Optional context where citation appears
    """
    
    source: str = Field(
        ...,
        description="Source paper ID"
    )
    
    target: str = Field(
        ...,
        description="Target paper ID"
    )
    
    weight: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Edge weight (similarity or strength)"
    )
    
    edge_type: str = Field(
        ...,
        description="Type of edge"
    )
    
    citation_context: Optional[str] = Field(
        None,
        description="Citation context text"
    )
    
    @validator('edge_type')
    def validate_edge_type(cls, v: str) -> str:
        """Validate edge type is one of allowed values."""
        allowed_types = ['cites', 'cited_by', 'semantic', 'co-citation']
        
        if v not in allowed_types:
            logger.error(f"Invalid edge type: {v}")
            raise ValueError(f"Edge type must be one of {allowed_types}")
        
        return v
    
    @validator('source', 'target')
    def validate_paper_ids(cls, v: str) -> str:
        """Validate paper IDs are not empty."""
        if not v or not v.strip():
            logger.error("Empty paper ID in edge")
            raise ValueError("Paper ID cannot be empty")
        return v
    
    class Config:
        """Pydantic configuration."""
        from_attributes = True
        json_schema_extra = {
            "example": {
                "source": "arxiv:2401.12345",
                "target": "science:2021:rosettafold",
                "weight": 0.94,
                "edge_type": "semantic",
                "citation_context": None
            }
        }


class GraphMetadata(BaseModel):
    """
    Metadata about the graph structure.
    
    Contains information about the graph composition and generation.
    
    Attributes:
        total_nodes: Total number of nodes in graph
        total_edges: Total number of edges in graph
        layout_algorithm: Algorithm used for layout
        generated_at: When graph was generated
        reference_paper_id: Central reference paper ID
    """
    
    total_nodes: int = Field(
        default=0,
        ge=0,
        description="Total node count"
    )
    
    total_edges: int = Field(
        default=0,
        ge=0,
        description="Total edge count"
    )
    
    layout_algorithm: str = Field(
        default="force_directed",
        description="Layout algorithm used"
    )
    
    generated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Generation timestamp"
    )
    
    reference_paper_id: str = Field(
        ...,
        description="Reference paper ID"
    )
    
    class Config:
        """Pydantic configuration."""
        from_attributes = True
        json_schema_extra = {
            "example": {
                "total_nodes": 26,
                "total_edges": 45,
                "layout_algorithm": "force_directed",
                "generated_at": "2025-11-08T10:30:00",
                "reference_paper_id": "arxiv:2401.12345"
            }
        }


class CitationNetwork(BaseModel):
    """
    Complete citation network structure.
    
    Contains the full graph with nodes, edges, and metadata.
    
    Attributes:
        reference_paper: The central reference paper
        related_papers: List of related papers (nodes)
        edges: List of relationships (edges)
        graph_metadata: Graph statistics and info
    """
    
    reference_paper: GraphNode = Field(
        ...,
        description="Reference paper (central node)"
    )
    
    related_papers: List[GraphNode] = Field(
        default_factory=list,
        description="Related papers (other nodes)"
    )
    
    edges: List[GraphEdge] = Field(
        default_factory=list,
        description="Graph edges (relationships)"
    )
    
    graph_metadata: GraphMetadata = Field(
        ...,
        description="Graph metadata"
    )
    
    @validator('reference_paper')
    def validate_reference_paper(cls, v: GraphNode) -> GraphNode:
        """Ensure reference paper is marked as reference."""
        if not v.is_reference:
            logger.warning("Reference paper not marked as is_reference=True, fixing")
            v.is_reference = True
        return v
    
    @validator('edges')
    def validate_edges(cls, v: List[GraphEdge], values: dict) -> List[GraphEdge]:
        """
        Validate edges reference existing nodes.
        
        Checks that all edge source/target IDs correspond to actual nodes
        in the graph.
        """
        if not v:
            return v
        
        # Get all node IDs
        reference_paper = values.get('reference_paper')
        related_papers = values.get('related_papers', [])
        
        if not reference_paper:
            return v
        
        node_ids = {reference_paper.paper_id}
        node_ids.update(p.paper_id for p in related_papers)
        
        # Validate each edge
        valid_edges = []
        for edge in v:
            if edge.source not in node_ids:
                logger.warning(f"Edge source not in nodes: {edge.source}")
                continue
            
            if edge.target not in node_ids:
                logger.warning(f"Edge target not in nodes: {edge.target}")
                continue
            
            valid_edges.append(edge)
        
        if len(valid_edges) != len(v):
            logger.warning(
                f"Removed {len(v) - len(valid_edges)} invalid edges"
            )
        
        return valid_edges
    
    class Config:
        """Pydantic configuration."""
        from_attributes = True
        json_schema_extra = {
            "example": {
                "reference_paper": {
                    "paper_id": "arxiv:2401.12345",
                    "title": "AlphaFold 2",
                    "authors": ["Jumper, J."],
                    "year": 2021,
                    "citation_count": 9432,
                    "domain": "healthcare",
                    "similarity_score": 1.0,
                    "relationship_type": "reference",
                    "position_x": 250.0,
                    "position_y": 200.0,
                    "is_reference": True
                },
                "related_papers": [],
                "edges": [],
                "graph_metadata": {
                    "total_nodes": 26,
                    "total_edges": 45,
                    "layout_algorithm": "force_directed",
                    "generated_at": "2025-11-08T10:30:00",
                    "reference_paper_id": "arxiv:2401.12345"
                }
            }
        }


# Initialize module logger
logger.info("Graph models module loaded successfully")
