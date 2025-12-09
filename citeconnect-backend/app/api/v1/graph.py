"""
Citation Graph API Endpoints
Routes for fetching citation network data for visualization.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional
from pydantic import BaseModel, Field
from app.db.connection import get_db, DatabaseConnection
from app.services.graph_service import GraphService
from app.db.repositories.user_repo import UserRepository
import structlog
from app.config import settings
logger = structlog.get_logger()
from typing import List

router = APIRouter()


# Request models
class CitationGraphRequest(BaseModel):
    """Request body for citation graph generation."""
    depth: int = Field(
        default=1,
        ge=1,
        le=2,
        description="Graph depth: 1=direct connections, 2=connections between related papers"
    )
    max_nodes: int = Field(
        default=50,
        ge=10,
        le=100,
        description="Maximum number of nodes to include in graph"
    )
    include_metadata: bool = Field(
        default=True,
        description="Include full paper metadata in nodes"
    )
    embedding_model: Optional[str] = Field(
        default=None,
        pattern="^(minilm|specter)$",
        description="Embedding model for semantic similarity: 'minilm' or 'specter' (default: from config)"
    )
    recommended_papers: Optional[List[str]] = Field(
        default=None,
        description="List of paper IDs from same recommendation batch (shows recommendation context)"
    )


# Response models
class GraphNode(BaseModel):
    """Node in citation graph."""
    id: str
    label: str
    type: str
    year: Optional[int] = None
    citation_count: Optional[int] = None
    domain: Optional[str] = None
    authors: Optional[list[str]] = None
    venue: Optional[str] = None
    abstract: Optional[str] = None
    size: Optional[int] = None
    color: Optional[str] = None


class GraphEdge(BaseModel):
    """Edge in citation graph."""
    source: str
    target: str
    type: str
    strength: float
    label: str
    distance: Optional[int] = 100


class GraphStats(BaseModel):
    """Statistics about the graph."""
    total_nodes: int
    total_edges: int
    direct_citations: int
    co_citations: int
    bibliographic_couples: int
    network_centrality: float
    avg_citation_count: float


class GraphMetadata(BaseModel):
    """Metadata about the graph."""
    central_paper_id: str
    depth: int
    total_nodes: int
    total_edges: int
    has_semantic_fallback: Optional[bool] = False
    embedding_model_used: Optional[str] = None


class CitationGraphResponse(BaseModel):
    """Complete citation graph response."""
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    stats: GraphStats
    metadata: GraphMetadata


@router.post(
    "/citation-network/{paper_id}",
    response_model=CitationGraphResponse,
    summary="Get citation network graph for a paper"
)
async def get_citation_network(
    paper_id: str,
    request: CitationGraphRequest,
    db: DatabaseConnection = Depends(get_db)
):
    """
    Generate citation network graph for visualization.
    
    **Use Cases:**
    - User clicks "View Citation Graph" on a recommended paper
    - User explores paper relationships after reading recommendations
    - User navigates research landscape interactively
    
    **Graph Types by Node:**
    - `central`: The selected paper (red node)
    - `recommended_peer`: Other papers from same recommendation batch (gold)
    - `direct_citation`: Papers directly cited/citing (teal)
    - `co_cited`: Papers frequently co-cited with this one (light teal)
    - `bibliographic_couple`: Papers sharing references (yellow)
    - `semantic_similar`: AI-matched papers based on content similarity (mint green)
    
    **Embedding Models:**
    - `minilm`: Fast, efficient (384 dimensions) - all-MiniLM-L6-v2
    - `specter`: More accurate (768 dimensions) - allenai/specter2
    - If not specified, uses default from config (GRAPH_DEFAULT_MODEL)
    
    **Semantic Fallback:**
    - Automatically triggered when citation graph has < 5 nodes (configurable)
    - Uses pgvector cosine similarity to find related papers
    - Supplements citation network with AI-matched papers
    
    **Network Topology:**
    - Papers connect to each other via semantic bridges
    - Recommended peers show how selected paper relates to other recommendations
    - Edge distances vary by relationship strength
    
    **Frontend Integration:**
    ```javascript
    // After user receives recommendations and clicks on one
    const selectedPaper = "2fc7d040b64164126f0a56cf1562c7659bc2b146";
    const otherRecommendations = ["163b4d6a...", "c62de1db...", ...];
    
    const response = await fetch(
      `/api/v1/graph/citation-network/${selectedPaper}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          depth: 1,
          max_nodes: 50,
          embedding_model: 'specter',
          recommended_papers: otherRecommendations
        })
      }
    );
    
    const graphData = await response.json();
    
    // Graph includes:
    // - Red node: Selected paper
    // - Gold nodes: Other recommendations
    // - Green nodes: Semantic similar
    // - Teal/Yellow: Citations
    renderGraph(graphData.nodes, graphData.edges);
    ```
    
    **Request Body:**
    - `depth`: Graph depth (1 or 2)
    - `max_nodes`: Maximum nodes (10-100)
    - `include_metadata`: Include paper details
    - `embedding_model`: 'minilm' or 'specter'
    - `recommended_papers`: List of peer paper IDs from same recommendation batch
    
    **Returns:**
    - `nodes`: Papers with visualization properties (id, label, type, size, color)
    - `edges`: Connections with type, strength, label, and distance
    - `stats`: Graph statistics
    - `metadata`: Graph information including model used
    """
    try:
        logger.info(
            "Fetching citation network",
            paper_id=paper_id,
            depth=request.depth,
            max_nodes=request.max_nodes,
            embedding_model=request.embedding_model,
            recommended_peers=len(request.recommended_papers) if request.recommended_papers else 0
        )
        
        graph_service = GraphService(db)
        
        graph_data = await graph_service.get_citation_graph(
            paper_id=paper_id,
            depth=request.depth,
            max_nodes=request.max_nodes,
            include_metadata=request.include_metadata,
            embedding_model=request.embedding_model,
            recommended_papers=request.recommended_papers or []
        )
        
        logger.info(
            "Citation network generated successfully",
            paper_id=paper_id,
            nodes=graph_data['metadata']['total_nodes'],
            edges=graph_data['metadata']['total_edges'],
            semantic_used=graph_data['metadata'].get('has_semantic_fallback', False),
            model=graph_data['metadata'].get('embedding_model_used', False)
        )
        
        return graph_data
        
    except Exception as e:
        logger.error(
            "Failed to generate citation graph",
            paper_id=paper_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate citation graph: {str(e)}"
        )