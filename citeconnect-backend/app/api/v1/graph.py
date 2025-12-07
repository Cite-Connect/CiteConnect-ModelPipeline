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

router = APIRouter()

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
    embedding_model_used: Optional[str] = None  # NEW: Track which model was used


class CitationGraphResponse(BaseModel):
    """Complete citation graph response."""
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    stats: GraphStats
    metadata: GraphMetadata


class GraphSummaryResponse(BaseModel):
    """Quick summary of citation graph."""
    paper_id: str
    total_citations: int
    total_references: Optional[int] = None
    co_cited_papers: Optional[int] = None
    bibliographic_couples: Optional[int] = None
    network_centrality: Optional[float] = None
    has_ground_truth: bool


@router.get(
    "/citation-network/{paper_id}",
    response_model=CitationGraphResponse,
    summary="Get citation network graph for a paper"
)
async def get_citation_network(
    paper_id: str,
    depth: int = Query(
        1,
        ge=1,
        le=2,
        description="Graph depth: 1=direct connections, 2=connections between related papers"
    ),
    max_nodes: int = Query(
        50,
        ge=10,
        le=100,
        description="Maximum number of nodes to include in graph"
    ),
    include_metadata: bool = Query(
        True,
        description="Include full paper metadata in nodes"
    ),
    embedding_model: str = Query(
        None,
        regex="^(minilm|specter)$",
        description="Embedding model for semantic similarity: 'minilm' or 'specter' (default: from config)"
    ),
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
    - `direct_citation`: Papers directly cited/citing (teal)
    - `co_cited`: Papers frequently co-cited with this one (light teal)
    - `bibliographic_couple`: Papers sharing references (yellow)
    
    **Frontend Integration:**
    ```javascript
    // After user clicks on a recommended paper
    const response = await fetch(`/api/v1/graph/citation-network/${paperId}?depth=1&max_nodes=50`);
    const graphData = await response.json();
    
    // Render with D3.js, Cytoscape, or vis.js
    renderGraph(graphData.nodes, graphData.edges);
    ```
    
    **Parameters:**
    - `paper_id`: The paper to center the graph around
    - `depth`: 1 = show direct connections only, 2 = show connections between related papers
    - `max_nodes`: Limit graph size (10-100 nodes)
    - `include_metadata`: Whether to include full paper details in nodes
    
    **Returns:**
    - `nodes`: Array of papers with visualization properties (id, label, type, size, color)
    - `edges`: Array of connections with type and strength
    - `stats`: Graph statistics (citation counts, centrality, etc.)
    - `metadata`: Overall graph information
    """
    try:
        logger.info(
            "Fetching citation network",
            paper_id=paper_id,
            depth=depth,
            max_nodes=max_nodes
        )
        
        graph_service = GraphService(db)
        
        graph_data = await graph_service.get_citation_graph(
            paper_id=paper_id,
            depth=depth,
            max_nodes=max_nodes,
            include_metadata=include_metadata,
            embedding_model=embedding_model # ← THIS LINE MUST BE HERE

        )
        
        return graph_data
        
    except Exception as e:
        logger.error(
            "Failed to generate citation graph",
            paper_id=paper_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate citation graph: {str(e)}"
        )


# @router.get(
#     "/summary/{paper_id}",
#     response_model=GraphSummaryResponse,
#     summary="Get quick citation graph summary"
# )
# async def get_graph_summary(
#     paper_id: str,
#     current_user: dict = Depends(get_current_user),
#     db: DatabaseConnection = Depends(get_db)
# ):
#     """
#     Get a quick summary of citation graph without building full graph.
    
#     **Use Cases:**
#     - Show citation counts in recommendation cards
#     - Display graph size in tooltips ("View citation network with 45 papers")
#     - Quick preview before loading full graph
    
#     **Frontend Integration:**
#     ```javascript
#     // Show summary in recommendation card
#     const summary = await fetch(`/api/v1/graph/summary/${paperId}`);
#     // Display: "📊 Citation Network: 23 citations, 15 co-cited papers"
#     ```
    
#     **Returns:**
#     - `total_citations`: Number of direct citations
#     - `co_cited_papers`: Number of papers co-cited with this one
#     - `bibliographic_couples`: Papers sharing references
#     - `network_centrality`: Importance score (0-1)
#     - `has_ground_truth`: Whether enhanced relationships exist
#     """
#     try:
#         logger.debug(
#             "Fetching graph summary",
#             paper_id=paper_id,
#             user_id=current_user['user_id']
#         )
        
#         graph_service = GraphService(db)
#         summary = await graph_service.get_graph_summary(paper_id)
        
#         return summary
        
#     except Exception as e:
#         logger.error(
#             "Failed to fetch graph summary",
#             paper_id=paper_id,
#             error=str(e)
#         )
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to fetch graph summary: {str(e)}"
#         )


# @router.get(
#     "/explore/{paper_id}",
#     summary="Get explorable citation paths"
# )
# async def get_citation_paths(
#     paper_id: str,
#     target_paper_id: Optional[str] = Query(
#         None,
#         description="Find paths between source and target paper"
#     ),
#     max_paths: int = Query(
#         3,
#         ge=1,
#         le=10,
#         description="Maximum citation paths to return"
#     ),
#     current_user: dict = Depends(get_current_user),
#     db: DatabaseConnection = Depends(get_db)
# ):
#     """
#     Find citation paths between papers (coming soon).
    
#     **Use Case:**
#     - User asks: "How is Paper A related to Paper B?"
#     - Show citation chain: A → C → D → B
    
#     **Status:** Planned feature for citation graph exploration
#     """
#     raise HTTPException(
#         status_code=501,
#         detail="Citation path exploration is coming soon!"
#     )