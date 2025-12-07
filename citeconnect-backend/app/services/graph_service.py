"""
Citation Graph Service
Generates citation network data for frontend visualization.
"""

from typing import Dict, List, Optional, Set
from app.db.connection import DatabaseConnection
from app.db.repositories.ground_truth_repo import GroundTruthRepository
from app.db.repositories.paper_repo import PaperRepository
import structlog

logger = structlog.get_logger()


class GraphService:
    """Service for generating citation graph data."""
    
    def __init__(self, db: DatabaseConnection):
        self.db = db
        self.gt_repo = GroundTruthRepository(db)
        self.paper_repo = PaperRepository(db)
    
    async def get_citation_graph(
        self,
        paper_id: str,
        depth: int = 1,
        max_nodes: int = 50,
        include_metadata: bool = True
    ) -> Dict:
        """
        Generate citation graph for a paper.
        
        Args:
            paper_id: Central paper ID
            depth: How many levels deep (1 or 2)
            max_nodes: Maximum nodes to include
            include_metadata: Include paper metadata in nodes
            
        Returns:
            Graph data in format ready for D3.js/Cytoscape/vis.js
        """
        logger.info(
            "Generating citation graph",
            paper_id=paper_id,
            depth=depth,
            max_nodes=max_nodes
        )
        
        # Get ground truth relationships for the paper
        gt_relationships = await self.gt_repo.get_ground_truth_relationships(
            paper_id
        )
        
        if not gt_relationships:
            # Fall back to basic citation/reference data
            return await self._build_basic_graph(paper_id, depth, max_nodes)
        
        # Build graph from ground truth relationships
        nodes = {}
        edges = []
        visited = set()
        
        # Add central node
        central_paper = await self.paper_repo.find_by_paper_id(paper_id)
        nodes[paper_id] = self._create_node(
            central_paper,
            node_type='central',
            metadata=include_metadata
        )
        visited.add(paper_id)
        
        # Add citation network (direct citations/references)
        citation_network = gt_relationships.get('citation_network', [])
        for cited_id in citation_network[:max_nodes//2]:
            if cited_id not in visited:
                cited_paper = await self.paper_repo.find_by_paper_id(cited_id)
                if cited_paper:
                    nodes[cited_id] = self._create_node(
                        cited_paper,
                        node_type='direct_citation',
                        metadata=include_metadata
                    )
                    edges.append({
                        'source': paper_id,
                        'target': cited_id,
                        'type': 'cites',
                        'strength': 1.0,
                        'label': 'cites'
                    })
                    visited.add(cited_id)
        
        # Add co-cited papers (if space allows)
        remaining_slots = max_nodes - len(nodes)
        co_cited = gt_relationships.get('co_cited_papers', [])
        
        for co_cited_id in co_cited[:remaining_slots//2]:
            if co_cited_id not in visited:
                co_paper = await self.paper_repo.find_by_paper_id(co_cited_id)
                if co_paper:
                    nodes[co_cited_id] = self._create_node(
                        co_paper,
                        node_type='co_cited',
                        metadata=include_metadata
                    )
                    edges.append({
                        'source': paper_id,
                        'target': co_cited_id,
                        'type': 'co_cited',
                        'strength': 0.7,
                        'label': 'frequently co-cited with'
                    })
                    visited.add(co_cited_id)
        
        # Add bibliographic couples (if space allows)
        remaining_slots = max_nodes - len(nodes)
        bib_couples = gt_relationships.get('bibliographic_couples', [])
        
        for couple_id in bib_couples[:remaining_slots]:
            if couple_id not in visited:
                couple_paper = await self.paper_repo.find_by_paper_id(couple_id)
                if couple_paper:
                    nodes[couple_id] = self._create_node(
                        couple_paper,
                        node_type='bibliographic_couple',
                        metadata=include_metadata
                    )
                    edges.append({
                        'source': paper_id,
                        'target': couple_id,
                        'type': 'bibliographic_coupling',
                        'strength': 0.5,
                        'label': 'shares references with'
                    })
                    visited.add(couple_id)
        
        # If depth=2, add connections between related papers
        if depth == 2:
            await self._add_second_degree_connections(
                nodes, edges, visited, max_nodes
            )
        
        # Calculate graph statistics
        stats = self._calculate_graph_stats(nodes, edges, gt_relationships)
        
        logger.info(
            "Citation graph generated",
            paper_id=paper_id,
            node_count=len(nodes),
            edge_count=len(edges)
        )
        
        return {
            'nodes': list(nodes.values()),
            'edges': edges,
            'stats': stats,
            'metadata': {
                'central_paper_id': paper_id,
                'depth': depth,
                'total_nodes': len(nodes),
                'total_edges': len(edges)
            }
        }
    
    async def _build_basic_graph(
        self,
        paper_id: str,
        depth: int,
        max_nodes: int
    ) -> Dict:
        """
        Fallback: Build basic graph from paper citations/references.
        Used when ground truth relationships don't exist.
        """
        logger.debug("Building basic citation graph", paper_id=paper_id)
        
        nodes = {}
        edges = []
        visited = set()
        
        # Central paper
        central_paper = await self.paper_repo.find_by_paper_id(paper_id)
        nodes[paper_id] = self._create_node(
            central_paper,
            node_type='central',
            metadata=True
        )
        visited.add(paper_id)
        
        # Get direct citations and references
        citations = await self.paper_repo.get_paper_citations(paper_id)
        references = await self.paper_repo.get_paper_references(paper_id)
        
        # Add references (papers this paper cites)
        for ref_id in references[:max_nodes//2]:
            if ref_id not in visited:
                ref_paper = await self.paper_repo.find_by_paper_id(ref_id)
                if ref_paper:
                    nodes[ref_id] = self._create_node(
                        ref_paper,
                        node_type='reference',
                        metadata=True
                    )
                    edges.append({
                        'source': paper_id,
                        'target': ref_id,
                        'type': 'cites',
                        'strength': 1.0,
                        'label': 'cites'
                    })
                    visited.add(ref_id)
        
        # Add citations (papers citing this paper)
        remaining = max_nodes - len(nodes)
        for cite_id in citations[:remaining]:
            if cite_id not in visited:
                cite_paper = await self.paper_repo.find_by_paper_id(cite_id)
                if cite_paper:
                    nodes[cite_id] = self._create_node(
                        cite_paper,
                        node_type='citation',
                        metadata=True
                    )
                    edges.append({
                        'source': cite_id,
                        'target': paper_id,
                        'type': 'cites',
                        'strength': 1.0,
                        'label': 'cites'
                    })
                    visited.add(cite_id)
        
        stats = {
            'total_citations': len(citations),
            'total_references': len(references),
            'network_centrality': 0.0
        }
        
        return {
            'nodes': list(nodes.values()),
            'edges': edges,
            'stats': stats,
            'metadata': {
                'central_paper_id': paper_id,
                'depth': depth,
                'total_nodes': len(nodes),
                'total_edges': len(edges)
            }
        }
    
    def _create_node(
        self,
        paper: Dict,
        node_type: str,
        metadata: bool = True
    ) -> Dict:
        """
        Create a node object for the graph.
        
        Args:
            paper: Paper data from database
            node_type: Type of node (central, direct_citation, etc.)
            metadata: Whether to include full metadata
        """
        node = {
            'id': paper['paper_id'],
            'label': paper.get('title', 'Unknown'),
            'type': node_type,
        }
        
        if metadata:
            node.update({
                'year': paper.get('year'),
                'citation_count': paper.get('citation_count', 0),
                'domain': paper.get('domain'),
                'authors': paper.get('authors', [])[:3],  # First 3 authors
                'venue': paper.get('venue'),
                'abstract': paper.get('abstract', '')[:200],  # First 200 chars
                # Visual properties based on type
                'size': self._calculate_node_size(
                    paper.get('citation_count', 0),
                    node_type
                ),
                'color': self._get_node_color(node_type),
            })
        
        return node
    
    def _calculate_node_size(
        self,
        citation_count: int,
        node_type: str
    ) -> int:
        """Calculate node size based on citation count and type."""
        if node_type == 'central':
            return 30  # Central node is always largest
        
        # Size based on citation count (logarithmic scale)
        import math
        base_size = 10
        citation_factor = math.log(citation_count + 1) * 3
        
        return int(min(base_size + citation_factor, 25))
    
    def _get_node_color(self, node_type: str) -> str:
        """Get node color based on type."""
        color_map = {
            'central': '#FF6B6B',           # Red
            'direct_citation': '#4ECDC4',   # Teal
            'co_cited': '#95E1D3',          # Light teal
            'bibliographic_couple': '#FFE66D',  # Yellow
            'reference': '#6C5CE7',         # Purple
            'citation': '#74B9FF'           # Light blue
        }
        return color_map.get(node_type, '#95A5A6')  # Gray default
    
    async def _add_second_degree_connections(
        self,
        nodes: Dict,
        edges: List[Dict],
        visited: Set[str],
        max_nodes: int
    ):
        """
        Add connections between related papers (depth=2).
        Only adds if nodes already exist in graph.
        """
        # Get all current node IDs except central
        node_ids = [nid for nid in nodes.keys() if nodes[nid]['type'] != 'central']
        
        # For each node, check if it cites other nodes in graph
        for node_id in node_ids[:10]:  # Limit checks to avoid too many queries
            references = await self.paper_repo.get_paper_references(node_id)
            
            for ref_id in references:
                if ref_id in nodes and ref_id != node_id:
                    # Add edge between two existing nodes
                    edges.append({
                        'source': node_id,
                        'target': ref_id,
                        'type': 'secondary_citation',
                        'strength': 0.3,
                        'label': 'cites'
                    })
    
    def _calculate_graph_stats(
        self,
        nodes: Dict,
        edges: List[Dict],
        gt_relationships: Dict
    ) -> Dict:
        """Calculate graph statistics for frontend display."""
        return {
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'direct_citations': sum(
                1 for e in edges if e['type'] == 'cites'
            ),
            'co_citations': sum(
                1 for e in edges if e['type'] == 'co_cited'
            ),
            'bibliographic_couples': sum(
                1 for e in edges if e['type'] == 'bibliographic_coupling'
            ),
            'network_centrality': gt_relationships.get('network_centrality', 0.0),
            'avg_citation_count': sum(
                n.get('citation_count', 0) for n in nodes.values()
            ) / len(nodes) if nodes else 0,
        }
    
    async def get_graph_summary(self, paper_id: str) -> Dict:
        """
        Get a quick summary of the citation graph without building full graph.
        Used for preview/tooltips.
        """
        gt_relationships = await self.gt_repo.get_ground_truth_relationships(
            paper_id
        )
        
        if not gt_relationships:
            citations = await self.paper_repo.get_paper_citations(paper_id)
            references = await self.paper_repo.get_paper_references(paper_id)
            
            return {
                'paper_id': paper_id,
                'total_citations': len(citations),
                'total_references': len(references),
                'has_ground_truth': False
            }
        
        return {
            'paper_id': paper_id,
            'total_citations': len(gt_relationships.get('citation_network', [])),
            'co_cited_papers': len(gt_relationships.get('co_cited_papers', [])),
            'bibliographic_couples': len(gt_relationships.get('bibliographic_couples', [])),
            'network_centrality': gt_relationships.get('network_centrality', 0.0),
            'has_ground_truth': True
        }