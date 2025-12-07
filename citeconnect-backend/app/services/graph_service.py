"""
Citation Graph Service
Generates citation network data for frontend visualization.
"""

from typing import Dict, List, Optional, Set
from app.db.connection import DatabaseConnection
from app.db.repositories.ground_truth_repo import GroundTruthRepository
from app.db.repositories.paper_repo import PaperRepository
import structlog
from app.config import settings

logger = structlog.get_logger()

class GraphService:
    """Service for generating citation graph data with semantic fallback."""
    
    def __init__(self, db: DatabaseConnection):
        self.db = db
        self.gt_repo = GroundTruthRepository(db)
        self.paper_repo = PaperRepository(db)
        
        # Load configuration
        self.min_similarity = settings.GRAPH_SEMANTIC_MIN_SIMILARITY
        self.semantic_limit = settings.GRAPH_SEMANTIC_LIMIT
        self.hybrid_threshold = settings.GRAPH_HYBRID_TRIGGER_THRESHOLD
        self.default_model = settings.GRAPH_DEFAULT_MODEL
        self.embedding_models = settings.EMBEDDING_MODELS
        
        logger.info(
            "GraphService initialized",
            min_similarity=self.min_similarity,
            semantic_limit=self.semantic_limit,
            hybrid_threshold=self.hybrid_threshold,
            default_model=self.default_model
        )
    
    async def get_citation_graph(
        self,
        paper_id: str,
        depth: int = 1,
        max_nodes: int = 50,
        include_metadata: bool = True,
        embedding_model: str = None  # NEW: 'minilm' or 'specter'
    ) -> Dict:
        """
        Generate citation graph for a paper.
        
        Args:
            paper_id: Central paper ID
            depth: How many levels deep (1 or 2)
            max_nodes: Maximum nodes to include
            include_metadata: Include paper metadata in nodes
            embedding_model: Model for semantic similarity ('minilm' or 'specter')
            
        Returns:
            Graph data in format ready for D3.js/Cytoscape/vis.js
        """
        # Validate and set embedding model
        if embedding_model is None:
            embedding_model = self.default_model
        
        if embedding_model not in self.embedding_models:
            logger.warning(
                "Invalid embedding model requested, using default",
                requested=embedding_model,
                default=self.default_model
            )
            embedding_model = self.default_model
        
        logger.info(
            "Generating citation graph",
            paper_id=paper_id,
            depth=depth,
            max_nodes=max_nodes,
            embedding_model=embedding_model
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
        
        # HYBRID: Supplement with semantic similarity if sparse graph
        if len(nodes) < self.hybrid_threshold:
            logger.info(
                "Citation graph sparse, triggering semantic fallback",
                current_nodes=len(nodes),
                threshold=self.hybrid_threshold,
                embedding_model=embedding_model
            )
            
            try:
                hybrid_result = await self._build_hybrid_graph(
                    paper_id,
                    list(nodes.values()),
                    edges,
                    max_nodes,
                    embedding_model
                )
                nodes = {n['id']: n for n in hybrid_result['nodes']}
                edges = hybrid_result['edges']
                
                logger.info(
                    "Semantic fallback completed",
                    final_node_count=len(nodes),
                    semantic_nodes_added=len(nodes) - len(list(nodes.values()))
                )
            except Exception as e:
                logger.error(
                    "Semantic fallback failed, continuing with citation graph",
                    error=str(e),
                    exc_info=True
                )
        else:
            logger.debug(
                "Citation graph sufficient, skipping semantic fallback",
                node_count=len(nodes),
                threshold=self.hybrid_threshold
            )
        
        # Calculate graph statistics
        stats = self._calculate_graph_stats(nodes, edges, gt_relationships)
        
        logger.info(
            "Citation graph generated",
            paper_id=paper_id,
            node_count=len(nodes),
            edge_count=len(edges),
            embedding_model=embedding_model
        )
        
        return {
            'nodes': list(nodes.values()),
            'edges': edges,
            'stats': stats,
            'metadata': {
                'central_paper_id': paper_id,
                'depth': depth,
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'has_semantic_fallback': len(nodes) < self.hybrid_threshold,
                'embedding_model_used': embedding_model  # NEW: Track which model was used
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
            'citation': '#74B9FF',          # Light blue
            'semantic_similar': '#A8E6CF'   # Mint green - NEW!
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
    
    async def _get_semantic_similar_papers(
        self,
        paper_id: str,
        remaining_slots: int,
        embedding_model: str,
        min_similarity: float = None
    ) -> List[Dict]:
        """
        Find semantically similar papers using specified embedding model.
        Supports both MiniLM and SPECTER embeddings via pgvector.
        
        Args:
            paper_id: Central paper
            remaining_slots: How many more nodes to add
            embedding_model: 'minilm' or 'specter'
            min_similarity: Override minimum similarity threshold
            
        Returns:
            List of similar papers with similarity scores
        """
        if min_similarity is None:
            min_similarity = self.min_similarity
        
        # Get model configuration
        model_config = self.embedding_models.get(embedding_model)
        if not model_config:
            logger.error(
                "Invalid embedding model",
                model=embedding_model,
                available=list(self.embedding_models.keys())
            )
            return []
        
        embeddings_table = model_config['table']
        model_name = model_config['name']
        
        logger.debug(
            "Finding semantic similar papers",
            paper_id=paper_id,
            slots=remaining_slots,
            model=embedding_model,
            table=embeddings_table,
            min_similarity=min_similarity
        )
        
        # Get paper details for domain filtering
        try:
            paper = await self.paper_repo.find_by_paper_id(paper_id)
            if not paper:
                logger.warning("Paper not found", paper_id=paper_id)
                return []
        except Exception as e:
            logger.error(
                "Failed to fetch paper details",
                paper_id=paper_id,
                error=str(e),
                exc_info=True
            )
            return []
        
        # Query for semantically similar papers using pgvector
        query = f"""
            WITH target_embedding AS (
                SELECT embedding
                FROM {embeddings_table}
                WHERE paper_id = $1
            )
            SELECT 
                p.paper_id,
                p.title,
                p.year,
                p.citation_count,
                p.domain,
                p.authors,
                p.venue,
                p.abstract,
                -- Cosine similarity: 1 - (embedding <=> target)
                1 - (pe.embedding <=> (SELECT embedding FROM target_embedding)) as similarity_score
            FROM papers p
            JOIN {embeddings_table} pe ON p.paper_id = pe.paper_id
            CROSS JOIN target_embedding
            WHERE p.paper_id != $1
              AND p.domain = $2
              AND pe.embedding IS NOT NULL
              -- Cosine similarity threshold
              AND (1 - (pe.embedding <=> (SELECT embedding FROM target_embedding))) >= $3
            ORDER BY similarity_score DESC
            LIMIT $4
        """
        
        try:
            similar_papers = await self.db.fetch(
                query,
                paper_id,
                paper['domain'],
                min_similarity,
                min(remaining_slots, self.semantic_limit)
            )
            
            logger.info(
                "Semantic similar papers retrieved",
                paper_id=paper_id,
                count=len(similar_papers),
                model=embedding_model,
                domain=paper['domain']
            )
            
            return list(similar_papers)
            
        except Exception as e:
            logger.error(
                "Semantic similarity query failed",
                paper_id=paper_id,
                embedding_model=embedding_model,
                table=embeddings_table,
                error=str(e),
                exc_info=True
            )
            
            # Fallback to domain filtering
            logger.info(
                "Falling back to domain filtering",
                paper_id=paper_id
            )
            return await self._get_domain_filtered_papers(
                paper_id,
                paper['domain'],
                paper.get('year', 2020),
                remaining_slots
            )
    
    async def _get_domain_filtered_papers(
        self,
        paper_id: str,
        domain: str,
        year: int,
        limit: int
    ) -> List[Dict]:
        """
        Fallback: Find papers by domain and year similarity.
        Used when pgvector query fails or embeddings don't exist.
        
        Args:
            paper_id: Paper to exclude
            domain: Domain filter
            year: Paper year for range filtering
            limit: Maximum papers to return
        """
        year_range = settings.GRAPH_DOMAIN_YEAR_RANGE
        
        logger.debug(
            "Using domain filtering fallback",
            domain=domain,
            year_range=year_range
        )
        
        query = """
            SELECT 
                paper_id, title, year, citation_count,
                domain, authors, venue, abstract
            FROM papers
            WHERE paper_id != $1
              AND domain = $2
              AND ABS(year - $3) <= $4
            ORDER BY citation_count DESC
            LIMIT $5
        """
        
        try:
            results = await self.db.fetch(
                query,
                paper_id,
                domain,
                year,
                year_range,
                limit
            )
            
            logger.info(
                "Domain filtered papers retrieved",
                count=len(results),
                domain=domain
            )
            
            return list(results)
            
        except Exception as e:
            logger.error(
                "Domain filtering failed",
                paper_id=paper_id,
                domain=domain,
                error=str(e),
                exc_info=True
            )
            return []
    
    async def _build_hybrid_graph(
        self,
        paper_id: str,
        citation_nodes: List[Dict],
        citation_edges: List[Dict],
        max_nodes: int,
        embedding_model: str
    ) -> Dict:
        """
        Build hybrid graph combining citation + semantic relationships.
        Adds semantic nodes when citation graph is sparse.
        
        Args:
            paper_id: Central paper
            citation_nodes: Existing citation nodes
            citation_edges: Existing citation edges
            max_nodes: Maximum total nodes
            embedding_model: Embedding model to use for semantic similarity
        """
        current_node_count = len(citation_nodes)
        
        logger.info(
            "Building hybrid graph",
            citation_nodes=current_node_count,
            target_nodes=max_nodes,
            embedding_model=embedding_model
        )
        
        # Calculate remaining slots
        remaining_slots = max_nodes - current_node_count
        
        if remaining_slots <= 0:
            logger.debug("No slots remaining for semantic nodes")
            return {
                'nodes': citation_nodes,
                'edges': citation_edges
            }
        
        # Get semantic similar papers
        try:
            similar_papers = await self._get_semantic_similar_papers(
                paper_id,
                remaining_slots=remaining_slots,
                embedding_model=embedding_model
            )
            
            if not similar_papers:
                logger.warning(
                    "No semantic similar papers found",
                    paper_id=paper_id,
                    model=embedding_model
                )
                return {
                    'nodes': citation_nodes,
                    'edges': citation_edges
                }
            
        except Exception as e:
            logger.error(
                "Failed to retrieve semantic papers",
                paper_id=paper_id,
                error=str(e),
                exc_info=True
            )
            return {
                'nodes': citation_nodes,
                'edges': citation_edges
            }
        
        # Add semantic nodes and edges
        existing_ids = {node['id'] for node in citation_nodes}
        semantic_nodes_added = 0
        
        for sim_paper in similar_papers:
            if sim_paper['paper_id'] not in existing_ids:
                # Add node
                citation_nodes.append(
                    self._create_node(
                        sim_paper,
                        node_type='semantic_similar',
                        metadata=True
                    )
                )
                
                # Add edge with similarity score
                citation_edges.append({
                    'source': paper_id,
                    'target': sim_paper['paper_id'],
                    'type': 'semantic_similarity',
                    'strength': float(sim_paper.get('similarity_score', 0.5)),
                    'label': f"{int(sim_paper.get('similarity_score', 0.5) * 100)}% similar"
                })
                
                semantic_nodes_added += 1
        
        logger.info(
            "Hybrid graph built",
            total_nodes=len(citation_nodes),
            semantic_added=semantic_nodes_added,
            model=embedding_model
        )
        
        return {
            'nodes': citation_nodes,
            'edges': citation_edges
        }