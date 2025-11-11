# app/db/weaviate_client.py

"""
Weaviate Vector Database Client Module

This module manages connections to Weaviate vector database for
embedding storage and semantic search operations.

Features:
- Weaviate client initialization and connection management
- Schema creation and management for Paper collection
- Vector insertion and batch operations
- Semantic similarity search
- Connection health checks
- Retry logic with exponential backoff

Usage:
    from app.db.weaviate_client import get_weaviate_client, search_papers
    
    # Get client
    client = await get_weaviate_client()
    
    # Search papers
    results = await search_papers(
        query_vector=[0.1, 0.2, ...],
        limit=10,
        filters={"domain": "healthcare"}
    )
"""

import logging
import asyncio
from typing import Optional, List, Dict, Any

import weaviate
from weaviate.client import Client
from weaviate.exceptions import WeaviateBaseError

from app.core.config import get_settings
from app.core.exceptions import DatabaseError

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Global Weaviate client
_weaviate_client: Optional[Client] = None


# Weaviate schema for Paper collection
PAPER_SCHEMA = {
    "class": "Paper",
    "description": "Research paper with SPECTER embeddings",
    "vectorizer": "none",  # We provide embeddings manually
    "properties": [
        {
            "name": "paper_id",
            "dataType": ["text"],
            "description": "Unique paper identifier",
            "indexFilterable": True,
            "indexSearchable": False
        },
        {
            "name": "title",
            "dataType": ["text"],
            "description": "Paper title",
            "indexFilterable": False,
            "indexSearchable": True
        },
        {
            "name": "abstract",
            "dataType": ["text"],
            "description": "Paper abstract",
            "indexFilterable": False,
            "indexSearchable": True
        },
        {
            "name": "summary",
            "dataType": ["text"],
            "description": "AI-generated summary",
            "indexFilterable": False,
            "indexSearchable": True
        },
        {
            "name": "domain",
            "dataType": ["text"],
            "description": "Domain: healthcare, fintech, or quantum_computing",
            "indexFilterable": True,
            "indexSearchable": False
        },
        {
            "name": "year",
            "dataType": ["int"],
            "description": "Publication year",
            "indexFilterable": True,
            "indexSearchable": False
        },
        {
            "name": "citation_count",
            "dataType": ["int"],
            "description": "Number of citations",
            "indexFilterable": True,
            "indexSearchable": False
        },
        {
            "name": "authors",
            "dataType": ["text[]"],
            "description": "List of authors",
            "indexFilterable": False,
            "indexSearchable": True
        }
    ],
    "vectorIndexConfig": {
        "distance": "cosine",  # Cosine similarity for embeddings
        "ef": 64,              # HNSW parameter (query time accuracy)
        "efConstruction": 128, # HNSW parameter (build time accuracy)
        "maxConnections": 32   # HNSW parameter (graph connectivity)
    }
}


def create_weaviate_client() -> Client:
    """
    Create and return a Weaviate client.
    
    Creates a Weaviate client with configured connection settings
    and timeout parameters. Implements retry logic for robustness.
    
    Returns:
        weaviate.Client: Weaviate client instance
    
    Raises:
        DatabaseError: If client creation fails after retries
    
    Example:
        >>> client = create_weaviate_client()
        >>> print(client.is_ready())
    """
    logger.info(
        "Creating Weaviate client",
        extra={
            "url": settings.WEAVIATE_URL,
            "timeout": settings.WEAVIATE_TIMEOUT
        }
    )
    
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"Connection attempt {attempt + 1}/{max_retries}")
            
            # Create client configuration
            if settings.WEAVIATE_API_KEY:
                # With API key (for cloud/production)
                auth_config = weaviate.AuthApiKey(api_key=settings.WEAVIATE_API_KEY)
                client = weaviate.Client(
                    url=settings.WEAVIATE_URL,
                    auth_client_secret=auth_config,
                    timeout_config=(5, settings.WEAVIATE_TIMEOUT)
                )
            else:
                # Without API key (for local development)
                client = weaviate.Client(
                    url=settings.WEAVIATE_URL,
                    timeout_config=(5, settings.WEAVIATE_TIMEOUT)
                )
            
            # Test connection
            if not client.is_ready():
                raise ConnectionError("Weaviate is not ready")
            
            logger.info("Weaviate client created successfully")
            
            return client
            
        except Exception as e:
            logger.error(
                f"Failed to create Weaviate client (attempt {attempt + 1}/{max_retries}): {str(e)}",
                exc_info=True
            )
            
            if attempt < max_retries - 1:
                # Exponential backoff
                wait_time = retry_delay * (2 ** attempt)
                logger.info(f"Retrying in {wait_time} seconds...")
                import time
                time.sleep(wait_time)
            else:
                # Final attempt failed
                raise DatabaseError(
                    message=f"Failed to create Weaviate client after {max_retries} attempts",
                    operation="create_client",
                    details={"error": str(e)}
                )


def get_weaviate_client() -> Client:
    """
    Get or create the global Weaviate client.
    
    Returns the existing client if available, otherwise creates a new one.
    
    Returns:
        weaviate.Client: Weaviate client instance
    
    Raises:
        DatabaseError: If client creation fails
    
    Example:
        >>> client = get_weaviate_client()
        >>> schema = client.schema.get()
    """
    global _weaviate_client
    
    logger.debug("Getting Weaviate client")
    
    if _weaviate_client is None:
        logger.info("Weaviate client not initialized, creating new client")
        _weaviate_client = create_weaviate_client()
    
    return _weaviate_client


def close_weaviate_client() -> None:
    """
    Close the global Weaviate client.
    
    Note: Weaviate client doesn't require explicit closing in most cases,
    but this function is provided for consistency with other database modules.
    
    Example:
        >>> close_weaviate_client()
    """
    global _weaviate_client
    
    logger.info("Closing Weaviate client")
    
    if _weaviate_client is not None:
        try:
            # Weaviate client doesn't have explicit close method
            # Just set to None to allow garbage collection
            _weaviate_client = None
            logger.info("Weaviate client closed successfully")
        except Exception as e:
            logger.error(f"Error closing Weaviate client: {str(e)}", exc_info=True)
    else:
        logger.debug("No Weaviate client to close")


def create_schema() -> bool:
    """
    Create the Paper schema in Weaviate if it doesn't exist.
    
    Returns:
        True if schema was created or already exists, False on error
    
    Example:
        >>> success = create_schema()
        >>> print(f"Schema ready: {success}")
    """
    logger.info("Creating Weaviate schema")
    
    try:
        client = get_weaviate_client()
        
        # Check if Paper class already exists
        existing_schema = client.schema.get()
        existing_classes = [cls["class"] for cls in existing_schema.get("classes", [])]
        
        if "Paper" in existing_classes:
            logger.info("Paper schema already exists")
            return True
        
        # Create Paper class
        client.schema.create_class(PAPER_SCHEMA)
        
        logger.info("Paper schema created successfully")
        
        return True
        
    except WeaviateBaseError as e:
        logger.error(f"Weaviate error creating schema: {str(e)}", exc_info=True)
        raise DatabaseError(
            message=f"Failed to create schema: {str(e)}",
            operation="create_schema"
        )
        
    except Exception as e:
        logger.error(f"Unexpected error creating schema: {str(e)}", exc_info=True)
        return False


def insert_paper(
    paper_id: str,
    title: str,
    abstract: str,
    embedding: List[float],
    domain: str,
    year: int,
    citation_count: int = 0,
    authors: Optional[List[str]] = None,
    summary: Optional[str] = None
) -> bool:
    """
    Insert a single paper with its embedding into Weaviate.
    
    Args:
        paper_id: Unique paper identifier
        title: Paper title
        abstract: Paper abstract
        embedding: 768-dimensional SPECTER embedding vector
        domain: Domain (healthcare, fintech, quantum_computing)
        year: Publication year
        citation_count: Number of citations
        authors: List of author names
        summary: Optional AI-generated summary
    
    Returns:
        True if successful, False otherwise
    
    Raises:
        DatabaseError: If insertion fails
    
    Example:
        >>> embedding = [0.1, 0.2, ...] # 768 dimensions
        >>> success = insert_paper(
        ...     paper_id="arxiv:2401.12345",
        ...     title="Deep Learning for Healthcare",
        ...     abstract="This paper presents...",
        ...     embedding=embedding,
        ...     domain="healthcare",
        ...     year=2024,
        ...     citation_count=10,
        ...     authors=["John Doe", "Jane Smith"]
        ... )
    """
    logger.info(
        f"Inserting paper into Weaviate: {paper_id}",
        extra={"paper_id": paper_id, "domain": domain}
    )
    
    try:
        client = get_weaviate_client()
        
        # Prepare paper data
        paper_data = {
            "paper_id": paper_id,
            "title": title,
            "abstract": abstract,
            "domain": domain,
            "year": year,
            "citation_count": citation_count,
            "authors": authors or [],
            "summary": summary or ""
        }
        
        # Insert with embedding
        client.data_object.create(
            data_object=paper_data,
            class_name="Paper",
            vector=embedding
        )
        
        logger.debug(f"Paper inserted successfully: {paper_id}")
        
        return True
        
    except WeaviateBaseError as e:
        logger.error(f"Weaviate error inserting paper: {str(e)}", exc_info=True)
        raise DatabaseError(
            message=f"Failed to insert paper: {str(e)}",
            operation="insert_paper",
            details={"paper_id": paper_id}
        )
        
    except Exception as e:
        logger.error(f"Unexpected error inserting paper: {str(e)}", exc_info=True)
        return False


def batch_insert_papers(papers: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Insert multiple papers in batch for better performance.
    
    Args:
        papers: List of paper dictionaries, each containing:
            - paper_id, title, abstract, embedding, domain, year,
              citation_count, authors (optional), summary (optional)
    
    Returns:
        Dictionary with success and failure counts
    
    Example:
        >>> papers = [
        ...     {
        ...         "paper_id": "arxiv:2401.001",
        ...         "title": "Paper 1",
        ...         "abstract": "Abstract 1",
        ...         "embedding": [0.1, 0.2, ...],
        ...         "domain": "healthcare",
        ...         "year": 2024,
        ...         "citation_count": 5,
        ...         "authors": ["Author 1"]
        ...     },
        ...     # ... more papers
        ... ]
        >>> result = batch_insert_papers(papers)
        >>> print(f"Inserted {result['success']} papers")
    """
    logger.info(f"Batch inserting {len(papers)} papers into Weaviate")
    
    try:
        client = get_weaviate_client()
        
        # Configure batch
        client.batch.configure(
            batch_size=100,
            dynamic=True,
            timeout_retries=3,
            callback=None
        )
        
        success_count = 0
        failure_count = 0
        
        with client.batch as batch:
            for paper in papers:
                try:
                    # Prepare paper data
                    paper_data = {
                        "paper_id": paper["paper_id"],
                        "title": paper["title"],
                        "abstract": paper["abstract"],
                        "domain": paper["domain"],
                        "year": paper["year"],
                        "citation_count": paper.get("citation_count", 0),
                        "authors": paper.get("authors", []),
                        "summary": paper.get("summary", "")
                    }
                    
                    # Add to batch with embedding
                    batch.add_data_object(
                        data_object=paper_data,
                        class_name="Paper",
                        vector=paper["embedding"]
                    )
                    
                    success_count += 1
                    
                except Exception as e:
                    logger.error(
                        f"Error adding paper to batch: {paper.get('paper_id')}: {str(e)}"
                    )
                    failure_count += 1
        
        logger.info(
            f"Batch insert completed",
            extra={
                "total": len(papers),
                "success": success_count,
                "failure": failure_count
            }
        )
        
        return {
            "total": len(papers),
            "success": success_count,
            "failure": failure_count
        }
        
    except WeaviateBaseError as e:
        logger.error(f"Weaviate error during batch insert: {str(e)}", exc_info=True)
        raise DatabaseError(
            message=f"Batch insert failed: {str(e)}",
            operation="batch_insert"
        )
        
    except Exception as e:
        logger.error(f"Unexpected error during batch insert: {str(e)}", exc_info=True)
        return {
            "total": len(papers),
            "success": 0,
            "failure": len(papers)
        }


def search_papers(
    query_vector: List[float],
    limit: int = 20,
    filters: Optional[Dict[str, Any]] = None,
    certainty: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Search for papers using vector similarity.
    
    Args:
        query_vector: 768-dimensional query embedding vector
        limit: Maximum number of results to return
        filters: Optional filters (e.g., {"domain": "healthcare", "year": 2024})
        certainty: Minimum certainty score (0-1), equivalent to cosine similarity
    
    Returns:
        List of paper dictionaries with similarity scores
    
    Example:
        >>> query_embedding = [0.1, 0.2, ...] # 768 dimensions
        >>> results = search_papers(
        ...     query_vector=query_embedding,
        ...     limit=10,
        ...     filters={"domain": "healthcare", "year": 2024},
        ...     certainty=0.7
        ... )
        >>> for paper in results:
        ...     print(f"{paper['title']}: {paper['_additional']['certainty']}")
    """
    logger.info(
        "Searching papers in Weaviate",
        extra={
            "limit": limit,
            "filters": filters,
            "certainty": certainty
        }
    )
    
    try:
        client = get_weaviate_client()
        
        # Build query
        query = client.query.get("Paper", [
            "paper_id",
            "title",
            "abstract",
            "summary",
            "domain",
            "year",
            "citation_count",
            "authors"
        ]).with_near_vector({
            "vector": query_vector,
            "certainty": certainty
        }).with_limit(limit)
        
        # Add filters if provided
        if filters:
            where_filter = _build_where_filter(filters)
            if where_filter:
                query = query.with_where(where_filter)
        
        # Add additional metadata (certainty score, distance)
        query = query.with_additional(["certainty", "distance"])
        
        # Execute query
        result = query.do()
        
        # Extract papers from result
        papers = result.get("data", {}).get("Get", {}).get("Paper", [])
        
        logger.info(f"Found {len(papers)} papers matching query")
        
        return papers
        
    except WeaviateBaseError as e:
        logger.error(f"Weaviate error during search: {str(e)}", exc_info=True)
        raise DatabaseError(
            message=f"Search failed: {str(e)}",
            operation="search_papers"
        )
        
    except Exception as e:
        logger.error(f"Unexpected error during search: {str(e)}", exc_info=True)
        return []


def _build_where_filter(filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Build Weaviate where filter from simple filter dictionary.
    
    Args:
        filters: Dictionary of field: value pairs
    
    Returns:
        Weaviate where filter dictionary
    
    Example:
        >>> filters = {"domain": "healthcare", "year": 2024}
        >>> where = _build_where_filter(filters)
    """
    logger.debug(f"Building where filter: {filters}")
    
    if not filters:
        return None
    
    # Build filter conditions
    conditions = []
    
    for field, value in filters.items():
        if isinstance(value, str):
            conditions.append({
                "path": [field],
                "operator": "Equal",
                "valueText": value
            })
        elif isinstance(value, int):
            conditions.append({
                "path": [field],
                "operator": "Equal",
                "valueInt": value
            })
        elif isinstance(value, dict):
            # Handle range queries (e.g., {"year": {">=": 2020, "<=": 2024}})
            for op, val in value.items():
                operator_map = {
                    ">=": "GreaterThanEqual",
                    ">": "GreaterThan",
                    "<=": "LessThanEqual",
                    "<": "LessThan",
                    "=": "Equal"
                }
                conditions.append({
                    "path": [field],
                    "operator": operator_map.get(op, "Equal"),
                    "valueInt": val
                })
    
    if not conditions:
        return None
    
    # If multiple conditions, combine with AND
    if len(conditions) == 1:
        return conditions[0]
    else:
        return {
            "operator": "And",
            "operands": conditions
        }


def get_paper_by_id(paper_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a paper by its paper_id.
    
    Args:
        paper_id: Paper identifier
    
    Returns:
        Paper dictionary or None if not found
    
    Example:
        >>> paper = get_paper_by_id("arxiv:2401.12345")
        >>> if paper:
        ...     print(paper["title"])
    """
    logger.debug(f"Getting paper by ID: {paper_id}")
    
    try:
        client = get_weaviate_client()
        
        # Query for specific paper
        result = client.query.get("Paper", [
            "paper_id",
            "title",
            "abstract",
            "summary",
            "domain",
            "year",
            "citation_count",
            "authors"
        ]).with_where({
            "path": ["paper_id"],
            "operator": "Equal",
            "valueText": paper_id
        }).do()
        
        papers = result.get("data", {}).get("Get", {}).get("Paper", [])
        
        if papers:
            logger.debug(f"Paper found: {paper_id}")
            return papers[0]
        else:
            logger.debug(f"Paper not found: {paper_id}")
            return None
        
    except Exception as e:
        logger.error(f"Error getting paper by ID: {str(e)}", exc_info=True)
        return None


def delete_paper(paper_id: str) -> bool:
    """
    Delete a paper from Weaviate.
    
    Args:
        paper_id: Paper identifier
    
    Returns:
        True if deleted, False if not found or error
    
    Example:
        >>> success = delete_paper("arxiv:2401.12345")
    """
    logger.info(f"Deleting paper: {paper_id}")
    
    try:
        client = get_weaviate_client()
        
        # Find paper UUID
        result = client.query.get("Paper").with_where({
            "path": ["paper_id"],
            "operator": "Equal",
            "valueText": paper_id
        }).with_additional(["id"]).do()
        
        papers = result.get("data", {}).get("Get", {}).get("Paper", [])
        
        if not papers:
            logger.warning(f"Paper not found for deletion: {paper_id}")
            return False
        
        # Delete by UUID
        uuid = papers[0]["_additional"]["id"]
        client.data_object.delete(uuid, class_name="Paper")
        
        logger.info(f"Paper deleted: {paper_id}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error deleting paper: {str(e)}", exc_info=True)
        return False


def check_weaviate_health() -> bool:
    """
    Check if Weaviate is healthy and ready.
    
    Returns:
        True if Weaviate is healthy, False otherwise
    
    Example:
        >>> is_healthy = check_weaviate_health()
        >>> print(f"Weaviate status: {'OK' if is_healthy else 'Down'}")
    """
    logger.debug("Checking Weaviate health")
    
    try:
        client = get_weaviate_client()
        
        is_ready = client.is_ready()
        
        if is_ready:
            logger.debug("Weaviate health check passed")
        else:
            logger.warning("Weaviate is not ready")
        
        return is_ready
        
    except Exception as e:
        logger.error(f"Weaviate health check failed: {str(e)}", exc_info=True)
        return False


def get_weaviate_stats() -> Dict[str, Any]:
    """
    Get Weaviate statistics.
    
    Returns:
        Dictionary with Weaviate stats
    
    Example:
        >>> stats = get_weaviate_stats()
        >>> print(f"Paper count: {stats.get('object_count', 0)}")
    """
    logger.debug("Getting Weaviate statistics")
    
    try:
        client = get_weaviate_client()
        
        # Get schema
        schema = client.schema.get()
        
        # Count papers
        result = client.query.aggregate("Paper").with_meta_count().do()
        paper_count = result.get("data", {}).get("Aggregate", {}).get("Paper", [{}])[0].get("meta", {}).get("count", 0)
        
        stats = {
            "is_ready": client.is_ready(),
            "schema_classes": len(schema.get("classes", [])),
            "object_count": paper_count
        }
        
        logger.debug("Weaviate statistics retrieved", extra=stats)
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get Weaviate stats: {str(e)}", exc_info=True)
        return {}


# Initialize module logger
logger.info("Weaviate client module loaded successfully")