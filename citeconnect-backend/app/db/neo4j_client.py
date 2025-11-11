# app/db/neo4j_client.py

"""
Neo4j Graph Database Client Module

This module manages connections to Neo4j graph database for
citation network storage and graph analysis operations.

Features:
- Neo4j driver initialization and connection management
- Cypher query execution with parameterization
- Transaction management
- Citation graph operations
- Connection health checks
- Retry logic with exponential backoff

Usage:
    from app.db.neo4j_client import get_neo4j_driver, execute_query
    
    # Execute query
    result = await execute_query(
        "MATCH (p:Paper {paper_id: $paper_id}) RETURN p",
        {"paper_id": "arxiv:2401.12345"}
    )
"""

import logging
import asyncio
from typing import Optional, List, Dict, Any, Tuple
from contextlib import asynccontextmanager

from neo4j import GraphDatabase, AsyncGraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from app.core.config import get_settings
from app.core.exceptions import DatabaseError

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Global Neo4j driver
_neo4j_driver = None


async def create_neo4j_driver():
    """
    Create and return a Neo4j async driver.
    
    Creates an async Neo4j driver with configured connection settings.
    Implements retry logic with exponential backoff for robustness.
    
    Returns:
        neo4j.AsyncDriver: Neo4j async driver instance
    
    Raises:
        DatabaseError: If driver creation fails after retries
    
    Example:
        >>> driver = await create_neo4j_driver()
        >>> await driver.verify_connectivity()
    """
    logger.info(
        "Creating Neo4j driver",
        extra={
            "uri": settings.NEO4J_URI,
            "user": settings.NEO4J_USER,
            "max_pool_size": settings.NEO4J_MAX_CONNECTION_POOL_SIZE
        }
    )
    
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"Connection attempt {attempt + 1}/{max_retries}")
            
            # Create async driver
            driver = AsyncGraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
                max_connection_lifetime=settings.NEO4J_MAX_CONNECTION_LIFETIME,
                max_connection_pool_size=settings.NEO4J_MAX_CONNECTION_POOL_SIZE,
                connection_timeout=10,
                connection_acquisition_timeout=60
            )
            
            # Verify connectivity
            await driver.verify_connectivity()
            
            logger.info("Neo4j driver created successfully")
            
            return driver
            
        except ServiceUnavailable as e:
            logger.error(
                f"Neo4j service unavailable (attempt {attempt + 1}/{max_retries}): {str(e)}",
                exc_info=True
            )
            
            if attempt < max_retries - 1:
                # Exponential backoff
                wait_time = retry_delay * (2 ** attempt)
                logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                # Final attempt failed
                raise DatabaseError(
                    message=f"Failed to connect to Neo4j after {max_retries} attempts",
                    operation="create_driver",
                    details={"error": str(e)}
                )
                
        except Exception as e:
            logger.error(
                f"Failed to create Neo4j driver (attempt {attempt + 1}/{max_retries}): {str(e)}",
                exc_info=True
            )
            
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                await asyncio.sleep(wait_time)
            else:
                raise DatabaseError(
                    message=f"Failed to create Neo4j driver: {str(e)}",
                    operation="create_driver"
                )


async def get_neo4j_driver():
    """
    Get or create the global Neo4j driver.
    
    Returns the existing driver if available, otherwise creates a new one.
    
    Returns:
        neo4j.AsyncDriver: Neo4j async driver instance
    
    Raises:
        DatabaseError: If driver creation fails
    
    Example:
        >>> driver = await get_neo4j_driver()
        >>> async with driver.session() as session:
        ...     result = await session.run("RETURN 1 AS n")
    """
    global _neo4j_driver
    
    logger.debug("Getting Neo4j driver")
    
    if _neo4j_driver is None:
        logger.info("Neo4j driver not initialized, creating new driver")
        _neo4j_driver = await create_neo4j_driver()
    
    return _neo4j_driver


async def close_neo4j_driver() -> None:
    """
    Close the global Neo4j driver.
    
    Gracefully closes all connections in the driver. Should be called
    during application shutdown.
    
    Example:
        >>> await close_neo4j_driver()
    """
    global _neo4j_driver
    
    logger.info("Closing Neo4j driver")
    
    if _neo4j_driver is not None:
        try:
            await _neo4j_driver.close()
            logger.info("Neo4j driver closed successfully")
        except Exception as e:
            logger.error(f"Error closing Neo4j driver: {str(e)}", exc_info=True)
        finally:
            _neo4j_driver = None
    else:
        logger.debug("No Neo4j driver to close")


async def execute_query(
    query: str,
    parameters: Optional[Dict[str, Any]] = None,
    database: str = "neo4j"
) -> List[Dict[str, Any]]:
    """
    Execute a Cypher query and return results.
    
    Args:
        query: Cypher query string (use $param for parameters)
        parameters: Query parameters dictionary
        database: Database name (default: "neo4j")
    
    Returns:
        List of result records as dictionaries
    
    Raises:
        DatabaseError: If query execution fails
    
    Example:
        >>> # Simple query
        >>> result = await execute_query("MATCH (p:Paper) RETURN count(p) as count")
        >>> print(result[0]["count"])
        
        >>> # Parameterized query
        >>> result = await execute_query(
        ...     "MATCH (p:Paper {paper_id: $paper_id}) RETURN p",
        ...     {"paper_id": "arxiv:2401.12345"}
        ... )
    """
    logger.info(
        "Executing Neo4j query",
        extra={
            "query_preview": query[:100],
            "has_parameters": parameters is not None
        }
    )
    
    driver = await get_neo4j_driver()
    
    try:
        async with driver.session(database=database) as session:
            result = await session.run(query, parameters or {})
            
            # Convert result to list of dictionaries
            records = []
            async for record in result:
                records.append(dict(record))
            
            logger.debug(f"Query returned {len(records)} records")
            
            return records
            
    except Neo4jError as e:
        logger.error(
            f"Neo4j error executing query: {str(e)}",
            extra={"query": query[:200], "error_code": e.code},
            exc_info=True
        )
        raise DatabaseError(
            message=f"Query execution failed: {str(e)}",
            operation="execute_query",
            details={"error_code": e.code, "query": query[:200]}
        )
        
    except Exception as e:
        logger.error(f"Unexpected error executing query: {str(e)}", exc_info=True)
        raise DatabaseError(
            message=f"Query execution failed: {str(e)}",
            operation="execute_query"
        )


async def execute_write_query(
    query: str,
    parameters: Optional[Dict[str, Any]] = None,
    database: str = "neo4j"
) -> Dict[str, Any]:
    """
    Execute a write query (CREATE, UPDATE, DELETE) in a transaction.
    
    Args:
        query: Cypher query string
        parameters: Query parameters dictionary
        database: Database name (default: "neo4j")
    
    Returns:
        Summary of the query execution
    
    Raises:
        DatabaseError: If query execution fails
    
    Example:
        >>> # Create a paper node
        >>> await execute_write_query(
        ...     "CREATE (p:Paper {paper_id: $paper_id, title: $title})",
        ...     {"paper_id": "arxiv:001", "title": "Sample Paper"}
        ... )
    """
    logger.info(
        "Executing Neo4j write query",
        extra={"query_preview": query[:100]}
    )
    
    driver = await get_neo4j_driver()
    
    try:
        async with driver.session(database=database) as session:
            result = await session.run(query, parameters or {})
            
            # Get summary
            summary = await result.consume()
            
            logger.debug(
                "Write query executed",
                extra={
                    "nodes_created": summary.counters.nodes_created,
                    "relationships_created": summary.counters.relationships_created,
                    "properties_set": summary.counters.properties_set
                }
            )
            
            return {
                "nodes_created": summary.counters.nodes_created,
                "nodes_deleted": summary.counters.nodes_deleted,
                "relationships_created": summary.counters.relationships_created,
                "relationships_deleted": summary.counters.relationships_deleted,
                "properties_set": summary.counters.properties_set
            }
            
    except Neo4jError as e:
        logger.error(f"Neo4j error executing write query: {str(e)}", exc_info=True)
        raise DatabaseError(
            message=f"Write query failed: {str(e)}",
            operation="execute_write_query",
            details={"error_code": e.code}
        )
        
    except Exception as e:
        logger.error(f"Unexpected error executing write query: {str(e)}", exc_info=True)
        raise DatabaseError(
            message=f"Write query failed: {str(e)}",
            operation="execute_write_query"
        )


async def execute_transaction(queries: List[Tuple[str, Dict[str, Any]]]) -> None:
    """
    Execute multiple queries in a transaction.
    
    All queries succeed or all fail (atomic operation).
    
    Args:
        queries: List of tuples (query, parameters)
    
    Raises:
        DatabaseError: If transaction fails
    
    Example:
        >>> await execute_transaction([
        ...     ("CREATE (p:Paper {paper_id: $id})", {"id": "arxiv:001"}),
        ...     ("CREATE (p:Paper {paper_id: $id})", {"id": "arxiv:002"}),
        ...     ("MATCH (p1:Paper {paper_id: $id1}), (p2:Paper {paper_id: $id2}) "
        ...      "CREATE (p1)-[:CITES]->(p2)", {"id1": "arxiv:001", "id2": "arxiv:002"})
        ... ])
    """
    logger.info(f"Executing Neo4j transaction with {len(queries)} queries")
    
    driver = await get_neo4j_driver()
    
    async def _transaction_function(tx):
        """Execute queries within transaction."""
        for i, (query, params) in enumerate(queries):
            logger.debug(f"Executing transaction query {i + 1}/{len(queries)}")
            await tx.run(query, params)
    
    try:
        async with driver.session() as session:
            await session.execute_write(_transaction_function)
            
        logger.info("Transaction completed successfully")
        
    except Neo4jError as e:
        logger.error(f"Neo4j transaction failed: {str(e)}", exc_info=True)
        raise DatabaseError(
            message=f"Transaction failed: {str(e)}",
            operation="execute_transaction",
            details={"query_count": len(queries)}
        )
        
    except Exception as e:
        logger.error(f"Unexpected error in transaction: {str(e)}", exc_info=True)
        raise DatabaseError(
            message=f"Transaction failed: {str(e)}",
            operation="execute_transaction"
        )


async def create_paper_node(
    paper_id: str,
    title: str,
    year: int,
    domain: str,
    citation_count: int = 0
) -> bool:
    """
    Create a Paper node in Neo4j.
    
    Args:
        paper_id: Unique paper identifier
        title: Paper title
        year: Publication year
        domain: Domain (healthcare, fintech, quantum_computing)
        citation_count: Number of citations
    
    Returns:
        True if successful, False otherwise
    
    Example:
        >>> success = await create_paper_node(
        ...     paper_id="arxiv:2401.12345",
        ...     title="Deep Learning Paper",
        ...     year=2024,
        ...     domain="healthcare",
        ...     citation_count=10
        ... )
    """
    logger.info(f"Creating paper node: {paper_id}")
    
    query = """
    MERGE (p:Paper {paper_id: $paper_id})
    ON CREATE SET 
        p.title = $title,
        p.year = $year,
        p.domain = $domain,
        p.citation_count = $citation_count
    ON MATCH SET
        p.citation_count = $citation_count
    RETURN p
    """
    
    try:
        await execute_write_query(query, {
            "paper_id": paper_id,
            "title": title,
            "year": year,
            "domain": domain,
            "citation_count": citation_count
        })
        
        logger.debug(f"Paper node created: {paper_id}")
        return True
        
    except DatabaseError:
        logger.error(f"Failed to create paper node: {paper_id}")
        return False


async def create_citation_relationship(
    citing_paper_id: str,
    cited_paper_id: str,
    citation_context: Optional[str] = None
) -> bool:
    """
    Create a CITES relationship between two papers.
    
    Args:
        citing_paper_id: ID of the paper that cites
        cited_paper_id: ID of the paper being cited
        citation_context: Optional context where citation appears
    
    Returns:
        True if successful, False otherwise
    
    Example:
        >>> success = await create_citation_relationship(
        ...     citing_paper_id="arxiv:001",
        ...     cited_paper_id="arxiv:002",
        ...     citation_context="In the introduction"
        ... )
    """
    logger.info(
        f"Creating citation: {citing_paper_id} -> {cited_paper_id}",
        extra={"citing": citing_paper_id, "cited": cited_paper_id}
    )
    
    query = """
    MATCH (citing:Paper {paper_id: $citing_id})
    MATCH (cited:Paper {paper_id: $cited_id})
    MERGE (citing)-[r:CITES]->(cited)
    ON CREATE SET r.citation_context = $context
    RETURN r
    """
    
    try:
        await execute_write_query(query, {
            "citing_id": citing_paper_id,
            "cited_id": cited_paper_id,
            "context": citation_context or ""
        })
        
        logger.debug("Citation relationship created")
        return True
        
    except DatabaseError:
        logger.error("Failed to create citation relationship")
        return False


async def get_cited_papers(paper_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Get papers cited by a given paper.
    
    Args:
        paper_id: Paper identifier
        limit: Maximum number of results
    
    Returns:
        List of cited paper dictionaries
    
    Example:
        >>> cited = await get_cited_papers("arxiv:2401.12345", limit=10)
        >>> for paper in cited:
        ...     print(paper["paper_id"])
    """
    logger.info(f"Getting papers cited by: {paper_id}", extra={"limit": limit})
    
    query = """
    MATCH (p:Paper {paper_id: $paper_id})-[:CITES]->(cited:Paper)
    RETURN cited.paper_id as paper_id,
           cited.title as title,
           cited.year as year,
           cited.citation_count as citation_count
    LIMIT $limit
    """
    
    try:
        results = await execute_query(query, {
            "paper_id": paper_id,
            "limit": limit
        })
        
        logger.debug(f"Found {len(results)} cited papers")
        return results
        
    except DatabaseError:
        logger.error("Failed to get cited papers")
        return []


async def get_citing_papers(paper_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Get papers that cite a given paper.
    
    Args:
        paper_id: Paper identifier
        limit: Maximum number of results
    
    Returns:
        List of citing paper dictionaries
    
    Example:
        >>> citing = await get_citing_papers("arxiv:2401.12345", limit=10)
        >>> for paper in citing:
        ...     print(paper["paper_id"])
    """
    logger.info(f"Getting papers citing: {paper_id}", extra={"limit": limit})
    
    query = """
    MATCH (citing:Paper)-[:CITES]->(p:Paper {paper_id: $paper_id})
    RETURN citing.paper_id as paper_id,
           citing.title as title,
           citing.year as year,
           citing.citation_count as citation_count
    LIMIT $limit
    """
    
    try:
        results = await execute_query(query, {
            "paper_id": paper_id,
            "limit": limit
        })
        
        logger.debug(f"Found {len(results)} citing papers")
        return results
        
    except DatabaseError:
        logger.error("Failed to get citing papers")
        return []


async def get_co_cited_papers(paper_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Get papers that are co-cited with a given paper.
    
    Co-cited papers are papers that appear together in reference lists.
    
    Args:
        paper_id: Paper identifier
        limit: Maximum number of results
    
    Returns:
        List of co-cited paper dictionaries with co-occurrence count
    
    Example:
        >>> co_cited = await get_co_cited_papers("arxiv:2401.12345", limit=10)
        >>> for paper in co_cited:
        ...     print(f"{paper['paper_id']}: {paper['co_occurrence_count']} co-citations")
    """
    logger.info(f"Getting co-cited papers for: {paper_id}", extra={"limit": limit})
    
    query = """
    MATCH (p:Paper {paper_id: $paper_id})<-[:CITES]-(citing:Paper)-[:CITES]->(co_cited:Paper)
    WHERE co_cited.paper_id <> $paper_id
    RETURN co_cited.paper_id as paper_id,
           co_cited.title as title,
           co_cited.year as year,
           COUNT(citing) as co_occurrence_count
    ORDER BY co_occurrence_count DESC
    LIMIT $limit
    """
    
    try:
        results = await execute_query(query, {
            "paper_id": paper_id,
            "limit": limit
        })
        
        logger.debug(f"Found {len(results)} co-cited papers")
        return results
        
    except DatabaseError:
        logger.error("Failed to get co-cited papers")
        return []


async def get_citation_network(
    paper_id: str,
    depth: int = 1,
    limit: int = 25
) -> Dict[str, Any]:
    """
    Get citation network around a paper.
    
    Returns papers at specified depth from the given paper,
    including both citations and cited-by relationships.
    
    Args:
        paper_id: Central paper identifier
        depth: How many hops to traverse (1 or 2 recommended)
        limit: Maximum papers per direction
    
    Returns:
        Dictionary with nodes and edges for graph visualization
    
    Example:
        >>> network = await get_citation_network("arxiv:2401.12345", depth=1, limit=20)
        >>> print(f"Nodes: {len(network['nodes'])}, Edges: {len(network['edges'])}")
    """
    logger.info(
        f"Getting citation network for: {paper_id}",
        extra={"depth": depth, "limit": limit}
    )
    
    # Get cited papers
    cited = await get_cited_papers(paper_id, limit)
    
    # Get citing papers
    citing = await get_citing_papers(paper_id, limit)
    
    # Get co-cited papers
    co_cited = await get_co_cited_papers(paper_id, limit)
    
    # Build nodes
    nodes = {}
    
    # Add central node
    nodes[paper_id] = {
        "paper_id": paper_id,
        "type": "reference"
    }
    
    # Add cited nodes
    for paper in cited:
        nodes[paper["paper_id"]] = {
            "paper_id": paper["paper_id"],
            "title": paper["title"],
            "year": paper["year"],
            "type": "cited"
        }
    
    # Add citing nodes
    for paper in citing:
        nodes[paper["paper_id"]] = {
            "paper_id": paper["paper_id"],
            "title": paper["title"],
            "year": paper["year"],
            "type": "citing"
        }
    
    # Add co-cited nodes
    for paper in co_cited:
        if paper["paper_id"] not in nodes:
            nodes[paper["paper_id"]] = {
                "paper_id": paper["paper_id"],
                "title": paper["title"],
                "year": paper["year"],
                "type": "co-cited"
            }
    
    # Build edges
    edges = []
    
    # Citation edges
    for paper in cited:
        edges.append({
            "source": paper_id,
            "target": paper["paper_id"],
            "type": "cites"
        })
    
    for paper in citing:
        edges.append({
            "source": paper["paper_id"],
            "target": paper_id,
            "type": "cites"
        })
    
    logger.info(
        f"Citation network built",
        extra={"nodes": len(nodes), "edges": len(edges)}
    )
    
    return {
        "nodes": list(nodes.values()),
        "edges": edges
    }


async def check_neo4j_health() -> bool:
    """
    Check if Neo4j connection is healthy.
    
    Performs a simple query to verify connectivity and responsiveness.
    
    Returns:
        True if Neo4j is healthy, False otherwise
    
    Example:
        >>> is_healthy = await check_neo4j_health()
        >>> print(f"Neo4j status: {'OK' if is_healthy else 'Down'}")
    """
    logger.debug("Checking Neo4j health")
    
    try:
        driver = await get_neo4j_driver()
        
        # Verify connectivity
        await driver.verify_connectivity()
        
        logger.debug("Neo4j health check passed")
        return True
        
    except Exception as e:
        logger.error(f"Neo4j health check failed: {str(e)}", exc_info=True)
        return False


async def get_neo4j_stats() -> Dict[str, Any]:
    """
    Get Neo4j database statistics.
    
    Returns:
        Dictionary with database stats
    
    Example:
        >>> stats = await get_neo4j_stats()
        >>> print(f"Paper count: {stats.get('paper_count', 0)}")
    """
    logger.debug("Getting Neo4j statistics")
    
    try:
        # Count papers
        paper_result = await execute_query("MATCH (p:Paper) RETURN count(p) as count")
        paper_count = paper_result[0]["count"] if paper_result else 0
        
        # Count citations
        citation_result = await execute_query(
            "MATCH ()-[r:CITES]->() RETURN count(r) as count"
        )
        citation_count = citation_result[0]["count"] if citation_result else 0
        
        stats = {
            "is_healthy": await check_neo4j_health(),
            "paper_count": paper_count,
            "citation_count": citation_count
        }
        
        logger.debug("Neo4j statistics retrieved", extra=stats)
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get Neo4j stats: {str(e)}", exc_info=True)
        return {}


# Initialize module logger
logger.info("Neo4j client module loaded successfully")