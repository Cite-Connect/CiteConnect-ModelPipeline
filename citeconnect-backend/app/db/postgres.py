# app/db/postgres.py

"""
PostgreSQL Database Connection Module

This module manages PostgreSQL database connections using asyncpg for
async operations and psycopg2 for sync operations when needed.

Features:
- Async connection pooling with asyncpg
- Automatic retry logic with exponential backoff
- Connection health checks
- Proper connection lifecycle management
- Query logging and performance monitoring

Usage:
    from app.db.postgres import get_db_pool, get_db_connection
    
    # Get connection pool
    pool = await get_db_pool()
    
    # Execute query
    async with pool.acquire() as conn:
        result = await conn.fetchrow("SELECT * FROM users WHERE user_id = $1", user_id)
"""

import logging
import asyncio
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

import asyncpg
from asyncpg.pool import Pool

from app.core.config import get_settings
from app.core.exceptions import DatabaseError

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Global connection pool
_pool: Optional[Pool] = None


async def create_db_pool() -> Pool:
    """
    Create and return a PostgreSQL connection pool.
    
    Creates an asyncpg connection pool with configured size and timeout settings.
    Implements retry logic with exponential backoff for robustness.
    
    Returns:
        asyncpg.Pool: Connection pool instance
    
    Raises:
        DatabaseError: If pool creation fails after retries
    
    Example:
        >>> pool = await create_db_pool()
        >>> async with pool.acquire() as conn:
        ...     result = await conn.fetch("SELECT 1")
    """
    logger.info(
        "Creating PostgreSQL connection pool",
        extra={
            "host": settings.POSTGRES_HOST,
            "port": settings.POSTGRES_PORT,
            "database": settings.POSTGRES_DB,
            "pool_size": settings.DB_POOL_SIZE
        }
    )
    
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"Connection attempt {attempt + 1}/{max_retries}")
            
            # Create connection pool
            pool = await asyncpg.create_pool(
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD,
                database=settings.POSTGRES_DB,
                min_size=settings.DB_POOL_SIZE // 2,  # Minimum connections
                max_size=settings.DB_POOL_SIZE + settings.DB_MAX_OVERFLOW,
                command_timeout=60,  # Command timeout in seconds
                max_queries=50000,   # Max queries per connection before recycling
                max_inactive_connection_lifetime=300  # 5 minutes
            )
            
            # Test connection
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            
            logger.info(
                "PostgreSQL connection pool created successfully",
                extra={
                    "min_size": settings.DB_POOL_SIZE // 2,
                    "max_size": settings.DB_POOL_SIZE + settings.DB_MAX_OVERFLOW
                }
            )
            
            return pool
            
        except Exception as e:
            logger.error(
                f"Failed to create connection pool (attempt {attempt + 1}/{max_retries}): {str(e)}",
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
                    message=f"Failed to create database pool after {max_retries} attempts",
                    operation="create_pool",
                    details={"error": str(e)}
                )


async def get_db_pool() -> Pool:
    """
    Get or create the global database connection pool.
    
    Returns the existing pool if available, otherwise creates a new one.
    This function ensures only one pool exists throughout the application lifecycle.
    
    Returns:
        asyncpg.Pool: Connection pool instance
    
    Raises:
        DatabaseError: If pool creation fails
    
    Example:
        >>> pool = await get_db_pool()
        >>> async with pool.acquire() as conn:
        ...     users = await conn.fetch("SELECT * FROM users LIMIT 10")
    """
    global _pool
    
    logger.debug("Getting database connection pool")
    
    if _pool is None:
        logger.info("Pool not initialized, creating new pool")
        _pool = await create_db_pool()
    
    return _pool


async def close_db_pool() -> None:
    """
    Close the global database connection pool.
    
    Gracefully closes all connections in the pool. Should be called
    during application shutdown.
    
    Example:
        >>> await close_db_pool()
    """
    global _pool
    
    logger.info("Closing database connection pool")
    
    if _pool is not None:
        try:
            await _pool.close()
            logger.info("Database connection pool closed successfully")
        except Exception as e:
            logger.error(f"Error closing database pool: {str(e)}", exc_info=True)
        finally:
            _pool = None
    else:
        logger.debug("No pool to close")


@asynccontextmanager
async def get_db_connection() -> AsyncGenerator[asyncpg.Connection, None]:
    """
    Get a database connection from the pool (context manager).
    
    Provides a context manager that automatically acquires and releases
    a connection from the pool.
    
    Yields:
        asyncpg.Connection: Database connection
    
    Raises:
        DatabaseError: If connection cannot be acquired
    
    Example:
        >>> async with get_db_connection() as conn:
        ...     result = await conn.fetchrow("SELECT * FROM users WHERE user_id = $1", 123)
        ...     print(result['email'])
    """
    logger.debug("Acquiring database connection from pool")
    
    pool = await get_db_pool()
    
    try:
        async with pool.acquire() as connection:
            logger.debug("Database connection acquired")
            yield connection
            logger.debug("Database connection released")
            
    except Exception as e:
        logger.error(f"Error with database connection: {str(e)}", exc_info=True)
        raise DatabaseError(
            message="Database connection error",
            operation="acquire_connection",
            details={"error": str(e)}
        )


async def execute_query(
    query: str,
    *args,
    fetch_one: bool = False,
    fetch_all: bool = False
):
    """
    Execute a database query with automatic connection management.
    
    Convenience function that handles connection acquisition and execution.
    
    Args:
        query: SQL query string (use $1, $2, etc. for parameters)
        *args: Query parameters
        fetch_one: If True, return single row
        fetch_all: If True, return all rows
    
    Returns:
        Query result (row, list of rows, or None)
    
    Raises:
        DatabaseError: If query execution fails
    
    Example:
        >>> # Insert
        >>> await execute_query(
        ...     "INSERT INTO users (email, name) VALUES ($1, $2)",
        ...     "user@example.com", "John Doe"
        ... )
        
        >>> # Fetch one
        >>> user = await execute_query(
        ...     "SELECT * FROM users WHERE user_id = $1",
        ...     123,
        ...     fetch_one=True
        ... )
        
        >>> # Fetch all
        >>> users = await execute_query(
        ...     "SELECT * FROM users WHERE domain = $1",
        ...     "healthcare",
        ...     fetch_all=True
        ... )
    """
    logger.info(
        "Executing database query",
        extra={
            "query_preview": query[:100],
            "param_count": len(args),
            "fetch_one": fetch_one,
            "fetch_all": fetch_all
        }
    )
    
    async with get_db_connection() as conn:
        try:
            if fetch_one:
                result = await conn.fetchrow(query, *args)
                logger.debug(f"Query returned single row: {result is not None}")
                return result
                
            elif fetch_all:
                result = await conn.fetch(query, *args)
                logger.debug(f"Query returned {len(result)} rows")
                return result
                
            else:
                # Execute without fetching (INSERT, UPDATE, DELETE)
                result = await conn.execute(query, *args)
                logger.debug(f"Query executed: {result}")
                return result
                
        except asyncpg.PostgresError as e:
            logger.error(
                f"PostgreSQL error executing query: {str(e)}",
                extra={"query": query[:200], "error_code": e.sqlstate},
                exc_info=True
            )
            raise DatabaseError(
                message=f"Query execution failed: {str(e)}",
                operation="execute_query",
                details={"error_code": e.sqlstate, "query": query[:200]}
            )
            
        except Exception as e:
            logger.error(f"Unexpected error executing query: {str(e)}", exc_info=True)
            raise DatabaseError(
                message=f"Query execution failed: {str(e)}",
                operation="execute_query"
            )


async def execute_transaction(queries: list) -> None:
    """
    Execute multiple queries in a transaction.
    
    All queries succeed or all fail (atomic operation).
    
    Args:
        queries: List of tuples (query, args)
    
    Raises:
        DatabaseError: If transaction fails
    
    Example:
        >>> await execute_transaction([
        ...     ("INSERT INTO users (email, name) VALUES ($1, $2)", "user@example.com", "John"),
        ...     ("INSERT INTO user_interests (user_id, keyword) VALUES ($1, $2)", 123, "ML")
        ... ])
    """
    logger.info(f"Executing transaction with {len(queries)} queries")
    
    async with get_db_connection() as conn:
        transaction = conn.transaction()
        
        try:
            await transaction.start()
            logger.debug("Transaction started")
            
            for i, query_data in enumerate(queries):
                query = query_data[0]
                args = query_data[1:] if len(query_data) > 1 else ()
                
                logger.debug(f"Executing query {i + 1}/{len(queries)}")
                await conn.execute(query, *args)
            
            await transaction.commit()
            logger.info("Transaction committed successfully")
            
        except Exception as e:
            await transaction.rollback()
            logger.error(f"Transaction failed, rolled back: {str(e)}", exc_info=True)
            raise DatabaseError(
                message=f"Transaction failed: {str(e)}",
                operation="execute_transaction",
                details={"query_count": len(queries)}
            )


async def check_db_health() -> bool:
    """
    Check if database connection is healthy.
    
    Performs a simple query to verify database connectivity and responsiveness.
    
    Returns:
        True if database is healthy, False otherwise
    
    Example:
        >>> is_healthy = await check_db_health()
        >>> print(f"Database status: {'OK' if is_healthy else 'Down'}")
    """
    logger.debug("Checking database health")
    
    try:
        async with get_db_connection() as conn:
            result = await conn.fetchval("SELECT 1")
            
            if result == 1:
                logger.debug("Database health check passed")
                return True
            else:
                logger.warning("Database health check returned unexpected result")
                return False
                
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}", exc_info=True)
        return False


async def get_db_stats() -> dict:
    """
    Get database connection pool statistics.
    
    Returns:
        Dictionary with pool statistics
    
    Example:
        >>> stats = await get_db_stats()
        >>> print(f"Active connections: {stats['size']}")
    """
    logger.debug("Getting database pool statistics")
    
    pool = await get_db_pool()
    
    stats = {
        "size": pool.get_size(),
        "free": pool.get_size() - pool.get_idle_size(),
        "idle": pool.get_idle_size(),
        "min_size": pool.get_min_size(),
        "max_size": pool.get_max_size()
    }
    
    logger.debug("Database pool statistics retrieved", extra=stats)
    
    return stats


# Initialize module logger
logger.info("PostgreSQL database module loaded successfully")