"""
Database connection management for Supabase PostgreSQL.
Handles connection pooling and provides async database access.
"""
import asyncpg
from typing import Optional
from contextlib import asynccontextmanager
from supabase import create_client, Client
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseConnection:
    """
    Manages database connections with connection pooling.
    Supports both Supabase client (for auth/storage) and asyncpg (for direct SQL).
    """
    
    def __init__(self):
        """Initialize database connection manager."""
        self._pool: Optional[asyncpg.Pool] = None
        self._supabase_client: Optional[Client] = None
        logger.info("DatabaseConnection initialized")
    
    async def connect(self) -> None:
        """
        Establish database connection pool.
        Called during application startup.
        """
        try:
            logger.info(
                "Creating database connection pool",
                pool_size=settings.DB_POOL_SIZE,
                max_overflow=settings.DB_MAX_OVERFLOW
            )
            
            # Create asyncpg connection pool
            self._pool = await asyncpg.create_pool(
                dsn=settings.DATABASE_URL,
                min_size=settings.DB_POOL_SIZE,
                max_size=settings.DB_POOL_SIZE + settings.DB_MAX_OVERFLOW,
                timeout=settings.DB_POOL_TIMEOUT,
                command_timeout=60,
                statement_cache_size=0,  # Required for pgbouncer compatibility

            )
            
            # Initialize Supabase client (optional - only if key is provided)
            if settings.SUPABASE_KEY and settings.SUPABASE_KEY != "your-supabase-anon-key":
                try:
                    self._supabase_client = create_client(
                        settings.SUPABASE_URL,
                        settings.SUPABASE_KEY
                    )
                    logger.info("Supabase client initialized")
                except Exception as e:
                    logger.warning(
                        "Supabase client initialization failed, continuing without it",
                        error=str(e)
                    )
                    self._supabase_client = None
            else:
                logger.info("Supabase client not initialized (no valid API key)")
                self._supabase_client = None
            
            logger.info("Database connection pool created successfully")
            
        except Exception as e:
            logger.error(
                "Failed to create database connection pool",
                error=str(e),
                exc_info=True
            )
            raise
    
    async def disconnect(self) -> None:
        """
        Close database connection pool.
        Called during application shutdown.
        """
        try:
            if self._pool:
                logger.info("Closing database connection pool")
                await self._pool.close()
                self._pool = None
                logger.info("Database connection pool closed")
        except Exception as e:
            logger.error(
                "Error closing database connection pool",
                error=str(e),
                exc_info=True
            )
    
    @asynccontextmanager
    async def acquire(self):
        """
        Acquire a database connection from the pool.
        
        Usage:
            async with db.acquire() as conn:
                result = await conn.fetch("SELECT * FROM papers")
        
        Yields:
            asyncpg.Connection: Database connection
        """
        if not self._pool:
            logger.error("Connection pool not initialized")
            raise RuntimeError("Database connection pool not initialized")
        
        async with self._pool.acquire() as connection:
            logger.debug("Database connection acquired from pool")
            try:
                yield connection
            finally:
                logger.debug("Database connection returned to pool")
    
    async def execute(
        self,
        query: str,
        *args,
        timeout: Optional[float] = None
    ) -> str:
        """
        Execute a query without returning results.
        
        Args:
            query: SQL query to execute
            *args: Query parameters
            timeout: Query timeout in seconds
            
        Returns:
            str: Query execution status
        """
        logger.debug(
            "Executing query",
            query=query[:100],  # Log first 100 chars
            params_count=len(args)
        )
        
        async with self.acquire() as conn:
            try:
                result = await conn.execute(query, *args, timeout=timeout)
                logger.debug("Query executed successfully", result=result)
                return result
            except Exception as e:
                logger.error(
                    "Query execution failed",
                    query=query[:100],
                    error=str(e),
                    exc_info=True
                )
                raise
    
    async def fetch(
        self,
        query: str,
        *args,
        timeout: Optional[float] = None
    ) -> list[asyncpg.Record]:
        """
        Fetch multiple rows from database.
        
        Args:
            query: SQL query
            *args: Query parameters
            timeout: Query timeout in seconds
            
        Returns:
            list[Record]: Query results
        """
        logger.debug(
            "Fetching rows",
            query=query[:100],
            params_count=len(args)
        )
        
        async with self.acquire() as conn:
            try:
                results = await conn.fetch(query, *args, timeout=timeout)
                logger.debug("Rows fetched successfully", row_count=len(results))
                return results
            except Exception as e:
                logger.error(
                    "Fetch query failed",
                    query=query[:100],
                    error=str(e),
                    exc_info=True
                )
                raise
    
    async def fetchrow(
        self,
        query: str,
        *args,
        timeout: Optional[float] = None
    ) -> Optional[asyncpg.Record]:
        """
        Fetch single row from database.
        
        Args:
            query: SQL query
            *args: Query parameters
            timeout: Query timeout in seconds
            
        Returns:
            Optional[Record]: Single row or None
        """
        logger.debug(
            "Fetching single row",
            query=query[:100],
            params_count=len(args)
        )
        
        async with self.acquire() as conn:
            try:
                result = await conn.fetchrow(query, *args, timeout=timeout)
                logger.debug(
                    "Row fetch complete",
                    found=result is not None
                )
                return result
            except Exception as e:
                logger.error(
                    "Fetchrow query failed",
                    query=query[:100],
                    error=str(e),
                    exc_info=True
                )
                raise
    
    async def fetchval(
        self,
        query: str,
        *args,
        column: int = 0,
        timeout: Optional[float] = None
    ):
        """
        Fetch single value from database.
        
        Args:
            query: SQL query
            *args: Query parameters
            column: Column index to return
            timeout: Query timeout in seconds
            
        Returns:
            Any: Single value
        """
        logger.debug(
            "Fetching single value",
            query=query[:100],
            column=column
        )
        
        async with self.acquire() as conn:
            try:
                result = await conn.fetchval(
                    query, *args, column=column, timeout=timeout
                )
                logger.debug("Value fetched successfully")
                return result
            except Exception as e:
                logger.error(
                    "Fetchval query failed",
                    query=query[:100],
                    error=str(e),
                    exc_info=True
                )
                raise
    
    @property
    def supabase(self) -> Client:
        """
        Get Supabase client for auth/storage operations.
        
        Returns:
            Client: Supabase client instance
            
        Raises:
            RuntimeError: If Supabase client is not initialized
        """
        if not self._supabase_client:
            logger.warning("Supabase client not available")
            raise RuntimeError(
                "Supabase client not initialized. "
                "Set SUPABASE_KEY in .env to enable Supabase features."
            )
        return self._supabase_client
    
    async def health_check(self) -> bool:
        """
        Check database connection health.
        
        Returns:
            bool: True if database is accessible
        """
        try:
            logger.debug("Performing database health check")
            result = await self.fetchval("SELECT 1")
            is_healthy = result == 1
            logger.info("Database health check complete", healthy=is_healthy)
            return is_healthy
        except Exception as e:
            logger.error(
                "Database health check failed",
                error=str(e),
                exc_info=True
            )
            return False


# Global database connection instance
db = DatabaseConnection()


async def get_db() -> DatabaseConnection:
    """
    Dependency function for FastAPI endpoints.
    
    Returns:
        DatabaseConnection: Database connection instance
    """
    return db