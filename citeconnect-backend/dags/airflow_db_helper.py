"""
Simplified database helper for Airflow DAGs
Minimal version - just connection
"""
import asyncpg
from typing import Optional
import os
import asyncio


class SimpleDB:
    """Minimal database connection for Airflow tasks"""
    
    def __init__(self):
        self._pool: Optional[asyncpg.Pool] = None
        self.database_url = os.getenv('DATABASE_URL')
    
    async def connect(self):
        """Create connection pool"""
        if not self.database_url:
            raise ValueError("DATABASE_URL not set")
        
        # Disable statement cache for pgbouncer compatibility
        # pgbouncer doesn't support prepared statements properly
        self._pool = await asyncpg.create_pool(
            dsn=self.database_url,
            min_size=1,
            max_size=5,
            timeout=30.0,
            command_timeout=60,
            statement_cache_size=0,  # Disable prepared statement cache for pgbouncer
        )
    
    async def disconnect(self):
        """Close pool with timeout"""
        if self._pool:
            try:
                await asyncio.wait_for(self._pool.close(), timeout=3.0)
            except asyncio.TimeoutError:
                self._pool.terminate()
            finally:
            self._pool = None
    
    async def fetchval(self, query: str, *args, timeout: float = 10.0):
        """Fetch single value"""
        if not self._pool:
            raise RuntimeError("Not connected")
        
        async with self._pool.acquire() as conn:
            return await conn.fetchval(query, *args, timeout=timeout)
    
    async def fetchrow(self, query: str, *args, timeout: float = 10.0):
        """Fetch single row"""
        if not self._pool:
            raise RuntimeError("Not connected")
        
        async with self._pool.acquire() as conn:
            return await conn.fetchrow(query, *args, timeout=timeout)
    
        