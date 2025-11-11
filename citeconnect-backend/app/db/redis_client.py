# app/db/redis_client.py

"""
Redis Client Module

This module manages Redis connections for caching and session management.

Features:
- Connection pooling with redis-py
- Automatic retry logic with exponential backoff
- Key expiration (TTL) management
- Serialization/deserialization helpers
- Connection health checks
- Comprehensive error handling

Usage:
    from app.db.redis_client import get_redis_client, cache_set, cache_get
    
    # Get Redis client
    redis = await get_redis_client()
    
    # Set value with TTL
    await cache_set("user:123", {"name": "John"}, ttl=3600)
    
    # Get value
    user_data = await cache_get("user:123")
"""

import logging
import json
import asyncio
from typing import Optional, Any, Union
from datetime import timedelta

import redis.asyncio as aioredis
from redis.asyncio import Redis
from redis.exceptions import RedisError, ConnectionError as RedisConnectionError

from app.core.config import get_settings
from app.core.exceptions import CachingError

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Global Redis client
_redis_client: Optional[Redis] = None


async def create_redis_client() -> Redis:
    """
    Create and return a Redis client with connection pool.
    
    Creates an async Redis client with configured connection pooling
    and timeout settings. Implements retry logic for robustness.
    
    Returns:
        Redis: Async Redis client instance
    
    Raises:
        CachingError: If client creation fails after retries
    
    Example:
        >>> redis = await create_redis_client()
        >>> await redis.set("key", "value")
    """
    logger.info(
        "Creating Redis client",
        extra={
            "host": settings.REDIS_HOST,
            "port": settings.REDIS_PORT,
            "db": settings.REDIS_DB,
            "max_connections": settings.REDIS_MAX_CONNECTIONS
        }
    )
    
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"Connection attempt {attempt + 1}/{max_retries}")
            
            # Create Redis client with connection pool
            client = aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                max_connections=settings.REDIS_MAX_CONNECTIONS,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            
            # Test connection
            await client.ping()
            
            logger.info("Redis client created successfully")
            
            return client
            
        except RedisConnectionError as e:
            logger.error(
                f"Redis connection failed (attempt {attempt + 1}/{max_retries}): {str(e)}",
                exc_info=True
            )
            
            if attempt < max_retries - 1:
                # Exponential backoff
                wait_time = retry_delay * (2 ** attempt)
                logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                # Final attempt failed
                raise CachingError(
                    message=f"Failed to connect to Redis after {max_retries} attempts",
                    operation="create_client",
                    details={"error": str(e)}
                )
                
        except Exception as e:
            logger.error(f"Unexpected error creating Redis client: {str(e)}", exc_info=True)
            
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                await asyncio.sleep(wait_time)
            else:
                raise CachingError(
                    message=f"Failed to create Redis client: {str(e)}",
                    operation="create_client"
                )


async def get_redis_client() -> Redis:
    """
    Get or create the global Redis client.
    
    Returns the existing client if available, otherwise creates a new one.
    
    Returns:
        Redis: Async Redis client instance
    
    Raises:
        CachingError: If client creation fails
    
    Example:
        >>> redis = await get_redis_client()
        >>> await redis.set("key", "value", ex=3600)
    """
    global _redis_client
    
    logger.debug("Getting Redis client")
    
    if _redis_client is None:
        logger.info("Redis client not initialized, creating new client")
        _redis_client = await create_redis_client()
    
    return _redis_client


async def close_redis_client() -> None:
    """
    Close the global Redis client.
    
    Gracefully closes the Redis connection. Should be called
    during application shutdown.
    
    Example:
        >>> await close_redis_client()
    """
    global _redis_client
    
    logger.info("Closing Redis client")
    
    if _redis_client is not None:
        try:
            await _redis_client.close()
            logger.info("Redis client closed successfully")
        except Exception as e:
            logger.error(f"Error closing Redis client: {str(e)}", exc_info=True)
        finally:
            _redis_client = None
    else:
        logger.debug("No Redis client to close")


def serialize_value(value: Any) -> str:
    """
    Serialize a Python object to JSON string for Redis storage.
    
    Args:
        value: Python object to serialize
    
    Returns:
        JSON string representation
    
    Raises:
        CachingError: If serialization fails
    
    Example:
        >>> data = {"name": "John", "age": 30}
        >>> serialized = serialize_value(data)
        >>> print(serialized)
        '{"name": "John", "age": 30}'
    """
    logger.debug(f"Serializing value of type: {type(value).__name__}")
    
    try:
        # Handle simple types that don't need JSON encoding
        if isinstance(value, (str, int, float, bool)):
            return str(value)
        
        # Complex types: serialize to JSON
        serialized = json.dumps(value)
        logger.debug("Value serialized successfully")
        return serialized
        
    except (TypeError, ValueError) as e:
        logger.error(f"Failed to serialize value: {str(e)}", exc_info=True)
        raise CachingError(
            message=f"Failed to serialize value: {str(e)}",
            operation="serialize"
        )


def deserialize_value(value: Optional[str]) -> Any:
    """
    Deserialize a JSON string from Redis to Python object.
    
    Args:
        value: JSON string from Redis
    
    Returns:
        Deserialized Python object, or None if value is None
    
    Example:
        >>> serialized = '{"name": "John", "age": 30}'
        >>> data = deserialize_value(serialized)
        >>> print(data["name"])
        John
    """
    if value is None:
        return None
    
    logger.debug("Deserializing value")
    
    try:
        # Try to parse as JSON
        deserialized = json.loads(value)
        logger.debug(f"Value deserialized to type: {type(deserialized).__name__}")
        return deserialized
        
    except json.JSONDecodeError:
        # If not JSON, return as string
        logger.debug("Value is not JSON, returning as string")
        return value


async def cache_set(
    key: str,
    value: Any,
    ttl: Optional[int] = None
) -> bool:
    """
    Set a value in Redis cache with optional TTL.
    
    Args:
        key: Cache key
        value: Value to cache (will be JSON serialized)
        ttl: Time to live in seconds (None for no expiration)
    
    Returns:
        True if successful, False otherwise
    
    Example:
        >>> # Set with 1 hour TTL
        >>> await cache_set("user:123", {"name": "John"}, ttl=3600)
        
        >>> # Set without expiration
        >>> await cache_set("config:app", {"debug": True})
    """
    logger.info(
        f"Setting cache key: {key}",
        extra={"key": key, "ttl": ttl}
    )
    
    try:
        redis = await get_redis_client()
        
        # Serialize value
        serialized_value = serialize_value(value)
        
        # Set value with optional TTL
        if ttl:
            await redis.setex(key, ttl, serialized_value)
            logger.debug(f"Cache key set with TTL: {ttl} seconds")
        else:
            await redis.set(key, serialized_value)
            logger.debug("Cache key set without expiration")
        
        return True
        
    except CachingError:
        # Re-raise our custom exceptions
        raise
        
    except RedisError as e:
        logger.error(f"Redis error setting cache key: {str(e)}", exc_info=True)
        raise CachingError(
            message=f"Failed to set cache key: {str(e)}",
            operation="set",
            details={"key": key}
        )
        
    except Exception as e:
        logger.error(f"Unexpected error setting cache key: {str(e)}", exc_info=True)
        # Don't raise for caching errors - log and continue
        return False


async def cache_get(key: str) -> Any:
    """
    Get a value from Redis cache.
    
    Args:
        key: Cache key
    
    Returns:
        Cached value (deserialized), or None if not found
    
    Example:
        >>> user_data = await cache_get("user:123")
        >>> if user_data:
        ...     print(user_data["name"])
    """
    logger.info(f"Getting cache key: {key}", extra={"key": key})
    
    try:
        redis = await get_redis_client()
        
        # Get value
        value = await redis.get(key)
        
        if value is None:
            logger.debug(f"Cache miss for key: {key}")
            return None
        
        logger.debug(f"Cache hit for key: {key}")
        
        # Deserialize and return
        return deserialize_value(value)
        
    except RedisError as e:
        logger.error(f"Redis error getting cache key: {str(e)}", exc_info=True)
        # Return None instead of raising - cache miss is acceptable
        return None
        
    except Exception as e:
        logger.error(f"Unexpected error getting cache key: {str(e)}", exc_info=True)
        return None


async def cache_delete(key: str) -> bool:
    """
    Delete a key from Redis cache.
    
    Args:
        key: Cache key to delete
    
    Returns:
        True if key was deleted, False if key didn't exist
    
    Example:
        >>> await cache_delete("user:123")
    """
    logger.info(f"Deleting cache key: {key}", extra={"key": key})
    
    try:
        redis = await get_redis_client()
        
        # Delete key
        result = await redis.delete(key)
        
        if result > 0:
            logger.debug(f"Cache key deleted: {key}")
            return True
        else:
            logger.debug(f"Cache key not found: {key}")
            return False
        
    except RedisError as e:
        logger.error(f"Redis error deleting cache key: {str(e)}", exc_info=True)
        return False
        
    except Exception as e:
        logger.error(f"Unexpected error deleting cache key: {str(e)}", exc_info=True)
        return False


async def cache_exists(key: str) -> bool:
    """
    Check if a key exists in Redis cache.
    
    Args:
        key: Cache key
    
    Returns:
        True if key exists, False otherwise
    
    Example:
        >>> if await cache_exists("user:123"):
        ...     print("User cached")
    """
    logger.debug(f"Checking if cache key exists: {key}")
    
    try:
        redis = await get_redis_client()
        
        result = await redis.exists(key)
        
        exists = result > 0
        logger.debug(f"Cache key {'exists' if exists else 'does not exist'}: {key}")
        
        return exists
        
    except RedisError as e:
        logger.error(f"Redis error checking key existence: {str(e)}", exc_info=True)
        return False
        
    except Exception as e:
        logger.error(f"Unexpected error checking key existence: {str(e)}", exc_info=True)
        return False


async def cache_expire(key: str, ttl: int) -> bool:
    """
    Set expiration time for a cache key.
    
    Args:
        key: Cache key
        ttl: Time to live in seconds
    
    Returns:
        True if expiration was set, False otherwise
    
    Example:
        >>> # Set key to expire in 1 hour
        >>> await cache_expire("user:123", 3600)
    """
    logger.info(f"Setting expiration for key: {key}", extra={"key": key, "ttl": ttl})
    
    try:
        redis = await get_redis_client()
        
        result = await redis.expire(key, ttl)
        
        if result:
            logger.debug(f"Expiration set for key: {key}")
            return True
        else:
            logger.warning(f"Failed to set expiration (key may not exist): {key}")
            return False
        
    except RedisError as e:
        logger.error(f"Redis error setting expiration: {str(e)}", exc_info=True)
        return False
        
    except Exception as e:
        logger.error(f"Unexpected error setting expiration: {str(e)}", exc_info=True)
        return False


async def cache_get_ttl(key: str) -> Optional[int]:
    """
    Get remaining TTL for a cache key.
    
    Args:
        key: Cache key
    
    Returns:
        Remaining TTL in seconds, -1 if no expiration, None if key doesn't exist
    
    Example:
        >>> ttl = await cache_get_ttl("user:123")
        >>> print(f"Key expires in {ttl} seconds")
    """
    logger.debug(f"Getting TTL for key: {key}")
    
    try:
        redis = await get_redis_client()
        
        ttl = await redis.ttl(key)
        
        if ttl == -2:
            # Key doesn't exist
            logger.debug(f"Key does not exist: {key}")
            return None
        elif ttl == -1:
            # Key exists but has no expiration
            logger.debug(f"Key has no expiration: {key}")
            return -1
        else:
            logger.debug(f"Key TTL: {ttl} seconds")
            return ttl
        
    except RedisError as e:
        logger.error(f"Redis error getting TTL: {str(e)}", exc_info=True)
        return None
        
    except Exception as e:
        logger.error(f"Unexpected error getting TTL: {str(e)}", exc_info=True)
        return None


async def cache_increment(key: str, amount: int = 1) -> Optional[int]:
    """
    Increment a counter in Redis.
    
    Useful for rate limiting and counters.
    
    Args:
        key: Cache key
        amount: Amount to increment by (default: 1)
    
    Returns:
        New value after increment, or None if operation failed
    
    Example:
        >>> # Increment API request counter
        >>> count = await cache_increment("rate_limit:user:123")
        >>> print(f"Request count: {count}")
    """
    logger.debug(f"Incrementing cache key: {key}", extra={"key": key, "amount": amount})
    
    try:
        redis = await get_redis_client()
        
        new_value = await redis.incrby(key, amount)
        
        logger.debug(f"Key incremented to: {new_value}")
        
        return new_value
        
    except RedisError as e:
        logger.error(f"Redis error incrementing key: {str(e)}", exc_info=True)
        return None
        
    except Exception as e:
        logger.error(f"Unexpected error incrementing key: {str(e)}", exc_info=True)
        return None


async def cache_keys_pattern(pattern: str) -> list:
    """
    Get all keys matching a pattern.
    
    WARNING: This scans all keys and can be slow. Use with caution.
    
    Args:
        pattern: Redis key pattern (e.g., "user:*", "cache:session:*")
    
    Returns:
        List of matching keys
    
    Example:
        >>> # Get all user cache keys
        >>> user_keys = await cache_keys_pattern("user:*")
        >>> print(f"Found {len(user_keys)} user keys")
    """
    logger.info(f"Searching for keys matching pattern: {pattern}")
    
    try:
        redis = await get_redis_client()
        
        keys = await redis.keys(pattern)
        
        logger.debug(f"Found {len(keys)} keys matching pattern")
        
        return keys
        
    except RedisError as e:
        logger.error(f"Redis error searching keys: {str(e)}", exc_info=True)
        return []
        
    except Exception as e:
        logger.error(f"Unexpected error searching keys: {str(e)}", exc_info=True)
        return []


async def cache_delete_pattern(pattern: str) -> int:
    """
    Delete all keys matching a pattern.
    
    WARNING: This can delete many keys at once. Use with caution.
    
    Args:
        pattern: Redis key pattern (e.g., "temp:*")
    
    Returns:
        Number of keys deleted
    
    Example:
        >>> # Delete all temporary cache entries
        >>> deleted = await cache_delete_pattern("temp:*")
        >>> print(f"Deleted {deleted} temporary keys")
    """
    logger.warning(f"Deleting keys matching pattern: {pattern}")
    
    try:
        redis = await get_redis_client()
        
        # Get all matching keys
        keys = await redis.keys(pattern)
        
        if not keys:
            logger.debug("No keys found matching pattern")
            return 0
        
        # Delete all keys
        deleted = await redis.delete(*keys)
        
        logger.info(f"Deleted {deleted} keys matching pattern: {pattern}")
        
        return deleted
        
    except RedisError as e:
        logger.error(f"Redis error deleting keys: {str(e)}", exc_info=True)
        return 0
        
    except Exception as e:
        logger.error(f"Unexpected error deleting keys: {str(e)}", exc_info=True)
        return 0


async def check_redis_health() -> bool:
    """
    Check if Redis connection is healthy.
    
    Performs a ping to verify Redis connectivity and responsiveness.
    
    Returns:
        True if Redis is healthy, False otherwise
    
    Example:
        >>> is_healthy = await check_redis_health()
        >>> print(f"Redis status: {'OK' if is_healthy else 'Down'}")
    """
    logger.debug("Checking Redis health")
    
    try:
        redis = await get_redis_client()
        
        # Ping Redis
        pong = await redis.ping()
        
        if pong:
            logger.debug("Redis health check passed")
            return True
        else:
            logger.warning("Redis health check returned unexpected response")
            return False
        
    except Exception as e:
        logger.error(f"Redis health check failed: {str(e)}", exc_info=True)
        return False


async def get_redis_info() -> dict:
    """
    Get Redis server information.
    
    Returns:
        Dictionary with Redis server stats
    
    Example:
        >>> info = await get_redis_info()
        >>> print(f"Redis version: {info.get('redis_version')}")
    """
    logger.debug("Getting Redis server info")
    
    try:
        redis = await get_redis_client()
        
        info = await redis.info()
        
        logger.debug("Redis info retrieved successfully")
        
        return info
        
    except Exception as e:
        logger.error(f"Failed to get Redis info: {str(e)}", exc_info=True)
        return {}


# Initialize module logger
logger.info("Redis client module loaded successfully")