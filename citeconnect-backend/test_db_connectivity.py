"""
Database Connectivity Test Script

Tests connections to all databases:
- PostgreSQL
- Redis
- Weaviate
- Neo4j

This script attempts to connect to each database and perform basic operations.
"""

import asyncio
import sys
from datetime import datetime


def print_section(title):
    """Print formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


async def test_postgres():
    """Test PostgreSQL connection."""
    print_section("Testing PostgreSQL Connection")
    
    try:
        from app.db.postgres import get_db_pool, check_db_health
        
        print("Attempting to create connection pool...")
        pool = await get_db_pool()
        print("✓ Connection pool created")
        
        print("Testing connection health...")
        is_healthy = await check_db_health()
        
        if is_healthy:
            print("✓ PostgreSQL is healthy and responding")
            
            # Test query execution
            print("Testing query execution...")
            from app.db.postgres import execute_query
            result = await execute_query("SELECT 1 as test", fetch_one=True)
            if result and result['test'] == 1:
                print("✓ Query execution successful")
            
            return True
        else:
            print("✗ PostgreSQL health check failed")
            return False
            
    except Exception as e:
        print(f"✗ PostgreSQL test failed: {str(e)}")
        print(f"   Make sure PostgreSQL is running: docker-compose up -d postgres")
        return False


async def test_redis():
    """Test Redis connection."""
    print_section("Testing Redis Connection")
    
    try:
        from app.db.redis_client import get_redis_client, check_redis_health, cache_set, cache_get
        
        print("Attempting to create Redis client...")
        redis = await get_redis_client()
        print("✓ Redis client created")
        
        print("Testing connection health...")
        is_healthy = await check_redis_health()
        
        if is_healthy:
            print("✓ Redis is healthy and responding")
            
            # Test cache operations
            print("Testing cache operations...")
            test_key = "test:connectivity"
            test_value = {"timestamp": datetime.utcnow().isoformat()}
            
            await cache_set(test_key, test_value, ttl=60)
            print("  ✓ Cache set successful")
            
            retrieved = await cache_get(test_key)
            if retrieved:
                print("  ✓ Cache get successful")
            
            return True
        else:
            print("✗ Redis health check failed")
            return False
            
    except Exception as e:
        print(f"✗ Redis test failed: {str(e)}")
        print(f"   Make sure Redis is running: docker-compose up -d redis")
        return False


def test_weaviate():
    """Test Weaviate connection."""
    print_section("Testing Weaviate Connection")
    
    try:
        from app.db.weaviate_client import get_weaviate_client, check_weaviate_health, create_schema
        
        print("Attempting to create Weaviate client...")
        client = get_weaviate_client()
        print("✓ Weaviate client created")
        
        print("Testing connection health...")
        is_healthy = check_weaviate_health()
        
        if is_healthy:
            print("✓ Weaviate is healthy and responding")
            
            # Test schema creation
            print("Testing schema creation...")
            create_schema()
            print("  ✓ Schema created/verified")
            
            return True
        else:
            print("✗ Weaviate health check failed")
            return False
            
    except Exception as e:
        print(f"✗ Weaviate test failed: {str(e)}")
        print(f"   Make sure Weaviate is running: docker-compose up -d weaviate")
        return False


async def test_neo4j():
    """Test Neo4j connection."""
    print_section("Testing Neo4j Connection")
    
    try:
        from app.db.neo4j_client import get_neo4j_driver, check_neo4j_health, execute_query
        
        print("Attempting to create Neo4j driver...")
        driver = await get_neo4j_driver()
        print("✓ Neo4j driver created")
        
        print("Testing connection health...")
        is_healthy = await check_neo4j_health()
        
        if is_healthy:
            print("✓ Neo4j is healthy and responding")
            
            # Test query execution
            print("Testing query execution...")
            result = await execute_query("RETURN 1 as test")
            if result and result[0]['test'] == 1:
                print("  ✓ Query execution successful")
            
            return True
        else:
            print("✗ Neo4j health check failed")
            return False
            
    except Exception as e:
        print(f"✗ Neo4j test failed: {str(e)}")
        print(f"   Make sure Neo4j is running: docker-compose up -d neo4j")
        return False


def generate_report(results):
    """Generate test report."""
    print_section("Database Connectivity Test Summary")
    
    total = len(results)
    passed = sum(1 for r in results.values() if r)
    failed = total - passed
    
    print(f"Total Databases: {total}")
    print(f"Connected:       {passed} ✓")
    print(f"Failed:          {failed} ✗")
    print(f"Success Rate:    {(passed/total)*100:.1f}%")
    
    print("\n" + "="*60)
    print("Detailed Results:")
    print("="*60 + "\n")
    
    for db_name, result in results.items():
        status = "✓ CONNECTED" if result else "✗ FAILED"
        print(f"{status:15s} - {db_name}")
    
    print("\n" + "="*60)
    
    if passed == total:
        print("\n✓ All databases connected successfully!")
        print("\nYou can now:")
        print("  1. Start the FastAPI application: uvicorn app.main:app --reload")
        print("  2. Visit http://localhost:8000/docs for API documentation")
        print("  3. Check health: http://localhost:8000/api/v1/health")
    elif passed > 0:
        print(f"\n⚠️  {failed} database(s) not connected.")
        print("\nThe application can start but some features will be unavailable.")
        print("Make sure all required databases are running:")
        print("  docker-compose up -d postgres redis weaviate neo4j")
    else:
        print("\n✗ No databases connected!")
        print("\nPlease start the required services:")
        print("  docker-compose up -d")
    
    print("="*60 + "\n")
    
    return passed == total


async def main():
    """Run all database connectivity tests."""
    print("\n" + "="*60)
    print("  CiteConnect Database Connectivity Test")
    print("  " + str(datetime.now()))
    print("="*60)
    
    results = {
        "PostgreSQL": await test_postgres(),
        "Redis": await test_redis(),
        "Weaviate": test_weaviate(),
        "Neo4j": await test_neo4j()
    }
    
    all_connected = generate_report(results)
    
    sys.exit(0 if all_connected else 1)


if __name__ == "__main__":
    asyncio.run(main())
