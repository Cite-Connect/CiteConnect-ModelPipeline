"""
Generate embeddings for all users in the database.
Run this after users have created profiles.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.utils.logger import setup_logging, get_logger
from app.db.connection import db
from app.services.user_embedding_service import UserEmbeddingService

setup_logging()
logger = get_logger(__name__)


async def generate_embeddings_for_all_users():
    """Generate embeddings for all users with profiles."""
    
    logger.info("="*60)
    logger.info("Starting bulk user embedding generation")
    logger.info("="*60)
    
    await db.connect()
    
    try:
        # Get all users with profiles
        query = """
            SELECT DISTINCT u.user_id, u.email, p.primary_domain
            FROM users u
            JOIN user_profiles_extended p ON u.user_id = p.user_id
            WHERE u.is_active = true
            ORDER BY u.user_id
        """
        
        users = await db.fetch(query)
        
        if not users:
            logger.warning("No users with profiles found")
            return
        
        logger.info(
            "Found users with profiles",
            count=len(users)
        )
        
        # Initialize service
        embedding_service = UserEmbeddingService(db)
        
        # Process each user
        success_count = 0
        error_count = 0
        
        for i, user in enumerate(users, 1):
            user_id = user['user_id']
            email = user['email']
            domain = user['primary_domain']
            
            logger.info(
                f"Processing user {i}/{len(users)}",
                user_id=user_id,
                email=email,
                domain=domain
            )
            
            try:
                # Generate embeddings for both models
                embeddings = await embedding_service.get_or_generate_user_embeddings(user_id)
                
                logger.info(
                    "Embeddings generated",
                    user_id=user_id,
                    minilm_shape=embeddings['minilm'].shape,
                    specter_shape=embeddings['specter'].shape
                )
                
                success_count += 1
                
            except Exception as e:
                logger.error(
                    "Embedding generation failed for user",
                    user_id=user_id,
                    email=email,
                    error=str(e),
                    exc_info=True
                )
                error_count += 1
        
        # Summary
        logger.info("="*60)
        logger.info("Bulk embedding generation complete")
        logger.info(f"Total users: {len(users)}")
        logger.info(f"Success: {success_count}")
        logger.info(f"Errors: {error_count}")
        logger.info("="*60)
        
        # Verify results
        minilm_count = await db.fetchval(
            "SELECT COUNT(*) FROM user_embeddings_minilm"
        )
        specter_count = await db.fetchval(
            "SELECT COUNT(*) FROM user_embeddings_specter"
        )
        
        logger.info(
            "Database verification",
            minilm_embeddings=minilm_count,
            specter_embeddings=specter_count
        )
        
    finally:
        await db.disconnect()


async def main():
    """Main entry point."""
    try:
        await generate_embeddings_for_all_users()
        sys.exit(0)
    except Exception as e:
        logger.error(
            "Bulk generation failed",
            error=str(e),
            exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())