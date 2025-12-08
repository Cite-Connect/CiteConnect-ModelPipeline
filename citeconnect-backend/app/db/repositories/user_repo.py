"""
User repository for managing user accounts and profiles.
Handles user data, profiles, and state management.
UPDATED: Interests stored in separate user_interest_hierarchy table.
"""
from typing import Optional, Dict, Any, List
import asyncpg
from app.db.repositories.base import BaseRepository
from app.db.connection import DatabaseConnection
from app.utils.logger import get_logger
import numpy as np
from app.config import settings

logger = get_logger(__name__)


class UserRepository(BaseRepository):
    """Repository for user-related operations."""
    
    @property
    def table_name(self) -> str:
        return "users"
    
    def __init__(self, db: DatabaseConnection):
        super().__init__(db)
        logger.info("UserRepository initialized")
    
    async def find_by_email(self, email: str) -> Optional[asyncpg.Record]:
        """
        Find user by email address.
        
        Args:
            email: User email
            
        Returns:
            Optional[Record]: User record or None
        """
        logger.info("Finding user by email", email=email)
        
        query = """
            SELECT * FROM users
            WHERE email = $1
        """
        
        try:
            logger.debug("Executing email lookup query", email=email)
            result = await self.db.fetchrow(query, email)
            logger.debug(
                "User email lookup complete",
                email=email,
                found=result is not None
            )
            return result
        except Exception as e:
            logger.error(
                "User email lookup failed",
                email=email,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def get_profile(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Get user's extended profile with interests from hierarchy table.
        
        Args:
            user_id: User identifier
            
        Returns:
            Optional[Dict]: Profile with interests included
        """
        logger.debug("Getting user profile", user_id=user_id)
        
        query = """
            SELECT * FROM user_profiles_extended
            WHERE user_id = $1
        """
        
        try:
            result = await self.db.fetchrow(query, user_id)
            
            if not result:
                logger.debug(
                    "Profile not found",
                    user_id=user_id
                )
                return None
            
            # Convert to dict
            profile = dict(result)
            
            # Get interests from hierarchy table
            interests = await self.get_user_interests(user_id)
            
            # Group interests by level
            profile['interests'] = {
                'level_1': [i['interest_term'] for i in interests if i['interest_level'] == 1],
                'level_2': [i['interest_term'] for i in interests if i['interest_level'] == 2],
                'level_3': [i['interest_term'] for i in interests if i['interest_level'] == 3],
                'all': [i['interest_term'] for i in interests]
            }
            
            logger.debug(
                "Profile retrieved",
                user_id=user_id,
                found=True,
                interest_count=len(interests)
            )
            return profile
            
        except Exception as e:
            logger.error(
                "Profile retrieval failed",
                user_id=user_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def create_profile(
        self,
        user_id: int,
        profile_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create extended user profile.
        NOTE: Interests are saved to user_interest_hierarchy table separately.
        
        Args:
            user_id: User identifier
            profile_data: Profile fields (may include 'interests' key)
            
        Returns:
            Dict: Created profile with interests
        """
        logger.info("Creating user profile", user_id=user_id)
        
        # Extract interests (will be saved separately)
        interests = profile_data.pop('interests', None)
        
        # Insert into user_profiles_extended (NO interests field)
        query = """
            INSERT INTO user_profiles_extended (
                user_id, research_stage, primary_domain, 
                sub_domains, research_methods, research_goals, 
                reading_level, time_availability,
                years_experience, h_index,
                prefers_recent_papers, prefers_high_impact, prefers_open_access,
                preferred_venues, institution, department,
                looking_for_collaborators, google_scholar_url, semantic_scholar_author_id
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
            RETURNING *
        """
        
        try:
            result = await self.db.fetchrow(
                query,
                user_id,
                profile_data.get('research_stage'),
                profile_data.get('primary_domain'),
                profile_data.get('sub_domains', []),
                profile_data.get('research_methods', []),
                profile_data.get('research_goals', []),
                profile_data.get('reading_level', 'intermediate'),
                profile_data.get('time_availability'),
                profile_data.get('years_experience'),
                profile_data.get('h_index'),
                profile_data.get('prefers_recent_papers', True),
                profile_data.get('prefers_high_impact', False),
                profile_data.get('prefers_open_access', True),
                profile_data.get('preferred_venues', []),
                profile_data.get('institution'),
                profile_data.get('department'),
                profile_data.get('looking_for_collaborators', False),
                profile_data.get('google_scholar_url'),
                profile_data.get('semantic_scholar_author_id')
            )
            
            profile = dict(result)
            
            # Save interests to user_interest_hierarchy table
            if interests and len(interests) > 0:
                for interest in interests:
                    await self.add_interest(
                        user_id=user_id,
                        interest=interest,
                        level=1,  # Default to broad level
                        confidence=1.0,
                        source='explicit'
                    )
                
                logger.debug(
                    "Interests saved to hierarchy table",
                    user_id=user_id,
                    count=len(interests)
                )
                
                # Add interests to return value
                profile['interests'] = {
                    'level_1': interests,
                    'level_2': [],
                    'level_3': [],
                    'all': interests
                }
            else:
                profile['interests'] = {
                    'level_1': [],
                    'level_2': [],
                    'level_3': [],
                    'all': []
                }
            
            logger.info(
                "Profile created successfully",
                user_id=user_id,
                completeness=profile.get('profile_completeness')
            )
            
            return profile
            
        except Exception as e:
            logger.error(
                "Profile creation failed",
                user_id=user_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def update_profile(
        self,
        user_id: int,
        updates: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Update user profile.
        NOTE: 'interests' handled separately in user_interest_hierarchy table.
        
        Args:
            user_id: User identifier
            updates: Fields to update (may include 'interests')
            
        Returns:
            Optional[Dict]: Updated profile with interests
        """
        logger.info(
            "Updating user profile",
            user_id=user_id,
            fields=list(updates.keys())
        )
        
        # Extract interests if provided (will be updated separately)
        interests = updates.pop('interests', None)
        
        if not updates and not interests:
            logger.warning("No fields to update", user_id=user_id)
            return await self.get_profile(user_id)
        
        # Build update query for profile fields (excluding interests)
        result = None
        if updates:
            set_parts = []
            values = [user_id]
            param_idx = 2
            
            for key, value in updates.items():
                set_parts.append(f"{key} = ${param_idx}")
                values.append(value)
                param_idx += 1
            
            query = f"""
                UPDATE user_profiles_extended
                SET {', '.join(set_parts)}, updated_at = NOW()
                WHERE user_id = $1
                RETURNING *
            """
            
            try:
                result = await self.db.fetchrow(query, *values)
                
                if not result:
                    logger.warning("Profile not found for update", user_id=user_id)
                    return None
                
                logger.info(
                    "Profile updated successfully",
                    user_id=user_id,
                    completeness=result.get('profile_completeness')
                )
            except Exception as e:
                logger.error(
                    "Profile update failed",
                    user_id=user_id,
                    error=str(e),
                    exc_info=True
                )
                raise
        
        # Update interests if provided
        if interests:
            # Clear old explicit interests and add new ones
            delete_query = """
                DELETE FROM user_interest_hierarchy
                WHERE user_id = $1 AND source = 'explicit'
            """
            await self.db.execute(delete_query, user_id)
            
            for interest in interests:
                await self.add_interest(
                    user_id=user_id,
                    interest=interest,
                    level=1,
                    confidence=1.0,
                    source='explicit'
                )
            
            logger.info(
                "Interests updated in hierarchy table",
                user_id=user_id,
                count=len(interests)
            )
        
        # Get updated profile with interests
        return await self.get_profile(user_id)
    
    async def get_user_interests(self, user_id: int) -> List[asyncpg.Record]:
        """
        Get user's interest hierarchy from user_interest_hierarchy table.
        
        Args:
            user_id: User identifier
            
        Returns:
            List[Record]: Interest records with levels
        """
        logger.debug("Getting user interests", user_id=user_id)
        
        query = """
            SELECT *
            FROM user_interest_hierarchy
            WHERE user_id = $1
            ORDER BY interest_level, confidence_score DESC
        """
        
        try:
            results = await self.db.fetch(query, user_id)
            logger.debug(
                "Interests retrieved",
                user_id=user_id,
                count=len(results)
            )
            return results
        except Exception as e:
            logger.error(
                "Interests retrieval failed",
                user_id=user_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def add_interest(
        self,
        user_id: int,
        interest: str,
        level: int = 1,
        confidence: float = 1.0,
        source: str = "explicit"
    ) -> asyncpg.Record:
        """
        Add interest to user's hierarchy in user_interest_hierarchy table.
        
        Args:
            user_id: User identifier
            interest: Interest term (max 100 chars)
            level: Interest level (1=broad, 2=specific, 3=narrow)
            confidence: Confidence score (0.0-1.0)
            source: 'explicit', 'inferred', or 'imported'
            
        Returns:
            Record: Created interest record
        """
        logger.debug(
            "Adding user interest",
            user_id=user_id,
            interest=interest,
            level=level
        )
        
        query = """
            INSERT INTO user_interest_hierarchy (
                user_id, interest_term, interest_level,
                confidence_score, source
            )
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (user_id, interest_level, interest_term)
            DO UPDATE SET
                confidence_score = EXCLUDED.confidence_score,
                source = EXCLUDED.source
            RETURNING *
        """
        
        try:
            result = await self.db.fetchrow(
                query,
                user_id,
                interest,
                level,
                confidence,
                source
            )
            logger.info(
                "Interest added to hierarchy",
                user_id=user_id,
                interest=interest,
                level=level
            )
            return result
        except Exception as e:
            logger.error(
                "Interest addition failed",
                user_id=user_id,
                interest=interest,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def get_recommendation_state(
        self,
        user_id: int
    ) -> Optional[asyncpg.Record]:
        """
        Get user's recommendation state.
        
        Args:
            user_id: User identifier
            
        Returns:
            Optional[Record]: State record or None
        """
        logger.debug("Getting recommendation state", user_id=user_id)
        
        query = """
            SELECT * FROM user_recommendation_state
            WHERE user_id = $1
        """
        
        try:
            result = await self.db.fetchrow(query, user_id)
            logger.debug(
                "State retrieved",
                user_id=user_id,
                found=result is not None
            )
            return result
        except Exception as e:
            logger.error(
                "State retrieval failed",
                user_id=user_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def initialize_recommendation_state(
        self,
        user_id: int,
        initial_stage: str = "cold_start"
    ) -> asyncpg.Record:
        """
        Initialize recommendation state for new user.
        
        Args:
            user_id: User identifier
            initial_stage: Starting stage
            
        Returns:
            Record: Created state record
        """
        logger.info(
            "Initializing recommendation state",
            user_id=user_id,
            stage=initial_stage
        )
        
        query = """
            INSERT INTO user_recommendation_state (
                user_id, recommendation_stage
            )
            VALUES ($1, $2)
            RETURNING *
        """
        
        try:
            result = await self.db.fetchrow(query, user_id, initial_stage)
            logger.info(
                "State initialized",
                user_id=user_id,
                stage=initial_stage
            )
            return result
        except Exception as e:
            logger.error(
                "State initialization failed",
                user_id=user_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def update_recommendation_state(
        self,
        user_id: int,
        updates: Dict[str, Any]
    ) -> Optional[asyncpg.Record]:
        """
        Update user's recommendation state.
        
        Args:
            user_id: User identifier
            updates: State fields to update
            
        Returns:
            Optional[Record]: Updated state
        """
        logger.debug(
            "Updating recommendation state",
            user_id=user_id,
            fields=list(updates.keys())
        )
        
        set_parts = []
        values = [user_id]
        param_idx = 2
        
        for key, value in updates.items():
            set_parts.append(f"{key} = ${param_idx}")
            values.append(value)
            param_idx += 1
        
        query = f"""
            UPDATE user_recommendation_state
            SET {', '.join(set_parts)}, updated_at = NOW()
            WHERE user_id = $1
            RETURNING *
        """
        
        try:
            result = await self.db.fetchrow(query, *values)
            logger.info(
                "State updated",
                user_id=user_id,
                updates=updates
            )
            return result
        except Exception as e:
            logger.error(
                "State update failed",
                user_id=user_id,
                error=str(e),
                exc_info=True
            )
            raise

    async def _get_positive_interactions(
        self,
        user_id: int,
        limit: int = 50
    ) -> List[Dict]:
        """
        Get user's positive interactions (saved, liked, long views).
        
        Args:
            user_id: User identifier
            limit: Maximum interactions to retrieve
            
        Returns:
            List of interaction records
        """
        query = """
            SELECT 
                paper_id,
                interaction_type,
                interaction_strength,
                duration_seconds,
                created_at
            FROM user_interactions
            WHERE user_id = $1
              AND interaction_strength > 0
            ORDER BY interaction_strength DESC, created_at DESC
            LIMIT $2
        """
        
        interactions = await self.db.fetch(query, user_id, limit)
        
        return [dict(i) for i in interactions]    
    

    
    
    async def _get_existing_embedding(
        self,
        user_id: int,
        model: str
    ) -> Optional[np.ndarray]:
        """
        Get existing user embedding from database.
        
        Args:
            user_id: User identifier
            model: Model name ('minilm' or 'specter')
            
        Returns:
            Embedding vector or None
        """
        table = f'user_embeddings_{model}'
        
        query = f"""
            SELECT embedding
            FROM {table}
            WHERE user_id = $1
        """
        
        result = await self.db.fetchrow(query, user_id)
        
        if result:
            return np.array(result['embedding'])
        
        return None
    
    async def _store_embedding(
        self,
        user_id: int,
        model: str,
        embedding: np.ndarray,
        generation_method: str,
        based_on_papers: Optional[List[str]],
        interaction_count: int
    ):
        """
        Store or update user embedding in database.
        
        Args:
            user_id: User identifier
            model: Model name ('minilm' or 'specter')
            embedding: Embedding vector
            generation_method: How embedding was generated
            based_on_papers: Papers used (if any)
            interaction_count: Current interaction count
        """
        table = f'user_embeddings_{model}'
        
        # Convert numpy array to PostgreSQL vector string format
        embedding_str = '[' + ','.join(map(str, embedding.tolist())) + ']'
        
        query = f"""
            INSERT INTO {table} (
                user_id,
                embedding,
                generation_method,
                based_on_papers,
                interaction_count,
                created_at,
                last_updated
            )
            VALUES ($1, $2::vector, $3, $4, $5, NOW(), NOW())
            ON CONFLICT (user_id)
            DO UPDATE SET
                embedding = EXCLUDED.embedding,
                generation_method = EXCLUDED.generation_method,
                based_on_papers = EXCLUDED.based_on_papers,
                interaction_count = EXCLUDED.interaction_count,
                last_updated = NOW()
        """
        
        await self.db.execute(
            query,
            user_id,
            embedding_str,
            generation_method,
            based_on_papers,
            interaction_count
        )
        
        logger.debug(
            "Embedding stored",
            user_id=user_id,
            model=model,
            table=table
        )
    
    async def _should_regenerate_embeddings(self, user_id: int) -> bool:
        """
        Check if embeddings should be regenerated.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if regeneration needed
        """
        state = await self.get_recommendation_state(user_id)
        
        if not state:
            return False
        
        interaction_count = state['interaction_count']
        
        minilm = await self.db.fetchrow(
            "SELECT interaction_count FROM user_embeddings_minilm WHERE user_id = $1",
            user_id
        )
        
        if not minilm:
            return True
        
        stored_count = minilm['interaction_count']
        print(f"Interaction count: {interaction_count}, Stored count: {stored_count}"); # Debug print
        if interaction_count >= stored_count + settings.UPDATE_EVERY_N_INTERACTIONS:
            print(f"Regenerating minilm embedding for user {user_id}"); # Debug print
            logger.info(
                "Embedding regeneration needed",
                user_id=user_id,
                current_interactions=interaction_count,
                stored_interactions=stored_count
            )
            return True
        
        return False
    
    async def _update_embedding_timestamps(self, user_id: int):
        """
        Update embedding generation timestamps in recommendation state.
        
        Args:
            user_id: User identifier
        """
        query = """
            UPDATE user_recommendation_state
            SET 
                last_embedding_update_minilm = NOW(),
                last_embedding_update_specter = NOW(),
                updated_at = NOW()
            WHERE user_id = $1
        """
        
        await self.db.execute(query, user_id)
    
    async def _check_stage_transition(
        self,
        user_id: int,
        interaction_count: int
    ):
        """
        Check and update recommendation stage based on interaction count.
        
        Args:
            user_id: User identifier
            interaction_count: Current interaction count
        """
        if interaction_count >= settings.EXPERT_STAGE_THRESHOLD:
            new_stage = 'expert'
        elif interaction_count >= settings.MATURE_STAGE_THRESHOLD:
            new_stage = 'mature'
        elif interaction_count >= settings.EARLY_STAGE_THRESHOLD:
            new_stage = 'early'
        else:
            new_stage = 'cold_start'
        
        current = await self.db.fetchrow(
            "SELECT recommendation_stage FROM user_recommendation_state WHERE user_id = $1",
            user_id
        )
        
        if current and current['recommendation_stage'] != new_stage:
            logger.info(
                "Stage transition detected",
                user_id=user_id,
                old_stage=current['recommendation_stage'],
                new_stage=new_stage,
                interaction_count=interaction_count
            )
            
            await self.db.execute(
                """
                UPDATE user_recommendation_state
                SET recommendation_stage = $1, updated_at = NOW()
                WHERE user_id = $2
                """,
                new_stage,
                user_id
            )