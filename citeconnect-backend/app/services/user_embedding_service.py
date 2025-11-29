"""
User embedding generation service.
Handles dual embedding models (MiniLM and SPECTER) for user profiles.
Supports profile-based (cold start) and interaction-based (warm start) generation.
"""
from typing import Optional, List, Dict, Tuple
import numpy as np
from datetime import datetime

from app.services.bootstrap.embedding_service import get_embedding_service
from app.db.connection import DatabaseConnection
from app.db.repositories.user_repo import UserRepository
from app.utils.logger import get_logger

logger = get_logger(__name__)


class UserEmbeddingService:
    """
    Service for generating and managing user embeddings.
    
    Supports:
    - Dual models (MiniLM 384-dim, SPECTER 768-dim)
    - Profile-based generation (cold start)
    - Interaction-based generation (warm start)
    - Hybrid generation (mature users)
    """
    
    # Interaction thresholds for stage transitions
    EARLY_STAGE_THRESHOLD = 10      # cold_start → early
    MATURE_STAGE_THRESHOLD = 50     # early → mature
    EXPERT_STAGE_THRESHOLD = 200    # mature → expert
    
    # Update frequency
    UPDATE_EVERY_N_INTERACTIONS = 10
    
    def __init__(self, db: DatabaseConnection):
        """
        Initialize user embedding service.
        
        Args:
            db: Database connection
        """
        self.db = db
        self.embedding_service = get_embedding_service()
        self.user_repo = UserRepository(db)
        
        logger.info("UserEmbeddingService initialized")
    
    async def get_or_generate_user_embeddings(
        self,
        user_id: int
    ) -> Dict[str, np.ndarray]:
        """
        Get existing embeddings or generate new ones for both models.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with 'minilm' and 'specter' embeddings
        """
        logger.info(
            "Getting user embeddings for both models",
            user_id=user_id
        )
        
        # Check if embeddings exist
        minilm_emb = await self._get_existing_embedding(user_id, 'minilm')
        specter_emb = await self._get_existing_embedding(user_id, 'specter')
        
        # Check if regeneration needed
        should_regenerate = await self._should_regenerate_embeddings(user_id)
        
        if minilm_emb is None or specter_emb is None or should_regenerate:
            logger.info(
                "Generating/updating embeddings",
                user_id=user_id,
                minilm_exists=minilm_emb is not None,
                specter_exists=specter_emb is not None,
                should_regenerate=should_regenerate
            )
            
            # Generate both embeddings
            minilm_emb, specter_emb = await self.generate_user_embeddings(user_id)
        else:
            logger.debug(
                "Using existing embeddings",
                user_id=user_id
            )
        
        return {
            'minilm': minilm_emb,
            'specter': specter_emb
        }
    
    async def generate_user_embeddings(
        self,
        user_id: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate embeddings for both models based on user's current stage.
        
        Args:
            user_id: User identifier
            
        Returns:
            Tuple of (minilm_embedding, specter_embedding)
        """
        logger.info(
            "Generating user embeddings",
            user_id=user_id
        )
        
        # Get user's recommendation state
        state = await self.user_repo.get_recommendation_state(user_id)
        
        if not state:
            raise ValueError(f"No recommendation state found for user {user_id}")
        
        interaction_count = state['interaction_count']
        
        # Determine generation method based on interaction count
        if interaction_count < self.EARLY_STAGE_THRESHOLD:
            # COLD START: Use profile only
            method = 'profile_based'
            minilm_emb, specter_emb = await self._generate_from_profile(user_id)
            based_on_papers = None
            
        elif interaction_count < self.MATURE_STAGE_THRESHOLD:
            # EARLY: Use interactions only
            method = 'interaction_based'
            minilm_emb, specter_emb = await self._generate_from_interactions(user_id)
            
            # Get papers used
            interactions = await self._get_positive_interactions(user_id, limit=20)
            based_on_papers = [i['paper_id'] for i in interactions]
            
        else:
            # MATURE/EXPERT: Hybrid approach
            method = 'hybrid'
            minilm_emb, specter_emb = await self._generate_hybrid(user_id)
            
            # Get papers used
            interactions = await self._get_positive_interactions(user_id, limit=50)
            based_on_papers = [i['paper_id'] for i in interactions]
        
        # Store both embeddings
        await self._store_embedding(
            user_id=user_id,
            model='minilm',
            embedding=minilm_emb,
            generation_method=method,
            based_on_papers=based_on_papers,
            interaction_count=interaction_count
        )
        
        await self._store_embedding(
            user_id=user_id,
            model='specter',
            embedding=specter_emb,
            generation_method=method,
            based_on_papers=based_on_papers,
            interaction_count=interaction_count
        )
        
        # Update recommendation state timestamps
        await self._update_embedding_timestamps(user_id)
        
        # Check for stage transition
        await self._check_stage_transition(user_id, interaction_count)
        
        logger.info(
            "User embeddings generated successfully",
            user_id=user_id,
            method=method,
            interaction_count=interaction_count,
            minilm_dim=len(minilm_emb),
            specter_dim=len(specter_emb)
        )
        
        return minilm_emb, specter_emb
    
    async def _generate_from_profile(
        self,
        user_id: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate embeddings from user profile and interests (COLD START).
        
        Args:
            user_id: User identifier
            
        Returns:
            Tuple of (minilm_embedding, specter_embedding)
        """
        logger.debug(
            "Generating embeddings from profile",
            user_id=user_id
        )
        
        # Get user profile
        profile = await self.user_repo.get_profile(user_id)
        
        if not profile:
            raise ValueError(f"No profile found for user {user_id}")
        
        # Get interests
        interests = await self.user_repo.get_user_interests(user_id)
        
        if not interests:
            raise ValueError(f"No interests found for user {user_id}")
        
        # Build text representation
        text = self._build_profile_text(profile, interests)
        
        logger.debug(
            "Profile text generated",
            user_id=user_id,
            text_preview=text[:100],
            text_length=len(text)
        )
        
        # Generate embeddings with both models
        minilm_emb = self.embedding_service.encode_text(text, model='minilm')
        specter_emb = self.embedding_service.encode_text(text, model='specter')
        
        return minilm_emb, specter_emb
    
    async def _generate_from_interactions(
        self,
        user_id: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate embeddings from user's interaction history (EARLY/WARM START).
        
        Args:
            user_id: User identifier
            
        Returns:
            Tuple of (minilm_embedding, specter_embedding)
        """
        logger.debug(
            "Generating embeddings from interactions",
            user_id=user_id
        )
        
        # Get positive interactions (saved, liked, long views)
        interactions = await self._get_positive_interactions(user_id, limit=50)
        
        if not interactions:
            logger.warning(
                "No positive interactions found, falling back to profile",
                user_id=user_id
            )
            return await self._generate_from_profile(user_id)
        
        paper_ids = [i['paper_id'] for i in interactions]
        
        logger.debug(
            "Retrieved interactions",
            user_id=user_id,
            paper_count=len(paper_ids)
        )
        
        # Get paper embeddings
        minilm_embeddings = await self._get_paper_embeddings(paper_ids, 'minilm')
        specter_embeddings = await self._get_paper_embeddings(paper_ids, 'specter')
        
        if not minilm_embeddings or not specter_embeddings:
            logger.warning(
                "Could not retrieve paper embeddings, falling back to profile",
                user_id=user_id
            )
            return await self._generate_from_profile(user_id)
        
        # Calculate weighted average based on interaction strength
        weights = [i['interaction_strength'] for i in interactions]
        
        minilm_emb = np.average(minilm_embeddings, axis=0, weights=weights)
        specter_emb = np.average(specter_embeddings, axis=0, weights=weights)
        
        logger.debug(
            "Embeddings averaged from interactions",
            user_id=user_id,
            papers_used=len(paper_ids),
            avg_weight=np.mean(weights)
        )
        
        return minilm_emb, specter_emb
    
    async def _generate_hybrid(
        self,
        user_id: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate hybrid embeddings (profile + interactions) for MATURE users.
        
        Args:
            user_id: User identifier
            
        Returns:
            Tuple of (minilm_embedding, specter_embedding)
        """
        logger.debug(
            "Generating hybrid embeddings",
            user_id=user_id
        )
        
        # Get profile-based embeddings
        profile_minilm, profile_specter = await self._generate_from_profile(user_id)
        
        # Get interaction-based embeddings
        interaction_minilm, interaction_specter = await self._generate_from_interactions(user_id)
        
        # Weighted combination (70% interactions, 30% profile)
        interaction_weight = 0.7
        profile_weight = 0.3
        
        minilm_emb = (
            interaction_weight * interaction_minilm +
            profile_weight * profile_minilm
        )
        
        specter_emb = (
            interaction_weight * interaction_specter +
            profile_weight * profile_specter
        )
        
        logger.debug(
            "Hybrid embeddings generated",
            user_id=user_id,
            interaction_weight=interaction_weight,
            profile_weight=profile_weight
        )
        
        return minilm_emb, specter_emb
    
    def _build_profile_text(
        self,
        profile: Dict,
        interests: List[Dict]
    ) -> str:
        """
        Build text representation from profile and interests.
        
        Args:
            profile: User profile dict
            interests: List of interest dicts
            
        Returns:
            Text representation
        """
        text_parts = []
        
        # Add domain
        text_parts.append(profile['primary_domain'])
        
        # Add research stage context
        if profile.get('research_stage'):
            stage_context = {
                'undergraduate': 'learning fundamentals',
                'masters': 'graduate research',
                'phd': 'doctoral research',
                'postdoc': 'postdoctoral research',
                'professor': 'academic research teaching',
                'industry': 'industry application development',
                'independent': 'independent research'
            }
            context = stage_context.get(
                profile['research_stage'],
                profile['research_stage']
            )
            text_parts.append(context)
        
        # Add sub-domains
        if profile.get('sub_domains'):
            text_parts.extend(profile['sub_domains'])
        
        # Add interests (sorted by level and confidence)
        sorted_interests = sorted(
            interests,
            key=lambda x: (x['interest_level'], -x['confidence_score'])
        )
        
        for interest in sorted_interests:
            # Repeat high-confidence interests for emphasis
            if interest['confidence_score'] >= 0.8:
                text_parts.append(interest['interest_term'])
                text_parts.append(interest['interest_term'])  # Twice for weight
            else:
                text_parts.append(interest['interest_term'])
        
        # Combine
        text = ' '.join(text_parts)
        
        return text
    
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
    
    async def _get_paper_embeddings(
        self,
        paper_ids: List[str],
        model: str
    ) -> Optional[List[np.ndarray]]:
        """
        Get paper embeddings from database.
        
        Args:
            paper_ids: List of paper identifiers
            model: Model name ('minilm' or 'specter')
            
        Returns:
            List of embedding vectors or None
        """
        table = f'paper_embeddings_{model}'
        
        query = f"""
            SELECT embedding
            FROM {table}
            WHERE paper_id = ANY($1::text[])
            ORDER BY array_position($1::text[], paper_id)
        """
        
        results = await self.db.fetch(query, paper_ids)
        
        if not results:
            return None
        
        embeddings = [np.array(r['embedding']) for r in results]
        
        return embeddings
    
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
        state = await self.user_repo.get_recommendation_state(user_id)
        
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
        
        if interaction_count >= stored_count + self.UPDATE_EVERY_N_INTERACTIONS:
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
        if interaction_count >= self.EXPERT_STAGE_THRESHOLD:
            new_stage = 'expert'
        elif interaction_count >= self.MATURE_STAGE_THRESHOLD:
            new_stage = 'mature'
        elif interaction_count >= self.EARLY_STAGE_THRESHOLD:
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