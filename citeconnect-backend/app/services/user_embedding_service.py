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
from app.db.repositories.paper_repo import PaperRepository
from app.config import settings

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
    
    def __init__(self, db: DatabaseConnection):
        """
        Initialize user embedding service.
        
        Args:
            db: Database connection
        """
        self.db = db
        self.embedding_service = get_embedding_service()
        self.user_repo = UserRepository(db)
        self.paper_repo = PaperRepository(db)
        
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
        minilm_emb = await self.user_repo._get_existing_embedding(user_id, 'minilm')
        specter_emb = await self.user_repo._get_existing_embedding(user_id, 'specter')
        
        # Check if regeneration needed
        should_regenerate = await self.user_repo._should_regenerate_embeddings(user_id)
        
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
        if interaction_count < settings.EARLY_STAGE_THRESHOLD:
            # COLD START: Use profile only
            method = 'profile_based'
            minilm_emb, specter_emb = await self._generate_from_profile(user_id)
            based_on_papers = None
            
        elif interaction_count < settings.MATURE_STAGE_THRESHOLD:
            # EARLY: Use interactions only
            method = 'interaction_based'
            minilm_emb, specter_emb = await self._generate_from_interactions(user_id)
            
            # Get papers used
            interactions = await self.user_repo._get_positive_interactions(user_id, limit=20)
            based_on_papers = [i['paper_id'] for i in interactions]
            
        else:
            # MATURE/EXPERT: Hybrid approach
            method = 'hybrid'
            minilm_emb, specter_emb = await self._generate_hybrid(user_id)
            
            # Get papers used
            interactions = await self.user_repo._get_positive_interactions(user_id, limit=50)
            based_on_papers = [i['paper_id'] for i in interactions]
        
        # Store both embeddings
        await self.user_repo._store_embedding(
            user_id=user_id,
            model='minilm',
            embedding=minilm_emb,
            generation_method=method,
            based_on_papers=based_on_papers,
            interaction_count=interaction_count
        )
        
        await self.user_repo._store_embedding(
            user_id=user_id,
            model='specter',
            embedding=specter_emb,
            generation_method=method,
            based_on_papers=based_on_papers,
            interaction_count=interaction_count
        )
        
        # Update recommendation state timestamps
        await self.user_repo._update_embedding_timestamps(user_id)
        
        # Check for stage transition
        await self.user_repo._check_stage_transition(user_id, interaction_count)
        
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
        interactions = await self.user_repo._get_positive_interactions(user_id, limit=50)
        
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
        
        # Get paper embeddings (returns Dict[paper_id, np.ndarray])
        minilm_embeddings_dict = await self.paper_repo._get_paper_embeddings(paper_ids, 'minilm')
        specter_embeddings_dict = await self.paper_repo._get_paper_embeddings(paper_ids, 'specter')
        
        if not minilm_embeddings_dict or not specter_embeddings_dict:
            logger.warning(
                "Could not retrieve paper embeddings, falling back to profile",
                user_id=user_id
            )
            return await self._generate_from_profile(user_id)
        
        # ✅ FIX: Build aligned arrays from dictionaries
        # Only include papers that have embeddings in BOTH models
        minilm_embeddings_list = []
        specter_embeddings_list = []
        valid_weights = []
        
        for interaction in interactions:
            paper_id = interaction['paper_id']
            
            # Check if this paper has embeddings in both models
            if paper_id in minilm_embeddings_dict and paper_id in specter_embeddings_dict:
                minilm_embeddings_list.append(minilm_embeddings_dict[paper_id])
                specter_embeddings_list.append(specter_embeddings_dict[paper_id])
                valid_weights.append(float(interaction['interaction_strength']))
        
        # Check if we have any valid embeddings
        if not minilm_embeddings_list:
            logger.warning(
                "No valid embeddings found for interacted papers, falling back to profile",
                user_id=user_id,
                total_interactions=len(interactions),
                papers_with_embeddings=0
            )
            return await self._generate_from_profile(user_id)
        
        # Convert to numpy arrays with proper shape
        minilm_embeddings = np.array(minilm_embeddings_list, dtype=np.float64)
        specter_embeddings = np.array(specter_embeddings_list, dtype=np.float64)
        weights = np.array(valid_weights, dtype=np.float64)  # Shape: (n_papers,)
        
        logger.debug(
            "Embedding arrays prepared",
            user_id=user_id,
            papers_used=len(weights),
            minilm_shape=minilm_embeddings.shape,
            specter_shape=specter_embeddings.shape,
            weights_shape=weights.shape
        )
        
        # Calculate weighted average based on interaction strength
        minilm_emb = np.average(minilm_embeddings, axis=0, weights=weights)
        specter_emb = np.average(specter_embeddings, axis=0, weights=weights)
        
        logger.info(
            "Embeddings averaged from interactions",
            user_id=user_id,
            papers_used=len(weights),
            avg_weight=float(np.mean(weights)),
            minilm_dim=len(minilm_emb),
            specter_dim=len(specter_emb)
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