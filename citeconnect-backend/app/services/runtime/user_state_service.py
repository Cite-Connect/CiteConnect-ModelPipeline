"""
User state service for managing user journey transitions.
Handles progression from cold_start → early → mature → expert.
"""
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from app.config import settings
from app.utils.logger import get_logger
from app.db.repositories.user_repo import UserRepository
from app.db.repositories.interaction_repo import InteractionRepository

logger = get_logger(__name__)


class UserStateService:
    """
    Manages user state transitions based on interaction patterns.
    Implements state machine logic from HLD.
    """
    
    # State transition rules from config
    STATE_TRANSITIONS = {
        'cold_start': {
            'threshold': {
                'saves': settings.STATE_COLD_START_TO_EARLY_MIN_SAVES,
                'meaningful_interactions': settings.STATE_COLD_START_TO_EARLY_MIN_INTERACTIONS
            },
            'next': 'early'
        },
        'early': {
            'threshold': {
                'saves': settings.STATE_EARLY_TO_MATURE_MIN_SAVES,
                'interactions': settings.STATE_EARLY_TO_MATURE_MIN_INTERACTIONS,
                'domains_explored': settings.STATE_EARLY_TO_MATURE_MIN_DOMAINS,
                'days_active': settings.STATE_EARLY_TO_MATURE_MIN_DAYS
            },
            'next': 'mature'
        },
        'mature': {
            'threshold': {
                'interactions': settings.STATE_MATURE_TO_EXPERT_MIN_INTERACTIONS,
                'days_active': settings.STATE_MATURE_TO_EXPERT_MIN_DAYS
            },
            'next': 'expert'
        }
    }
    
    def __init__(
        self,
        user_repo: UserRepository,
        interaction_repo: InteractionRepository
    ):
        """
        Initialize user state service.
        
        Args:
            user_repo: User repository
            interaction_repo: Interaction repository
        """
        self.user_repo = user_repo
        self.interaction_repo = interaction_repo
        logger.info("UserStateService initialized")
    
    async def get_user_stage(self, user_id: int) -> str:
        """
        Get user's current recommendation stage.
        
        Args:
            user_id: User identifier
            
        Returns:
            str: Current stage (cold_start, early, mature, expert)
        """
        logger.debug("Getting user stage", user_id=user_id)
        
        state = await self.user_repo.get_recommendation_state(user_id)
        
        if not state:
            # Initialize state for new user
            logger.info(
                "No state found, initializing",
                user_id=user_id
            )
            state = await self.user_repo.initialize_recommendation_state(
                user_id,
                initial_stage='cold_start'
            )
        
        stage = state['recommendation_stage']
        
        logger.debug(
            "User stage retrieved",
            user_id=user_id,
            stage=stage
        )
        
        return stage
    
    async def check_state_transition(self, user_id: int) -> bool:
        """
        Check if user should transition to next stage.
        
        Args:
            user_id: User identifier
            
        Returns:
            bool: True if transition occurred
        """
        logger.debug(
            "Checking state transition",
            user_id=user_id
        )
        
        # Get current stage
        current_stage = await self.get_user_stage(user_id)
        
        if current_stage == 'expert':
            logger.debug(
                "User already at final stage",
                user_id=user_id
            )
            return False
        
        # Get transition rules
        transition_rule = self.STATE_TRANSITIONS.get(current_stage)
        if not transition_rule:
            logger.warning(
                "No transition rule found",
                user_id=user_id,
                current_stage=current_stage
            )
            return False
        
        # Check if thresholds are met
        meets_threshold = await self._check_thresholds(
            user_id,
            transition_rule['threshold']
        )
        
        if meets_threshold:
            # Transition to next stage
            next_stage = transition_rule['next']
            
            await self.update_user_state(
                user_id,
                new_stage=next_stage
            )
            
            logger.info(
                "User transitioned to new stage",
                user_id=user_id,
                from_stage=current_stage,
                to_stage=next_stage
            )
            
            return True
        
        logger.debug(
            "Thresholds not met for transition",
            user_id=user_id,
            current_stage=current_stage
        )
        
        return False
    
    async def _check_thresholds(
        self,
        user_id: int,
        thresholds: Dict[str, int]
    ) -> bool:
        """
        Check if user meets all thresholds.
        
        Args:
            user_id: User identifier
            thresholds: Dict of threshold requirements
            
        Returns:
            bool: True if all thresholds met
        """
        logger.debug(
            "Checking thresholds",
            user_id=user_id,
            thresholds=thresholds
        )
        
        # Get interaction counts
        counts = await self.interaction_repo.get_interaction_counts(
            user_id,
            days=365  # Look back 1 year
        )
        
        # Check saves
        if 'saves' in thresholds:
            save_count = counts.get('save', 0)
            if save_count < thresholds['saves']:
                logger.debug(
                    "Save threshold not met",
                    user_id=user_id,
                    required=thresholds['saves'],
                    actual=save_count
                )
                return False
        
        # Check meaningful interactions
        if 'meaningful_interactions' in thresholds:
            meaningful_count = await self.interaction_repo.get_meaningful_interactions_count(
                user_id
            )
            if meaningful_count < thresholds['meaningful_interactions']:
                logger.debug(
                    "Meaningful interactions threshold not met",
                    user_id=user_id,
                    required=thresholds['meaningful_interactions'],
                    actual=meaningful_count
                )
                return False
        
        # Check total interactions
        if 'interactions' in thresholds:
            total_interactions = sum(counts.values())
            if total_interactions < thresholds['interactions']:
                logger.debug(
                    "Total interactions threshold not met",
                    user_id=user_id,
                    required=thresholds['interactions'],
                    actual=total_interactions
                )
                return False
        
        # Check domains explored
        if 'domains_explored' in thresholds:
            domains_count = await self.interaction_repo.get_domains_explored(
                user_id
            )
            if domains_count < thresholds['domains_explored']:
                logger.debug(
                    "Domains explored threshold not met",
                    user_id=user_id,
                    required=thresholds['domains_explored'],
                    actual=domains_count
                )
                return False
        
        # Check days active
        if 'days_active' in thresholds:
            state = await self.user_repo.get_recommendation_state(user_id)
            if state:
                days_active = (datetime.now() - state['created_at']).days
                if days_active < thresholds['days_active']:
                    logger.debug(
                        "Days active threshold not met",
                        user_id=user_id,
                        required=thresholds['days_active'],
                        actual=days_active
                    )
                    return False
        
        logger.debug(
            "All thresholds met",
            user_id=user_id
        )
        
        return True
    
    async def update_user_state(
        self,
        user_id: int,
        new_stage: str
    ) -> None:
        """
        Update user's recommendation stage.
        
        Args:
            user_id: User identifier
            new_stage: New stage to set
        """
        logger.info(
            "Updating user state",
            user_id=user_id,
            new_stage=new_stage
        )
        
        await self.user_repo.update_recommendation_state(
            user_id,
            {'recommendation_stage': new_stage}
        )
        
        logger.info(
            "User state updated",
            user_id=user_id,
            stage=new_stage
        )
    
    async def get_stage_appropriate_strategies(
        self,
        stage: str
    ) -> List[str]:
        """
        Get recommendation strategies appropriate for user stage.
        
        Args:
            stage: User stage
            
        Returns:
            List[str]: Strategy names
        """
        logger.debug(
            "Getting stage-appropriate strategies",
            stage=stage
        )
        
        # Strategy mapping from HLD
        STAGE_STRATEGIES = {
            'cold_start': [
                'canonical',
                'profile_based',
                'trending',
                'exploration'
            ],
            'early': [
                'profile_based',
                'interaction_based',
                'canonical',
                'exploration'
            ],
            'mature': [
                'personalized',
                'citation_network',
                'trending',
                'exploration'
            ],
            'expert': [
                'personalized',
                'citation_network',
                'collaborative',
                'domain_expert'
            ]
        }
        
        strategies = STAGE_STRATEGIES.get(stage, ['profile_based'])
        
        logger.debug(
            "Strategies selected",
            stage=stage,
            strategies=strategies
        )
        
        return strategies
    
    async def get_user_context(self, user_id: int) -> Dict:
        """
        Get comprehensive user context for recommendations.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with user context
        """
        logger.debug("Getting user context", user_id=user_id)
        
        # Get profile
        profile = await self.user_repo.get_profile(user_id)
        
        # Get state
        state = await self.user_repo.get_recommendation_state(user_id)
        
        # Get recent interactions
        recent_interactions = await self.interaction_repo.get_user_interactions(
            user_id,
            limit=50,
            min_strength=0.0
        )
        
        # Get filtered papers
        filtered_papers = await self.interaction_repo.get_filtered_papers(
            user_id
        )
        
        # Get interaction counts
        interaction_counts = await self.interaction_repo.get_interaction_counts(
            user_id,
            days=30
        )
        
        context = {
            'user_id': user_id,
            'profile': dict(profile) if profile else {},
            'stage': state['recommendation_stage'] if state else 'cold_start',
            'recent_interactions': [dict(i) for i in recent_interactions],
            'filtered_papers': filtered_papers,
            'interaction_counts': interaction_counts,
            'strategies': await self.get_stage_appropriate_strategies(
                state['recommendation_stage'] if state else 'cold_start'
            )
        }
        
        logger.debug(
            "User context compiled",
            user_id=user_id,
            stage=context['stage'],
            interaction_count=len(recent_interactions)
        )
        
        return context
    
    async def increment_interaction_count(self, user_id: int) -> None:
        """
        Increment user's interaction count.
        
        Args:
            user_id: User identifier
        """
        logger.debug(
            "Incrementing interaction count",
            user_id=user_id
        )
        
        state = await self.user_repo.get_recommendation_state(user_id)
        
        if state:
            new_count = state['interaction_count'] + 1
            
            await self.user_repo.update_recommendation_state(
                user_id,
                {'interaction_count': new_count}
            )
            
            logger.debug(
                "Interaction count incremented",
                user_id=user_id,
                new_count=new_count
            )
            
            # Check if state transition should occur
            await self.check_state_transition(user_id)