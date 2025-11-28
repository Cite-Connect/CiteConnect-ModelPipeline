"""
Recommendation orchestrator - central coordination of recommendation generation.
Implements fallback hierarchy and quality validation from HLD.
"""
from typing import List, Dict, Optional, Tuple
import time
import numpy as np
from app.config import settings
from app.utils.logger import get_logger
from app.db.repositories.paper_repo import PaperRepository
from app.db.repositories.embedding_repo import EmbeddingRepository
from app.services.bootstrap.embedding_service import EmbeddingService
from app.services.bootstrap.ground_truth_service import GroundTruthService
from app.services.runtime.user_state_service import UserStateService

logger = get_logger(__name__)


class RecommendationOrchestrator:
    """
    Orchestrates recommendation generation with fallback strategies.
    Implements multi-level fallback hierarchy from HLD.
    """
    
    # Fallback hierarchy from HLD
    FALLBACK_HIERARCHY = [
        'personalized_embedding_search',
        'profile_based_search',
        'domain_canonical',
        'trending_papers'
    ]
    
    # Mixing weights by user stage from HLD
    MIXING_WEIGHTS = {
        'cold_start': {
            'canonical': 0.4,
            'profile_based': 0.3,
            'trending': 0.2,
            'exploration': 0.1
        },
        'early': {
            'profile_based': 0.3,
            'interaction_based': 0.3,
            'canonical': 0.2,
            'exploration': 0.2
        },
        'mature': {
            'personalized': 0.5,
            'citation_network': 0.3,
            'trending': 0.1,
            'exploration': 0.1
        },
        'expert': {
            'personalized': 0.6,
            'citation_network': 0.3,
            'exploration': 0.1
        }
    }
    
    def __init__(
        self,
        paper_repo: PaperRepository,
        embedding_repo: EmbeddingRepository,
        embedding_service: EmbeddingService,
        ground_truth_service: GroundTruthService,
        user_state_service: UserStateService
    ):
        """
        Initialize recommendation orchestrator.
        
        Args:
            paper_repo: Paper repository
            embedding_repo: Embedding repository
            embedding_service: Embedding service
            ground_truth_service: Ground truth service
            user_state_service: User state service
        """
        self.paper_repo = paper_repo
        self.embedding_repo = embedding_repo
        self.embedding_service = embedding_service
        self.ground_truth_service = ground_truth_service
        self.user_state_service = user_state_service
        
        logger.info("RecommendationOrchestrator initialized")
    
    async def generate_recommendations(
        self,
        user_id: int,
        model_name: str,
        count: int = 10,
        filters: Optional[Dict] = None
    ) -> Dict:
        """
        Generate recommendations with fallback hierarchy.
        
        Args:
            user_id: User identifier
            model_name: Embedding model to use
            count: Number of recommendations
            filters: Optional filters
            
        Returns:
            Dict with recommendations and metadata
        """
        start_time = time.time()
        
        logger.info(
            "Generating recommendations",
            user_id=user_id,
            model=model_name,
            count=count
        )
        
        try:
            # Get user context
            user_context = await self.user_state_service.get_user_context(
                user_id
            )
            
            user_stage = user_context['stage']
            
            logger.debug(
                "User context retrieved",
                user_id=user_id,
                stage=user_stage
            )
            
            # Try strategies in fallback order
            recommendations = None
            strategy_used = None
            
            for strategy in self.FALLBACK_HIERARCHY:
                try:
                    logger.debug(
                        "Trying strategy",
                        strategy=strategy,
                        user_id=user_id
                    )
                    
                    recommendations = await self._execute_strategy(
                        strategy=strategy,
                        user_context=user_context,
                        model_name=model_name,
                        count=count,
                        filters=filters
                    )
                    
                    if recommendations and len(recommendations) >= count // 2:
                        strategy_used = strategy
                        logger.info(
                            "Strategy succeeded",
                            strategy=strategy,
                            count=len(recommendations)
                        )
                        break
                    
                except Exception as e:
                    logger.warning(
                        "Strategy failed, trying next",
                        strategy=strategy,
                        error=str(e)
                    )
                    continue
            
            if not recommendations:
                logger.error(
                    "All strategies failed",
                    user_id=user_id
                )
                raise RuntimeError("Failed to generate recommendations")
            
            # Apply diversity constraints
            recommendations = await self._apply_diversity_constraints(
                recommendations,
                user_stage
            )
            
            # Evaluate quality
            evaluation = await self._evaluate_recommendations(
                recommendations,
                user_context
            )
            
            # Prepare response
            generation_time_ms = (time.time() - start_time) * 1000
            
            response = {
                "recommendations": recommendations[:count],
                "metadata": {
                    "user_stage": user_stage,
                    "strategy_used": strategy_used,
                    "model_used": model_name,
                    "evaluation_scores": evaluation,
                    "cache_hit": False,
                    "generation_time_ms": round(generation_time_ms, 2)
                },
                "explanations": self._generate_explanations(
                    recommendations[:count],
                    user_context,
                    strategy_used
                )
            }
            
            logger.info(
                "Recommendations generated successfully",
                user_id=user_id,
                count=len(recommendations),
                strategy=strategy_used,
                time_ms=round(generation_time_ms, 2)
            )
            
            return response
            
        except Exception as e:
            logger.error(
                "Recommendation generation failed",
                user_id=user_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def _execute_strategy(
        self,
        strategy: str,
        user_context: Dict,
        model_name: str,
        count: int,
        filters: Optional[Dict]
    ) -> List[Dict]:
        """
        Execute specific recommendation strategy.
        
        Args:
            strategy: Strategy name
            user_context: User context
            model_name: Model name
            count: Number of recommendations
            filters: Optional filters
            
        Returns:
            List of paper dicts
        """
        logger.debug(
            "Executing strategy",
            strategy=strategy,
            user_id=user_context['user_id']
        )
        
        if strategy == 'personalized_embedding_search':
            return await self._personalized_embedding_search(
                user_context,
                model_name,
                count,
                filters
            )
        
        elif strategy == 'profile_based_search':
            return await self._profile_based_search(
                user_context,
                model_name,
                count,
                filters
            )
        
        elif strategy == 'domain_canonical':
            return await self._domain_canonical(
                user_context,
                count
            )
        
        elif strategy == 'trending_papers':
            return await self._trending_papers(
                user_context,
                count,
                filters
            )
        
        else:
            logger.warning(
                "Unknown strategy",
                strategy=strategy
            )
            return []
    
    async def _personalized_embedding_search(
        self,
        user_context: Dict,
        model_name: str,
        count: int,
        filters: Optional[Dict]
    ) -> List[Dict]:
        """
        Search using user's learned embedding.
        Requires user to have interaction history.
        """
        logger.debug(
            "Personalized embedding search",
            user_id=user_context['user_id']
        )
        
        # Get user embedding
        embedding_data = await self.embedding_repo.get_user_embedding(
            user_context['user_id'],
            model_name
        )
        
        if not embedding_data:
            logger.debug("No user embedding found")
            return []
        
        user_embedding, method, updated_at = embedding_data
        
        # Get excluded papers
        excluded_papers = user_context.get('filtered_papers', [])
        
        # Extract filters
        domain_filter = None
        min_year = None
        
        if filters:
            domain_filter = filters.get('domains', [None])[0] if filters.get('domains') else None
            min_year = filters.get('min_year')
        
        # Find similar papers
        similar_papers = await self.embedding_repo.find_similar_papers(
            query_embedding=user_embedding,
            model_name=model_name,
            limit=count * 2,  # Get more for filtering
            excluded_paper_ids=excluded_papers,
            domain_filter=domain_filter,
            min_year=min_year
        )
        
        # Fetch full paper data
        paper_ids = [pid for pid, score in similar_papers]
        papers = await self.paper_repo.find_by_ids(paper_ids)
        
        # Add similarity scores
        paper_dict = {p['paper_id']: dict(p) for p in papers}
        
        results = []
        for paper_id, similarity in similar_papers:
            if paper_id in paper_dict:
                paper = paper_dict[paper_id]
                paper['relevance_score'] = float(similarity)
                paper['matching_aspects'] = ['semantic_similarity', 'interaction_based']
                results.append(paper)
        
        logger.debug(
            "Personalized search complete",
            count=len(results)
        )
        
        return results
    
    async def _profile_based_search(
        self,
        user_context: Dict,
        model_name: str,
        count: int,
        filters: Optional[Dict]
    ) -> List[Dict]:
        """
        Search using profile embedding.
        Works for cold-start users.
        """
        logger.debug(
            "Profile-based search",
            user_id=user_context['user_id']
        )
        
        profile = user_context.get('profile', {})
        
        if not profile:
            logger.debug("No profile found")
            return []
        
        # Generate profile embedding
        profile_embedding = await self.embedding_service.embed_user_profile(
            user_id=user_context['user_id'],
            research_stage=profile.get('research_stage', 'phd'),
            primary_domain=profile.get('primary_domain', 'machine_learning'),
            interests=profile.get('interests', []),
            research_goals=profile.get('research_goals'),
            model_name=model_name,
            save_to_db=False  # Don't save, just use for search
        )
        
        # Get excluded papers
        excluded_papers = user_context.get('filtered_papers', [])
        
        # Extract filters
        domain_filter = profile.get('primary_domain')
        min_year = filters.get('min_year') if filters else None
        
        # Find similar papers
        similar_papers = await self.embedding_repo.find_similar_papers(
            query_embedding=profile_embedding,
            model_name=model_name,
            limit=count * 2,
            excluded_paper_ids=excluded_papers,
            domain_filter=domain_filter,
            min_year=min_year
        )
        
        # Fetch full paper data
        paper_ids = [pid for pid, score in similar_papers]
        papers = await self.paper_repo.find_by_ids(paper_ids)
        
        # Add scores
        paper_dict = {p['paper_id']: dict(p) for p in papers}
        
        results = []
        for paper_id, similarity in similar_papers:
            if paper_id in paper_dict:
                paper = paper_dict[paper_id]
                paper['relevance_score'] = float(similarity)
                paper['matching_aspects'] = ['profile_match', 'semantic_similarity']
                results.append(paper)
        
        logger.debug(
            "Profile-based search complete",
            count=len(results)
        )
        
        return results
    
    async def _domain_canonical(
        self,
        user_context: Dict,
        count: int
    ) -> List[Dict]:
        """
        Return canonical papers for user's domain.
        Guaranteed fallback that always works.
        """
        logger.debug(
            "Domain canonical fallback",
            user_id=user_context['user_id']
        )
        
        profile = user_context.get('profile', {})
        domain = profile.get('primary_domain', 'machine_learning')
        stage = user_context.get('stage', 'cold_start')
        research_stage = profile.get('research_stage', 'phd')
        
        # Get mix of foundational and recent
        foundational = await self.ground_truth_service.get_canonical_papers(
            domain=domain,
            tier='foundational',
            user_stage=research_stage,
            count=count // 2
        )
        
        recent = await self.ground_truth_service.get_canonical_papers(
            domain=domain,
            tier='recent',
            user_stage=research_stage,
            count=count // 2
        )
        
        results = []
        
        for paper in foundational + recent:
            paper_dict = dict(paper)
            paper_dict['relevance_score'] = 0.8
            paper_dict['matching_aspects'] = ['canonical_paper', 'domain_match']
            results.append(paper_dict)
        
        logger.debug(
            "Canonical papers retrieved",
            count=len(results)
        )
        
        return results
    
    async def _trending_papers(
        self,
        user_context: Dict,
        count: int,
        filters: Optional[Dict]
    ) -> List[Dict]:
        """
        Return trending papers in user's domain.
        """
        logger.debug(
            "Trending papers fallback",
            user_id=user_context['user_id']
        )
        
        profile = user_context.get('profile', {})
        domain = profile.get('primary_domain')
        
        papers = await self.paper_repo.get_trending_papers(
            domain=domain,
            days=30,
            limit=count
        )
        
        results = []
        for paper in papers:
            paper_dict = dict(paper)
            paper_dict['relevance_score'] = 0.7
            paper_dict['matching_aspects'] = ['trending', 'recent']
            results.append(paper_dict)
        
        logger.debug(
            "Trending papers retrieved",
            count=len(results)
        )
        
        return results
    
    async def _apply_diversity_constraints(
        self,
        papers: List[Dict],
        user_stage: str
    ) -> List[Dict]:
        """
        Apply diversity constraints from HLD business rules.
        
        Args:
            papers: List of papers
            user_stage: User stage
            
        Returns:
            List of diverse papers
        """
        logger.debug(
            "Applying diversity constraints",
            count=len(papers),
            user_stage=user_stage
        )
        
        # Business rules from HLD
        MAX_SAME_AUTHOR = 2
        MAX_SAME_VENUE = 3
        MAX_SAME_DOMAIN = 0.7
        
        diverse_papers = []
        author_counts = {}
        venue_counts = {}
        domain_counts = {}
        
        for paper in papers:
            # Check author diversity
            authors = paper.get('authors', [])
            author_ok = all(
                author_counts.get(author, 0) < MAX_SAME_AUTHOR
                for author in authors
            )
            
            # Check venue diversity
            venue = paper.get('venue')
            venue_ok = venue_counts.get(venue, 0) < MAX_SAME_VENUE or not venue
            
            # Check domain diversity
            domain = paper.get('domain')
            total_so_far = len(diverse_papers)
            domain_count = domain_counts.get(domain, 0)
            domain_ok = (
                total_so_far == 0 or
                domain_count / total_so_far < MAX_SAME_DOMAIN
            )
            
            if author_ok and venue_ok and domain_ok:
                diverse_papers.append(paper)
                
                # Update counts
                for author in authors:
                    author_counts[author] = author_counts.get(author, 0) + 1
                
                if venue:
                    venue_counts[venue] = venue_counts.get(venue, 0) + 1
                
                if domain:
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        logger.debug(
            "Diversity constraints applied",
            original_count=len(papers),
            diverse_count=len(diverse_papers)
        )
        
        return diverse_papers
    
    async def _evaluate_recommendations(
        self,
        recommendations: List[Dict],
        user_context: Dict
    ) -> Dict:
        """
        Evaluate recommendation quality.
        
        Args:
            recommendations: List of papers
            user_context: User context
            
        Returns:
            Dict with evaluation scores
        """
        logger.debug(
            "Evaluating recommendations",
            count=len(recommendations)
        )
        
        paper_ids = [p['paper_id'] for p in recommendations]
        
        # Evaluate against ground truth
        gt_eval = await self.ground_truth_service.evaluate_against_ground_truth(
            paper_ids,
            user_context
        )
        
        # Calculate profile alignment for cold start
        profile_alignment = None
        if user_context['stage'] == 'cold_start':
            profile_alignment = self._calculate_profile_alignment(
                recommendations,
                user_context.get('profile', {})
            )
        
        evaluation = {
            "profile_alignment": profile_alignment,
            "ground_truth_quality": gt_eval.get('ground_truth_quality'),
            "combined_score": None
        }
        
        # Calculate combined score for cold start
        if profile_alignment is not None:
            evaluation['combined_score'] = (
                profile_alignment * 0.6 +
                gt_eval.get('ground_truth_quality', 0.0) * 0.4
            )
        
        logger.debug(
            "Evaluation complete",
            evaluation=evaluation
        )
        
        return evaluation
    
    def _calculate_profile_alignment(
        self,
        papers: List[Dict],
        profile: Dict
    ) -> float:
        """
        Calculate how well papers align with user profile.
        
        Args:
            papers: List of papers
            profile: User profile
            
        Returns:
            float: Alignment score (0-1)
        """
        if not profile or not papers:
            return 0.0
        
        user_domain = profile.get('primary_domain', '')
        user_interests = set(profile.get('interests', []))
        
        alignment_scores = []
        
        for paper in papers:
            score = 0.0
            
            # Domain match (50% weight)
            if paper.get('domain') == user_domain:
                score += 0.5
            
            # Interest keyword match (50% weight)
            paper_text = (
                paper.get('title', '') + ' ' +
                paper.get('abstract', '')
            ).lower()
            
            interest_matches = sum(
                1 for interest in user_interests
                if interest.lower() in paper_text
            )
            
            if user_interests:
                score += 0.5 * (interest_matches / len(user_interests))
            
            alignment_scores.append(score)
        
        return sum(alignment_scores) / len(alignment_scores)
    
    def _generate_explanations(
        self,
        papers: List[Dict],
        user_context: Dict,
        strategy: str
    ) -> Dict[str, str]:
        """
        Generate explanations for why papers were recommended.
        
        Args:
            papers: List of papers
            user_context: User context
            strategy: Strategy used
            
        Returns:
            Dict mapping paper_id to explanation
        """
        explanations = {}
        
        profile = user_context.get('profile', {})
        
        for paper in papers:
            explanation_parts = []
            
            # Domain match
            if paper.get('domain') == profile.get('primary_domain'):
                explanation_parts.append(
                    f"Matches your {profile.get('primary_domain')} research area"
                )
            
            # Recent and high quality
            if paper.get('citation_count', 0) > 100:
                explanation_parts.append("Highly cited paper")
            
            # Strategy-specific
            if strategy == 'personalized_embedding_search':
                explanation_parts.append(
                    "Based on your reading history"
                )
            elif strategy == 'profile_based_search':
                explanation_parts.append(
                    "Matches your research interests"
                )
            elif strategy == 'domain_canonical':
                explanation_parts.append(
                    "Foundational paper in your field"
                )
            
            explanations[paper['paper_id']] = ". ".join(explanation_parts)
        
        return explanations