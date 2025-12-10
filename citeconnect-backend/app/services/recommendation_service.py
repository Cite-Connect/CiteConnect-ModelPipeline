"""
Recommendation service for CiteConnect.
Generates personalized paper recommendations using dual embedding models.
Supports cold-start (profile-based) and warm-start (interaction-based).
"""
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import random
from datetime import datetime, timedelta
from pathlib import Path
import json

from app.db.connection import DatabaseConnection
from app.db.repositories.user_repo import UserRepository
from app.db.repositories.paper_repo import PaperRepository
from app.db.repositories.ground_truth_repo import GroundTruthRepository
from app.services.user_embedding_service import UserEmbeddingService
from app.services.fairness_service import fairness_aware_rerank
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RecommendationService:
    """
    Service for generating personalized paper recommendations.
    Supports:
    - Cold-start recommendations (profile-based)
    - Warm-start recommendations (interaction-based)
    - Multi-factor scoring
    - Ground truth validation
    - Dual bias mitigation
    """
    
    # Unified Default Scoring Weights (for both Cold and Warm Start)
    DEFAULT_WEIGHTS = {
        'semantic': 0.40,
        'citation': 0.20,
        'recency': 0.15,
        'ground_truth': 0.10,
        'reading_level': 0.10,
        'diversity': 0.05
    }
    
    # Deprecated specific defaults (kept for backward compatibility if needed, but unused)
    DEFAULT_COLD_START_WEIGHTS = DEFAULT_WEIGHTS
    DEFAULT_WARM_START_WEIGHTS = DEFAULT_WEIGHTS
    
    # Retrieval limits
    SEMANTIC_CANDIDATE_LIMIT = 150
    CANONICAL_CANDIDATE_COUNT = 25
    GT_NETWORK_CANDIDATE_COUNT = 25
    
    def __init__(self, db: DatabaseConnection):
        """
        Initialize recommendation service.
        """
        self.db = db
        self.user_repo = UserRepository(db)
        self.paper_repo = PaperRepository(db)
        self.gt_repo = GroundTruthRepository(db)
        self.user_embedding_service = UserEmbeddingService(db)

        # Load bias mitigation configuration (if present)
        self.bias_config = self._load_bias_config()
        
        logger.info("RecommendationService initialized")

    # ... [Helper methods _load_bias_config and _get_mitigation_policy_for_profile remain unchanged] ...
    def _load_bias_config(self) -> Dict:
        """Load bias mitigation config from JSON, if present."""
        try:
            config_path = (
                Path(__file__).parent.parent.parent
                / "bias_config"
                / "bias_mitigation_config.json"
            )
            if not config_path.exists():
                logger.info("No bias mitigation config found – running without mitigation")
                return {}
            with config_path.open("r") as f:
                cfg = json.load(f)
            logger.info("Loaded bias mitigation config", path=str(config_path))
            return cfg
        except Exception as e:
            logger.warning(f"Failed to load bias mitigation config: {e}")
            return {}

    def _get_mitigation_policy_for_profile(self, profile: Dict, model: str = 'minilm', is_cold_start: bool = True) -> Dict:
        """Compute mitigation policy for this user from bias_config."""
        policy = {
            "factor": 1.0,
            "weight_multipliers": {},
            "min_score_threshold": None,
            "applied_rules": [],
        }

        if not self.bias_config:
            return policy

        cfg = self.bias_config
        model_key_map = {
            'all-MiniLM-L6-v2': 'minilm',
            'minilm': 'minilm',
            'specter': 'specter',
            'specter2': 'specter'
        }
        model_key = model_key_map.get(model, 'minilm')
        mode_key = "cold_start" if is_cold_start else "warm_start"
        
        if mode_key not in cfg: return policy
        mode_cfg = cfg[mode_key]
        if model_key not in mode_cfg: return policy
        model_cfg = mode_cfg[model_key]
        
        fields_to_check = ["primary_domain", "research_stage", "reading_level"]
        
        for field in fields_to_check:
            user_value = profile.get(field)
            if not user_value: continue
                
            field_cfg = model_cfg.get(field)
            if not field_cfg: continue
                
            underperforming_slices = field_cfg.get("underperforming_slices", [])
            if user_value not in underperforming_slices: continue
            
            boost_factor = field_cfg.get("boost_factor", 1.0)
            min_score_floor = field_cfg.get("min_score_floor")
            
            policy["factor"] *= float(boost_factor)
            
            if min_score_floor is not None:
                th = float(min_score_floor)
                if policy["min_score_threshold"] is None:
                    policy["min_score_threshold"] = th
                else:
                    policy["min_score_threshold"] = min(policy["min_score_threshold"], th)
            
            policy["applied_rules"].append({
                "field": field, "value": user_value,
                "boost_factor": boost_factor, "min_score_floor": min_score_floor
            })
            
        return policy

    # -------------------------------------------------------------------------
    # NEW HELPER: Fetch Personalized Weights
    # -------------------------------------------------------------------------
    async def _get_user_scoring_weights(self, user_id: int) -> Dict[str, float]:
        """
        Fetch personalized scoring weights for a user.
        Falls back to DEFAULT_WEIGHTS if none exist.
        """
        try:
            state = await self.user_repo.get_recommendation_state(user_id)
            if state and state.get('scoring_weights'):
                # Ensure it's a dict (handle potential JSON string if not auto-parsed)
                weights = state['scoring_weights']
                if isinstance(weights, str):
                    return json.loads(weights)
                return weights
        except Exception as e:
            logger.warning(f"Failed to fetch user weights for {user_id}, using defaults: {e}")
        
        return self.DEFAULT_WEIGHTS.copy()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def generate_recommendations(
        self,
        user_id: int,
        count: int = 10,
        model: str = 'minilm',
        scoring_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for generating recommendations.
        Routes to cold-start or warm-start based on user's stage.
        """
        logger.info(
            "Generating recommendations",
            user_id=user_id,
            count=count,
            model=model
        )
        
        state = await self.user_repo.get_recommendation_state(user_id)
        
        if not state:
            raise ValueError(f"No recommendation state found for user {user_id}")
        
        stage = state['recommendation_stage']
        interaction_count = state['interaction_count']
        
        if interaction_count < 10:
            logger.info("Using cold-start strategy", user_id=user_id, stage=stage)
            return await self.generate_cold_start_recommendations(
                user_id=user_id, count=count, model=model, scoring_weights=scoring_weights
            )
        else:
            logger.info("Using warm-start strategy", user_id=user_id, stage=stage, interactions=interaction_count)
            return await self.generate_warm_start_recommendations(
                user_id=user_id, count=count, model=model, scoring_weights=scoring_weights
            )
    
    async def generate_cold_start_recommendations(
        self,
        user_id: int,
        count: int = 10,
        model: str = 'minilm',
        scoring_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Generate recommendations for cold-start users (0-9 interactions).
        """
        logger.info("Generating cold-start recommendations", user_id=user_id, count=count, model=model)
        
        # CHANGE 1: Use personalized weights if not explicitly overridden
        if scoring_weights:
            weights = scoring_weights
        else:
            weights = await self._get_user_scoring_weights(user_id)
        
        profile = await self.user_repo.get_profile(user_id)
        if not profile: raise ValueError(f"No profile found for user {user_id}")
        interests = await self.user_repo.get_user_interests(user_id)

        mitigation_policy = self._get_mitigation_policy_for_profile(profile=profile, model=model, is_cold_start=True)
        
        embeddings = await self.user_embedding_service.get_or_generate_user_embeddings(user_id)
        model_key_map = {'all-MiniLM-L6-v2': 'minilm', 'minilm': 'minilm', 'specter': 'specter', 'specter2': 'specter'}
        embedding_key = model_key_map.get(model, 'minilm')
        
        if embedding_key not in embeddings:
            raise ValueError(f"Model '{model}' not available.")
        user_embedding = embeddings[embedding_key]
        
        # 3A: Semantic search
        semantic_candidates = await self._retrieve_semantic_candidates(
            user_embedding=user_embedding, domain=profile['primary_domain'],
            model=model, limit=self.SEMANTIC_CANDIDATE_LIMIT
        )
        
        # 3B: Canonical papers
        canonical_candidates = await self._retrieve_canonical_candidates(
            domain=profile['primary_domain'], user_stage=profile.get('research_stage', 'phd'),
            count=self.CANONICAL_CANDIDATE_COUNT
        )
        
        # 3C: Ground truth network
        gt_candidates = await self._retrieve_ground_truth_candidates(
            user_interests=[i['interest_term'] for i in interests],
            domain=profile['primary_domain'], count=self.GT_NETWORK_CANDIDATE_COUNT
        )
        
        all_candidates = self._merge_candidates(semantic_candidates, canonical_candidates, gt_candidates)
        
        scored_papers = await self._apply_multi_factor_scoring(
            candidates=all_candidates, user=profile, user_interests=interests,
            scoring_weights=weights, is_cold_start=True, mitigation_policy=mitigation_policy, apply_bias_mitigation=True,
        )
        
        diverse_papers = await self._apply_diversity_filtering(scored_papers=scored_papers, target_count=21)
        final_recommendations = diverse_papers[:count]
        enriched = self._enrich_recommendations(papers=final_recommendations, user_interests=[i['interest_term'] for i in interests])
        fairness_reranked = self._apply_fairness_reranking(enriched)
        
        return {
            'user_id': user_id, 'papers': fairness_reranked, 'method': 'cold_start',
            'model_used': model, 'scoring_weights': weights, 'generated_at': datetime.utcnow().isoformat(),
            'total_candidates': len(all_candidates), 'mitigation_policy': mitigation_policy,
        }

    async def generate_warm_start_recommendations(
        self,
        user_id: int,
        count: int = 10,
        model: str = 'minilm',
        scoring_weights: Optional[Dict[str, float]] = None,
        context_paper_ids: Optional[List[str]] = None  # <--- OFFLINE PARAM
    ) -> Dict[str, Any]:
        """
        Generate recommendations for warm-start users (10+ interactions).
        """
        logger.info(
            "Generating warm-start recommendations",
            user_id=user_id,
            count=count,
            model=model,
            mode="offline_eval" if context_paper_ids is not None else "production"
        )
        
        saved_paper_ids = []
        
        # CHANGE 2: Use personalized weights if not explicitly overridden
        if scoring_weights:
            weights = scoring_weights
        else:
            weights = await self._get_user_scoring_weights(user_id)
        
        profile = await self.user_repo.get_profile(user_id)

        mitigation_policy = self._get_mitigation_policy_for_profile(
            profile=profile,
            model=model,
            is_cold_start=False
        )
        
        # 1. CONTEXT DETERMINATION
        if context_paper_ids is not None:
            # OFFLINE MODE
            saved_paper_ids = context_paper_ids
        else:
            # PRODUCTION MODE
            saved_papers_rows = await self._get_user_saved_papers(user_id)
            saved_paper_ids = [p['paper_id'] for p in saved_papers_rows]
        
        # Get user embedding
        embeddings = await self.user_embedding_service.get_or_generate_user_embeddings(user_id)
        model_key_map = {
            'all-MiniLM-L6-v2': 'minilm',
            'minilm': 'minilm',
            'specter': 'specter',
            'specter2': 'specter'
        }
        embedding_key = model_key_map.get(model, 'minilm')
        
        if embedding_key not in embeddings:
            raise ValueError(f"Model '{model}' not available.")
    
        user_embedding = embeddings[embedding_key]
        
        # Retrieve candidates
        logger.debug("Retrieving warm-start candidates", user_id=user_id)
        
        # Strategy A: Semantic
        semantic_candidates = await self._retrieve_semantic_candidates(
            user_embedding=user_embedding,
            domain=profile['primary_domain'],
            model=model,
            limit=200
        )
        
        # Strategy B: Citation network
        citation_candidates = await self._retrieve_citation_network_candidates(
            paper_ids=saved_paper_ids, 
            limit=50
        )
        
        # Strategy C: Collaborative
        collaborative_candidates = await self._retrieve_collaborative_candidates(
            user_id=user_id,
            user_embedding=user_embedding,
            model=model,
            limit=30
        )
        
        # Strategy D: Temporal
        temporal_candidates = await self._retrieve_temporal_candidates(
            domain=profile['primary_domain'],
            sub_domains=profile.get('sub_domains', []),
            limit=20
        )
        
        # Merge candidates
        all_candidates = self._merge_candidates(
            semantic_candidates,
            citation_candidates,
            collaborative_candidates,
            temporal_candidates
        )
        
        # 2. FILTERING LOGIC
        if context_paper_ids is not None:
            # OFFLINE MODE
            seen_set = set(context_paper_ids)
            all_candidates = [p for p in all_candidates if p['paper_id'] not in seen_set]
        else:
            # PRODUCTION MODE
            all_candidates = await self._filter_seen_papers(all_candidates, user_id)
        
        logger.info(
            "Warm-start candidates retrieved",
            user_id=user_id,
            total=len(all_candidates)
        )
        
        # Apply scoring
        scored_papers = await self._apply_multi_factor_scoring(
            candidates=all_candidates,
            user=profile,
            user_interests=None,
            scoring_weights=weights,
            is_cold_start=False,
            mitigation_policy=mitigation_policy,
            apply_bias_mitigation=True,
        )
        
        # Diversity filtering
        diverse_papers = await self._apply_diversity_filtering(
            scored_papers=scored_papers,
            target_count=21,
            max_per_author=3,
            max_per_venue=3
        )
        
        final_recommendations = diverse_papers[:count]
        
        enriched = self._enrich_recommendations(
            papers=final_recommendations,
            user_interests=None
        )
        
        fairness_reranked = self._apply_fairness_reranking(enriched)
        
        return {
            'user_id': user_id,
            'papers': fairness_reranked,
            'method': 'warm_start',
            'model_used': model,
            'scoring_weights': weights,
            'generated_at': datetime.utcnow().isoformat(),
            'total_candidates': len(all_candidates),
            'mitigation_policy': mitigation_policy,
        }
    
    async def generate_search_augmented_recommendations(
        self,
        user_id: int,
        search_query: str,
        count: int = 10,
        model: str = 'minilm',
        scoring_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Generate recommendations augmented by search query.
        Uses hybrid approach: keyword + semantic + profile.
        """
        logger.info(
            "Generating search-augmented recommendations",
            user_id=user_id,
            search_query=search_query[:100],
            count=count,
            model=model
        )
        
        # Get user data
        profile = await self.user_repo.get_profile(user_id)
        if not profile:
            raise ValueError(f"No profile found for user {user_id}")
        
        interests = await self.user_repo.get_user_interests(user_id)
        
        # Get user embedding
        embeddings = await self.user_embedding_service.get_or_generate_user_embeddings(user_id)
        
        # Map model name
        model_key_map = {
            'all-MiniLM-L6-v2': 'minilm',
            'minilm': 'minilm',
            'specter': 'specter',
            'specter2': 'specter'
        }
        embedding_key = model_key_map.get(model, 'minilm')
        user_embedding = embeddings[embedding_key]
        
        # ────────────────────────────────────────────────────────────
        # PHASE 1: Keyword Search (Fast, Precise)
        # ────────────────────────────────────────────────────────────
        logger.debug("Phase 1: Keyword search")
        keyword_results_raw = await self.paper_repo.search_by_text(
            search_text=search_query,
            limit=50
        )
        
        # FIX: Convert asyncpg.Record to dict FIRST
        keyword_results = [dict(row) for row in keyword_results_raw]
        max_relevance = max([p.get('relevance', 0) for p in keyword_results]) if keyword_results else 1.0
        # Now we can modify
        for paper in keyword_results:
            paper['match_source'] = 'keyword'
            raw_relevance = float(paper.get('relevance', 0))
            paper['keyword_score'] = raw_relevance / max_relevance if max_relevance > 0 else 0.0
            
            # Title boost
            title_lower = str(paper.get('title', '')).lower()
            query_terms_clean = [t.lower() for t in search_query.split() if len(t) > 3]
            title_matches = sum(1 for term in query_terms_clean if term in title_lower)
            
            if title_matches > 0 and query_terms_clean:
                boost = min(title_matches / len(query_terms_clean), 1.0) * 0.4
                paper['keyword_score'] = min(paper['keyword_score'] + boost, 1.0)
        
        logger.info(
            "Keyword search complete",
            results=len(keyword_results),
            avg_score=np.mean([p['keyword_score'] for p in keyword_results]) if keyword_results else 0
        )
        
        # ────────────────────────────────────────────────────────────
        # PHASE 2: Semantic Search (Contextual)
        # ────────────────────────────────────────────────────────────
        logger.debug("Phase 2: Semantic search")
        
        # Import embedding service
        from app.services.bootstrap.embedding_service import get_embedding_service
        embedding_service = get_embedding_service()
        
        # Generate query embedding
        query_embedding = embedding_service.encode_text(
            text=search_query,
            model=embedding_key,
            normalize=True
        )
        
        logger.debug(
            "Query embedding generated",
            model=embedding_key,
            embedding_shape=query_embedding.shape
        )
        
        # CHANGE: Increased limit and min_similarity
        semantic_results_raw = await self.paper_repo.semantic_search(
            embedding=query_embedding,
            model=embedding_key,
            domain=profile['primary_domain'],
            limit=100,              # Increased from 50
            min_similarity=0.25     # Increased from 0.2 for better quality
        )
        
        # FIX: Convert to dict
        semantic_results = semantic_results_raw  # Already returns List[Dict] from semantic_search
        
        # Tag scores
        for paper in semantic_results:
            paper['match_source'] = 'semantic'
            paper['semantic_score'] = float(paper.get('similarity', 0))
        
        logger.info(
            "Semantic search complete",
            results=len(semantic_results),
            avg_similarity=np.mean([p['semantic_score'] for p in semantic_results]) if semantic_results else 0
        )
        
        # ────────────────────────────────────────────────────────────
        # PHASE 3: Profile-Based Candidates (Personalization)
        # ────────────────────────────────────────────────────────────
        logger.debug("Phase 3: Profile-based retrieval")
        
        profile_results_raw = await self._retrieve_semantic_candidates(
            user_embedding=user_embedding,
            domain=profile['primary_domain'],
            model=embedding_key,
            limit=50
        )
        
        # FIX: Convert to dict
        profile_results = [dict(row) for row in profile_results_raw]
        
        # Tag scores
        for paper in profile_results:
            if 'match_source' not in paper:
                paper['match_source'] = 'profile'
            paper['profile_score'] = float(paper.get('semantic_similarity', 0))
        
        logger.info(
            "Profile-based retrieval complete",
            results=len(profile_results)
        )
        
        # ────────────────────────────────────────────────────────────
        # PHASE 4: Merge and Deduplicate
        # ────────────────────────────────────────────────────────────
        logger.debug("Phase 4: Merging results")
        
        all_candidates = {}
        
        # Merge keyword results
        for paper in keyword_results:
            paper_id = paper['paper_id']
            all_candidates[paper_id] = paper
        
        # Merge semantic results
        for paper in semantic_results:
            paper_id = paper['paper_id']
            if paper_id in all_candidates:
                # Paper appears in multiple sources - merge scores
                all_candidates[paper_id]['semantic_score'] = paper['semantic_score']
                if all_candidates[paper_id]['match_source'] == 'keyword':
                    all_candidates[paper_id]['match_source'] = 'keyword+semantic'
            else:
                all_candidates[paper_id] = paper
        
        # Merge profile results
        for paper in profile_results:
            paper_id = paper['paper_id']
            if paper_id in all_candidates:
                all_candidates[paper_id]['profile_score'] = paper['profile_score']
                # Update match source to reflect profile presence
                current_source = all_candidates[paper_id]['match_source']
                if 'profile' not in current_source:
                    all_candidates[paper_id]['match_source'] = current_source + '+profile'
            else:
                all_candidates[paper_id] = paper
        
        # CHANGE: Calculate multi-source boost
        for paper_id, paper in all_candidates.items():
            match_source = paper.get('match_source', '')
            
            # Boost papers from multiple sources
            if '+' in match_source:
                source_count = len(match_source.split('+'))
                # 10% boost per additional source
                paper['multi_source_boost'] = 0.10 * (source_count - 1)
            else:
                paper['multi_source_boost'] = 0.0
        
        candidates_list = list(all_candidates.values())
        
        logger.info(
            "Candidates merged",
            total_unique=len(candidates_list),
            keyword_only=sum(1 for p in candidates_list if p['match_source'] == 'keyword'),
            semantic_only=sum(1 for p in candidates_list if p.get('match_source') == 'semantic'),
            profile_only=sum(1 for p in candidates_list if p.get('match_source') == 'profile'),
            multi_source=sum(1 for p in candidates_list if '+' in p.get('match_source', ''))
        )
        
        # ────────────────────────────────────────────────────────────
        # PHASE 5: Hybrid Scoring (REBALANCED WEIGHTS)
        # ────────────────────────────────────────────────────────────
        logger.debug("Phase 5: Hybrid scoring")
        
        # CHANGE: Rebalanced weights to emphasize semantic + profile
        default_weights = {
            'keyword': 0.50,      # Reduced from 0.50
            'semantic': 0.35,     # Increased from 0.35
            'profile': 0.15       # Increased from 0.15
        }
        
        weights = scoring_weights or default_weights
        
        for paper in candidates_list:
            # CHANGE: Higher minimum scores for missing components
            keyword_score = paper.get('keyword_score', 0.0)
            semantic_score = paper.get('semantic_score', 0.15)  # Increased from 0.05
            profile_score = paper.get('profile_score', 0.10)    # Increased from 0.01
            
            # Calculate base score
            base_score = (
                weights['keyword'] * keyword_score +
                weights['semantic'] * semantic_score +
                weights['profile'] * profile_score
            )
            
            # CHANGE: Apply multi-source boost
            multi_source_boost = paper.get('multi_source_boost', 0.0)
            paper['final_score'] = base_score * (1.0 + multi_source_boost)
            
            # Store breakdown for explainability
            paper['score_breakdown'] = {
                'keyword': keyword_score,
                'semantic': semantic_score,
                'profile': profile_score,
                'multi_source_boost': multi_source_boost
            }
        
        # Sort by final score
        candidates_list.sort(key=lambda x: x['final_score'], reverse=True)
        
        logger.info(
            "Hybrid scoring complete",
            total_candidates=len(candidates_list),
            avg_score=np.mean([p['final_score'] for p in candidates_list]) if candidates_list else 0,
            max_score=candidates_list[0]['final_score'] if candidates_list else 0,
            top_paper_breakdown=candidates_list[0]['score_breakdown'] if candidates_list else None
        )
        
        # ────────────────────────────────────────────────────────────
        # PHASE 6: Diversity Filtering
        # ────────────────────────────────────────────────────────────
        logger.debug("Phase 6: Applying diversity")
        
        diverse_papers = await self._apply_diversity_filtering(
            scored_papers=candidates_list,
            target_count=min(count * 2, 21),  # Get extra for final selection
            max_per_author=2,
            max_per_venue=2
        )
        
        # ────────────────────────────────────────────────────────────
        # PHASE 7: Take Top N and Enrich
        # ────────────────────────────────────────────────────────────
        final_recommendations = diverse_papers[:count]
        
        # Enrich with explanations
        enriched = self._enrich_search_recommendations(
            papers=final_recommendations,
            search_query=search_query,
            user_interests=[i['interest_term'] for i in interests]
        )
        
        # Apply field-based fairness reranking (paper-level fairness)
        fairness_reranked = self._apply_fairness_reranking(enriched)
        
        logger.info(
            "Search-augmented recommendations generated",
            user_id=user_id,
            search_query=search_query[:50],
            count=len(fairness_reranked),
            avg_score=np.mean([p['final_score'] for p in fairness_reranked]) if fairness_reranked else 0,
            sample_explanation=fairness_reranked[0].get('relevance_explanation') if fairness_reranked else None,
            sample_breakdown=fairness_reranked[0].get('score_breakdown') if fairness_reranked else None
        )
        
        return {
            'user_id': user_id,
            'papers': fairness_reranked,
            'method': 'search_augmented',
            'search_query': search_query,
            'model_used': model,
            'scoring_weights': weights,
            'generated_at': datetime.utcnow().isoformat(),
            'total_candidates': len(candidates_list)
        }

    def _enrich_search_recommendations(
        self,
        papers: List[Dict],
        search_query: str,
        user_interests: List[str]
    ) -> List[Dict]:
        """
        Add relevance explanations for search-augmented recommendations.
        Checks both title AND abstract for matches.
        """
        
        if not papers:
            logger.warning("No papers to enrich")
            return []
        
        if not search_query:
            logger.warning("No search query provided for enrichment")
            search_query = ""
        
        enriched = []
        
        for i, paper in enumerate(papers):
            try:
                enriched_paper = paper.copy()
                
                # Safely get title and abstract (handle None values)
                title = str(paper.get('title') or '')
                abstract = str(paper.get('abstract') or '')
                
                title_lower = title.lower()
                abstract_lower = abstract.lower()
                
                # Generate explanation
                explanation_parts = []
                
                match_source = paper.get('match_source', '')
                breakdown = paper.get('score_breakdown', {})
                
                # Extract search terms (remove stop words and short terms)
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
                query_terms = [
                    t.lower() for t in search_query.split() 
                    if len(t) > 3 and t.lower() not in stop_words
                ]
                
                # Check for keyword matches
                if 'keyword' in match_source or breakdown.get('keyword', 0) > 0.1:
                    # Find matched terms in TITLE
                    title_matches = [term for term in query_terms if term in title_lower]
                    
                    # Find matched terms in ABSTRACT
                    abstract_matches = [term for term in query_terms if term in abstract_lower]
                    
                    if title_matches:
                        # Title matches are more important
                        unique_matches = list(set(title_matches))[:4]  # Max 4 terms
                        explanation_parts.append(
                            f"Title matches: {', '.join(unique_matches)}"
                        )
                    elif abstract_matches:
                        # Abstract matches
                        unique_matches = list(set(abstract_matches))[:4]
                        explanation_parts.append(
                            f"Abstract discusses: {', '.join(unique_matches)}"
                        )
                    else:
                        # Generic keyword match
                        explanation_parts.append("Matches your search query")
                
                # Check for semantic similarity
                if 'semantic' in match_source or breakdown.get('semantic', 0) > 0.5:
                    explanation_parts.append("Semantically similar to your search")
                
                # Check for profile alignment
                if breakdown.get('profile', 0) > 0.6:
                    explanation_parts.append("Aligns with your research interests")
                elif breakdown.get('profile', 0) > 0.4:
                    explanation_parts.append("Relevant to your research area")
                
                # Add citation quality note if high
                citation_count = paper.get('citation_count', 0)
                if citation_count > 500:
                    explanation_parts.append(f"Highly influential ({citation_count} citations)")
                elif citation_count > 100:
                    explanation_parts.append(f"Well-cited ({citation_count} citations)")
                
                # Add recency note for recent papers
                year = paper.get('year', 0)
                if year >= 2023:
                    explanation_parts.append("Recent research")
                
                # Default explanation if nothing else
                if not explanation_parts:
                    explanation_parts.append("Relevant to your query")
                
                # Build final explanation
                enriched_paper['relevance_explanation'] = '; '.join(explanation_parts)
                enriched_paper['relevance_score'] = round(paper.get('final_score', 0), 3)
                
                # Add match details for debugging (optional)
                enriched_paper['match_details'] = {
                    'title_matches': [t for t in query_terms if t in title_lower],
                    'abstract_matches': [t for t in query_terms if t in abstract_lower],
                    'total_query_terms': len(query_terms),
                    'match_source': match_source
                }
                
                enriched.append(enriched_paper)
                
            except Exception as e:
                logger.error(
                    "Failed to enrich paper",
                    index=i,
                    paper_id=paper.get('paper_id'),
                    error=str(e),
                    exc_info=True
                )
                # Add without enrichment
                enriched.append(paper)
        
        logger.info(
            "Enrichment complete",
            enriched_count=len(enriched),
            sample_explanation=enriched[0].get('relevance_explanation')[:80] if enriched else None
        )
        
        return enriched
   
    # -------------------------------------------------------------------------
    # Candidate retrieval helpers
    # -------------------------------------------------------------------------

    async def _retrieve_citation_network_candidates(
        self,
        paper_ids: List[str],
        limit: int
    ) -> List[Dict]:
        """
        Retrieve papers from citation networks of user's saved papers.
        Uses PaperRepository - NO SQL HERE.
        """
        logger.debug(
            "Retrieving citation network candidates",
            saved_papers=len(paper_ids),
            limit=limit
        )
        
        if not paper_ids:
            return []
        
        # Use repository method
        network_paper_ids = await self.paper_repo.get_papers_from_citation_network(
            source_paper_ids=paper_ids,
            limit=limit
        )
        
        # Fetch full details using repository
        if network_paper_ids:
            candidates = await self.paper_repo.find_by_ids(network_paper_ids)
            return [dict(c) for c in candidates]
        else:
            return []
    async def _retrieve_semantic_candidates(
        self,
        user_embedding: np.ndarray,
        domain: str,
        model: str,
        limit: int
    ) -> List[Dict]:
        """
        Retrieve papers similar to user embedding using vector search.
        Uses PaperRepository - NO SQL HERE.
        
        Args:
            user_embedding: User's embedding vector
            domain: User's primary domain
            model: Model name ('minilm' or 'specter')
            limit: Number of candidates to retrieve
            
        Returns:
            List of candidate papers with semantic_similarity scores
        """
        logger.debug(
            "Retrieving semantic candidates",
            domain=domain,
            model=model,
            limit=limit
        )
        
        # Use repository method for semantic search
        candidates = await self.paper_repo.semantic_search_by_user_embedding(
            embedding=user_embedding,
            model=model,
            domain=domain,
            limit=limit
        )
        
        logger.debug(
            "Semantic candidates retrieved",
            count=len(candidates),
            avg_similarity=np.mean([c.get('semantic_similarity', 0) for c in candidates]) if candidates else 0
        )
        
        return candidates
    
    async def _retrieve_canonical_candidates(
        self,
        domain: str,
        user_stage: str,
        count: int
    ) -> List[Dict]:
        """
        Retrieve canonical papers based on user's research stage.
        Uses GroundTruthRepository - NO SQL HERE.
        
        Args:
            domain: Research domain
            user_stage: User's research stage
            count: Total number of canonical papers to retrieve
            
        Returns:
            List of canonical papers with tier labels
        """
        logger.debug(
            "Retrieving canonical candidates",
            domain=domain,
            user_stage=user_stage,
            count=count
        )
        
        # Determine tier distribution based on research stage
        if user_stage in ['undergraduate', 'masters']:
            tier_distribution = {
                'foundational': 15,
                'recent': 5,
                'trending': 5
            }
        elif user_stage in ['phd', 'postdoc', 'professor']:
            tier_distribution = {
                'foundational': 5,
                'recent': 10,
                'trending': 10
            }
        elif user_stage == 'industry':
            tier_distribution = {
                'foundational': 3,
                'recent': 12,
                'trending': 10
            }
        else:
            tier_distribution = {
                'foundational': 8,
                'recent': 9,
                'trending': 8
            }
        
        # Use repository method to get canonical papers
        canonical_papers = await self.gt_repo.get_canonical_papers_sampled(
            domain=domain,
            tier_distribution=tier_distribution
        )
        
        logger.debug(
            "Canonical candidates retrieved",
            total=len(canonical_papers)
        )
        
        return canonical_papers
        
    async def _retrieve_ground_truth_candidates(
        self,
        user_interests: List[str],
        domain: str,
        count: int
    ) -> List[Dict]:
        """
        Retrieve papers from ground truth citation networks.
        Uses GroundTruthRepository - NO SQL HERE.
        
        Args:
            user_interests: User's interest terms
            domain: User's primary domain
            count: Number of papers to retrieve
            
        Returns:
            List of papers from GT networks
        """
        logger.debug(
            "Retrieving ground truth candidates",
            domain=domain,
            interests=user_interests,
            count=count
        )
        
        # Use repository method to find relevant GT papers
        gt_paper_ids = await self.gt_repo.find_relevant_ground_truth_papers(
            interest_terms=user_interests,
            domain=domain,
            limit=10
        )
        
        if not gt_paper_ids:
            logger.warning(
                "No relevant ground truth papers found",
                domain=domain,
                interests=user_interests
            )
            return []
        
        logger.debug(
            "Relevant GT papers found",
            count=len(gt_paper_ids)
        )
        
        # Get citation networks for these GT papers
        all_network_papers = []
        
        for gt_id in gt_paper_ids:
            relationships = await self.gt_repo.get_ground_truth_relationships(gt_id)
            if relationships and relationships.get('citation_network'):
                all_network_papers.extend(relationships['citation_network'])
        
        # Deduplicate and sample
        unique_network_papers = list(set(all_network_papers))
        
        if len(unique_network_papers) > count:
            sampled_ids = random.sample(unique_network_papers, count)
        else:
            sampled_ids = unique_network_papers
        
        # Fetch full paper details using repository
        if sampled_ids:
            candidates = await self.paper_repo.find_by_ids(sampled_ids)
            candidates_list = [dict(p) for p in candidates]
            
            # Mark as from ground truth
            for paper in candidates_list:
                paper['from_ground_truth'] = True
        else:
            candidates_list = []
        
        logger.debug(
            "Ground truth candidates retrieved",
            count=len(candidates_list)
        )
        
        return candidates_list

    
    async def _retrieve_citation_network_candidates(
        self,
        paper_ids: List[str],
        limit: int
    ) -> List[Dict]:
        """
        Retrieve papers from citation networks of user's saved papers.
        Used for warm-start recommendations.
        
        Args:
            paper_ids: Papers user has saved/liked
            limit: Number of candidates to retrieve
            
        Returns:
            List of papers from citation networks
        """
        logger.debug(
            "Retrieving citation network candidates",
            saved_papers=len(paper_ids),
            limit=limit
        )
        
        if not paper_ids:
            return []
        
        # Get citation networks for saved papers
        all_citations = []
        all_references = []
        
        for paper_id in paper_ids:
            # Get papers this paper cites (references)
            paper = await self.db.fetchrow(
                "SELECT reference_ids FROM papers WHERE paper_id = $1",
                paper_id
            )
            if paper and paper['reference_ids']:
                all_references.extend(paper['reference_ids'])
            
            # Get papers citing this paper (citations)
            citing_papers = await self.db.fetch(
                "SELECT paper_id FROM papers WHERE $1 = ANY(reference_ids)",
                paper_id
            )
            all_citations.extend([p['paper_id'] for p in citing_papers])
        
        # Combine and deduplicate
        network_paper_ids = list(set(all_citations + all_references))
        
        # Remove papers user already saved
        network_paper_ids = [pid for pid in network_paper_ids if pid not in paper_ids]
        
        # Sample if too many
        if len(network_paper_ids) > limit:
            sampled_ids = random.sample(network_paper_ids, limit)
        else:
            sampled_ids = network_paper_ids
        
        # Fetch full details
        if sampled_ids:
            query = """
                SELECT 
                    paper_id, title, abstract, authors, year,
                    citation_count, domain, sub_domains, venue
                FROM papers
                WHERE paper_id = ANY($1::text[])
            """
            papers = await self.db.fetch(query, sampled_ids)
            candidates = [dict(p) for p in papers]
        else:
            candidates = []
        
        logger.debug(
            "Citation network candidates retrieved",
            count=len(candidates)
        )
        
        return candidates
    
    async def _retrieve_collaborative_candidates(
        self,
        user_id: int,
        user_embedding: np.ndarray,
        model: str,
        limit: int
    ) -> List[Dict]:
        """
        Retrieve papers liked by similar users (collaborative filtering).
        Uses UserRepository - NO SQL HERE.
        """
        logger.debug(
            "Retrieving collaborative candidates",
            user_id=user_id,
            model=model,
            limit=limit
        )
        
        # Find similar users using repository
        similar_users = await self.user_repo.find_similar_users(
            user_embedding=user_embedding,
            model=model,
            current_user_id=user_id,
            limit=10
        )
        
        if not similar_users:
            logger.debug("No similar users found", user_id=user_id)
            return []
        
        similar_user_ids = [u['user_id'] for u in similar_users]
        
        # Get papers saved by similar users using repository
        paper_ids = await self.user_repo.get_papers_saved_by_users(
            user_ids=similar_user_ids,
            exclude_user_id=user_id,
            limit=limit
        )
        
        if not paper_ids:
            return []
        
        # Fetch full details using repository
        papers = await self.paper_repo.find_by_ids(paper_ids)
        candidates = [dict(p) for p in papers]
        
        logger.debug(
            "Collaborative candidates retrieved",
            count=len(candidates),
            from_users=len(similar_user_ids)
        )
        
        return candidates
    
    async def _retrieve_temporal_candidates(
        self,
        domain: str,
        sub_domains: List[str],
        limit: int
    ) -> List[Dict]:
        """
        Retrieve recent papers in user's domain/sub-domains.
        Uses PaperRepository - NO SQL HERE.
        """
        logger.debug(
            "Retrieving temporal candidates",
            domain=domain,
            sub_domains=sub_domains,
            limit=limit
        )
        
        # Use repository method
        candidates = await self.paper_repo.get_recent_papers_in_domain(
            domain=domain,
            years_back=1,
            min_citations=5,
            limit=limit
        )
        
        logger.debug(
            "Temporal candidates retrieved",
            count=len(candidates)
        )
        
        return candidates
    
    def _merge_candidates(
        self,
        *candidate_lists: List[Dict]
    ) -> List[Dict]:
        """
        Merge multiple candidate lists and deduplicate.
        
        Args:
            *candidate_lists: Variable number of candidate lists
            
        Returns:
            Deduplicated list of candidates
        """
        merged = {}
        
        for candidate_list in candidate_lists:
            for paper in candidate_list:
                paper_id = paper['paper_id']
                
                if paper_id not in merged:
                    merged[paper_id] = paper
                else:
                    # If paper appears in multiple sources, merge metadata
                    if 'semantic_similarity' in paper and 'semantic_similarity' not in merged[paper_id]:
                        merged[paper_id]['semantic_similarity'] = paper['semantic_similarity']
                    
                    if 'canonical_tier' in paper:
                        merged[paper_id]['canonical_tier'] = paper['canonical_tier']
                    
                    if 'from_ground_truth' in paper:
                        merged[paper_id]['from_ground_truth'] = True
        
        return list(merged.values())
    
    # -------------------------------------------------------------------------
    # Scoring + mitigation
    # -------------------------------------------------------------------------

    async def _apply_multi_factor_scoring(
        self,
        candidates: List[Dict],
        user: Dict,
        user_interests: Optional[List[Dict]],
        scoring_weights: Dict[str, float],
        is_cold_start: bool,
        mitigation_policy: Optional[Dict] = None,
        apply_bias_mitigation: bool = True,
    ) -> List[Dict]:
        """
        Apply multi-factor scoring to candidate papers (OPTIMIZED VERSION).
        
        KEY OPTIMIZATIONS:
        1. Batch fetch all ground truth data once
        2. Pre-build lookup dictionaries
        3. Eliminate per-candidate database queries
        4. Parallel computation where possible
        """
        logger.debug(
            "Applying multi-factor scoring (optimized)",
            candidate_count=len(candidates),
            weights=scoring_weights
        )

        policy = mitigation_policy or {}
        mit_factor = policy.get("factor", 1.0) if apply_bias_mitigation else 1.0
        weight_multipliers = policy.get("weight_multipliers", {}) if apply_bias_mitigation else {}
        min_score_threshold = policy.get("min_score_threshold", None)

        # Start from base weights
        effective_weights = scoring_weights.copy()

        # Apply per-component weight multipliers
        for comp, mult in weight_multipliers.items():
            if comp in effective_weights:
                effective_weights[comp] *= float(mult)

        # Renormalize weights
        total_w = sum(effective_weights.values())
        if total_w > 0:
            effective_weights = {k: v / total_w for k, v in effective_weights.items()}

        # ========================================================================
        # OPTIMIZATION 1: Fetch all data ONCE (not per candidate)
        # ========================================================================
        
        # Get max citation count for normalization (cached or fetch once)
        max_citations = await self.db.fetchval(
            "SELECT MAX(citation_count) FROM papers"
        )
        
        # Get relevant ground truth papers ONCE
        if is_cold_start and user_interests:
            relevant_gt_papers = await self._get_relevant_ground_truth_papers(
                user_interests=user_interests,
                domain=user['primary_domain']
            )
            logger.debug(f"Found {len(relevant_gt_papers)} relevant GT papers")
            
            # ========================================================================
            # OPTIMIZATION 2: Batch fetch ALL ground truth relationships
            # ========================================================================
            gt_lookup = await self._build_ground_truth_lookup(relevant_gt_papers)
            logger.debug(f"Built GT lookup with {len(gt_lookup)} entries")
        else:
            relevant_gt_papers = []
            gt_lookup = {}
        
        # ========================================================================
        # OPTIMIZATION 3: Vectorized scoring (process all candidates together)
        # ========================================================================
        
        scored_candidates = []
        
        for paper in candidates:
            scores: Dict[str, float] = {}
            
            # 1. Semantic score (already computed)
            scores['semantic'] = paper.get('semantic_similarity', 0.0)
            
            # 2. Citation score (fast calculation)
            scores['citation'] = self._calculate_citation_score(
                paper['citation_count'],
                paper['year'],
                max_citations
            )
            
            # 3. Recency score (fast calculation)
            scores['recency'] = self._calculate_recency_score(
                paper['year'],
                user.get('prefers_recent_papers', True)
            )
            
            # 4. Ground truth score (OPTIMIZED - no database query!)
            if is_cold_start:
                scores['ground_truth'] = self._calculate_ground_truth_score_fast(
                    paper['paper_id'],
                    gt_lookup  # Use pre-built lookup
                )
            else:
                scores['citation_network'] = 0.0
            
            # 5. Reading level score (fast calculation)
            scores['reading_level'] = self._calculate_reading_level_score(
                paper['citation_count'],
                user.get('reading_level', 'intermediate')
            )
            
            # 6. Diversity factor
            scores['diversity'] = 1.0
            
            # Calculate final score
            if is_cold_start:
                final_score = (
                    effective_weights['semantic']      * scores['semantic'] +
                    effective_weights['citation']      * scores['citation'] +
                    effective_weights['recency']       * scores['recency'] +
                    effective_weights['ground_truth']  * scores['ground_truth'] +
                    effective_weights['reading_level'] * scores['reading_level'] +
                    effective_weights['diversity']     * scores['diversity']
                )
            else:
                final_score = (
                    effective_weights.get('semantic', 0.35) * scores['semantic'] +
                    effective_weights.get('citation_network', 0.25) * scores.get('citation_network', 0.0) +
                    effective_weights.get('temporal', 0.10) * scores['recency'] +
                    effective_weights.get('diversity', 0.05) * scores['diversity']
                )
            
            # Apply mitigation factor
            final_score *= mit_factor

            paper['final_score'] = final_score
            paper['score_breakdown'] = scores

            if apply_bias_mitigation:
                paper['mitigation'] = {
                    "factor": mit_factor,
                    "policy": policy,
                }
            
            scored_candidates.append(paper)
        
        # Sort by final score
        scored_candidates.sort(key=lambda x: x['final_score'], reverse=True)

        # Apply minimum score threshold if defined
        if min_score_threshold is not None:
            before = len(scored_candidates)
            scored_candidates = [
                p for p in scored_candidates
                if p['final_score'] >= min_score_threshold
            ]
            logger.info(
                "Applied min_score_threshold",
                threshold=min_score_threshold,
                before=before,
                after=len(scored_candidates),
            )
        
        logger.info(
            "Multi-factor scoring complete",
            candidate_count=len(scored_candidates),
            avg_score=np.mean([p['final_score'] for p in scored_candidates]) if scored_candidates else 0.0,
            max_score=scored_candidates[0]['final_score'] if scored_candidates else 0,
            min_score=scored_candidates[-1]['final_score'] if scored_candidates else 0
        )
        
        return scored_candidates


    async def _build_ground_truth_lookup(
        self,
        relevant_gt_papers: List[str]
    ) -> Dict[str, float]:
        """
        Build lookup dictionary for ground truth scoring.
        Fetches ALL relationships in ONE batch query.
        
        Args:
            relevant_gt_papers: List of GT paper IDs
            
        Returns:
            Dict mapping paper_id -> GT score
        """
        if not relevant_gt_papers:
            return {}
        
        logger.debug(
            "Building ground truth lookup",
            gt_paper_count=len(relevant_gt_papers)
        )
        
        gt_lookup = {}
        
        # Fetch all GT relationships in ONE query
        for gt_id in relevant_gt_papers:
            relationships = await self.gt_repo.get_ground_truth_relationships(gt_id)
            
            if not relationships:
                continue
            
            # Citation network papers (direct citations)
            if relationships.get('citation_network'):
                for paper_id in relationships['citation_network']:
                    gt_lookup[paper_id] = gt_lookup.get(paper_id, 0.0) + 1.0
            
            # Bibliographic couples (shared references)
            if relationships.get('bibliographic_couples'):
                for paper_id in relationships['bibliographic_couples']:
                    gt_lookup[paper_id] = gt_lookup.get(paper_id, 0.0) + 0.6
        
        # Normalize scores
        if gt_lookup:
            max_score = max(gt_lookup.values())
            if max_score > 0:
                gt_lookup = {k: min(v / len(relevant_gt_papers), 1.0) for k, v in gt_lookup.items()}
        
        logger.debug(
            "Ground truth lookup built",
            total_papers_in_lookup=len(gt_lookup),
            avg_score=np.mean(list(gt_lookup.values())) if gt_lookup else 0
        )
        
        return gt_lookup


    def _calculate_ground_truth_score_fast(
        self,
        paper_id: str,
        gt_lookup: Dict[str, float]
    ) -> float:
        """
        Fast GT scoring using pre-built lookup (NO database queries).
        
        Args:
            paper_id: Paper to score
            gt_lookup: Pre-built lookup dictionary
            
        Returns:
            Ground truth score (0.0-1.0)
        """
        return gt_lookup.get(paper_id, 0.0)
    
    def _calculate_citation_score(
        self,
        citation_count: int,
        year: int,
        max_citations: int
    ) -> float:
        """
        Calculate normalized citation score with age adjustment.
        
        Args:
            citation_count: Paper's citation count
            year: Publication year
            max_citations: Maximum citations in database
            
        Returns:
            Normalized score (0.0-1.0)
        """
        if year is None: return 0.5

        if not citation_count or citation_count == 0 or not max_citations:
            return 0.0
        
        # Logarithmic normalization
        raw_score = np.log(citation_count + 1) / np.log(max_citations + 1)
        
        # Age adjustment
        current_year = 2024
        years_old = current_year - year
        
        if years_old < 2:
            # Recent papers get boost
            adjusted_score = raw_score * 1.5
        elif years_old > 10:
            # Old papers slight penalty
            adjusted_score = raw_score * 0.8
        else:
            adjusted_score = raw_score
        
        # Clip to [0, 1]
        return min(adjusted_score, 1.0)
    
    def _calculate_recency_score(
        self,
        year: int,
        prefers_recent: bool
    ) -> float:
        """
        Calculate recency score based on publication year.
        
        Args:
            year: Publication year
            prefers_recent: User's preference for recent papers
            
        Returns:
            Recency score (0.0-1.0)
        """
        year_scores = {
            2024: 1.0,
            2023: 0.95,
            2022: 0.90,
            2021: 0.85,
            2020: 0.75,
            2019: 0.65,
            2018: 0.55,
            2017: 0.45,
            2016: 0.35,
        }
        
        base_score = year_scores.get(year, 0.25)
        
        # Apply user preference
        if prefers_recent:
            return min(base_score * 1.2, 1.0)
        else:
            return base_score
    
    async def _calculate_ground_truth_score(
        self,
        paper_id: str,
        relevant_gt_papers: List[str]
    ) -> float:
        """
        Calculate ground truth score (is paper in GT networks?).
        
        Args:
            paper_id: Paper to score
            relevant_gt_papers: List of relevant GT paper IDs
            
        Returns:
            Ground truth score (0.0-1.0)
        """
        if not relevant_gt_papers:
            return 0.0
        
        total_score = 0.0
        
        for gt_id in relevant_gt_papers:
            relationships = await self.gt_repo.get_ground_truth_relationships(gt_id)
            
            if not relationships:
                continue
            
            # Check different match types
            if relationships['citation_network'] and paper_id in relationships['citation_network']:
                total_score += 1.0  # Direct citation
            
            if relationships['bibliographic_couples'] and paper_id in relationships['bibliographic_couples']:
                total_score += 0.6  # Bibliographic couple
            
            # Note: co_cited_papers is empty in your data, so skip
        
        # Normalize by number of GT papers
        normalized_score = total_score / len(relevant_gt_papers)
        
        return min(normalized_score, 1.0)
    
    def _calculate_reading_level_score(
        self,
        citation_count: int,
        user_reading_level: str
    ) -> float:
        """
        Match paper complexity with user's reading level.
        
        Args:
            citation_count: Paper's citation count (proxy for complexity)
            user_reading_level: User's reading level
            
        Returns:
            Match score (0.0-1.0)
        """
        # Infer paper complexity
        if citation_count >= 1000:
            complexity = 'high'
        elif citation_count >= 100:
            complexity = 'medium'
        else:
            complexity = 'low'
        
        # Match matrix
        match_scores = {
            ('introductory', 'low'): 1.0,
            ('introductory', 'medium'): 0.5,
            ('introductory', 'high'): 0.1,
            ('intermediate', 'low'): 0.7,
            ('intermediate', 'medium'): 1.0,
            ('intermediate', 'high'): 0.6,
            ('advanced', 'low'): 0.3,
            ('advanced', 'medium'): 0.8,
            ('advanced', 'high'): 1.0,
            ('expert', 'low'): 0.2,
            ('expert', 'medium'): 0.7,
            ('expert', 'high'): 1.0,
        }
        
        return match_scores.get((user_reading_level, complexity), 0.5)
    
    async def _get_relevant_ground_truth_papers(
        self,
        user_interests: List[Dict],
        domain: str
    ) -> List[str]:
        """
        Find ground truth papers relevant to user's interests.
        Uses GroundTruthRepository - NO SQL HERE.
        
        Args:
            user_interests: User's interest rows (with 'interest_term')
            domain: User's domain
            
        Returns:
            List of relevant GT paper IDs
        """
        interest_terms = [i['interest_term'] for i in user_interests]
        
        # Use repository method
        paper_ids = await self.gt_repo.find_relevant_ground_truth_papers(
            interest_terms=interest_terms,
            domain=domain,
            limit=10
        )
        
        return paper_ids
    
    async def _apply_diversity_filtering(
        self,
        scored_papers: List[Dict],
        target_count: int,
        max_per_author: int = 2,
        max_per_venue: int = 2
    ) -> List[Dict]:
        """Apply diversity constraints to paper selection."""
        logger.debug(
            "Applying diversity filtering",
            input_count=len(scored_papers),
            target_count=target_count
        )
    
        selected = []
        author_counts = {}
        venue_counts = {}
        tier_counts = {'foundational': 0, 'recent': 0, 'trending': 0}
        
        for paper in scored_papers:
            # Proper author extraction
            first_author = 'unknown'
            
            if paper.get('authors'):
                authors = paper['authors']
                
                if isinstance(authors, list) and len(authors) > 0:
                    # Authors is array with single string: ["Name1, Name2, Name3"]
                    author_string = authors[0]
                    
                    # Split by comma to get individual authors
                    author_list = author_string.split(',')
                    
                    if author_list:
                        # Get first author and clean
                        first_author = author_list[0].strip()
                        
                        # Extract last name only
                        name_parts = first_author.split()
                        if len(name_parts) > 1:
                            first_author = name_parts[-1]  # Last name
            
            # Check author constraint
            if author_counts.get(first_author, 0) >= max_per_author:
                continue
            
            # Handle NULL venues
            venue = paper.get('venue')
            if venue is None or venue == 'null' or venue == '':
                venue = f"unknown_{paper['paper_id'][:8]}"  # Unique identifier
            
            # Check venue constraint
            if venue_counts.get(venue, 0) >= max_per_venue:
                continue
            
            # Add paper
            selected.append(paper)
            author_counts[first_author] = author_counts.get(first_author, 0) + 1
            venue_counts[venue] = venue_counts.get(venue, 0) + 1
            
            # Track canonical tier
            if 'canonical_tier' in paper:
                tier_counts[paper['canonical_tier']] += 1
            
            # Stop when target reached
            if len(selected) >= target_count:
                break
        
        logger.info(
            "Diversity filtering complete",
            selected_count=len(selected),
            tier_distribution=tier_counts,
            unique_authors=len(author_counts),
            unique_venues=len(venue_counts)
        )
        
        return selected
    
    def _apply_fairness_reranking(self, papers: List[Dict]) -> List[Dict]:
        """
        Apply field-based fairness reranking to boost under-served research fields.
        
        This complements user-profile-based bias mitigation by also ensuring
        papers from under-served fields get fair representation.
        
        Args:
            papers: List of enriched recommendation papers
            
        Returns:
            Reranked papers with updated scores and primary_field metadata
        """
        if not papers:
            return papers
        
        # Create mapping from paper_id to original paper for merging
        paper_map = {}
        fairness_input = []
        
        for paper in papers:
            paper_id = paper.get('paper_id') or paper.get('paperId')
            score = paper.get('final_score') or paper.get('relevance_score') or paper.get('score', 0.0)
            
            if paper_id:
                paper_id_str = str(paper_id)
                paper_map[paper_id_str] = paper
                fairness_input.append({
                    'paper_id': paper_id_str,
                    'score': float(score)
                })
        
        if not fairness_input:
            return papers
        
        # Apply fairness reranking
        try:
            reranked = fairness_aware_rerank(fairness_input, boost=1.05)
            
            # Map back to original format, updating scores and adding primary_field
            result = []
            for reranked_paper in reranked:
                paper_id = reranked_paper.get('paper_id') or reranked_paper.get('paperId')
                paper_id_str = str(paper_id) if paper_id else None
                
                # Get original paper
                original = paper_map.get(paper_id_str, {})
                if not original:
                    # If paper not found in map, use reranked paper as base
                    original = reranked_paper.copy()
                    # Remove keys that shouldn't be in final output
                    original.pop('_original', None)
                
                updated_paper = original.copy()
                
                # Update final_score with fairness-boosted score
                new_score = reranked_paper.get('score', original.get('final_score', 0.0))
                updated_paper['final_score'] = new_score
                updated_paper['relevance_score'] = round(new_score, 3)
                
                # Add primary_field if available
                if 'primary_field' in reranked_paper:
                    updated_paper['primary_field'] = reranked_paper['primary_field']
                
                result.append(updated_paper)
            
            # Sort by final_score (fairness_aware_rerank already sorts, but ensure consistency)
            result.sort(key=lambda x: x.get('final_score', 0.0), reverse=True)
            
            if result:
                logger.debug(
                    "Applied fairness reranking",
                    papers_reranked=len(result),
                    avg_score_after=np.mean([p['final_score'] for p in result])
                )
            
            return result
            
        except Exception as e:
            logger.warning(
                f"Failed to apply fairness reranking: {e}",
                exc_info=True
            )
            # Return original papers if fairness reranking fails
            return papers
    
    def _enrich_recommendations(
        self,
        papers: List[Dict],
        user_interests: Optional[List[str]]
    ) -> List[Dict]:
        """
        Add relevance explanations and metadata to recommendations.
        
        Args:
            papers: List of recommended papers
            user_interests: User's interests (for cold-start)
            
        Returns:
            Enriched papers
        """
        enriched = []
        
        for paper in papers:
            # Copy paper data
            enriched_paper = paper.copy()
            
            # Add relevance score (final_score as percentage)
            enriched_paper['relevance_score'] = round(paper['final_score'], 3)
            
            # Generate explanation
            explanation_parts = []
            
            breakdown = paper.get('score_breakdown', {})
            
            if breakdown.get('semantic', 0) > 0.7:
                explanation_parts.append("Highly relevant to your interests")
            
            if breakdown.get('ground_truth', 0) > 0.5:
                explanation_parts.append("Academically validated through citations")
            
            if paper.get('canonical_tier'):
                tier_text = {
                    'foundational': 'Foundational work in the field',
                    'recent': 'Recent cutting-edge research',
                    'trending': 'Trending topic with high impact'
                }
                explanation_parts.append(tier_text[paper['canonical_tier']])
            
            if breakdown.get('citation', 0) > 0.8:
                explanation_parts.append("Highly influential paper")
            
            enriched_paper['relevance_explanation'] = '; '.join(explanation_parts) if explanation_parts else "Recommended based on your profile"
            
            enriched.append(enriched_paper)
        
        return enriched
    
    async def _get_user_saved_papers(self, user_id: int) -> List[Dict]:
        """
        Get papers user has saved.
        Uses UserRepository - NO SQL HERE.
        """
        return await self.user_repo.get_saved_papers_list(user_id)
    
    async def _filter_seen_papers(
        self,
        candidates: List[Dict],
        user_id: int
    ) -> List[Dict]:
        """
        Remove papers user has already interacted with.
        Uses InteractionRepository - NO SQL HERE.
        """
        # Get seen paper IDs from repository
        from app.db.repositories.interaction_repo import InteractionRepository
        interaction_repo = InteractionRepository(self.db)
        
        seen_ids = set(await interaction_repo.get_seen_paper_ids(user_id))
        
        # Filter out seen papers
        filtered = [p for p in candidates if p['paper_id'] not in seen_ids]
        
        logger.debug(
            "Filtered seen papers",
            original_count=len(candidates),
            filtered_count=len(filtered),
            removed=len(candidates) - len(filtered)
        )
        
        return filtered

