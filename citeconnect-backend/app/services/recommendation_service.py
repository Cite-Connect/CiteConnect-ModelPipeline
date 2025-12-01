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
    - Bias mitigation via configurable slice rules
    """
    
    # Default scoring weights (tunable hyperparameters)
    DEFAULT_COLD_START_WEIGHTS = {
        'semantic': 0.40,
        'citation': 0.20,
        'recency': 0.15,
        'ground_truth': 0.10,
        'reading_level': 0.10,
        'diversity': 0.05
    }
    
    DEFAULT_WARM_START_WEIGHTS = {
        'semantic': 0.35,
        'citation_network': 0.25,
        'collaborative': 0.15,
        'temporal': 0.10,
        'venue': 0.10,
        'diversity': 0.05
    }
    
    # Retrieval limits
    SEMANTIC_CANDIDATE_LIMIT = 150
    CANONICAL_CANDIDATE_COUNT = 25
    GT_NETWORK_CANDIDATE_COUNT = 25
    
    def __init__(self, db: DatabaseConnection):
        """
        Initialize recommendation service.
        
        Args:
            db: Database connection
        """
        self.db = db
        self.user_repo = UserRepository(db)
        self.paper_repo = PaperRepository(db)
        self.gt_repo = GroundTruthRepository(db)
        self.user_embedding_service = UserEmbeddingService(db)

        # Load bias mitigation configuration (if present)
        self.bias_config = self._load_bias_config()
        
        logger.info("RecommendationService initialized")

    # -------------------------------------------------------------------------
    # Bias mitigation config helpers
    # -------------------------------------------------------------------------

    def _load_bias_config(self) -> Dict:
        """
        Load bias mitigation config from JSON, if present.

        Expected path (from backend root):
        citeconnect-backend/bias_config/bias_mitigation_config.json
        """
        try:
            config_path = (
                Path(__file__).parent.parent.parent
                / "bias_config"
                / "bias_mitigation_config.json"
            )
            if not config_path.exists():
                logger.info(
                    "No bias mitigation config found – running without mitigation",
                    path=str(config_path),
                )
                return {}
            with config_path.open("r") as f:
                cfg = json.load(f)
            logger.info(
                "Loaded bias mitigation config",
                path=str(config_path),
            )
            return cfg
        except Exception as e:
            logger.warning(f"Failed to load bias mitigation config: {e}")
            return {}

    def _get_mitigation_policy_for_profile(self, profile: Dict) -> Dict:
        """
        Compute mitigation policy for this user from bias_config.

        Returns:
            {
              "factor": float,
              "weight_multipliers": {"semantic": 1.0, "ground_truth": 1.2, ...},
              "min_score_threshold": float | None,
              "applied_rules": [...]
            }
        """
        policy = {
            "factor": 1.0,
            "weight_multipliers": {},
            "min_score_threshold": None,
            "applied_rules": [],
        }

        if not self.bias_config:
            return policy

        cfg = self.bias_config
        slice_rules = cfg.get("slice_rules", {})

        # Global default threshold if provided
        global_th = cfg.get("global_min_score_threshold")
        if global_th is not None:
            policy["min_score_threshold"] = float(global_th)

        # For each slicing field, see if this profile matches a rule
        for field, field_rules in slice_rules.items():
            user_value = profile.get(field)
            if not user_value:
                continue

            rules_for_value = field_rules.get(user_value)
            if not rules_for_value:
                continue

            # 1) Score multiplier
            sf = rules_for_value.get("score_factor")
            if sf is not None:
                policy["factor"] *= float(sf)

            # 2) Per-component weight multipliers
            for comp, mult in rules_for_value.get("weight_multipliers", {}).items():
                current = policy["weight_multipliers"].get(comp, 1.0)
                policy["weight_multipliers"][comp] = current * float(mult)

            # 3) Threshold override – pick the most lenient (lowest)
            if "min_score_threshold" in rules_for_value:
                th = float(rules_for_value["min_score_threshold"])
                if policy["min_score_threshold"] is None:
                    policy["min_score_threshold"] = th
                else:
                    policy["min_score_threshold"] = min(policy["min_score_threshold"], th)

            policy["applied_rules"].append(
                {"field": field, "value": user_value, "config": rules_for_value}
            )

        return policy

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
        
        Args:
            user_id: User identifier
            count: Number of recommendations to return
            model: Embedding model ('minilm' or 'specter')
            scoring_weights: Optional custom weights
            
        Returns:
            Dict with recommendations and metadata
        """
        logger.info(
            "Generating recommendations",
            user_id=user_id,
            count=count,
            model=model
        )
        
        # Get user's recommendation state
        state = await self.user_repo.get_recommendation_state(user_id)
        
        if not state:
            raise ValueError(f"No recommendation state found for user {user_id}")
        
        # Route based on stage
        stage = state['recommendation_stage']
        interaction_count = state['interaction_count']
        
        if interaction_count < 10:
            logger.info(
                "Using cold-start strategy",
                user_id=user_id,
                stage=stage
            )
            return await self.generate_cold_start_recommendations(
                user_id=user_id,
                count=count,
                model=model,
                scoring_weights=scoring_weights
            )
        else:
            logger.info(
                "Using warm-start strategy",
                user_id=user_id,
                stage=stage,
                interactions=interaction_count
            )
            return await self.generate_warm_start_recommendations(
                user_id=user_id,
                count=count,
                model=model,
                scoring_weights=scoring_weights
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
        Uses profile-based embeddings and ground truth boost.
        
        Args:
            user_id: User identifier
            count: Number of recommendations
            model: Embedding model to use
            scoring_weights: Optional custom weights
            
        Returns:
            Dict with recommendations and metadata
        """
        logger.info(
            "Generating cold-start recommendations",
            user_id=user_id,
            count=count,
            model=model
        )
        
        # Use default weights if not provided
        weights = scoring_weights or self.DEFAULT_COLD_START_WEIGHTS
        
        # Step 1: Get user data
        profile = await self.user_repo.get_profile(user_id)
        if not profile:
            raise ValueError(f"No profile found for user {user_id}")
        
        interests = await self.user_repo.get_user_interests(user_id)

        # NEW: compute mitigation policy for this user
        mitigation_policy = self._get_mitigation_policy_for_profile(profile)
        
        # Step 2: Get user embedding
        embeddings = await self.user_embedding_service.get_or_generate_user_embeddings(user_id)
        user_embedding = embeddings[model]
        
        # Step 3: Retrieve candidates (3 strategies)
        logger.debug("Retrieving candidates", user_id=user_id)
        
        # 3A: Semantic search
        semantic_candidates = await self._retrieve_semantic_candidates(
            user_embedding=user_embedding,
            domain=profile['primary_domain'],
            model=model,
            limit=self.SEMANTIC_CANDIDATE_LIMIT
        )
        
        # 3B: Canonical papers
        canonical_candidates = await self._retrieve_canonical_candidates(
            domain=profile['primary_domain'],
            user_stage=profile.get('research_stage', 'phd'),
            count=self.CANONICAL_CANDIDATE_COUNT
        )
        
        # 3C: Ground truth network papers
        gt_candidates = await self._retrieve_ground_truth_candidates(
            user_interests=[i['interest_term'] for i in interests],
            domain=profile['primary_domain'],
            count=self.GT_NETWORK_CANDIDATE_COUNT
        )
        
        # Merge and deduplicate
        all_candidates = self._merge_candidates(
            semantic_candidates,
            canonical_candidates,
            gt_candidates
        )
        
        logger.info(
            "Candidates retrieved",
            user_id=user_id,
            total_candidates=len(all_candidates),
            semantic=len(semantic_candidates),
            canonical=len(canonical_candidates),
            ground_truth=len(gt_candidates)
        )
        
        # Step 4: Apply multi-factor scoring (WITH mitigation)
        scored_papers = await self._apply_multi_factor_scoring(
            candidates=all_candidates,
            user=profile,
            user_interests=interests,
            scoring_weights=weights,
            is_cold_start=True,
            mitigation_policy=mitigation_policy,
            apply_bias_mitigation=True,
        )
        
        # Step 5: Apply diversity filtering and select top N
        # We want 21 papers for clustering (3 clusters × 7 papers)
        diverse_papers = await self._apply_diversity_filtering(
            scored_papers=scored_papers,
            target_count=21,
            max_per_author=2,
            max_per_venue=2
        )
        
        # Step 6: Take top 'count' papers
        final_recommendations = diverse_papers[:count]
        
        # Step 7: Enrich with explanations
        enriched = self._enrich_recommendations(
            papers=final_recommendations,
            user_interests=[i['interest_term'] for i in interests]
        )
        
        logger.info(
            "Cold-start recommendations generated",
            user_id=user_id,
            count=len(enriched),
            avg_score=np.mean([p['final_score'] for p in enriched]) if enriched else 0.0
        )
        
        return {
            'user_id': user_id,
            'papers': enriched,
            'method': 'cold_start',
            'model_used': model,
            'scoring_weights': weights,
            'generated_at': datetime.utcnow().isoformat(),
            'total_candidates': len(all_candidates),
            'mitigation_policy': mitigation_policy,
        }
    
    async def generate_warm_start_recommendations(
        self,
        user_id: int,
        count: int = 10,
        model: str = 'minilm',
        scoring_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Generate recommendations for warm-start users (10+ interactions).
        Uses interaction-based embeddings and citation networks.
        
        Args:
            user_id: User identifier
            count: Number of recommendations
            model: Embedding model to use
            scoring_weights: Optional custom weights
            
        Returns:
            Dict with recommendations and metadata
        """
        logger.info(
            "Generating warm-start recommendations",
            user_id=user_id,
            count=count,
            model=model
        )
        
        # Use default weights if not provided
        weights = scoring_weights or self.DEFAULT_WARM_START_WEIGHTS
        
        # Get user data
        profile = await self.user_repo.get_profile(user_id)

        # NEW: mitigation policy for warm-start users too
        mitigation_policy = self._get_mitigation_policy_for_profile(profile)
        
        # Get user's interaction history
        saved_papers = await self._get_user_saved_papers(user_id)
        
        # Get user embedding (should be interaction-based or hybrid)
        embeddings = await self.user_embedding_service.get_or_generate_user_embeddings(user_id)
        user_embedding = embeddings[model]
        
        # Retrieve candidates (4 strategies for warm-start)
        logger.debug("Retrieving warm-start candidates", user_id=user_id)
        
        # Strategy A: Semantic (200 candidates)
        semantic_candidates = await self._retrieve_semantic_candidates(
            user_embedding=user_embedding,
            domain=profile['primary_domain'],
            model=model,
            limit=200
        )
        
        # Strategy B: Citation network (50 candidates)
        citation_candidates = await self._retrieve_citation_network_candidates(
            paper_ids=[p['paper_id'] for p in saved_papers],
            limit=50
        )
        
        # Strategy C: Collaborative filtering (30 candidates)
        # Find similar users and get their saved papers
        collaborative_candidates = await self._retrieve_collaborative_candidates(
            user_id=user_id,
            user_embedding=user_embedding,
            model=model,
            limit=30
        )
        
        # Strategy D: Temporal (20 candidates)
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
        
        # Filter out papers user already interacted with
        all_candidates = await self._filter_seen_papers(all_candidates, user_id)
        
        logger.info(
            "Warm-start candidates retrieved",
            user_id=user_id,
            total=len(all_candidates)
        )
        
        # Apply scoring (WITH mitigation)
        scored_papers = await self._apply_multi_factor_scoring(
            candidates=all_candidates,
            user=profile,
            user_interests=None,  # Not used for warm-start
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
        
        # Select top N
        final_recommendations = diverse_papers[:count]
        
        # Enrich
        enriched = self._enrich_recommendations(
            papers=final_recommendations,
            user_interests=None
        )
        
        logger.info(
            "Warm-start recommendations generated",
            user_id=user_id,
            count=len(enriched)
        )
        
        return {
            'user_id': user_id,
            'papers': enriched,
            'method': 'warm_start',
            'model_used': model,
            'scoring_weights': weights,
            'generated_at': datetime.utcnow().isoformat(),
            'total_candidates': len(all_candidates),
            'mitigation_policy': mitigation_policy,
        }
    
    # -------------------------------------------------------------------------
    # Candidate retrieval helpers
    # -------------------------------------------------------------------------

    async def _retrieve_semantic_candidates(
        self,
        user_embedding: np.ndarray,
        domain: str,
        model: str,
        limit: int
    ) -> List[Dict]:
        """
        Retrieve papers similar to user embedding using vector search.
        
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
        
        # Determine table based on model
        embedding_table = f'paper_embeddings_{model}'
        
        # Convert embedding to PostgreSQL vector string
        embedding_str = user_embedding.tolist()
        
        # Vector similarity search
        query = f"""
            WITH user_emb AS (
                SELECT $1::vector as emb
            )
            SELECT 
                p.paper_id,
                p.title,
                p.abstract,
                p.authors,
                p.year,
                p.citation_count,
                p.domain,
                p.sub_domains,
                p.venue,
                1 - (pe.embedding <=> ue.emb) as semantic_similarity
            FROM papers p
            JOIN {embedding_table} pe ON p.paper_id = pe.paper_id
            CROSS JOIN user_emb ue
            WHERE p.domain = $2
            ORDER BY pe.embedding <=> ue.emb
            LIMIT $3
        """
        
        results = await self.db.fetch(query, embedding_str, domain, limit)
        
        candidates = [dict(r) for r in results]
        
        logger.debug(
            "Semantic candidates retrieved",
            count=len(candidates),
            avg_similarity=np.mean([c['semantic_similarity'] for c in candidates]) if candidates else 0
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
            # Focus on fundamentals
            tier_distribution = {
                'foundational': 15,
                'recent': 5,
                'trending': 5
            }
        elif user_stage in ['phd', 'postdoc', 'professor']:
            # Focus on cutting-edge
            tier_distribution = {
                'foundational': 5,
                'recent': 10,
                'trending': 10
            }
        elif user_stage == 'industry':
            # Focus on practical and trending
            tier_distribution = {
                'foundational': 3,
                'recent': 12,
                'trending': 10
            }
        else:
            # Default balanced
            tier_distribution = {
                'foundational': 8,
                'recent': 9,
                'trending': 8
            }
        
        # Retrieve canonical papers for each tier
        canonical_papers = []
        
        for tier, tier_count in tier_distribution.items():
            # Get paper IDs for this tier
            query = """
                SELECT paper_ids
                FROM domain_canonical_papers
                WHERE domain = $1
                  AND recommendation_tier = $2
            """
            
            result = await self.db.fetchrow(query, domain, tier)
            
            if result and result['paper_ids']:
                paper_ids = result['paper_ids']
                
                # Sample requested count (or all if fewer available)
                sample_size = min(tier_count, len(paper_ids))
                sampled_ids = random.sample(paper_ids, sample_size)
                
                # Fetch full paper details
                papers_query = """
                    SELECT 
                        paper_id,
                        title,
                        abstract,
                        authors,
                        year,
                        citation_count,
                        domain,
                        sub_domains,
                        venue
                    FROM papers
                    WHERE paper_id = ANY($1::text[])
                """
                
                papers = await self.db.fetch(papers_query, sampled_ids)
                
                # Add tier label
                for paper in papers:
                    paper_dict = dict(paper)
                    paper_dict['canonical_tier'] = tier
                    canonical_papers.append(paper_dict)
                
                logger.debug(
                    "Canonical tier retrieved",
                    tier=tier,
                    count=len(papers)
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
        # Find relevant ground truth papers
        interest_patterns = ' OR '.join([
            f"p.title ILIKE '%{interest}%' OR p.abstract ILIKE '%{interest}%'"
            for interest in user_interests
        ])
        print(f"Interest patterns: {interest_patterns}")
        
        query = f"""
            SELECT p.paper_id
            FROM papers p
            INNER JOIN ground_truth_papers gtp ON p.paper_id = gtp.paper_id
            WHERE p.domain = $1
              AND ({interest_patterns})
            LIMIT 10
        """
        
        print(f"Executing GT paper query: {query}")
        
        gt_papers = await self.db.fetch(query, domain)
        print(f"GT papers fetched: {gt_papers}")
        if not gt_papers:
            logger.warning(
                "No relevant ground truth papers found",
                domain=domain,
                interests=user_interests
            )
            return []
        
        gt_paper_ids = [p['paper_id'] for p in gt_papers]
        print(f"Relevant GT paper IDs: {gt_paper_ids}")
        logger.debug(
            "Relevant GT papers found",
            count=len(gt_paper_ids)
        )
        
        # Get citation networks for these GT papers
        all_network_papers = []
        
        for gt_id in gt_paper_ids:
            relationships = await self.gt_repo.get_ground_truth_relationships(gt_id)
            if relationships and relationships['citation_network']:
                all_network_papers.extend(relationships['citation_network'])
        # Deduplicate
        unique_network_papers = list(set(all_network_papers))
        # Sample requested count
        if len(unique_network_papers) > count:
            sampled_ids = random.sample(unique_network_papers, count)
            print(f"Sampled GT network paper IDs: {sampled_ids}")
        else:
            sampled_ids = unique_network_papers
        
        # Fetch full paper details
        if sampled_ids:
            query = """
                SELECT 
                    paper_id,
                    title,
                    abstract,
                    authors,
                    year,
                    citation_count,
                    domain,
                    sub_domains,
                    venue
                FROM papers
                WHERE paper_id = ANY($1::text[])
            """
            papers = await self.db.fetch(query, sampled_ids)

            candidates = [dict(p) for p in papers]
            
            # Mark as from ground truth
            for paper in candidates:
                paper['from_ground_truth'] = True
        else:
            candidates = []
        
        logger.debug(
            "Ground truth candidates retrieved",
            count=len(candidates)
        )
        
        return candidates
    
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
        
        Args:
            user_id: Current user
            user_embedding: User's embedding
            model: Model name
            limit: Number of candidates
            
        Returns:
            List of papers from similar users
        """
        logger.debug(
            "Retrieving collaborative candidates",
            user_id=user_id,
            model=model,
            limit=limit
        )
        
        # Find similar users
        embedding_str = '[' + ','.join(map(str, user_embedding.tolist())) + ']'
        embedding_table = f'user_embeddings_{model}'
        
        query = f"""
            SELECT 
                user_id,
                1 - (embedding <=> $1::vector) as similarity
            FROM {embedding_table}
            WHERE user_id != $2
            ORDER BY embedding <=> $1::vector
            LIMIT 10
        """
        
        similar_users = await self.db.fetch(query, embedding_str, user_id)
        
        if not similar_users:
            return []
        
        similar_user_ids = [u['user_id'] for u in similar_users]
        
        # Get papers saved by similar users
        query = """
            SELECT DISTINCT paper_id
            FROM user_saved_papers
            WHERE user_id = ANY($1::int[])
              AND paper_id NOT IN (
                SELECT paper_id FROM user_saved_papers WHERE user_id = $2
              )
            LIMIT $3
        """
        
        paper_ids = await self.db.fetch(query, similar_user_ids, user_id, limit)
        
        if not paper_ids:
            return []
        
        # Fetch full details
        query = """
            SELECT paper_id, title, abstract, authors, year,
                   citation_count, domain, sub_domains, venue
            FROM papers
            WHERE paper_id = ANY($1::text[])
        """
        
        papers = await self.db.fetch(
            query,
            [p['paper_id'] for p in paper_ids]
        )
        
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
        
        Args:
            domain: Primary domain
            sub_domains: User's sub-domains
            limit: Number of papers
            
        Returns:
            List of recent papers
        """
        logger.debug(
            "Retrieving temporal candidates",
            domain=domain,
            sub_domains=sub_domains,
            limit=limit
        )
        
        # Get papers from last 6 months
        query = """
            SELECT paper_id, title, abstract, authors, year,
                   citation_count, domain, sub_domains, venue
            FROM papers
            WHERE domain = $1
              AND year >= EXTRACT(YEAR FROM CURRENT_DATE) - 1
              AND citation_count >= 5
            ORDER BY citation_count DESC
            LIMIT $2
        """
        
        papers = await self.db.fetch(query, domain, limit)
        candidates = [dict(p) for p in papers]
        
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
        Apply multi-factor scoring to candidate papers, with optional bias mitigation.

        Args:
            candidates: List of candidate papers
            user: User profile
            user_interests: User's interests (for cold-start)
            scoring_weights: Base weights for each scoring component
            is_cold_start: Whether this is cold-start mode
            mitigation_policy: Per-user mitigation policy (slice-based)
            apply_bias_mitigation: Toggle mitigation on/off
            
        Returns:
            List of papers with final_score and score_breakdown
        """
        logger.debug(
            "Applying multi-factor scoring",
            candidate_count=len(candidates),
            weights=scoring_weights
        )

        policy = mitigation_policy or {}
        mit_factor = policy.get("factor", 1.0) if apply_bias_mitigation else 1.0
        weight_multipliers = policy.get("weight_multipliers", {}) if apply_bias_mitigation else {}
        min_score_threshold = policy.get("min_score_threshold", None)

        # Start from base weights
        effective_weights = scoring_weights.copy()

        # Apply per-component weight multipliers for this slice
        for comp, mult in weight_multipliers.items():
            if comp in effective_weights:
                effective_weights[comp] *= float(mult)

        # Optional: renormalize so weights sum to 1 (keeps global scale stable)
        total_w = sum(effective_weights.values())
        if total_w > 0:
            effective_weights = {k: v / total_w for k, v in effective_weights.items()}

        # Get max citation count for normalization
        max_citations = await self.db.fetchval(
            "SELECT MAX(citation_count) FROM papers"
        )
        
        # Get relevant ground truth papers for this user (cold-start only)
        if is_cold_start and user_interests:
            relevant_gt_papers = await self._get_relevant_ground_truth_papers(
                user_interests=user_interests,
                domain=user['primary_domain']
            )
            print(f"Relevant GT papers for scoring: {relevant_gt_papers}")
        else:
            relevant_gt_papers = []
        
        # Score each candidate
        scored_candidates = []
        
        for paper in candidates:
            scores: Dict[str, float] = {}
            
            # 1. Semantic score
            scores['semantic'] = paper.get('semantic_similarity', 0.0)
            
            # 2. Citation score
            scores['citation'] = self._calculate_citation_score(
                paper['citation_count'],
                paper['year'],
                max_citations
            )
            
            # 3. Recency score
            scores['recency'] = self._calculate_recency_score(
                paper['year'],
                user.get('prefers_recent_papers', True)
            )
            
            # 4. Ground truth score (cold-start) or citation network (warm-start)
            if is_cold_start:
                scores['ground_truth'] = await self._calculate_ground_truth_score(
                    paper['paper_id'],
                    relevant_gt_papers
                )
            else:
                # For warm-start, this becomes citation_network score (placeholder)
                scores['citation_network'] = 0.0
            
            # 5. Reading level score
            scores['reading_level'] = self._calculate_reading_level_score(
                paper['citation_count'],
                user.get('reading_level', 'intermediate')
            )
            
            # 6. Diversity factor (placeholder)
            scores['diversity'] = 1.0
            
            # Calculate final score using effective_weights + mitigation factor
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
                    effective_weights.get('semantic', scoring_weights['semantic']) *
                        scores['semantic'] +
                    effective_weights.get('citation_network', scoring_weights.get('citation_network', 0.25)) *
                        scores.get('citation_network', 0.0) +
                    effective_weights.get('temporal', scoring_weights.get('temporal', 0.10)) *
                        scores['recency'] +
                    effective_weights.get('diversity', scoring_weights['diversity']) *
                        scores['diversity']
                    # Note: collaborative / venue left as placeholders
                )
            
            # Apply global/slice multiplier
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
        
        Args:
            user_interests: User's interest rows (with 'interest_term')
            domain: User's domain
            
        Returns:
            List of relevant GT paper IDs
        """
        interest_terms = [i['interest_term'] for i in user_interests]
        
        # Build ILIKE patterns
        conditions = ' OR '.join([
            f"p.title ILIKE '%{term}%' OR p.abstract ILIKE '%{term}%'"
            for term in interest_terms
        ])
        
        query = f"""
            SELECT p.paper_id
            FROM papers p
            INNER JOIN ground_truth_papers gtp ON p.paper_id = gtp.paper_id
            WHERE p.domain = $1
              AND ({conditions})
            LIMIT 10
        """
        print(f"GT interest query: {query}")
        results = await self.db.fetch(query, domain)
        print(f"GT interest results: {results}")
        
        return [r['paper_id'] for r in results]
    
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
        """Get papers user has saved."""
        query = """
            SELECT p.*
            FROM papers p
            JOIN user_saved_papers usp ON p.paper_id = usp.paper_id
            WHERE usp.user_id = $1
            ORDER BY usp.saved_at DESC
        """
        
        results = await self.db.fetch(query, user_id)
        return [dict(r) for r in results]
    
    async def _filter_seen_papers(
        self,
        candidates: List[Dict],
        user_id: int
    ) -> List[Dict]:
        """
        Remove papers user has already interacted with.
        
        Args:
            candidates: Candidate papers
            user_id: User identifier
            
        Returns:
            Filtered candidates
        """
        # Get all paper IDs user has interacted with
        query = """
            SELECT DISTINCT paper_id
            FROM user_interactions
            WHERE user_id = $1
        """
        
        seen_papers = await self.db.fetch(query, user_id)
        seen_ids = set([p['paper_id'] for p in seen_papers])
        
        # Filter out seen papers
        filtered = [p for p in candidates if p['paper_id'] not in seen_ids]
        
        logger.debug(
            "Filtered seen papers",
            original_count=len(candidates),
            filtered_count=len(filtered),
            removed=len(candidates) - len(filtered)
        )
        
        return filtered