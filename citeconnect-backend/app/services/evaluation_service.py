"""
Evaluation service for CiteConnect recommendations.
Measures recommendation quality using ground truth and profile alignment.
"""
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from datetime import datetime
from collections import Counter
import re

from app.db.connection import DatabaseConnection
from app.db.repositories.user_repo import UserRepository
from app.db.repositories.ground_truth_repo import GroundTruthRepository
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EvaluationService:
    """
    Service for evaluating recommendation quality.
    
    Supports:
    - Cold-start evaluation (profile alignment + ground truth)
    - Warm-start evaluation (precision, recall, NDCG)
    - Batch evaluation for hyperparameter tuning
    - MLflow integration for experiment tracking
    """
    
    # Evaluation thresholds
    PROFILE_ALIGNMENT_THRESHOLD = 0.60
    GROUND_TRUTH_QUALITY_THRESHOLD = 0.50
    COMBINED_SCORE_THRESHOLD = 0.60
    
    # Synonym dictionary for keyword matching
    SYNONYMS = {
        'machine learning': ['ml', 'deep learning', 'neural networks', 'artificial intelligence', 'ai'],
        'medical imaging': ['radiology', 'medical image', 'imaging', 'radiological', 'diagnostic imaging'],
        'diagnostics': ['diagnosis', 'diagnostic', 'clinical diagnosis'],
        'computer vision': ['cv', 'image recognition', 'visual recognition', 'image processing'],
        'natural language processing': ['nlp', 'text processing', 'language models', 'text analysis'],
        'quantum computing': ['quantum', 'qubit', 'quantum algorithms'],
    }
    
    def __init__(self, db: DatabaseConnection):
        """
        Initialize evaluation service.
        
        Args:
            db: Database connection
        """
        self.db = db
        self.user_repo = UserRepository(db)
        self.gt_repo = GroundTruthRepository(db)
        
        logger.info("EvaluationService initialized")
    
    async def evaluate_cold_start_recommendations(
        self,
        user_id: int,
        recommendations: List[Dict],
        model: str = 'minilm',  # ADD THIS PARAMETER
        store_result: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate recommendations for cold-start user.
        
        Args:
            user_id: User identifier
            recommendations: List of recommended papers
            model: Embedding model used ('minilm' or 'specter')
            store_result: Whether to store evaluation in database
            
        Returns:
            Dict with evaluation metrics
        """
        logger.info(
            "Evaluating cold-start recommendations",
            user_id=user_id,
            recommendation_count=len(recommendations),
            model=model
        )
        # Get user data
        profile = await self.user_repo.get_profile(user_id)
        if not profile:
            raise ValueError(f"No profile found for user {user_id}")
        
        interests = await self.user_repo.get_user_interests(user_id)
        if not interests:
            raise ValueError(f"No interests found for user {user_id}")
        
        interest_terms = [i['interest_term'] for i in interests]
        
        # Calculate profile alignment
        profile_alignment = await self._calculate_profile_alignment(
            recommendations=recommendations,
            user_interests=interest_terms
        )
        
        # Calculate ground truth quality
        ground_truth_quality = await self._calculate_ground_truth_quality(
            recommendations=recommendations,
            user_domain=profile['primary_domain'],
            user_interests=interest_terms
        )
        
        # Calculate combined score
        combined_score = (
            0.6 * profile_alignment +
            0.4 * ground_truth_quality
        )
        
        # Determine pass/fail
        passes_threshold = combined_score >= self.COMBINED_SCORE_THRESHOLD
        
        evaluation_result = {
        'user_id': user_id,
        'evaluation_type': 'cold_start',
        'model_used': model,  # ADD THIS
        'profile_alignment': round(profile_alignment, 4),
        'ground_truth_quality': round(ground_truth_quality, 4),
        'combined_score': round(combined_score, 4),
        'passes_threshold': passes_threshold,
        'recommendation_count': len(recommendations),
        'evaluated_at': datetime.utcnow().isoformat(),
        'thresholds': {
            'profile_alignment': self.PROFILE_ALIGNMENT_THRESHOLD,
            'ground_truth_quality': self.GROUND_TRUTH_QUALITY_THRESHOLD,
            'combined_score': self.COMBINED_SCORE_THRESHOLD
            }
        }
        
        logger.info(
            "Cold-start evaluation complete",
            user_id=user_id,
            profile_alignment=profile_alignment,
            ground_truth_quality=ground_truth_quality,
            combined_score=combined_score,
            passes=passes_threshold
        )
        
        # Store evaluation result
        if store_result:
            await self._store_cold_start_evaluation(evaluation_result)
        
        return evaluation_result
    
    async def evaluate_warm_start_recommendations(
        self,
        user_id: int,
        recommendations: List[Dict],
        ground_truth_papers: Optional[List[str]] = None,
        store_result: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate recommendations for warm-start user.
        
        Metrics:
        - Precision@K: Fraction of recommendations that are relevant
        - Recall@K: Fraction of relevant papers that were recommended
        - NDCG@K: Ranking quality
        - Click-through rate: Estimated engagement
        
        Args:
            user_id: User identifier
            recommendations: List of recommended papers
            ground_truth_papers: Known relevant papers (if available)
            store_result: Whether to store evaluation
            
        Returns:
            Dict with evaluation metrics
        """
        logger.info(
            "Evaluating warm-start recommendations",
            user_id=user_id,
            recommendation_count=len(recommendations)
        )
        
        # If no ground truth provided, use saved papers as proxy
        if ground_truth_papers is None:
            saved_papers = await self.db.fetch(
                "SELECT paper_id FROM user_saved_papers WHERE user_id = $1",
                user_id
            )
            ground_truth_papers = [p['paper_id'] for p in saved_papers]
        
        # Calculate metrics
        k = len(recommendations)
        recommended_ids = [p['paper_id'] for p in recommendations]
        
        # Precision@K
        relevant_in_recs = len(set(recommended_ids) & set(ground_truth_papers))
        precision_at_k = relevant_in_recs / k if k > 0 else 0.0
        
        # Recall@K
        recall_at_k = relevant_in_recs / len(ground_truth_papers) if ground_truth_papers else 0.0
        
        # NDCG@K (simplified - assumes binary relevance)
        ndcg_at_k = self._calculate_ndcg(recommended_ids, ground_truth_papers, k)
        
        # Estimated CTR (based on scores)
        avg_score = np.mean([p.get('final_score', 0) for p in recommendations])
        estimated_ctr = min(avg_score * 0.4, 1.0)  # Heuristic
        
        evaluation_result = {
            'user_id': user_id,
            'evaluation_type': 'warm_start',
            'precision_at_10': round(precision_at_k, 4),
            'recall_at_10': round(recall_at_k, 4),
            'ndcg_at_10': round(ndcg_at_k, 4),
            'estimated_ctr': round(estimated_ctr, 4),
            'recommendation_count': len(recommendations),
            'ground_truth_count': len(ground_truth_papers),
            'relevant_retrieved': relevant_in_recs,
            'recommended_paper_ids': recommended_ids,
            'evaluated_at': datetime.utcnow().isoformat()
        }
        
        logger.info(
            "Warm-start evaluation complete",
            user_id=user_id,
            precision=precision_at_k,
            recall=recall_at_k,
            ndcg=ndcg_at_k
        )
        
        if store_result:
            await self._store_warm_start_evaluation(evaluation_result)
        
        return evaluation_result
    
    async def batch_evaluate_cold_start(
        self,
        user_ids: Optional[List[int]] = None,
        model: str = 'minilm',
        scoring_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate recommendations for multiple cold-start users.
        Used for hyperparameter tuning and model comparison.
        
        Args:
            user_ids: List of user IDs (if None, evaluates all cold-start users)
            model: Model to use for recommendations
            scoring_weights: Optional custom weights to test
            
        Returns:
            Aggregated evaluation results
        """
        logger.info(
            "Starting batch cold-start evaluation",
            user_count=len(user_ids) if user_ids else "all",
            model=model
        )
        
        # If no user_ids provided, get all cold-start users
        if user_ids is None:
            query = """
                SELECT user_id
                FROM user_recommendation_state
                WHERE recommendation_stage = 'cold_start'
                  AND user_id IN (SELECT user_id FROM user_profiles_extended)
                ORDER BY user_id
            """
            results = await self.db.fetch(query)
            user_ids = [r['user_id'] for r in results]
        
        if not user_ids:
            logger.warning("No cold-start users found for evaluation")
            return {
                'total_users': 0,
                'evaluations': []
            }
        
        logger.info(f"Evaluating {len(user_ids)} cold-start users")
        
        # Import recommendation service (avoid circular import)
        from app.services.recommendation_service import RecommendationService
        rec_service = RecommendationService(self.db)
        
        # Evaluate each user
        evaluations = []
        profile_alignments = []
        ground_truth_qualities = []
        combined_scores = []
        
        for i, user_id in enumerate(user_ids, 1):
            try:
                logger.debug(
                    f"Evaluating user {i}/{len(user_ids)}",
                    user_id=user_id
                )
                
                # Generate recommendations
                rec_result = await rec_service.generate_cold_start_recommendations(
                    user_id=user_id,
                    count=10,
                    model=model,
                    scoring_weights=scoring_weights
                )
                
                # Evaluate
                eval_result = await self.evaluate_cold_start_recommendations(
                    user_id=user_id,
                    recommendations=rec_result['papers'],
                    store_result=True
                )
                
                evaluations.append(eval_result)
                profile_alignments.append(eval_result['profile_alignment'])
                ground_truth_qualities.append(eval_result['ground_truth_quality'])
                combined_scores.append(eval_result['combined_score'])
                
            except Exception as e:
                logger.error(
                    "Evaluation failed for user",
                    user_id=user_id,
                    error=str(e),
                    exc_info=True
                )
        
        # Aggregate results
        aggregated = {
            'total_users': len(user_ids),
            'successful_evaluations': len(evaluations),
            'failed_evaluations': len(user_ids) - len(evaluations),
            'model_used': model,
            'scoring_weights': scoring_weights or rec_service.DEFAULT_COLD_START_WEIGHTS,
            'aggregate_metrics': {
                'avg_profile_alignment': round(np.mean(profile_alignments), 4) if profile_alignments else 0.0,
                'avg_ground_truth_quality': round(np.mean(ground_truth_qualities), 4) if ground_truth_qualities else 0.0,
                'avg_combined_score': round(np.mean(combined_scores), 4) if combined_scores else 0.0,
                'std_profile_alignment': round(np.std(profile_alignments), 4) if profile_alignments else 0.0,
                'std_ground_truth_quality': round(np.std(ground_truth_qualities), 4) if ground_truth_qualities else 0.0,
                'std_combined_score': round(np.std(combined_scores), 4) if combined_scores else 0.0,
                'min_combined_score': round(min(combined_scores), 4) if combined_scores else 0.0,
                'max_combined_score': round(max(combined_scores), 4) if combined_scores else 0.0,
                'pass_rate': round(sum(1 for s in combined_scores if s >= self.COMBINED_SCORE_THRESHOLD) / len(combined_scores), 4) if combined_scores else 0.0
            },
            'evaluations': evaluations,
            'evaluated_at': datetime.utcnow().isoformat()
        }
        
        logger.info(
            "Batch evaluation complete",
            total_users=len(user_ids),
            successful=len(evaluations),
            avg_combined_score=aggregated['aggregate_metrics']['avg_combined_score'],
            pass_rate=aggregated['aggregate_metrics']['pass_rate']
        )
        
        return aggregated
    
    async def _calculate_profile_alignment(
        self,
        recommendations: List[Dict],
        user_interests: List[str]
    ) -> float:
        """
        Calculate how well recommendations match user's stated interests.
        
        Args:
            recommendations: List of recommended papers
            user_interests: User's interest terms
            
        Returns:
            Profile alignment score (0.0-1.0)
        """
        logger.debug(
            "Calculating profile alignment",
            recommendation_count=len(recommendations),
            interest_count=len(user_interests)
        )
        
        if not recommendations or not user_interests:
            return 0.0
        
        paper_scores = []
        
        for paper in recommendations:
            # Extract keywords from paper
            paper_keywords = self._extract_paper_keywords(
                title=paper.get('title', ''),
                abstract=paper.get('abstract', '')
            )
            
            # Calculate match score
            match_score = self._calculate_keyword_match_score(
                paper_keywords=paper_keywords,
                user_interests=user_interests
            )
            
            paper_scores.append(match_score)
            
            logger.debug(
                "Paper alignment calculated",
                paper_id=paper.get('paper_id'),
                match_score=match_score,
                keywords_found=len(paper_keywords)
            )
        
        # Average across all papers
        profile_alignment = np.mean(paper_scores)
        
        logger.info(
            "Profile alignment calculated",
            score=profile_alignment,
            individual_scores=paper_scores
        )
        
        return float(profile_alignment)
    
    async def _calculate_ground_truth_quality(
        self,
        recommendations: List[Dict],
        user_domain: str,
        user_interests: List[str]
    ) -> float:
        """
        Calculate ground truth quality by checking citation network overlap.
        
        Args:
            recommendations: List of recommended papers
            user_domain: User's primary domain
            user_interests: User's interest terms
            
        Returns:
            Ground truth quality score (0.0-1.0)
        """
        logger.debug(
            "Calculating ground truth quality",
            recommendation_count=len(recommendations),
            domain=user_domain
        )
        
        # Find relevant ground truth papers
        relevant_gt_papers = await self._get_relevant_ground_truth_papers(
            user_interests=user_interests,
            domain=user_domain
        )
        
        if not relevant_gt_papers:
            logger.warning(
                "No relevant ground truth papers found",
                domain=user_domain,
                interests=user_interests
            )
            return 0.0
        
        logger.debug(
            "Relevant GT papers found",
            count=len(relevant_gt_papers)
        )
        
        # Check each recommendation against GT networks
        total_weight = 0.0
        matches_detail = []
        
        for paper in recommendations:
            paper_id = paper['paper_id']
            
            # Check if paper appears in any GT network
            match_weight = await self._check_citation_network_match(
                paper_id=paper_id,
                gt_paper_ids=relevant_gt_papers
            )
            
            total_weight += match_weight
            
            matches_detail.append({
                'paper_id': paper_id,
                'title': paper.get('title', ''),
                'match_weight': match_weight,
                'in_ground_truth': match_weight > 0
            })
        
        # Normalize by number of recommendations
        ground_truth_quality = total_weight / len(recommendations)
        
        # Count how many papers matched
        matched_count = sum(1 for m in matches_detail if m['in_ground_truth'])
        
        logger.info(
            "Ground truth quality calculated",
            score=ground_truth_quality,
            matched_papers=matched_count,
            total_recommendations=len(recommendations),
            relevant_gt_papers=len(relevant_gt_papers)
        )
        
        return float(ground_truth_quality)
    
    def _extract_paper_keywords(
        self,
        title: str,
        abstract: str,
        top_k: int = 15
    ) -> List[str]:
        """
        Extract keywords from paper using simple TF-IDF approach.
        
        Args:
            title: Paper title
            abstract: Paper abstract
            top_k: Number of keywords to extract
            
        Returns:
            List of keyword terms
        """
        # Combine title (weighted 3x) and abstract
        text = f"{title} {title} {title} {abstract}"
        
        if not text.strip():
            return []
        
        # Simple preprocessing
        text = text.lower()
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
            'those', 'we', 'our', 'their', 'its', 'such', 'which', 'using'
        }
        
        # Extract words (simple tokenization)
        words = re.findall(r'\b[a-z]+\b', text)
        
        # Filter stop words and short words
        words = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Count frequencies
        word_counts = Counter(words)
        
        # Get top K most frequent
        top_words = [word for word, count in word_counts.most_common(top_k)]
        
        return top_words
    
    def _calculate_keyword_match_score(
        self,
        paper_keywords: List[str],
        user_interests: List[str]
    ) -> float:
        """
        Calculate match score between paper keywords and user interests.
        Handles synonyms and partial matches.
        
        Args:
            paper_keywords: Keywords from paper
            user_interests: User's interest terms
            
        Returns:
            Match score (0.0-1.0)
        """
        if not paper_keywords or not user_interests:
            return 0.0
        
        total_match_score = 0.0
        
        for interest in user_interests:
            interest_lower = interest.lower()
            max_match_for_interest = 0.0
            
            # Check exact match
            if interest_lower in paper_keywords:
                max_match_for_interest = 1.0
            else:
                # Check partial matches and synonyms
                for keyword in paper_keywords:
                    # Partial match (e.g., "learning" in "machine learning")
                    if interest_lower in keyword or keyword in interest_lower:
                        max_match_for_interest = max(max_match_for_interest, 0.7)
                    
                    # Check synonyms
                    synonyms = self.SYNONYMS.get(interest_lower, [])
                    for synonym in synonyms:
                        if synonym in keyword or keyword in synonym:
                            max_match_for_interest = max(max_match_for_interest, 0.8)
            
            total_match_score += max_match_for_interest
        
        # Normalize by number of interests
        match_score = total_match_score / len(user_interests)
        
        return match_score
    
    async def _get_relevant_ground_truth_papers(
        self,
        user_interests: List[str],
        domain: str,
        limit: int = 10
    ) -> List[str]:
        """
        Find ground truth papers relevant to user's interests.
        
        Args:
            user_interests: User's interest terms
            domain: User's domain
            limit: Maximum GT papers to return
            
        Returns:
            List of relevant GT paper IDs
        """
        # Build search conditions for interests
        conditions = []
        for interest in user_interests:
            conditions.append(f"title ILIKE '%{interest}%'")
            conditions.append(f"abstract ILIKE '%{interest}%'")
        
        where_clause = ' OR '.join(conditions)
        
        query = f"""
            SELECT gtp.paper_id, p.title
            FROM ground_truth_papers gtp
            JOIN papers p ON gtp.paper_id = p.paper_id
            WHERE gtp.domain = $1
              AND ({where_clause})
            ORDER BY gtp.quality_score DESC
            LIMIT $2
        """
        
        results = await self.db.fetch(query, domain, limit)
        
        gt_paper_ids = [r['paper_id'] for r in results]
        
        logger.debug(
            "Relevant GT papers identified",
            count=len(gt_paper_ids),
            domain=domain
        )
        
        return gt_paper_ids
    
    async def _check_citation_network_match(
        self,
        paper_id: str,
        gt_paper_ids: List[str]
    ) -> float:
        """
        Check if paper appears in citation networks of GT papers.
        
        Args:
            paper_id: Paper to check
            gt_paper_ids: Relevant ground truth paper IDs
            
        Returns:
            Match weight (0.0-1.0+)
        """
        total_weight = 0.0
        
        for gt_id in gt_paper_ids:
            # Get relationships
            relationships = await self.gt_repo.get_ground_truth_relationships(gt_id)
            
            if not relationships:
                continue
            
            # Check citation network (direct citation)
            '''if relationships.get('citation_network') and paper_id in relationships['citation_network']:
                total_weight += 1.0
                logger.debug(
                    "Citation network match found",
                    paper_id=paper_id,
                    gt_paper_id=gt_id,
                    match_type='direct_citation'
                )'''
            
            # Check bibliographic couples
            if relationships.get('bibliographic_couples') and paper_id in relationships['bibliographic_couples']:
                total_weight += 0.6
                logger.debug(
                    "Bibliographic couple match found",
                    paper_id=paper_id,
                    gt_paper_id=gt_id,
                    match_type='bibliographic_couple'
                )
            
            # Check co-cited papers (if available)
            if relationships.get('co_cited_papers') and paper_id in relationships['co_cited_papers']:
                total_weight += 0.8
                logger.debug(
                    "Co-citation match found",
                    paper_id=paper_id,
                    gt_paper_id=gt_id,
                    match_type='co_cited'
                )
        
        # Normalize by number of GT papers
        if gt_paper_ids:
            normalized_weight = total_weight / len(gt_paper_ids)
        else:
            normalized_weight = 0.0
        
        return min(normalized_weight, 1.0)
    
    def _calculate_ndcg(
        self,
        recommended_ids: List[str],
        relevant_ids: List[str],
        k: int
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain @K.
        
        Args:
            recommended_ids: Recommended paper IDs
            relevant_ids: Ground truth relevant paper IDs
            k: Cutoff position
            
        Returns:
            NDCG@K score (0.0-1.0)
        """
        if not recommended_ids or not relevant_ids:
            return 0.0
        
        # Create relevance array (1 if relevant, 0 otherwise)
        relevance = [1 if pid in relevant_ids else 0 for pid in recommended_ids[:k]]
        
        # Calculate DCG
        dcg = relevance[0]
        for i in range(1, len(relevance)):
            dcg += relevance[i] / np.log2(i + 1)
        
        # Calculate ideal DCG
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = ideal_relevance[0]
        for i in range(1, len(ideal_relevance)):
            idcg += ideal_relevance[i] / np.log2(i + 1)
        
        # Calculate NDCG
        if idcg == 0:
            return 0.0
        
        ndcg = dcg / idcg
        
        return float(ndcg)
    
    async def _store_cold_start_evaluation(
        self,
        evaluation: Dict[str, Any]
    ) -> None:
        """
        Store cold-start evaluation result in database.
        
        Args:
            evaluation: Evaluation result dict
        """
        query = """
            INSERT INTO cold_start_evaluations (
                user_id,
                embedding_model,
                profile_alignment,
                ground_truth_quality,
                recommendation_count,
                evaluation_timestamp
            )
            VALUES ($1, $2, $3, $4, $5, NOW())
            RETURNING evaluation_id, combined_score
        """
        
        # Determine model from context (you'll need to pass this)
        # For now, default to 'minilm'
        embedding_model = evaluation.get('model_used', 'all-MiniLM-L6-v2')
        
        result = await self.db.fetchrow(
            query,
            evaluation['user_id'],
            embedding_model,
            evaluation['profile_alignment'],
            evaluation['ground_truth_quality'],
            evaluation['recommendation_count']
        )
        
        logger.debug(
            "Cold-start evaluation stored",
            evaluation_id=result['evaluation_id'] if result else None,
            user_id=evaluation['user_id'],
            combined_score=result['combined_score'] if result else None
        )
    
    async def _store_warm_start_evaluation(
        self,
        evaluation: Dict[str, Any]
    ) -> None:
        """
        Store warm-start evaluation result in database.
        
        Args:
            evaluation: Evaluation result dict
        """
        query = """
            INSERT INTO warm_start_evaluations (
                user_id,
                precision_at_10,
                recall_at_10,
                ndcg_at_10,
                estimated_ctr,
                recommended_paper_ids,
                evaluation_metadata,
                evaluated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
            RETURNING evaluation_id
        """
        
        metadata = {
            'ground_truth_count': evaluation['ground_truth_count'],
            'relevant_retrieved': evaluation['relevant_retrieved']
        }
        
        result = await self.db.fetchrow(
            query,
            evaluation['user_id'],
            evaluation['precision_at_10'],
            evaluation['recall_at_10'],
            evaluation['ndcg_at_10'],
            evaluation['estimated_ctr'],
            evaluation['recommended_paper_ids'],
            metadata
        )
        
        logger.debug(
            "Warm-start evaluation stored",
            evaluation_id=result['evaluation_id'] if result else None,
            user_id=evaluation['user_id']
        )
    
    async def compare_models(
        self,
        user_ids: List[int],
        model_a: str = 'minilm',
        model_b: str = 'specter'
    ) -> Dict[str, Any]:
        """
        Compare two embedding models using A/B testing.
        
        Args:
            user_ids: Users to evaluate
            model_a: First model
            model_b: Second model
            
        Returns:
            Comparison results with statistical significance
        """
        logger.info(
            "Comparing models",
            model_a=model_a,
            model_b=model_b,
            user_count=len(user_ids)
        )
        
        # Evaluate with model A
        results_a = await self.batch_evaluate_cold_start(
            user_ids=user_ids,
            model=model_a
        )
        
        # Evaluate with model B
        results_b = await self.batch_evaluate_cold_start(
            user_ids=user_ids,
            model=model_b
        )
        
        # Extract combined scores
        scores_a = [e['combined_score'] for e in results_a['evaluations']]
        scores_b = [e['combined_score'] for e in results_b['evaluations']]
        
        # Statistical comparison
        mean_a = np.mean(scores_a)
        mean_b = np.mean(scores_b)
        std_a = np.std(scores_a)
        std_b = np.std(scores_b)
        
        # Simple t-test (approximate)
        diff = abs(mean_a - mean_b)
        pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
        
        # Determine winner
        if diff > 0.05 and pooled_std < 0.15:  # Significant difference
            winner = model_a if mean_a > mean_b else model_b
            confidence = 'high'
        elif diff > 0.02:
            winner = model_a if mean_a > mean_b else model_b
            confidence = 'medium'
        else:
            winner = 'tie'
            confidence = 'low'
        
        comparison = {
            'model_a': model_a,
            'model_b': model_b,
            'user_count': len(user_ids),
            'results_a': {
                'mean_score': round(mean_a, 4),
                'std_score': round(std_a, 4),
                'min_score': round(min(scores_a), 4),
                'max_score': round(max(scores_a), 4)
            },
            'results_b': {
                'mean_score': round(mean_b, 4),
                'std_score': round(std_b, 4),
                'min_score': round(min(scores_b), 4),
                'max_score': round(max(scores_b), 4)
            },
            'winner': winner,
            'confidence': confidence,
            'score_difference': round(diff, 4),
            'compared_at': datetime.utcnow().isoformat()
        }
        
        logger.info(
            "Model comparison complete",
            winner=winner,
            model_a_score=mean_a,
            model_b_score=mean_b,
            difference=diff
        )
        
        return comparison
    
    async def get_evaluation_summary(
        self,
        evaluation_type: str = 'cold_start',
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Get summary of recent evaluations.
        
        Args:
            evaluation_type: 'cold_start' or 'warm_start'
            limit: Number of recent evaluations
            
        Returns:
            Summary statistics
        """
        logger.info(
            "Getting evaluation summary",
            type=evaluation_type,
            limit=limit
        )
        
        if evaluation_type == 'cold_start':
            query = """
                SELECT 
                    profile_alignment,
                    ground_truth_quality,
                    combined_score,
                    passes_threshold,
                    evaluated_at
                FROM cold_start_evaluations
                ORDER BY evaluated_at DESC
                LIMIT $1
            """
        else:
            query = """
                SELECT 
                    precision_at_10,
                    recall_at_10,
                    ndcg_at_10,
                    estimated_ctr,
                    evaluated_at
                FROM warm_start_evaluations
                ORDER BY evaluated_at DESC
                LIMIT $1
            """
        
        results = await self.db.fetch(query, limit)
        
        if not results:
            return {
                'evaluation_type': evaluation_type,
                'total_evaluations': 0,
                'summary': {}
            }
        
        if evaluation_type == 'cold_start':
            summary = {
                'total_evaluations': len(results),
                'avg_profile_alignment': round(np.mean([r['profile_alignment'] for r in results]), 4),
                'avg_ground_truth_quality': round(np.mean([r['ground_truth_quality'] for r in results]), 4),
                'avg_combined_score': round(np.mean([r['combined_score'] for r in results]), 4),
                'pass_rate': round(sum(1 for r in results if r['passes_threshold']) / len(results), 4),
                'std_combined_score': round(np.std([r['combined_score'] for r in results]), 4),
                'min_combined_score': round(min([r['combined_score'] for r in results]), 4),
                'max_combined_score': round(max([r['combined_score'] for r in results]), 4)
            }
        else:
            summary = {
                'total_evaluations': len(results),
                'avg_precision': round(np.mean([r['precision_at_10'] for r in results]), 4),
                'avg_recall': round(np.mean([r['recall_at_10'] for r in results]), 4),
                'avg_ndcg': round(np.mean([r['ndcg_at_10'] for r in results]), 4),
                'avg_ctr': round(np.mean([r['estimated_ctr'] for r in results]), 4)
            }
        
        logger.info(
            "Evaluation summary retrieved",
            type=evaluation_type,
            total=len(results)
        )
        
        return {
            'evaluation_type': evaluation_type,
            'total_evaluations': len(results),
            'summary': summary,
            'recent_evaluations': [dict(r) for r in results[:10]]
        }
    
    async def get_user_evaluation_history(
        self,
        user_id: int
    ) -> Dict[str, Any]:
        """
        Get evaluation history for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            User's evaluation history
        """
        logger.info(
            "Getting user evaluation history",
            user_id=user_id
        )
        
        # Get cold-start evaluations
        cold_start_query = """
            SELECT *
            FROM cold_start_evaluations
            WHERE user_id = $1
            ORDER BY evaluated_at DESC
        """
        cold_start = await self.db.fetch(cold_start_query, user_id)
        
        # Get warm-start evaluations
        warm_start_query = """
            SELECT *
            FROM warm_start_evaluations
            WHERE user_id = $1
            ORDER BY evaluated_at DESC
        """
        warm_start = await self.db.fetch(warm_start_query, user_id)
        
        return {
            'user_id': user_id,
            'cold_start_evaluations': [dict(e) for e in cold_start],
            'warm_start_evaluations': [dict(e) for e in warm_start],
            'total_evaluations': len(cold_start) + len(warm_start)
        }