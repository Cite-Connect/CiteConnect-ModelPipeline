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
from app.db.repositories.evaluation_repo import EvaluationRepository
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EvaluationService:
    """
    Service for evaluating recommendation quality.
    Delegates database operations to EvaluationRepository.
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
        """
        self.db = db
        self.user_repo = UserRepository(db)
        self.gt_repo = GroundTruthRepository(db)
        self.eval_repo = EvaluationRepository(db)
        
        logger.info("EvaluationService initialized")
    
    async def evaluate_cold_start_recommendations(
        self,
        user_id: int,
        recommendations: List[Dict],
        model: str = 'minilm',
        store_result: bool = True
    ) -> Dict[str, Any]:
        """Evaluate recommendations for cold-start user."""
        logger.info(
            "Evaluating cold-start recommendations",
            user_id=user_id,
            recommendation_count=len(recommendations),
            model=model
        )
        
        profile = await self.user_repo.get_profile(user_id)
        if not profile:
            raise ValueError(f"No profile found for user {user_id}")
        
        interests = await self.user_repo.get_user_interests(user_id)
        if not interests:
            raise ValueError(f"No interests found for user {user_id}")
        
        interest_terms = [i['interest_term'] for i in interests]
        
        # Calculate scores
        profile_alignment = await self._calculate_profile_alignment(
            recommendations=recommendations,
            user_interests=interest_terms
        )
        
        ground_truth_quality = await self._calculate_ground_truth_quality(
            recommendations=recommendations,
            user_domain=profile['primary_domain'],
            user_interests=interest_terms
        )
        
        # Note: Database will calculate its own combined_score (GENERATED ALWAYS).
        # We calculate it here for immediate feedback and logic checks.
        combined_score = (
            0.6 * profile_alignment +
            0.4 * ground_truth_quality
        )
        
        passes_threshold = combined_score >= self.COMBINED_SCORE_THRESHOLD
        
        evaluation_result = {
            'user_id': user_id,
            'evaluation_type': 'cold_start',
            'model_used': model,
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
        
        if store_result:
            # We do NOT pass combined_score; the DB generates it.
            await self.eval_repo.save_cold_start_evaluation(
                user_id=user_id,
                embedding_model=model,
                profile_alignment=evaluation_result['profile_alignment'],
                ground_truth_quality=evaluation_result['ground_truth_quality'],
                recommendation_count=evaluation_result['recommendation_count']
            )
        
        return evaluation_result
    
    async def evaluate_warm_start_recommendations(
        self,
        user_id: int,
        recommendations: List[Dict],
        ground_truth_papers: Optional[List[str]] = None,
        store_result: bool = True
    ) -> Dict[str, Any]:
        """Evaluate recommendations for warm-start user."""
        logger.info(
            "Evaluating warm-start recommendations",
            user_id=user_id,
            recommendation_count=len(recommendations)
        )
        
        # If no ground truth provided, fetch saved papers via Repo
        if ground_truth_papers is None:
            ground_truth_papers = await self.eval_repo.get_user_saved_paper_ids(user_id)
        
        # Calculate metrics
        k = len(recommendations)
        recommended_ids = [p['paper_id'] for p in recommendations]
        
        # Precision@K
        relevant_in_recs = len(set(recommended_ids) & set(ground_truth_papers))
        precision_at_k = relevant_in_recs / k if k > 0 else 0.0
        
        # Recall@K
        recall_at_k = relevant_in_recs / len(ground_truth_papers) if ground_truth_papers else 0.0
        
        # NDCG@K
        ndcg_at_k = self._calculate_ndcg(recommended_ids, ground_truth_papers, k)
        
        # Estimated CTR
        avg_score = np.mean([p.get('final_score', 0) for p in recommendations])
        estimated_ctr = min(avg_score * 0.4, 1.0)
        
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
        
        if store_result:
            metadata = {
                'ground_truth_count': evaluation_result['ground_truth_count'],
                'relevant_retrieved': evaluation_result['relevant_retrieved']
            }
            await self.eval_repo.save_warm_start_evaluation(
                user_id=user_id,
                precision_at_10=evaluation_result['precision_at_10'],
                recall_at_10=evaluation_result['recall_at_10'],
                ndcg_at_10=evaluation_result['ndcg_at_10'],
                estimated_ctr=evaluation_result['estimated_ctr'],
                recommended_paper_ids=recommended_ids,
                metadata=metadata
            )
        
        return evaluation_result
    
    async def evaluate_search_augmented_recommendations(
        self,
        user_id: int,
        search_query: str,
        recommendations: List[Dict],
        model: str = 'minilm',
        store_result: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate search-augmented recommendations.
        Note: The DB combined_score is GENERATED ALWAYS using standard weights (0.6/0.4).
        This method uses a search-specific formula (0.5/0.4/0.1).
        We store the search-specific score in metadata to preserve the feature flow.
        """
        logger.info(
            "Evaluating search-augmented recommendations",
            user_id=user_id,
            search_query=search_query[:50],
            recommendation_count=len(recommendations),
            model=model
        )
        
        profile = await self.user_repo.get_profile(user_id)
        if not profile:
            raise ValueError(f"No profile found for user {user_id}")
        
        interests = await self.user_repo.get_user_interests(user_id)
        interest_terms = [i['interest_term'] for i in interests]
        
        # Calculate Metrics
        search_alignment = await self._calculate_search_alignment(
            recommendations=recommendations,
            search_query=search_query
        )
        
        profile_alignment = await self._calculate_profile_alignment(
            recommendations=recommendations,
            user_interests=interest_terms
        )
        
        ground_truth_quality = await self._calculate_ground_truth_quality(
            recommendations=recommendations,
            user_domain=profile['primary_domain'],
            user_interests=interest_terms
        )
        
        # Custom Combined Score for Search
        combined_score = (
            0.50 * search_alignment +
            0.40 * profile_alignment +
            0.10 * ground_truth_quality
        )
        
        passes_threshold = combined_score >= 0.50
        
        evaluation_result = {
            'user_id': user_id,
            'evaluation_type': 'search_augmented',
            'search_query': search_query,
            'model_used': model,
            'search_alignment': round(search_alignment, 4),
            'profile_alignment': round(profile_alignment, 4),
            'ground_truth_quality': round(ground_truth_quality, 4),
            'combined_score': round(combined_score, 4),
            'passes_threshold': passes_threshold,
            'recommendation_count': len(recommendations),
            'evaluated_at': datetime.utcnow().isoformat(),
        }
        
        if store_result:
            await self._store_search_evaluation(evaluation_result)
        
        return evaluation_result

    async def _calculate_search_alignment(
        self,
        recommendations: List[Dict],
        search_query: str
    ) -> float:
        """Calculate how well recommendations match the search query."""
        if not recommendations or not search_query:
            return 0.0
        
        search_terms = set(search_query.lower().split())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        search_terms = {term for term in search_terms if term not in stop_words and len(term) > 2}
        
        if not search_terms:
            return 0.0
        
        paper_scores = []
        for paper in recommendations:
            paper_text = (paper.get('title', '') + ' ' + paper.get('abstract', '')).lower()
            matches = sum(1 for term in search_terms if term in paper_text)
            match_score = matches / len(search_terms)
            
            title_text = paper.get('title', '').lower()
            title_matches = sum(1 for term in search_terms if term in title_text)
            if title_matches > 0:
                match_score *= 1.2
            
            match_score = min(match_score, 1.0)
            paper_scores.append(match_score)
        
        return float(np.mean(paper_scores))

    async def _store_search_evaluation(self, evaluation: Dict[str, Any]) -> None:
        """
        Store search-augmented evaluation result.
        Passes the custom combined_score into metadata since the DB column is generated.
        """
        metadata = {
            'evaluation_type': 'search_augmented',
            'search_query': evaluation['search_query'],
            'search_alignment': evaluation['search_alignment'],
            'search_combined_score': evaluation['combined_score'] # Storing custom score here
        }
        
        await self.eval_repo.save_cold_start_evaluation(
            user_id=evaluation['user_id'],
            embedding_model=evaluation['model_used'],
            profile_alignment=evaluation['profile_alignment'],
            ground_truth_quality=evaluation['ground_truth_quality'],
            recommendation_count=evaluation['recommendation_count'],
            metadata=metadata
        )
    
    async def batch_evaluate_cold_start(
        self,
        user_ids: Optional[List[int]] = None,
        model: str = 'minilm',
        scoring_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Evaluate recommendations for multiple cold-start users."""
        logger.info(
            "Starting batch cold-start evaluation",
            user_count=len(user_ids) if user_ids else "all",
            model=model
        )
        
        if user_ids is None:
            user_ids = await self.eval_repo.get_cold_start_users_for_evaluation()
        
        if not user_ids:
            return {'total_users': 0, 'evaluations': []}
        
        # Import recommendation service (avoid circular import)
        from app.services.recommendation_service import RecommendationService
        rec_service = RecommendationService(self.db)
        
        evaluations = []
        scores = []
        
        for user_id in user_ids:
            try:
                rec_result = await rec_service.generate_cold_start_recommendations(
                    user_id=user_id,
                    count=10,
                    model=model,
                    scoring_weights=scoring_weights
                )
                
                eval_result = await self.evaluate_cold_start_recommendations(
                    user_id=user_id,
                    recommendations=rec_result['papers'],
                    store_result=True
                )
                
                evaluations.append(eval_result)
                scores.append(eval_result['combined_score'])
                
            except Exception as e:
                logger.error("Evaluation failed", user_id=user_id, error=str(e))
        
        if not evaluations:
             return {'total_users': len(user_ids), 'successful_evaluations': 0}

        return {
            'total_users': len(user_ids),
            'successful_evaluations': len(evaluations),
            'model_used': model,
            'avg_combined_score': round(np.mean(scores), 4),
            'pass_rate': round(sum(1 for s in scores if s >= self.COMBINED_SCORE_THRESHOLD) / len(scores), 4),
            'evaluations': evaluations
        }
    
    async def _calculate_profile_alignment(
        self,
        recommendations: List[Dict],
        user_interests: List[str]
    ) -> float:
        """Calculate how well recommendations match user's stated interests."""
        if not recommendations or not user_interests:
            return 0.0
        
        paper_scores = []
        for paper in recommendations:
            paper_keywords = self._extract_paper_keywords(
                title=paper.get('title', ''),
                abstract=paper.get('abstract', '')
            )
            match_score = self._calculate_keyword_match_score(
                paper_keywords=paper_keywords,
                user_interests=user_interests
            )
            paper_scores.append(match_score)
        
        return float(np.mean(paper_scores))
    
    async def _calculate_ground_truth_quality(
        self,
        recommendations: List[Dict],
        user_domain: str,
        user_interests: List[str]
    ) -> float:
        """Calculate ground truth quality via Repo."""
        relevant_gt_papers = await self.eval_repo.find_relevant_ground_truth_papers(
            user_interests=user_interests,
            domain=user_domain
        )
        
        if not relevant_gt_papers:
            return 0.0
        
        total_weight = 0.0
        for paper in recommendations:
            match_weight = await self._check_citation_network_match(
                paper_id=paper['paper_id'],
                gt_paper_ids=relevant_gt_papers
            )
            total_weight += match_weight
        
        return float(total_weight / len(recommendations))
    
    def _extract_paper_keywords(self, title: str, abstract: str, top_k: int = 15) -> List[str]:
        """Extract keywords from paper using simple TF-IDF approach."""
        text = f"{title} {title} {title} {abstract}".lower()
        if not text.strip():
            return []
        
        stop_words = {'the', 'a', 'an', 'and', 'or', 'is', 'in', 'at', 'of', 'for', 'with', 'to'}
        words = re.findall(r'\b[a-z]+\b', text)
        words = [w for w in words if w not in stop_words and len(w) > 3]
        return [word for word, count in Counter(words).most_common(top_k)]
    
    def _calculate_keyword_match_score(self, paper_keywords: List[str], user_interests: List[str]) -> float:
        """Calculate match score handling synonyms and partial matches."""
        if not paper_keywords or not user_interests:
            return 0.0
        
        total = 0.0
        for interest in user_interests:
            interest = interest.lower()
            score = 0.0
            if interest in paper_keywords:
                score = 1.0
            else:
                for kw in paper_keywords:
                    if interest in kw or kw in interest:
                        score = max(score, 0.7)
                    for syn in self.SYNONYMS.get(interest, []):
                        if syn in kw or kw in syn:
                            score = max(score, 0.8)
            total += score
        return total / len(user_interests)
    
    async def _check_citation_network_match(self, paper_id: str, gt_paper_ids: List[str]) -> float:
        """Check if paper appears in citation networks of GT papers."""
        total_weight = 0.0
        for gt_id in gt_paper_ids:
            # We assume gt_repo handles the detailed graph lookups
            rels = await self.gt_repo.get_ground_truth_relationships(gt_id)
            if not rels:
                continue
            
            if rels.get('bibliographic_couples') and paper_id in rels['bibliographic_couples']:
                total_weight += 0.6
            if rels.get('co_cited_papers') and paper_id in rels['co_cited_papers']:
                total_weight += 0.8
                
        return min(total_weight / len(gt_paper_ids) if gt_paper_ids else 0.0, 1.0)
    
    def _calculate_ndcg(self, recommended_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """Calculate NDCG@K."""
        if not recommended_ids or not relevant_ids:
            return 0.0
        
        relevance = [1 if pid in relevant_ids else 0 for pid in recommended_ids[:k]]
        
        dcg = relevance[0]
        for i in range(1, len(relevance)):
            dcg += relevance[i] / np.log2(i + 1)
        
        ideal = sorted(relevance, reverse=True)
        idcg = ideal[0]
        for i in range(1, len(ideal)):
            idcg += ideal[i] / np.log2(i + 1)
            
        return float(dcg / idcg) if idcg > 0 else 0.0

    async def compare_models(self, user_ids: List[int], model_a: str, model_b: str) -> Dict[str, Any]:
        """Compare two embedding models using A/B testing."""
        return await self.eval_repo.get_model_comparison(model_a, model_b)
    
    async def get_evaluation_summary(self, evaluation_type: str = 'cold_start', limit: int = 100) -> Dict[str, Any]:
        """Get summary of recent evaluations."""
        recent_evals = await self.eval_repo.get_recent_evaluations(evaluation_type, limit)
        recent_list = [dict(r) for r in recent_evals]
        
        if not recent_list:
             return {'evaluation_type': evaluation_type, 'total_evaluations': 0, 'summary': {}}

        # Calculate lightweight summary in-memory for the recent batch
        if evaluation_type == 'cold_start':
            scores = [r['combined_score'] for r in recent_list if r['combined_score'] is not None]
            summary = {
                'avg_combined_score': round(np.mean(scores), 4) if scores else 0.0,
                'pass_rate': round(sum(1 for s in scores if s >= self.COMBINED_SCORE_THRESHOLD) / len(scores), 4) if scores else 0.0
            }
        else:
            ctrs = [r['estimated_ctr'] for r in recent_list if r['estimated_ctr'] is not None]
            summary = {
                'avg_ctr': round(np.mean(ctrs), 4) if ctrs else 0.0
            }
            
        return {
            'evaluation_type': evaluation_type,
            'total_evaluations': len(recent_list),
            'summary': summary,
            'recent_evaluations': recent_list[:10]
        }
    
    async def get_user_evaluation_history(self, user_id: int) -> Dict[str, Any]:
        """Get evaluation history for a specific user."""
        cold = await self.eval_repo.get_evaluations_by_user(user_id, 'cold_start')
        warm = await self.eval_repo.get_evaluations_by_user(user_id, 'warm_start')
        return {
            'user_id': user_id,
            'cold_start_evaluations': [dict(r) for r in cold],
            'warm_start_evaluations': [dict(r) for r in warm],
            'total_evaluations': len(cold) + len(warm)
        }