"""
Evaluation service for assessing recommendation quality.
Implements cold-start evaluation and bias detection from Model Development Guidelines.
"""
from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import defaultdict
from app.config import settings
from app.utils.logger import get_logger
from app.db.repositories.paper_repo import PaperRepository
from app.services.bootstrap.ground_truth_service import GroundTruthService

logger = get_logger(__name__)


class EvaluationService:
    """
    Evaluates recommendation quality using multiple metrics.
    Implements simplified 2-metric evaluation for cold-start and richer metrics for mature users.
    """
    
    # Success thresholds from config
    SUCCESS_THRESHOLDS = {
        'cold_start': {
            'profile_alignment': settings.COLD_START_PROFILE_ALIGNMENT_THRESHOLD,
            'ground_truth_quality': settings.COLD_START_GROUND_TRUTH_THRESHOLD
        },
        'mature': {
            'precision_at_10': settings.MATURE_PRECISION_AT_10_THRESHOLD,
            'ctr': settings.MATURE_CTR_THRESHOLD
        }
    }
    
    # Slicing dimensions for bias detection
    SLICING_DIMENSIONS = [
        'research_stage',
        'domain',
        'reading_level',
        'years_experience'
    ]
    
    def __init__(
        self,
        paper_repo: PaperRepository,
        ground_truth_service: GroundTruthService
    ):
        """
        Initialize evaluation service.
        
        Args:
            paper_repo: Paper repository
            ground_truth_service: Ground truth service
        """
        self.paper_repo = paper_repo
        self.ground_truth_service = ground_truth_service
        
        logger.info("EvaluationService initialized")
    
    async def evaluate_cold_start(
        self,
        recommendations: List[Dict],
        user_profile: Dict
    ) -> Dict[str, float]:
        """
        Simplified 2-metric evaluation for cold-start users.
        
        Metrics:
        1. Profile alignment: How well papers match user interests/domain
        2. Ground truth quality: Are papers in known good citation networks
        
        Args:
            recommendations: List of recommended papers
            user_profile: User profile data
            
        Returns:
            Dict with evaluation metrics
        """
        logger.debug(
            "Cold-start evaluation",
            rec_count=len(recommendations),
            user_stage=user_profile.get('research_stage')
        )
        
        # Calculate profile alignment
        profile_alignment = self._calculate_profile_alignment(
            recommendations,
            user_profile
        )
        
        # Evaluate against ground truth
        paper_ids = [p['paper_id'] for p in recommendations]
        gt_eval = await self.ground_truth_service.evaluate_against_ground_truth(
            paper_ids,
            user_profile
        )
        
        ground_truth_quality = gt_eval.get('ground_truth_quality', 0.0)
        
        # Combined score (weighted average)
        combined_score = (
            profile_alignment * 0.6 +
            ground_truth_quality * 0.4
        )
        
        # Check if passes threshold
        passed = (
            profile_alignment >= self.SUCCESS_THRESHOLDS['cold_start']['profile_alignment'] and
            ground_truth_quality >= self.SUCCESS_THRESHOLDS['cold_start']['ground_truth_quality']
        )
        
        result = {
            'profile_alignment': profile_alignment,
            'ground_truth_quality': ground_truth_quality,
            'combined_score': combined_score,
            'passed_threshold': passed
        }
        
        logger.info(
            "Cold-start evaluation complete",
            profile_alignment=profile_alignment,
            ground_truth_quality=ground_truth_quality,
            passed=passed
        )
        
        return result
    
    def _calculate_profile_alignment(
        self,
        recommendations: List[Dict],
        user_profile: Dict
    ) -> float:
        """
        Calculate how well recommendations align with user profile.
        
        Args:
            recommendations: List of papers
            user_profile: User profile
            
        Returns:
            float: Alignment score (0-1)
        """
        if not recommendations or not user_profile:
            return 0.0
        
        user_domain = user_profile.get('primary_domain', '')
        user_interests = set(
            interest.lower()
            for interest in user_profile.get('interests', [])
        )
        user_sub_domains = set(
            sd.lower()
            for sd in user_profile.get('sub_domains', [])
        )
        
        alignment_scores = []
        
        for paper in recommendations:
            score = 0.0
            
            # Domain match (40% weight)
            if paper.get('domain', '').lower() == user_domain.lower():
                score += 0.4
            
            # Sub-domain match (20% weight)
            paper_domain = paper.get('domain', '').lower()
            if paper_domain in user_sub_domains:
                score += 0.2
            
            # Interest keyword match (40% weight)
            paper_text = (
                paper.get('title', '') + ' ' +
                paper.get('abstract', '')
            ).lower()
            
            interest_matches = sum(
                1 for interest in user_interests
                if interest in paper_text
            )
            
            if user_interests:
                score += 0.4 * min(interest_matches / len(user_interests), 1.0)
            
            alignment_scores.append(score)
        
        avg_alignment = sum(alignment_scores) / len(alignment_scores)
        
        logger.debug(
            "Profile alignment calculated",
            score=avg_alignment,
            rec_count=len(recommendations)
        )
        
        return avg_alignment
    
    async def evaluate_mature_user(
        self,
        recommendations: List[Dict],
        user_interactions: List[Dict],
        ground_truth_papers: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Richer evaluation for mature users with interaction history.
        
        Metrics:
        - Precision@10: Fraction of relevant recommendations
        - Recall@10: Fraction of relevant papers found
        - Click-through rate: Actual user engagement
        - Save rate: Strong positive signals
        
        Args:
            recommendations: List of recommended papers
            user_interactions: User's past interactions
            ground_truth_papers: Optional ground truth for comparison
            
        Returns:
            Dict with evaluation metrics
        """
        logger.debug(
            "Mature user evaluation",
            rec_count=len(recommendations),
            interaction_count=len(user_interactions)
        )
        
        # Build set of papers user has positively interacted with
        positive_interactions = {
            i['paper_id']
            for i in user_interactions
            if i.get('interaction_strength', 0) > 0.3
        }
        
        # Calculate precision@10
        top_10 = recommendations[:10]
        top_10_ids = [p['paper_id'] for p in top_10]
        
        relevant_in_top_10 = len(
            set(top_10_ids) & positive_interactions
        )
        
        precision_at_10 = relevant_in_top_10 / 10 if top_10 else 0.0
        
        # Calculate recall (if ground truth provided)
        recall_at_10 = 0.0
        if ground_truth_papers:
            recall_at_10 = (
                len(set(top_10_ids) & set(ground_truth_papers)) /
                len(ground_truth_papers)
                if ground_truth_papers else 0.0
            )
        
        # Calculate engagement metrics from interactions
        clicked = sum(
            1 for i in user_interactions[-20:]  # Last 20 interactions
            if i.get('interaction_type') in ['click', 'view']
        )
        saved = sum(
            1 for i in user_interactions[-20:]
            if i.get('interaction_type') == 'save'
        )
        
        ctr = clicked / 20 if len(user_interactions) >= 20 else 0.0
        save_rate = saved / 20 if len(user_interactions) >= 20 else 0.0
        
        # Check if passes threshold
        passed = (
            precision_at_10 >= self.SUCCESS_THRESHOLDS['mature']['precision_at_10'] and
            ctr >= self.SUCCESS_THRESHOLDS['mature']['ctr']
        )
        
        result = {
            'precision_at_10': precision_at_10,
            'recall_at_10': recall_at_10,
            'click_through_rate': ctr,
            'save_rate': save_rate,
            'passed_threshold': passed
        }
        
        logger.info(
            "Mature user evaluation complete",
            precision=precision_at_10,
            ctr=ctr,
            passed=passed
        )
        
        return result
    
    async def detect_bias(
        self,
        recommendation_events: List[Dict],
        slicing_dimensions: Optional[List[str]] = None
    ) -> Dict:
        """
        Detect bias across user segments using slicing techniques.
        Implements Model Development Guidelines requirement.
        
        Args:
            recommendation_events: List of recommendation events with outcomes
            slicing_dimensions: Dimensions to slice by
            
        Returns:
            BiasReport with detected biases
        """
        if slicing_dimensions is None:
            slicing_dimensions = self.SLICING_DIMENSIONS
        
        logger.info(
            "Bias detection started",
            event_count=len(recommendation_events),
            dimensions=slicing_dimensions
        )
        
        bias_results = {}
        
        for dimension in slicing_dimensions:
            logger.debug(f"Analyzing dimension: {dimension}")
            
            # Group events by dimension value
            slices = defaultdict(list)
            for event in recommendation_events:
                user_profile = event.get('user_profile', {})
                dimension_value = user_profile.get(dimension, 'unknown')
                slices[dimension_value].append(event)
            
            # Calculate metrics per slice
            slice_metrics = {}
            for slice_value, slice_events in slices.items():
                if len(slice_events) < 10:  # Skip small slices
                    continue
                
                metrics = self._calculate_slice_metrics(slice_events)
                slice_metrics[slice_value] = metrics
            
            # Check for bias (>20% variance)
            if len(slice_metrics) >= 2:
                precision_values = [
                    m['precision'] for m in slice_metrics.values()
                ]
                
                max_precision = max(precision_values)
                min_precision = min(precision_values)
                
                if max_precision > 0:
                    variance = (max_precision - min_precision) / max_precision
                    
                    if variance > settings.BIAS_VARIANCE_THRESHOLD:
                        worst_slice = min(
                            slice_metrics.items(),
                            key=lambda x: x[1]['precision']
                        )
                        
                        bias_results[dimension] = {
                            'bias_detected': True,
                            'variance': variance,
                            'worst_performing_slice': worst_slice[0],
                            'worst_precision': worst_slice[1]['precision'],
                            'best_precision': max_precision,
                            'slice_breakdown': slice_metrics
                        }
                        
                        logger.warning(
                            "Bias detected",
                            dimension=dimension,
                            variance=variance,
                            worst_slice=worst_slice[0]
                        )
        
        bias_detected = bool(bias_results)
        
        report = {
            'bias_detected': bias_detected,
            'dimensions_with_bias': list(bias_results.keys()),
            'details': bias_results,
            'recommendations': self._generate_bias_mitigation_recommendations(
                bias_results
            ) if bias_detected else []
        }
        
        logger.info(
            "Bias detection complete",
            bias_detected=bias_detected,
            affected_dimensions=len(bias_results)
        )
        
        return report
    
    def _calculate_slice_metrics(self, slice_events: List[Dict]) -> Dict:
        """
        Calculate metrics for a slice of events.
        
        Args:
            slice_events: Events in this slice
            
        Returns:
            Dict with metrics
        """
        total = len(slice_events)
        
        # Count clicks and saves
        clicks = sum(
            1 for event in slice_events
            if any(
                i.get('interaction_type') == 'click'
                for i in event.get('interactions', [])
            )
        )
        
        saves = sum(
            1 for event in slice_events
            if any(
                i.get('interaction_type') == 'save'
                for i in event.get('interactions', [])
            )
        )
        
        # Calculate precision (clicked or saved papers)
        relevant = sum(
            1 for event in slice_events
            if any(
                i.get('interaction_strength', 0) > 0.3
                for i in event.get('interactions', [])
            )
        )
        
        precision = relevant / total if total > 0 else 0.0
        ctr = clicks / total if total > 0 else 0.0
        save_rate = saves / total if total > 0 else 0.0
        
        return {
            'count': total,
            'precision': precision,
            'ctr': ctr,
            'save_rate': save_rate
        }
    
    def _generate_bias_mitigation_recommendations(
        self,
        bias_results: Dict
    ) -> List[str]:
        """
        Generate recommendations for mitigating detected bias.
        
        Args:
            bias_results: Detected biases
            
        Returns:
            List of mitigation recommendations
        """
        recommendations = []
        
        for dimension, details in bias_results.items():
            worst_slice = details['worst_performing_slice']
            variance = details['variance']
            
            recommendations.append(
                f"Increase training data for {dimension}={worst_slice} "
                f"(current variance: {variance:.2%})"
            )
            
            recommendations.append(
                f"Apply fairness constraints to ensure {dimension} "
                f"performance variance < 20%"
            )
            
            recommendations.append(
                f"Consider separate models or re-weighting for {dimension} groups"
            )
        
        return recommendations
    
    async def calculate_diversity_score(
        self,
        recommendations: List[Dict]
    ) -> float:
        """
        Calculate diversity score of recommendations.
        
        Checks:
        - Author diversity
        - Venue diversity
        - Domain diversity
        - Temporal diversity
        
        Args:
            recommendations: List of papers
            
        Returns:
            float: Diversity score (0-1)
        """
        if not recommendations:
            return 0.0
        
        # Author diversity
        all_authors = []
        for paper in recommendations:
            all_authors.extend(paper.get('authors', []))
        
        unique_authors = len(set(all_authors))
        total_authors = len(all_authors)
        author_diversity = unique_authors / total_authors if total_authors > 0 else 0.0
        
        # Venue diversity
        venues = [p.get('venue') for p in recommendations if p.get('venue')]
        venue_diversity = (
            len(set(venues)) / len(venues)
            if venues else 1.0
        )
        
        # Domain diversity
        domains = [p.get('domain') for p in recommendations]
        domain_diversity = (
            len(set(domains)) / len(domains)
            if domains else 1.0
        )
        
        # Temporal diversity (year spread)
        years = [p.get('year') for p in recommendations if p.get('year')]
        if years:
            year_span = max(years) - min(years)
            temporal_diversity = min(year_span / 10, 1.0)  # Normalize to 10-year span
        else:
            temporal_diversity = 0.0
        
        # Combined diversity score
        diversity_score = (
            author_diversity * 0.3 +
            venue_diversity * 0.2 +
            domain_diversity * 0.3 +
            temporal_diversity * 0.2
        )
        
        logger.debug(
            "Diversity score calculated",
            score=diversity_score,
            author_div=author_diversity,
            domain_div=domain_diversity
        )
        
        return diversity_score