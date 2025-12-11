#!/usr/bin/env python3
"""
CI-friendly model validation script.

Validates model performance metrics (Precision@10, Recall@10, MRR) against thresholds.
Exits with appropriate exit codes for CI/CD pipelines.

Thresholds:
- Precision@10 ≥ 0.60
- Recall@10 ≥ 0.75
- MRR ≥ 0.70

Usage:
    python scripts/validate_model_metrics.py

Exit codes:
    0: All metrics meet thresholds
    1: One or more metrics below threshold
    2: Validation error (no data, database error, etc.)
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.connection import DatabaseConnection
from app.services.recommendation_service import RecommendationService
from app.services.evaluation_service import EvaluationService
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Metric thresholds
PRECISION_THRESHOLD = 0.60
RECALL_THRESHOLD = 0.75
MRR_THRESHOLD = 0.70

# Minimum number of users required for validation
MIN_USERS_REQUIRED = 1


def calculate_mrr(recommended_ids: List[str], ground_truth_ids: List[str]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    Args:
        recommended_ids: List of recommended paper IDs (in rank order)
        ground_truth_ids: Set of relevant paper IDs
        
    Returns:
        MRR value (0.0 to 1.0)
    """
    ground_truth_set = set(ground_truth_ids)
    for i, paper_id in enumerate(recommended_ids):
        if paper_id in ground_truth_set:
            return 1.0 / (i + 1)
    return 0.0


async def validate_model_metrics() -> Dict[str, Any]:
    """
    Validate model metrics by running recommendations on test users.
    
    Returns:
        Dict with validation results and metrics
    """
    print("\n" + "=" * 80)
    print("  Model Validation - CI/CD Pipeline")
    print("=" * 80 + "\n")
    
    db = DatabaseConnection()
    
    try:
        await db.connect()
        print("✓ Database connected\n")
        
        # Get test users
        query = """
            SELECT u.user_id, u.email, u.name, ud.domain
            FROM users u
            JOIN user_domains ud ON u.user_id = ud.user_id
            WHERE u.email LIKE '%@example.com'
            ORDER BY u.user_id
            LIMIT 10
        """
        users = await db.fetch(query)
        
        if not users:
            print("⚠ No test users found in database")
            print("  Cannot perform validation without test users.")
            return {
                'success': False,
                'error': 'no_test_users',
                'metrics': {},
                'message': 'No test users found'
            }
        
        print(f"Found {len(users)} test users for validation\n")
        
        # Initialize services
        recommendation_service = RecommendationService(db)
        evaluation_service = EvaluationService(db)
        
        all_metrics = []
        validation_errors = []
        
        # Evaluate each user
        for user in users[:5]:  # Limit to 5 users for CI speed
            user_id = user['user_id']
            email = user['email']
            
            try:
                print(f"  Evaluating user {user_id} ({email})...", end=" ", flush=True)
                
                # Generate recommendations
                recommendations = await recommendation_service.generate_cold_start_recommendations(
                    user_id=user_id,
                    count=10,
                    model='minilm'
                )
                
                if not recommendations or not recommendations.get('papers'):
                    print("⚠ (no recommendations)")
                    continue
                
                rec_papers = recommendations['papers']
                rec_ids = [p['paper_id'] for p in rec_papers]
                
                # Get ground truth for evaluation
                # Use saved papers or ground truth papers as proxy
                saved_query = """
                    SELECT paper_id FROM user_saved_papers WHERE user_id = $1
                """
                saved_papers = await db.fetch(saved_query, user_id)
                ground_truth_ids = [p['paper_id'] for p in saved_papers]
                
                # If no saved papers, try ground truth papers
                if not ground_truth_ids:
                    gt_query = """
                        SELECT paper_id FROM ground_truth_papers
                        WHERE domain = (
                            SELECT domain FROM user_domains WHERE user_id = $1 LIMIT 1
                        )
                        LIMIT 20
                    """
                    gt_papers = await db.fetch(gt_query, user_id)
                    ground_truth_ids = [p['paper_id'] for p in gt_papers]
                
                if not ground_truth_ids:
                    print("⚠ (no ground truth)")
                    # Calculate metrics without ground truth (use score-based proxy)
                    # For CI purposes, we'll use a simplified approach
                    precision = 0.0
                    recall = 0.0
                    mrr = 0.0
                else:
                    # Calculate Precision@10
                    k = min(10, len(rec_ids))
                    top_k_ids = rec_ids[:k]
                    hits = len(set(top_k_ids) & set(ground_truth_ids))
                    precision = hits / k if k > 0 else 0.0
                    
                    # Calculate Recall@10
                    recall = hits / len(ground_truth_ids) if ground_truth_ids else 0.0
                    
                    # Calculate MRR
                    mrr = calculate_mrr(rec_ids, ground_truth_ids)
                
                metrics = {
                    'user_id': user_id,
                    'email': email,
                    'precision_at_10': float(precision),
                    'recall_at_10': float(recall),
                    'mrr': float(mrr),
                    'ground_truth_size': len(ground_truth_ids),
                    'recommendations_count': len(rec_ids)
                }
                
                all_metrics.append(metrics)
                print(f"✓ (P@{metrics['precision_at_10']:.3f}, R@{metrics['recall_at_10']:.3f}, MRR@{metrics['mrr']:.3f})")
                
            except Exception as e:
                error_msg = f"Error evaluating user {user_id}: {str(e)}"
                print(f"✗ ({error_msg})")
                validation_errors.append(error_msg)
                logger.error(error_msg, exc_info=True)
        
        if not all_metrics:
            print("\n⚠ No metrics collected - validation cannot complete")
            return {
                'success': False,
                'error': 'no_metrics',
                'metrics': {},
                'errors': validation_errors,
                'message': 'Failed to collect metrics from any users'
            }
        
        # Calculate aggregate metrics
        avg_precision = np.mean([m['precision_at_10'] for m in all_metrics])
        avg_recall = np.mean([m['recall_at_10'] for m in all_metrics])
        avg_mrr = np.mean([m['mrr'] for m in all_metrics])
        
        # Check thresholds
        precision_pass = avg_precision >= PRECISION_THRESHOLD
        recall_pass = avg_recall >= RECALL_THRESHOLD
        mrr_pass = avg_mrr >= MRR_THRESHOLD
        
        all_pass = precision_pass and recall_pass and mrr_pass
        
        # Print summary
        print("\n" + "-" * 80)
        print("  Validation Results")
        print("-" * 80 + "\n")
        
        print(f"Users evaluated: {len(all_metrics)}")
        print(f"\nAverage Metrics:")
        print(f"  Precision@10: {avg_precision:.3f} {'✓' if precision_pass else '✗'} (threshold: ≥{PRECISION_THRESHOLD})")
        print(f"  Recall@10:    {avg_recall:.3f} {'✓' if recall_pass else '✗'} (threshold: ≥{RECALL_THRESHOLD})")
        print(f"  MRR:          {avg_mrr:.3f} {'✓' if mrr_pass else '✗'} (threshold: ≥{MRR_THRESHOLD})")
        
        print(f"\n{'=' * 80}")
        if all_pass:
            print("  ✓ ALL METRICS MEET THRESHOLDS")
        else:
            print("  ✗ ONE OR MORE METRICS BELOW THRESHOLD")
            if not precision_pass:
                print(f"    - Precision@10 ({avg_precision:.3f}) < {PRECISION_THRESHOLD}")
            if not recall_pass:
                print(f"    - Recall@10 ({avg_recall:.3f}) < {RECALL_THRESHOLD}")
            if not mrr_pass:
                print(f"    - MRR ({avg_mrr:.3f}) < {MRR_THRESHOLD}")
        print("=" * 80 + "\n")
        
        # Prepare results
        results = {
            'success': all_pass,
            'generated_at': datetime.utcnow().isoformat(),
            'thresholds': {
                'precision_at_10': PRECISION_THRESHOLD,
                'recall_at_10': RECALL_THRESHOLD,
                'mrr': MRR_THRESHOLD
            },
            'aggregate_metrics': {
                'precision_at_10': float(avg_precision),
                'recall_at_10': float(avg_recall),
                'mrr': float(avg_mrr),
                'users_evaluated': len(all_metrics)
            },
            'individual_metrics': all_metrics,
            'errors': validation_errors,
            'validation_passed': all_pass
        }
        
        # Save results to file
        results_file = Path(__file__).parent.parent / "model_validation_results.json"
        results_file.write_text(json.dumps(results, indent=2))
        
        # Save text report
        report_file = Path(__file__).parent.parent / "validation_report.txt"
        with report_file.open('w') as f:
            f.write(f"Model Validation Report\n")
            f.write(f"Generated: {results['generated_at']}\n\n")
            f.write(f"Users Evaluated: {len(all_metrics)}\n\n")
            f.write(f"Average Metrics:\n")
            f.write(f"  Precision@10: {avg_precision:.3f} ({'PASS' if precision_pass else 'FAIL'})\n")
            f.write(f"  Recall@10:    {avg_recall:.3f} ({'PASS' if recall_pass else 'FAIL'})\n")
            f.write(f"  MRR:          {avg_mrr:.3f} ({'PASS' if mrr_pass else 'FAIL'})\n\n")
            f.write(f"Overall Status: {'PASS' if all_pass else 'FAIL'}\n")
        
        return results
        
    except Exception as e:
        error_msg = f"Validation error: {str(e)}"
        print(f"\n✗ {error_msg}")
        logger.error(error_msg, exc_info=True)
        return {
            'success': False,
            'error': 'validation_error',
            'message': error_msg,
            'metrics': {}
        }
    
    finally:
        await db.disconnect()


def main():
    """Main entry point for CI validation."""
    try:
        results = asyncio.run(validate_model_metrics())
        
        if results.get('error') == 'no_test_users' or results.get('error') == 'no_metrics':
            # Exit code 2: Validation cannot proceed (missing data)
            sys.exit(2)
        elif results.get('error') == 'validation_error':
            # Exit code 2: Validation error
            sys.exit(2)
        elif results.get('validation_passed', False):
            # Exit code 0: All metrics pass
            sys.exit(0)
        else:
            # Exit code 1: Metrics below threshold
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nUnexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
