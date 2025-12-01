"""
Batch evaluation script for CiteConnect recommendations.
Evaluates all cold-start users and generates comprehensive report.
Includes statistical significance testing (A/B testing).
"""
import asyncio
import sys
from pathlib import Path
import json
from datetime import datetime
from typing import Optional
import numpy as np
from scipy import stats

# Add parent directory to path so we can import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.utils.logger import setup_logging, get_logger
from app.db.connection import db
from app.services.recommendation_service import RecommendationService
from app.services.evaluation_service import EvaluationService
from app.db.repositories.evaluation_repo import EvaluationRepository

setup_logging()
logger = get_logger(__name__)


def calculate_p_value(mean_a: float, std_a: float, n_a: int, 
                     mean_b: float, std_b: float, n_b: int) -> float:
    """
    Calculate p-value using Welch's t-test (for unequal variances).
    """
    if n_a <= 1 or n_b <= 1 or std_a == 0 or std_b == 0:
        return 1.0

    se_a = std_a**2 / n_a
    se_b = std_b**2 / n_b
    
    if se_a + se_b == 0:
        return 1.0
        
    t_stat = (mean_a - mean_b) / np.sqrt(se_a + se_b)
    
    numerator = (se_a + se_b)**2
    denominator = (se_a**2 / (n_a - 1)) + (se_b**2 / (n_b - 1))
    
    if denominator == 0:
        return 1.0
        
    df = numerator / denominator
    p_value = 2 * stats.t.sf(abs(t_stat), df)
    
    return float(p_value)


async def evaluate_all_cold_start_users(
    model: str = 'minilm',
    max_users: Optional[int] = None
):
    """
    Evaluate all cold-start users and generate report.
    """
    logger.info("="*70)
    logger.info(f"BATCH COLD-START EVALUATION: {model.upper()}")
    logger.info("="*70)
    
    await db.connect()

    # Generate run ID
    run_id = f"exp_{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    current_weights = RecommendationService.DEFAULT_COLD_START_WEIGHTS
    
    await db.execute("""
        INSERT INTO experiment_runs (
            run_id, embedding_model, embedding_dimension, hyperparameters,
            experiment_type, user_segment, status, started_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
    """,
    run_id,
    'all-MiniLM-L6-v2' if model == 'minilm' else 'specter2',
    384 if model == 'minilm' else 768,
    json.dumps(current_weights),
    'baseline', 'cold_start', 'running'
    )
    
    logger.info(f"Experiment run started: {run_id}")
    
    try:
        query = """
            SELECT DISTINCT u.user_id, u.email, p.research_stage, 
                   p.primary_domain, p.reading_level
            FROM users u
            JOIN user_profiles_extended p ON u.user_id = p.user_id
            JOIN user_recommendation_state s ON u.user_id = s.user_id
            WHERE s.recommendation_stage = 'cold_start'
              AND u.is_active = true
            ORDER BY u.user_id
        """
        if max_users:
            query += f" LIMIT {max_users}"
        
        users = await db.fetch(query)
        if not users:
            logger.warning("No cold-start users found")
            return
        
        logger.info(f"Found {len(users)} cold-start users to evaluate")
        rec_service = RecommendationService(db)
        eval_service = EvaluationService(db)
        
        results = []
        successes = 0
        failures = 0
        
        for i, user in enumerate(users, 1):
            user_id = user['user_id']
            if i % 10 == 0 or i == 1:
                logger.info(f"Processing user {i}/{len(users)} (ID: {user_id})")
            
            try:
                rec_result = await rec_service.generate_cold_start_recommendations(
                    user_id=user_id, count=10, model=model
                )
                eval_result = await eval_service.evaluate_cold_start_recommendations(
                    user_id=user_id, recommendations=rec_result['papers'],
                    model=model, store_result=True
                )
                
                results.append({
                    'user_id': user_id,
                    'email': user['email'],
                    'research_stage': user['research_stage'],
                    'primary_domain': user['primary_domain'],
                    'reading_level': user['reading_level'],
                    'profile_alignment': eval_result['profile_alignment'],
                    'ground_truth_quality': eval_result['ground_truth_quality'],
                    'combined_score': eval_result['combined_score'],
                    'passes': eval_result['passes_threshold']
                })
                successes += 1
            except Exception as e:
                logger.error(f"User {user_id} failed: {e}")
                failures += 1
        
        # Calculate statistics gracefully
        combined_scores = [r['combined_score'] for r in results] if results else []
        profile_scores = [r['profile_alignment'] for r in results] if results else []
        gt_scores = [r['ground_truth_quality'] for r in results] if results else []
        passed = sum(1 for r in results if r['passes']) if results else 0
        
        avg_combined = np.mean(combined_scores) if combined_scores else 0.0
        std_combined = np.std(combined_scores) if combined_scores else 0.0
        
        # Identify bias
        stages = {}
        for r in results:
            stage = r['research_stage'] or 'unknown'
            if stage not in stages: stages[stage] = []
            stages[stage].append(r['combined_score'])
            
        # FIX: Initialize bias_magnitude safely to avoid UnboundLocalError
        bias_magnitude = 0.0
        if len(stages) > 1:
            stage_avgs = [np.mean(s) for s in stages.values()]
            if stage_avgs:
                bias_magnitude = max(stage_avgs) - min(stage_avgs)

        await db.execute("""
            UPDATE experiment_runs
            SET status = 'completed', ended_at = NOW(), results = $1
            WHERE run_id = $2
        """,
        json.dumps({
            'total_users': len(users),
            'successful_evaluations': successes,
            'avg_combined_score': float(avg_combined),
            'std_combined_score': float(std_combined),
            'avg_profile_alignment': float(np.mean(profile_scores)) if profile_scores else 0.0,
            'avg_ground_truth_quality': float(np.mean(gt_scores)) if gt_scores else 0.0,
            'pass_rate': passed/len(results) if results else 0.0,
            'bias_magnitude': float(bias_magnitude)
        }),
        run_id
        )

        logger.info("="*70)
        logger.info(f"EVALUATION COMPLETE: {model}")
        logger.info(f"Avg Score: {avg_combined:.4f} (Std: {std_combined:.4f})")
        logger.info("="*70)
        
    finally:
        await db.disconnect()


async def compare_models():
    """Compare MiniLM vs SPECTER on same users."""
    logger.info("="*70)
    logger.info("MODEL COMPARISON (MiniLM vs SPECTER)")
    logger.info("="*70)
    
    await db.connect()
    
    try:
        run_a = await db.fetchrow("""
            SELECT * FROM experiment_runs 
            WHERE embedding_model = 'all-MiniLM-L6-v2' AND status = 'completed'
            ORDER BY started_at DESC LIMIT 1
        """)
        run_b = await db.fetchrow("""
            SELECT * FROM experiment_runs 
            WHERE embedding_model = 'specter2' AND status = 'completed'
            ORDER BY started_at DESC LIMIT 1
        """)

        if not run_a or not run_b:
            logger.error("Could not find completed runs for both models.")
            return

        stats_a = json.loads(run_a['results'])
        stats_b = json.loads(run_b['results'])

        print(f"\n📊 MODEL A: {run_a['embedding_model']}")
        print(f"   Score: {stats_a['avg_combined_score']:.4f} ± {stats_a.get('std_combined_score', 0):.4f}")
        print(f"\n📊 MODEL B: {run_b['embedding_model']}")
        print(f"   Score: {stats_b['avg_combined_score']:.4f} ± {stats_b.get('std_combined_score', 0):.4f}")

        p_value = calculate_p_value(
            mean_a=stats_a['avg_combined_score'],
            std_a=stats_a.get('std_combined_score', 0),
            n_a=stats_a['successful_evaluations'],
            mean_b=stats_b['avg_combined_score'],
            std_b=stats_b.get('std_combined_score', 0),
            n_b=stats_b['successful_evaluations']
        )
        
        score_a = stats_a['avg_combined_score']
        score_b = stats_b['avg_combined_score']
        
        winner = run_a['embedding_model'] if score_a > score_b else run_b['embedding_model']
        if abs(score_a - score_b) < 0.01: 
            winner = "Tie"

        significance = "SIGNIFICANT" if p_value < 0.05 else "NOT SIGNIFICANT"
        
        print(f"\n🏆 WINNER: {winner}")
        print(f"   P-Value: {p_value:.5f} ({significance})")

        # CRITICAL FIX: Changed created_at to started_at to match DB schema
        await db.execute("""
            INSERT INTO ab_test_comparisons (
                test_name, model_a, model_a_run_id, model_b, model_b_run_id,
                winner, p_value, confidence_level, status, started_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
        """,
        'minilm_vs_specter_baseline',
        run_a['embedding_model'], run_a['run_id'],
        run_b['embedding_model'], run_b['run_id'],
        winner, p_value, 0.95, 'completed'
        )
        logger.info(f"Comparison saved. Winner: {winner}, p={p_value:.4f}")
        
    finally:
        await db.disconnect()


async def main():
    import argparse
    parser = argparse.ArgumentParser(description='Batch evaluate CiteConnect recommendations')
    parser.add_argument('--model', type=str, default='minilm', choices=['minilm', 'specter'], help='Embedding model to use')
    parser.add_argument('--max-users', type=int, default=None, help='Maximum number of users to evaluate')
    parser.add_argument('--compare', action='store_true', help='Compare MiniLM vs SPECTER models')
    args = parser.parse_args()
    
    try:
        if args.compare:
            await compare_models()
        else:
            await evaluate_all_cold_start_users(model=args.model, max_users=args.max_users)
    except Exception as e:
        logger.error("Batch evaluation failed", error=str(e), exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())