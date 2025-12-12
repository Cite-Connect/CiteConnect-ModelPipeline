"""
Batch evaluation script for CiteConnect recommendations.
Evaluates both Cold-Start (Profile) and Warm-Start (Hold-Out) users.
Generates comprehensive reports and stores results in experiment_runs.
Includes statistical significance testing (A/B testing).
"""
import asyncio
import sys
from pathlib import Path
import json
from datetime import datetime
from typing import Optional, List, Dict
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


async def evaluate_users(
    model: str = 'minilm',
    segment: str = 'cold_start',
    max_users: Optional[int] = None
):
    """
    Evaluate users (Cold-Start or Warm-Start) and log to DB.
    
    Segment Logic:
    - 'cold_start': Uses Profile Alignment & GT Quality metrics.
    - 'early', 'mature', 'expert', 'all': Uses Hold-Out Strategy (Precision/Recall).
    """
    logger.info("="*70)
    logger.info(f"BATCH EVALUATION: {segment.upper()} | Model: {model.upper()}")
    logger.info("="*70)
    
    await db.connect()

    # Generate run ID
    run_id = f"exp_{segment}_{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Select weights based on whether we are in cold or warm start logic
    if segment == 'cold_start':
        current_weights = RecommendationService.DEFAULT_COLD_START_WEIGHTS
    else:
        current_weights = RecommendationService.DEFAULT_WARM_START_WEIGHTS
    
    # Start Experiment Run
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
    'baseline', segment, 'running'
    )
    
    logger.info(f"Experiment run started: {run_id}")
    
    rec_service = RecommendationService(db)
    eval_service = EvaluationService(db)
    
    try:
        # 1. Select Users based on Segment
        if segment == 'cold_start':
            # COLD START QUERY
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
        else:
            # WARM START QUERY (Early, Mature, Expert, All)
            # Must have at least 5 saved papers for hold-out split
            base_query = """
                SELECT u.user_id, u.email, p.research_stage, 
                       p.primary_domain, p.reading_level, COUNT(usp.paper_id) as saved_count
                FROM users u
                JOIN user_profiles_extended p ON u.user_id = p.user_id
                JOIN user_recommendation_state s ON u.user_id = s.user_id
                JOIN user_saved_papers usp ON u.user_id = usp.user_id
                WHERE u.is_active = true
            """
            
            # Filter by specific segment if not 'all'
            if segment != 'all':
                base_query += f" AND s.recommendation_stage = '{segment}'"
            else:
                # 'all' implies all warm start users (exclude cold_start explicitly if needed, 
                # though HAVING count >= 5 usually implies warm start)
                base_query += " AND s.recommendation_stage != 'cold_start'"

            query = base_query + """
                GROUP BY u.user_id, u.email, p.research_stage, p.primary_domain, p.reading_level
                HAVING COUNT(usp.paper_id) >= 5
                ORDER BY u.user_id
            """

        if max_users:
            query += f" LIMIT {max_users}"
        
        users = await db.fetch(query)
        if not users:
            logger.warning(f"No eligible '{segment}' users found")
            return
        
        logger.info(f"Found {len(users)} users to evaluate")
        
        results = []
        successes = 0
        failures = 0
        
        # 2. Process Each User
        for i, user in enumerate(users, 1):
            user_id = user['user_id']
            if i % 10 == 0 or i == 1:
                logger.info(f"Processing user {i}/{len(users)} (ID: {user_id})")
            
            try:
                eval_metrics = {}
                
                # --- COLD START LOGIC ---
                if segment == 'cold_start':
                    rec_result = await rec_service.generate_cold_start_recommendations(
                        user_id=user_id, count=10, model=model
                    )
                    eval_result = await eval_service.evaluate_cold_start_recommendations(
                        user_id=user_id, recommendations=rec_result['papers'],
                        model=model, store_result=True
                    )
                    
                    eval_metrics = {
                        'score': eval_result['combined_score'],
                        'profile_alignment': eval_result['profile_alignment'],
                        'ground_truth_quality': eval_result['ground_truth_quality'],
                        'passes': eval_result['passes_threshold']
                    }

                # --- WARM START LOGIC (Hold-Out) ---
                else:
                    # Fetch all saved papers
                    all_saved = await rec_service.user_repo.get_saved_papers_list(user_id)
                    
                    # 80/20 Split
                    split_idx = int(len(all_saved) * 0.8)
                    context_set = [p['paper_id'] for p in all_saved[:split_idx]] # Past
                    target_set = [p['paper_id'] for p in all_saved[split_idx:]]  # Future
                    
                    if not target_set: continue

                    # Generate using only Context (Context Injection)
                    rec_result = await rec_service.generate_warm_start_recommendations(
                        user_id=user_id,
                        count=10,
                        model=model,
                        context_paper_ids=context_set 
                    )
                    
                    # Evaluate against Target (Hold-Out)
                    eval_result = await eval_service.evaluate_warm_start_recommendations(
                        user_id=user_id,
                        recommendations=rec_result['papers'],
                        ground_truth_papers=target_set, # Pass hold-out set explicitly
                        store_result=True
                    )
                    
                    # For warm start, using NDCG as primary score for aggregate tracking
                    eval_metrics = {
                        'score': eval_result['ndcg_at_10'], 
                        'precision': eval_result['precision_at_10'],
                        'recall': eval_result['recall_at_10'],
                        'ndcg': eval_result['ndcg_at_10'],
                        'passes': eval_result['recall_at_10'] > 0 
                    }

                # Add common metadata
                results.append({
                    'user_id': user_id,
                    'email': user['email'],
                    'research_stage': user['research_stage'],
                    **eval_metrics
                })
                successes += 1
                
            except Exception as e:
                logger.error(f"User {user_id} failed: {e}")
                failures += 1
        
        # 3. Calculate Aggregate Statistics
        scores = [r['score'] for r in results] if results else []
        avg_score = np.mean(scores) if scores else 0.0
        std_score = np.std(scores) if scores else 0.0
        pass_rate = sum(1 for r in results if r['passes']) / len(results) if results else 0.0
        
        # Determine bias magnitude
        stages = {}
        for r in results:
            stg = r.get('research_stage') or 'unknown'
            if stg not in stages: stages[stg] = []
            stages[stg].append(r['score'])
            
        bias_magnitude = 0.0
        if len(stages) > 1:
            stage_avgs = [np.mean(s) for s in stages.values()]
            if stage_avgs:
                bias_magnitude = max(stage_avgs) - min(stage_avgs)

        # Prepare results JSON structure
        result_json = {
            'total_users': len(users),
            'successful_evaluations': successes,
            'avg_combined_score': float(avg_score), # Maps to score for both types
            'std_combined_score': float(std_score),
            'pass_rate': float(pass_rate),
            'bias_magnitude': float(bias_magnitude)
        }
        
        # Add segment-specific metrics to JSON
        if segment == 'cold_start':
            result_json.update({
                'avg_profile_alignment': float(np.mean([r['profile_alignment'] for r in results])) if results else 0.0,
                'avg_ground_truth_quality': float(np.mean([r['ground_truth_quality'] for r in results])) if results else 0.0
            })
        else:
            result_json.update({
                'avg_precision': float(np.mean([r['precision'] for r in results])) if results else 0.0,
                'avg_recall': float(np.mean([r['recall'] for r in results])) if results else 0.0,
                'avg_ndcg': float(np.mean([r['ndcg'] for r in results])) if results else 0.0
            })

        # 4. Save to DB
        await db.execute("""
            UPDATE experiment_runs
            SET status = 'completed', ended_at = NOW(), results = $1
            WHERE run_id = $2
        """,
        json.dumps(result_json),
        run_id
        )

        logger.info("="*70)
        logger.info(f"EVALUATION COMPLETE: {segment.upper()} - {model.upper()}")
        logger.info(f"Avg Score: {avg_score:.4f} (Std: {std_score:.4f})")
        if segment != 'cold_start':
            logger.info(f"Precision: {result_json['avg_precision']:.4f} | Recall: {result_json['avg_recall']:.4f}")
        logger.info("="*70)
        
    finally:
        await db.disconnect()


async def compare_models(segment: str = 'cold_start'):
    """Compare MiniLM vs SPECTER for a specific segment."""
    logger.info("="*70)
    logger.info(f"MODEL COMPARISON: {segment.upper()} (MiniLM vs SPECTER)")
    logger.info("="*70)
    
    await db.connect()
    
    try:
        # Fetch last completed run for each model in this segment
        run_a = await db.fetchrow("""
            SELECT * FROM experiment_runs 
            WHERE embedding_model = 'all-MiniLM-L6-v2' 
              AND user_segment = $1 
              AND status = 'completed'
            ORDER BY started_at DESC LIMIT 1
        """, segment)
        
        run_b = await db.fetchrow("""
            SELECT * FROM experiment_runs 
            WHERE embedding_model = 'specter' 
              AND user_segment = $1 
              AND status = 'completed'
            ORDER BY started_at DESC LIMIT 1
        """, segment)

        if not run_a or not run_b:
            logger.error(f"Could not find completed runs for both models in segment '{segment}'.")
            return

        stats_a = json.loads(run_a['results'])
        stats_b = json.loads(run_b['results'])

        print(f"\n📊 MODEL A: {run_a['embedding_model']}")
        print(f"   Score: {stats_a['avg_combined_score']:.4f} ± {stats_a.get('std_combined_score', 0):.4f}")
        
        print(f"\n📊 MODEL B: {run_b['embedding_model']}")
        print(f"   Score: {stats_b['avg_combined_score']:.4f} ± {stats_b.get('std_combined_score', 0):.4f}")

        # Calculate Significance
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

        # Save Result to AB Test Table
        test_name = f"minilm_vs_specter_{segment}"
        
        await db.execute("""
            INSERT INTO ab_test_comparisons (
                test_name, model_a, model_a_run_id, model_b, model_b_run_id,
                winner, p_value, confidence_level, status, started_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
        """,
        test_name,
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
    # Updated choices to match database constraint
    parser.add_argument('--segment', type=str, default='cold_start', 
                        choices=['cold_start', 'early', 'mature', 'expert', 'all'], 
                        help='User segment to evaluate')
    parser.add_argument('--max-users', type=int, default=None, help='Maximum number of users to evaluate')
    parser.add_argument('--compare', action='store_true', help='Compare MiniLM vs SPECTER models')
    
    args = parser.parse_args()
    
    try:
        if args.compare:
            await compare_models(segment=args.segment)
        else:
            await evaluate_users(
                model=args.model, 
                segment=args.segment,
                max_users=args.max_users
            )
    except Exception as e:
        logger.error("Batch evaluation failed", error=str(e), exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())