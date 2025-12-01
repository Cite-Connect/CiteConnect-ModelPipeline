"""
Batch evaluation script for CiteConnect recommendations.
Evaluates all cold-start users and generates comprehensive report.
"""
import asyncio
import sys
from pathlib import Path
import json
from datetime import datetime
from typing import Optional 

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.utils.logger import setup_logging, get_logger
from app.db.connection import db
from app.services.recommendation_service import RecommendationService
from app.services.evaluation_service import EvaluationService
from app.db.repositories.evaluation_repo import EvaluationRepository

setup_logging()
logger = get_logger(__name__)


async def evaluate_all_cold_start_users(
    model: str = 'minilm',
    max_users: Optional[int] = None
):
    """
    Evaluate all cold-start users and generate report.
    
    Args:
        model: Embedding model to use
        max_users: Optional limit on number of users to evaluate
    """
    logger.info("="*70)
    logger.info("BATCH COLD-START EVALUATION")
    logger.info("="*70)
    
    await db.connect()

     # Generate run ID
    run_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # Get current weights from recommendation service
    from app.services.recommendation_service import RecommendationService
    current_weights = RecommendationService.DEFAULT_COLD_START_WEIGHTS
    
    # INSERT experiment run (START)
    await db.execute("""
        INSERT INTO experiment_runs (
            run_id,
            embedding_model,
            embedding_dimension,
            hyperparameters,
            experiment_type,
            user_segment,
            status,
            started_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
    """,
    run_id,
    'all-MiniLM-L6-v2' if model == 'minilm' else 'specter2',
    384 if model == 'minilm' else 768,
    json.dumps(current_weights),  # JSON format
    'baseline',
    'cold_start',
    'running'
    )
    
    logger.info(f"Experiment run started: {run_id}")
    try:
        # Get all cold-start users with profiles
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
        
        # Initialize services
        rec_service = RecommendationService(db)
        eval_service = EvaluationService(db)
        
        # Track results
        results = []
        successes = 0
        failures = 0
        
        # Evaluate each user
        for i, user in enumerate(users, 1):
            user_id = user['user_id']
            email = user['email']
            stage = user['research_stage']
            domain = user['primary_domain']
            
            logger.info(
                f"Evaluating user {i}/{len(users)}",
                user_id=user_id,
                email=email,
                stage=stage,
                domain=domain
            )
            
            try:
                # Generate recommendations
                rec_result = await rec_service.generate_cold_start_recommendations(
                    user_id=user_id,
                    count=10,
                    model=model
                )
                
                # Evaluate recommendations
                eval_result = await eval_service.evaluate_cold_start_recommendations(
                    user_id=user_id,
                    recommendations=rec_result['papers'],
                    model=model,
                    store_result=True
                )
                
                # Track result
                results.append({
                    'user_id': user_id,
                    'email': email,
                    'research_stage': stage,
                    'primary_domain': domain,
                    'reading_level': user['reading_level'],
                    'profile_alignment': eval_result['profile_alignment'],
                    'ground_truth_quality': eval_result['ground_truth_quality'],
                    'combined_score': eval_result['combined_score'],
                    'passes': eval_result['passes_threshold']
                })
                
                successes += 1
                
                logger.info(
                    "Evaluation complete",
                    user_id=user_id,
                    combined_score=eval_result['combined_score'],
                    passes=eval_result['passes_threshold']
                )
                
            except Exception as e:
                logger.error(
                    "Evaluation failed",
                    user_id=user_id,
                    error=str(e),
                    exc_info=True
                )
                failures += 1
        
        # Generate summary report
        logger.info("="*70)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*70)
        
        print(f"\nTotal users evaluated: {len(users)}")
        print(f"Successful: {successes}")
        print(f"Failed: {failures}")
        
        logger.info(f"Experiment run completed: {run_id}")
        if results:
            # Overall statistics
            profile_scores = [r['profile_alignment'] for r in results]
            gt_scores = [r['ground_truth_quality'] for r in results]
            combined_scores = [r['combined_score'] for r in results]
            passed = sum(1 for r in results if r['passes'])
            
            print(f"\n📊 OVERALL METRICS:")
            print(f"   Avg Profile Alignment:    {sum(profile_scores)/len(profile_scores):.4f}")
            print(f"   Avg Ground Truth Quality: {sum(gt_scores)/len(gt_scores):.4f}")
            print(f"   Avg Combined Score:       {sum(combined_scores)/len(combined_scores):.4f}")
            print(f"   Pass Rate:                {passed}/{len(results)} ({passed/len(results)*100:.1f}%)")
            print(f"   Score Range:              [{min(combined_scores):.3f}, {max(combined_scores):.3f}]")
            
            # Bias analysis by research stage
            print(f"\n📈 BIAS ANALYSIS (Research Stage):")
            stages = {}
            for r in results:
                stage = r['research_stage'] or 'unknown'
                if stage not in stages:
                    stages[stage] = []
                stages[stage].append(r['combined_score'])
            
            for stage, scores in stages.items():
                avg = sum(scores) / len(scores)
                print(f"   {stage:15s}: {avg:.4f} ({len(scores)} users)")
            
            # Calculate bias magnitude
            if len(stages) > 1:
                stage_avgs = [sum(s)/len(s) for s in stages.values()]
                bias_magnitude = max(stage_avgs) - min(stage_avgs)
                print(f"\n   Bias Magnitude: {bias_magnitude:.4f}")
                if bias_magnitude > 0.20:
                    print(f"   ⚠️  BIAS DETECTED (threshold: 0.20)")
                else:
                    print(f"   ✅ No significant bias")
            
            # Bias analysis by domain
            print(f"\n📈 BIAS ANALYSIS (Domain):")
            domains = {}
            for r in results:
                domain = r['primary_domain'] or 'unknown'
                if domain not in domains:
                    domains[domain] = []
                domains[domain].append(r['combined_score'])
            
            for domain, scores in domains.items():
                avg = sum(scores) / len(scores)
                print(f"   {domain:20s}: {avg:.4f} ({len(scores)} users)")
            
            # Bias analysis by reading level
            print(f"\n📈 BIAS ANALYSIS (Reading Level):")
            levels = {}
            for r in results:
                level = r['reading_level'] or 'unknown'
                if level not in levels:
                    levels[level] = []
                levels[level].append(r['combined_score'])
            
            for level, scores in levels.items():
                avg = sum(scores) / len(scores)
                print(f"   {level:15s}: {avg:.4f} ({len(scores)} users)")
            
            # Save detailed results to JSON
            output_file = f"evaluation_results_{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump({
                    'model': model,
                    'total_users': len(users),
                    'successful_evaluations': successes,
                    'failed_evaluations': failures,
                    'aggregate_metrics': {
                        'avg_profile_alignment': sum(profile_scores)/len(profile_scores),
                        'avg_ground_truth_quality': sum(gt_scores)/len(gt_scores),
                        'avg_combined_score': sum(combined_scores)/len(combined_scores),
                        'pass_rate': passed/len(results)
                    },
                    'bias_analysis': {
                        'by_research_stage': {
                            stage: sum(scores)/len(scores)
                            for stage, scores in stages.items()
                        },
                        'by_domain': {
                            domain: sum(scores)/len(scores)
                            for domain, scores in domains.items()
                        },
                        'by_reading_level': {
                            level: sum(scores)/len(scores)
                            for level, scores in levels.items()
                        }
                    },
                    'individual_results': results,
                    'evaluated_at': datetime.utcnow().isoformat()
                }, f, indent=2)
            
            logger.info(f"Results saved to {output_file}")
            print(f"\n💾 Detailed results saved to: {output_file}")

        # UPDATE experiment run (COMPLETE)
        await db.execute("""
            UPDATE experiment_runs
            SET status = 'completed',
                ended_at = NOW(),
                results = $1
            WHERE run_id = $2
        """,
        json.dumps({
            'total_users': len(users),
            'successful_evaluations': successes,
            'avg_combined_score': sum(combined_scores)/len(combined_scores),
            'avg_profile_alignment': sum(profile_scores)/len(profile_scores),
            'avg_ground_truth_quality': sum(gt_scores)/len(gt_scores),
            'pass_rate': passed/len(results),
            'bias_magnitude': bias_magnitude
        }),
        run_id
        )

        logger.info("="*70)
        logger.info("BATCH EVALUATION COMPLETE")
        logger.info("="*70)
        
    finally:
        await db.disconnect()


async def compare_models():
    """
    Compare MiniLM vs SPECTER on same users.
    """
    logger.info("="*70)
    logger.info("MODEL COMPARISON (MiniLM vs SPECTER)")
    logger.info("="*70)
    
    await db.connect()
    
    try:
        eval_repo = EvaluationRepository(db)
        
        # Get comparison
        comparison = await eval_repo.get_model_comparison(
            model_a='all-MiniLM-L6-v2',
            model_b='specter2'
        )
        
        print(f"\n📊 MODEL COMPARISON RESULTS:")
        print(f"   Model A: {comparison['model_a']}")
        print(f"   Model B: {comparison['model_b']}")
        
        if 'all-MiniLM-L6-v2' in comparison['results']:
            res_a = comparison['results']['all-MiniLM-L6-v2']
            print(f"\n   MiniLM:")
            print(f"      Evaluations: {res_a['evaluation_count']}")
            print(f"      Avg Score:   {res_a['avg_combined_score']:.4f}")
            print(f"      Std Dev:     {res_a['std_combined_score']:.4f}")
        
        if 'specter2' in comparison['results']:
            res_b = comparison['results']['specter2']
            print(f"\n   SPECTER:")
            print(f"      Evaluations: {res_b['evaluation_count']}")
            print(f"      Avg Score:   {res_b['avg_combined_score']:.4f}")
            print(f"      Std Dev:     {res_b['std_combined_score']:.4f}")
        
        print(f"\n   Winner: {comparison.get('winner', 'Unknown')}")
        print(f"   Confidence: {comparison.get('confidence', 'Unknown')}")
        print(f"   Score Difference: {comparison.get('score_difference', 0):.4f}")

        # Get both experiment runs
        run_a = await db.fetchrow("""
            SELECT * FROM experiment_runs 
            WHERE embedding_model = 'all-MiniLM-L6-v2'
            ORDER BY started_at DESC LIMIT 1
        """)

        run_b = await db.fetchrow("""
            SELECT * FROM experiment_runs 
            WHERE embedding_model = 'specter2'
            ORDER BY started_at DESC LIMIT 1
        """)

        # Statistical comparison
        score_a = run_a['results']['avg_combined_score']
        score_b = run_b['results']['avg_combined_score']

        # t-test or similar
        p_value = calculate_p_value(...)
        winner = 'all-MiniLM-L6-v2' if score_a > score_b else 'specter2'

        # INSERT A/B test result
        await db.execute("""
            INSERT INTO ab_test_comparisons (
                test_name,
                model_a, model_a_run_id,
                model_b, model_b_run_id,
                winner, p_value, confidence_level,
                status
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """,
        'minilm_vs_specter_baseline',
        'all-MiniLM-L6-v2', run_a['run_id'],
        'specter2', run_b['run_id'],
        winner, p_value, 0.95,
        'completed'
        )
        
    finally:
        await db.disconnect()


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch evaluate CiteConnect recommendations')
    parser.add_argument('--model', type=str, default='minilm', choices=['minilm', 'specter'],
                       help='Embedding model to use')
    parser.add_argument('--max-users', type=int, default=None,
                       help='Maximum number of users to evaluate')
    parser.add_argument('--compare', action='store_true',
                       help='Compare MiniLM vs SPECTER models')
    
    args = parser.parse_args()
    
    try:
        if args.compare:
            await compare_models()
        else:
            await evaluate_all_cold_start_users(
                model=args.model,
                max_users=args.max_users
            )
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(
            "Batch evaluation failed",
            error=str(e),
            exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())