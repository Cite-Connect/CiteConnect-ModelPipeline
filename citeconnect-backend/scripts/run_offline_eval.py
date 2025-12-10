import asyncio
import os
import sys

# Add project root to python path so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.connection import DatabaseConnection
from app.services.recommendation_service import RecommendationService
from app.services.evaluation_service import EvaluationService
from app.utils.logger import get_logger

logger = get_logger("offline_eval")

async def run_offline_evaluation(user_sample_size=5):
    """
    Runs offline metrics evaluation using Time-Based Split (Hold-Out).
    Does NOT affect production data.
    """
    # 1. Initialize Services
    db = DatabaseConnection()
    await db.connect()
    
    rec_service = RecommendationService(db)
    eval_service = EvaluationService(db)
    
    try:
        # 2. Find suitable test users (must have enough history to split)
        # We need users with at least 5 saved papers to create a meaningful split
        query = """
            SELECT user_id, COUNT(*) as count 
            FROM user_saved_papers 
            GROUP BY user_id 
            HAVING COUNT(*) >= 5 
            LIMIT $1
        """
        users = await db.fetch(query, user_sample_size)
        
        logger.info(f"Found {len(users)} users eligible for offline evaluation")
        
        total_precision = 0
        total_recall = 0
        
        for record in users:
            user_id = record['user_id']
            
            # 3. FETCH FULL HISTORY
            all_saved = await rec_service.user_repo.get_saved_papers_list(user_id)
            
            # 4. CREATE THE SPLIT (80% Context / 20% Target)
            split_idx = int(len(all_saved) * 0.8)
            
            # The "Past" (Input for model)
            context_set = [p['paper_id'] for p in all_saved[:split_idx]]
            
            # The "Future" (Target for metrics)
            target_set = [p['paper_id'] for p in all_saved[split_idx:]]
            
            if not target_set: 
                continue

            print(f"\nUser {user_id}: Context={len(context_set)}, Target={len(target_set)}")

            # 5. GENERATE (Passing explicit context)
            # This uses the "Backdoor" we added in Step 1
            result = await rec_service.generate_warm_start_recommendations(
                user_id=user_id,
                count=10,
                context_paper_ids=context_set 
            )
            
            recs = result['papers']
            rec_ids = [p['paper_id'] for p in recs]
            
            # 6. EVALUATE (Compare Recs vs Target)
            # We calculate metrics manually here or use eval_service helper
            hits = set(rec_ids) & set(target_set)
            
            precision = len(hits) / len(rec_ids) if rec_ids else 0
            recall = len(hits) / len(target_set) if target_set else 0
            
            print(f"  -> Hits: {len(hits)} / {len(target_set)}")
            print(f"  -> Precision: {precision:.4f}, Recall: {recall:.4f}")
            
            total_precision += precision
            total_recall += recall

        # 7. SUMMARY
        if users:
            avg_p = total_precision / len(users)
            avg_r = total_recall / len(users)
            print(f"\n=== FINAL OFFLINE RESULTS ({len(users)} users) ===")
            print(f"Average Precision: {avg_p:.4f}")
            print(f"Average Recall:    {avg_r:.4f}")
            
    finally:
        await db.disconnect()

if __name__ == "__main__":
    asyncio.run(run_offline_evaluation())