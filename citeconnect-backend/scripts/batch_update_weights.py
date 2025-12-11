"""
Batch Weight Update Script (The "Learning" Step).
Analyzes recent user interactions to calculate personalized scoring weights.
Updates 'user_recommendation_state' and handles stage transitions based on interaction counts.

Transitions:
- cold_start (0-9) -> early (10-49) -> mature (50-199) -> expert (200+)
"""
import asyncio
import os
import sys
import json

# Add project root to python path to allow imports
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.db.connection import DatabaseConnection
from app.utils.logger import get_logger

logger = get_logger("batch_weight_update")

# Configuration
# Only calculate weights if they have at least 5 interactions (to avoid noise)
# However, they won't leave 'cold_start' until they hit 10.
MIN_INTERACTIONS_THRESHOLD = 5 
POSITIVE_STRENGTH_THRESHOLD = 0.3  

async def update_user_weights(domain: str = None):
    db = DatabaseConnection()
    await db.connect()
    
    try:
        logger.info("Starting batch weight update...")
        
        query = """
        WITH calculated_weights AS (
            SELECT 
                ui.user_id,
                AVG(COALESCE((ui.context->'score_breakdown'->>'semantic')::numeric, 0)) AS raw_semantic,
                AVG(COALESCE((ui.context->'score_breakdown'->>'citation')::numeric, 0)) AS raw_citation,
                AVG(COALESCE((ui.context->'score_breakdown'->>'recency')::numeric, 0)) AS raw_recency,
                AVG(COALESCE((ui.context->'score_breakdown'->>'reading_level')::numeric, 0)) AS raw_reading,
                AVG(COALESCE(
                    COALESCE(ui.context->'score_breakdown'->>'citation_network', 
                             ui.context->'score_breakdown'->>'ground_truth')::numeric, 
                    0
                )) AS raw_ground_truth,
                COUNT(*) as valid_interaction_count
            FROM 
                user_interactions ui
            WHERE 
                ui.interaction_strength >= $1
                AND ui.context IS NOT NULL 
                AND (ui.context->>'score_breakdown') IS NOT NULL
            GROUP BY 
                ui.user_id
            HAVING COUNT(*) >= $2
        ),
        final_weights AS (
            SELECT 
                user_id,
                (raw_semantic + raw_citation + raw_recency + raw_reading + raw_ground_truth) as total_score,
                raw_semantic, raw_citation, raw_recency, raw_reading, raw_ground_truth,
                valid_interaction_count
            FROM 
                calculated_weights
            WHERE 
                (raw_semantic + raw_citation + raw_recency + raw_reading + raw_ground_truth) > 0
        )
        UPDATE user_recommendation_state urs
        SET 
            scoring_weights = jsonb_build_object(
                'semantic',      ROUND((fw.raw_semantic / fw.total_score * 0.95)::numeric, 4),
                'citation',      ROUND((fw.raw_citation / fw.total_score * 0.95)::numeric, 4),
                'recency',       ROUND((fw.raw_recency / fw.total_score * 0.95)::numeric, 4),
                'ground_truth',  ROUND((fw.raw_ground_truth / fw.total_score * 0.95)::numeric, 4),
                'reading_level', ROUND((fw.raw_reading / fw.total_score * 0.95)::numeric, 4),
                'diversity',     0.05
            ),
            last_retrained_at = NOW(),
            
            -- Sync interaction count from source of truth (user_interactions table)
            interaction_count = (SELECT count(*) FROM user_interactions WHERE user_id = urs.user_id),
            
            -- DYNAMIC STAGE TRANSITION based on strict count logic
            recommendation_stage = CASE 
                WHEN (SELECT count(*) FROM user_interactions WHERE user_id = urs.user_id) >= 200 THEN 'expert'
                WHEN (SELECT count(*) FROM user_interactions WHERE user_id = urs.user_id) >= 50 THEN 'mature'
                WHEN (SELECT count(*) FROM user_interactions WHERE user_id = urs.user_id) >= 10 THEN 'early'
                ELSE 'cold_start' -- If < 10, they stay cold_start (even if we updated weights)
            END
            
        FROM 
            final_weights fw
        WHERE 
            urs.user_id = fw.user_id;
        """
        
        result = await db.execute(query, POSITIVE_STRENGTH_THRESHOLD, MIN_INTERACTIONS_THRESHOLD)
        updated_count = int(result.split()[-1]) if result else 0
        
        logger.info(f"Weight update complete. Updated {updated_count} users.")

    except Exception as e:
        logger.error(f"Batch update failed: {e}", exc_info=True)
    finally:
        await db.disconnect()

if __name__ == "__main__":
    asyncio.run(update_user_weights())