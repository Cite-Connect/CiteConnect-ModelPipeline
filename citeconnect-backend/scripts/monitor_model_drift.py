"""
Model Drift Monitor (Weighted Aggregation).
Calculates online metrics (CTR, NDCG) weighted by interaction volume.
Triggers 'batch_update_weights.py' ONLY if the weighted performance drops.
"""
import asyncio
import os
import sys
import json  # <--- Added json import
import numpy as np
from datetime import datetime, timedelta

# Add path
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.db.connection import DatabaseConnection
from scripts.batch_update_weights import update_user_weights
from app.utils.logger import get_logger

logger = get_logger("model_drift_monitor")

# Configuration
RETRAIN_THRESHOLD_NDCG = 0.45  # Weighted NDCG threshold
LOOKBACK_HOURS = 24            # Analyze last 24 hours of data

async def calculate_online_metrics():
    db = DatabaseConnection()
    await db.connect()
    
    try:
        logger.info(f"Analyzing interactions from last {LOOKBACK_HOURS} hours...")
        
        # 1. Fetch Interactions grouped by Session
        query = """
            SELECT 
                ui.user_id,
                ui.context->>'session_id' as session_id,
                p.primary_domain,
                COUNT(*) as interaction_count,
                json_agg(ui.context->>'position') as ranks
            FROM user_interactions ui
            JOIN user_profiles_extended p ON ui.user_id = p.user_id
            WHERE ui.created_at >= NOW() - INTERVAL '$1 hours'
            GROUP BY ui.user_id, ui.context->>'session_id', p.primary_domain
        """
        
        rows = await db.fetch(query.replace('$1', str(LOOKBACK_HOURS)))
        
        if not rows:
            logger.warning("No interactions found in lookback period.")
            return

        # 2. Process Metrics in Memory
        metrics_bucket = {} 

        for row in rows:
            domain = row['primary_domain'] or 'unknown'
            count = row['interaction_count']
            
            # --- FIX STARTS HERE ---
            # Safely parse ranks whether returned as list or string
            raw_ranks = row['ranks']
            if isinstance(raw_ranks, str):
                try:
                    raw_ranks = json.loads(raw_ranks)
                except json.JSONDecodeError:
                    raw_ranks = []
            
            # Filter None and convert to int safely
            ranks = []
            if raw_ranks:
                for r in raw_ranks:
                    if r is not None:
                        try:
                            ranks.append(int(r))
                        except (ValueError, TypeError):
                            continue
            # --- FIX ENDS HERE ---
            
            # --- Binning Logic ---
            if count < 10:
                user_bin = 'low_activity'
            elif 10 <= count <= 20:
                user_bin = 'medium_activity'
            else:
                user_bin = 'high_activity'
            
            # --- Calculate Proxies per Session ---
            
            # Precision Proxy (CTR)
            precision = min(count / 10.0, 1.0)
            
            # NDCG Proxy (Rank Awareness)
            dcg = sum([1.0 / np.log2(r + 1) for r in ranks])
            idcg = sum([1.0 / np.log2(i + 2) for i in range(len(ranks))]) 
            ndcg = dcg / idcg if idcg > 0 else 0.0
            
            # Recall Proxy (Coverage)
            recall = min(len(set(ranks)) / 10.0, 1.0)

            # Store metric + weight
            if domain not in metrics_bucket: metrics_bucket[domain] = {}
            if user_bin not in metrics_bucket[domain]: metrics_bucket[domain][user_bin] = []
            
            metrics_bucket[domain][user_bin].append({
                'precision': precision,
                'recall': recall,
                'ndcg': ndcg,
                'weight': count 
            })

        # 3. Aggregate (Weighted) and Save
        trigger_retrain = False
        
        for domain, bins in metrics_bucket.items():
            for bin_name, session_data in bins.items():
                
                # --- WEIGHTED AVERAGE CALCULATION ---
                total_weight = sum(s['weight'] for s in session_data)
                
                if total_weight == 0:
                    continue

                avg_precision = sum(s['precision'] * s['weight'] for s in session_data) / total_weight
                avg_recall = sum(s['recall'] * s['weight'] for s in session_data) / total_weight
                avg_ndcg = sum(s['ndcg'] * s['weight'] for s in session_data) / total_weight
                
                sample_size = len(session_data)
                
                # Check for Drift
                metric_bad = False
                if bin_name == 'high_activity' and avg_ndcg < RETRAIN_THRESHOLD_NDCG:
                    metric_bad = True
                    trigger_retrain = True
                
                # Insert Snapshot into DB
                await db.execute("""
                    INSERT INTO online_evaluation_metrics 
                    (timestamp, domain, user_bin, precision_proxy, recall_proxy, ndcg_proxy, sample_size, triggered_retrain)
                    VALUES (NOW(), $1, $2, $3, $4, $5, $6, $7)
                """, domain, bin_name, avg_precision, avg_recall, avg_ndcg, sample_size, metric_bad)
                
                logger.info(f"Saved metrics for {domain}/{bin_name}: Weighted NDCG={avg_ndcg:.3f} (Sessions: {sample_size})")

        # 4. Conditional Retraining
        if trigger_retrain:
            logger.warning(f"📉 Model Drift Detected (Weighted NDCG < {RETRAIN_THRESHOLD_NDCG}). Triggering Weight Update...")
            await update_user_weights(domain=None)
        else:
            logger.info("✅ Metrics are healthy. No retraining needed.")

    except Exception as e:
        logger.error(f"Monitoring failed: {e}", exc_info=True)
    finally:
        await db.disconnect()

if __name__ == "__main__":
    asyncio.run(calculate_online_metrics())