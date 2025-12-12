"""
Model Drift Monitor (Weighted Aggregation).
Calculates online metrics (CTR, NDCG) weighted by interaction volume.
Triggers 'batch_update_weights.py' based on OVERALL PRECISION across ALL activity bins.
Recall = saved_papers / recommended_papers
"""
import asyncio
import os
import sys
import json
import numpy as np
from datetime import datetime, timedelta

# Add path
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.db.connection import DatabaseConnection
from scripts.batch_update_weights import update_user_weights
from app.utils.logger import get_logger

logger = get_logger("model_drift_monitor")

# Configuration
RETRAIN_THRESHOLD_PRECISION = 0.25  # Overall system precision threshold
LOOKBACK_HOURS = 24                 # Analyze last 24 hours of data

async def calculate_online_metrics():
    db = DatabaseConnection()
    await db.connect()
    
    try:
        logger.info(f"Analyzing interactions from last {LOOKBACK_HOURS} hours...")
        
        # 1. Fetch Interactions grouped by Session
        query = """
            WITH session_data AS (
                SELECT 
                    ui.user_id,
                    ui.context->>'session_id' as session_id,
                    p.primary_domain,
                    COUNT(*) as interaction_count,
                    json_agg(ui.context->>'position') as ranks,
                    -- Count saved papers in this session
                    COUNT(CASE WHEN ui.interaction_type = 'save' THEN 1 END) as saved_count,
                    -- Count recommended papers (assume from context or fixed number)
                    COALESCE(
                        (ui.context->>'total_recommended')::int,
                        10  -- Default: assume 10 papers recommended
                    ) as recommended_count
                FROM user_interactions ui
                JOIN user_profiles_extended p ON ui.user_id = p.user_id
                WHERE ui.created_at >= NOW() - INTERVAL '%s hours'
                GROUP BY ui.user_id, ui.context->>'session_id', p.primary_domain, ui.context->>'total_recommended'
            )
            SELECT 
                user_id,
                session_id,
                primary_domain,
                interaction_count,
                ranks,
                saved_count,
                recommended_count
            FROM session_data
        """ % LOOKBACK_HOURS
        
        rows = await db.fetch(query)
        
        if not rows:
            logger.warning("No interactions found in lookback period.")
            return

        # 2. Process Metrics in Memory
        metrics_bucket = {} 
        
        # OPTION 3: Track overall system metrics
        overall_metrics = {
            'total_weight': 0,
            'weighted_precision_sum': 0,
            'weighted_recall_sum': 0,
            'weighted_ndcg_sum': 0,
            'total_sessions': 0
        }

        for row in rows:
            domain = row['primary_domain'] or 'unknown'
            count = row['interaction_count']
            saved_count = row['saved_count']
            recommended_count = row['recommended_count']
            
            # Parse ranks
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
            
            # Binning Logic (for per-bin reporting, not for drift check)
            if count < 10:
                user_bin = 'low_activity'
            elif 10 <= count <= 20:
                user_bin = 'medium_activity'
            else:
                user_bin = 'high_activity'
            
            # --- Calculate Metrics per Session ---
            
            # Precision Proxy (CTR) - clicks / recommendations
            precision = min(count / recommended_count, 1.0) if recommended_count > 0 else 0.0
            
            # Recall Proxy - saved_papers / recommended_papers
            recall = min(saved_count / recommended_count, 1.0) if recommended_count > 0 else 0.0
            
            # NDCG Proxy (Rank Awareness)
            if len(ranks) > 0:
                dcg = sum([1.0 / np.log2(r + 1) for r in ranks])
                idcg = sum([1.0 / np.log2(i + 2) for i in range(len(ranks))]) 
                ndcg = dcg / idcg if idcg > 0 else 0.0
            else:
                ndcg = 0.0

            # Store metric + weight (for per-bin reporting)
            if domain not in metrics_bucket:
                metrics_bucket[domain] = {}
            if user_bin not in metrics_bucket[domain]:
                metrics_bucket[domain][user_bin] = []
            
            metrics_bucket[domain][user_bin].append({
                'precision': precision,
                'recall': recall,
                'ndcg': ndcg,
                'weight': count,
                'saved_count': saved_count,
                'recommended_count': recommended_count
            })
            
            # OPTION 3: Accumulate overall weighted metrics
            weight = count  # Use interaction count as weight
            overall_metrics['total_weight'] += weight
            overall_metrics['weighted_precision_sum'] += precision * weight
            overall_metrics['weighted_recall_sum'] += recall * weight
            overall_metrics['weighted_ndcg_sum'] += ndcg * weight
            overall_metrics['total_sessions'] += 1

        # 3. Calculate Overall System Metrics (OPTION 3)
        if overall_metrics['total_weight'] > 0:
            overall_precision = overall_metrics['weighted_precision_sum'] / overall_metrics['total_weight']
            overall_recall = overall_metrics['weighted_recall_sum'] / overall_metrics['total_weight']
            overall_ndcg = overall_metrics['weighted_ndcg_sum'] / overall_metrics['total_weight']
        else:
            overall_precision = 0.0
            overall_recall = 0.0
            overall_ndcg = 0.0
        
        logger.info(
            f"📊 OVERALL SYSTEM METRICS (weighted across all bins):\n"
            f"  Precision: {overall_precision:.3f}\n"
            f"  Recall: {overall_recall:.3f}\n"
            f"  NDCG: {overall_ndcg:.3f}\n"
            f"  Total Sessions: {overall_metrics['total_sessions']}\n"
            f"  Total Interactions: {overall_metrics['total_weight']}"
        )
        
        # 4. Check for Drift (OPTION 3: System-wide check)
        trigger_retrain = False
        
        if overall_precision < RETRAIN_THRESHOLD_PRECISION:
            trigger_retrain = True
            logger.warning(
                f"⚠️ SYSTEM-WIDE PRECISION DRIFT DETECTED!\n"
                f"  Overall Precision: {overall_precision:.3f} < Threshold: {RETRAIN_THRESHOLD_PRECISION}\n"
                f"  Triggering weight update for ALL users..."
            )
        else:
            logger.info(
                f"✅ System precision healthy: {overall_precision:.3f} >= {RETRAIN_THRESHOLD_PRECISION}"
            )
        
        # 5. Save Per-Bin Metrics (for detailed monitoring in Grafana)
        for domain, bins in metrics_bucket.items():
            for bin_name, session_data in bins.items():
                
                # Weighted average calculation per bin
                total_weight = sum(s['weight'] for s in session_data)
                
                if total_weight == 0:
                    continue

                avg_precision = sum(s['precision'] * s['weight'] for s in session_data) / total_weight
                avg_recall = sum(s['recall'] * s['weight'] for s in session_data) / total_weight
                avg_ndcg = sum(s['ndcg'] * s['weight'] for s in session_data) / total_weight
                
                sample_size = len(session_data)
                
                # Total saved/recommended for this bin
                total_saved = sum(s['saved_count'] for s in session_data)
                total_recommended = sum(s['recommended_count'] for s in session_data)
                
                # Mark if this bin contributed to drift
                # (not used for triggering, just for analysis)
                metric_bad = avg_precision < RETRAIN_THRESHOLD_PRECISION
                
                # Insert Snapshot into DB
                await db.execute("""
                    INSERT INTO online_evaluation_metrics 
                    (timestamp, domain, user_bin, precision_proxy, recall_proxy, ndcg_proxy, sample_size, triggered_retrain)
                    VALUES (NOW(), $1, $2, $3, $4, $5, $6, $7)
                """, domain, bin_name, avg_precision, avg_recall, avg_ndcg, sample_size, metric_bad)
                
                logger.info(
                    f"  └─ {domain}/{bin_name}: "
                    f"Precision={avg_precision:.3f}, Recall={avg_recall:.3f}, NDCG={avg_ndcg:.3f} "
                    f"(Sessions: {sample_size}, Saved: {total_saved}/{total_recommended})"
                )
        
        # 6. Save Overall System Metrics (special row for system-wide tracking)
        await db.execute("""
            INSERT INTO online_evaluation_metrics 
            (timestamp, domain, user_bin, precision_proxy, recall_proxy, ndcg_proxy, sample_size, triggered_retrain)
            VALUES (NOW(), $1, $2, $3, $4, $5, $6, $7)
        """, 'system_wide', 'all_activity', overall_precision, overall_recall, overall_ndcg, 
            overall_metrics['total_sessions'], trigger_retrain)
        
        logger.info(f"💾 Saved system-wide metrics to database")

        # 7. Conditional Retraining (OPTION 3: Based on overall precision)
        if trigger_retrain:
            logger.warning(
                f"📉 Model Drift Detected (Overall Precision < {RETRAIN_THRESHOLD_PRECISION}). "
                f"Triggering Weight Update..."
            )
            await update_user_weights(domain=None)
            logger.info("✅ Weight update completed")
        else:
            logger.info("✅ Metrics are healthy. No retraining needed.")

    except Exception as e:
        logger.error(f"Monitoring failed: {e}", exc_info=True)
    finally:
        await db.disconnect()

if __name__ == "__main__":
    asyncio.run(calculate_online_metrics())