#!/usr/bin/env python3
"""
Sensitivity analysis for cold-start recommendation weights.

What this script does (for your report):

1. Load the "baseline" cold-start weights
   - Prefer best_hyperparameters_cold_start.json if present
   - Otherwise fall back to RecommendationService.DEFAULT_COLD_START_WEIGHTS

2. Build a set of weight scenarios:
   - baseline
   - For each component (semantic, citation, recency, ground_truth,
     reading_level, diversity), create:
       * <name>_plus_20  (weight * 1.2, renormalize to sum=1)
       * <name>_minus_20 (weight * 0.8, renormalize to sum=1)

3. For each scenario:
   - Find cold-start users (interaction_count < 10, LIMITED to a small sample)
   - For each such user, generate cold-start recommendations with those weights
   - Collect final_score from each recommended paper

4. Compute metrics:
   - mean_final_score       (global mean over all recommended papers)
   - std_final_score        (standard deviation)
   - mean_user_score        (mean of per-user averages)
   - user_count             (# users evaluated)
   - total_recommendations  (# of recs evaluated)

5. Save results to:
      bias_config/sensitivity_cold_start_weights.json

6. Print a small summary table to terminal so you can screenshot / copy
   into your report.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np

from app.db.connection import DatabaseConnection
from app.services.recommendation_service import RecommendationService
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers to load baseline weights and generate variants
# ---------------------------------------------------------------------------

def load_baseline_weights() -> Dict[str, float]:
    """
    Try to load best cold-start weights from:
        bias_config/best_hyperparameters_cold_start.json

    If not present or malformed, fall back to
        RecommendationService.DEFAULT_COLD_START_WEIGHTS
    """
    # Start with class-level defaults
    baseline = RecommendationService.DEFAULT_COLD_START_WEIGHTS.copy()

    try:
        base_dir = Path(__file__).parent.parent  # citeconnect-backend/
        cfg_path = base_dir / "bias_config" / "best_hyperparameters_cold_start.json"

        if not cfg_path.exists():
            logger.info(
                "No best_hyperparameters_cold_start.json found – using default weights",
                path=str(cfg_path),
            )
            return baseline

        data = json.loads(cfg_path.read_text())

        # Try multiple possible structures:
        # 1. New format: {"best_config": {"weights": {...}}}
        # 2. Old format: {"cold_start": {"minilm": {...}}}
        # 3. Direct format: weights at top level
        weights = None
        
        if "best_config" in data and "weights" in data["best_config"]:
            weights = data["best_config"]["weights"]
        elif "cold_start" in data and "minilm" in data["cold_start"]:
            weights = data["cold_start"]["minilm"]
        elif all(k in data for k in ["semantic", "citation", "recency"]):
            # Direct format - weights are at top level
            weights = {k: v for k, v in data.items() if k in baseline.keys()}

        if not weights:
            logger.warning(
                "best_hyperparameters_cold_start.json did not have expected keys – "
                "falling back to defaults",
                path=str(cfg_path),
            )
            return baseline

        # Make sure it sums to 1.0
        total = sum(weights.values())
        if total <= 0:
            logger.warning(
                "Loaded weights sum to <= 0 – falling back to defaults",
                weights=weights,
            )
            return baseline

        normalized = {k: float(v) / total for k, v in weights.items()}
        logger.info(
            "Loaded baseline weights from best_hyperparameters_cold_start.json",
            path=str(cfg_path),
            weights=normalized,
        )
        return normalized

    except Exception as e:
        logger.warning(f"Failed to load baseline weights, using defaults: {e}")
        return baseline


def renormalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Ensure weights sum to 1.0."""
    total = sum(weights.values())
    if total <= 0:
        raise ValueError(f"Weights sum to <= 0: {weights}")
    return {k: float(v) / total for k, v in weights.items()}


def generate_weight_scenarios(
    baseline: Dict[str, float],
    delta: float = 0.20,
    quick_mode: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Build weight scenarios:

    - "baseline": baseline weights
    - For each component k:
        * f"{k}_plus_20"
        * f"{k}_minus_20"

    Where plus/minus multiplies that component by (1 ± delta)
    and then we renormalize all weights.
    
    If quick_mode=True, only test baseline + semantic + citation variations (5 scenarios).
    """
    scenarios: Dict[str, Dict[str, float]] = {}

    scenarios["baseline"] = baseline.copy()

    # In quick mode, only test the most important weight components
    keys_to_test = ["semantic", "citation"] if quick_mode else baseline.keys()

    for key in keys_to_test:
        # +20% scenario
        plus = baseline.copy()
        plus[key] = plus[key] * (1.0 + delta)
        plus = renormalize_weights(plus)
        scenarios[f"{key}_plus_20"] = plus

        # -20% scenario
        minus = baseline.copy()
        minus[key] = minus[key] * (1.0 - delta)
        minus = renormalize_weights(minus)
        scenarios[f"{key}_minus_20"] = minus

    return scenarios


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

async def get_cold_start_user_ids(
    db: DatabaseConnection,
    limit: int = 1,
) -> List[int]:
    """
    Find a SAMPLE of cold-start users (interaction_count < 10).

    We limit to 1 user (default) so the sensitivity script
    runs quickly but still shows how metrics change.
    """
    query = """
        SELECT user_id
        FROM user_recommendation_state
        WHERE interaction_count < 10
        ORDER BY user_id
        LIMIT $1
    """
    rows = await db.fetch(query, limit)
    return [r["user_id"] for r in rows]


async def evaluate_weights_for_users(
    db: DatabaseConnection,
    weights: Dict[str, float],
    model: str = "minilm",
    top_k: int = 10,
) -> Dict[str, float]:
    """
    For a given set of weights, generate cold-start recs for a small sample
    of cold-start users and compute summary metrics.

    Returns a dict with:
      - mean_final_score
      - std_final_score
      - mean_user_score
      - user_count
      - total_recommendations
    """
    service = RecommendationService(db)

    # Reduce candidate limits for faster sensitivity analysis
    # This reduces the number of papers loaded from ~200 to ~20 per user
    original_semantic = service.SEMANTIC_CANDIDATE_LIMIT
    original_canonical = service.CANONICAL_CANDIDATE_COUNT
    original_gt = service.GT_NETWORK_CANDIDATE_COUNT
    
    # Ultra-minimal limits for fast sensitivity analysis
    service.SEMANTIC_CANDIDATE_LIMIT = 5  # Minimal - just 5 papers from semantic search
    service.CANONICAL_CANDIDATE_COUNT = 3  # Just 3 canonical papers
    service.GT_NETWORK_CANDIDATE_COUNT = 2  # Just 2 GT papers

    # *** only 1 user for faster sensitivity analysis ***
    user_ids = await get_cold_start_user_ids(db, limit=1)
    if not user_ids:
        logger.warning("No cold-start users found for sensitivity analysis")
        return {
            "mean_final_score": 0.0,
            "std_final_score": 0.0,
            "mean_user_score": 0.0,
            "user_count": 0,
            "total_recommendations": 0,
        }

    all_scores: List[float] = []
    per_user_means: List[float] = []

    try:
        for uid in user_ids:
            try:
                print(f"    Loading ~10 candidate papers for user {uid}...", end="", flush=True)
                step_start = time.time()
                result = await service.generate_cold_start_recommendations(
                    user_id=uid,
                    count=top_k,
                    model=model,
                    scoring_weights=weights,
                )
                step_time = time.time() - step_start
                print(f" ✓ ({step_time:.1f}s)", flush=True)
            except Exception as e:
                print(" ✗", flush=True)
                logger.warning(
                    f"Failed to generate recs for user {uid} in sensitivity analysis: {e}"
                )
                continue

            papers = result.get("papers", [])
            if not papers:
                continue

            user_scores = [p.get("final_score", 0.0) for p in papers]
            all_scores.extend(user_scores)
            per_user_means.append(float(np.mean(user_scores)))
    finally:
        # Restore original limits
        service.SEMANTIC_CANDIDATE_LIMIT = original_semantic
        service.CANONICAL_CANDIDATE_COUNT = original_canonical
        service.GT_NETWORK_CANDIDATE_COUNT = original_gt

    if not all_scores:
        logger.warning("No scores collected in sensitivity analysis")
        return {
            "mean_final_score": 0.0,
            "std_final_score": 0.0,
            "mean_user_score": 0.0,
            "user_count": 0,
            "total_recommendations": 0,
        }

    all_scores_arr = np.array(all_scores, dtype=float)
    per_user_arr = np.array(per_user_means, dtype=float) if per_user_means else None

    metrics = {
        "mean_final_score": float(all_scores_arr.mean()),
        "std_final_score": float(all_scores_arr.std()),
        "mean_user_score": float(per_user_arr.mean()) if per_user_arr is not None else 0.0,
        "user_count": int(len(per_user_means)),
        "total_recommendations": int(len(all_scores)),
    }

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    print("\n" + "=" * 80)
    print("  Cold-Start Hyperparameter Sensitivity Analysis")
    print("=" * 80 + "\n")

    # Use minimal connection pool for scripts (Supabase has connection limits)
    from app.config import settings
    original_pool_size = settings.DB_POOL_SIZE
    original_max_overflow = settings.DB_MAX_OVERFLOW
    
    # Temporarily reduce pool size for script
    settings.DB_POOL_SIZE = 2  # Minimal pool for scripts
    settings.DB_MAX_OVERFLOW = 1  # Minimal overflow

    db = DatabaseConnection()
    await db.connect()
    
    # Restore original settings after connection
    settings.DB_POOL_SIZE = original_pool_size
    settings.DB_MAX_OVERFLOW = original_max_overflow

    try:
        baseline = load_baseline_weights()
        
        # Quick mode: only test baseline + semantic + citation (5 scenarios instead of 13)
        # Set to False to test all weight components
        QUICK_MODE = True  # Change to False for full analysis
        
        scenarios = generate_weight_scenarios(baseline, delta=0.20, quick_mode=QUICK_MODE)
        
        if QUICK_MODE:
            print("⚡ QUICK MODE: Only testing baseline + semantic + citation variations\n")

        print("Scenarios to evaluate:")
        for name, w in scenarios.items():
            print(f"  {name:15} -> {w}")
        print()

        results = {}
        total_scenarios = len(scenarios)
        
        print(f"📊 Processing {total_scenarios} scenarios × 1 user = {total_scenarios} user evaluations")
        print(f"   Estimated papers to load: ~{total_scenarios * 10} (ultra-minimal from ~{total_scenarios * 200})")
        print(f"   ⚠️  Each scenario reloads candidates - this takes time!\n")

        for scenario_num, (name, weights) in enumerate(scenarios.items(), 1):
            print(f"\n--- [{scenario_num}/{total_scenarios}] Evaluating scenario: {name} ---")
            print(f"    Weights: {weights}")
            start_time = time.time()
            metrics = await evaluate_weights_for_users(db, weights)
            elapsed = time.time() - start_time
            results[name] = {
                "weights": weights,
                "metrics": metrics,
            }

            print(
                f"  ✓ Completed ({elapsed:.1f}s): users={metrics['user_count']:2d}, "
                f"total_recs={metrics['total_recommendations']:3d}, "
                f"mean_final_score={metrics['mean_final_score']:.3f}, "
                f"std_final_score={metrics['std_final_score']:.3f}, "
                f"mean_user_score={metrics['mean_user_score']:.3f}"
            )

        # Save to JSON
        base_dir = Path(__file__).parent.parent
        out_dir = base_dir / "bias_config"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "sensitivity_cold_start_weights.json"
        out_path.write_text(json.dumps({
            "generated_at": datetime.utcnow().isoformat(),
            "baseline_weights": baseline,
            "scenarios": results,
        }, indent=2))

        print(f"\n✅ Sensitivity report saved to {out_path.resolve()}")

        # Quick comparison table by mean_final_score
        print("\n--- Scenario ranking by mean_final_score ---")
        sorted_items: List[Tuple[str, Dict]] = sorted(
            results.items(),
            key=lambda kv: kv[1]["metrics"]["mean_final_score"],
            reverse=True,
        )
        for name, data in sorted_items:
            m = data["metrics"]
            print(
                f"{name:15} | mean_final={m['mean_final_score']:.3f} "
                f"| std={m['std_final_score']:.3f} | users={m['user_count']:2d}"
            )

    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())