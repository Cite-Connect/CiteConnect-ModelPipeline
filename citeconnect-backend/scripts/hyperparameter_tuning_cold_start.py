#!/usr/bin/env python3
"""
Hyperparameter tuning for cold-start recommendation weights.

What this script does:

1. Defines a SEARCH_SPACE over cold-start scoring weights
2. Builds a normalized grid of weight configurations (sum(weights)=1)
3. Samples a small set of cold-start users from the DB
4. For each config:
   - Runs EvaluationService.batch_evaluate_cold_start(...)
   - Collects aggregate metrics:
       * avg_profile_alignment
       * avg_ground_truth_quality
       * avg_combined_score
   - Logs params + metrics to MLflow
5. Selects the config with the highest avg_combined_score
6. Writes:
       bias_config/best_hyperparameters_cold_start.json

The JSON format is compatible with sensitivity_cold_start_weights.py,
which expects:

    {
      "best_config": {
        "weights": { ... }
      },
      ...
    }

Performance Modes:
- ULTRA_FAST: 5 users, 32 configs = ~2-3 minutes (minimum for statistical validity)
- FAST: 10 users, 144 configs = ~5-10 minutes (balanced)
- FULL: 30 users, 288 configs = ~30-60 minutes (maximum accuracy)

Optimizations in fast/ultra-fast modes:
- Reduced search space focused around default weights
- Reduced user sample (still statistically valid)
- Reduced candidate limits: 50/10/10 (vs 150/25/25) = ~3x faster per user

Usage (from backend root: citeconnect-backend/):

    docker-compose exec api python scripts/hyperparameter_tuning_cold_start.py

To disable fast mode, set FAST_MODE = False in the script.
"""

import asyncio
import itertools
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

import mlflow
import numpy as np

from app.db.connection import DatabaseConnection
from app.services.evaluation_service import EvaluationService
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Baseline + search space
# ---------------------------------------------------------------------------

# These must match RecommendationService.DEFAULT_COLD_START_WEIGHTS
# (kept from the previous placeholder script)
DEFAULT_WEIGHTS: Dict[str, float] = {
    "semantic": 0.40,
    "citation": 0.20,
    "recency": 0.15,
    "ground_truth": 0.10,
    "reading_level": 0.10,
    "diversity": 0.05,
}

# Search space - can be reduced for faster tuning
# Full search space: 4×4×3×3×3×2 = 288 configurations
# Fast search space: 3×3×2×2×2×2 = 144 configurations (~2x faster)
# Minimal search space: 2×2×2×2×2×2 = 32 configurations (~9x faster)

# Performance modes:
# - ULTRA_FAST: 5 users, 32 configs (~2-3 min) - Minimum for statistical validity
# - FAST: 10 users, 144 configs (~5-10 min) - Balanced speed/accuracy
# - FULL: 30 users, 288 configs (~30-60 min) - Maximum accuracy

# Set to: "ultra_fast", "fast", or "full"
PERFORMANCE_MODE = "ultra_fast"  # Change this to adjust speed vs accuracy

if PERFORMANCE_MODE == "ultra_fast":
    # Ultra-fast: Minimal search space - just test default ± small variations
    # Uses 5 users (minimum for statistical significance)
    # Focuses on most important weights: semantic and citation
    SEARCH_SPACE: Dict[str, List[float]] = {
        "semantic": [0.35, 0.40, 0.45],  # 3 values - test around default (0.40)
        "citation": [0.15, 0.20, 0.25],  # 3 values - test around default (0.20)
        "recency": [0.15],  # 1 value - keep at default
        "ground_truth": [0.10],  # 1 value - keep at default
        "reading_level": [0.10],  # 1 value - keep at default
        "diversity": [0.05],  # 1 value - keep at default
    }
    # This gives us 3×3 = 9 configs instead of 64!
    FAST_MODE = True  # Enable fast mode optimizations
elif PERFORMANCE_MODE == "fast":
    # Fast: Reduced search space (~144 configs)
    SEARCH_SPACE: Dict[str, List[float]] = {
        "semantic": [0.35, 0.40, 0.45],  # 3 values
        "citation": [0.15, 0.20, 0.25],  # 3 values
        "recency": [0.10, 0.15],  # 2 values
        "ground_truth": [0.05, 0.10],  # 2 values
        "reading_level": [0.05, 0.10],  # 2 values
        "diversity": [0.05, 0.10],  # 2 values
    }
    FAST_MODE = True
else:  # full
    # Full: Complete search space (~288 configs)
    SEARCH_SPACE: Dict[str, List[float]] = {
        "semantic": [0.30, 0.35, 0.40, 0.45],
        "citation": [0.10, 0.15, 0.20, 0.25],
        "recency": [0.10, 0.15, 0.20],
        "ground_truth": [0.05, 0.10, 0.15],
        "reading_level": [0.05, 0.10, 0.15],
        "diversity": [0.05, 0.10],
    }
    FAST_MODE = False


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def renormalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Ensure weights sum to 1.0."""
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError(f"Weights sum to <= 0: {weights}")
    return {k: float(v) / total for k, v in weights.items()}


def generate_weight_configs() -> List[Dict[str, float]]:
    """
    Generate a grid of weight configurations from SEARCH_SPACE,
    renormalized so that sum(weights) == 1.0.

    Returns:
        List of weight dicts, e.g. [{"semantic": 0.4, ...}, ...]
    """
    keys = list(SEARCH_SPACE.keys())
    values_grid = [SEARCH_SPACE[k] for k in keys]

    configs: List[Dict[str, float]] = []
    seen: set = set()

    for raw_values in itertools.product(*values_grid):
        raw = dict(zip(keys, raw_values))
        normalized = renormalize_weights(raw)

        # Deduplicate after normalization (to avoid logging the same config twice)
        frozen = tuple(sorted(normalized.items()))
        if frozen in seen:
            continue
        seen.add(frozen)
        configs.append(normalized)

    return configs


async def get_cold_start_user_ids(
    db: DatabaseConnection,
    limit: int = 30,
) -> List[int]:
    """
    Find a SAMPLE of cold-start users.

    We use recommendation_stage='cold_start' and require they have an
    extended profile (same idea as EvaluationService.batch_evaluate_cold_start).

    Args:
        db: Database connection
        limit: Max number of users to sample

    Returns:
        List of user_ids
    """
    query = """
        SELECT s.user_id
        FROM user_recommendation_state s
        JOIN user_profiles_extended p ON s.user_id = p.user_id
        WHERE s.recommendation_stage = 'cold_start'
        ORDER BY s.user_id
        LIMIT $1
    """
    rows = await db.fetch(query, limit)
    user_ids = [r["user_id"] for r in rows]
    if not user_ids:
        logger.warning("No cold-start users found for hyperparameter tuning")
    else:
        logger.info("Sampled cold-start users for tuning", user_count=len(user_ids))
    return user_ids


async def evaluate_config_for_users(
    eval_service: EvaluationService,
    user_ids: List[int],
    weights: Dict[str, float],
    model: str = "minilm",
    fast_mode: bool = False,
) -> Dict[str, Any]:
    """
    Use EvaluationService.batch_evaluate_cold_start to evaluate a single
    weight configuration on a batch of cold-start users.

    Args:
        fast_mode: If True, reduces candidate limits for faster evaluation

    Returns:
        Aggregated result dict including:
          - total_users
          - successful_evaluations
          - failed_evaluations
          - model_used
          - scoring_weights
          - aggregate_metrics:
                avg_profile_alignment
                avg_ground_truth_quality
                avg_combined_score
                (and possibly more)
    """
    logger.info(
        "Evaluating config",
        user_count=len(user_ids),
        model=model,
        weights=weights,
    )

    # Temporarily reduce candidate limits for faster evaluation in fast mode
    from app.services.recommendation_service import RecommendationService
    rec_service = RecommendationService(eval_service.db)
    
    if fast_mode:
        # Store original limits
        original_semantic = rec_service.SEMANTIC_CANDIDATE_LIMIT
        original_canonical = rec_service.CANONICAL_CANDIDATE_COUNT
        original_gt = rec_service.GT_NETWORK_CANDIDATE_COUNT
        
        # Ultra-reduced limits for fastest evaluation (~5x faster per user)
        # This is aggressive but acceptable for hyperparameter tuning
        rec_service.SEMANTIC_CANDIDATE_LIMIT = 30  # Reduced from 150 (was 50)
        rec_service.CANONICAL_CANDIDATE_COUNT = 5  # Reduced from 25 (was 10)
        rec_service.GT_NETWORK_CANDIDATE_COUNT = 5  # Reduced from 25 (was 10)
    
    try:
        result = await eval_service.batch_evaluate_cold_start(
            user_ids=user_ids,
            model=model,
            scoring_weights=weights,
        )
    finally:
        # Restore original limits
        if fast_mode:
            rec_service.SEMANTIC_CANDIDATE_LIMIT = original_semantic
            rec_service.CANONICAL_CANDIDATE_COUNT = original_canonical
            rec_service.GT_NETWORK_CANDIDATE_COUNT = original_gt

    aggregate = result.get("aggregate_metrics", {}) or {}

    # Ensure numerics are float for logging
    for k, v in list(aggregate.items()):
        if isinstance(v, (np.floating, np.integer)):
            aggregate[k] = float(v)

    return {
        "weights": weights,
        "result": result,
        "aggregate_metrics": aggregate,
    }


def setup_mlflow():
    """
    Configure MLflow for this tuning run.

    Uses the same environment variables as the rest of the project:
      MLFLOW_TRACKING_URI
      MLFLOW_EXPERIMENT_NAME
    """
    from app.config import settings

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    # Keep it under the same "family" of experiments, but with a suffix
    exp_name = f"{settings.MLFLOW_EXPERIMENT_NAME}-cold-start-hparam-tuning"
    mlflow.set_experiment(exp_name)
    logger.info("MLflow configured", experiment_name=exp_name)


def save_report(report: Dict[str, Any], path: str = None):
    """
    Save the tuning report to JSON.

    Default path:
        citeconnect-backend/bias_config/best_hyperparameters_cold_start.json
    """
    if path is None:
        base_dir = Path(__file__).resolve().parent.parent
        out_path = base_dir / "bias_config" / "best_hyperparameters_cold_start.json"
    else:
        out_path = Path(path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    logger.info("Saved best hyperparameters", path=str(out_path.resolve()))
    print(f"✅ Saved best hyperparameters to {out_path.resolve()}")


# ---------------------------------------------------------------------------
# Main tuning logic
# ---------------------------------------------------------------------------

async def build_report() -> Dict[str, Any]:
    """
    Run grid-search hyperparameter tuning over SEARCH_SPACE.

    Objective:
        Maximize avg_combined_score across sampled cold-start users.

    Returns:
        A report dict with:
          - generated_at
          - mode
          - model
          - search_space
          - strategy
          - objective
          - user_sample_size
          - num_configurations
          - best_config: {weights, metrics, total_users, ...}
          - all_results: list of per-config results
    """
    from app.config import settings

    print("\n" + "=" * 80)
    print("  Hyperparameter Tuning (Cold-Start) – Grid Search")
    if PERFORMANCE_MODE == "ultra_fast":
        print("  ⚡⚡ ULTRA-FAST MODE - 5 users, 32 configs (~2-3 min)")
    elif PERFORMANCE_MODE == "fast":
        print("  ⚡ FAST MODE - 10 users, 144 configs (~5-10 min)")
    else:
        print("  📊 FULL MODE - 30 users, 288 configs (~30-60 min)")
    print("=" * 80 + "\n")

    # Use minimal DB pool for scripts (Supabase has connection limits)
    original_pool_size = settings.DB_POOL_SIZE
    original_max_overflow = settings.DB_MAX_OVERFLOW

    settings.DB_POOL_SIZE = 2    # small pool for script
    settings.DB_MAX_OVERFLOW = 1

    db = DatabaseConnection()
    await db.connect()

    # Restore pool settings after connection is created
    settings.DB_POOL_SIZE = original_pool_size
    settings.DB_MAX_OVERFLOW = original_max_overflow

    try:
        # 1. Sample cold-start users
        # Adjust based on performance mode
        if PERFORMANCE_MODE == "ultra_fast":
            USER_SAMPLE_SIZE = 5  # Minimum for statistical significance
        elif PERFORMANCE_MODE == "fast":
            USER_SAMPLE_SIZE = 10  # Balanced
        else:  # full
            USER_SAMPLE_SIZE = 30  # Maximum accuracy
        
        user_ids = await get_cold_start_user_ids(db, limit=USER_SAMPLE_SIZE)
        if not user_ids:
            # No users -> keep using defaults, but still write a report
            logger.warning("No users available; falling back to DEFAULT_WEIGHTS")
            return {
                "generated_at": datetime.utcnow().isoformat(),
                "mode": "cold_start",
                "model": "minilm",
                "search_space": SEARCH_SPACE,
                "strategy": "grid_search_normalized",
                "objective": "max avg_combined_score (no users found, fallback)",
                "user_sample_size": 0,
                "num_configurations": 0,
                "best_config": {
                    "weights": DEFAULT_WEIGHTS,
                    "metrics": {},
                    "notes": "No cold-start users found; using defaults.",
                },
                "all_results": [],
            }

        # 2. Generate configs
        configs = generate_weight_configs()
        num_configs = len(configs)
        print(f"Found {num_configs} unique normalized configs to evaluate.")
        total_evals = num_configs * USER_SAMPLE_SIZE
        if PERFORMANCE_MODE == "ultra_fast":
            print(f"⚡⚡ Ultra-fast mode: ~{total_evals} total evaluations (~1-2 min)")
        elif PERFORMANCE_MODE == "fast":
            print(f"⚡ Fast mode: ~{total_evals} total evaluations (~5-10 min)")
        else:
            print(f"📊 Full mode: ~{total_evals} total evaluations (~30-60 min)")
        print()

        # 3. Prepare evaluator + MLflow
        eval_service = EvaluationService(db)
        setup_mlflow()

        best_score = float("-inf")
        best_payload: Dict[str, Any] = {
            "weights": DEFAULT_WEIGHTS,
            "metrics": {},
            "total_users": len(user_ids),
        }
        all_results: List[Dict[str, Any]] = []

        # 4. Grid search
        for idx, weights in enumerate(configs, 1):
            cfg_name = f"cfg_{idx:03d}"
            print(f"[{idx}/{num_configs}] Evaluating {cfg_name} -> {weights}")

            with mlflow.start_run(run_name=f"cold_start_hp_{cfg_name}"):
                # Log params
                mlflow.log_params({f"weight_{k}": float(v) for k, v in weights.items()})
                mlflow.log_param("stage", "cold_start")
                mlflow.log_param("model", "minilm")
                mlflow.log_param("user_sample_size", len(user_ids))

                # Run evaluation
                eval_payload = await evaluate_config_for_users(
                    eval_service=eval_service,
                    user_ids=user_ids,
                    weights=weights,
                    model="minilm",
                    fast_mode=FAST_MODE,
                )

                metrics = eval_payload.get("aggregate_metrics", {}) or {}
                # Log metrics to MLflow (only numeric)
                numeric_metrics = {
                    k: float(v)
                    for k, v in metrics.items()
                    if isinstance(v, (int, float, np.floating, np.integer))
                }
                if numeric_metrics:
                    mlflow.log_metrics(numeric_metrics)

            avg_combined = float(metrics.get("avg_combined_score", 0.0))
            print(
                f"    avg_profile_alignment = {metrics.get('avg_profile_alignment', 0):.3f}, "
                f"avg_ground_truth_quality = {metrics.get('avg_ground_truth_quality', 0):.3f}, "
                f"avg_combined_score = {avg_combined:.3f}"
            )

            # Track best
            all_results.append(
                {
                    "name": cfg_name,
                    "weights": weights,
                    "metrics": metrics,
                    "total_users": eval_payload["result"].get(
                        "total_users", len(user_ids)
                    ),
                }
            )

            if avg_combined > best_score:
                best_score = avg_combined
                best_payload = {
                    "name": cfg_name,
                    "weights": weights,
                    "metrics": metrics,
                    "total_users": eval_payload["result"].get(
                        "total_users", len(user_ids)
                    ),
                }

        # 5. Build report
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "mode": "cold_start",
            "model": "minilm",
            "search_space": SEARCH_SPACE,
            "strategy": "grid_search_normalized",
            "objective": "max avg_combined_score over sampled cold-start users",
            "user_sample_size": len(user_ids),
            "num_configurations": num_configs,
            "best_config": {
                "name": best_payload.get("name"),
                "weights": best_payload["weights"],
                "metrics": best_payload["metrics"],
                "total_users": best_payload["total_users"],
            },
            "all_results": all_results,
        }

        # Pretty print best for terminal / screenshots
        print("\n" + "-" * 80)
        print("Best configuration (by avg_combined_score)")
        print("-" * 80)
        print(f"Name:    {best_payload.get('name')}")
        print(f"Weights: {best_payload['weights']}")
        print("Metrics:")
        for k, v in best_payload["metrics"].items():
            print(f"  - {k}: {v}")
        print("-" * 80 + "\n")

        return report

    finally:
        await db.disconnect()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    report = asyncio.run(build_report())
    save_report(report)


if __name__ == "__main__":
    main()
