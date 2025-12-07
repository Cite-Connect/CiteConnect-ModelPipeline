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

Performance modes:
- ULTRA_FAST: 3 users, 9 configs      ≈ quickest (for debugging / fast sweeps)
- FAST:        10 users, 144 configs   ≈ balanced speed/accuracy
- FULL:        30 users, 288 configs   ≈ most accurate, most expensive

Optimizations in ultra_fast / fast:
- Reduced search space focussed around default weights
- Reduced user sample
- Reduced candidate limits: 30 / 5 / 5 (vs 150 / 25 / 25)

Usage (from backend root: citeconnect-backend/):

    docker-compose exec api python scripts/hyperparameter_tuning_cold_start.py

To change performance mode, edit PERFORMANCE_MODE below
or wire it to an environment variable if you prefer.
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
DEFAULT_WEIGHTS: Dict[str, float] = {
    "semantic": 0.40,
    "citation": 0.20,
    "recency": 0.15,
    "ground_truth": 0.10,
    "reading_level": 0.10,
    "diversity": 0.05,
}

# Performance modes:
# - "ultra_fast": minimal search space, tiny user sample
# - "fast":        reduced search space, medium user sample
# - "full":        full search space, large user sample
PERFORMANCE_MODE = "ultra_fast"  # "ultra_fast", "fast", or "full"

if PERFORMANCE_MODE == "ultra_fast":
    # Ultra-fast:
    # - Only tune semantic & citation weights around defaults
    # - Keep other weights fixed
    # - Very small grid: 3 x 3 = 9 configs
    SEARCH_SPACE: Dict[str, List[float]] = {
        "semantic": [0.35, 0.40, 0.45],  # around default 0.40
        "citation": [0.15, 0.20, 0.25],  # around default 0.20
        "recency": [0.15],
        "ground_truth": [0.10],
        "reading_level": [0.10],
        "diversity": [0.05],
    }
    FAST_MODE = True
elif PERFORMANCE_MODE == "fast":
    # Fast:
    # - Wider search space but still smaller than full grid
    SEARCH_SPACE = {
        "semantic": [0.35, 0.40, 0.45],      # 3
        "citation": [0.15, 0.20, 0.25],      # 3
        "recency": [0.10, 0.15],             # 2
        "ground_truth": [0.05, 0.10],        # 2
        "reading_level": [0.05, 0.10],       # 2
        "diversity": [0.05, 0.10],           # 2
    }
    FAST_MODE = True
else:  # "full"
    # Full:
    # - Complete grid as originally designed
    SEARCH_SPACE = {
        "semantic": [0.30, 0.35, 0.40, 0.45],   # 4
        "citation": [0.10, 0.15, 0.20, 0.25],   # 4
        "recency": [0.10, 0.15, 0.20],          # 3
        "ground_truth": [0.05, 0.10, 0.15],     # 3
        "reading_level": [0.05, 0.10, 0.15],    # 3
        "diversity": [0.05, 0.10],              # 2
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
    seen: set[Tuple[Tuple[str, float], ...]] = set()

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
    Evaluate a single weight configuration on a batch of cold-start users.

    Uses EvaluationService.batch_evaluate_cold_start. When fast_mode is True,
    we shrink the RecommendationService candidate limits at the *class level*
    so that all instances created inside the evaluation use the reduced limits.
    """
    logger.info(
        "Evaluating config",
        user_count=len(user_ids),
        model=model,
        weights=weights,
    )

    from app.services.recommendation_service import RecommendationService

    # Class-level overrides so every RecommendationService instance sees them
    if fast_mode:
        original_semantic = RecommendationService.SEMANTIC_CANDIDATE_LIMIT
        original_canonical = RecommendationService.CANONICAL_CANDIDATE_COUNT
        original_gt = RecommendationService.GT_NETWORK_CANDIDATE_COUNT

        # Ultra-reduced limits for fastest evaluation
        RecommendationService.SEMANTIC_CANDIDATE_LIMIT = 30
        RecommendationService.CANONICAL_CANDIDATE_COUNT = 5
        RecommendationService.GT_NETWORK_CANDIDATE_COUNT = 5
    else:
        original_semantic = original_canonical = original_gt = None

    try:
        result = await eval_service.batch_evaluate_cold_start(
            user_ids=user_ids,
            model=model,
            scoring_weights=weights,
        )
    finally:
        # Restore original limits so the rest of the system is unaffected
        if fast_mode:
            RecommendationService.SEMANTIC_CANDIDATE_LIMIT = original_semantic
            RecommendationService.CANONICAL_CANDIDATE_COUNT = original_canonical
            RecommendationService.GT_NETWORK_CANDIDATE_COUNT = original_gt

    aggregate = result.get("aggregate_metrics", {}) or {}

    # Ensure numerics are plain float for logging / JSON
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


def save_report(report: Dict[str, Any], path: str | None = None):
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
        print("  ⚡⚡ ULTRA-FAST MODE - tiny search space, small user sample")
    elif PERFORMANCE_MODE == "fast":
        print("  ⚡ FAST MODE - reduced search space, medium user sample")
    else:
        print("  📊 FULL MODE - full search space, larger user sample")
    print("=" * 80 + "\n")

    # Use minimal DB pool for scripts (Supabase has connection limits)
    original_pool_size = settings.DB_POOL_SIZE
    original_max_overflow = settings.DB_MAX_OVERFLOW

    settings.DB_POOL_SIZE = 2
    settings.DB_MAX_OVERFLOW = 1

    db = DatabaseConnection()
    await db.connect()

    # Restore pool settings *after* the connection is created
    settings.DB_POOL_SIZE = original_pool_size
    settings.DB_MAX_OVERFLOW = original_max_overflow

    try:
        # 1. Sample cold-start users based on performance mode
        if PERFORMANCE_MODE == "ultra_fast":
            USER_SAMPLE_SIZE = 3  # minimum but still gives some variance
        elif PERFORMANCE_MODE == "fast":
            USER_SAMPLE_SIZE = 10
        else:  # "full"
            USER_SAMPLE_SIZE = 30

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
        total_evals = num_configs * len(user_ids)
        print(f"Approx total user-evaluations: {total_evals}")
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

        # 4. Grid search over configs
        for idx, weights in enumerate(configs, start=1):
            cfg_name = f"cfg_{idx:03d}"
            print(f"[{idx}/{num_configs}] Evaluating {cfg_name} -> {weights}")

            with mlflow.start_run(run_name=f"cold_start_hp_{cfg_name}"):
                # Log params
                mlflow.log_params({f"weight_{k}": float(v) for k, v in weights.items()})
                mlflow.log_param("stage", "cold_start")
                mlflow.log_param("model", "minilm")
                mlflow.log_param("user_sample_size", len(user_ids))
                mlflow.log_param("performance_mode", PERFORMANCE_MODE)

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
                f"    avg_profile_alignment = "
                f"{metrics.get('avg_profile_alignment', 0):.3f}, "
                f"avg_ground_truth_quality = "
                f"{metrics.get('avg_ground_truth_quality', 0):.3f}, "
                f"avg_combined_score = {avg_combined:.3f}"
            )

            # Track all results
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

            # Track best config by avg_combined_score
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
        report: Dict[str, Any] = {
            "generated_at": datetime.utcnow().isoformat(),
            "mode": "cold_start",
            "model": "minilm",
            "performance_mode": PERFORMANCE_MODE,
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
