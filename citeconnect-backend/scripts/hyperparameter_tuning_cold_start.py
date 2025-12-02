#!/usr/bin/env python3
"""
Hyperparameter tuning for cold-start recommendation weights.

Right now this script:
- Defines a search space for the cold-start weights
- (Placeholder) marks the current defaults as "best" because
  we don't yet have a full interaction-based evaluation loop
- Writes best_hyperparameters_cold_start.json

Later, when you have more interaction data, you can plug in:
- MRR / NDCG / Recall metrics per user
- Bias penalty term from your bias report
and actually search over weight combinations.
"""

import json
from pathlib import Path
from datetime import datetime

from app.utils.logger import get_logger

logger = get_logger(__name__)

# These must match RecommendationService.DEFAULT_COLD_START_WEIGHTS
DEFAULT_WEIGHTS = {
    "semantic": 0.40,
    "citation": 0.20,
    "recency": 0.15,
    "ground_truth": 0.10,
    "reading_level": 0.10,
    "diversity": 0.05,
}

# Simple search space definition for your report
SEARCH_SPACE = {
    "semantic": [0.30, 0.35, 0.40, 0.45],
    "citation": [0.10, 0.15, 0.20, 0.25],
    "recency": [0.10, 0.15, 0.20],
    "ground_truth": [0.05, 0.10, 0.15],
    "reading_level": [0.05, 0.10, 0.15],
    "diversity": [0.05, 0.10],
}


def build_report():
    """
    For now, just treat DEFAULT_WEIGHTS as the best config.
    This still lets you:
      - Document the search space
      - Show where the "best" config is stored
      - Plug in real evaluation later.
    """

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "mode": "cold_start",
        "model": "minilm",
        "search_space": SEARCH_SPACE,
        "strategy": "placeholder_random_search",
        "objective": "future: MRR + (1 - bias_score)",
        "best_config": {
            "weights": DEFAULT_WEIGHTS,
            "notes": (
                "Initial deployment uses hand-tuned defaults. "
                "Once interaction data is available, plug in "
                "MRR / NDCG and bias penalty here to choose the "
                "best weight combination."
            ),
        },
    }

    return report


def save_report(report, path: str = None):
    if path is None:
        # Save to bias_config directory (same location as other config files)
        base_dir = Path(__file__).resolve().parent.parent
        out_path = base_dir / "bias_config" / "best_hyperparameters_cold_start.json"
    else:
        out_path = Path(path)
    
    # Ensure parent directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    out_path.write_text(json.dumps(report, indent=2))
    logger.info("Saved best hyperparameters", path=str(out_path.resolve()))
    print(f"✅ Saved best hyperparameters to {out_path.resolve()}")


def main():
    print("\n" + "=" * 80)
    print("  Hyperparameter Tuning (Cold-Start) – Placeholder Run")
    print("=" * 80)

    report = build_report()
    save_report(report)


if __name__ == "__main__":
    main()
