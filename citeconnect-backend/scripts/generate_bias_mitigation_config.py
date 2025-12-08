#!/usr/bin/env python3
"""
Generate bias_mitigation_config.json from bias_report_cold_start_before.json.

What it does:

1) Read bias_report_cold_start_before.json
2) For each bias finding on mean_combined_score:
   - Identify the worst-performing slice (e.g. healthcare, masters, advanced)
   - Compute a boost factor and a minimum score floor
3) Write bias_config/bias_mitigation_config.json in the format that
   RecommendationService expects:

{
  "cold_start": {
    "primary_domain": {
      "healthcare": {
        "score_boost_factor": 1.2,
        "min_score_floor": 0.24
      }
    },
    "research_stage": {
      "masters": { ... }
    },
    "reading_level": {
      "advanced": { ... }
    }
  }
}
"""

import json
from pathlib import Path

# Only treat disparities above this as "bias" (same idea as your slicing script)
BIAS_DISPARITY_THRESHOLD = 0.15  # 15%

# Maximum extra boost we allow (e.g. 0.3 → up to 1.3x)
MAX_EXTRA_BOOST = 0.30


def main():
    repo_root = Path(__file__).parent.parent  # citeconnect-backend
    report_path = repo_root / "bias_report_cold_start_before.json"

    if not report_path.exists():
        raise FileNotFoundError(
            f"Bias report not found at {report_path}. "
            f"Run bias_slicing_cold_start.py first."
        )

    # Load the bias report your slicing script generated
    data = json.loads(report_path.read_text())

    slice_metrics = data.get("slice_metrics", {})
    bias_findings = data.get("bias_findings", [])

    # This is the structure RecommendationService is expecting
    config = {
        "cold_start": {
            # "primary_domain": { "healthcare": { ... } },
            # "research_stage": { "masters": { ... } },
            # "reading_level": { "advanced": { ... } },
        }
    }

    # We only use findings for mean_combined_score (you could extend later)
    for finding in bias_findings:
        metric = finding.get("metric")
        if metric != "mean_combined_score":
            continue

        field = finding["field"]            # e.g. "primary_domain"
        worst_slice = finding["worst_slice"]  # e.g. "healthcare"
        best_val = finding["best_value"]
        worst_val = finding["worst_value"]
        disparity = finding["disparity"]    # 0.42 → 42% worse

        if disparity < BIAS_DISPARITY_THRESHOLD:
            continue  # don’t mitigate tiny differences

        # Compute a boost factor between 1.0 and 1.3 (max)
        extra_boost = min(disparity, MAX_EXTRA_BOOST)
        score_boost_factor = round(1.0 + extra_boost, 3)

        # Set a floor so worst group can approach the best group’s average
        # e.g. 90% of best_val
        min_score_floor = round(best_val * 0.90, 3)

        # Insert into config
        field_cfg = config["cold_start"].setdefault(field, {})
        field_cfg[worst_slice] = {
            "score_boost_factor": score_boost_factor,
            "min_score_floor": min_score_floor,
        }

    # Make sure output folder exists
    out_dir = repo_root / "bias_config"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "bias_mitigation_config.json"
    out_path.write_text(json.dumps(config, indent=2))

    print("\n✅ Generated bias mitigation config:")
    print(f"   Input report : {report_path.resolve()}")
    print(f"   Output config: {out_path.resolve()}\n")
    print("Config content:")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()