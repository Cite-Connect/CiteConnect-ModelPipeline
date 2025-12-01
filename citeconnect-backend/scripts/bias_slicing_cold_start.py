#!/usr/bin/env python3
"""
Cold-start bias slicing + automatic mitigation config generation.

What this script does:

1) JOIN cold_start_evaluations with user_profiles_extended
2) Slice by:
   - primary_domain (healthcare / fintech / quantum_computing)
   - research_stage (undergraduate / masters / phd / industry / etc.)
   - reading_level (introductory / intermediate / advanced / expert)
3) For each slice, compute:
   - mean_combined_score
   - mean_profile_alignment
   - mean_ground_truth_quality
   - count of users in that slice
4) Detect bias if the gap between best and worst slice
   for a metric is > 0.15 (15 percentage points)
5) Save a JSON report for your writeup:
   - bias_report_cold_start_before.json
6) AUTOMATICALLY generate bias mitigation config:
   - bias_config/bias_mitigation_config.json

The mitigation config is consumed by RecommendationService, which:
  - boosts final_score for underperforming slices
  - applies a minimum score floor
"""

import asyncio
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.db.connection import DatabaseConnection
from app.utils.logger import get_logger

logger = get_logger(__name__)

# How big a gap we treat as "biased"
BIAS_DISPARITY_THRESHOLD = 0.15  # 15 percentage points
MIN_USERS_PER_SLICE = 2          # ignore slices with < 2 users

# Base directory (backend root: /app or citeconnect-backend/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Paths for outputs
BIAS_REPORT_PATH = BASE_DIR / "bias_report_cold_start_before.json"
BIAS_CONFIG_DIR = BASE_DIR / "bias_config"
BIAS_CONFIG_PATH = BIAS_CONFIG_DIR / "bias_mitigation_config.json"


async def load_joined_data(db: DatabaseConnection):
    """
    Pull the data we need from Postgres:

    - cold_start_evaluations: how good were the cold-start recs
    - user_profiles_extended: who the user is (domain, stage, level)

    Returns: list of dict rows.
    """
    query = """
    SELECT
        c.user_id,
        c.embedding_model,
        c.profile_alignment,
        c.ground_truth_quality,
        c.combined_score,
        c.recommendation_count,
        c.evaluation_timestamp,

        p.primary_domain,
        p.research_stage,
        p.reading_level,
        p.years_experience
    FROM cold_start_evaluations c
    JOIN user_profiles_extended p
      ON c.user_id = p.user_id
    ORDER BY c.user_id;
    """
    rows = await db.fetch(query)
    return [dict(r) for r in rows]


def aggregate_by_slice(rows):
    """
    Group rows by each slice dimension and compute metrics.

    Slices:
      - primary_domain
      - research_stage
      - reading_level

    Returns:
      slice_metrics = {
        "primary_domain": {
           "healthcare": {metrics...},
           "fintech": {metrics...},
           ...
        },
        "research_stage": {...},
        "reading_level": {...},
      }
    """
    slice_fields = ["primary_domain", "research_stage", "reading_level"]

    # raw_data[field][slice_value] = list of row dicts
    raw_data = {
        field: defaultdict(list)
        for field in slice_fields
    }

    for row in rows:
        for field in slice_fields:
            value = row.get(field) or "unknown"
            raw_data[field][value].append(row)

    slice_metrics = {}

    for field in slice_fields:
        field_metrics = {}
        for value, group in raw_data[field].items():
            if len(group) == 0:
                continue

            combined_scores = [
                g["combined_score"]
                for g in group
                if g["combined_score"] is not None
            ]
            profile_alignments = [
                g["profile_alignment"]
                for g in group
                if g["profile_alignment"] is not None
            ]
            ground_truth_qualities = [
                g["ground_truth_quality"]
                for g in group
                if g["ground_truth_quality"] is not None
            ]

            if not combined_scores:
                continue

            field_metrics[value] = {
                "user_count": len(group),
                "mean_combined_score": float(np.mean(combined_scores)),
                "std_combined_score": float(np.std(combined_scores)),
                "mean_profile_alignment": float(np.mean(profile_alignments)) if profile_alignments else None,
                "mean_ground_truth_quality": float(np.mean(ground_truth_qualities)) if ground_truth_qualities else None,
            }

        slice_metrics[field] = field_metrics

    return slice_metrics


def detect_bias(slice_metrics):
    """
    Look for big gaps (disparities) between slices.

    For each field (e.g. primary_domain), for each metric (e.g. mean_combined_score),
    we find best and worst slice and compute disparity.

    If disparity > BIAS_DISPARITY_THRESHOLD → we flag it as a bias finding.
    """
    bias_findings = []

    metrics_to_check = [
        "mean_combined_score",
        "mean_profile_alignment",
        "mean_ground_truth_quality",
    ]

    for field, slices in slice_metrics.items():
        # Only use slices that have enough users
        valid_slices = {
            name: m
            for name, m in slices.items()
            if m.get("user_count", 0) >= MIN_USERS_PER_SLICE
        }
        if len(valid_slices) < 2:
            continue

        for metric_name in metrics_to_check:
            # collect (slice_name, metric_value)
            values = []
            for slice_name, m in valid_slices.items():
                value = m.get(metric_name)
                if value is not None:
                    values.append((slice_name, value))

            if len(values) < 2:
                continue

            # find best & worst
            best_slice, best_val = max(values, key=lambda x: x[1])
            worst_slice, worst_val = min(values, key=lambda x: x[1])

            if best_val == 0:
                continue

            disparity = (best_val - worst_val) / best_val

            if disparity > BIAS_DISPARITY_THRESHOLD:
                bias_findings.append({
                    "field": field,
                    "metric": metric_name,
                    "best_slice": best_slice,
                    "best_value": best_val,
                    "worst_slice": worst_slice,
                    "worst_value": worst_val,
                    "disparity": disparity,
                })

    return bias_findings


def build_report(rows, slice_metrics, bias_findings):
    """Package everything into a JSON-serializable dict."""
    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "total_users_in_eval": len({r["user_id"] for r in rows}),
        "total_eval_rows": len(rows),
        "slices_analyzed": list(slice_metrics.keys()),
        "slice_metrics": slice_metrics,
        "bias_findings": bias_findings,
    }
    return report


def save_report(report, path: Path = BIAS_REPORT_PATH):
    path.write_text(json.dumps(report, indent=2))
    print(f"\n✅ Bias report saved to {path.resolve()}")


def build_mitigation_config_from_bias(bias_findings):
    """
    Build a bias_mitigation_config.json structure from bias findings.

    We ONLY use mean_combined_score for mitigation:

    For each field where mean_combined_score is biased:
      - Take the worst_slice as underperforming.
      - Compute:
          boost_factor      = 1 + min(disparity, 0.25)         (up to +25%)
          min_score_floor   = best_value * (1 - disparity*0.5) (never > best)
    """
    by_field = {}

    for b in bias_findings:
        if b["metric"] != "mean_combined_score":
            continue

        field = b["field"]
        worst_slice = b["worst_slice"]
        disparity = b["disparity"]  # 0–1
        best_val = b["best_value"]

        # heuristic boosts
        boost_factor = 1.0 + min(disparity, 0.25)   # cap at +25%
        min_score_floor = max(best_val * (1.0 - disparity * 0.5), 0.0)

        if field not in by_field:
            by_field[field] = {
                "underperforming_slices": [worst_slice],
                "boost_factor": round(boost_factor, 3),
                "min_score_floor": round(min_score_floor, 3),
            }
        else:
            # merge if multiple findings per field
            field_cfg = by_field[field]
            if worst_slice not in field_cfg["underperforming_slices"]:
                field_cfg["underperforming_slices"].append(worst_slice)
            # keep the max boost, min floor (most protective)
            field_cfg["boost_factor"] = round(
                max(field_cfg["boost_factor"], boost_factor), 3
            )
            field_cfg["min_score_floor"] = round(
                min(field_cfg["min_score_floor"], min_score_floor), 3
            )

    if not by_field:
        return {}

    # Shape expected by RecommendationService:
    # {
    #   "cold_start": {
    #     "minilm": {
    #       "<field>": {
    #          "underperforming_slices": [...],
    #          "boost_factor": ...,
    #          "min_score_floor": ...
    #       }
    #     }
    #   }
    # }
    cfg = {
        "cold_start": {
            "minilm": by_field
        }
    }
    return cfg


def save_mitigation_config(config: dict):
    """
    Save mitigation config to bias_config/bias_mitigation_config.json
    (backend root / app root).
    """
    if not config:
        print("\nℹ️ No mitigation config generated (no biased slices for mean_combined_score).")
        return

    BIAS_CONFIG_DIR.mkdir(exist_ok=True)
    BIAS_CONFIG_PATH.write_text(json.dumps(config, indent=2))
    print(f"✅ Bias mitigation config saved to {BIAS_CONFIG_PATH.resolve()}")
    print("\nGenerated mitigation config:\n")
    print(json.dumps(config, indent=2))


async def main():
    print("\n" + "=" * 80)
    print("  Cold-Start Bias Slicing (BEFORE MITIGATION)")
    print("=" * 80)

    db = DatabaseConnection()
    await db.connect()

    try:
        rows = await load_joined_data(db)
        print(f"Loaded {len(rows)} evaluation rows")

        if not rows:
            print("⚠ No data in cold_start_evaluations JOIN user_profiles_extended.")
            return

        # 1) Aggregate by slices
        slice_metrics = aggregate_by_slice(rows)

        # 2) Detect bias
        bias_findings = detect_bias(slice_metrics)

        # 3) Build + save bias report
        report = build_report(rows, slice_metrics, bias_findings)
        save_report(report)

        # 4) Build + save mitigation config automatically
        mitigation_cfg = build_mitigation_config_from_bias(bias_findings)
        save_mitigation_config(mitigation_cfg)

        # 5) Human-readable console summary
        print("\n--- Slice Summary (mean_combined_score) ---")
        for field, slices in slice_metrics.items():
            print(f"\n[{field}]")
            for name, m in slices.items():
                print(
                    f"  {name:20} | users={m['user_count']:2} | "
                    f"combined={m['mean_combined_score']:.3f} | "
                    f"profile_align={m['mean_profile_alignment']}"
                )

        print("\n--- Bias Findings ---")
        if not bias_findings:
            print("No disparities above threshold.")
        else:
            for b in bias_findings:
                print(
                    f"Field={b['field']}, metric={b['metric']}, "
                    f"best={b['best_slice']}({b['best_value']:.3f}), "
                    f"worst={b['worst_slice']}({b['worst_value']:.3f}), "
                    f"disparity={b['disparity']:.2%}"
                )

    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())