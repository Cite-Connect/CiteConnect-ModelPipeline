#!/usr/bin/env python3
"""
Cold-start bias slicing + automatic mitigation config generation.

What this script does:

1) JOIN cold_start_evaluations with user_profiles_extended
2) Slice by:
   - primary_domain (healthcare / fintech / quantum_computing / etc.)
   - research_stage (undergraduate / masters / phd / industry / etc.)
   - reading_level (introductory / intermediate / advanced / expert / etc.)
3) For each slice, compute:
   - mean_combined_score
   - mean_profile_alignment
   - mean_ground_truth_quality
   - user_count
4) Detect bias if the gap between best and worst slice
   for a metric is > 0.15 (15 percentage points)
5) Save a detailed JSON report for analysis/writeup:
   - bias_report_cold_start_before.json
6) Save a compact JSON summary for Prometheus/Grafana:
   - bias_reports.json
7) AUTOMATICALLY generate bias mitigation config:
   - bias_config/bias_mitigation_config.json

The mitigation config is consumed by RecommendationService, which:
  - boosts final_score for underperforming slices
  - applies a minimum score floor for those slices
"""

import asyncio
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import json
import sys
from typing import Dict, List, Any, Tuple, Optional

import numpy as np

# Ensure backend root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.connection import DatabaseConnection  # type: ignore
from app.utils.logger import get_logger           # type: ignore

logger = get_logger(__name__)

# -----------------------------------------------------------------------------
# Constants & paths
# -----------------------------------------------------------------------------

# How big a gap we treat as "biased"
BIAS_DISPARITY_THRESHOLD: float = 0.15  # 15 percentage points
MIN_USERS_PER_SLICE: int = 2           # ignore slices with < 2 users

# Base directory (backend root: /app or citeconnect-backend/)
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Paths for outputs
BIAS_REPORT_PATH: Path = BASE_DIR / "bias_report_cold_start_before.json"
BIAS_REPORTS_METRICS_PATH: Path = BASE_DIR / "bias_reports.json"
BIAS_CONFIG_DIR: Path = BASE_DIR / "bias_config"
BIAS_CONFIG_PATH: Path = BIAS_CONFIG_DIR / "bias_mitigation_config.json"


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

async def load_joined_data(db: DatabaseConnection) -> List[Dict[str, Any]]:
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


# -----------------------------------------------------------------------------
# Aggregation by slices
# -----------------------------------------------------------------------------

def aggregate_by_slice(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
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
    raw_data: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        field: defaultdict(list) for field in slice_fields
    }

    for row in rows:
        for field in slice_fields:
            value = row.get(field) or "unknown"
            raw_data[field][value].append(row)

    slice_metrics: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for field in slice_fields:
        field_metrics: Dict[str, Dict[str, Any]] = {}
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
                "mean_profile_alignment": float(np.mean(profile_alignments))
                if profile_alignments
                else None,
                "mean_ground_truth_quality": float(np.mean(ground_truth_qualities))
                if ground_truth_qualities
                else None,
            }

        slice_metrics[field] = field_metrics

    return slice_metrics


# -----------------------------------------------------------------------------
# Bias detection
# -----------------------------------------------------------------------------

def detect_bias(slice_metrics: Dict[str, Dict[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Look for big gaps (disparities) between slices.

    For each field (e.g. primary_domain), for each metric (e.g. mean_combined_score),
    we find best and worst slice and compute disparity:

        disparity = (best_val - worst_val) / best_val

    If disparity > BIAS_DISPARITY_THRESHOLD → we flag it as a bias finding.
    """
    bias_findings: List[Dict[str, Any]] = []

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
            values: List[Tuple[str, float]] = []
            for slice_name, m in valid_slices.items():
                value = m.get(metric_name)
                if value is not None:
                    values.append((slice_name, float(value)))

            if len(values) < 2:
                continue

            # find best & worst
            best_slice, best_val = max(values, key=lambda x: x[1])
            worst_slice, worst_val = min(values, key=lambda x: x[1])

            if best_val == 0:
                continue

            disparity = (best_val - worst_val) / best_val

            if disparity > BIAS_DISPARITY_THRESHOLD:
                bias_findings.append(
                    {
                        "field": field,
                        "metric": metric_name,
                        "best_slice": best_slice,
                        "best_value": best_val,
                        "worst_slice": worst_slice,
                        "worst_value": worst_val,
                        "disparity": disparity,
                    }
                )

    return bias_findings


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def build_report(
    rows: List[Dict[str, Any]],
    slice_metrics: Dict[str, Dict[str, Dict[str, Any]]],
    bias_findings: List[Dict[str, Any]],
) -> Dict[str, Any]:
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


def save_report(report: Dict[str, Any], path: Path = BIAS_REPORT_PATH) -> None:
    path.write_text(json.dumps(report, indent=2))
    print(f"\n✅ Bias report saved to {path.resolve()}")


# -----------------------------------------------------------------------------
# Compact fairness summary for /metrics
# -----------------------------------------------------------------------------

def build_overall_citation_fairness(
    slice_metrics: Dict[str, Dict[str, Dict[str, Any]]]
) -> Optional[Dict[str, Any]]:
    """
    Build an overall 'citation_fairness' style summary from slice_metrics,
    to feed Prometheus /metrics via bias_reports.json.

    We use mean_combined_score as the fairness metric and consider
    all slices across all fields that have >= MIN_USERS_PER_SLICE.
    """
    group_means: List[float] = []
    group_sizes: List[int] = []

    for field, slices in slice_metrics.items():
        for name, m in slices.items():
            if m.get("user_count", 0) < MIN_USERS_PER_SLICE:
                continue
            mean_val = m.get("mean_combined_score")
            if mean_val is None:
                continue
            group_means.append(float(mean_val))
            group_sizes.append(int(m["user_count"]))

    if not group_means:
        return None

    max_mean = max(group_means)
    min_mean = min(group_means)
    max_group_size = max(group_sizes)
    min_group_size = min(group_sizes)
    num_groups = len(group_means)

    if min_mean <= 0:
        disparity_ratio = None
    else:
        disparity_ratio = max_mean / min_mean

    disparity_difference = max_mean - min_mean

    return {
        "max_group_mean": float(max_mean),
        "min_group_mean": float(min_mean),
        "disparity_ratio": float(disparity_ratio) if disparity_ratio is not None else None,
        "disparity_difference": float(disparity_difference),
        "min_group_size": int(min_group_size),
        "max_group_size": int(max_group_size),
        "num_groups": int(num_groups),
    }


def save_metrics_bias_report(slice_metrics: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    """
    Save a compact bias_reports.json for Prometheus /metrics.

    main.py expects:
      bias_reports.json -> { "citation_fairness": { "disparity_ratio", "disparity_difference", ... } }
    """
    cf = build_overall_citation_fairness(slice_metrics)
    if cf is None:
        print("\nℹ️ No valid slices to build citation_fairness summary for metrics.")
        return

    data = {
        "citation_fairness": cf,
        "metadata": {
            "source": "bias_slicing_cold_start.py",
            "generated_at": datetime.utcnow().isoformat(),
            "metric": "mean_combined_score",
        },
    }

    BIAS_REPORTS_METRICS_PATH.write_text(json.dumps(data, indent=2))
    print(f"✅ Metrics-compatible bias report saved to {BIAS_REPORTS_METRICS_PATH.resolve()}")


# -----------------------------------------------------------------------------
# Mitigation config
# -----------------------------------------------------------------------------

def build_mitigation_config_from_bias(
    bias_findings: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Build a bias_mitigation_config.json structure from bias findings.

    We ONLY use mean_combined_score for mitigation:

    For each field where mean_combined_score is biased:
      - Take the worst_slice as underperforming.
      - Compute:
          boost_factor      = 1 + min(disparity, 0.25)         (up to +25%)
          min_score_floor   = best_value * (1 - disparity*0.5) (never > best)
    """
    by_field: Dict[str, Dict[str, Any]] = {}

    for b in bias_findings:
        if b["metric"] != "mean_combined_score":
            continue

        field = b["field"]
        worst_slice = b["worst_slice"]
        disparity = float(b["disparity"])  # 0–1
        best_val = float(b["best_value"])

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
    cfg: Dict[str, Any] = {
        "cold_start": {
            "minilm": by_field
        }
    }
    return cfg


def save_mitigation_config(config: Dict[str, Any]) -> None:
    """
    Save mitigation config to bias_config/bias_mitigation_config.json
    (backend root / app root).
    """
    if not config:
        print("\nℹ️ No mitigation config generated (no biased slices for mean_combined_score).")
        return

    BIAS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    BIAS_CONFIG_PATH.write_text(json.dumps(config, indent=2))
    print(f"✅ Bias mitigation config saved to {BIAS_CONFIG_PATH.resolve()}")
    print("\nGenerated mitigation config:\n")
    print(json.dumps(config, indent=2))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

async def main() -> None:
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

        # 3) Build + save detailed bias report (for writeup)
        report = build_report(rows, slice_metrics, bias_findings)
        save_report(report)

        # 4) Build + save metrics-compatible bias summary (for Prometheus/Grafana)
        save_metrics_bias_report(slice_metrics)

        # 5) Build + save mitigation config automatically
        mitigation_cfg = build_mitigation_config_from_bias(bias_findings)
        save_mitigation_config(mitigation_cfg)

        # 6) Human-readable console summary
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
