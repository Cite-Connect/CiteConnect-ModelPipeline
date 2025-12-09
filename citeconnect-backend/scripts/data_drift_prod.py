#!/usr/bin/env python3
"""
Production data drift detection for CiteConnect.

This script:
  1) Connects to Postgres via DatabaseConnection
  2) Pulls paper-level features from the `papers` table:
       - domain (categorical)
       - year (numeric)
       - citation_count (numeric)
       - updated_at (timestamp)  → aliased as created_at
  3) Splits into:
       - baseline_df: oldest 60% of rows (training-era)
       - current_df : newest 20% of rows (latest production data)
  4) Computes per-feature drift scores:
       - domain: L1 distribution shift in [0, 1]
       - year, citation_count: PSI-based score in [0, 1]
  5) Writes:
       bias_config/data_drift_report.json

This JSON is consumed by main.py/update_monitoring_metrics()
to set the Prometheus gauge:
    DATA_DRIFT_OVERALL_SCORE
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import sys

import numpy as np
import pandas as pd

# Make backend importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.connection import DatabaseConnection  # type: ignore
from app.utils.logger import get_logger          # type: ignore

logger = get_logger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parent.parent
BIAS_CONFIG_DIR = BACKEND_ROOT / "bias_config"
DRIFT_REPORT_PATH = BIAS_CONFIG_DIR / "data_drift_report.json"

# Which features to track
CATEGORICAL_FEATURES: List[str] = ["domain"]
NUMERIC_FEATURES: List[str] = ["citation_count", "year"]

# How to split baseline vs current in time (percentiles of updated_at/created_at)
BASELINE_FRACTION = 0.60  # oldest 60% of data
CURRENT_FRACTION = 0.20   # newest 20% of data


# --------------------------------------------------------------------
# DB loading
# --------------------------------------------------------------------


async def load_paper_features(db: DatabaseConnection) -> pd.DataFrame:
    """
    Load paper-level features from Postgres.

    Assumes `papers` table has at least:
      - domain (text)
      - year (int)
      - citation_count (int)
      - updated_at (timestamp)  # aliased as created_at
    """
    query = """
    SELECT
        domain,
        year,
        citation_count,
        updated_at AS created_at
    FROM papers
    WHERE updated_at IS NOT NULL;
    """

    rows = await db.fetch(query)
    records = [dict(r) for r in rows]

    if not records:
        logger.warning("No rows returned from papers table for drift analysis")
        return pd.DataFrame(columns=["domain", "year", "citation_count", "created_at"])

    df = pd.DataFrame.from_records(records)
    logger.info(
        "Loaded paper features for drift",
        rows=len(df),
        columns=list(df.columns),
    )
    return df


def split_baseline_current(df: pd.DataFrame, ts_col: str = "created_at") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into baseline vs current using the timestamp column.

    - Sort by created_at ascending
    - Baseline = oldest BASELINE_FRACTION of rows
    - Current  = newest CURRENT_FRACTION of rows
    - Middle portion is ignored so windows don't overlap
    """
    if df.empty:
        return df, df

    if ts_col not in df.columns:
        raise ValueError(f"Missing timestamp column '{ts_col}' in dataframe")

    df_sorted = df.sort_values(ts_col)
    n = len(df_sorted)

    if n < 20:
        # Not enough data for a meaningful split
        logger.warning("Not enough rows for drift split; returning empty current_df")
        baseline_df = df_sorted.copy()
        current_df = df_sorted.iloc[0:0].copy()
        return baseline_df, current_df

    baseline_end = int(BASELINE_FRACTION * n)
    current_start = int((1.0 - CURRENT_FRACTION) * n)

    baseline_df = df_sorted.iloc[:baseline_end].reset_index(drop=True)
    current_df = df_sorted.iloc[current_start:].reset_index(drop=True)

    logger.info(
        "Split baseline/current",
        total_rows=n,
        baseline_rows=len(baseline_df),
        current_rows=len(current_df),
    )
    return baseline_df, current_df


# --------------------------------------------------------------------
# Drift metrics
# --------------------------------------------------------------------


def _categorical_drift_score(baseline: pd.Series, current: pd.Series) -> float:
    """
    Simple categorical drift metric:
      score = 0.5 * sum |p_i - q_i|
    where p, q are normalized frequency distributions.

    Range: [0, 1]
    """
    categories = sorted(set(baseline.dropna().unique()) | set(current.dropna().unique()))
    if not categories:
        return 0.0

    p = baseline.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    q = current.value_counts(normalize=True).reindex(categories, fill_value=0.0)

    score = 0.5 * float(np.abs(p - q).sum())
    return float(max(0.0, min(score, 1.0)))


def _population_stability_index(baseline: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """
    Compute PSI (Population Stability Index) for a numeric feature.

    Steps:
      - Define bins based on baseline quantiles
      - Compute histograms for baseline and current
      - PSI = sum((p_i - q_i) * ln(p_i / q_i))

    We then map PSI into [0, 1] for reporting:
      score = min(PSI / 1.0, 1.0)
    """
    base = baseline.dropna().astype(float)
    curr = current.dropna().astype(float)

    if len(base) == 0 or len(curr) == 0:
        return 0.0

    # Use baseline quantiles for bin edges
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(base, quantiles))
    if len(edges) < 2:
        return 0.0

    base_hist, _ = np.histogram(base, bins=edges)
    curr_hist, _ = np.histogram(curr, bins=edges)

    base_pct = base_hist / (len(base) + 1e-9)
    curr_pct = curr_hist / (len(curr) + 1e-9)

    # Avoid zero values
    base_pct = np.clip(base_pct, 1e-6, 1.0)
    curr_pct = np.clip(curr_pct, 1e-6, 1.0)

    psi_values = (base_pct - curr_pct) * np.log(base_pct / curr_pct)
    psi = float(np.sum(psi_values))

    # Normalize PSI to [0, 1]
    score = min(psi / 1.0, 1.0)  # PSI >= 1 considered max drift
    return float(max(0.0, score))


def compute_drift_report(
    baseline_df: pd.DataFrame, current_df: pd.DataFrame
) -> Dict[str, Any]:
    feature_reports: Dict[str, Any] = {}
    scores: List[float] = []

    if current_df.empty:
        # No current data → drift = 0, but flag in metadata
        logger.warning("Current dataframe is empty; overall drift set to 0")
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "overall_drift_score": 0.0,
            "features": {},
            "metadata": {
                "note": "current_df empty; check data pipeline",
            },
        }

    # Categorical features
    for col in CATEGORICAL_FEATURES:
        if col not in baseline_df.columns or col not in current_df.columns:
            logger.warning("Categorical feature missing", feature=col)
            continue

        score = _categorical_drift_score(baseline_df[col], current_df[col])
        feature_reports[col] = {
            "type": "categorical",
            "metric": "l1_distribution_shift",
            "score": score,
        }
        scores.append(score)

    # Numeric features
    for col in NUMERIC_FEATURES:
        if col not in baseline_df.columns or col not in current_df.columns:
            logger.warning("Numeric feature missing", feature=col)
            continue

        score = _population_stability_index(baseline_df[col], current_df[col])
        feature_reports[col] = {
            "type": "numeric",
            "metric": "psi_normalized",
            "score": score,
        }
        scores.append(score)

    overall = float(np.mean(scores)) if scores else 0.0

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "overall_drift_score": overall,
        "features": feature_reports,
        "metadata": {
            "baseline_rows": int(len(baseline_df)),
            "current_rows": int(len(current_df)),
            "baseline_fraction": BASELINE_FRACTION,
            "current_fraction": CURRENT_FRACTION,
        },
    }
    return report


def save_drift_report(report: Dict[str, Any]) -> None:
    BIAS_CONFIG_DIR.mkdir(exist_ok=True)
    DRIFT_REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"✅ Data drift report saved to {DRIFT_REPORT_PATH.resolve()}")
    print(f"Overall drift score: {report['overall_drift_score']:.3f}")


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------


async def main_async() -> None:
    print("\n" + "=" * 80)
    print("  Production Data Drift Analysis (Postgres → JSON)")
    print("=" * 80)

    db = DatabaseConnection()
    await db.connect()

    try:
        df = await load_paper_features(db)
    finally:
        await db.disconnect()

    baseline_df, current_df = split_baseline_current(df, ts_col="created_at")
    report = compute_drift_report(baseline_df, current_df)
    save_drift_report(report)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
