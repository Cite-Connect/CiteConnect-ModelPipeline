# app/services/fairness_service.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# --------------------------------------------------------
# Paths (inside container, /app is the backend root)
# --------------------------------------------------------

# __file__ = /app/app/services/fairness_service.py
# parents[0] = /app/app/services
# parents[1] = /app/app
# parents[2] = /app
BACKEND_ROOT = Path(__file__).resolve().parents[2]

FAIRNESS_CONFIG_PATH = BACKEND_ROOT / "fairness_config.json"
METADATA_PARQUET_PATH = BACKEND_ROOT / "data" / "combined_gcs_data.parquet"

# --------------------------------------------------------
# Caches so we don't reload on every request
# --------------------------------------------------------
_fairness_config_cache: Optional[Dict[str, Any]] = None
_fairness_config_mtime: Optional[float] = None

_paper_field_map: Optional[Dict[str, str]] = None


def _extract_primary_field(fields: Any) -> str:
    """
    Take the 'fieldsOfStudy' value and turn it into a single primary field.
    """
    if isinstance(fields, list) and fields:
        return str(fields[0])
    if isinstance(fields, str) and fields:
        return fields
    return "Unknown"


def _load_paper_field_map() -> Dict[str, str]:
    """
    Lazy load a mapping: paper_id -> primary_field.
    Loaded once from combined_gcs_data.parquet and cached.
    """
    global _paper_field_map
    if _paper_field_map is not None:
        return _paper_field_map

    if not METADATA_PARQUET_PATH.exists():
        # Fallback: no metadata, just return empty mapping
        _paper_field_map = {}
        return _paper_field_map

    df = pd.read_parquet(METADATA_PARQUET_PATH)

    # Align column names to how we use them elsewhere
    if "paperId" in df.columns and "paper_id" not in df.columns:
        df = df.rename(columns={"paperId": "paper_id"})

    if "fieldsOfStudy" not in df.columns:
        df["fieldsOfStudy"] = [[]]

    df["primary_field"] = df["fieldsOfStudy"].apply(_extract_primary_field)

    _paper_field_map = dict(zip(df["paper_id"].astype(str), df["primary_field"]))
    return _paper_field_map


def load_fairness_config() -> Dict[str, Any]:
    """
    Load fairness_config.json with a small cache so we don't re-read
    on every request.
    """
    global _fairness_config_cache, _fairness_config_mtime

    if not FAIRNESS_CONFIG_PATH.exists():
        # No config yet => no mitigation
        return {"under_served_fields": []}

    mtime = FAIRNESS_CONFIG_PATH.stat().st_mtime

    if _fairness_config_cache is not None and _fairness_config_mtime == mtime:
        return _fairness_config_cache

    with FAIRNESS_CONFIG_PATH.open() as f:
        _fairness_config_cache = json.load(f)
        _fairness_config_mtime = mtime

    return _fairness_config_cache


def fairness_aware_rerank(
    recommendations: List[Dict[str, Any]],
    boost: float = 1.05,
) -> List[Dict[str, Any]]:
    """
    Take a list of recommendations and lightly boost scores for papers
    belonging to under-served fields, as defined in fairness_config.json.

    Expected rec format (adapt if yours differs):
        {
          "paper_id": "...",    # or "paperId"
          "score": 0.92,
          ...
        }
    """
    if not recommendations:
        return recommendations

    cfg = load_fairness_config()
    under_served = set(cfg.get("under_served_fields", []))

    # If no under-served fields, just return as-is
    if not under_served:
        return recommendations

    field_map = _load_paper_field_map()

    boosted: List[Dict[str, Any]] = []
    for rec in recommendations:
        # Try both keys in case your code uses paperId instead of paper_id
        pid = rec.get("paper_id") or rec.get("paperId")
        if pid is None:
            boosted.append(rec)
            continue

        pid = str(pid)
        field = field_map.get(pid, "Unknown")

        score = rec.get("score")
        if score is None:
            boosted.append(rec)
            continue

        if field in under_served:
            score = score * boost

        new_rec = {**rec, "score": score, "primary_field": field}
        boosted.append(new_rec)

    boosted.sort(key=lambda r: r.get("score", 0.0), reverse=True)
    return boosted