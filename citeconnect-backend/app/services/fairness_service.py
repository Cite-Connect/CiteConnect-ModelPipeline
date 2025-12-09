# app/services/fairness_service.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# --------------------------------------------------------
# Paths (inside container, /app is the backend root)
# --------------------------------------------------------

# __file__ = /app/app/services/fairness_service.py
# parents[0] = /app/app/services
# parents[1] = /app/app
# parents[2] = /app
BACKEND_ROOT = Path(__file__).resolve().parents[2]

FAIRNESS_CONFIG_PATH = BACKEND_ROOT / "fairness_config.json"

# --------------------------------------------------------
# Caches so we don't reload on every request
# --------------------------------------------------------
_fairness_config_cache: Optional[Dict[str, Any]] = None
_fairness_config_mtime: Optional[float] = None

_paper_domain_map: Optional[Dict[str, str]] = None


def load_fairness_config() -> Dict[str, Any]:
    """
    Load fairness_config.json with a small cache so we don't re-read
    on every request.
    
    Supports multiple config formats:
    - New: paper_domain_fairness.under_served_domains (from domain_representation_fairness.py)
    - Old: under_served_fields (backward compatibility)
    """
    global _fairness_config_cache, _fairness_config_mtime

    if not FAIRNESS_CONFIG_PATH.exists():
        # No config yet => no mitigation
        return {"paper_domain_fairness": {"under_served_domains": []}}

    mtime = FAIRNESS_CONFIG_PATH.stat().st_mtime

    if _fairness_config_cache is not None and _fairness_config_mtime == mtime:
        return _fairness_config_cache

    with FAIRNESS_CONFIG_PATH.open() as f:
        _fairness_config_cache = json.load(f)
        _fairness_config_mtime = mtime

    return _fairness_config_cache


def get_under_served_domains(config: Dict[str, Any]) -> List[str]:
    """
    Extract under-served domains from config.
    
    Args:
        config: Loaded fairness config
        
    Returns:
        List of under-served domain names
    """
    # New format: paper_domain_fairness.under_served_domains
    if "paper_domain_fairness" in config:
        return config["paper_domain_fairness"].get("under_served_domains", [])
    
    # Old format: under_served_fields (backward compatibility)
    if "under_served_fields" in config:
        return config["under_served_fields"]
    
    return []


def get_domain_boost_factor(config: Dict[str, Any], domain: str) -> float:
    """
    Get boost factor for a specific domain from config.
    
    Args:
        config: Loaded fairness config
        domain: Domain name (e.g., "fintech", "healthcare")
        
    Returns:
        Boost factor (default 1.0 if domain not under-served)
    """
    # New format: paper_domain_fairness.domains[domain].boost_factor
    if "paper_domain_fairness" in config:
        domains = config["paper_domain_fairness"].get("domains", {})
        if domain in domains:
            return float(domains[domain].get("boost_factor", 1.0))
    
    # Fallback: check if domain is under-served and use default boost
    under_served = get_under_served_domains(config)
    if domain in under_served:
        # Use boost_factor from config or default 1.05
        fairness = config.get("paper_domain_fairness", {})
        return float(fairness.get("boost_factor", 1.05))
    
    return 1.0


async def fairness_aware_rerank(
    recommendations: List[Dict[str, Any]],
    db: Optional[Any] = None,
    boost: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Take a list of recommendations and boost scores for papers
    belonging to under-served domains, as defined in fairness_config.json.
    
    Uses PostgreSQL to get paper domains (no parquet file needed).

    Expected rec format:
        {
          "paper_id": "...",    # or "paperId"
          "score": 0.92,
          ...
        }
        
    Args:
        recommendations: List of recommendation dicts
        db: DatabaseConnection (optional, for loading domain map from PostgreSQL)
        boost: Override boost factor (optional, uses config if not provided)
        
    Returns:
        Reranked recommendations with boosted scores
    """
    if not recommendations:
        return recommendations

    cfg = load_fairness_config()
    under_served = set(get_under_served_domains(cfg))

    # If no under-served domains, just return as-is
    if not under_served:
        return recommendations

    # Load domain mapping from PostgreSQL if db provided
    domain_map: Dict[str, str] = {}
    if db:
        try:
            query = "SELECT paper_id, domain FROM papers WHERE domain IS NOT NULL"
            rows = await db.fetch(query)
            domain_map = {str(row['paper_id']): row['domain'] for row in rows}
        except Exception:
            # Fallback: use domain from paper dict if available
            pass
    
    # Fallback: try to get domain from paper dict if already present
    if not domain_map:
        for rec in recommendations:
            pid = rec.get("paper_id") or rec.get("paperId")
            if pid and "domain" in rec:
                domain_map[str(pid)] = rec["domain"]

    # Get default boost from config if not provided
    if boost is None:
        fairness = cfg.get("paper_domain_fairness", {})
        boost = float(fairness.get("boost_factor", 1.05))

    boosted: List[Dict[str, Any]] = []
    for rec in recommendations:
        # Try both keys in case your code uses paperId instead of paper_id
        pid = rec.get("paper_id") or rec.get("paperId")
        if pid is None:
            boosted.append(rec)
            continue

        pid = str(pid)
        # Get domain from map or from paper dict
        domain = domain_map.get(pid) or rec.get("domain", "Unknown")

        score = rec.get("score") or rec.get("final_score") or rec.get("relevance_score")
        if score is None:
            boosted.append(rec)
            continue

        # Apply domain-specific boost
        domain_boost = get_domain_boost_factor(cfg, domain)
        if domain_boost > 1.0:
            score = score * domain_boost

        new_rec = {**rec, "score": score, "domain": domain}
        boosted.append(new_rec)

    boosted.sort(key=lambda r: r.get("score", 0.0), reverse=True)
    return boosted


# Backward compatibility: synchronous version
def fairness_aware_rerank_sync(
    recommendations: List[Dict[str, Any]],
    boost: float = 1.05,
) -> List[Dict[str, Any]]:
    """
    Synchronous version for backward compatibility.
    Uses domain from paper dict if available, otherwise no boost.
    """
    if not recommendations:
        return recommendations

    cfg = load_fairness_config()
    under_served = set(get_under_served_domains(cfg))

    if not under_served:
        return recommendations

    boosted: List[Dict[str, Any]] = []
    for rec in recommendations:
        domain = rec.get("domain", "Unknown")
        score = rec.get("score") or rec.get("final_score") or rec.get("relevance_score")
        
        if score is None:
            boosted.append(rec)
            continue

        if domain in under_served:
            domain_boost = get_domain_boost_factor(cfg, domain)
            score = score * domain_boost

        new_rec = {**rec, "score": score, "domain": domain}
        boosted.append(new_rec)

    boosted.sort(key=lambda r: r.get("score", 0.0), reverse=True)
    return boosted