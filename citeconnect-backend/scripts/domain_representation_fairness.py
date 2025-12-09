#!/usr/bin/env python3
"""
Domain-level *representation* fairness using only Postgres.

We measure how many papers exist per domain and boost
domains that are under-represented in the corpus.

Example domains:
  - healthcare
  - fintech
  - quantum_computing

Logic:
  - Count papers per domain (papers.domain)
  - Let max_count = largest domain count
  - A domain is "under-served" if count < 50% of max_count
  - Under-served domains get a boost_factor (e.g. 1.05x)

Writes fairness_config.json:

{
  "paper_domain_fairness": {
    "metric": "representation_count",
    "disparity_threshold_ratio": 0.5,
    "under_served_domains": [...],
    "domains": {
      "fintech": {
        "num_papers": 123,
        "under_served": true,
        "boost_factor": 1.05
      },
      "healthcare": {
        "num_papers": 500,
        "under_served": false,
        "boost_factor": 1.0
      }
    }
  },
  "metadata": {
    "source": "domain_representation_fairness.py",
    "generated_at": "...",
    "max_domain_count": 500
  }
}
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.connection import DatabaseConnection  # type: ignore
from app.utils.logger import get_logger           # type: ignore

logger = get_logger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parent.parent
FAIRNESS_CONFIG_PATH = BACKEND_ROOT / "fairness_config.json"

# Domains with count < 50% of the biggest domain are under-represented
UNDERREP_THRESHOLD_RATIO = 0.5

# Boost applied to under-served domains at runtime
DEFAULT_DOMAIN_BOOST = 1.05


async def load_domain_counts(db: DatabaseConnection) -> Dict[str, int]:
    """
    Query Postgres for paper counts per domain.

    Assumes a 'papers' table with columns:
      - paper_id (text)
      - domain (text)  ← IMPORTANT
    """
    query = """
    SELECT domain, COUNT(*) AS num_papers
    FROM papers
    GROUP BY domain;
    """

    rows = await db.fetch(query)
    counts: Dict[str, int] = {}
    for r in rows:
        domain = r["domain"] or "unknown"
        counts[domain] = int(r["num_papers"])

    logger.info("Loaded domain counts", num_domains=len(counts))
    return counts


def build_fairness_config(counts: Dict[str, int]) -> Dict[str, Any]:
    if not counts:
        raise ValueError("No domain counts found – cannot build fairness config")

    max_count = max(counts.values())
    threshold = max_count * UNDERREP_THRESHOLD_RATIO

    under_served_domains = []
    domains_cfg: Dict[str, Dict[str, Any]] = {}

    for domain, n in counts.items():
        under_served = n < threshold
        boost = DEFAULT_DOMAIN_BOOST if under_served else 1.0

        if under_served:
            under_served_domains.append(domain)

        domains_cfg[domain] = {
            "num_papers": n,
            "under_served": under_served,
            "boost_factor": boost,
        }

    cfg = {
        "paper_domain_fairness": {
            "metric": "representation_count",
            "disparity_threshold_ratio": UNDERREP_THRESHOLD_RATIO,
            "under_served_domains": under_served_domains,
            "domains": domains_cfg,
        },
        "metadata": {
            "source": "domain_representation_fairness.py",
            "generated_at": datetime.utcnow().isoformat(),
            "max_domain_count": max_count,
        },
    }
    return cfg


def save_fairness_config(cfg: Dict[str, Any]) -> None:
    FAIRNESS_CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    print(f"✅ fairness_config.json saved to {FAIRNESS_CONFIG_PATH.resolve()}")
    print("Under-served domains:", cfg["paper_domain_fairness"]["under_served_domains"])


async def main() -> None:
    print("\n" + "=" * 80)
    print("  Domain Representation Fairness (Postgres only)")
    print("=" * 80)

    db = DatabaseConnection()
    await db.connect()
    try:
        counts = await load_domain_counts(db)
    finally:
        await db.disconnect()

    if not counts:
        print("⚠ No domain counts found – check 'papers' table / domain column")
        return

    cfg = build_fairness_config(counts)
    save_fairness_config(cfg)

    print("\n--- Domain representation counts ---")
    for domain, n in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{domain:25} | n={n:5d}")


if __name__ == "__main__":
    asyncio.run(main())
