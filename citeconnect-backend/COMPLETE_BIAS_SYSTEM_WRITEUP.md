# Complete Bias Detection & Mitigation System - Full Writeup

## Table of Contents

1. [Overview](#overview)
2. [Type 1: User-Profile Bias](#type-1-user-profile-bias)
3. [Type 2: Domain Representation Bias](#type-2-domain-representation-bias)
4. [How They Work Together](#how-they-work-together)
5. [Complete Workflow](#complete-workflow)
6. [Data Sources](#data-sources)
7. [Config Files](#config-files)
8. [Runtime Application](#runtime-application)
9. [Examples](#examples)

---

## Overview

CiteConnect uses a **dual-layer bias mitigation system** to ensure fair recommendations:

1. **User-Profile Bias Mitigation** - Ensures all users get fair treatment
2. **Domain Representation Bias Mitigation** - Ensures all domains get fair representation

Both systems work **automatically** during recommendation generation.

---

## Type 1: User-Profile Bias

### What It Detects

**Problem:** Some user groups consistently receive lower-quality recommendations.

**Example:**
- Fintech users: average `combined_score = 0.20`
- Quantum computing users: average `combined_score = 0.35`
- **Disparity:** Fintech users are 42.9% worse off

**Why it happens:**
- Training data imbalance (more quantum computing papers)
- Model learns patterns better for some domains
- Cold-start users have limited interaction history

### Detection Process

**Script:** `scripts/bias_slicing_cold_start.py`

**Step 1: Load Data from PostgreSQL**

```sql
SELECT
    c.user_id,
    c.combined_score,           -- How good were recommendations for this user
    c.profile_alignment,
    c.ground_truth_quality,
    p.primary_domain,           -- fintech, healthcare, quantum_computing
    p.research_stage,           -- undergraduate, masters, phd
    p.reading_level             -- introductory, intermediate, advanced
FROM cold_start_evaluations c
JOIN user_profiles_extended p ON c.user_id = p.user_id
```

**Data Sources:**
- `cold_start_evaluations` table - User evaluation results (populated by `evaluate_all_users.py`)
- `user_profiles_extended` table - User demographics

**Step 2: Slice by Demographics**

Groups users by:
- `primary_domain` (fintech, healthcare, quantum_computing)
- `research_stage` (undergraduate, masters, phd)
- `reading_level` (introductory, intermediate, advanced)

**Step 3: Compute Metrics per Slice**

For each slice, calculates:
- `mean_combined_score` - Average recommendation quality
- `mean_profile_alignment` - How well papers match profile
- `mean_ground_truth_quality` - Papers in citation networks
- `user_count` - Number of users in slice

**Step 4: Detect Disparities**

```python
# For each field (primary_domain, research_stage, reading_level)
best_slice = slice with highest mean_combined_score
worst_slice = slice with lowest mean_combined_score

disparity = (best_value - worst_value) / best_value

if disparity > 0.15:  # 15% threshold
    # Flag as bias
```

**Example Finding:**
```json
{
  "field": "primary_domain",
  "metric": "mean_combined_score",
  "best_slice": "quantum_computing",
  "best_value": 0.35,
  "worst_slice": "fintech",
  "worst_value": 0.20,
  "disparity": 0.429  // 42.9% worse
}
```

**Step 5: Generate Mitigation Config**

For each biased field:
- Identifies `worst_slice` as underperforming
- Calculates:
  - `boost_factor = 1.0 + min(disparity, 0.25)` (max 1.25x)
  - `min_score_floor = best_value * (1 - disparity * 0.5)`

**Outputs:**
- `bias_report_cold_start_before.json` - Full analysis report
- `bias_config/bias_mitigation_config.json` - Auto-generated mitigation rules

### Mitigation Config Structure

**File:** `bias_config/bias_mitigation_config.json`

```json
{
  "cold_start": {
    "minilm": {
      "primary_domain": {
        "underperforming_slices": ["fintech"],
        "boost_factor": 1.25,
        "min_score_floor": 0.253
      },
      "research_stage": {
        "underperforming_slices": ["masters"],
        "boost_factor": 1.25,
        "min_score_floor": 0.246
      },
      "reading_level": {
        "underperforming_slices": ["intermediate"],
        "boost_factor": 1.25,
        "min_score_floor": 0.256
      }
    }
  }
}
```

### How Mitigation Works at Runtime

**Location:** `RecommendationService._apply_multi_factor_scoring()` (Line 1683)

**Step 1: Get User Profile**

```python
profile = await user_repo.get_profile(user_id)
# → {primary_domain: "fintech", research_stage: "masters", reading_level: "intermediate"}
```

**Step 2: Compute Mitigation Policy**

```python
mitigation_policy = self._get_mitigation_policy_for_profile(profile, model='minilm')
# Checks bias_mitigation_config.json
# User matches: fintech + masters + intermediate
# Returns: {factor: 1.953, min_score_threshold: 0.246, applied_rules: [...]}
```

**Step 3: Apply Boost During Scoring**

```python
# Calculate base score
final_score = (
    semantic_score * 0.40 +
    citation_score * 0.20 +
    recency_score * 0.15 +
    ground_truth_score * 0.10 +
    reading_level_score * 0.10 +
    diversity_score * 0.05
)

# Apply user-profile boost (Line 1683)
final_score *= mitigation_policy.factor  # 1.953x for fintech+masters+intermediate

# Filter low-scoring papers (Line 1700)
if final_score < mitigation_policy.min_score_threshold:  # 0.246
    # Remove from recommendations
```

**Result:**
- Fintech user's papers get 1.953x boost
- Papers below 0.246 are filtered out
- User gets better recommendations

---

## Type 2: Domain Representation Bias

### What It Detects

**Problem:** Some domains are under-represented in the corpus, leading to fewer recommendations from those domains.

**Example:**
- AI domain: 6,379 papers (max)
- Fintech domain: 3,803 papers
- **Threshold:** 3,189.5 (50% of max)
- **Status:** Fintech is NOT under-served (3,803 > 3,189.5)

**Why it matters:**
- If a domain has very few papers, it gets less representation
- Users might not see papers from their domain
- System appears biased toward domains with more papers

### Detection Process

**Script:** `scripts/domain_representation_fairness.py`

**Step 1: Query PostgreSQL**

```sql
SELECT domain, COUNT(*) AS num_papers
FROM papers
GROUP BY domain;
```

**Results:**
```
ai: 6,379 papers
healthcare: 5,925 papers
quantum_computing: 4,796 papers
fintech: 3,803 papers
```

**Step 2: Calculate Under-Served Domains**

```python
max_count = 6379  # AI has most papers
threshold = max_count * 0.5  # 50% of max = 3,189.5

# Check each domain
for domain, count in domain_counts.items():
    if count < threshold:
        under_served_domains.append(domain)
```

**Step 3: Generate Fairness Config**

```json
{
  "paper_domain_fairness": {
    "metric": "representation_count",
    "disparity_threshold_ratio": 0.5,
    "under_served_domains": [],  // Empty if all domains are above threshold
    "domains": {
      "fintech": {
        "num_papers": 3803,
        "under_served": false,
        "boost_factor": 1.0
      },
      "ai": {
        "num_papers": 6379,
        "under_served": false,
        "boost_factor": 1.0
      }
    }
  }
}
```

**Output:** `fairness_config.json`

### How Mitigation Works at Runtime

**Location:** `RecommendationService._apply_fairness_reranking()` (Line 437)

**Step 1: After Enrichment**

```python
# Papers already have final_score (may be boosted by user-profile bias)
enriched = self._enrich_recommendations(papers)
# → Papers with final_score, metadata, etc.
```

**Step 2: Call Fairness Service**

```python
fairness_reranked = await self._apply_fairness_reranking(enriched)
    ↓
fairness_service.fairness_aware_rerank(recommendations, db=self.db)
```

**Step 3: Fairness Service Logic**

```python
# 1. Load config
cfg = load_fairness_config()
under_served = get_under_served_domains(cfg)
# → ["fintech"] (if fintech is under-served)

# 2. Query PostgreSQL for domain mapping
query = "SELECT paper_id, domain FROM papers WHERE domain IS NOT NULL"
domain_map = {"paper_123": "fintech", "paper_456": "healthcare", ...}

# 3. For each recommendation
for rec in recommendations:
    paper_id = rec["paper_id"]
    domain = domain_map[paper_id]  # Look up from PostgreSQL
    
    # 4. Check if domain is under-served
    if domain in under_served:
        boost = get_domain_boost_factor(cfg, domain)  # → 1.05
        rec["score"] *= boost  # Boost score
    
    # 5. Add domain to paper
    rec["domain"] = domain

# 6. Rerank by boosted scores
return sorted(recommendations, key=lambda r: r["score"], reverse=True)
```

**Result:**
- Papers from under-served domains get 1.05x boost
- Domains get fair representation in recommendations

---

## How They Work Together

### Sequential Application

```
User requests recommendations
    ↓
1. Generate candidates
    ↓
2. Multi-factor scoring
    ↓
3. USER-PROFILE BIAS MITIGATION (Line 1683)
   → Boost scores for underperforming users
   → Fintech user: all papers × 1.953
    ↓
4. Diversity filtering
    ↓
5. Top-N selection
    ↓
6. Enrichment
    ↓
7. DOMAIN FAIRNESS RERANKING (Line 437)
   → Boost papers from under-served domains
   → Fintech papers: × 1.05
    ↓
8. Return final recommendations
```

### Multiplicative Effect

**Example: Fintech paper for Fintech user**

```
Base score: 0.20

Step 1: User-Profile Boost
- User is fintech + masters + intermediate
- Boost: 1.953x
- Score: 0.20 × 1.953 = 0.39

Step 2: Domain Fairness Boost
- Paper is from fintech domain
- If fintech is under-served: boost 1.05x
- Score: 0.39 × 1.05 = 0.41

Final: 0.41 (instead of 0.20)
```

---

## Complete Workflow

### Phase 1: Data Collection

**For User-Profile Bias:**
```bash
# Evaluate all users (populates cold_start_evaluations table)
python scripts/evaluate_all_users.py
```

**For Domain Bias:**
```bash
# No separate step needed - uses papers table directly
```

### Phase 2: Bias Detection

**User-Profile Bias:**
```bash
python scripts/bias_slicing_cold_start.py
```

**What it does:**
1. Queries `cold_start_evaluations` JOIN `user_profiles_extended`
2. Groups by `primary_domain`, `research_stage`, `reading_level`
3. Computes average `combined_score` per slice
4. Detects disparities > 15%
5. **Auto-generates** `bias_config/bias_mitigation_config.json`

**Domain Representation Bias:**
```bash
python scripts/domain_representation_fairness.py
```

**What it does:**
1. Queries `papers` table: `SELECT domain, COUNT(*) FROM papers GROUP BY domain`
2. Calculates max count and threshold (50% of max)
3. Identifies domains with count < threshold
4. **Generates** `fairness_config.json`

### Phase 3: Mitigation Application (Automatic)

**No manual step needed!** When a user requests recommendations:

1. **User-Profile Mitigation** (during scoring):
   - Loads `bias_mitigation_config.json`
   - Checks user profile against underperforming slices
   - Applies boost factor to all paper scores
   - Filters papers below min_score_floor

2. **Domain Fairness** (during reranking):
   - Loads `fairness_config.json`
   - Queries PostgreSQL for paper domains
   - Boosts papers from under-served domains
   - Reranks by boosted scores

### Phase 4: Monitoring

**Re-evaluate after mitigation:**
```bash
# Re-run evaluations
python scripts/evaluate_all_users.py

# Re-run bias detection
python scripts/bias_slicing_cold_start.py

# Compare before/after disparities
```

---

## Data Sources

### User-Profile Bias

**Detection:**
- `cold_start_evaluations` table (PostgreSQL)
  - Columns: `user_id`, `combined_score`, `profile_alignment`, `ground_truth_quality`
  - Populated by: `scripts/evaluate_all_users.py`
- `user_profiles_extended` table (PostgreSQL)
  - Columns: `user_id`, `primary_domain`, `research_stage`, `reading_level`

**Mitigation:**
- `bias_config/bias_mitigation_config.json` (file)
  - Contains: underperforming slices, boost factors, min score floors

### Domain Representation Bias

**Detection:**
- `papers` table (PostgreSQL)
  - Columns: `paper_id`, `domain`
  - Query: `SELECT domain, COUNT(*) FROM papers GROUP BY domain`

**Mitigation:**
- `fairness_config.json` (file)
  - Contains: under-served domains list
- `papers` table (PostgreSQL)
  - Queried at runtime to get domain for each paper

---

## Config Files

### 1. User-Profile Bias Config

**File:** `bias_config/bias_mitigation_config.json`

**Structure:**
```json
{
  "cold_start": {
    "minilm": {
      "primary_domain": {
        "underperforming_slices": ["fintech"],
        "boost_factor": 1.25,
        "min_score_floor": 0.253
      },
      "research_stage": {
        "underperforming_slices": ["masters"],
        "boost_factor": 1.25,
        "min_score_floor": 0.246
      }
    }
  }
}
```

**Generated by:** `bias_slicing_cold_start.py` (auto-generated)

**Used by:** `RecommendationService._get_mitigation_policy_for_profile()`

### 2. Domain Fairness Config

**File:** `fairness_config.json`

**Structure:**
```json
{
  "paper_domain_fairness": {
    "metric": "representation_count",
    "disparity_threshold_ratio": 0.5,
    "under_served_domains": ["fintech"],
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
  }
}
```

**Generated by:** `domain_representation_fairness.py`

**Used by:** `fairness_service.fairness_aware_rerank()`

---

## Runtime Application

### Code Flow

```
User Request → RecommendationService.generate_recommendations()
    ↓
generate_cold_start_recommendations()
    ↓
Step 1: Get user profile
    ↓
Step 2: Get mitigation policy (Line 337)
    → _get_mitigation_policy_for_profile()
    → Reads bias_mitigation_config.json
    → Returns: {factor: 1.953, min_score_threshold: 0.246, ...}
    ↓
Step 3: Retrieve candidates
    ↓
Step 4: Multi-factor scoring (Line 408)
    → _apply_multi_factor_scoring(mitigation_policy=...)
    → Line 1683: final_score *= mit_factor  // USER-PROFILE BOOST
    → Line 1700: Filter papers below threshold
    ↓
Step 5: Diversity filtering
    ↓
Step 6: Top-N selection
    ↓
Step 7: Enrichment
    ↓
Step 8: Domain fairness reranking (Line 437)
    → _apply_fairness_reranking()
    → fairness_aware_rerank(db=self.db)
    → Reads fairness_config.json
    → Queries PostgreSQL for domains
    → Boosts under-served domains  // DOMAIN BOOST
    → Reranks
    ↓
Return recommendations
```

### Key Code Locations

| Component | File | Line | What It Does |
|-----------|------|------|--------------|
| **User-Profile Detection** | `scripts/bias_slicing_cold_start.py` | 71-100 | Queries PostgreSQL, detects bias |
| **User-Profile Config** | `bias_config/bias_mitigation_config.json` | - | Mitigation rules |
| **User-Profile Policy** | `recommendation_service.py` | 116-230 | Computes mitigation policy |
| **User-Profile Boost** | `recommendation_service.py` | 1683 | Applies boost: `final_score *= factor` |
| **Domain Detection** | `scripts/domain_representation_fairness.py` | 71-92 | Queries PostgreSQL, detects under-served |
| **Domain Config** | `fairness_config.json` | - | Under-served domains list |
| **Domain Boost** | `fairness_service.py` | 181-184 | Applies boost: `score *= domain_boost` |
| **Domain Reranking** | `recommendation_service.py` | 437 | Calls fairness service |

---

## Examples

### Example 1: Fintech User Gets Recommendations

**User Profile:**
- `primary_domain`: "fintech"
- `research_stage`: "masters"
- `reading_level`: "intermediate"

**Configs:**
- User-profile: fintech, masters, intermediate are underperforming
- Domain: fintech is under-served (hypothetically)

**Flow:**

```
1. Generate candidates
   → 100 candidate papers

2. Score papers
   → Base scores: 0.15 - 0.35

3. User-Profile Boost (Line 1683)
   → User matches: fintech + masters + intermediate
   → Boost factor: 1.25 × 1.25 × 1.25 = 1.953
   → All papers: score × 1.953
   → Filter papers below 0.246

4. Domain Fairness Boost (Line 437)
   → Fintech papers: score × 1.05
   → Rerank by boosted scores

5. Return top 10
   → Fintech papers rank higher
   → User gets better recommendations
```

**Result:**
- Papers boosted by 1.953x (user-profile)
- Fintech papers boosted by additional 1.05x (domain)
- Combined: 2.05x total boost for fintech papers for fintech users

### Example 2: Healthcare User Gets Recommendations

**User Profile:**
- `primary_domain`: "healthcare"
- `research_stage`: "phd"
- `reading_level`: "advanced"

**Configs:**
- User-profile: healthcare, phd, advanced are NOT underperforming
- Domain: healthcare is NOT under-served

**Flow:**

```
1. Generate candidates
   → 100 candidate papers

2. Score papers
   → Base scores: 0.20 - 0.40

3. User-Profile Boost (Line 1683)
   → User does NOT match underperforming slices
   → Boost factor: 1.0 (no boost)
   → Scores unchanged

4. Domain Fairness Boost (Line 437)
   → Healthcare is NOT under-served
   → Boost factor: 1.0 (no boost)
   → Scores unchanged

5. Return top 10
   → No boosts applied
   → Normal recommendations
```

**Result:**
- No bias mitigation needed
- User gets standard recommendations

---

## Summary

### User-Profile Bias

- **Detects:** Users in underperforming demographic slices
- **Mitigates:** Boosts all paper scores for those users
- **Applied:** During multi-factor scoring (Line 1683)
- **Config:** `bias_config/bias_mitigation_config.json`
- **Data:** PostgreSQL (`cold_start_evaluations`, `user_profiles_extended`)

### Domain Representation Bias

- **Detects:** Domains with low paper counts (< 50% of max)
- **Mitigates:** Boosts papers from under-served domains
- **Applied:** During reranking (Line 437)
- **Config:** `fairness_config.json`
- **Data:** PostgreSQL (`papers` table)

### Together

- **User-profile** ensures fairness to users
- **Domain fairness** ensures fairness to content
- **Both apply automatically** during recommendation generation
- **Multiplicative effect** when both apply to same paper

**The system is fully automated and works together to ensure fair and diverse recommendations!**
