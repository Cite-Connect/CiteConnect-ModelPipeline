# Bias Mitigation Workflow Guide

## Complete Workflow Overview

```
1. Run Bias Detection → 2. Config Generated → 3. Mitigation Applied → 4. Monitor Results
```

---

## Step-by-Step Process

### Step 1a: Run User-Profile Bias Detection ✅ (COMPLETED)

You've already done this! The script analyzed your data and found:

- **Fintech users**: 37.99% worse than quantum_computing
- **Masters students**: 44.51% worse than undergraduate  
- **Intermediate readers**: 33.98% worse than introductory

**Outputs:**
- `bias_report_cold_start_before.json` - Full analysis report
- `bias_config/bias_mitigation_config.json` - Mitigation rules (auto-generated)

**Command:**
```bash
python scripts/bias_slicing_cold_start.py
```

---

### Step 1b: Run Paper-Field Bias Detection (Optional but Recommended)

Analyze model performance by research field to identify under-served fields.

**Command:**
```bash
python scripts/model_bias_slicing.py
```

**Outputs:**
- `model_bias_report.json` - Full field-level analysis
- `fairness_config.json` - Field fairness rules (auto-generated)
- `bias_plots/model_precision_by_field.png` - Visualization

**What it does:**
- Analyzes precision@10 by `primary_field` (paper's research field)
- Identifies fields with precision < 80% of best field
- Generates fairness config for field-based boosting

---

### Step 2: Verify Mitigation Configs ✅ (COMPLETED)

The config file is now correctly structured and the code has been fixed to read it.

**Config Locations:**
- `bias_config/bias_mitigation_config.json` - User-profile mitigation
- `fairness_config.json` - Paper-field fairness

**User-Profile Config contains:**
- Underperforming slices: `fintech`, `masters`, `intermediate`
- Boost factors: `1.25x` for each
- Min score floors: `0.252`, `0.242`, `0.255`

**Paper-Field Config contains:**
- Under-served fields: List of research fields with low precision
- Metric: `precision_at_10`
- Threshold: Fields with precision < 80% of best field

---

### Step 3: Test Mitigation (NEXT STEP)

**Run the test script to verify mitigation is working:**

```bash
python scripts/test_bias_mitigation.py
```

**What it does:**
1. Finds users in underperforming slices
2. Generates recommendations for them
3. Checks if `mitigation_policy` is applied
4. Verifies boost factors are > 1.0
5. Shows which rules were applied

**Expected Output:**
```
✅ MITIGATION IS ACTIVE!
   Scores are being boosted by 1.953x  (if user matches all 3 slices)
   Applied Rules: 3
     - primary_domain=fintech (boost: 1.25)
     - research_stage=masters (boost: 1.25)
     - reading_level=intermediate (boost: 1.25)
```

---

### Step 4: Generate Recommendations with Mitigation

**Test with a specific user:**

```python
from app.db.connection import db
from app.services.recommendation_service import RecommendationService
import asyncio

async def test():
    await db.connect()
    service = RecommendationService(db)
    
    # Find a fintech user
    user_id = 7  # Replace with actual fintech user ID
    
    result = await service.generate_recommendations(
        user_id=user_id,
        count=10,
        model='minilm'
    )
    
    # Check mitigation
    policy = result['mitigation_policy']
    print(f"Boost Factor: {policy['factor']}")
    print(f"Applied Rules: {policy['applied_rules']}")
    
    await db.disconnect()

asyncio.run(test())
```

**What to look for:**
- `mitigation_policy.factor` should be > 1.0 for underperforming users
- `mitigation_policy.applied_rules` should list matched slices
- Paper scores should be higher than without mitigation

---

### Step 5: Evaluate Impact

**After running recommendations with mitigation, evaluate the results:**

```bash
# This will evaluate all users and store results in cold_start_evaluations
python scripts/evaluate_all_users.py
```

**Then re-run bias detection to see if disparities are reduced:**

```bash
python scripts/bias_slicing_cold_start.py
```

**Compare results:**
- Before mitigation: Disparities of 37.99%, 44.51%, 33.98%
- After mitigation: Should see reduced disparities

---

### Step 6: Monitor Over Time

**Regular monitoring workflow:**

1. **Weekly/Monthly**: Run bias detection
   ```bash
   python scripts/bias_slicing_cold_start.py
   ```

2. **Check if new biases emerge:**
   - Review `bias_report_cold_start_before.json`
   - Check if new slices are underperforming

3. **Update config if needed:**
   - Config is auto-generated, but you can manually edit if needed
   - Restart the service to reload config

4. **Track metrics:**
   - Monitor `cold_start_evaluations` table
   - Check if combined_score improves for underperforming slices

---

## How Mitigation Works in Production

### Dual-Layer Bias Mitigation System

The system uses **two complementary approaches** to ensure fairness:

#### 1. User-Profile-Based Mitigation (Cold-Start Bias)
- **Config**: `bias_config/bias_mitigation_config.json`
- **Purpose**: Boosts recommendations for users in underperforming demographic slices
- **Applied during**: Multi-factor scoring phase
- **Mechanism**: 
  - Checks user's `primary_domain`, `research_stage`, `reading_level`
  - Applies `boost_factor` (e.g., 1.25x) to final scores
  - Filters papers below `min_score_floor`

#### 2. Paper-Field-Based Fairness (Field-Level Fairness)
- **Config**: `fairness_config.json`
- **Purpose**: Boosts papers from under-served research fields
- **Applied during**: Final reranking phase (after enrichment)
- **Mechanism**:
  - Identifies under-served fields via `model_bias_slicing.py`
  - Applies 1.05x boost to papers in those fields
  - Reranks recommendations by boosted scores

### Automatic Application Flow

When a user requests recommendations:

1. **Service loads configs** (on startup, cached)
   - `bias_mitigation_config.json` → User-profile mitigation
   - `fairness_config.json` → Paper-field fairness

2. **Phase 1: User-Profile Mitigation** (during scoring)
   - Checks user profile against `underperforming_slices`
   - If match found:
     - Applies `boost_factor` to final scores
     - Filters papers below `min_score_floor`

3. **Phase 2: Paper-Field Fairness** (after enrichment)
   - Checks each paper's `primary_field`
   - If field is under-served:
     - Applies 1.05x boost to score
   - Reranks by boosted scores

4. **Returns recommendations** with both mitigations applied

### Example Flow

**User Profile:**
- Domain: `fintech`
- Stage: `masters`
- Reading: `intermediate`

**Phase 1: User-Profile Mitigation (during scoring)**
```
Base Score: 0.20
  × 1.25 (fintech boost)
  × 1.25 (masters boost)
  × 1.25 (intermediate boost)
= 0.391 (after user-profile mitigation)
```

**Min Score Floor:**
- Papers below `0.242` are filtered out

**Phase 2: Paper-Field Fairness (after enrichment)**
```
Paper A (Computer Science field): 0.391
  × 1.05 (under-served field boost)
= 0.411 (final score)

Paper B (Physics field): 0.391
  (not under-served, no boost)
= 0.391 (final score)
```

**Result**: Paper A ranks higher due to field-based fairness boost

---

## Troubleshooting

### Mitigation Not Working?

1. **Check config is loaded:**
   ```python
   service = RecommendationService(db)
   print(service.bias_config)  # Should not be empty
   ```

2. **Verify user profile matches:**
   ```sql
   SELECT primary_domain, research_stage, reading_level
   FROM user_profiles_extended
   WHERE user_id = <your_user_id>
   ```

3. **Check logs:**
   - Look for "Applied bias mitigation" debug logs
   - Check if `mitigation_policy` is in recommendation response

4. **Verify model name:**
   - Config is under `cold_start.minilm`
   - Make sure you're using `model='minilm'` in requests

---

## Files Reference

### Config Files
- `bias_config/bias_mitigation_config.json` - User-profile mitigation rules
- `fairness_config.json` - Paper-field fairness rules
- `bias_report_cold_start_before.json` - User-profile bias analysis
- `model_bias_report.json` - Paper-field bias analysis

### Scripts
- `scripts/bias_slicing_cold_start.py` - Analyze user-profile bias, generates mitigation config
- `scripts/model_bias_slicing.py` - Analyze paper-field bias, generates fairness config
- `scripts/generate_bias_mitigation_config.py` - Alternative config generator
- `scripts/test_bias_mitigation.py` - Test user-profile mitigation
- `scripts/evaluate_all_users.py` - Evaluate all users

### Code
- `app/services/recommendation_service.py` - Main recommendation engine
  - `_load_bias_config()` - Loads user-profile mitigation config
  - `_get_mitigation_policy_for_profile()` - Computes user-profile policy
  - `_apply_multi_factor_scoring()` - Applies user-profile boosts
  - `_apply_fairness_reranking()` - Applies paper-field fairness (NEW)
- `app/services/fairness_service.py` - Paper-field fairness service
  - `fairness_aware_rerank()` - Reranks by field-based fairness
  - `load_fairness_config()` - Loads fairness config

---

## Next Actions

1. ✅ **Run test script:**
   ```bash
   python scripts/test_bias_mitigation.py
   ```

2. ✅ **Generate recommendations for underperforming users** and verify boosts

3. ✅ **Re-evaluate users** after mitigation:
   ```bash
   python scripts/evaluate_all_users.py
   ```

4. ✅ **Re-run bias detection** to measure improvement:
   ```bash
   python scripts/bias_slicing_cold_start.py
   ```

5. ✅ **Compare before/after** disparities in the reports

---

## Success Criteria

✅ **Mitigation is working if:**
- `mitigation_policy.factor > 1.0` for underperforming users
- `mitigation_policy.applied_rules` contains matched slices
- Recommendation scores are higher for boosted users
- Bias disparities are reduced in follow-up analysis

---

## Questions?

- Check logs for "Applied bias mitigation" messages
- Verify config structure matches expected format
- Ensure user profiles match underperforming_slices values
- Test with a known underperforming user
