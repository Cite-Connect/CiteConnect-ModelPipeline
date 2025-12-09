# CI Bias Detection - Implementation Summary

## ✅ What Was Done

### 1. Updated CI Configuration
**File:** `.github/workflows/ci.yml`

Added a new `bias-detection` job that:
- ✅ Runs **after** tests pass (depends on test job)
- ✅ **Non-blocking** (uses `continue-on-error: true`)
- ✅ Analyzes user-profile bias using `scripts/bias_slicing_cold_start.py`
- ✅ Extracts disparity metrics and compares against thresholds
- ✅ Uploads bias reports as downloadable artifacts
- ✅ Posts PR comments when bias is detected (≥20%)

### 2. Bias Detection Features
- **Alert Threshold:** 20% disparity → Shows warning
- **Critical Threshold:** 30% disparity → Shows critical warning
- **Non-Blocking:** CI always passes, just shows warnings
- **Automatic Reports:** Generates and uploads bias analysis reports
- **PR Comments:** Automatically comments on PRs when bias detected

### 3. File Requirements Clarified
✅ **Only runs `bias_slicing_cold_start.py`** (user-profile bias)
- Needs: DATABASE_URL (queries `cold_start_evaluations` and `user_profiles_extended`)
- No files required

❌ **Skipped `model_bias_slicing.py`** (paper-field bias)
- Requires: `offline_evaluation_results.json` (you don't have this)
- Parquet file in git is not sufficient alone

---

## 🎯 What You Need to Do

### Step 1: Add DATABASE_URL Secret to GitHub

**Instructions:** See `GITHUB_SECRETS_SETUP.md` for detailed steps.

**Quick Steps:**
1. Go to your repo: `https://github.com/YOUR_USERNAME/CiteConnect-ModelPipeline`
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Name: `DATABASE_URL`
5. Value: Your database connection string (e.g., `postgresql://user:pass@host:5432/db`)
6. Click **Add secret**

**Where to find your DATABASE_URL:**
- Check your `.env` file
- Supabase dashboard → Settings → Database → Connection String
- Your hosting provider's dashboard

### Step 2: Test the CI

Push a commit to trigger the workflow:

```bash
git add .
git commit -m "test: trigger CI with bias detection"
git push
```

### Step 3: Check Results

1. Go to GitHub → **Actions** tab
2. Click on your workflow run
3. You'll see two jobs:
   - ✅ **test** - Runs all tests
   - ⚠️ **bias-detection** - Runs bias analysis

4. Download artifacts to see reports

---

## 📊 What Happens When You Push Code

### Flow Diagram

```
You push code
     ↓
GitHub Actions triggered
     ↓
┌─────────────────────┐
│  JOB 1: Test        │
│  ✅ Run all tests   │
│  ✅ Check coverage  │
└─────────────────────┘
     ↓ (only if tests pass)
┌─────────────────────────────────┐
│  JOB 2: Bias Detection          │
│  📊 Connect to database          │
│  📊 Run bias_slicing_cold_start │
│  📊 Analyze disparities          │
│  📊 Generate reports             │
│  📊 Upload artifacts             │
│  💬 Comment on PR (if needed)   │
└─────────────────────────────────┘
     ↓
✅ CI Passes (always)
⚠️  Shows warnings if bias found
```

### Example Output (if bias detected)

```
================================================
  BIAS DETECTION RESULTS
================================================
Max Disparity: 44.51%
Alert Threshold: 20%
Critical Threshold: 30%
================================================

🚨 CRITICAL: High bias detected!
   Disparity 44.51% >= 30%

⚠️  This would normally block deployment in production.
   Currently set to WARNING only (non-blocking).

Action required:
  1. Review bias reports in artifacts
  2. Apply mitigation strategies
  3. Re-run evaluation

================================================
```

### PR Comment (if bias ≥20%)

On pull requests, you'll see an automated comment like:

```
🚨 Bias Detection CRITICAL

🔴 Max Disparity Detected: 44.51%

🚨 Critical bias levels detected! This would block deployment in production.

### What This Means
User-profile bias analysis found significant disparities between 
different user groups (e.g., by domain, research stage, or reading level).

### Next Steps
1. 📥 Download `bias-reports` from workflow artifacts
2. 📊 Review `bias_report_cold_start_before.json`
3. 🔧 Check if `bias_mitigation_config.json` was auto-generated
4. ✅ Verify mitigation is applied in RecommendationService
5. 🔄 Re-run evaluation after applying fixes

---
⚙️ This is currently non-blocking and won't prevent merging.
```

---

## 📁 Artifacts Available

After each CI run, you can download:

### `bias-reports` artifact contains:
- `bias_report_cold_start_before.json` - Full bias analysis
  - Slice-by-slice metrics
  - Bias findings with disparities
  - User counts per slice
- `bias_config/bias_mitigation_config.json` - Auto-generated mitigation config
  - Underperforming slices
  - Boost factors
  - Min score floors

### `coverage-report` artifact contains:
- HTML coverage reports from tests

---

## 🔧 Configuration Options

### Current Thresholds

In the CI file (lines 148-149):
```yaml
alert_threshold = 0.20  # 20%
critical_threshold = 0.30  # 30%
```

**To adjust:**
- Edit these values in `.github/workflows/ci.yml`
- Lower values = more sensitive
- Higher values = less sensitive

### Make it Blocking (Optional)

To make bias detection **block merges** in the future:

1. Remove line 77 from `.github/workflows/ci.yml`:
```yaml
continue-on-error: true  # Remove this line
```

2. The CI will then **fail** if disparity ≥ critical threshold

---

## 🗃️ Database Requirements

The bias detection needs:

### Tables Required:
1. **`cold_start_evaluations`** - Must have data
   - Contains: user_id, embedding_model, combined_score, etc.
2. **`user_profiles_extended`** - Must have data
   - Contains: user_id, primary_domain, research_stage, reading_level

### Query Type:
- **READ-ONLY** - No data is modified
- Joins the two tables and computes aggregate metrics

### If Tables Are Empty:
- Script will report "No data found"
- CI will still pass (non-blocking)
- No bias reports generated

---

## 🚀 Next Steps After Setup

### 1. First Run
- Add DATABASE_URL secret
- Push code to trigger CI
- Check Actions tab for results
- Download artifacts to review reports

### 2. If Bias Detected
- Review `bias_report_cold_start_before.json`
- Check auto-generated `bias_mitigation_config.json`
- Verify RecommendationService loads the config
- Apply mitigation and re-evaluate

### 3. Monitor Over Time
- Bias detection runs on every push/PR
- Track trends in disparity over time
- Adjust mitigation as needed

### 4. Optional: Add Scheduled Monitoring
Create `.github/workflows/bias-monitoring.yml` to run weekly:
- Monitors production bias regularly
- Creates GitHub issues when bias detected
- Independent of code pushes

---

## ❓ FAQ

### Q: What if I don't have evaluation data yet?
**A:** The script will report "No data found" but CI will still pass. Populate the tables first.

### Q: Can I test this locally?
**A:** Yes! Run:
```bash
cd citeconnect-backend
export DATABASE_URL="your_db_url"
python scripts/bias_slicing_cold_start.py
```

### Q: Will this slow down my CI?
**A:** Adds ~2-3 minutes. Runs only after tests pass, so doesn't delay feedback.

### Q: Can I skip bias detection for specific commits?
**A:** Yes, add `[skip ci]` or `[ci skip]` to commit message. But better to leave it running!

### Q: What about the paper-field bias detection?
**A:** Skipped for now since you don't have `offline_evaluation_results.json`. 
Once you generate that file, we can add it to CI.

---

## 📚 Files Created/Modified

### Modified:
- `.github/workflows/ci.yml` - Added bias-detection job

### Created:
- `GITHUB_SECRETS_SETUP.md` - Instructions for adding secrets
- `CI_BIAS_DETECTION_SUMMARY.md` - This summary (you're reading it!)

### Referenced:
- `scripts/bias_slicing_cold_start.py` - Existing bias detection script
- `BIAS_MITIGATION_WORKFLOW.md` - Existing bias mitigation guide

---

## ✅ Checklist

- [ ] Read `GITHUB_SECRETS_SETUP.md`
- [ ] Add `DATABASE_URL` secret to GitHub
- [ ] Push a commit to test CI
- [ ] Check Actions tab for results
- [ ] Download and review bias reports
- [ ] Verify mitigation config is generated
- [ ] Apply mitigation if needed

---

## 🎉 You're All Set!

The CI now includes automated bias detection that:
- ✅ Runs on every push and PR
- ✅ Non-blocking (warnings only)
- ✅ Generates detailed reports
- ✅ Auto-creates mitigation configs
- ✅ Comments on PRs with findings

Just add the DATABASE_URL secret and you're good to go! 🚀
