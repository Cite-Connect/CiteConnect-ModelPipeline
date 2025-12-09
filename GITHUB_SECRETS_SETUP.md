# GitHub Secrets Setup Guide

## How to Add DATABASE_URL Secret to GitHub

Your CI now requires the `DATABASE_URL` secret to run bias detection. Follow these steps:

### Step 1: Get Your Database URL

Your database URL should be in this format:
```
postgresql://username:password@host:port/database
```

**Example:**
```
postgresql://myuser:mypassword@db.example.com:5432/citeconnect
```

**For Supabase:**
```
postgresql://postgres.xxxxx:password@aws-0-us-east-1.pooler.supabase.com:5432/postgres
```

You can find your database URL in:
- Your `.env` file (look for `DATABASE_URL`)
- Supabase dashboard → Settings → Database → Connection String
- Your hosting provider's dashboard

### Step 2: Add Secret to GitHub

1. **Go to your GitHub repository**
   ```
   https://github.com/YOUR_USERNAME/CiteConnect-ModelPipeline
   ```

2. **Navigate to Settings**
   - Click on "Settings" tab (top right of repo page)

3. **Go to Secrets and Variables**
   - In the left sidebar, click "Secrets and variables"
   - Click "Actions"

4. **Add New Secret**
   - Click "New repository secret" button
   - Name: `DATABASE_URL`
   - Secret: Paste your database connection string
   - Click "Add secret"

### Step 3: Verify Secret is Added

You should see:
```
✅ DATABASE_URL
   Updated 1 minute ago
```

### Step 4: Test the CI

Push a commit or create a PR to test:

```bash
git add .
git commit -m "test: trigger CI with bias detection"
git push
```

Go to the "Actions" tab in GitHub to see the workflow run.

---

## What the CI Will Do Now

When you push code:

1. ✅ **Test Job** - Runs all tests from `tests/` folder
2. ⚠️ **Bias Detection Job** - Runs user-profile bias analysis
   - Connects to your database using `DATABASE_URL`
   - Queries `cold_start_evaluations` and `user_profiles_extended` tables
   - Analyzes disparities across user slices
   - Generates bias reports
   - **Non-blocking**: Won't fail CI, just shows warnings

---

## Important Notes

### Database Access
- The CI uses **READ-ONLY** queries
- It queries existing evaluation data
- No data is modified or deleted

### Required Tables
The bias detection needs these tables to exist and have data:
- `cold_start_evaluations`
- `user_profiles_extended`

If these tables are empty, the script will report "No data found" but won't fail.

### Thresholds
Current bias detection thresholds:
- **Alert**: 20% disparity → Shows warning
- **Critical**: 30% disparity → Shows critical warning (but doesn't block)

### Non-Blocking Behavior
The bias detection job has `continue-on-error: true`, which means:
- ✅ If bias is found: Shows warning, CI passes
- ✅ If script fails: CI still passes
- ✅ You can always merge PRs

To make it **blocking** in the future, remove this line from `.github/workflows/ci.yml`:
```yaml
continue-on-error: true  # Remove this to make it blocking
```

---

## Artifacts Available After Each Run

After CI runs, you can download:

1. **bias-reports** artifact contains:
   - `bias_report_cold_start_before.json` - Full analysis
   - `bias_config/bias_mitigation_config.json` - Auto-generated mitigation config

2. **coverage-report** artifact contains:
   - HTML coverage reports from tests

---

## Troubleshooting

### "DATABASE_URL not found"
- Make sure you added the secret exactly as `DATABASE_URL`
- Check spelling and case (must be uppercase)
- Re-push to trigger workflow again

### "No data in cold_start_evaluations"
- Your database tables might be empty
- Run evaluation scripts locally first to populate data
- Or use staging/test database with sample data

### Script fails with connection error
- Check your database URL is correct
- Verify database allows connections from GitHub Actions IPs
- For cloud databases, check firewall/security group settings

### Want to use different database for CI?
Create a separate `DATABASE_URL_CI` secret pointing to test/staging DB:
```yaml
env:
  DATABASE_URL: ${{ secrets.DATABASE_URL_CI }}
```

---

## Next Steps

1. ✅ Add `DATABASE_URL` secret to GitHub
2. ✅ Push a commit to test the CI
3. ✅ Check Actions tab to see results
4. ✅ Download artifacts to review bias reports
5. ✅ Set up mitigation if bias is detected

---

## Questions?

If you need help:
1. Check the Actions tab for detailed logs
2. Look at the bias detection job output
3. Download artifacts to see the full reports
