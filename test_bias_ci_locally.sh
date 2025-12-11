#!/bin/bash
# Local test script to simulate CI bias detection steps

set -e  # Exit on error

echo "=================================================="
echo "  Testing Bias Detection CI Steps Locally"
echo "=================================================="
echo ""

# Set environment variables (same as CI)
export DATABASE_URL="${DATABASE_URL:-your_database_url_here}"
export ENVIRONMENT="test"
export SUPABASE_URL="http://example.com"
export SECRET_KEY="dummy-secret-key"

echo "✅ Environment variables set"
echo ""

# Navigate to citeconnect-backend directory
cd citeconnect-backend

echo "=================================================="
echo "  Step 1: User-Profile Bias Detection"
echo "=================================================="

# Run user-profile bias detection
python scripts/bias_slicing_cold_start.py

if [ -f "bias_report_cold_start_before.json" ]; then
  echo "✅ User bias report generated"
  
  # Extract max disparity using heredoc (matches CI exactly)
  MAX_DISPARITY=$(python3 << 'PYSCRIPT'
import json
with open("bias_report_cold_start_before.json") as f:
    data = json.load(f)
findings = data.get("bias_findings", [])
if findings:
    print(f"{max(f['disparity'] for f in findings):.4f}")
else:
    print("0.0000")
PYSCRIPT
)
  
  # Extract bias count
  BIAS_COUNT=$(python3 << 'PYSCRIPT'
import json
with open("bias_report_cold_start_before.json") as f:
    data = json.load(f)
print(len(data.get("bias_findings", [])))
PYSCRIPT
)
  
  # Calculate percentage
  MAX_DISPARITY_PCT=$(python3 -c "print(f'{float($MAX_DISPARITY) * 100:.1f}%')" 2>/dev/null || echo "N/A")
  
  echo ""
  echo "📊 Results:"
  echo "   - Max Disparity: $MAX_DISPARITY ($MAX_DISPARITY_PCT)"
  echo "   - Bias Findings: $BIAS_COUNT"
else
  echo "⚠️ Bias report not generated (possibly no data in DB)"
  MAX_DISPARITY="0.0000"
  BIAS_COUNT="0"
fi

echo ""
echo "=================================================="
echo "  Step 2: Paper-Field Bias Detection"
echo "=================================================="

# Check for required files (relative to citeconnect-backend directory)
if [ -f "offline_evaluation_results.json" ] && [ -f "data/combined_gcs_data.parquet" ]; then
  echo "✅ Required files found"
  
  # Run paper-field bias detection
  python scripts/model_bias_slicing.py
  
  if [ -f "model_bias_report.json" ]; then
    echo "✅ Field bias report generated"
    
    # Extract field ratio (simulating CI)
    FIELD_RATIO=$(python -c 'import json; data=json.load(open("model_bias_report.json")); ur=data.get("underrepresented_fields",[]); print(f"{len(ur):.0f}" if ur else "0")' 2>/dev/null || echo "0")
    
    echo "📊 Underrepresented Fields Found: $FIELD_RATIO"
  else
    echo "⚠️ Field bias report not generated"
    FIELD_RATIO="0"
  fi
else
  echo "⚠️ Required files not found (offline_evaluation_results.json or data/combined_gcs_data.parquet)"
  echo "⏭️  Skipping field bias detection"
  FIELD_RATIO="0"
fi

echo ""
echo "============================================================"
echo "  📊 BIAS DETECTION RESULTS (Matching CI)"
echo "============================================================"
echo "User-Profile Max Disparity: ${MAX_DISPARITY}"
echo "User-Profile Bias Count: ${BIAS_COUNT}"
echo ""
echo "Thresholds:"
echo "  - Alert: 0.20 (20%)"
echo "  - Critical (BLOCKS): 0.30 (30%)"
echo "============================================================"
echo ""

ALERT_THRESHOLD="0.20"
CRITICAL_THRESHOLD="0.30"
SHOULD_BLOCK=false

# Check if bias exceeds thresholds
if (( $(echo "$MAX_DISPARITY >= $CRITICAL_THRESHOLD" | bc -l) )); then
  echo "🚨 USER-PROFILE BIAS: CRITICAL!"
  echo "   Disparity ${MAX_DISPARITY} >= ${CRITICAL_THRESHOLD}"
  echo "   This BLOCKS deployment!"
  BIAS_STATUS="critical"
  SHOULD_BLOCK=true
elif (( $(echo "$MAX_DISPARITY >= $ALERT_THRESHOLD" | bc -l) )); then
  echo "⚠️  USER-PROFILE BIAS: WARNING"
  echo "   Disparity ${MAX_DISPARITY} >= ${ALERT_THRESHOLD}"
  echo "   Review recommended (non-blocking)"
  BIAS_STATUS="warning"
else
  echo "✅ USER-PROFILE BIAS: PASS"
  echo "   Disparity ${MAX_DISPARITY} < ${ALERT_THRESHOLD}"
  BIAS_STATUS="ok"
fi

if [ "$FIELD_RATIO" != "0" ]; then
  echo "⚠️  WARNING: Found $FIELD_RATIO underrepresented research fields"
  if [ "$BIAS_STATUS" = "ok" ]; then
    BIAS_STATUS="warning"
  fi
fi

echo ""
echo "============================================================"
echo "  📁 Generated Artifacts (Would be uploaded to CI)"
echo "============================================================"

# Check all artifacts that CI would upload
ARTIFACTS_FOUND=0

if [ -f "bias_report_cold_start_before.json" ]; then
  echo "✅ bias_report_cold_start_before.json"
  ARTIFACTS_FOUND=$((ARTIFACTS_FOUND + 1))
else
  echo "❌ bias_report_cold_start_before.json"
fi

if [ -f "bias_reports.json" ]; then
  echo "✅ bias_reports.json (metrics summary)"
  ARTIFACTS_FOUND=$((ARTIFACTS_FOUND + 1))
else
  echo "❌ bias_reports.json"
fi

if [ -f "bias_config/bias_mitigation_config.json" ]; then
  echo "✅ bias_config/bias_mitigation_config.json"
  ARTIFACTS_FOUND=$((ARTIFACTS_FOUND + 1))
else
  echo "❌ bias_config/bias_mitigation_config.json"
fi

if [ -f "model_bias_report.json" ]; then
  echo "✅ model_bias_report.json (field bias)"
  ARTIFACTS_FOUND=$((ARTIFACTS_FOUND + 1))
else
  echo "ℹ️  model_bias_report.json (not generated - field bias skipped)"
fi

if [ -f "fairness_config.json" ]; then
  echo "✅ fairness_config.json"
  ARTIFACTS_FOUND=$((ARTIFACTS_FOUND + 1))
else
  echo "ℹ️  fairness_config.json (not generated - field bias skipped)"
fi

echo ""
echo "Total artifacts generated: $ARTIFACTS_FOUND"

echo ""
echo "=================================================="
echo "  Test Complete!"
echo "=================================================="
echo "Bias Status: $BIAS_STATUS"
echo ""

if [ "$BIAS_STATUS" = "critical" ]; then
  echo "❌ DEPLOYMENT BLOCKED - This would FAIL the CI pipeline!"
  echo ""
  echo "In CI, this would:"
  echo "  1. Exit with code 1"
  echo "  2. Block the pipeline"
  echo "  3. Prevent deployment"
  echo ""
  exit 1  # Exit with error to simulate CI blocking
elif [ "$BIAS_STATUS" = "warning" ]; then
  echo "⚠️  This would show as WARNING in CI (non-blocking)"
  echo "Pipeline would continue but review is recommended"
  exit 0
else
  echo "✅ This would PASS in CI and allow deployment"
  exit 0
fi
