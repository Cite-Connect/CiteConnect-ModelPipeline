# Screenshots Checklist for Model Bias & Hyperparameter Tuning Report

## Quick Checklist

- [ ] Hyperparameter tuning script execution
- [ ] Best hyperparameters JSON file
- [ ] Sensitivity analysis script execution  
- [ ] Sensitivity analysis results JSON file
- [ ] Bias mitigation configuration JSON file
- [ ] Bias analysis report (if available)
- [ ] Code structure/file organization

---

## Detailed Screenshot Instructions

### 1. Hyperparameter Tuning Script Execution

**Command to run**:
```bash
cd citeconnect-backend
python scripts/hyperparameter_tuning_cold_start.py
```

**What to capture**:
- Terminal output showing:
  ```
  ================================================================================
    Hyperparameter Tuning (Cold-Start) – Placeholder Run
  ================================================================================
  ✅ Saved best hyperparameters to /path/to/bias_config/best_hyperparameters_cold_start.json
  ```

**Tips**:
- Make terminal window wide enough to see full path
- Include the command prompt showing you're in the right directory
- Highlight the success message

---

### 2. Best Hyperparameters Configuration File

**File**: `citeconnect-backend/bias_config/best_hyperparameters_cold_start.json`

**What to capture**:
- Full JSON file content
- Highlight these sections:
  - `search_space`: Shows all possible weight values
  - `best_config.weights`: Shows final selected weights
  - `generated_at`: Timestamp

**Tips**:
- Use a code editor with syntax highlighting
- Zoom in to make text readable
- Show file path in editor tab

---

### 3. Sensitivity Analysis Script Execution

**Command to run**:
```bash
cd citeconnect-backend
python scripts/sensitivity_cold_start_weights.py
```

**What to capture**:
- Full terminal output including:
  - Initial header with scenario count
  - Each scenario evaluation with timing
  - Final ranking table
  - Success message with file path

**Key sections to show**:
```
📊 Processing 5 scenarios × 1 user = 5 user evaluations
   Estimated papers to load: ~50 (ultra-minimal from ~1000)

--- [1/5] Evaluating scenario: baseline ---
    Weights: {'semantic': 0.4, 'citation': 0.2, ...}
    Loading ~10 candidate papers for user X... ✓ (X.Xs)
  ✓ Completed (X.Xs): users=1, total_recs=10, mean_final_score=0.533...

--- Scenario ranking by mean_final_score ---
semantic_minus_20 | mean_final=0.566 | std=0.052 | users=1
citation_plus_20  | mean_final=0.554 | std=0.063 | users=1
...
```

**Tips**:
- Scroll to capture all scenarios
- Make sure timing information is visible
- Include the final ranking table

---

### 4. Sensitivity Analysis Results File

**File**: `citeconnect-backend/bias_config/sensitivity_cold_start_weights.json`

**What to capture**:
- JSON file showing:
  - `baseline_weights`: Starting point
  - `scenarios`: All tested variations
  - `metrics`: Performance measurements

**Key sections to highlight**:
- `baseline.metrics.mean_final_score`: Baseline performance
- `semantic_minus_20.metrics`: Best performing scenario
- Comparison of all scenarios

**Tips**:
- Use a JSON formatter/viewer for better readability
- Create a side-by-side comparison if possible
- Highlight the best performing scenario

---

### 5. Bias Mitigation Configuration

**File**: `citeconnect-backend/bias_config/bias_mitigation_config.json`

**What to capture**:
- Full JSON structure showing:
  - `underperforming_slices`: Which groups need help
  - `boost_factor`: Multiplier (1.25 = 25% boost)
  - `min_score_floor`: Quality threshold

**Key information**:
- Shows fintech domain needs 1.25× boost
- Industry research stage needs 1.25× boost
- Intermediate reading level needs 1.25× boost

**Tips**:
- Highlight the underperforming slices
- Show the boost factors clearly
- Include context about what these mean

---

### 6. Bias Analysis Report (Optional)

**If you have bias slicing analysis output**, capture:
- Slice metrics table
- Disparity calculations
- Underperforming group identification

**File might be**: `bias_report_cold_start_before.json` or similar

---

### 7. Code Structure

**What to capture**:
- File explorer or IDE showing:
  ```
  citeconnect-backend/
  ├── scripts/
  │   ├── hyperparameter_tuning_cold_start.py
  │   ├── sensitivity_cold_start_weights.py
  │   └── generate_bias_mitigation_config.py
  └── bias_config/
      ├── best_hyperparameters_cold_start.json
      ├── sensitivity_cold_start_weights.json
      └── bias_mitigation_config.json
  ```

**Tips**:
- Use tree view if available
- Show file sizes or modification dates
- Highlight the key files

---

## Screenshot Quality Tips

1. **Resolution**: Use high resolution (at least 1920x1080)
2. **Text Size**: Ensure all text is readable when zoomed
3. **Annotations**: Add arrows or highlights to key sections
4. **Consistency**: Use same terminal/editor theme across screenshots
5. **File Paths**: Always show full file paths when possible
6. **Timestamps**: Include timestamps to show when work was done

---

## Suggested Presentation Order

1. **Introduction Slide**: Concepts diagram
2. **Hyperparameter Tuning**:
   - Script execution (Figure 1)
   - Configuration file (Figure 2)
3. **Sensitivity Analysis**:
   - Script execution (Figure 3)
   - Results file (Figure 4)
   - Results comparison table (create from JSON data)
4. **Bias Mitigation**:
   - Bias detection (if available)
   - Mitigation config (Figure 5)
5. **Code Structure** (Figure 6)
6. **Summary**: Key findings and recommendations

---

## Creating Comparison Tables

From the sensitivity analysis JSON, create a table:

| Scenario | Semantic | Citation | Recency | Mean Score | Std Dev | Change |
|----------|----------|----------|---------|------------|---------|--------|
| baseline | 0.400 | 0.200 | 0.150 | 0.533 | 0.089 | - |
| semantic_minus_20 | 0.348 | 0.217 | 0.163 | **0.566** | 0.052 | **+6.1%** |
| citation_plus_20 | 0.385 | 0.231 | 0.144 | 0.554 | 0.063 | +3.9% |
| semantic_plus_20 | 0.444 | 0.185 | 0.139 | 0.534 | 0.093 | +0.1% |
| citation_minus_20 | 0.417 | 0.167 | 0.156 | 0.543 | 0.084 | +1.9% |

**Screenshot this table** or create it in Excel/Google Sheets for better formatting.

---

## Quick Commands Reference

```bash
# Navigate to project
cd citeconnect-backend

# Run hyperparameter tuning
python scripts/hyperparameter_tuning_cold_start.py

# Run sensitivity analysis
python scripts/sensitivity_cold_start_weights.py

# View results
cat bias_config/best_hyperparameters_cold_start.json
cat bias_config/sensitivity_cold_start_weights.json
cat bias_config/bias_mitigation_config.json
```

---

## Final Checklist Before Submission

- [ ] All screenshots are clear and readable
- [ ] File paths are visible
- [ ] Key metrics are highlighted
- [ ] Comparison tables are included
- [ ] Code structure is shown
- [ ] All JSON files are properly formatted in screenshots
- [ ] Terminal output shows successful completion
- [ ] Timestamps are visible (if relevant)

