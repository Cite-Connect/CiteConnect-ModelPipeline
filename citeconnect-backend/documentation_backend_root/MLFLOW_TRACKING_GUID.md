# CiteConnect - MLflow Experiment Tracking Guide

## 🎯 Overview

This guide explains how CiteConnect tracks experiments, logs metrics, and evaluates model performance using MLflow, fulfilling the **Model Development Guidelines** requirements.

---

## 📊 What Gets Tracked

### **1. Cold-Start Evaluations** (Logged Immediately)

**When:** Every time recommendations are generated for users with 0-5 interactions

**Metrics Logged:**
- `profile_alignment` (0-1): How well papers match user profile
- `ground_truth_quality` (0-1): Papers in known citation networks
- `combined_score` (0-1): Weighted average (60% alignment, 40% GT)
- `recommendation_count`: Number of papers recommended
- `generation_time_ms`: Time taken to generate

**Parameters Logged:**
- `user_id`: Which user
- `model`: Which embedding model (all-MiniLM-L6-v2 or specter2)
- `user_stage`: cold_start, early, mature, expert
- `user_domain`: healthcare, fintech, quantum_computing

**Logged to:**
- ✅ MLflow run (immediate)
- ✅ `cold_start_evaluations` table (database)

---

### **2. Recommendation Events** (Every Recommendation)

**When:** Every recommendation request

**What's Logged:**
```json
{
  "user_id": 123,
  "model": "all-MiniLM-L6-v2",
  "user_stage": "cold_start",
  "recommendation_count": 10,
  "generation_time_ms": 145.23,
  "strategy_used": "profile_based_search",
  "evaluation_scores": {
    "profile_alignment": 0.75,
    "ground_truth_quality": 0.62
  }
}
```

**Logged to:**
- ✅ MLflow run
- ✅ `recommendation_events` table

---

### **3. Interaction-Based Evaluations** (After User Feedback)

**When:** After user has 10+ interactions (mature stage)

**Metrics:**
- `precision_at_10`: % of top 10 that user engaged with
- `recall_at_10`: % of relevant papers found
- `click_through_rate` (CTR): % of recommendations clicked
- `save_rate`: % of recommendations saved
- `interaction_count`: Number of interactions analyzed

**Logged to:**
- ✅ MLflow run
- ✅ `interaction_evaluations` table

---

### **4. Model Comparisons** (A/B Testing)

**When:** Comparing MiniLM vs SPECTER2 performance

**What's Compared:**
- Precision@10 for both models
- CTR for both models
- Lift percentage (improvement)
- Statistical significance (p-value)

**Logged to:**
- ✅ MLflow comparison run
- ✅ `ab_test_comparisons` table

---

## 🔄 Complete Tracking Flow

### **Flow 1: User Gets Recommendations (LOGS IMMEDIATELY)**

```
1. User requests recommendations
   ↓
2. RecommendationOrchestrator generates papers
   ↓
3. EvaluationService calculates metrics
   ↓
4. ExperimentService logs to MLflow ← TRACKING HAPPENS HERE
   ├─ Log run with parameters (user_id, model, stage)
   ├─ Log metrics (profile_alignment, ground_truth_quality)
   └─ Save to database (cold_start_evaluations)
   ↓
5. Return recommendations to user
```

### **Flow 2: User Interacts (LOGS AFTER INTERACTION)**

```
1. User clicks/saves/dismisses paper
   ↓
2. InteractionService records interaction
   ↓
3. Check if user has 10+ interactions
   ↓
4. If yes → EvaluationService.evaluate_mature_user()
   ├─ Calculate precision@10, CTR, save_rate
   ├─ ExperimentService.log_interaction_evaluation() ← TRACKING
   └─ MLflow logs performance metrics
   ↓
5. Update user embedding (if threshold met)
```

### **Flow 3: Model Comparison (PERIODIC)**

```
1. Run weekly/monthly comparison
   ↓
2. ExperimentService.compare_models('minilm', 'specter2')
   ├─ Get all evaluation data from last 7 days
   ├─ Calculate average metrics per model
   ├─ Compute statistical significance
   ├─ Log comparison to MLflow ← TRACKING
   └─ Save to ab_test_comparisons table
   ↓
3. Determine winner
   ↓
4. (Optional) Auto-deploy winning model
```

---

## 🧪 Testing MLflow Integration (AFTER PAPER DATA)

### Test 1: Verify MLflow is Running

```bash
# Check MLflow service
curl http://localhost:5000/health

# Open MLflow UI
open http://localhost:5000
```

### Test 2: Generate Recommendation with Tracking

```bash
# This will automatically log to MLflow
curl -X POST http://localhost:8000/api/v1/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "count": 10,
    "model_preference": "all-MiniLM-L6-v2",
    "session_id": "test-session-1"
  }'

# Check MLflow UI - you should see a new run
open http://localhost:5000/#/experiments/1
```

### Test 3: Verify Metrics in MLflow

In MLflow UI, you should see:

**Run Name:** `rec_1_2024-11-28T...`

**Parameters:**
- user_id: 1
- model: all-MiniLM-L6-v2
- user_stage: cold_start
- recommendation_count: 10

**Metrics:**
- generation_time_ms: ~150
- profile_alignment: 0.75
- ground_truth_quality: 0.62
- combined_score: 0.69

**Tags:**
- user_domain: healthcare
- user_stage: cold_start
- timestamp: 2024-11-28T...

### Test 4: Check Database Logging

```bash
docker-compose exec api python -c "
import asyncio
from app.db.connection import db

async def check():
    await db.connect()
    
    # Check cold-start evaluations
    evals = await db.fetch('''
        SELECT 
            user_id, 
            embedding_model,
            profile_alignment,
            ground_truth_quality,
            combined_score,
            evaluation_timestamp
        FROM cold_start_evaluations
        ORDER BY evaluation_timestamp DESC
        LIMIT 5
    ''')
    
    print('📊 Recent Cold-Start Evaluations:')
    for e in evals:
        print(f'  User {e[\"user_id\"]}: {e[\"embedding_model\"]}')
        print(f'    Profile: {e[\"profile_alignment\"]:.3f}, GT: {e[\"ground_truth_quality\"]:.3f}, Combined: {e[\"combined_score\"]:.3f}')
    
    # Check recommendation events
    events = await db.fetch('''
        SELECT 
            user_id,
            embedding_model,
            array_length(recommended_paper_ids, 1) as count,
            recommendation_strategy,
            event_timestamp
        FROM recommendation_events
        ORDER BY event_timestamp DESC
        LIMIT 5
    ''')
    
    print('\n📊 Recent Recommendation Events:')
    for e in events:
        print(f'  User {e[\"user_id\"]}: {e[\"embedding_model\"]} - {e[\"count\"]} papers via {e[\"recommendation_strategy\"]}')
    
    await db.disconnect()

asyncio.run(check())
"
```

---

## 📈 MLflow Dashboard Views

### **Experiment View** (Main Dashboard)

Navigate to: http://localhost:5000/#/experiments/1

You'll see:
- **All runs** for CiteConnect recommendations
- **Metrics comparison** across runs
- **Parameter filtering** (by user_stage, model, etc.)

### **Run Comparison**

Select multiple runs and click "Compare":
- Compare profile_alignment across user stages
- Compare generation_time_ms for different models
- See which model performs better for which segment

### **Metric Charts**

Built-in charts show:
- Profile alignment over time
- Ground truth quality trends
- Generation time distribution
- Success rate by user stage

---

## 🎓 Model Development Guidelines Compliance

### **Requirement 1: Model Validation** ✅

**Implemented in:** `EvaluationService.evaluate_cold_start()`

**What's Logged:**
- Performance metrics (profile alignment, ground truth quality)
- Logged to MLflow on every recommendation
- Stored in `cold_start_evaluations` table

**Thresholds:**
- Profile alignment >= 0.6
- Ground truth quality >= 0.5

### **Requirement 2: Bias Detection Using Slicing** ✅

**Implemented in:** `EvaluationService.detect_bias()`

**Slicing Dimensions:**
- research_stage
- domain
- reading_level
- years_experience

**How to Run:**
```python
# Will be called after collecting 100+ recommendation events
bias_report = await evaluation_service.detect_bias(
    recommendation_events=events,
    slicing_dimensions=['research_stage', 'domain', 'reading_level']
)

# Logs to MLflow with:
# - Variance per dimension
# - Worst performing slices
# - Mitigation recommendations
```

### **Requirement 3: Experiment Tracking** ✅

**Implemented in:** `ExperimentService`

**What's Tracked:**
- Every recommendation event
- All evaluation metrics
- Model parameters
- User segments

**Access in MLflow:**
```python
# All experiments
mlflow.search_experiments()

# Filter by user segment
mlflow.search_runs(filter_string="params.user_stage = 'cold_start'")
```

### **Requirement 4: Model Comparison** ✅

**Implemented in:** `ExperimentService.compare_models()`

**How to Run:**
```python
comparison = await experiment_service.compare_models(
    model_a='all-MiniLM-L6-v2',
    model_b='specter2',
    metric_name='precision_at_10'
)

# Returns:
# {
#   "winner": "all-MiniLM-L6-v2",
#   "lift_percentage": 12.5,
#   "model_a_value": 0.32,
#   "model_b_value": 0.36,
#   "sample_size_a": 150,
#   "sample_size_b": 145
# }
```

### **Requirement 5: CI/CD Pipeline** 🟡 (Partially Implemented)

**What's Ready:**
- Model validation code ✅
- Bias detection code ✅
- Experiment tracking ✅

**What's Missing:**
- GitHub Actions workflow
- Automated model push to registry

---

## 🔧 How to Use MLflow for Your Demo

### **Scenario 1: Show Model Performance Over Time**

1. Generate 10+ recommendations for different users
2. Open MLflow: http://localhost:5000
3. Navigate to experiment: "citeconnect-recommendations"
4. View runs sorted by timestamp
5. Show metrics chart for `profile_alignment`

**Demo Script:**
```bash
# Generate recommendations for 5 users
for user_id in {1..5}; do
  curl -X POST http://localhost:8000/api/v1/recommendations \
    -H "Content-Type: application/json" \
    -d "{
      \"user_id\": $user_id,
      \"count\": 10,
      \"model_preference\": \"all-MiniLM-L6-v2\",
      \"session_id\": \"demo-$user_id\"
    }"
done

# Then show MLflow dashboard with all runs
```

### **Scenario 2: Compare User Segments**

In MLflow UI:
1. Filter runs by `params.user_stage = 'cold_start'`
2. Note average `profile_alignment`
3. Filter by `params.user_stage = 'mature'` (when you have data)
4. Compare metrics

Shows that recommendations improve as users progress!

### **Scenario 3: Model A/B Test Results**

```bash
# After collecting 100+ interactions
docker-compose exec api python -c "
import asyncio
from app.db.connection import db
from app.services.bootstrap.experiment_service import ExperimentService

async def compare():
    await db.connect()
    exp_service = ExperimentService(db)
    await exp_service.initialize()
    
    comparison = await exp_service.compare_models(
        model_a='all-MiniLM-L6-v2',
        model_b='specter2',
        metric_name='precision_at_10'
    )
    
    print('🔬 Model Comparison Results:')
    print(f'  Winner: {comparison[\"winner\"]}')
    print(f'  Lift: {comparison[\"lift_percentage\"]}%')
    print(f'  MiniLM: {comparison[\"model_a_value\"]:.3f}')
    print(f'  SPECTER2: {comparison[\"model_b_value\"]:.3f}')
    
    await db.disconnect()

asyncio.run(compare())
"
```

---

## 📋 Evaluation Timeline

### **Phase 1: NOW (Without Paper Data)**
- ❌ Cannot test recommendations yet
- ❌ Cannot log recommendation events
- ❌ Cannot collect interaction metrics
- ✅ Can verify MLflow service is running
- ✅ Can verify database tables exist

### **Phase 2: After Paper Data Loaded**
- ✅ Generate recommendations → Auto-logged to MLflow
- ✅ Each recommendation gets:
  - Profile alignment score
  - Ground truth quality score
  - MLflow run with all metrics
- ✅ View in MLflow UI

### **Phase 3: After 10+ Interactions Per User**
- ✅ Interaction-based evaluation kicks in
- ✅ Calculate precision@10, CTR, save_rate
- ✅ Log mature user metrics to MLflow
- ✅ Compare with cold-start performance

### **Phase 4: After 100+ Total Interactions**
- ✅ Run bias detection across user segments
- ✅ Compare model A vs model B
- ✅ Generate statistical significance reports
- ✅ Present results in course submission

---

## 🎓 For Course Submission

### **What to Show:**

1. **MLflow Dashboard Screenshot**
   - Navigate to http://localhost:5000
   - Show experiment with 50+ runs
   - Show metrics over time

2. **Cold-Start Evaluation**
   - Show users getting quality recommendations immediately
   - Profile alignment > 0.6
   - Ground truth quality > 0.5

3. **Bias Detection Results**
   - Show performance across research_stage slices
   - Variance < 20% across segments
   - Mitigation strategies if bias found

4. **Model Comparison**
   - MiniLM vs SPECTER2 performance
   - Statistical significance
   - Winner selection

5. **Experiment Tracking**
   - All runs logged with parameters
   - Metrics tracked over time
   - Reproducible experiments

---

## 📊 MLflow UI Navigation

### **View All Runs**
```
http://localhost:5000/#/experiments/1
```

### **Compare Runs**
1. Select 2+ runs (checkboxes)
2. Click "Compare" button
3. View side-by-side metrics

### **Filter Runs**
```
# By user stage
params.user_stage = 'cold_start'

# By model
params.model = 'all-MiniLM-L6-v2'

# By domain
tags.user_domain = 'healthcare'

# By date
start_time > '2024-11-28'
```

### **Export Results**
1. Select runs
2. Click "..." menu
3. Download CSV

---

## 🧪 Testing Checklist

Before course submission, verify:

- [ ] MLflow service running
- [ ] Experiment created: "citeconnect-recommendations"
- [ ] 50+ runs logged
- [ ] Cold-start evaluations: avg profile_alignment > 0.6
- [ ] Interaction evaluations: avg precision@10 > 0.3
- [ ] Bias detection run with results
- [ ] Model comparison with winner identified
- [ ] All metrics visible in MLflow UI
- [ ] Runs can be filtered and compared
- [ ] Results exported to CSV

---

## 🎯 Summary

**Evaluation happens at TWO stages:**

1. **Recommendation Time** (Immediate)
   - Profile alignment
   - Ground truth quality
   - Logged to MLflow automatically ✅

2. **After Interactions** (Post-hoc)
   - Precision@10, CTR, save_rate
   - Logged when user reaches interaction threshold
   - Needs to be triggered ⏳

**MLflow tracks:**
- Every recommendation request
- All evaluation metrics
- Model parameters
- User context
- Performance over time

**For your demo today:**
- ✅ Verify MLflow is accessible: http://localhost:5000
- ⏳ Wait for paper data to generate actual recommendations
- ⏳ Then MLflow will start populating with runs automatically!