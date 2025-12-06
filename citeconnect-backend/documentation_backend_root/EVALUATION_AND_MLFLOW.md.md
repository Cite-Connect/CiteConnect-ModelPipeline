# Evaluation & MLflow Tracking - Complete Flow Diagram

## 🎯 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    User Request Flow                         │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│          Recommendation Orchestrator                         │
│  1. Get user context                                         │
│  2. Generate recommendations                                 │
│  3. Evaluate quality ──────────┐                            │
│  4. Log to MLflow ─────────────┼───► EVALUATION HAPPENS     │
│  5. Return to user              │                            │
└─────────────────────────────────┼───────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│                  Evaluation Service                          │
│  • Calculate profile_alignment                               │
│  • Calculate ground_truth_quality                            │
│  • Calculate diversity_score                                 │
│  • Detect bias (if enough data)                             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  Experiment Service                          │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  MLflow Tracking                                       │ │
│  │  • Start run                                           │ │
│  │  • Log parameters (user_id, model, stage)             │ │
│  │  • Log metrics (alignment, quality, time)             │ │
│  │  • Set tags (domain, timestamp)                       │ │
│  │  • End run                                             │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Database Logging                                      │ │
│  │  • recommendation_events table                         │ │
│  │  • cold_start_evaluations table                        │ │
│  │  • interaction_evaluations table (later)               │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Evaluation Metrics by User Stage

### **Cold Start Users (0-5 interactions)**

```
┌──────────────────────────────────────┐
│  Evaluation Metrics (Logged to MLflow) │
├──────────────────────────────────────┤
│  1. profile_alignment (0-1)           │
│     ↳ How well papers match interests │
│                                        │
│  2. ground_truth_quality (0-1)        │
│     ↳ Papers in citation networks     │
│                                        │
│  3. combined_score (0-1)              │
│     ↳ 60% alignment + 40% GT          │
│                                        │
│  4. diversity_score (0-1)             │
│     ↳ Author, venue, temporal mix     │
│                                        │
│  5. generation_time_ms                │
│     ↳ Latency tracking                │
└──────────────────────────────────────┘
        │
        ▼
  Logged Immediately
  When Recommendations
  Are Generated
```

### **Mature Users (10+ interactions)**

```
┌──────────────────────────────────────┐
│  Additional Metrics (Logged After    │
│  User Interactions)                  │
├──────────────────────────────────────┤
│  1. precision_at_10                  │
│     ↳ % of top 10 user engaged with  │
│                                       │
│  2. recall_at_10                     │
│     ↳ % of relevant papers found     │
│                                       │
│  3. click_through_rate (CTR)         │
│     ↳ % recommendations clicked      │
│                                       │
│  4. save_rate                        │
│     ↳ % recommendations saved        │
│                                       │
│  5. engagement_rate                  │
│     ↳ Overall positive signal rate   │
└──────────────────────────────────────┘
        │
        ▼
  Logged Periodically
  After Interactions
  Accumulate
```

---

## 🔄 Complete Request-Response Flow with Evaluation

### **1. User Requests Recommendations**

```bash
POST /api/v1/recommendations
{
  "user_id": 1,
  "model_preference": "all-MiniLM-L6-v2",
  "count": 10
}
```

### **2. Backend Processing (What Happens)**

```python
# In RecommendationOrchestrator.generate_recommendations()

# Step 1: Get user context
user_context = await user_state_service.get_user_context(user_id)
# Returns: {stage: 'cold_start', profile: {...}, interactions: []}

# Step 2: Generate recommendations
recommendations = await self._execute_strategy(...)
# Returns: [paper1, paper2, ..., paper10]

# Step 3: EVALUATE (EvaluationService)
evaluation = await self._evaluate_recommendations(recommendations, user_context)
# Returns: {
#   profile_alignment: 0.75,
#   ground_truth_quality: 0.62,
#   combined_score: 0.69
# }

# Step 4: LOG TO MLFLOW (ExperimentService) ← HAPPENS HERE
if self.experiment_service:
    await self.experiment_service.log_recommendation_event(
        user_id=user_id,
        model_name=model_name,
        recommendations=recommendations,
        evaluation_scores=evaluation,
        user_context=user_context,
        generation_time_ms=generation_time_ms
    )
    # This creates an MLflow run with all metrics!

# Step 5: Return response
return {
    "recommendations": [...],
    "metadata": {
        "evaluation_scores": evaluation,  # Included in response
        "model_used": "all-MiniLM-L6-v2",
        ...
    }
}
```

### **3. User Response Includes Evaluation**

```json
{
  "recommendations": [
    {"paper_id": "...", "title": "...", "relevance_score": 0.89},
    ...
  ],
  "metadata": {
    "user_stage": "cold_start",
    "strategy_used": "profile_based_search",
    "model_used": "all-MiniLM-L6-v2",
    "evaluation_scores": {
      "profile_alignment": 0.75,
      "ground_truth_quality": 0.62,
      "combined_score": 0.69
    },
    "generation_time_ms": 145.23
  }
}
```

### **4. Simultaneously in MLflow**

Navigate to: http://localhost:5000

You see a new run with:
- **Run Name**: `rec_1_2024-11-28T15:30:45`
- **Parameters**: user_id=1, model=all-MiniLM-L6-v2, user_stage=cold_start
- **Metrics**: profile_alignment=0.75, ground_truth_quality=0.62, combined_score=0.69
- **Tags**: user_domain=healthcare, timestamp=2024-11-28T15:30:45

---

## 🎯 Summary: Where Evaluation Fits

### **In the Request Flow:**
```
User → API → Orchestrator → [EVALUATE] → Log MLflow → Return Response
                                  ↑
                          Happens here!
```

### **What Gets Evaluated:**
1. **Before serving** (Quality gate):
   - Profile alignment
   - Ground truth quality
   - Diversity

2. **After interactions** (Performance measurement):
   - Precision, Recall
   - CTR, Save rate
   - User engagement

### **Where It's Logged:**
1. **MLflow** - For visualization and comparison
2. **Database** - For persistence and queries
3. **Response** - User sees quality scores

### **When You Can Test:**
- ✅ **NOW**: Verify MLflow is running
- ⏳ **AFTER PAPER DATA**: See actual metrics being logged
- ⏳ **AFTER INTERACTIONS**: See mature user metrics
- ⏳ **BEFORE SUBMISSION**: Run model comparison

---

## 🎓 Model Development Guidelines Compliance

All requirements are implemented and will be automatically tracked:

| Requirement | Implementation | Status |
|-------------|----------------|--------|
| Model Validation | EvaluationService.evaluate_cold_start() | ✅ Done |
| Bias Detection | EvaluationService.detect_bias() | ✅ Done |
| Experiment Tracking | ExperimentService with MLflow | ✅ Done |
| Model Comparison | ExperimentService.compare_models() | ✅ Done |
| Automated Logging | Every recommendation auto-logs | ✅ Done |
| CI/CD Pipeline | GitHub Actions | 🟡 Pending |

The system is **ready** - it just needs paper data to start generating metrics!