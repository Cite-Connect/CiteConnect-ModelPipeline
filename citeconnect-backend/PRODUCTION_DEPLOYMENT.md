# Bias Mitigation - Production Deployment Guide

## 🚀 How It Works in Production

### Overview

The bias mitigation system is **fully automatic** and requires **zero manual intervention** during normal operation. It's designed to be:
- ✅ **Zero-config** - Works automatically if configs exist
- ✅ **Graceful degradation** - Continues working if configs are missing
- ✅ **Performance-optimized** - Configs cached in memory
- ✅ **Hot-reloadable** - Fairness config auto-reloads on file changes

---

## 📋 Production Flow

### 1. Application Startup (`app/main.py`)

```python
# During FastAPI lifespan startup:
app.state.recommendation_service = RecommendationService(db)
```

**What happens:**
1. ✅ `RecommendationService.__init__()` is called
2. ✅ `_load_bias_config()` reads `bias_config/bias_mitigation_config.json`
3. ✅ Config is loaded into memory (`self.bias_config`)
4. ✅ Logs: `"Loaded bias mitigation config"` or `"No bias mitigation config found"`
5. ✅ Service is ready to use

**Performance:**
- Config loaded **once** at startup
- Stored in memory (no file I/O per request)
- Startup time: ~10-50ms (JSON parsing)

**If config missing:**
- Returns empty dict `{}`
- System continues without user-profile mitigation
- No errors, just logs a warning

---

### 2. Fairness Service (Lazy Loading)

The fairness service uses **lazy loading with smart caching**:

```python
# First time fairness_aware_rerank() is called:
_load_paper_field_map()  # Loads parquet file (one-time, ~1-2 seconds)
load_fairness_config()    # Loads JSON with mtime check
```

**What happens:**
1. **First request** that uses fairness:
   - Loads `data/combined_gcs_data.parquet` (20K papers, ~4MB)
   - Creates `paper_id → primary_field` mapping
   - Caches in memory (`_paper_field_map`)
   - Loads `fairness_config.json`
   - Caches with modification time check

2. **Subsequent requests:**
   - Uses cached mapping (instant)
   - Checks config file mtime (fast stat call)
   - Reloads config only if file changed

**Performance:**
- First load: ~1-2 seconds (parquet read)
- Subsequent: ~0ms (memory cache)
- Config reload: ~1ms (if file changed)

---

### 3. Recommendation Request Flow

When a user requests recommendations:

```
API Request → RecommendationOrchestrator → RecommendationService
```

**Step-by-step:**

1. **User-Profile Mitigation (During Scoring)**
   ```python
   # In _get_mitigation_policy_for_profile()
   - Checks user's primary_domain, research_stage, reading_level
   - Matches against bias_config["cold_start"]["minilm"]
   - If match found:
     * Applies boost_factor (e.g., 1.25x)
     * Sets min_score_floor
   - Returns mitigation_policy dict
   ```

2. **Multi-Factor Scoring**
   ```python
   # In _apply_multi_factor_scoring()
   - Uses mitigation_policy to boost scores
   - Filters papers below min_score_floor
   - Applies weight multipliers if specified
   ```

3. **Paper-Field Fairness (After Enrichment)**
   ```python
   # In _apply_fairness_reranking()
   - Calls fairness_aware_rerank()
   - Checks each paper's primary_field
   - If field in under_served_fields:
     * Applies 1.05x boost
   - Reranks by boosted scores
   ```

4. **Response**
   ```json
   {
     "papers": [...],
     "mitigation_policy": {
       "factor": 1.25,
       "applied_rules": [...],
       "min_score_threshold": 0.252
     }
   }
   ```

**Performance Impact:**
- User-profile check: ~0.1ms (dict lookup)
- Fairness reranking: ~1-5ms (for 10-20 papers)
- **Total overhead: < 10ms per request**

---

## 🔄 Config Management

### User-Profile Config (`bias_mitigation_config.json`)

**Loading:**
- ✅ Loaded **once** at service startup
- ✅ Cached in `self.bias_config`
- ❌ **Does NOT auto-reload** (requires service restart)

**To Update:**
1. Edit `bias_config/bias_mitigation_config.json`
2. Restart service (or redeploy)
3. New config loaded on next startup

**Best Practice:**
- Update configs during scheduled maintenance
- Or use rolling deployment (new instances get new config)

---

### Paper-Field Config (`fairness_config.json`)

**Loading:**
- ✅ Lazy-loaded on first use
- ✅ **Auto-reloads** if file modification time changes
- ✅ Checks mtime on every request (fast stat call)

**To Update:**
1. Edit `fairness_config.json`
2. **No restart needed!** - Auto-reloads on next request
3. System picks up changes automatically

**Best Practice:**
- Can update anytime
- Changes take effect on next recommendation request
- Monitor logs for reload messages

---

### Paper Metadata (`combined_gcs_data.parquet`)

**Loading:**
- ✅ Loaded **once** on first fairness request
- ✅ Cached in `_paper_field_map` (global variable)
- ❌ **Does NOT auto-reload** (requires service restart)

**To Update:**
1. Regenerate parquet file:
   ```bash
   python scripts/export_paper_metadata_to_parquet.py
   ```
2. Restart service to reload mapping

**Best Practice:**
- Update monthly/quarterly as new papers are added
- Or during scheduled maintenance

---

## 📊 Performance Characteristics

### Memory Usage

| Component | Memory | Notes |
|-----------|--------|-------|
| Bias config | ~1 KB | Small JSON |
| Fairness config | ~0.5 KB | Small JSON |
| Paper field map | ~2-5 MB | 20K papers × ~100 bytes |
| **Total** | **~5 MB** | Negligible |

### Latency Impact

| Operation | Time | Frequency |
|-----------|------|-----------|
| User-profile check | 0.1ms | Every request |
| Fairness reranking | 1-5ms | Every request |
| Config reload (fairness) | 1ms | Only if file changed |
| **Total overhead** | **< 10ms** | Per recommendation |

### Throughput

- ✅ **No bottleneck** - All operations are in-memory
- ✅ **Scalable** - Each worker has its own cache
- ✅ **No external dependencies** - No API calls or DB queries

---

## 🏗️ Deployment Architecture

### File Structure in Production

```
/app (container root)
├── bias_config/
│   └── bias_mitigation_config.json  # User-profile mitigation
├── fairness_config.json              # Paper-field fairness
├── data/
│   └── combined_gcs_data.parquet     # Paper metadata (optional)
└── app/
    └── services/
        ├── recommendation_service.py
        └── fairness_service.py
```

### Docker/Container Deployment

**Include in Dockerfile:**
```dockerfile
# Copy configs
COPY bias_config/ /app/bias_config/
COPY fairness_config.json /app/fairness_config.json

# Copy metadata (optional)
COPY data/combined_gcs_data.parquet /app/data/combined_gcs_data.parquet
```

**Or mount as volumes:**
```yaml
# docker-compose.yml
volumes:
  - ./bias_config:/app/bias_config:ro
  - ./fairness_config.json:/app/fairness_config.json:ro
  - ./data:/app/data:ro
```

### Kubernetes Deployment

**ConfigMap for configs:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: bias-configs
data:
  bias_mitigation_config.json: |
    {
      "cold_start": {
        "minilm": { ... }
      }
    }
  fairness_config.json: |
    {
      "under_served_fields": []
    }
```

**PersistentVolume for parquet:**
```yaml
# For large parquet file
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: paper-metadata
spec:
  accessModes: [ReadOnlyMany]
  resources:
    requests:
      storage: 100Mi
```

---

## 🔍 Monitoring & Observability

### Logs to Monitor

**Startup:**
```
[INFO] Loaded bias mitigation config
[INFO] RecommendationService initialized
```

**Request-level (if enabled):**
```
[DEBUG] Applied bias mitigation field=primary_domain value=fintech boost_factor=1.25
[DEBUG] Applied fairness reranking papers_reranked=10
```

**Config reload:**
```
[INFO] Reloaded fairness config (file modified)
```

### Metrics to Track

**Recommended metrics:**
1. **Mitigation application rate:**
   - % of requests with `mitigation_policy.factor > 1.0`
   - Track by slice (fintech, masters, intermediate)

2. **Fairness boost rate:**
   - % of papers boosted by field-based fairness
   - Track by field

3. **Performance:**
   - P50/P95/P99 latency for recommendations
   - Compare with/without mitigation

4. **Effectiveness:**
   - Combined score improvement for underperforming slices
   - Track over time

### Health Checks

**Add to health endpoint:**
```python
@app.get("/health")
async def health_check():
    return {
        "bias_mitigation": {
            "user_profile": bool(service.bias_config),
            "paper_field": fairness_config_exists(),
            "paper_metadata": parquet_file_exists()
        }
    }
```

---

## 🔄 Updating Configs in Production

### Option 1: Rolling Deployment (Recommended)

1. **Update config files** in version control
2. **Deploy new version** with updated configs
3. **New instances** load new configs on startup
4. **Old instances** continue with old configs until replaced
5. **Zero downtime**

### Option 2: Hot Reload (Fairness Config Only)

1. **Update `fairness_config.json`** on filesystem
2. **No restart needed** - auto-reloads on next request
3. **Useful for** quick adjustments

### Option 3: Config Service (Advanced)

1. **Store configs in database** or config service
2. **Poll for updates** periodically
3. **Reload when changed**
4. **More complex** but more flexible

---

## 🛡️ Error Handling & Resilience

### Missing Configs

**User-profile config missing:**
- ✅ Service starts normally
- ✅ Logs: `"No bias mitigation config found"`
- ✅ Recommendations work without mitigation
- ✅ No errors thrown

**Fairness config missing:**
- ✅ Returns empty config: `{"under_served_fields": []}`
- ✅ No field-based boosting applied
- ✅ Recommendations work normally

**Parquet file missing:**
- ✅ Returns empty mapping: `{}`
- ✅ All papers mapped to "Unknown"
- ✅ No field-based boosting applied
- ✅ Recommendations work normally

### Invalid Configs

**Malformed JSON:**
- ✅ Catches exception
- ✅ Logs warning
- ✅ Falls back to empty config
- ✅ Service continues

**Missing required fields:**
- ✅ Checks for required keys
- ✅ Skips invalid sections
- ✅ Uses valid sections only

---

## 📈 Scaling Considerations

### Horizontal Scaling

**Each worker instance:**
- ✅ Has its own config cache
- ✅ Loads configs independently
- ✅ No shared state needed
- ✅ Scales linearly

**Best Practice:**
- Use same config files across all instances
- Deploy configs with application code
- Or use shared volume/ConfigMap

### Vertical Scaling

**Memory:**
- ✅ ~5 MB per instance (negligible)
- ✅ No memory concerns

**CPU:**
- ✅ < 10ms overhead per request
- ✅ No CPU concerns

---

## ✅ Production Checklist

### Pre-Deployment

- [ ] Config files exist and are valid JSON
- [ ] Parquet file exists (if using paper-field fairness)
- [ ] Configs included in Docker image or mounted volumes
- [ ] Health check includes bias mitigation status
- [ ] Logging configured for mitigation events

### Deployment

- [ ] Configs deployed with application
- [ ] Service starts without errors
- [ ] Logs show configs loaded successfully
- [ ] Health check passes

### Post-Deployment

- [ ] Monitor logs for config loading
- [ ] Verify mitigation is being applied
- [ ] Track metrics (mitigation rate, performance)
- [ ] Test with users in underperforming slices

### Ongoing

- [ ] Regular bias analysis (weekly/monthly)
- [ ] Update configs based on analysis
- [ ] Monitor effectiveness metrics
- [ ] Update parquet file as papers are added

---

## 🎯 Summary

**The bias mitigation system in production:**

1. ✅ **Automatic** - No manual intervention needed
2. ✅ **Fast** - < 10ms overhead per request
3. ✅ **Resilient** - Gracefully handles missing configs
4. ✅ **Scalable** - Works with any number of workers
5. ✅ **Hot-reloadable** - Fairness config updates without restart
6. ✅ **Observable** - Logs and metrics available

**Just deploy and it works!** 🚀
