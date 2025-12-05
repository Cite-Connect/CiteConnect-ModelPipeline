# Database Changes Summary: `test_recommendations` Task

## 🔍 What Happens When Task Runs

### Step 1: Find Cold-Start User
- **Action**: SELECT query (read-only)
- **Tables**: `users`, `user_interactions`
- **Impact**: ✅ No changes

### Step 2: Generate Recommendations
This is where the complexity happens:

#### 2a. Load EmbeddingService (First Time Only)
- **Action**: `EmbeddingService()` singleton initialization
- **What it does**: 
  - Loads **BOTH** MiniLM and SPECTER models into memory
  - This happens ONCE when the service is first created
  - Subsequent calls reuse the same instance (cached)
- **Time**: ~20-30 seconds first time (loading models), <1 second after
- **Impact**: ⚠️ **Model loading is slow but only happens once per Airflow worker**

#### 2b. Get User Embeddings
- **Action**: `get_or_generate_user_embeddings(user_id)`
- **What it checks**:
  1. Does `user_embeddings_minilm` exist? ✅ (from test: YES)
  2. Does `user_embeddings_specter` exist? ✅ (from test: YES)
  3. Should we regenerate? (checks if `interaction_count` changed significantly)
- **Decision**: 
  - If embeddings exist AND are up-to-date → **Just return them** ✅
  - If missing OR outdated → **Regenerate both** ⚠️

#### 2c. Generate Recommendations
- **Action**: Uses existing MiniLM embedding to find similar papers
- **Tables Read**: `papers`, `paper_embeddings_minilm`, `ground_truth_papers`
- **Impact**: ✅ Read-only

### Step 3: Evaluate Recommendations
- **Action**: Calculate scores (profile_alignment, ground_truth_quality, combined_score)
- **Tables Read**: `user_interests`, `ground_truth_papers`, `ground_truth_relationships`
- **Tables Written**: ❌ **NONE** (because `store_result=False`)
- **Impact**: ✅ Read-only

### Step 4: Print Scores
- **Action**: Just logging
- **Impact**: ✅ No database changes

---

## 📊 What Gets Modified in Database?

### ✅ **SAFE - Expected Changes** (Only if embeddings are missing/outdated):

#### 1. `user_embeddings_minilm` table
- **When**: Only if user has NO MiniLM embedding OR interaction_count increased by ≥10
- **Action**: 
  - `INSERT` if doesn't exist
  - `UPDATE` if outdated
- **Columns Updated**:
  - `embedding` (vector)
  - `generation_method` (e.g., 'profile_based')
  - `interaction_count` (current count)
  - `last_updated` (timestamp)
- **Impact**: ✅ **Harmless** - This is normal for cold-start users

#### 2. `user_embeddings_specter` table
- **When**: Only if user has NO SPECTER embedding OR interaction_count increased by ≥10
- **Action**: Same as above
- **Impact**: ✅ **Harmless** - Even though we only use MiniLM, both are generated

#### 3. `user_recommendation_state` table
- **When**: Only if embeddings were regenerated
- **Action**: `UPDATE` timestamps
- **Columns Updated**:
  - `last_embedding_update_minilm` (timestamp)
  - `last_embedding_update_specter` (timestamp)
  - `updated_at` (timestamp)
  - Potentially `recommendation_stage` (if user transitions stages)
- **Impact**: ✅ **Harmless** - Just metadata tracking

### ❌ **NO CHANGES** (Because `store_result=False`):

#### 4. `cold_start_evaluations` table
- **Action**: ❌ **NO INSERT**
- **Reason**: `store_result=False` in evaluation call
- **Impact**: ✅ **Safe** - Evaluation results are NOT stored

---

## ⚠️ Performance Concerns

### Issue 1: SPECTER Model Loading
**Problem**: EmbeddingService loads BOTH models even though we only need MiniLM

**Why it happens**:
- `EmbeddingService` is a singleton that loads both models on first initialization
- `get_or_generate_user_embeddings()` needs to return both embeddings (even if we only use one)

**Impact**:
- First run: ~20-30 seconds to load SPECTER model
- Subsequent runs: <1 second (models cached in memory)

**Solution Options**:
1. ✅ **Accept it** - Models are cached, so it's only slow the first time
2. ⚠️ **Modify EmbeddingService** - Make model loading lazy (load only when needed)
3. ⚠️ **Modify UserEmbeddingService** - Add method to get only one embedding

### Issue 2: Unnecessary Embedding Generation
**Problem**: If embeddings already exist and are up-to-date, we shouldn't regenerate

**Current Behavior**:
- ✅ `get_or_generate_user_embeddings()` already checks if embeddings exist
- ✅ It only regenerates if missing OR outdated (interaction_count changed by ≥10)
- ✅ From test: User already has embeddings from 2025-11-29, so **NO regeneration should happen**

**Expected**: If user has embeddings and interaction_count hasn't changed, it should just return existing ones ✅

---

## 🎯 Summary

### What Gets Modified?
**For a user with existing embeddings (like user_id=2)**:
- ✅ **NO database writes** - Embeddings already exist, so they're just read
- ✅ **NO evaluation writes** - `store_result=False`
- ✅ **Read-only operations only**

**For a user WITHOUT embeddings**:
- ⚠️ **2 INSERTs**: `user_embeddings_minilm`, `user_embeddings_specter`
- ⚠️ **1 UPDATE**: `user_recommendation_state` (timestamps)
- ✅ **Still safe** - This is expected behavior for cold-start users

### Performance
- **First run**: ~20-30 seconds (loading SPECTER model)
- **Subsequent runs**: <5 seconds (models cached)
- **If embeddings exist**: No regeneration, just reads ✅

### Safety
- ✅ **No harmful changes** - Only generates missing embeddings
- ✅ **No evaluation writes** - `store_result=False`
- ✅ **No user data deleted** - Only INSERT/UPDATE, never DELETE
- ✅ **Idempotent** - Running multiple times has same result

---

## 💡 Recommendations

1. **For Testing**: Current setup is fine - embeddings already exist, so no writes
2. **For Production**: Consider lazy-loading models or adding method to get only one embedding
3. **To Make Fully Read-Only**: Pre-generate embeddings for all test users before running DAG



