# CiteConnect Model Pipeline DAGs

This directory contains Apache Airflow DAGs for the CiteConnect model development pipeline.

## DAGs

### `model_pipeline_dag.py`
Main DAG for the model development pipeline.

**Current Tasks:**
1. `load_data_from_pipeline` - Loads data from Supabase and generates quality report

## Testing Without Airflow

You can test the DAG tasks directly without running Airflow:

```bash
# From citeconnect-backend directory
cd citeconnect-backend

# Test the load_data_from_pipeline task
python dags/test_load_data.py
```

This will:
- Connect to your Supabase database
- Load papers, embeddings, and user data
- Generate `data_quality_report.json` in the `citeconnect-backend/` directory
- Print a summary of the results

## Testing With Airflow

### 1. Set Airflow DAGs Folder

```bash
export AIRFLOW__CORE__DAGS_FOLDER=/Users/anusha/CiteConnect/CiteConnect-ModelPipeline/citeconnect-backend/dags
```

### 2. Test a Specific Task

```bash
# Test the task (dry run)
airflow tasks test citeconnect_model_pipeline load_data_from_pipeline 2025-01-01
```

### 3. Trigger the DAG

```bash
# Via CLI
airflow dags trigger citeconnect_model_pipeline

# Or via Airflow UI
# Navigate to DAGs → citeconnect_model_pipeline → Trigger DAG
```

## Output

The `load_data_from_pipeline` task generates:
- **File**: `citeconnect-backend/data_quality_report.json`
- **Content**: 
  - Data statistics (papers, embeddings, users)
  - Quality metrics
  - Versioning information
  - Validation errors (if any)

## Requirements

- Database connection configured in `.env`
- All required tables exist in Supabase:
  - `papers` (with `reference_ids`, `citation_ids` columns)
  - `paper_embeddings_specter`
  - `paper_embeddings_minilm`
  - `users`
  - `user_domains`
  - `user_interests`
  - `user_profile_embeddings`
  - `alembic_version`

## Next Steps

Future tasks to be added:
- `test_recommendation_strategies` - Compare SPECTER2, MiniLM, hybrid strategies
- `validate_models` - Calculate Precision@10, Recall@10, MRR
- `perform_data_slicing` - Slice by domain, user stage, interests
- `detect_bias_across_slices` - Bias detection per slice
- `push_to_model_registry` - Push validated models to GCP Artifact Registry



