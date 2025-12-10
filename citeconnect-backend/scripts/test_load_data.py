#!/usr/bin/env python3
"""
Standalone test script for load_data_from_pipeline task.

This allows testing the DAG task without Airflow.

Usage:
    python dags/test_load_data.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the task function from the DAG
from dags.model_pipeline_dag import load_data_from_pipeline

if __name__ == "__main__":
    print("=" * 80)
    print("Testing load_data_from_pipeline task")
    print("=" * 80)
    print()
    
    try:
        # Call the task function directly (without Airflow context)
        result = load_data_from_pipeline()
        
        print("\n✅ Task completed successfully!")
        print(f"Status: {result['status']}")
        print(f"Report saved to: citeconnect-backend/data_quality_report.json")
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Task failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

