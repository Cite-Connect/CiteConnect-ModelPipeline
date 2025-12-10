"""
Absolute minimal test - no custom imports
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def hello_world(**context):
    """Simplest possible task"""
    print("="*80)
    print("HELLO FROM AIRFLOW!")
    print("="*80)
    return "success"

with DAG(
    'test_minimal',
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=['test']
) as dag:
    
    task = PythonOperator(
        task_id='hello',
        python_callable=hello_world
    )