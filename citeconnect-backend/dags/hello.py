from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def say_hello():
    print("="*50)
    print("HELLO WORLD!")
    print("="*50)
    return "success"

with DAG(
    'hello_dag',
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False
) as dag:
    
    PythonOperator(
        task_id='hello',
        python_callable=say_hello
    )