# services/airflow/dags/test_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 2, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def test_function():
    print("Test DAG exécuté avec succès!")
    return "Success"

with DAG(
    'test_dag',
    default_args=default_args,
    description='Un DAG de test simple',
    schedule_interval=timedelta(days=1),
    catchup=False
) as dag:

    test_task = PythonOperator(
        task_id='test_task',
        python_callable=test_function,
    )