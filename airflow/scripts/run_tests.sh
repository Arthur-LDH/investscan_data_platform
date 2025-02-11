#!/bin/bash

echo "Executing tests..."
python -m pytest /opt/airflow/tests -v --cov=/opt/airflow/dags --cov-report=term-missing