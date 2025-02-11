"""
Configuration et fixtures pour les tests Pytest.
"""
import pytest
from airflow.models import DagBag
import os

@pytest.fixture(scope="session")
def dagbag():
    """Fixture pour charger les DAGs Airflow."""
    dagbag = DagBag(
        dag_folder=os.path.join(os.path.dirname(__file__), "../dags"),
        include_examples=False
    )
    return dagbag

@pytest.fixture(scope="session")
def iris_dag(dagbag):
    """Fixture pour accéder au DAG IRIS."""
    return dagbag.get_dag("iris_data_collection")