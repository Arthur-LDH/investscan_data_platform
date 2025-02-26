# tests/conftest.py
import os
import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark_session():
    """Fixture qui crée une SparkSession pour l'ensemble des tests"""
    session = (SparkSession.builder
               .master("local[*]")
               .appName("pytest-pyspark")
               .config("spark.sql.shuffle.partitions", "1")
               .config("spark.default.parallelism", "1")
               .getOrCreate())

    yield session

    session.stop()