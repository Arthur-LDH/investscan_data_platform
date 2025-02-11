"""
Configuration du package dags.
"""
from typing import Dict, Any

# Configuration globale pour tous les DAGs
DEFAULT_ARGS: Dict[str, Any] = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

# Configuration commune MinIO
MINIO_CONFIG = {
    'endpoint': "minio:9000",
    'access_key': "minioadmin",
    'secret_key': "minioadmin",
    'secure': False
}

# Configuration des APIs
API_CONFIG = {
    'iris': {
        'base_url': "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/georef-france-iris/records",
        'regions': ["Bretagne"],
        'limit': 100
    }
}