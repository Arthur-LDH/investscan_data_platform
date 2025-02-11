"""
DAG pour la collecte des données IRIS de l'OpenData et leur stockage en Parquet.
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
from io import BytesIO
import json
import logging
from apache_airflow.dags.utils.data_utils import prepare_parquet_data, create_metadata
from apache_airflow.dags.utils.minio_utils import get_minio_client, ensure_bucket_exists

logger = logging.getLogger(__name__)

# Configuration
DEFAULT_ARGS = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

MINIO_CONFIG = {
    'endpoint': "minio:9000",
    'access_key': "minioadmin",
    'secret_key': "minioadmin",
    'bucket_name': "iris-data"
}

API_CONFIG = {
    'base_url': "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/georef-france-iris/records",
    'regions': ["Bretagne"],
    'limit': 100
}

def fetch_api_data(region: str, offset: int) -> dict:
    """Récupère une page de données de l'API pour une région donnée."""
    params = {
        'where': f'reg_name = "{region}"',
        'limit': API_CONFIG['limit'],
        'offset': offset
    }

    response = requests.get(API_CONFIG['base_url'], params=params)
    if response.status_code != 200:
        raise Exception(f"Erreur API: {response.status_code}")

    return response.json()

def fetch_all_data() -> list:
    """Collecte toutes les données IRIS."""
    all_data = []

    for region in API_CONFIG['regions']:
        offset = 0
        total_records = None

        while total_records is None or offset < total_records:
            data = fetch_api_data(region, offset)

            if total_records is None:
                total_records = data['total_count']
                logger.info(f"Total à récupérer pour {region}: {total_records}")

            all_data.extend(data['results'])
            offset += API_CONFIG['limit']
            logger.info(f"Progression {region}: {len(all_data)}/{total_records}")

    return all_data

def setup_bucket(**context):
    """Crée le bucket si nécessaire."""
    return ensure_bucket_exists(MINIO_CONFIG['bucket_name'])

def process_data(**context):
    """Traite et stocke les données IRIS."""
    try:
        # Collecte des données
        raw_data = fetch_all_data()
        logger.info(f"Données collectées: {len(raw_data)} enregistrements")

        # Préparation du Parquet
        table, parquet_buffer = prepare_parquet_data(raw_data)
        buffer_size = len(parquet_buffer.getvalue())
        logger.info(f"Données converties en Parquet: {buffer_size} bytes")

        # Noms des fichiers
        current_date = datetime.now().strftime("%Y%m%d")
        parquet_name = f"iris_data_{current_date}.parquet"
        metadata_name = f"iris_data_{current_date}_metadata.json"

        # Création du client MinIO
        client = get_minio_client()

        # Upload du fichier Parquet
        logger.info(f"Upload du fichier {parquet_name}")
        client.put_object(
            bucket_name=MINIO_CONFIG['bucket_name'],
            object_name=parquet_name,
            data=parquet_buffer,
            length=buffer_size,
            content_type='application/octet-stream'
        )

        # Création et upload des métadonnées
        metadata = create_metadata(
            table=table,
            data_size=buffer_size,
            source_url=API_CONFIG['base_url'],
            additional_info={
                'regions': API_CONFIG['regions'],
                'date_extraction': datetime.now().isoformat()
            }
        )

        metadata_buffer = BytesIO(json.dumps(metadata, indent=2).encode())
        logger.info(f"Upload des métadonnées {metadata_name}")
        client.put_object(
            bucket_name=MINIO_CONFIG['bucket_name'],
            object_name=metadata_name,
            data=metadata_buffer,
            length=len(metadata_buffer.getvalue()),
            content_type='application/json'
        )

        return {
            'parquet_file': parquet_name,
            'metadata_file': metadata_name,
            'record_count': len(raw_data)
        }

    except Exception as e:
        logger.error(f"Erreur lors du traitement: {str(e)}")
        raise

# Configuration du DAG
dag = DAG(
    'iris_data_collection',
    default_args=DEFAULT_ARGS,
    description='Collecte des données IRIS en format Parquet',
    schedule_interval='@daily',
    start_date=datetime(2024, 2, 11),
    catchup=False,
    tags=['iris', 'bretagne', 'geodata', 'parquet'],
)

# Définition des tâches
create_bucket = PythonOperator(
    task_id='setup_bucket',
    python_callable=setup_bucket,
    dag=dag,
)

collect_data = PythonOperator(
    task_id='process_data',
    python_callable=process_data,
    dag=dag,
)

# Dépendances des tâches
create_bucket >> collect_data