from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
import pandas as pd
from io import BytesIO
from minio import Minio
import json

# Configuration MinIO
MINIO_ENDPOINT = "minio:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
BUCKET_NAME = "iris-data"

# Configuration API
BASE_URL = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/georef-france-iris/records"
REGIONS = ["Bretagne"]
LIMIT = 100


def create_minio_client():
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )


def ensure_bucket_exists(**context):
    client = create_minio_client()
    if not client.bucket_exists(BUCKET_NAME):
        client.make_bucket(BUCKET_NAME)
    return True


def fetch_iris_data(**context):
    all_data = []

    for region in REGIONS:
        offset = 0
        total_records = None

        while total_records is None or offset < total_records:
            # Construction de l'URL avec les paramètres
            params = {
                'where': f'reg_name = "{region}"',
                'limit': LIMIT,
                'offset': offset
            }

            # Requête à l'API
            response = requests.get(BASE_URL, params=params)

            if response.status_code != 200:
                raise Exception(f"Erreur lors de la récupération des données IRIS: {response.status_code}")

            data = response.json()

            # Mise à jour du nombre total d'enregistrements si pas encore fait
            if total_records is None:
                total_records = data['total_count']

            # Ajout des résultats à notre liste
            all_data.extend(data['results'])

            # Mise à jour de l'offset pour la prochaine requête
            offset += LIMIT

            print(f"Récupérés {len(all_data)}/{total_records} enregistrements pour {region}")

    # Conversion en DataFrame
    df = pd.json_normalize(all_data)

    # Sauvegarde en différents formats
    current_date = datetime.now().strftime("%Y%m%d")
    client = create_minio_client()

    # Sauvegarde en CSV
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    csv_object_name = f"iris_data_{current_date}.csv"
    client.put_object(
        bucket_name=BUCKET_NAME,
        object_name=csv_object_name,
        data=csv_buffer,
        length=len(csv_buffer.getvalue()),
        content_type='application/csv'
    )

    # Sauvegarde en JSON
    json_buffer = BytesIO()
    df.to_json(json_buffer, orient='records')
    json_buffer.seek(0)

    json_object_name = f"iris_data_{current_date}.json"
    client.put_object(
        bucket_name=BUCKET_NAME,
        object_name=json_object_name,
        data=json_buffer,
        length=len(json_buffer.getvalue()),
        content_type='application/json'
    )

    # Création d'un fichier de métadonnées
    metadata = {
        'date_extraction': datetime.now().isoformat(),
        'regions': REGIONS,
        'nombre_enregistrements': len(all_data),
        'colonnes': df.columns.tolist(),
        'source': BASE_URL
    }

    metadata_buffer = BytesIO(json.dumps(metadata, indent=2).encode())
    metadata_object_name = f"iris_data_{current_date}_metadata.json"

    client.put_object(
        bucket_name=BUCKET_NAME,
        object_name=metadata_object_name,
        data=metadata_buffer,
        length=len(metadata_buffer.getvalue()),
        content_type='application/json'
    )

    return {
        'csv_file': csv_object_name,
        'json_file': json_object_name,
        'metadata_file': metadata_object_name,
        'record_count': len(all_data)
    }


# Configuration du DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'iris_data_collection',
    default_args=default_args,
    description='Collecte des données IRIS pour la Bretagne',
    schedule_interval='@daily',  # Exécution quotidienne
    start_date=datetime(2024, 2, 11),
    catchup=False,
    tags=['iris', 'bretagne', 'geodata'],
)

create_bucket = PythonOperator(
    task_id='ensure_bucket_exists',
    python_callable=ensure_bucket_exists,
    dag=dag,
)

collect_data = PythonOperator(
    task_id='fetch_iris_data',
    python_callable=fetch_iris_data,
    dag=dag,
)

create_bucket >> collect_data