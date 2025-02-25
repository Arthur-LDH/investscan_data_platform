from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
import pandas as pd
from io import BytesIO
from minio import Minio
import json
import logging
import os
import gzip
import tempfile

# Configuration MinIO
MINIO_ENDPOINT = "minio:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
BUCKET_NAME = "dvf-data"

# Configuration pour DVF - Source de données correcte
DVF_URL = "https://files.data.gouv.fr/geo-dvf/latest/csv"
YEARS = ["2019", "2020", "2021", "2022", "2023", "2024"]
DEPARTMENTS = ["22", "29", "35", "56"]


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
    logging.info(f"Bucket {BUCKET_NAME} vérifié/créé avec succès")
    return True


def fetch_dvf_data(**context):
    client = create_minio_client()
    metadata = {
        'date_extraction': datetime.now().isoformat(),
        'departements': DEPARTMENTS,
        'annees': YEARS,
        'fichiers': []
    }

    # Structure de l'URL confirmée: https://files.data.gouv.fr/geo-dvf/latest/csv/2019/departements/56.csv.gz
    failed_downloads = []
    successful_downloads = []

    # Créer un répertoire temporaire pour stocker les fichiers décompressés
    temp_dir = tempfile.mkdtemp()

    for year in YEARS:
        for dept in DEPARTMENTS:
            # URL correcte pour les fichiers DVF
            file_url = f"{DVF_URL}/{year}/departements/{dept}.csv.gz"

            try:
                logging.info(
                    f"Téléchargement des données DVF pour le département {dept}, année {year} - URL: {file_url}")

                # Tentative de téléchargement
                response = requests.get(file_url, stream=True)

                if response.status_code != 200:
                    error_msg = f"Erreur lors du téléchargement: {response.status_code} - {file_url}"
                    logging.error(error_msg)
                    failed_downloads.append({
                        'url': file_url,
                        'status_code': response.status_code,
                        'departement': dept,
                        'annee': year
                    })
                    continue

                # Décompression du fichier
                logging.info(f"Décompression du fichier pour le département {dept}, année {year}")

                try:
                    # Décompresser les données en mémoire
                    compressed_data = BytesIO(response.content)
                    decompressed_data = gzip.GzipFile(fileobj=compressed_data, mode='rb')
                    csv_content = decompressed_data.read()

                    # Nom du fichier décompressé à stocker dans MinIO
                    object_name = f"dvf_{dept}_{year}.csv"

                    # Upload du fichier décompressé dans MinIO
                    client.put_object(
                        bucket_name=BUCKET_NAME,
                        object_name=object_name,
                        data=BytesIO(csv_content),
                        length=len(csv_content),
                        content_type='text/csv'
                    )

                    metadata['fichiers'].append({
                        'departement': dept,
                        'annee': year,
                        'fichier': object_name,
                        'taille': len(csv_content),
                        'url_source': file_url,
                        'decompressed': True
                    })

                    successful_downloads.append({
                        'departement': dept,
                        'annee': year,
                        'fichier': object_name,
                        'taille': len(csv_content),
                        'format': 'csv'
                    })

                    logging.info(f"Fichier décompressé {object_name} enregistré avec succès")

                except Exception as decomp_error:
                    # En cas d'erreur de décompression, on sauvegarde quand même le fichier compressé
                    error_msg = f"Erreur lors de la décompression: {str(decomp_error)}. Sauvegarde du fichier compressé."
                    logging.warning(error_msg)

                    # Nom du fichier compressé à stocker dans MinIO
                    object_name = f"dvf_{dept}_{year}.csv.gz"

                    # Upload du fichier compressé dans MinIO
                    client.put_object(
                        bucket_name=BUCKET_NAME,
                        object_name=object_name,
                        data=BytesIO(response.content),
                        length=len(response.content),
                        content_type='application/gzip'
                    )

                    metadata['fichiers'].append({
                        'departement': dept,
                        'annee': year,
                        'fichier': object_name,
                        'taille': len(response.content),
                        'url_source': file_url,
                        'decompressed': False,
                        'decompression_error': str(decomp_error)
                    })

                    successful_downloads.append({
                        'departement': dept,
                        'annee': year,
                        'fichier': object_name,
                        'taille': len(response.content),
                        'format': 'csv.gz'
                    })

                    logging.info(f"Fichier compressé {object_name} enregistré par défaut")

            except Exception as e:
                error_msg = f"Erreur lors du traitement du fichier {file_url}: {str(e)}"
                logging.error(error_msg)
                failed_downloads.append({
                    'url': file_url,
                    'error': str(e),
                    'departement': dept,
                    'annee': year
                })

    # Nettoyage du répertoire temporaire
    try:
        import shutil
        shutil.rmtree(temp_dir)
    except Exception as e:
        logging.warning(f"Erreur lors du nettoyage du répertoire temporaire: {str(e)}")

    # Enregistrer des informations sur les échecs éventuels
    if failed_downloads:
        failures_json = json.dumps(failed_downloads, indent=2)
        client.put_object(
            bucket_name=BUCKET_NAME,
            object_name="failed_downloads.json",
            data=BytesIO(failures_json.encode()),
            length=len(failures_json),
            content_type='application/json'
        )
        logging.warning(
            f"{len(failed_downloads)} téléchargements ont échoué. Voir failed_downloads.json pour les détails.")

    # Enregistrer des informations sur les téléchargements réussis
    if successful_downloads:
        success_json = json.dumps(successful_downloads, indent=2)
        client.put_object(
            bucket_name=BUCKET_NAME,
            object_name="successful_downloads.json",
            data=BytesIO(success_json.encode()),
            length=len(success_json),
            content_type='application/json'
        )
        logging.info(f"{len(successful_downloads)} téléchargements réussis.")

    # Création d'un fichier de métadonnées
    current_date = datetime.now().strftime("%Y%m%d")
    metadata_buffer = BytesIO(json.dumps(metadata, indent=2).encode())
    metadata_object_name = f"dvf_metadata_{current_date}.json"

    client.put_object(
        bucket_name=BUCKET_NAME,
        object_name=metadata_object_name,
        data=metadata_buffer,
        length=len(metadata_buffer.getvalue()),
        content_type='application/json'
    )

    return {
        'metadata_file': metadata_object_name,
        'files_count': len(metadata['fichiers']),
        'success_count': len(successful_downloads),
        'failure_count': len(failed_downloads)
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
    'dvf_data_collection',
    default_args=default_args,
    description='Collecte des données DVF pour les départements bretons (fichiers décompressés)',
    schedule_interval='@monthly',
    start_date=datetime(2024, 2, 11),
    catchup=False,
    tags=['dvf', 'bretagne', 'immobilier'],
)

create_bucket = PythonOperator(
    task_id='ensure_bucket_exists',
    python_callable=ensure_bucket_exists,
    dag=dag,
)

collect_data = PythonOperator(
    task_id='fetch_dvf_data',
    python_callable=fetch_dvf_data,
    dag=dag,
)

create_bucket >> collect_data