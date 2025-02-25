from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
import pandas as pd
from io import BytesIO
from minio import Minio
import json
import zipfile
import geopandas as gpd
import os
import tempfile
import shutil
import logging
import subprocess

# Configuration MinIO
MINIO_ENDPOINT = "minio:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
BUCKET_NAME = "iris-contours"

# URL du fichier de contours IRIS (source: Géoplateforme France)
CONTOURS_URL = "https://data.geopf.fr/telechargement/download/CONTOURS-IRIS/CONTOURS-IRIS_3-0__GPKG_LAMB93_FXX_2024-01-01/CONTOURS-IRIS_3-0__GPKG_LAMB93_FXX_2024-01-01.7z"

# Départements bretons
DEPARTMENTS = ["22", "29", "35", "56"]

# Variable globale pour le dossier temporaire
TEMP_DIR = None


def create_minio_client():
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )


def ensure_bucket_exists(**context):
    """Vérifier/créer le bucket MinIO"""
    client = create_minio_client()
    if not client.bucket_exists(BUCKET_NAME):
        client.make_bucket(BUCKET_NAME)
    logging.info(f"Bucket {BUCKET_NAME} vérifié/créé avec succès")
    return True


def download_and_extract(**context):
    """Télécharger et extraire le fichier"""
    global TEMP_DIR
    client = create_minio_client()
    current_date = datetime.now().strftime("%Y%m%d")

    # Création du répertoire temporaire
    TEMP_DIR = tempfile.mkdtemp()
    extract_dir = os.path.join(TEMP_DIR, "extracted")
    os.makedirs(extract_dir, exist_ok=True)

    try:
        # Téléchargement
        logging.info(f"Téléchargement depuis {CONTOURS_URL}...")
        response = requests.get(CONTOURS_URL, stream=True)

        if response.status_code != 200:
            error_msg = f"Erreur de téléchargement: {response.status_code}"
            client.put_object(
                bucket_name=BUCKET_NAME,
                object_name=f"download_error_{current_date}.txt",
                data=BytesIO(error_msg.encode()),
                length=len(error_msg),
                content_type='text/plain'
            )
            raise Exception(error_msg)

        # Enregistrement du fichier
        zip_path = os.path.join(TEMP_DIR, "contours.7z")
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)

        file_size = os.path.getsize(zip_path)
        logging.info(f"Téléchargement OK: {file_size} octets")

        # Extraction avec 7z
        logging.info(f"Extraction vers {extract_dir}...")
        process = subprocess.run(
            ['7z', 'x', zip_path, f'-o{extract_dir}', '-y'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if process.returncode != 0:
            error_msg = f"Erreur d'extraction: {process.stderr}"
            client.put_object(
                bucket_name=BUCKET_NAME,
                object_name=f"extract_error_{current_date}.txt",
                data=BytesIO(error_msg.encode()),
                length=len(error_msg),
                content_type='text/plain'
            )
            raise Exception(error_msg)

        # Liste des fichiers extraits
        all_files = []
        for root, _, files in os.walk(extract_dir):
            for file in files:
                all_files.append(os.path.join(root, file))

        logging.info(f"Extraction OK: {len(all_files)} fichiers")

        # Sauvegarder la liste des fichiers pour debug
        file_list = "\n".join(all_files)
        client.put_object(
            bucket_name=BUCKET_NAME,
            object_name=f"files_list_{current_date}.txt",
            data=BytesIO(file_list.encode()),
            length=len(file_list),
            content_type='text/plain'
        )

        # Recherche de fichiers .gpkg ou .shp
        source_file = None
        for file_path in all_files:
            if file_path.lower().endswith('.gpkg'):
                source_file = file_path
                break
            elif file_path.lower().endswith('.shp'):
                source_file = file_path
                break

        if not source_file:
            error_msg = "Aucun fichier géographique (.gpkg ou .shp) trouvé"
            client.put_object(
                bucket_name=BUCKET_NAME,
                object_name=f"no_geo_file_{current_date}.txt",
                data=BytesIO(error_msg.encode()),
                length=len(error_msg),
                content_type='text/plain'
            )
            raise Exception(error_msg)

        logging.info(f"Fichier géographique trouvé: {source_file}")
        return source_file

    except Exception as e:
        error_msg = f"Erreur: {str(e)}"
        logging.error(error_msg)
        client.put_object(
            bucket_name=BUCKET_NAME,
            object_name=f"error_{current_date}.txt",
            data=BytesIO(error_msg.encode()),
            length=len(error_msg),
            content_type='text/plain'
        )
        raise e


def process_and_save(**context):
    """Traiter et sauvegarder les données"""
    global TEMP_DIR
    client = create_minio_client()
    current_date = datetime.now().strftime("%Y%m%d")

    source_file = context['ti'].xcom_pull(task_ids='download_and_extract')

    try:
        # Charger le fichier
        logging.info(f"Chargement de {source_file}...")
        iris_gdf = gpd.read_file(source_file)
        logging.info(f"Fichier chargé: {len(iris_gdf)} entités")

        # Filtrer pour la Bretagne
        columns = iris_gdf.columns.tolist()

        # Recherche de la colonne de département
        dept_column = None
        for col in ['INSEE_DEP', 'CODE_DEP', 'DEP', 'DEPARTEMENT', 'DEPCOM']:
            if col in columns:
                dept_column = col
                break

        if not dept_column:
            for col in columns:
                if 'DEP' in col.upper():
                    dept_column = col
                    break

        # Filtrage
        if dept_column:
            try:
                bretagne_iris = iris_gdf[iris_gdf[dept_column].isin(DEPARTMENTS)]
                if len(bretagne_iris) == 0:
                    bretagne_iris = iris_gdf[iris_gdf[dept_column].astype(str).str.startswith(tuple(DEPARTMENTS))]
                if len(bretagne_iris) == 0:
                    bretagne_iris = iris_gdf
            except:
                bretagne_iris = iris_gdf
        else:
            bretagne_iris = iris_gdf

        logging.info(f"Après filtrage: {len(bretagne_iris)} entités")

        # Sauvegarde en GeoJSON uniquement
        logging.info("Sauvegarde en GeoJSON...")
        buffer = BytesIO()
        bretagne_iris.to_file(buffer, driver='GeoJSON')
        buffer.seek(0)

        geojson_name = f"iris_contours_bretagne_{current_date}.geojson"
        client.put_object(
            bucket_name=BUCKET_NAME,
            object_name=geojson_name,
            data=buffer,
            length=len(buffer.getvalue()),
            content_type='application/geo+json'
        )

        # Métadonnées
        metadata = {
            'date_extraction': datetime.now().isoformat(),
            'source_url': CONTOURS_URL,
            'departements': DEPARTMENTS,
            'fichiers': [
                {
                    'format': 'geojson',
                    'region': 'Bretagne',
                    'fichier': geojson_name
                }
            ]
        }

        metadata_buffer = BytesIO(json.dumps(metadata, indent=2).encode())
        metadata_name = f"iris_contours_metadata_{current_date}.json"

        client.put_object(
            bucket_name=BUCKET_NAME,
            object_name=metadata_name,
            data=metadata_buffer,
            length=len(metadata_buffer.getvalue()),
            content_type='application/json'
        )

        logging.info("Sauvegarde terminée")
        return {'geojson': geojson_name, 'metadata': metadata_name}

    except Exception as e:
        error_msg = f"Erreur de traitement: {str(e)}"
        logging.error(error_msg)
        client.put_object(
            bucket_name=BUCKET_NAME,
            object_name=f"process_error_{current_date}.txt",
            data=BytesIO(error_msg.encode()),
            length=len(error_msg),
            content_type='text/plain'
        )
        raise e
    finally:
        # Nettoyage
        if TEMP_DIR and os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)


# Configuration du DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,  # Pas de réessai pour faciliter le débogage
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'iris_contours_collection',
    default_args=default_args,
    description='Collecte simplifiée des contours IRIS pour la Bretagne',
    schedule_interval='@quarterly',
    start_date=datetime(2024, 2, 11),
    catchup=False,
    tags=['iris', 'contours', 'bretagne', 'geodata'],
)

# Définition des tâches
create_bucket = PythonOperator(
    task_id='ensure_bucket_exists',
    python_callable=ensure_bucket_exists,
    dag=dag,
)

download_and_extract = PythonOperator(
    task_id='download_and_extract',
    python_callable=download_and_extract,
    dag=dag,
)

process_and_save = PythonOperator(
    task_id='process_and_save',
    python_callable=process_and_save,
    dag=dag,
)

# Définition des dépendances
create_bucket >> download_and_extract >> process_and_save