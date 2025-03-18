import pandas as pd
import numpy as np
import geopandas as gpd
import io
import os
import tempfile
from minio import Minio


def get_minio_client():
    """
    Crée et retourne un client MinIO
    """
    minio_endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")

    # Configuration du client MinIO
    return Minio(
        endpoint=minio_endpoint,
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False
    )


def load_data_from_minio(bucket_name, object_name, file_format="csv", **kwargs):
    """
    Charge des données depuis MinIO

    Args:
        bucket_name (str): Nom du bucket MinIO
        object_name (str): Chemin de l'objet dans le bucket
        file_format (str): Format du fichier ('csv', 'parquet', 'json')
        **kwargs: Arguments supplémentaires pour la fonction de lecture

    Returns:
        pandas.DataFrame: DataFrame contenant les données
    """
    # Obtenir le client MinIO
    minio_client = get_minio_client()

    # Vérifier si le bucket existe
    if not minio_client.bucket_exists(bucket_name):
        raise Exception(f"Le bucket '{bucket_name}' n'existe pas")

    # Télécharger l'objet
    response = minio_client.get_object(bucket_name, object_name)
    data = response.read()

    # Convertir les bytes en un objet buffer
    buffer = io.BytesIO(data)

    # Charger les données selon le format
    if file_format.lower() == "csv":
        return pd.read_csv(buffer, **kwargs)
    elif file_format.lower() == "parquet":
        return pd.read_parquet(buffer, **kwargs)
    elif file_format.lower() == "json":
        return pd.read_json(buffer, **kwargs)
    else:
        raise ValueError(f"Format de fichier non supporté: {file_format}")


def load_geojson_from_minio(bucket_name, object_name):
    """
    Charge un fichier GeoJSON depuis MinIO

    Args:
        bucket_name (str): Nom du bucket MinIO
        object_name (str): Chemin de l'objet dans le bucket

    Returns:
        geopandas.GeoDataFrame: GeoDataFrame contenant les données géographiques
    """
    # Obtenir le client MinIO
    minio_client = get_minio_client()

    # Vérifier si le bucket existe
    if not minio_client.bucket_exists(bucket_name):
        raise Exception(f"Le bucket '{bucket_name}' n'existe pas")

    # Télécharger le fichier dans un dossier temporaire
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = os.path.join(temp_dir, "contours-iris.geojson")

        # Télécharger le fichier
        minio_client.fget_object(bucket_name, object_name, temp_file)

        # Charger avec geopandas
        gdf = gpd.read_file(temp_file)

        return gdf