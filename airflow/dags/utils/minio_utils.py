"""
Fonctions utilitaires pour interagir avec MinIO.
"""
from minio import Minio
from .. import MINIO_CONFIG
import logging

logger = logging.getLogger(__name__)


def get_minio_client() -> Minio:
    """Crée et retourne un client MinIO."""
    return Minio(
        MINIO_CONFIG['endpoint'],
        access_key=MINIO_CONFIG['access_key'],
        secret_key=MINIO_CONFIG['secret_key'],
        secure=MINIO_CONFIG['secure']
    )


def ensure_bucket_exists(bucket_name: str) -> bool:
    """
    S'assure qu'un bucket existe, le crée si nécessaire.

    Args:
        bucket_name: Nom du bucket à vérifier/créer

    Returns:
        bool: True si le bucket existe ou a été créé
    """
    client = get_minio_client()
    try:
        if not client.bucket_exists(bucket_name):
            logger.info(f"Création du bucket {bucket_name}")
            client.make_bucket(bucket_name)
        return True
    except Exception as e:
        logger.error(f"Erreur avec le bucket {bucket_name}: {str(e)}")
        raise