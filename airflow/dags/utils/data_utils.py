"""
Fonctions utilitaires pour le traitement des données.
"""
from typing import List, Tuple, Dict, Any
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from io import BytesIO
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def prepare_parquet_data(data: List[Dict[str, Any]]) -> Tuple[pa.Table, BytesIO]:
    """
    Prépare les données pour le format Parquet.

    Args:
        data: Liste de dictionnaires contenant les données

    Returns:
        Tuple contenant la table PyArrow et le buffer Parquet
    """
    logger.info("Conversion des données en DataFrame")
    df = pd.json_normalize(data)

    # Optimisation des types pour les colonnes catégorielles
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype('category')
            logger.debug(f"Colonne {col} convertie en catégorie")

    logger.info("Conversion en Table PyArrow")
    table = pa.Table.from_pandas(df)

    logger.info("Création du buffer Parquet")
    parquet_buffer = BytesIO()
    pq.write_table(
        table,
        parquet_buffer,
        compression='snappy',
        use_dictionary=True
    )
    parquet_buffer.seek(0)

    return table, parquet_buffer


def create_metadata(
        table: pa.Table,
        data_size: int,
        source_url: str,
        additional_info: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Crée les métadonnées pour le fichier Parquet.

    Args:
        table: Table PyArrow
        data_size: Taille des données en bytes
        source_url: URL source des données
        additional_info: Informations supplémentaires à inclure

    Returns:
        Dictionnaire des métadonnées
    """
    metadata = {
        'date_extraction': datetime.now().isoformat(),
        'schema': [
            {
                'nom': field.name,
                'type': str(field.type)
            }
            for field in table.schema
        ],
        'nombre_lignes': table.num_rows,
        'nombre_colonnes': table.num_columns,
        'taille_fichier': data_size,
        'source': source_url
    }

    if additional_info:
        metadata.update(additional_info)

    return metadata