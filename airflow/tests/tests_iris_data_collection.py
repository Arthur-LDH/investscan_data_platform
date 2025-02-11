"""
Tests unitaires pour le DAG de collecte des données IRIS.
"""
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import pyarrow as pa
from datetime import datetime
from io import BytesIO

# Import du DAG et ses composants
from apache_airflow.dags.iris_data_collection import (
    fetch_api_data,
    collect_data,
    process_data,
    BUCKET_NAME,
    API_CONF
)
from apache_airflow.dags.utils.data_utils import prepare_parquet_data, create_metadata
from apache_airflow.dags.utils.minio_utils import get_minio_client, ensure_bucket_exists


class TestIrisDAGUtils(unittest.TestCase):
    """Tests des fonctions utilitaires."""

    def setUp(self):
        """Initialisation des données de test."""
        self.sample_data = [
            {
                "iris": "123456789",
                "commune": "RENNES",
                "population": 1000,
                "superficie": 150.5
            },
            {
                "iris": "987654321",
                "commune": "BREST",
                "population": 800,
                "superficie": 120.3
            }
        ]

    def test_prepare_parquet_data(self):
        """Test de la préparation des données Parquet."""
        # Test de la fonction
        table, buffer = prepare_parquet_data(self.sample_data)

        # Vérifications
        self.assertIsInstance(table, pa.Table)
        self.assertIsInstance(buffer, BytesIO)

        # Vérification du contenu
        df = table.to_pandas()
        self.assertEqual(len(df), 2)
        self.assertTrue('iris' in df.columns)
        self.assertTrue('commune' in df.columns)

    def test_create_metadata(self):
        """Test de la création des métadonnées."""
        # Création d'une table test
        df = pd.DataFrame(self.sample_data)
        table = pa.Table.from_pandas(df)

        # Test
        metadata = create_metadata(
            table=table,
            data_size=1000,
            source_url=API_CONF['base_url']
        )

        # Vérifications
        self.assertIn('date_extraction', metadata)
        self.assertIn('schema', metadata)
        self.assertEqual(metadata['nombre_lignes'], 2)
        self.assertEqual(metadata['taille_fichier'], 1000)


class TestIrisDAGAPI(unittest.TestCase):
    """Tests des fonctions d'API."""

    @patch('requests.get')
    def test_fetch_api_data(self, mock_get):
        """Test de la récupération des données de l'API."""
        # Configuration du mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'total_count': 2,
            'results': [
                {"iris": "123", "commune": "RENNES"},
                {"iris": "456", "commune": "BREST"}
            ]
        }
        mock_get.return_value = mock_response

        # Test
        data = fetch_api_data("Bretagne", 0)

        # Vérifications
        self.assertEqual(data['total_count'], 2)
        self.assertEqual(len(data['results']), 2)
        mock_get.assert_called_once()

    @patch('requests.get')
    def test_fetch_api_data_error(self, mock_get):
        """Test de la gestion des erreurs d'API."""
        # Configuration du mock pour simuler une erreur
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        # Vérification que l'erreur est bien levée
        with self.assertRaises(Exception):
            fetch_api_data("Bretagne", 0)


class TestIrisDAGMinio(unittest.TestCase):
    """Tests des fonctions MinIO."""

    @patch('minio.Minio')
    def test_ensure_bucket_exists(self, mock_minio):
        """Test de la création/vérification du bucket."""
        # Configuration du mock
        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = False
        mock_minio.return_value = mock_client

        # Test
        result = ensure_bucket_exists(BUCKET_NAME)

        # Vérifications
        self.assertTrue(result)
        mock_client.bucket_exists.assert_called_once_with(BUCKET_NAME)
        mock_client.make_bucket.assert_called_once_with(BUCKET_NAME)


class TestIrisDAGIntegration(unittest.TestCase):
    """Tests d'intégration."""

    @patch('airflow.dags.iris_data_collection.get_minio_client')
    @patch('requests.get')
    def test_full_process(self, mock_get, mock_minio_client):
        """Test du processus complet."""
        # Configuration des mocks
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'total_count': 2,
            'results': [
                {"iris": "123", "commune": "RENNES"},
                {"iris": "456", "commune": "BREST"}
            ]
        }
        mock_get.return_value = mock_response

        mock_client = MagicMock()
        mock_minio_client.return_value = mock_client

        # Test
        result = process_data()

        # Vérifications
        self.assertIn('parquet_file', result)
        self.assertIn('metadata_file', result)
        self.assertEqual(result['record_count'], 2)

        # Vérification des appels MinIO
        self.assertEqual(mock_client.put_object.call_count, 2)  # Parquet + Metadata


def test_dag_structure(dagbag):
    """Test de la structure du DAG."""
    dag = dagbag.get_dag('iris_data_collection')

    # Vérification de l'existence du DAG
    assert dag is not None

    # Vérification des tâches
    tasks = dag.tasks
    task_ids = [task.task_id for task in tasks]
    assert 'setup_bucket' in task_ids
    assert 'process_data' in task_ids

    # Vérification des dépendances
    setup_bucket_task = dag.get_task('setup_bucket')
    process_data_task = dag.get_task('process_data')
    assert process_data_task in setup_bucket_task.downstream_list


if __name__ == '__main__':
    unittest.main()