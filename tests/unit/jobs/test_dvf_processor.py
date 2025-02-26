import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DateType
import pandas as pd
from datetime import date

from etl.jobs.dvf_processor import DVFProcessor
from etl.common.storage import MinIOConnector, PostgresConnector


# Fixtures pour les tests
@pytest.fixture
def spark():
    """Crée une session Spark locale pour les tests"""
    return (SparkSession.builder
            .master("local[*]")
            .appName("DVFProcessorTest")
            .getOrCreate())


@pytest.fixture
def sample_dvf_data(spark):
    """Crée un DataFrame d'exemple pour les tests"""
    # Définir le schéma
    schema = StructType([
        StructField("id_mutation", StringType(), True),
        StructField("date_mutation", StringType(), True),
        StructField("valeur_fonciere", StringType(), True),
        StructField("surface_reelle_bati", StringType(), True),
        StructField("type_local", StringType(), True),
        StructField("code_postal", StringType(), True),
        # Ajoutez d'autres champs selon besoin
    ])

    # Créer des données d'exemple
    data = [
        ("M001", "2023-01-15", "200000", "80", "Appartement", "75001"),
        ("M002", "2023-02-20", "350000", "120", "Maison", "75002"),
        ("M003", "2023-03-10", "null", "90", "Appartement", "75003"),  # Valeur manquante
        ("M004", "2023-04-05", "450000", "null", "Maison", "75004"),  # Surface manquante
        ("M005", "2023-05-18", "180000", "60", "  appartement  ", "75005"),  # Espaces à nettoyer
        ("M006", "2023-06-22", "1000000", "30", "Appartement", "75006"),  # Valeur aberrante (prix/m² > 20000)
    ]

    # Créer le DataFrame
    return spark.createDataFrame(data, schema)


class TestDVFProcessor:
    """Tests unitaires pour DVFProcessor"""

    def test_transform(self, spark, sample_dvf_data, monkeypatch):
        """Test de la méthode transform"""
        # Arrangement
        processor = DVFProcessor()
        processor.spark = spark

        # Action
        transformed_df = processor.transform(sample_dvf_data)
        transformed_data = transformed_df.collect()

        # Assertions
        assert transformed_df.count() == 3  # Seulement 3 lignes valides après filtrage

        # Vérifier les types convertis
        assert isinstance(transformed_df.schema["valeur_fonciere"].dataType, DoubleType)
        assert isinstance(transformed_df.schema["date_mutation"].dataType, DateType)

        # Vérifier le calcul du prix au m²
        assert "prix_m2" in transformed_df.columns

        # Vérifier le nettoyage du type_local
        assert transformed_data[2]["type_local"] == "appartement"  # Espaces supprimés et mis en minuscules

    def test_extract(self, spark, sample_dvf_data, monkeypatch):
        """Test de la méthode extract avec mock de MinIOConnector"""
        # Arrangement
        processor = DVFProcessor()
        processor.spark = spark

        # Mock de la méthode read du MinIOConnector
        class MockMinIO:
            def read(self, bucket, path, format="csv", options=None):
                return sample_dvf_data

        processor.minio = MockMinIO()

        # Action
        extracted_df = processor.extract()

        # Assertions
        assert extracted_df.count() == 6
        assert "id_mutation" in extracted_df.columns

    def test_load(self, spark, sample_dvf_data, monkeypatch):
        """Test de la méthode load avec mock des connecteurs"""
        # Arrangement
        processor = DVFProcessor()
        processor.spark = spark

        # Transformer d'abord les données
        transformed_df = processor.transform(sample_dvf_data)

        # Mock des méthodes d'écriture
        minio_write_called = False
        postgres_write_called = False

        class MockMinIO:
            def write(self_mock, df, bucket, path, **kwargs):
                nonlocal minio_write_called
                minio_write_called = True
                # Vérifier que le DataFrame est celui attendu
                assert df.count() == transformed_df.count()

        class MockPostgres:
            def write(self_mock, df, table_name, **kwargs):
                nonlocal postgres_write_called
                postgres_write_called = True
                # Vérifier que la table est correcte
                assert table_name == "transactions"

        processor.minio = MockMinIO()
        processor.postgres = MockPostgres()

        # Action
        processor.load(transformed_df)

        # Assertions
        assert minio_write_called
        assert postgres_write_called

    def test_execute_success(self, spark, sample_dvf_data, monkeypatch):
        """Test du flux complet avec succès"""
        # Arrangement
        processor = DVFProcessor()
        processor.spark = spark

        # Mock des méthodes
        def mock_extract():
            return sample_dvf_data

        extract_called = False
        transform_called = False
        load_called = False

        original_extract = processor.extract
        original_transform = processor.transform
        original_load = processor.load

        def patched_extract():
            nonlocal extract_called
            extract_called = True
            return original_extract()

        def patched_transform(df):
            nonlocal transform_called
            transform_called = True
            return original_transform(df)

        def patched_load(df):
            nonlocal load_called
            load_called = True
            return original_load(df)

        monkeypatch.setattr(processor, "extract", patched_extract)
        monkeypatch.setattr(processor, "transform", patched_transform)
        monkeypatch.setattr(processor, "load", patched_load)

        # Mock des connecteurs
        class MockMinIO:
            def read(self, bucket, path, **kwargs):
                return sample_dvf_data

            def write(self, df, bucket, path, **kwargs):
                pass

        class MockPostgres:
            def write(self, df, table_name, **kwargs):
                pass

        processor.minio = MockMinIO()
        processor.postgres = MockPostgres()

        # Action
        result = processor.execute()

        # Assertions
        assert extract_called
        assert transform_called
        assert load_called
        assert result["status"] == "success"
        assert "metrics" in result

    def test_execute_failure(self, spark, monkeypatch):
        """Test du flux avec erreur"""
        # Arrangement
        processor = DVFProcessor()
        processor.spark = spark

        # Force une erreur dans extract
        def failed_extract():
            raise ValueError("Erreur simulée dans extract")

        monkeypatch.setattr(processor, "extract", failed_extract)

        # Action & Assert
        with pytest.raises(Exception) as excinfo:
            processor.execute()

        # Vérifier que l'exception contient les informations d'erreur
        assert "ETL job failed" in str(excinfo.value)
        assert "Erreur simulée dans extract" in str(excinfo.value)