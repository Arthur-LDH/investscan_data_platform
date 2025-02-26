from abc import ABC, abstractmethod
from etl.common.spark import SparkManager
from etl.common.storage import MinIOConnector, PostgresConnector


class BaseETLJob(ABC):
    """Classe de base pour tous les jobs ETL"""

    def __init__(self, app_name=None):
        """Initialise les ressources communes pour tous les jobs"""
        if app_name is None:
            app_name = self.__class__.__name__

        # Initialiser Spark
        spark_manager = SparkManager.get_instance(app_name)
        self.spark = spark_manager.get_session()

        # Initialiser les connecteurs
        self.minio = MinIOConnector(self.spark)
        self.postgres = PostgresConnector(self.spark)

        # Métriques et logs
        self.metrics = {}

    @abstractmethod
    def extract(self):
        """Extrait les données des sources"""
        pass

    @abstractmethod
    def transform(self, data_frame):
        """Transforme les données"""
        pass

    @abstractmethod
    def load(self, data_frame):
        """Charge les données dans la destination"""
        pass

    def execute(self):
        """Exécute le job ETL complet"""
        try:
            # Extraction
            raw_data = self.extract()

            # Transformation
            transformed_data = self.transform(raw_data)

            # Chargement
            self.load(transformed_data)

            # Récapitulatif
            return {
                "status": "success",
                "metrics": self.metrics,
                "job_name": self.__class__.__name__
            }

        except Exception as e:
            # Gestion d'erreur
            error_info = {
                "status": "failed",
                "error": str(e),
                "job_name": self.__class__.__name__
            }
            raise Exception(f"ETL job failed: {error_info}")