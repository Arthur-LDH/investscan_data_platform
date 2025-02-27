from pyspark.sql import SparkSession


class SparkManager:
    """Gestionnaire de session Spark"""

    _instance = None

    @classmethod
    def get_instance(cls, app_name="InvestScan"):
        """Implémentation singleton pour réutiliser la session"""
        if cls._instance is None:
            cls._instance = cls(app_name)
        return cls._instance

    def __init__(self, app_name):
        """Initialise la session Spark avec la configuration requise"""
        self.spark = (SparkSession.builder
                      .appName(app_name)
                      .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.1")
                      .config("spark.hadoop.fs.s3a.endpoint", "http://localhost:9000")
                      .config("spark.hadoop.fs.s3a.access.key", "minioadmin")
                      .config("spark.hadoop.fs.s3a.secret.key", "minioadmin")
                      .config("spark.hadoop.fs.s3a.path.style.access", "true")
                      .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
                      .getOrCreate())

    def get_session(self):
        """Retourne la session Spark"""
        return self.spark

    def stop(self):
        """Arrête la session Spark"""
        if self.spark:
            self.spark.stop()