from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, DateType
from etl.jobs.base_job import BaseETLJob


class DVFProcessor(BaseETLJob):
    """Traitement des données DVF"""

    def __init__(self):
        super().__init__("DVFProcessor")

    def extract(self):
        """Extrait les données DVF depuis MinIO"""
        self.metrics["extraction_start"] = F.current_timestamp()

        # Lire les données brutes
        raw_df = self.minio.read("raw", "dvf/dvf_latest.csv")

        self.metrics["row_count_raw"] = raw_df.count()
        self.metrics["extraction_end"] = F.current_timestamp()

        return raw_df

    def transform(self, data_frame):
        """Transforme les données DVF"""
        self.metrics["transformation_start"] = F.current_timestamp()

        # Effectuer les transformations
        cleaned_df = (data_frame
                      # Convertir les types
                      .withColumn("valeur_fonciere", F.col("valeur_fonciere").cast(DoubleType()))
                      .withColumn("surface_reelle_bati", F.col("surface_reelle_bati").cast(DoubleType()))
                      .withColumn("date_mutation", F.to_date(F.col("date_mutation"), "yyyy-MM-dd"))

                      # Nettoyer les valeurs nulles
                      .filter(F.col("valeur_fonciere").isNotNull())
                      .filter(F.col("surface_reelle_bati").isNotNull())

                      # Normaliser les chaînes
                      .withColumn("type_local", F.lower(F.trim(F.col("type_local"))))

                      # Calculer de nouvelles variables
                      .withColumn("prix_m2", F.col("valeur_fonciere") / F.col("surface_reelle_bati"))

                      # Filtrer les valeurs aberrantes
                      .filter(F.col("prix_m2").between(100, 20000))
                      )

        self.metrics["row_count_transformed"] = cleaned_df.count()
        self.metrics["transformation_end"] = F.current_timestamp()

        return cleaned_df

    def load(self, data_frame):
        """Charge les données transformées dans MinIO et PostgreSQL"""
        self.metrics["loading_start"] = F.current_timestamp()

        # Sauvegarder dans MinIO
        self.minio.write(data_frame, "cleansed", "dvf/cleaned")

        # Préparer pour PostgreSQL
        db_ready_df = data_frame.select(
            F.col("id_mutation").alias("transaction_id"),
            F.col("date_mutation").alias("sale_date"),
            F.col("valeur_fonciere").alias("sale_price"),
            F.col("prix_m2").alias("price_per_sqm"),
            # ... autres colonnes
        )

        # Sauvegarder dans PostgreSQL
        self.postgres.write(db_ready_df, "transactions")

        self.metrics["loading_end"] = F.current_timestamp()