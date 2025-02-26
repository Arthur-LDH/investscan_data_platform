from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

from etl.common.spark import SparkManager
from etl.common.base_process_job import BaseETLJob


class DVFProcessor(BaseETLJob):
    """Traitement des données DVF"""

    def __init__(self):
        super().__init__("DVFProcessor")

    def extract(self):
        """Extrait les données DVF depuis MinIO"""
        print("Début de l'extraction")
        self.metrics["extraction_start"] = F.current_timestamp()

        print("Tentative de lecture du fichier")
        try:
            raw_df = self.minio.read("dvf-data", "dvf_22_2019.csv")
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier: {str(e)}")
            raise e

        print(f"DataFrame lu - nombre de lignes : {raw_df.count()}")
        print(f"Colonnes du DataFrame : {raw_df.columns}")

        if raw_df.count() == 0:
            print("ATTENTION : Le DataFrame est vide !")
            raise ValueError("Le fichier CSV est vide ou n'a pas été trouvé.")

        self.metrics["row_count_raw"] = raw_df.count()
        self.metrics["extraction_end"] = F.current_timestamp()

        return raw_df

    def transform(self, data_frame):
        """Transforme les données DVF"""
        self.metrics["transformation_start"] = F.current_timestamp()
        print("Début de la transformation")

        # Étape 1: Conversion des types de base
        typed_df = (data_frame
                    .withColumn("valeur_fonciere", F.col("valeur_fonciere").cast(DoubleType()))
                    .withColumn("surface_reelle_bati", F.col("surface_reelle_bati").cast(DoubleType()))
                    .withColumn("date_mutation", F.to_date(F.col("date_mutation"), "yyyy-MM-dd"))
                    .withColumn("longitude", F.col("longitude").cast(DoubleType()))
                    .withColumn("latitude", F.col("latitude").cast(DoubleType()))
                    .withColumn("surface_terrain", F.col("surface_terrain").cast(DoubleType()))
                    .withColumn("nombre_pieces_principales", F.col("nombre_pieces_principales").cast(IntegerType()))
                    )

        print("Types de données convertis")

        # Étape 2: Séparer les bâtiments des terrains
        batiments_df = typed_df.filter(F.col("type_local").isNotNull())
        terrains_df = typed_df.filter(F.col("type_local").isNull())

        print(f"Nombre de bâtiments : {batiments_df.count()}")

        # Étape 3: Traitement des bâtiments
        batiments_clean = (batiments_df
                           # Nettoyer les valeurs nulles essentielles
                           .filter(F.col("valeur_fonciere").isNotNull())
                           .filter(F.col("surface_reelle_bati").isNotNull())

                           # Normalisation des chaînes
                           .withColumn("type_local", F.lower(F.trim(F.col("type_local"))))

                           # Calcul du prix au m²
                           .withColumn("prix_m2", F.col("valeur_fonciere") / F.col("surface_reelle_bati"))

                           # Filtrer les valeurs aberrantes pour les bâtiments
                           .filter(F.col("prix_m2").between(100, 20000))
                           )

        print(f"Nombre de bâtiments après nettoyage : {batiments_clean.count()}")

        # Étape 4: Traitement des terrains
        terrains_clean = (terrains_df
                          # Nettoyer les valeurs nulles essentielles
                          .filter(F.col("valeur_fonciere").isNotNull())
                          .filter(F.col("surface_terrain").isNotNull())

                          # Normalisation des chaînes
                          .withColumn("nature_culture", F.lower(F.trim(F.col("nature_culture"))))

                          # Calcul du prix au m² pour le terrain
                          .withColumn("prix_m2_terrain", F.col("valeur_fonciere") / F.col("surface_terrain"))

                          # Filtrer les valeurs aberrantes pour les terrains
                          .filter(F.col("prix_m2_terrain").between(0.5, 1000))
                          )

        print(f"Nombre de terrains après nettoyage : {terrains_clean.count()}")

        # Étape 5: Agrégations pour résoudre le problème des lignes multiples par mutation

        # Pour les bâtiments, regrouper par id_mutation et prendre la somme des surfaces
        batiments_agg = (batiments_clean
                         .groupBy("id_mutation", "date_mutation", "valeur_fonciere", "code_postal",
                                  "nom_commune", "code_departement", "type_local", "nombre_pieces_principales")
                         .agg(
            F.sum("surface_reelle_bati").alias("surface_totale_bati"),
            F.max("surface_terrain").alias("surface_terrain_bati"),
            F.avg("longitude").alias("longitude"),
            F.avg("latitude").alias("latitude")
        )
                         .withColumn("prix_m2_recalcule", F.col("valeur_fonciere") / F.col("surface_totale_bati"))
                         )

        print(f"Nombre de bâtiments agrégés : {batiments_agg.count()}")

        # Pour les terrains, regrouper par id_mutation
        terrains_agg = (terrains_clean
                        .groupBy("id_mutation", "date_mutation", "valeur_fonciere", "code_postal",
                                 "nom_commune", "code_departement")
                        .agg(
            F.sum("surface_terrain").alias("surface_totale_terrain"),
            F.collect_list("nature_culture").alias("natures_culture"),
            F.avg("longitude").alias("longitude"),
            F.avg("latitude").alias("latitude")
        )
                        .withColumn("prix_m2_terrain_recalcule",
                                    F.col("valeur_fonciere") / F.col("surface_totale_terrain"))
                        .withColumn("type_bien", F.lit("terrain"))
                        )

        print(f"Nombre de terrains agrégés : {terrains_agg.count()}")

        # Étape 6: Traitement final des colonnes
        batiments_final = (batiments_agg
                           .withColumn("surface_terrain", F.col("surface_terrain_bati"))
                           .withColumn("prix_m2", F.col("prix_m2_recalcule"))
                           .drop("surface_terrain_bati", "prix_m2_recalcule")
                           )

        print(f"Nombre de bâtiments finaux : {batiments_final.count()}")

        terrains_final = (terrains_agg
                          .withColumn("surface_reelle_bati", F.lit(None).cast(DoubleType()))
                          .withColumn("prix_m2", F.col("prix_m2_terrain_recalcule"))
                          .withColumn("type_local", F.col("type_bien"))
                          .drop("prix_m2_terrain_recalcule", "type_bien")
                          )

        print(f"Nombre de terrains finaux : {terrains_final.count()}")

        # Étape 7: Union des bâtiments et terrains traités
        cleaned_df = batiments_final.unionByName(
            terrains_final.select(batiments_final.columns),
            allowMissingColumns=True
        )

        print(f"Nombre total de lignes après transformation : {cleaned_df.count()}")

        # Calculer métriques
        self.metrics["row_count_transformed"] = cleaned_df.count()
        self.metrics["transformation_end"] = F.current_timestamp()

        return cleaned_df

    def load(self, data_frame):
        """Charge les données transformées dans MinIO et PostgreSQL"""
        self.metrics["loading_start"] = F.current_timestamp()

        # Sauvegarder dans MinIO
        self.minio.write(data_frame, "cleansed", "dvf/cleaned", format="parquet")

        # Préparer pour PostgreSQL - adapter les colonnes selon votre modèle de base de données
        db_ready_df = data_frame.select(
            F.col("id_mutation").alias("transaction_id"),
            F.col("date_mutation").alias("sale_date"),
            F.col("valeur_fonciere").alias("sale_price"),
            F.col("prix_m2").alias("price_per_sqm"),
            F.col("type_local").alias("property_type"),
            F.col("surface_reelle_bati").alias("living_area"),
            F.col("surface_terrain").alias("land_area"),
            F.col("nombre_pieces_principales").alias("rooms"),
            F.col("code_postal").alias("postal_code"),
            F.col("nom_commune").alias("city_name"),
            F.col("code_departement").alias("department"),
            F.col("longitude"),
            F.col("latitude"),
            # Vous pouvez ajouter d'autres colonnes selon votre schéma
        )

        # Ajouter une colonne geom pour PostGIS si nécessaire
        if "longitude" in db_ready_df.columns and "latitude" in db_ready_df.columns:
            db_ready_df = db_ready_df.withColumn(
                "coordinates",
                F.expr("ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)")
            )

        self.postgres.write(db_ready_df, "transactions")
        self.metrics["loading_end"] = F.current_timestamp()


if __name__ == "__main__":
    spark_manager = SparkManager.get_instance()
    processor = None
    try:
        processor = DVFProcessor()
        result = processor.execute()
        print(f"Traitement terminé avec succès: {result}")
    except Exception as e:
        print(f"Erreur lors du traitement: {str(e)}")
    finally:
        if spark_manager:
            spark_manager.stop()
        if processor:
            processor.close()  # À ajouter dans votre BaseETLJob si nécessaire