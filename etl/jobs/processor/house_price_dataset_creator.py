from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from etl.common.base_process_job import BaseETLJob
import logging
import json
import io
import datetime
import tempfile
import os
import shutil
import numpy as np


class HousePriceDatasetCreator(BaseETLJob):
    """
    Job ETL pour la création d'un dataset d'apprentissage pour l'estimation des prix immobiliers
    """

    def __init__(self,
                 input_bucket="processed",
                 input_base_path="dvf_enriched",
                 input_version="latest",
                 output_bucket="ml-datasets",
                 output_base_path="house_price_model",
                 version=None,
                 reference_year=None):
        super().__init__("HousePriceDatasetCreator")
        self.input_bucket = input_bucket
        self.input_base_path = input_base_path
        self.input_version = input_version
        self.output_bucket = output_bucket

        # Gestion du versionnement des datasets
        self.version = version or f"v_{datetime.datetime.now().strftime('%Y%m%d')}"
        self.output_base_path = output_base_path
        self.output_path = f"{output_base_path}/{self.version}"

        # Année de référence pour l'inflation
        self.reference_year = reference_year or datetime.datetime.now().year
        self.dvf_actual_version = None

        # Configuration du logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def resolve_dvf_version(self):
        """Résout la version réelle des données DVF à utiliser"""
        if self.input_version != "latest":
            return self.input_version

        try:
            latest_info = self.minio.minio_client.get_object(
                bucket_name=self.input_bucket,
                object_name=f"{self.input_base_path}/latest/info.json"
            )
            actual_version = json.loads(latest_info.read().decode('utf-8')).get("redirect_to")
            if actual_version:
                return actual_version
        except:
            pass

        try:
            catalog = self.minio.minio_client.get_object(
                bucket_name=self.input_bucket,
                object_name=f"{self.input_base_path}/catalog.json"
            )
            versions = json.loads(catalog.read().decode('utf-8')).get("versions", [])
            if versions:
                versions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
                return versions[0]["version"]
        except Exception as e:
            self.logger.error(f"Impossible de déterminer la version DVF: {str(e)}")

        raise ValueError("Aucune version DVF trouvée")

    def extract(self):
        """Extrait les données DVF consolidées"""
        self.metrics["extraction_start"] = F.current_timestamp()

        # Résoudre la version réelle
        self.dvf_actual_version = self.resolve_dvf_version()
        input_path = f"{self.input_base_path}/{self.dvf_actual_version}/data.parquet"

        self.logger.info(f"Extraction depuis {self.input_bucket}/{input_path}")

        try:
            dvf_data = self.minio.read(
                bucket=self.input_bucket,
                path=input_path,
                format="parquet"
            )

            self.logger.info(f"Données chargées: {dvf_data.count()} lignes")
            self.metrics["rows_raw"] = dvf_data.count()
            self.metrics["dvf_version"] = self.dvf_actual_version
            self.metrics["extraction_end"] = F.current_timestamp()

            return dvf_data

        except Exception as e:
            self.logger.error(f"Erreur d'extraction: {str(e)}")
            raise ValueError(f"Échec de l'extraction: {str(e)}")

    def transform(self, data_frame):
        """Prépare les données pour le modèle d'estimation des prix immobiliers"""
        self.metrics["transformation_start"] = F.current_timestamp()
        self.logger.info("Transformation des données")

        try:
            # 1. Filtrer pour garder les ventes de maisons et d'appartements
            houses_df = data_frame.filter(
                (F.col("nature_mutation").ilike("%vente%")) &
                F.col("code_type_local").isin(1, 2)
            )

            # 2. Sélectionner les caractéristiques et nettoyer
            selected_df = houses_df.select(
                "annee_mutation", "mois_mutation",
                "valeur_fonciere", "code_postal",
                "surface_reelle_bati", "nombre_pieces_principales",
                "surface_terrain",
                "longitude", "latitude",
                "code_type_local",
                "DEC_MED21",  # Revenu médian - l'indicateur le plus important
                "DEC_D121",  # Premier décile (revenus les plus bas)
                "DEC_D921",  # Neuvième décile (revenus les plus hauts)
                "DEC_GI21"  # Indice de Gini (inégalité)
            ).filter(
                F.col("valeur_fonciere").isNotNull() &
                F.col("surface_reelle_bati").isNotNull() &
                F.col("nombre_pieces_principales").isNotNull() &
                (F.col("code_type_local").isNotNull()) &
                (F.col("code_postal").isNotNull()) &
                (F.col("surface_reelle_bati") > 0) &
                (F.col("nombre_pieces_principales") > 0) &
                (F.col("surface_terrain").isNotNull()) &
                (F.col("longitude").isNotNull()) &
                (F.col("latitude").isNotNull()) &
                (F.col("valeur_fonciere") > 10000) &
                (F.col("surface_reelle_bati") > 10) &
                F.col("DEC_MED21").isNotNull() &
                F.col("DEC_D121").isNotNull() &
                F.col("DEC_D921").isNotNull() &
                F.col("DEC_GI21").isNotNull()
            )

            # 3. Calculer les indicateurs dérivés
            enriched_df = (selected_df
            .withColumn(
                "ratio_terrain_bati",
                F.when(F.col("surface_terrain").isNotNull() & (F.col("surface_terrain") > 0) &
                       (F.col("surface_reelle_bati") > 0),
                       F.col("surface_terrain") / F.col("surface_reelle_bati")).otherwise(0)
            ))

            # Après les calculs du ratio_terrain_bati
            # final_df = self.add_sea_distance(enriched_df)
            final_df = enriched_df

            # Metrics
            self.metrics["rows_transformed"] = final_df.count()
            self.metrics["rows_filtered_out"] = self.metrics["rows_raw"] - final_df.count()
            self.metrics["reference_year"] = self.reference_year
            self.metrics["transformation_end"] = F.current_timestamp()

            self.logger.info(f"Transformation terminée: {final_df.count()} lignes")
            return final_df

        except Exception as e:
            self.logger.error(f"Erreur de transformation: {str(e)}")
            raise ValueError(f"Échec de la transformation: {str(e)}")

    def load(self, data_frame):
        """Sauvegarde le dataset avec un minimum de connexions MinIO"""
        self.metrics["loading_start"] = F.current_timestamp()
        self.logger.info(f"Chargement vers {self.output_bucket}/{self.output_path}")

        try:
            self.minio.ensure_bucket_exists(self.output_bucket)
            train_df, test_df, validation_df = self.split_dataset(data_frame)
            # inflation_factors = data_frame.select(
            #     "annee_mutation", "code_postal", "inflation_factor"
            # ).distinct()
            temp_dir = tempfile.mkdtemp()
            self.logger.info(f"Dossier temporaire: {temp_dir}")

            try:
                version_dir = os.path.join(temp_dir, self.version)
                os.makedirs(version_dir)
                latest_dir = os.path.join(temp_dir, "latest")
                os.makedirs(latest_dir)

                self.logger.info("Écriture locale des dataframes")
                data_frame.coalesce(1).write.parquet(os.path.join(version_dir, "full_dataset.parquet"))
                train_df.coalesce(1).write.parquet(os.path.join(version_dir, "train.parquet"))
                test_df.coalesce(1).write.parquet(os.path.join(version_dir, "test.parquet"))
                validation_df.coalesce(1).write.parquet(os.path.join(version_dir, "validation.parquet"))
                # inflation_factors.coalesce(1).write.parquet(os.path.join(version_dir, "inflation_factors.parquet"))

                with open(os.path.join(version_dir, "metadata.json"), 'w') as f:
                    json.dump(self.create_metadata(data_frame, train_df, test_df, validation_df), f, indent=2)

                self.update_catalog(temp_dir, latest_dir)

                # Upload vers MinIO
                self.logger.info("Upload vers MinIO")
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        local_path = os.path.join(root, file)
                        rel_path = os.path.relpath(local_path, temp_dir)
                        minio_path = f"{self.output_base_path}/{rel_path}"

                        if os.path.isdir(local_path) or file.endswith('.parquet'):
                            # Traitement des dossiers parquet
                            if os.path.isdir(local_path):
                                for p_root, p_dirs, p_files in os.walk(local_path):
                                    for p_file in p_files:
                                        local_p_path = os.path.join(p_root, p_file)
                                        rel_p_path = os.path.relpath(local_p_path, temp_dir)
                                        minio_p_path = f"{self.output_base_path}/{rel_p_path}"

                                        self.minio.minio_client.fput_object(
                                            bucket_name=self.output_bucket,
                                            object_name=minio_p_path,
                                            file_path=local_p_path
                                        )
                            else:
                                self.minio.minio_client.fput_object(
                                    bucket_name=self.output_bucket,
                                    object_name=minio_path,
                                    file_path=local_path
                                )
                        else:
                            # JSON et autres fichiers
                            self.minio.minio_client.fput_object(
                                bucket_name=self.output_bucket,
                                object_name=minio_path,
                                file_path=local_path
                            )

                self.metrics["loading_status"] = "success"
                self.metrics["loading_end"] = F.current_timestamp()
                self.logger.info(f"Chargement réussi pour la version {self.version}")

            finally:
                self.logger.info(f"Nettoyage du dossier temporaire")
                shutil.rmtree(temp_dir)

        except Exception as e:
            self.logger.error(f"Erreur de chargement: {str(e)}")
            self.metrics["loading_status"] = "failed"
            self.metrics["loading_error"] = str(e)
            raise ValueError(f"Échec du chargement: {str(e)}")

    def create_metadata(self, data_frame, train_df, test_df, validation_df):
        # Statistiques
        stats = data_frame.select(
            F.min("annee_mutation").alias("min_year"),
            F.max("annee_mutation").alias("max_year"),
            F.mean("valeur_fonciere").alias("mean_price"),
            F.min("valeur_fonciere").alias("min_price"),
            F.max("valeur_fonciere").alias("max_price"),
            F.mean("surface_reelle_bati").alias("mean_surface"),
            F.countDistinct("code_postal").alias("num_villes")
        ).collect()[0]

        # Métadonnées
        return {
            "version": self.version,
            "processed_at": str(self.metrics.get("loading_start", "unknown")),
            "input_data": {
                "bucket": self.input_bucket,
                "base_path": self.input_base_path,
                "version": self.dvf_actual_version
            },
            "reference_year": self.reference_year,
            "total_rows": self.metrics.get("rows_transformed", 0),
            "filtered_out": self.metrics.get("rows_filtered_out", 0),
            "train_rows": train_df.count(),
            "test_rows": test_df.count(),
            "validation_rows": validation_df.count(),
            "stats": {
                "year_range": f"{stats['min_year']} - {stats['max_year']}",
                "mean_price": round(stats["mean_price"], 2),
                "price_range": f"{round(stats['min_price'], 2)} - {round(stats['max_price'], 2)}",
                "mean_surface": round(stats["mean_surface"], 2),
                "code_postal": stats["num_villes"]
            },
            "features": [
                "surface_reelle_bati", "nombre_pieces_principales", "surface_terrain",
                "ratio_terrain_bati",
                "longitude", "latitude",
                "code_postal",
                "annee_mutation", "code_type_local"
            ],
            "target": "valeur_fonciere_ajustee",
            "inflation_adjustment": "Prix ajusté à l'inflation par rapport à l'année de référence"
        }

    def split_dataset(self, data_frame):
        """Divise le dataset en ensembles d'entraînement, test et validation"""
        self.logger.info("Division du dataset")

        df_with_rand = data_frame.withColumn("random", F.rand(seed=42))

        train_df = df_with_rand.filter(F.col("random") < 0.7).drop("random")
        test_df = df_with_rand.filter((F.col("random") >= 0.7) & (F.col("random") < 0.9)).drop("random")
        validation_df = df_with_rand.filter(F.col("random") >= 0.9).drop("random")

        self.logger.info(
            f"Division: Train: {train_df.count()}, Test: {test_df.count()}, Validation: {validation_df.count()}")
        return train_df, test_df, validation_df

    def update_catalog(self, temp_dir, latest_dir):
        try:
            catalog = {"versions": []}
            try:
                existing_catalog = self.minio.minio_client.get_object(
                    bucket_name=self.output_bucket,
                    object_name=f"{self.output_base_path}/catalog.json"
                )
                catalog = json.loads(existing_catalog.read().decode('utf-8'))
            except:
                self.logger.info("Création d'un nouveau catalogue")

            # Nouvelle entrée
            version_entry = {
                "version": self.version,
                "created_at": str(self.metrics.get("loading_start", "unknown")),
                "reference_year": self.reference_year,
                "total_rows": self.metrics.get("rows_transformed", 0),
                "input_data": {
                    "bucket": self.input_bucket,
                    "base_path": self.input_base_path,
                    "version": self.dvf_actual_version
                }
            }

            catalog["versions"].append(version_entry)
            catalog["versions"] = sorted(catalog["versions"],
                                         key=lambda x: x.get("created_at", ""),
                                         reverse=True)

            with open(os.path.join(temp_dir, "catalog.json"), 'w') as f:
                json.dump(catalog, f, indent=2)

            # Fichier latest
            with open(os.path.join(latest_dir, "info.json"), 'w') as f:
                json.dump({"redirect_to": self.version}, f, indent=2)

        except Exception as e:
            self.logger.warning(f"Erreur catalogue: {str(e)}")

    def add_sea_distance(self, data_frame):
        """
        Ajoute une colonne de distance à la mer pour les biens en Bretagne.
        Utilise une méthode simplifiée ne nécessitant aucune dépendance externe.
        """
        from pyspark.sql.functions import udf, substring, col
        from pyspark.sql.types import DoubleType
        import math

        self.logger.info("Calcul de la distance à la mer pour la Bretagne (méthode par points)")

        # Points de référence du littoral breton (lon, lat)
        littoral_bretagne = [
            # Côte nord (22)
            (-2.53, 48.64),  # Saint-Brieuc
            (-3.15, 48.78),  # Paimpol
            (-3.54, 48.79),  # Perros-Guirec

            # Finistère (29)
            (-3.97, 48.73),  # Roscoff
            (-4.37, 48.39),  # Brest
            (-4.10, 47.98),  # Quimper
            (-3.98, 48.02),  # Douarnenez
            (-4.49, 48.06),  # Pointe du Raz
            (-4.70, 47.87),  # Audierne

            # Morbihan (56)
            (-3.37, 47.75),  # Lorient
            (-2.92, 47.66),  # Vannes
            (-2.77, 47.50),  # Presqu'île de Rhuys
            (-3.13, 47.59),  # Quiberon
            (-3.09, 47.34),  # Belle-Île

            # Ille-et-Vilaine (35)
            (-2.03, 48.64),  # Saint-Malo
            (-1.85, 48.64),  # Cancale
            (-1.61, 48.51),  # Mont-Saint-Michel
        ]

        # Fonction pour calculer la distance minimale à la côte
        @udf(returnType=DoubleType())
        def min_distance_to_coast(lon, lat):
            """Calcule la distance minimale (en km) d'un point à la côte"""
            if lon is None or lat is None:
                return None

            min_dist = float('inf')
            for coast_lon, coast_lat in littoral_bretagne:
                # Formule de Haversine sans numpy
                R = 6371.0  # rayon de la Terre en km
                dlon = math.radians(coast_lon - lon)
                dlat = math.radians(coast_lat - lat)

                a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat)) * math.cos(
                    math.radians(coast_lat)) * math.sin(dlon / 2) ** 2
                c = 2 * math.asin(math.sqrt(a))
                distance = R * c

                min_dist = min(min_dist, distance)

            return min_dist

        # Ajouter la colonne de distance pour les biens en Bretagne
        bretagne_codes = ["22", "29", "35", "56"]
        bretagne_df = data_frame.filter(
            substring(col("code_postal"), 1, 2).isin(bretagne_codes)
        )

        non_bretagne_df = data_frame.filter(
            ~substring(col("code_postal"), 1, 2).isin(bretagne_codes)
        )

        # Calculer la distance à la mer
        bretagne_with_distance = bretagne_df.withColumn(
            "distance_mer_km", min_distance_to_coast(col("longitude"), col("latitude"))
        )

        # Ajouter une valeur NULL pour les autres biens
        non_bretagne_with_distance = non_bretagne_df.withColumn(
            "distance_mer_km", F.lit(None).cast(DoubleType())
        )

        # Combiner les résultats
        result_df = bretagne_with_distance.union(non_bretagne_with_distance)

        # Ajouter un indicateur binaire de proximité
        result_df = result_df.withColumn(
            "proche_mer",
            F.when(col("distance_mer_km").isNotNull() & (col("distance_mer_km") < 3), 1).otherwise(0)
        )

        self.logger.info(f"Distance à la mer calculée pour les biens en Bretagne")

        return result_df


if __name__ == "__main__":
    import sys
    import argparse
    from etl.common.spark import SparkManager

    parser = argparse.ArgumentParser(description='Dataset pour estimation des prix immobiliers')
    parser.add_argument('--input-bucket', type=str, default='processed')
    parser.add_argument('--input-base-path', type=str, default='dvf_enriched')
    parser.add_argument('--input-version', type=str, default='latest')
    parser.add_argument('--output-bucket', type=str, default='ml-datasets')
    parser.add_argument('--output-base-path', type=str, default='house_price_model')
    parser.add_argument('--version', type=str)
    parser.add_argument('--reference-year', type=int, default=datetime.datetime.now().year)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    logger = logging.getLogger("HousePriceDatasetCreator-Main")
    logger.info("Démarrage du processus")

    spark_manager = None
    dataset_creator = None

    try:
        spark_manager = SparkManager.get_instance()

        dataset_creator = HousePriceDatasetCreator(
            input_bucket=args.input_bucket,
            input_base_path=args.input_base_path,
            input_version=args.input_version,
            output_bucket=args.output_bucket,
            output_base_path=args.output_base_path,
            version=args.version,
            reference_year=args.reference_year
        )

        result = dataset_creator.execute()
        logger.info(f"Création réussie: {result}")
        logger.info(f"Dataset disponible: {args.output_bucket}/{dataset_creator.output_path}")
        logger.info(f"Source DVF: version {dataset_creator.dvf_actual_version}")

    except Exception as e:
        logger.error(f"Erreur: {str(e)}", exc_info=True)
        sys.exit(1)

    finally:
        if spark_manager:
            spark_manager.stop()
        logger.info("Processus terminé")
