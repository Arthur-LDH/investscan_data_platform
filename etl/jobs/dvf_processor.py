from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType
from etl.common.base_process_job import BaseETLJob
import logging
import json
import os
import io


class DVFConsolidator(BaseETLJob):
    """
    Job ETL pour la consolidation des données DVF
    - Lit les fichiers DVF collectés depuis le dossier le plus récent dans raw-data/dvf-data/
    - Nettoie et standardise les données
    - Consolide en un unique fichier parquet
    """

    def __init__(self,
                 input_bucket="raw-data",
                 input_prefix="dvf-data",
                 output_bucket="processed",
                 output_path="dvf_consolidated.parquet"):  # Modifié pour indiquer un fichier .parquet
        super().__init__("DVFConsolidator")
        self.input_bucket = input_bucket
        self.input_prefix = input_prefix
        self.output_bucket = output_bucket
        self.output_path = output_path

        # Configuration du logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)

        # Vérifier si le handler existe déjà pour éviter les doublons
        if not self.logger.handlers:
            # Créer un handler pour afficher les logs dans la console
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def find_latest_data_folder(self):
        """Trouve le dossier de données le plus récent"""
        try:
            # Utiliser la méthode list_files du connecteur MinIO
            folders = set()
            files = self.minio.list_files(self.input_bucket, prefix=f"{self.input_prefix}/")

            for file_path in files:
                if "/" in file_path[len(self.input_prefix) + 1:]:
                    folder = file_path.split("/")[1]
                    if folder:
                        folders.add(folder)

            self.logger.info(f"Dossiers datés trouvés: {folders}")

            if not folders:
                raise ValueError(f"Aucun dossier daté trouvé dans {self.input_bucket}/{self.input_prefix}/")

            # Trier les dossiers par date (format YYYYMMDD) et prendre le plus récent
            latest_folder = sorted(folders, reverse=True)[0]
            self.logger.info(f"Dossier le plus récent: {latest_folder}")

            return latest_folder

        except Exception as e:
            self.logger.error(f"Erreur lors de la recherche du dossier le plus récent: {str(e)}", exc_info=True)
            raise ValueError(f"Impossible de déterminer le dossier de données le plus récent: {str(e)}")

    def extract(self):
        """Extrait toutes les données DVF disponibles du dossier le plus récent"""
        self.metrics["extraction_start"] = F.current_timestamp()

        self.logger.info(f"Début de l'extraction des données depuis {self.input_bucket}/{self.input_prefix}")

        try:
            # Étape 1: Trouver le dossier le plus récent
            folder_date = self.find_latest_data_folder()
            self.logger.info(f"Utilisation du dossier de collecte daté du: {folder_date}")

            # Étape 2: Lister tous les fichiers CSV dans ce dossier
            file_list = self.minio.list_files(
                bucket=self.input_bucket,
                prefix=f"{self.input_prefix}/{folder_date}/",
                extension=".csv"
            )

            # Filtrer pour ne garder que les fichiers DVF
            dvf_files = [f for f in file_list if "dvf_" in f.split("/")[-1]]

            self.logger.info(f"Fichiers CSV à traiter: {len(dvf_files)}")
            for f in dvf_files:
                self.logger.info(f"- {f}")

            if not dvf_files:
                raise ValueError(
                    f"Aucun fichier DVF trouvé dans {self.input_bucket}/{self.input_prefix}/{folder_date}/")

            # Étape 3: Lire et fusionner tous les fichiers CSV
            dataframes = []

            for file_path in dvf_files:
                try:
                    # Extraire le nom du fichier et les informations (département, année)
                    file_name = file_path.split('/')[-1]
                    # Format attendu: dvf_DD_YYYY.csv
                    if file_name.startswith('dvf_') and file_name.endswith('.csv'):
                        parts = file_name.replace('.csv', '').split('_')
                        if len(parts) >= 3:
                            dept, year = parts[1], parts[2]

                            # Lire le fichier avec le connecteur MinIO
                            df = self.minio.read(
                                bucket=self.input_bucket,
                                path=file_path,
                                options={
                                    "header": "true",
                                    "inferSchema": "true",
                                    "multiLine": "true",
                                    "escape": "\"",
                                    "quote": "\"",
                                    "header": "true",
                                    "nullValue": ""
                                }
                            )

                            # Ajouter des colonnes pour le département et l'année source
                            df = df.withColumn("source_departement", F.lit(dept))
                            df = df.withColumn("source_annee", F.lit(year))

                            # Repartitionner pour éviter les tâches trop grandes
                            df = df.repartition(10)

                            dataframes.append(df)
                            self.logger.info(f"Fichier {file_path} lu avec succès: {df.count()} lignes")
                except Exception as e:
                    self.logger.error(f"Erreur lors de la lecture du fichier {file_path}: {str(e)}", exc_info=True)

            if not dataframes:
                raise ValueError(f"Aucun fichier n'a pu être lu correctement depuis {folder_date}")

            # Union de tous les dataframes
            raw_df = dataframes[0]
            for df in dataframes[1:]:
                # S'assurer que les schémas sont compatibles
                raw_df = raw_df.unionByName(df, allowMissingColumns=True)

            # Enregistrer les métriques
            self.metrics["rows_raw"] = raw_df.count()
            self.metrics["files_count"] = len(dataframes)
            self.metrics["collection_date"] = folder_date

            self.logger.info(f"Total des lignes après union: {raw_df.count()}")
            self.metrics["extraction_end"] = F.current_timestamp()

            return raw_df

        except Exception as e:
            self.logger.error(f"Erreur lors de l'extraction: {str(e)}", exc_info=True)
            raise ValueError(f"Échec de l'extraction des données DVF: {str(e)}")

    def transform(self, data_frame):
        """Nettoie et standardise les données DVF"""
        self.metrics["transformation_start"] = F.current_timestamp()
        self.logger.info("Début de la transformation des données")

        try:
            # 0. Repartitionner pour éviter les tâches trop grandes
            data_frame = data_frame.repartition(50)

            # 1. Conversion des types de données
            self.logger.info("Étape 1: Conversion des types de données")
            typed_df = (data_frame
                        .withColumn("id_mutation", F.col("id_mutation").cast(StringType()))
                        .withColumn("date_mutation", F.to_date(F.col("date_mutation"), "yyyy-MM-dd"))
                        .withColumn("valeur_fonciere", F.col("valeur_fonciere").cast(DoubleType()))
                        .withColumn("surface_reelle_bati", F.col("surface_reelle_bati").cast(DoubleType()))
                        .withColumn("nombre_pieces_principales", F.col("nombre_pieces_principales").cast(IntegerType()))
                        .withColumn("surface_terrain", F.col("surface_terrain").cast(DoubleType()))
                        .withColumn("longitude", F.col("longitude").cast(DoubleType()))
                        .withColumn("latitude", F.col("latitude").cast(DoubleType()))
                        )
            self.logger.info(f"Conversion des types terminée: {typed_df.count()} lignes")

            # 2. Standardisation des chaînes de caractères
            self.logger.info("Étape 2: Standardisation des chaînes")
            cleaned_df = (typed_df
                          .withColumn("type_local", F.when(F.col("type_local").isNotNull(),
                                                           F.lower(F.trim(F.col("type_local")))).otherwise(F.lit(None)))
                          .withColumn("nature_culture", F.when(F.col("nature_culture").isNotNull(),
                                                               F.lower(F.trim(F.col("nature_culture")))).otherwise(
                F.lit(None)))
                          .withColumn("nom_commune", F.when(F.col("nom_commune").isNotNull(),
                                                            F.initcap(F.trim(F.col("nom_commune")))).otherwise(
                F.lit(None)))
                          )
            self.logger.info(f"Standardisation terminée: {cleaned_df.count()} lignes")

            # 3. Élimination des valeurs aberrantes
            self.logger.info("Étape 3: Filtrage des valeurs aberrantes")
            filtered_df = (cleaned_df
                           # Filtrer les transactions avec valeur foncière nulle ou négative
                           .filter(F.col("valeur_fonciere").isNotNull() & (F.col("valeur_fonciere") > 0))
                           # Filtrer les transactions sans coordonnées géographiques valides
                           .filter((F.col("longitude").isNotNull() & F.col("latitude").isNotNull()) |
                                   (F.col("code_postal").isNotNull() & F.col("nom_commune").isNotNull()))
                           # Filtrer les transactions avec des valeurs aberrantes de surface (au cas par cas)
                           .filter((F.col("surface_reelle_bati").isNull()) | (F.col("surface_reelle_bati") > 0))
                           .filter((F.col("surface_terrain").isNull()) | (F.col("surface_terrain") > 0))
                           )
            self.logger.info(
                f"Filtrage terminé: {filtered_df.count()} lignes (suppression de {cleaned_df.count() - filtered_df.count()} lignes)")

            # 4. Calcul de métriques dérivées
            self.logger.info("Étape 4: Calcul des métriques dérivées")
            enriched_df = (filtered_df
                           # Prix au m² pour le bâti
                           .withColumn("prix_m2_bati",
                                       F.when(F.col("surface_reelle_bati").isNotNull() & (
                                               F.col("surface_reelle_bati") > 0),
                                              F.col("valeur_fonciere") / F.col("surface_reelle_bati")).otherwise(
                                           F.lit(None)))
                           # Prix au m² pour le terrain
                           .withColumn("prix_m2_terrain",
                                       F.when(F.col("surface_terrain").isNotNull() & (F.col("surface_terrain") > 0),
                                              F.col("valeur_fonciere") / F.col("surface_terrain")).otherwise(
                                           F.lit(None)))
                           # Année de vente
                           .withColumn("annee_mutation", F.year(F.col("date_mutation")))
                           # Mois de vente
                           .withColumn("mois_mutation", F.month(F.col("date_mutation")))
                           # Trim des colonnes de code pour éviter les espaces
                           .withColumn("code_postal", F.trim(F.col("code_postal")))
                           .withColumn("code_departement", F.trim(F.col("code_departement")))
                           )
            self.logger.info(f"Enrichissement terminé: {enriched_df.count()} lignes")

            # Cache pour optimiser la performance
            enriched_df = enriched_df.cache()

            # 5. Filtrage supplémentaire des valeurs aberrantes sur les prix au m²
            self.logger.info("Étape 5: Filtrage des prix aberrants")
            final_df = (enriched_df
                        .filter((F.col("prix_m2_bati").isNull()) |
                                (F.col("prix_m2_bati").between(50, 25000)))  # Plage plus large
                        .filter((F.col("prix_m2_terrain").isNull()) |
                                (F.col("prix_m2_terrain").between(0.1, 5000)))  # Plage plus large
                        )
            self.logger.info(
                f"Filtrage des prix terminé: {final_df.count()} lignes (suppression de {enriched_df.count() - final_df.count()} lignes)")

            # Collecte des métriques
            self.metrics["rows_transformed"] = final_df.count()
            self.metrics["rows_filtered_out"] = self.metrics["rows_raw"] - final_df.count()
            self.metrics["transformation_end"] = F.current_timestamp()

            self.logger.info(f"Transformation terminée: {final_df.count()} lignes après nettoyage")
            return final_df

        except Exception as e:
            self.logger.error(f"Erreur lors de la transformation: {str(e)}", exc_info=True)
            raise ValueError(f"Échec de la transformation des données DVF: {str(e)}")

    def load(self, data_frame):
        """Sauvegarde les données consolidées en un fichier parquet unique"""
        self.metrics["loading_start"] = F.current_timestamp()
        self.logger.info(f"Début du chargement des données vers {self.output_bucket}/{self.output_path}")

        try:
            # Assurer que le bucket de sortie existe
            self.minio.ensure_bucket_exists(self.output_bucket)
            self.logger.info(f"Le bucket {self.output_bucket} est prêt pour le chargement des données")

            # Coalesce pour créer un seul fichier parquet
            # La valeur optimale dépend de la taille de vos données. Une valeur trop
            # basse pourrait causer des problèmes de mémoire.
            coalesce_num = 1  # Pour un seul fichier

            # Si votre dataset est très volumineux (>1GB), vous pourriez avoir besoin d'ajuster
            # self.logger.info(f"Coalescence du DataFrame en {coalesce_num} partition(s)")
            # df_single = data_frame.coalesce(coalesce_num)

            # Écrire directement en un seul fichier parquet
            self.logger.info(f"Écriture d'un fichier parquet unique: {self.output_path}")

            # Utiliser coalesce(1) pour forcer un seul fichier de sortie
            self.minio.write(
                df=data_frame.coalesce(1),
                bucket=self.output_bucket,
                path=self.output_path,
                format="parquet",
                mode="overwrite"
            )

            self.logger.info(f"Données sauvegardées avec succès dans {self.output_bucket}/{self.output_path}")
            self.metrics["loading_status"] = "success"

            # Sauvegarder aussi un fichier de métadonnées
            try:
                self.logger.info("Création du fichier de métadonnées")
                metadata = {
                    "processed_at": str(self.metrics.get("loading_start", "unknown")),
                    "source_folder": self.metrics.get("collection_date", "unknown"),
                    "total_files_processed": self.metrics.get("files_count", 0),
                    "total_rows": self.metrics.get("rows_transformed", 0),
                    "filtered_out": self.metrics.get("rows_filtered_out", 0),
                    "departments": [row.code_departement for row in
                                    data_frame.select("code_departement").distinct().collect()],
                    "years": [row.annee_mutation for row in data_frame.select("annee_mutation").distinct().collect()],
                    "file_path": self.output_path  # Ajout du chemin du fichier parquet
                }

                # Convertir en JSON et sauvegarder directement via MinIO client
                metadata_json = json.dumps(metadata, indent=2)
                metadata_path = f"{os.path.splitext(self.output_path)[0]}_metadata.json"

                self.minio.minio_client.put_object(
                    bucket_name=self.output_bucket,
                    object_name=metadata_path,
                    data=io.BytesIO(metadata_json.encode()),
                    length=len(metadata_json),
                    content_type="application/json"
                )

                self.logger.info(f"Métadonnées sauvegardées dans {self.output_bucket}/{metadata_path}")

            except Exception as e:
                self.logger.warning(f"Erreur lors de la sauvegarde des métadonnées: {str(e)}")
                # Ne pas échouer l'ensemble du job pour les métadonnées

        except Exception as e:
            self.logger.error(f"Erreur lors du chargement: {str(e)}", exc_info=True)
            self.metrics["loading_status"] = "failed"
            self.metrics["loading_error"] = str(e)
            raise ValueError(f"Échec du chargement des données DVF: {str(e)}")

        self.metrics["loading_end"] = F.current_timestamp()
        self.logger.info("Chargement terminé avec succès")


if __name__ == "__main__":
    import sys
    from etl.common.spark import SparkManager

    # Configurer le logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    logger = logging.getLogger("DVFConsolidator-Main")
    logger.info("Démarrage du processus de consolidation DVF")

    spark_manager = None
    consolidator = None

    try:
        # Initialiser Spark
        logger.info("Initialisation de Spark")
        spark_manager = SparkManager.get_instance()

        # Créer et exécuter le consolidateur
        logger.info("Création du consolidateur")
        consolidator = DVFConsolidator(
            input_bucket="raw-data",
            input_prefix="dvf-data",  # Sans trailing slash
            output_bucket="processed",
            output_path="dvf_consolidated.parquet"  # Modifié pour spécifier un fichier parquet unique
        )

        logger.info("Exécution du job de consolidation")
        result = consolidator.execute()
        logger.info(f"Consolidation terminée avec succès: {result}")

    except Exception as e:
        logger.error(f"Erreur lors de la consolidation: {str(e)}", exc_info=True)
        sys.exit(1)

    finally:
        # S'assurer que les ressources sont bien libérées
        logger.info("Nettoyage des ressources")
        if spark_manager:
            spark_manager.stop()
        logger.info("Processus terminé")