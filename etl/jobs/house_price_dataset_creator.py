from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType
from etl.common.base_process_job import BaseETLJob
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
import logging
import json
import io
import datetime


class HousePriceDatasetCreator(BaseETLJob):
    """
    Job ETL pour la création d'un dataset d'apprentissage pour l'estimation des prix immobiliers
    - Filtre les données pour ne garder que les ventes de maisons
    - Nettoie et prépare les caractéristiques pertinentes
    - Calcule et intègre des facteurs d'inflation du marché immobilier
    """

    def __init__(self,
                 input_bucket="processed",
                 input_base_path="dvf_consolidated",
                 input_version="latest",  # Par défaut, utiliser la dernière version
                 output_bucket="ml-datasets",
                 output_base_path="house_price_model",
                 version=None,
                 reference_year=None):  # Année de référence pour l'inflation (par défaut: année courante)
        super().__init__("HousePriceDatasetCreator")
        self.input_bucket = input_bucket
        self.input_base_path = input_base_path
        self.input_version = input_version
        self.output_bucket = output_bucket

        # Gestion du versionnement des datasets
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        if version is None:
            version = f"v_{current_date}"
        self.version = version
        self.output_base_path = output_base_path
        self.output_path = f"{output_base_path}/{self.version}"

        # Si aucune année de référence n'est fournie, utiliser l'année courante
        self.reference_year = reference_year if reference_year else datetime.datetime.now().year

        # La version des données DVF sera déterminée lors de l'extraction
        self.dvf_actual_version = None

        # Configuration du logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)

        # Vérifier si le handler existe déjà pour éviter les doublons
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def resolve_dvf_version(self):
        """Résout la version réelle des données DVF à utiliser"""
        try:
            if self.input_version == "latest":
                # Lire le fichier info.json dans le dossier latest
                latest_info_path = f"{self.input_base_path}/latest/info.json"
                try:
                    latest_info = self.minio.minio_client.get_object(
                        bucket_name=self.input_bucket,
                        object_name=latest_info_path
                    )
                    latest_info_data = json.loads(latest_info.read().decode('utf-8'))
                    actual_version = latest_info_data.get("redirect_to")
                    if not actual_version:
                        raise ValueError("Le fichier info.json ne contient pas la clé 'redirect_to'")

                    self.logger.info(f"Résolution de 'latest' vers la version réelle: {actual_version}")
                    return actual_version

                except Exception as e:
                    self.logger.error(f"Erreur lors de la lecture du fichier latest/info.json: {str(e)}")

                    # Fallback: essayer de lire le catalogue et prendre la version la plus récente
                    self.logger.info("Tentative de récupération via le catalogue...")
                    catalog_path = f"{self.input_base_path}/catalog.json"
                    catalog = self.minio.minio_client.get_object(
                        bucket_name=self.input_bucket,
                        object_name=catalog_path
                    )
                    catalog_data = json.loads(catalog.read().decode('utf-8'))

                    if not catalog_data.get("versions"):
                        raise ValueError("Le catalogue ne contient pas de versions")

                    # Trier par date de création (supposée être dans un format comparable)
                    versions = sorted(catalog_data["versions"],
                                      key=lambda x: x.get("created_at", ""),
                                      reverse=True)
                    actual_version = versions[0]["version"]
                    self.logger.info(f"Version la plus récente dans le catalogue: {actual_version}")
                    return actual_version

            # Si ce n'est pas "latest", utiliser directement la version spécifiée
            self.logger.info(f"Utilisation de la version DVF spécifiée: {self.input_version}")
            return self.input_version

        except Exception as e:
            self.logger.error(f"Erreur lors de la résolution de la version DVF: {str(e)}", exc_info=True)
            raise ValueError(f"Impossible de déterminer la version DVF à utiliser: {str(e)}")

    def extract(self):
        """Extrait les données DVF consolidées de la version spécifiée"""
        self.metrics["extraction_start"] = F.current_timestamp()

        # Résoudre la version réelle à utiliser
        self.dvf_actual_version = self.resolve_dvf_version()
        input_path = f"{self.input_base_path}/{self.dvf_actual_version}/data.parquet"

        self.logger.info(
            f"Début de l'extraction des données depuis {self.input_bucket}/{input_path} (version: {self.dvf_actual_version})")

        try:
            # Lire le fichier parquet consolidé
            dvf_data = self.minio.read(
                bucket=self.input_bucket,
                path=input_path,
                format="parquet"
            )

            self.logger.info(f"Données DVF chargées avec succès: {dvf_data.count()} lignes")
            self.metrics["rows_raw"] = dvf_data.count()
            self.metrics["dvf_version"] = self.dvf_actual_version
            self.metrics["extraction_end"] = F.current_timestamp()

            return dvf_data

        except Exception as e:
            self.logger.error(f"Erreur lors de l'extraction depuis la version {self.dvf_actual_version}: {str(e)}",
                              exc_info=True)
            raise ValueError(f"Échec de l'extraction des données DVF: {str(e)}")

    def transform(self, data_frame):
        """
        Prépare les données pour le modèle d'estimation des prix immobiliers
        - Filtre les données (ventes de maisons uniquement)
        - Sélectionne et nettoie les caractéristiques pertinentes
        - Calcule les facteurs d'inflation du marché
        """
        self.metrics["transformation_start"] = F.current_timestamp()
        self.logger.info("Début de la transformation des données")

        try:
            # 1. Filtrer pour ne garder que les ventes de maisons
            self.logger.info("Étape 1: Filtrage des ventes de maisons")
            houses_df = data_frame.filter(
                (F.col("nature_mutation").ilike("%vente%")) &
                (F.col("type_local") == "maison")
            )

            self.logger.info(f"Filtrage terminé: {houses_df.count()} ventes de maisons")

            # 2. Sélectionner les caractéristiques pertinentes et éliminer les valeurs manquantes
            self.logger.info("Étape 2: Sélection des caractéristiques et nettoyage")

            selected_df = houses_df.select(
                "id_mutation",
                "date_mutation",
                "annee_mutation",
                "mois_mutation",
                "valeur_fonciere",
                "code_departement",
                "code_postal",
                "nom_commune",
                "surface_reelle_bati",
                "nombre_pieces_principales",
                "surface_terrain",
                "longitude",
                "latitude"
            ).filter(
                # Filtrer les lignes avec valeurs manquantes sur des caractéristiques critiques
                F.col("valeur_fonciere").isNotNull() &
                F.col("surface_reelle_bati").isNotNull() &
                F.col("nombre_pieces_principales").isNotNull() &
                (F.col("valeur_fonciere") > 10000) &  # Éliminer les prix aberrants trop bas
                (F.col("surface_reelle_bati") > 10)  # Éliminer les surfaces aberrantes trop petites
            )

            self.logger.info(f"Sélection et nettoyage terminés: {selected_df.count()} lignes valides")

            # 3. Calculer les indicateurs dérivés
            self.logger.info("Étape 3: Calcul des indicateurs dérivés")

            enriched_df = selected_df.withColumn(
                "prix_m2", F.col("valeur_fonciere") / F.col("surface_reelle_bati")
            ).withColumn(
                "ratio_terrain_bati",
                F.when(F.col("surface_terrain").isNotNull() & (F.col("surface_terrain") > 0) & (
                            F.col("surface_reelle_bati") > 0),
                       F.col("surface_terrain") / F.col("surface_reelle_bati")).otherwise(0)
            )

            # 4. Calculer le facteur d'inflation du marché immobilier
            self.logger.info("Étape 4: Calcul de l'inflation du marché immobilier")

            # Calculer le prix moyen au m² par année et département
            price_index = enriched_df.groupBy("annee_mutation", "code_departement").agg(
                F.avg("prix_m2").alias("avg_price_m2"),
                F.count("*").alias("transaction_count")
            )

            # Filtrer pour n'avoir que les années avec suffisamment de transactions
            price_index = price_index.filter(F.col("transaction_count") >= 30)

            # Créer une vue temporaire pour faciliter les requêtes SQL
            price_index.createOrReplaceTempView("price_index")

            # Calculer les facteurs d'inflation par année et département par rapport à l'année de référence
            inflation_query = f"""
            WITH ref_prices AS (
                SELECT 
                    code_departement,
                    AVG(avg_price_m2) AS ref_price
                FROM price_index
                WHERE annee_mutation = (
                    SELECT MAX(annee_mutation) 
                    FROM price_index 
                    WHERE annee_mutation <= {self.reference_year}
                )
                GROUP BY code_departement
            )
            SELECT 
                p.annee_mutation,
                p.code_departement,
                p.avg_price_m2,
                r.ref_price,
                (r.ref_price / p.avg_price_m2) AS inflation_factor
            FROM price_index p
            JOIN ref_prices r ON p.code_departement = r.code_departement
            """

            inflation_factors = self.spark.sql(inflation_query)
            self.logger.info(
                f"Facteurs d'inflation calculés pour {inflation_factors.count()} combinaisons année/département")

            # 5. Joindre les facteurs d'inflation au dataset principal
            self.logger.info("Étape 5: Application des facteurs d'inflation")

            # Joindre les facteurs d'inflation
            final_df = enriched_df.join(
                inflation_factors.select("annee_mutation", "code_departement", "inflation_factor"),
                on=["annee_mutation", "code_departement"],
                how="left"
            )

            # Pour les lignes sans facteur d'inflation (départements avec peu de données), utiliser 1.0
            final_df = final_df.withColumn(
                "inflation_factor",
                F.coalesce(F.col("inflation_factor"), F.lit(1.0))
            )

            # Calculer le prix ajusté à l'inflation (valeur actuelle)
            final_df = final_df.withColumn(
                "valeur_fonciere_ajustee",
                F.col("valeur_fonciere") * F.col("inflation_factor")
            )

            # 6. Préparer les caractéristiques finales
            self.logger.info("Étape 6: Préparation des caractéristiques finales")

            model_df = final_df.select(
                "id_mutation",
                "date_mutation",
                "annee_mutation",
                "code_departement",
                "code_postal",
                "nom_commune",
                "surface_reelle_bati",
                "nombre_pieces_principales",
                "surface_terrain",
                "ratio_terrain_bati",
                "longitude",
                "latitude",
                "valeur_fonciere",
                "valeur_fonciere_ajustee",
                "inflation_factor"
            )

            # Collecte des métriques
            self.metrics["rows_transformed"] = model_df.count()
            self.metrics["rows_filtered_out"] = self.metrics["rows_raw"] - model_df.count()
            self.metrics["reference_year"] = self.reference_year
            self.metrics["transformation_end"] = F.current_timestamp()

            self.logger.info(f"Transformation terminée: {model_df.count()} lignes dans le dataset final")
            return model_df

        except Exception as e:
            self.logger.error(f"Erreur lors de la transformation: {str(e)}", exc_info=True)
            raise ValueError(f"Échec de la transformation des données: {str(e)}")

    def load(self, data_frame):
        """Sauvegarde le dataset d'apprentissage et les métadonnées avec versionnement"""
        self.metrics["loading_start"] = F.current_timestamp()
        self.logger.info(
            f"Début du chargement des données vers {self.output_bucket}/{self.output_path} (version: {self.version})")

        try:
            # Assurer que le bucket de sortie existe
            self.minio.ensure_bucket_exists(self.output_bucket)

            # Partition du dataset pour l'entraînement et les tests
            train_df, test_df, validation_df = self.split_dataset(data_frame)

            # Sauvegarde du dataset complet
            self.logger.info("Sauvegarde du dataset complet")
            self.minio.write(
                df=data_frame.coalesce(1),
                bucket=self.output_bucket,
                path=f"{self.output_path}/full_dataset.parquet",
                format="parquet",
                mode="overwrite"
            )

            # Sauvegarde du jeu d'entraînement
            self.logger.info("Sauvegarde du jeu d'entraînement")
            self.minio.write(
                df=train_df.coalesce(1),
                bucket=self.output_bucket,
                path=f"{self.output_path}/train.parquet",
                format="parquet",
                mode="overwrite"
            )

            # Sauvegarde du jeu de test
            self.logger.info("Sauvegarde du jeu de test")
            self.minio.write(
                df=test_df.coalesce(1),
                bucket=self.output_bucket,
                path=f"{self.output_path}/test.parquet",
                format="parquet",
                mode="overwrite"
            )

            # Sauvegarde du jeu de validation
            self.logger.info("Sauvegarde du jeu de validation")
            self.minio.write(
                df=validation_df.coalesce(1),
                bucket=self.output_bucket,
                path=f"{self.output_path}/validation.parquet",
                format="parquet",
                mode="overwrite"
            )

            # Sauvegarde des facteurs d'inflation
            self.logger.info("Sauvegarde des facteurs d'inflation")
            inflation_factors = data_frame.select(
                "annee_mutation", "code_departement", "inflation_factor"
            ).distinct()

            self.minio.write(
                df=inflation_factors.coalesce(1),
                bucket=self.output_bucket,
                path=f"{self.output_path}/inflation_factors.parquet",
                format="parquet",
                mode="overwrite"
            )

            # Sauvegarde des métadonnées
            try:
                self.logger.info("Création du fichier de métadonnées")

                # Collecter des statistiques pour les métadonnées
                stats = data_frame.select(
                    F.min("annee_mutation").alias("min_year"),
                    F.max("annee_mutation").alias("max_year"),
                    F.mean("valeur_fonciere").alias("mean_price"),
                    F.min("valeur_fonciere").alias("min_price"),
                    F.max("valeur_fonciere").alias("max_price"),
                    F.mean("surface_reelle_bati").alias("mean_surface"),
                    F.countDistinct("code_departement").alias("num_departments")
                ).collect()[0]

                metadata = {
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
                        "departments_count": stats["num_departments"]
                    },
                    "features": [
                        "surface_reelle_bati",
                        "nombre_pieces_principales",
                        "surface_terrain",
                        "ratio_terrain_bati",
                        "longitude",
                        "latitude",
                        "code_departement (encodé)",
                        "annee_mutation"
                    ],
                    "target": "valeur_fonciere_ajustee",
                    "inflation_adjustment": "La cible 'valeur_fonciere_ajustee' représente le prix ajusté à l'inflation par rapport à l'année de référence"
                }

                # Convertir en JSON et sauvegarder
                metadata_json = json.dumps(metadata, indent=2)
                metadata_path = f"{self.output_path}/metadata.json"

                self.minio.minio_client.put_object(
                    bucket_name=self.output_bucket,
                    object_name=metadata_path,
                    data=io.BytesIO(metadata_json.encode()),
                    length=len(metadata_json),
                    content_type="application/json"
                )

                self.logger.info(f"Métadonnées sauvegardées dans {self.output_bucket}/{metadata_path}")

                # Créer un fichier catalog.json dans le dossier base pour répertorier toutes les versions
                try:
                    self.logger.info("Mise à jour du catalogue des versions")

                    # Structure du catalogue
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

                    # Essayer de lire le catalogue existant
                    catalog_path = f"{self.output_base_path}/catalog.json"
                    catalog = {"versions": []}

                    try:
                        existing_catalog = self.minio.minio_client.get_object(
                            bucket_name=self.output_bucket,
                            object_name=catalog_path
                        )
                        catalog = json.loads(existing_catalog.read().decode('utf-8'))
                    except Exception as e:
                        self.logger.info(f"Aucun catalogue existant trouvé, création d'un nouveau: {str(e)}")

                    # Ajouter la nouvelle version
                    catalog["versions"].append(version_entry)
                    # Trier par date de création décroissante
                    catalog["versions"] = sorted(catalog["versions"],
                                                 key=lambda x: x.get("created_at", ""),
                                                 reverse=True)

                    # Enregistrer le catalogue mis à jour
                    catalog_json = json.dumps(catalog, indent=2)
                    self.minio.minio_client.put_object(
                        bucket_name=self.output_bucket,
                        object_name=catalog_path,
                        data=io.BytesIO(catalog_json.encode()),
                        length=len(catalog_json),
                        content_type="application/json"
                    )

                    # Créer un lien symbolique "latest" vers la version la plus récente
                    latest_path = f"{self.output_base_path}/latest"
                    latest_json = json.dumps({"redirect_to": self.version}, indent=2)
                    self.minio.minio_client.put_object(
                        bucket_name=self.output_bucket,
                        object_name=f"{latest_path}/info.json",
                        data=io.BytesIO(latest_json.encode()),
                        length=len(latest_json),
                        content_type="application/json"
                    )

                    self.logger.info(f"Catalogue mis à jour dans {self.output_bucket}/{catalog_path}")

                except Exception as e:
                    self.logger.warning(f"Erreur lors de la mise à jour du catalogue: {str(e)}")

                self.metrics["loading_status"] = "success"
                self.metrics["loading_end"] = F.current_timestamp()
                self.logger.info(f"Chargement terminé avec succès pour la version {self.version}")

            except Exception as e:
                self.logger.warning(f"Erreur lors de la sauvegarde des métadonnées: {str(e)}")
                # Ne pas échouer l'ensemble du job pour les métadonnées

        except Exception as e:
            self.logger.error(f"Erreur lors du chargement: {str(e)}", exc_info=True)
            self.metrics["loading_status"] = "failed"
            self.metrics["loading_error"] = str(e)
            raise ValueError(f"Échec du chargement des données: {str(e)}")

    def split_dataset(self, data_frame):
        """
        Divise le dataset en ensembles d'entraînement, test et validation
        Train: 70%, Test: 20%, Validation: 10%
        """
        self.logger.info("Division du dataset en ensembles d'entraînement, test et validation")

        # Ajouter une colonne avec un nombre aléatoire entre 0 et 1
        df_with_rand = data_frame.withColumn("random", F.rand(seed=42))

        # Diviser le dataset
        train_df = df_with_rand.filter(F.col("random") < 0.7).drop("random")
        test_df = df_with_rand.filter((F.col("random") >= 0.7) & (F.col("random") < 0.9)).drop("random")
        validation_df = df_with_rand.filter(F.col("random") >= 0.9).drop("random")

        self.logger.info(
            f"Division terminée: Train: {train_df.count()}, Test: {test_df.count()}, Validation: {validation_df.count()}")

        return train_df, test_df, validation_df


if __name__ == "__main__":
    import sys
    import argparse
    from etl.common.spark import SparkManager

    # Configurer les arguments en ligne de commande
    parser = argparse.ArgumentParser(description='Créer un dataset versionné pour l\'estimation des prix immobiliers')
    parser.add_argument('--input-bucket', type=str, default='processed',
                        help='Bucket contenant les données DVF consolidées')
    parser.add_argument('--input-base-path', type=str, default='dvf_consolidated',
                        help='Dossier de base des données DVF')
    parser.add_argument('--input-version', type=str, default='latest',
                        help='Version des données DVF à utiliser (par défaut: latest)')
    parser.add_argument('--output-bucket', type=str, default='ml-datasets', help='Bucket de destination')
    parser.add_argument('--output-base-path', type=str, default='house_price_model',
                        help='Dossier de base pour les datasets versionnés')
    parser.add_argument('--version', type=str, help='Version du dataset (par défaut: v_YYYYMMDD)')
    parser.add_argument('--reference-year', type=int, default=datetime.datetime.now().year,
                        help='Année de référence pour l\'inflation')

    args = parser.parse_args()

    # Configurer le logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    logger = logging.getLogger("HousePriceDatasetCreator-Main")
    logger.info("Démarrage du processus de création du dataset pour l'estimation des prix immobiliers")
    logger.info(
        f"Arguments: input={args.input_bucket}/{args.input_base_path}/{args.input_version}, output={args.output_bucket}/{args.output_base_path}, version={args.version}, ref_year={args.reference_year}")

    spark_manager = None
    dataset_creator = None

    try:
        # Initialiser Spark
        logger.info("Initialisation de Spark")
        spark_manager = SparkManager.get_instance()

        # Créer et exécuter le job
        logger.info("Création du job de dataset")
        dataset_creator = HousePriceDatasetCreator(
            input_bucket=args.input_bucket,
            input_base_path=args.input_base_path,
            input_version=args.input_version,
            output_bucket=args.output_bucket,
            output_base_path=args.output_base_path,
            version=args.version,
            reference_year=args.reference_year
        )

        logger.info(f"Exécution du job de création de dataset (version: {dataset_creator.version})")
        result = dataset_creator.execute()
        logger.info(f"Création de dataset terminée avec succès: {result}")
        logger.info(f"Dataset disponible dans: {args.output_bucket}/{dataset_creator.output_path}")
        logger.info(f"Source DVF utilisée: version {dataset_creator.dvf_actual_version}")

    except Exception as e:
        logger.error(f"Erreur lors de la création du dataset: {str(e)}", exc_info=True)
        sys.exit(1)

    finally:
        # S'assurer que les ressources sont bien libérées
        logger.info("Nettoyage des ressources")
        if spark_manager:
            spark_manager.stop()
        logger.info("Processus terminé")