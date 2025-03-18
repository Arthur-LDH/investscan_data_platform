from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructType, StructField
from etl.common.base_process_job import BaseETLJob
import geopandas as gpd
import pandas as pd
import os
import tempfile
from shapely.geometry import Point
import datetime
import numpy as np
import json
import io


class DVFSocioEconomicEnrichment(BaseETLJob):
    """
    Job ETL pour enrichir les données DVF avec les données socio-économiques IRIS
    """

    def __init__(
            self,
            dvf_bucket="processed",
            dvf_base_path="dvf_consolidated",
            dvf_version=None,
            iris_contours_bucket="raw-data",
            iris_contours_prefix="iris-contours",
            filo_iris_bucket="raw-data",
            filo_iris_prefix="base-td-file-iris",
            output_bucket="processed",
            output_base_path="dvf_enriched",
            output_version=None,
    ):
        super().__init__("DVFSocioEconomicEnrichment")

        # Configuration sources/destinations
        self.dvf_bucket = dvf_bucket
        self.dvf_base_path = dvf_base_path
        self.dvf_version = dvf_version
        self.iris_contours_bucket = iris_contours_bucket
        self.iris_contours_prefix = iris_contours_prefix
        self.filo_iris_bucket = filo_iris_bucket
        self.filo_iris_prefix = filo_iris_prefix
        self.output_bucket = output_bucket
        self.output_base_path = output_base_path

        # Génération de version
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        if output_version is None:
            output_version = f"v_{current_date}"
        self.output_version = output_version
        self.output_path = f"{output_base_path}/{self.output_version}/data.parquet"

        # Dossier temporaire
        self.temp_dir = tempfile.mkdtemp()

    def _find_latest_folder(self, bucket, prefix):
        """Trouve le dossier le plus récent"""
        folders = set()
        files = self.minio.list_files(bucket, prefix=f"{prefix}/")

        for file_path in files:
            if "/" in file_path[len(prefix) + 1:]:
                folder = file_path.split("/")[1]
                if folder:
                    folders.add(folder)

        if not folders:
            raise ValueError(f"Aucun dossier trouvé dans {bucket}/{prefix}/")

        return sorted(folders, reverse=True)[0]

    def _find_latest_dvf_version(self):
        """Trouve la dernière version des données DVF"""
        if self.dvf_version:
            return self.dvf_version

        try:
            # Lire le catalogue
            catalog_path = f"{self.dvf_base_path}/catalog.json"
            catalog_content = self.minio.minio_client.get_object(
                bucket_name=self.dvf_bucket,
                object_name=catalog_path
            ).read().decode('utf-8')

            catalog = json.loads(catalog_content)
            if catalog and "versions" in catalog and catalog["versions"]:
                return catalog["versions"][0]["version"]
        except:
            pass

        # Si on ne peut pas lire le catalogue, utiliser la méthode des dossiers
        versions = set()
        files = self.minio.list_files(self.dvf_bucket, prefix=f"{self.dvf_base_path}/")

        for file_path in files:
            path_parts = file_path.split("/")
            if len(path_parts) > 2:
                version = path_parts[1]
                if version != "catalog.json" and version != "latest":
                    versions.add(version)

        if not versions:
            raise ValueError(f"Aucune version DVF trouvée")

        return sorted(versions, reverse=True)[0]

    def extract(self):
        """Extrait les données des trois sources"""
        # 1. Extraction des données DVF
        dvf_version = self._find_latest_dvf_version()
        dvf_path = f"{self.dvf_base_path}/{dvf_version}/data.parquet"
        dvf_df = self.minio.read(
            bucket=self.dvf_bucket,
            path=dvf_path,
            format="parquet"
        )

        # Filtrer sur les coordonnées valides
        dvf_df_filtered = dvf_df.filter(
            F.col("longitude").isNotNull() &
            F.col("latitude").isNotNull()
        )

        # 2. Extraction des contours IRIS
        iris_contours_folder = self._find_latest_folder(
            self.iris_contours_bucket,
            self.iris_contours_prefix
        )

        # Déterminer le format du fichier
        try:
            metadata_path = f"{self.iris_contours_prefix}/{iris_contours_folder}/metadata.json"
            metadata_content = self.minio.minio_client.get_object(
                bucket_name=self.iris_contours_bucket,
                object_name=metadata_path
            ).read().decode('utf-8')
            metadata = json.loads(metadata_content)
            file_format = metadata.get("format", "gpkg")
        except:
            file_format = "gpkg"

        # Télécharger le fichier des contours IRIS
        contours_file_path = f"{self.iris_contours_prefix}/{iris_contours_folder}/iris_contours.{file_format}"
        local_contours_path = os.path.join(self.temp_dir, f"iris_contours.{file_format}")

        contours_data = self.minio.minio_client.get_object(
            bucket_name=self.iris_contours_bucket,
            object_name=contours_file_path
        ).read()

        with open(local_contours_path, 'wb') as f:
            f.write(contours_data)

        # 3. Extraction des données socio-économiques IRIS
        filo_iris_folder = self._find_latest_folder(
            self.filo_iris_bucket,
            self.filo_iris_prefix
        )

        # Télécharger le fichier CSV
        filo_iris_path = f"{self.filo_iris_prefix}/{filo_iris_folder}/base_td_file_iris.csv"
        local_filo_path = os.path.join(self.temp_dir, "base_td_file_iris.csv")

        try:
            filo_data = self.minio.minio_client.get_object(
                bucket_name=self.filo_iris_bucket,
                object_name=filo_iris_path
            ).read()
        except:
            # Tenter avec un autre nom de fichier
            alternative_path = f"{self.filo_iris_prefix}/{filo_iris_folder}/base_td_filo_iris.csv"
            filo_data = self.minio.minio_client.get_object(
                bucket_name=self.filo_iris_bucket,
                object_name=alternative_path
            ).read()

        with open(local_filo_path, 'wb') as f:
            f.write(filo_data)

        return {
            "dvf_df": dvf_df_filtered,
            "contours_file": local_contours_path,
            "filo_file": local_filo_path
        }

    def transform(self, data):
        """Effectue la transformation et l'enrichissement des données"""
        dvf_df = data["dvf_df"]
        contours_file = data["contours_file"]
        filo_file = data["filo_file"]

        # 1. Charger les contours IRIS avec GeoPandas
        iris_gdf = gpd.read_file(contours_file)

        # Utiliser code_iris comme identifiant IRIS
        iris_id_col = "code_iris"
        if iris_id_col not in iris_gdf.columns:
            raise ValueError(f"La colonne {iris_id_col} est requise dans les contours IRIS")

        # S'assurer que le système de coordonnées est en WGS84
        if iris_gdf.crs is None:
            iris_gdf.set_crs(epsg=4326, inplace=True)
        elif iris_gdf.crs.to_epsg() != 4326:
            iris_gdf = iris_gdf.to_crs(epsg=4326)

        # 2. Charger les données socio-économiques IRIS
        try:
            filo_df = pd.read_csv(filo_file, delimiter=";", encoding='utf-8')
        except:
            try:
                filo_df = pd.read_csv(filo_file, delimiter=",", encoding='utf-8')
            except:
                filo_df = pd.read_csv(filo_file, delimiter=";", encoding='latin-1')

        # Identifier la colonne IRIS
        filo_iris_col = None
        for possible_col in ["IRIS", "iris", "code_iris", "CODE_IRIS"]:
            if possible_col in filo_df.columns:
                filo_iris_col = possible_col
                break

        if filo_iris_col is None:
            # Si aucune colonne évidente, prendre la première
            filo_iris_col = filo_df.columns[0]

        # 3. Standardiser les colonnes IRIS pour la jointure
        iris_gdf[iris_id_col] = iris_gdf[iris_id_col].astype(str).str.strip()
        filo_df[filo_iris_col] = filo_df[filo_iris_col].astype(str).str.strip()

        # Vérifier si le format est de 9 caractères
        if iris_gdf[iris_id_col].str.len().max() == 9 or filo_df[filo_iris_col].str.len().max() == 9:
            iris_gdf[iris_id_col] = iris_gdf[iris_id_col].str.zfill(9)
            filo_df[filo_iris_col] = filo_df[filo_iris_col].str.zfill(9)

        # Renommer les colonnes
        iris_gdf = iris_gdf.rename(columns={iris_id_col: "IRIS"})
        filo_df = filo_df.rename(columns={filo_iris_col: "IRIS"})

        # 4. Fusionner les données socio-économiques avec les contours
        iris_complet = iris_gdf.merge(filo_df, on="IRIS", how="left")

        # Vérifier si la jointure a peu de correspondances
        if len(filo_df.columns) > 1:
            sample_col = filo_df.columns[1]
            match_count = iris_complet[sample_col].notna().sum()
            match_percent = round(match_count / len(iris_complet) * 100, 2)

            if match_percent < 10:
                # Essayer une jointure alternative sur les 5 premiers caractères
                iris_gdf["IRIS_5"] = iris_gdf["IRIS"].str[:5]
                filo_df["IRIS_5"] = filo_df["IRIS"].str[:5]

                iris_complet = iris_gdf.merge(filo_df.drop(columns=["IRIS"]),
                                              left_on="IRIS_5",
                                              right_on="IRIS_5",
                                              how="left")

        # Identifier les colonnes socio-économiques
        socio_eco_cols = [col for col in iris_complet.columns if col.startswith("DEC_")]

        # 5. Convertir les données DVF pour la jointure spatiale
        cols_dvf = ["id_mutation", "longitude", "latitude"]

        # Ajouter les colonnes de localisation si disponibles
        for col in ["code_postal", "code_departement", "code_commune", "nom_commune"]:
            if col in dvf_df.columns:
                cols_dvf.append(col)

        dvf_pandas = dvf_df.select(cols_dvf).toPandas()

        # Créer les points pour la jointure spatiale
        geometry = [Point(lon, lat) for lon, lat in zip(dvf_pandas["longitude"], dvf_pandas["latitude"])]
        dvf_points = gpd.GeoDataFrame(dvf_pandas, geometry=geometry, crs="EPSG:4326")

        # Vérifier et harmoniser les CRS
        if dvf_points.crs != iris_complet.crs:
            dvf_points = dvf_points.to_crs(iris_complet.crs)

        # 6. Jointure spatiale
        # Corriger les géométries invalides
        if iris_complet.geometry.is_valid.sum() != len(iris_complet):
            iris_complet['geometry'] = iris_complet.geometry.buffer(0)

        try:
            # Jointure spatiale standard
            dvf_with_iris = gpd.sjoin(dvf_points, iris_complet, how="left", predicate="within")
        except Exception:
            try:
                # Tenter avec buffer
                dvf_points_buffered = dvf_points.copy()
                dvf_points_buffered['geometry'] = dvf_points_buffered.geometry.buffer(0.0001)
                dvf_with_iris = gpd.sjoin(dvf_points_buffered, iris_complet, how="left", predicate="intersects")
            except Exception:
                # Si tout échoue, créer un DataFrame minimal
                dvf_with_iris = dvf_points.copy()
                for col in iris_complet.columns:
                    if col not in dvf_with_iris.columns and col != 'geometry':
                        dvf_with_iris[col] = None

        # Renommer la colonne IRIS si dupliquée
        for col in dvf_with_iris.columns:
            if col.startswith('IRIS_') or col.endswith('_IRIS'):
                dvf_with_iris['IRIS'] = dvf_with_iris[col]
                break

        # 7. Sélectionner les colonnes pour Spark
        cols_to_keep = ["id_mutation", "IRIS"]
        cols_to_keep.extend([col for col in socio_eco_cols if col in dvf_with_iris.columns])

        dvf_enriched_pandas = dvf_with_iris[cols_to_keep]

        # Remplacer NaN par None
        dvf_enriched_pandas = dvf_enriched_pandas.replace({np.nan: None})

        # 8. Convertir en DataFrame Spark
        try:
            # Méthode avec Row
            from pyspark.sql import Row

            data_rows = []
            for _, row in dvf_enriched_pandas.iterrows():
                row_dict = {}
                for col in dvf_enriched_pandas.columns:
                    row_dict[col] = row[col]
                data_rows.append(Row(**row_dict))

            dvf_enriched_spark = self.spark.createDataFrame(data_rows)
        except Exception:
            # Si échec, utiliser une méthode plus basique
            schema = StructType([
                StructField("id_mutation", StringType(), True),
                StructField("IRIS", StringType(), True)
            ])

            for col in socio_eco_cols:
                if col in dvf_enriched_pandas.columns:
                    schema.add(StructField(col, DoubleType(), True))

            data_tuples = []
            for _, row in dvf_enriched_pandas.iterrows():
                row_tuple = []
                for col in dvf_enriched_pandas.columns:
                    row_tuple.append(row[col])
                data_tuples.append(tuple(row_tuple))

            dvf_enriched_spark = self.spark.createDataFrame(data_tuples, schema)

        # 9. Joindre avec les données DVF originales
        result_df = dvf_df.join(
            dvf_enriched_spark,
            on="id_mutation",
            how="left"
        )

        return result_df

    def load(self, data_frame):
        """Sauvegarde les données enrichies"""
        # Assurer que le bucket existe
        self.minio.ensure_bucket_exists(self.output_bucket)

        # Écriture en parquet
        self.minio.write(
            df=data_frame.coalesce(1),
            bucket=self.output_bucket,
            path=self.output_path,
            format="parquet",
            mode="overwrite"
        )

        # Métadonnées basiques
        metadata = {
            "version": self.output_version,
            "created_at": datetime.datetime.now().isoformat()
        }

        # Sérialisation sécurisée pour JSON
        def json_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [json_serializable(i) for i in obj]
            else:
                return obj

        serializable_metadata = json_serializable(metadata)
        metadata_json = json.dumps(serializable_metadata, indent=2)
        metadata_path = f"{self.output_base_path}/{self.output_version}/metadata.json"

        self.minio.minio_client.put_object(
            bucket_name=self.output_bucket,
            object_name=metadata_path,
            data=io.BytesIO(metadata_json.encode()),
            length=len(metadata_json),
            content_type="application/json"
        )

        # Nettoyer les fichiers temporaires
        self._cleanup()

    def _cleanup(self):
        """Nettoie les fichiers temporaires"""
        import shutil
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


if __name__ == "__main__":
    import sys
    import argparse
    from etl.common.spark import SparkManager

    parser = argparse.ArgumentParser(description='Enrichir les données DVF avec les données socio-économiques IRIS')
    parser.add_argument('--dvf-bucket', type=str, default='processed')
    parser.add_argument('--dvf-base-path', type=str, default='dvf_consolidated')
    parser.add_argument('--dvf-version', type=str)
    parser.add_argument('--iris-contours-bucket', type=str, default='raw-data')
    parser.add_argument('--iris-contours-prefix', type=str, default='iris-contours')
    parser.add_argument('--filo-iris-bucket', type=str, default='raw-data')
    parser.add_argument('--filo-iris-prefix', type=str, default='base-td-file-iris')
    parser.add_argument('--output-bucket', type=str, default='processed')
    parser.add_argument('--output-base-path', type=str, default='dvf_enriched')
    parser.add_argument('--output-version', type=str)

    args = parser.parse_args()

    spark_manager = None
    processor = None

    try:
        spark_manager = SparkManager.get_instance()

        processor = DVFSocioEconomicEnrichment(
            dvf_bucket=args.dvf_bucket,
            dvf_base_path=args.dvf_base_path,
            dvf_version=args.dvf_version,
            iris_contours_bucket=args.iris_contours_bucket,
            iris_contours_prefix=args.iris_contours_prefix,
            filo_iris_bucket=args.filo_iris_bucket,
            filo_iris_prefix=args.filo_iris_prefix,
            output_bucket=args.output_bucket,
            output_base_path=args.output_base_path,
            output_version=args.output_version
        )

        processor.execute()

    except Exception as e:
        print(f"Erreur: {str(e)}")
        sys.exit(1)

    finally:
        if spark_manager:
            spark_manager.stop()