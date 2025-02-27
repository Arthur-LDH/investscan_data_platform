class StorageConnector:
    """Classe de base pour les connecteurs de stockage"""

    def __init__(self, spark_session):
        self.spark = spark_session

    def read(self, *args, **kwargs):
        raise NotImplementedError("Sous-classes doivent implémenter cette méthode")

    def write(self, *args, **kwargs):
        raise NotImplementedError("Sous-classes doivent implémenter cette méthode")


class MinIOConnector(StorageConnector):
    """Connecteur pour MinIO/S3"""

    def __init__(self, spark_session, endpoint="localhost:9000",
                 access_key="minioadmin", secret_key="minioadmin", secure=False):
        super().__init__(spark_session)
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.secure = secure
        self._minio_client = None

    @property
    def minio_client(self):
        """Lazy initialization du client MinIO"""
        if self._minio_client is None:
            from minio import Minio
            self._minio_client = Minio(
                endpoint=self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure
            )
        return self._minio_client

    def ensure_bucket_exists(self, bucket_name):
        """
        Vérifie si le bucket existe et le crée si nécessaire.

        Args:
            bucket_name: Nom du bucket à vérifier/créer

        Returns:
            bool: True si le bucket existe ou a été créé
        """
        import logging
        logger = logging.getLogger(__name__)

        try:
            if not self.minio_client.bucket_exists(bucket_name):
                self.minio_client.make_bucket(bucket_name)
                logger.info(f"Bucket {bucket_name} créé avec succès")
            else:
                logger.info(f"Bucket {bucket_name} existe déjà")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la vérification/création du bucket: {str(e)}")
            raise

    def read(self, bucket, path, format="csv", options=None):
        """
        Lit des données depuis MinIO

        Args:
            bucket: Nom du bucket
            path: Chemin de l'objet ou du répertoire
            format: Format des données (csv, parquet, etc.)
            options: Options de lecture

        Returns:
            DataFrame: DataFrame Spark contenant les données lues
        """
        if options is None:
            options = {"header": "true", "inferSchema": "true"}

        return (self.spark.read
                .format(format)
                .options(**options)
                .load(f"s3a://{bucket}/{path}"))

    def write(self, df, bucket, path, format="parquet", mode="overwrite",
              partition_by=None, options=None):
        """
        Écrit des données dans MinIO

        Args:
            df: DataFrame à écrire
            bucket: Nom du bucket
            path: Chemin de destination
            format: Format d'écriture (parquet, csv, etc.)
            mode: Mode d'écriture (overwrite, append, etc.)
            partition_by: Liste des colonnes pour partitionnement
            options: Options d'écriture supplémentaires
        """
        # S'assurer que le bucket existe
        self.ensure_bucket_exists(bucket)

        if options is None:
            options = {}

        writer = df.write.format(format).mode(mode).options(**options)

        if partition_by:
            writer = writer.partitionBy(partition_by)

        writer.save(f"s3a://{bucket}/{path}")

    def list_files(self, bucket, prefix="", extension=None):
        """
        Liste les fichiers dans un bucket MinIO avec un préfixe donné.

        Args:
            bucket: Nom du bucket
            prefix: Préfixe des objets à lister
            extension: Extension de fichier à filtrer

        Returns:
            list: Liste des chemins des objets
        """
        import logging
        logger = logging.getLogger(__name__)

        try:
            objects = list(self.minio_client.list_objects(bucket, prefix=prefix, recursive=True))

            if extension:
                files = [obj.object_name for obj in objects if obj.object_name.endswith(extension)]
            else:
                files = [obj.object_name for obj in objects]

            return files

        except Exception as e:
            logger.error(f"Erreur lors du listage des fichiers: {str(e)}")
            raise


class PostgresConnector(StorageConnector):
    """Connecteur pour PostgreSQL"""

    def __init__(self, spark_session, host="postgres", port=5432,
                 database="real_estate_db", user="investscan", password="password"):
        super().__init__(spark_session)
        self.jdbc_url = f"jdbc:postgresql://{host}:{port}/{database}"
        self.properties = {
            "user": user,
            "password": password,
            "driver": "org.postgresql.Driver"
        }

    def read(self, table_name, conditions=None):
        """Lit des données depuis PostgreSQL"""
        query = f"SELECT * FROM {table_name}"
        if conditions:
            query += f" WHERE {conditions}"

        return (self.spark.read
                .jdbc(url=self.jdbc_url,
                      table=f"({query}) AS tmp",
                      properties=self.properties))

    def write(self, df, table_name, mode="overwrite"):
        """Écrit des données dans PostgreSQL"""
        (df.write
         .jdbc(url=self.jdbc_url,
               table=table_name,
               mode=mode,
               properties=self.properties))