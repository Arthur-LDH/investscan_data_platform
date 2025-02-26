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

    def read(self, bucket, path, format="csv", options=None):
        """Lit des données depuis MinIO"""
        if options is None:
            options = {"header": "true", "inferSchema": "true"}

        return (self.spark.read
                .format(format)
                .options(**options)
                .load(f"s3a://{bucket}/{path}"))

    def write(self, df, bucket, path, format="parquet", mode="overwrite", options=None):
        """Écrit des données dans MinIO"""
        if options is None:
            options = {}

        (df.write
         .format(format)
         .mode(mode)
         .options(**options)
         .save(f"s3a://{bucket}/{path}"))


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