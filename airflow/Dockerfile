FROM apache/airflow:2.10.4

USER root

# Installation des dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsnappy-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copie et configuration des scripts (en tant que root)
COPY scripts/run_tests.sh /usr/local/bin/
COPY scripts/healthcheck.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/run_tests.sh \
    && chmod +x /usr/local/bin/healthcheck.sh

USER airflow

# Installation des dépendances Python pour l'application
RUN pip install --no-cache-dir \
    apache-airflow-providers-amazon \
    pandas \
    requests \
    minio \
    pyarrow==14.0.1 \
    python-snappy==0.6.1

# Installation des dépendances de test
RUN pip install --no-cache-dir \
    pytest==8.0.0 \
    pytest-cov==4.1.0 \
    pytest-mock==3.12.0

# Les dags et tests seront montés via les volumes dans docker-compose
HEALTHCHECK CMD ["healthcheck.sh"]