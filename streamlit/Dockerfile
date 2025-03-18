FROM python:3.10-slim-bullseye

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

# Mettre à jour et installer les dépendances système
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    python3-dev \
    openjdk-11-jdk \
    curl \
    software-properties-common \
    git \
    procps \
    libgdal-dev \
    gdal-bin \
    && rm -rf /var/lib/apt/lists/*

# Définir les variables d'environnement
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin
ENV PYSPARK_PYTHON=/usr/bin/python3
ENV PYSPARK_DRIVER_PYTHON=/usr/bin/python3
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Copie et installation des autres dépendances
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copie des fichiers de l'application
COPY . .

# Exposition du port et commande de démarrage
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]