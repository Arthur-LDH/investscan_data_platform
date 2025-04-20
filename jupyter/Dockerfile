FROM jupyter/pyspark-notebook:spark-3.4.1

USER root

# Installation de packages système supplémentaires
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    git \
    vim \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copie des fichiers de configuration
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

USER jovyan

WORKDIR /home/jovyan/notebooks

# Exposer le port Jupyter
EXPOSE 8888

# Démarrer JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]