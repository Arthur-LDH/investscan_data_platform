FROM apache/airflow:2.10.4

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsnappy-dev \
    p7zip-full \
    gdal-bin \
    libgdal-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

USER airflow

RUN pip install --no-cache-dir \
    apache-airflow-providers-amazon \
    pandas \
    requests \
    minio \
    pyarrow==14.0.1 \
    python-snappy==0.6.1 \
    geopandas \
    shapely \
    fiona \
    pyproj \
    rtree \
    mapclassify \
    pyogrio