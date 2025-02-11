#!/bin/bash

# Vérifie que les modules Python nécessaires sont installés
python3 -c "import pandas, pyarrow, minio" || exit 1

# Vérifie que les DAGs peuvent être chargés sans erreur
airflow dags list 2>/dev/null || exit 1

# Vérifie l'accès aux dossiers nécessaires
test -d /opt/airflow/dags || exit 1
test -d /opt/airflow/logs || exit 1

exit 0