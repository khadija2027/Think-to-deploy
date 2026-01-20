FROM apache/airflow:2.8.1-python3.9

USER root
RUN apt-get update && apt-get install -y \
    gcc g++ libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Copier requirements.txt d'abord (pour meilleur cache)
COPY requirements.txt /requirements.txt

# Installer les dépendances
RUN pip install --no-cache-dir -r /requirements.txt

# Créer les dossiers nécessaires
RUN mkdir -p /opt/airflow/dags/anonymized_dataset \
    /opt/airflow/dags/parsed \
    /opt/airflow/dags/chunked \
    /opt/airflow/dags/failed \
    /opt/airflow/dags/logs_pipeline \
    /opt/airflow/dags/cache \
    /opt/airflow/dags/output \
    /opt/airflow/dags/backend

WORKDIR /opt/airflow