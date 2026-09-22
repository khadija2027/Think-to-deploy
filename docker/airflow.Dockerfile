FROM apache/airflow:2.8.1-python3.11
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libgomp1 poppler-utils tesseract-ocr tesseract-ocr-fra antiword \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/airflow/pipeline /opt/airflow/index /opt/airflow/model-cache \
    && chown -R airflow:root /opt/airflow/pipeline /opt/airflow/index /opt/airflow/model-cache
USER airflow
COPY requirements /requirements
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir "apache-airflow==2.8.1" -r /requirements/ingestion.txt
COPY T2D_Chatbot_for_HR/rag_core /opt/project/rag_core
COPY T2D_Chatbot_for_HR/init_airflow.py /opt/project/init_airflow.py
ENV PYTHONPATH=/opt/project
