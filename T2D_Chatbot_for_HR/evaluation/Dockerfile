FROM t2dphase2-fastapi:latest
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY T2D_Chatbot_for_HR/evaluation/requirements.txt /tmp/evaluation-requirements.txt
RUN pip install --no-cache-dir -r /tmp/evaluation-requirements.txt
WORKDIR /evaluation
ENTRYPOINT ["python", "-u", "/evaluation/run_ragas.py"]
