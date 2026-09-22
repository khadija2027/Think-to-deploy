FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements /requirements
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /requirements/api.txt
COPY T2D_Chatbot_for_HR/rag_core /opt/project/rag_core
COPY T2D_Chatbot_for_HR/frontend_chatbot /opt/project/frontend_chatbot
ENV PYTHONPATH=/opt/project
EXPOSE 8000
CMD ["uvicorn", "rag_core.web:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
