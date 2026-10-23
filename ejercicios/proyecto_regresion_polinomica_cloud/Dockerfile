FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    MODEL_PATH=/app/artifacts/model.joblib
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY scripts ./scripts
COPY train_model.py .
ENV PYTHONPATH=/app/src

# Para el ejemplo, el modelo se entrena sobre un CSV sintético dentro del build.
# En producción, versionar y publicar el artefacto por separado del contenedor.
RUN python scripts/generate_data.py --output /tmp/random_forest_demo.csv \
    && python train_model.py --csv /tmp/random_forest_demo.csv \
    && rm /tmp/random_forest_demo.csv

EXPOSE 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "4", "--timeout", "60", "--factory", "cloud_rf.app:create_app()"]
