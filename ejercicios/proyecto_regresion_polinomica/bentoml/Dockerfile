FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BENTOML_HOME=/var/lib/bentoml

WORKDIR /app

COPY bentoml/requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /app/requirements.txt

COPY bentoml/ /app/bentoml/
COPY ejemplos/datos_ejemplo_xy.csv /app/data/datos_ejemplo_xy.csv

WORKDIR /app/bentoml
EXPOSE 3000
