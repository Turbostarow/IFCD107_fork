FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/app

WORKDIR /app

RUN groupadd --system app \
    && useradd --system --gid app --create-home --home-dir /home/app app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-boosting.txt .
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements-boosting.txt

COPY --chown=app:app app.py modelo.py interfaz.py catalogo.py controles.py descripciones.py ./
COPY --chown=app:app cnn_modelo.py cnn_interfaz.py ./
COPY --chown=app:app geometria.py ./
COPY --chown=app:app boosting_ui.py ./
COPY --chown=app:app curvas.py ./
COPY --chown=app:app pages/ ./pages/
USER app

EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--server.fileWatcherType=none", "--browser.gatherUsageStats=false"]
